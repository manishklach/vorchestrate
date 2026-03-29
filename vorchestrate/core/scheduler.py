"""Asynchronous prefetch scheduling for weight promotions and demotions."""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass

from .registry import WeightBlockMeta, WeightBlockRegistry

LOGGER = logging.getLogger(__name__)

DEFAULT_PREFETCH_MARGIN_US = 500.0
DEFAULT_MAX_BANDWIDTH_BYTES_PER_SEC = 8 * 1024**3
DEFAULT_UTILIZATION_WINDOW_SEC = 1.0
PROMOTION_PRIORITY = 1
DEMOTION_PRIORITY = 10
DEMOTION_THROTTLE_THRESHOLD = 0.8
WORKER_IDLE_SLEEP_SEC = 0.01
SIMULATED_TRANSFER_CAP_US = 2_000.0


@dataclass(slots=True)
class TransferCommand:
    """A queued residency transition command."""

    priority: int
    command_type: str
    block_id: str
    enqueued_at: float


class PrefetchScheduler:
    """Background scheduler for prefetch and eviction traffic."""

    def __init__(
        self,
        registry: WeightBlockRegistry,
        transfer_margin_us: float = DEFAULT_PREFETCH_MARGIN_US,
        max_bandwidth_bytes_per_sec: float = DEFAULT_MAX_BANDWIDTH_BYTES_PER_SEC,
    ) -> None:
        """Initialize the prefetch scheduler and worker thread.

        Args:
            registry: Block registry dependency.
            transfer_margin_us: Safety margin added to prefetch windows.
            max_bandwidth_bytes_per_sec: Effective transfer bandwidth cap.
        """
        self.registry = registry
        self.transfer_margin_us = transfer_margin_us
        self.max_bandwidth_bytes_per_sec = max_bandwidth_bytes_per_sec
        self._queue: queue.PriorityQueue[tuple[int, int, TransferCommand]] = queue.PriorityQueue()
        self._counter = 0
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._completed_commands: list[TransferCommand] = []
        self._transfer_log: list[tuple[float, int]] = []
        self._thread = threading.Thread(target=self._run_worker, daemon=True, name="vorchestrate-scheduler")
        self._thread.start()

    def compute_prefetch_window(self, block: WeightBlockMeta) -> float:
        """Compute the prefetch window in microseconds."""
        return block.transfer_cost_us + block.decomp_cost_us + self.transfer_margin_us

    def should_prefetch(
        self,
        block: WeightBlockMeta,
        current_step: int,
        steps_per_second: float,
    ) -> bool:
        """Return whether the block should be prefetched now."""
        if steps_per_second <= 0.0 or block.predicted_next_access < current_step:
            return False
        time_to_access_us = ((block.predicted_next_access - current_step) / steps_per_second) * 1_000_000.0
        return time_to_access_us <= self.compute_prefetch_window(block)

    def enqueue_promotion(self, block_id: str, priority: int = PROMOTION_PRIORITY) -> None:
        """Enqueue a high-priority promotion command."""
        self._enqueue("promotion", block_id, priority)

    def enqueue_demotion(self, block_id: str, priority: int = DEMOTION_PRIORITY) -> None:
        """Enqueue a low-priority demotion command."""
        self._enqueue("demotion", block_id, priority)

    async def process_queue(self) -> None:
        """Coroutine worker that drains queued transfer commands."""
        while not self._stop_event.is_set():
            try:
                _, _, command = self._queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(WORKER_IDLE_SLEEP_SEC)
                continue

            if command.command_type == "demotion" and self.get_bandwidth_utilization() > DEMOTION_THROTTLE_THRESHOLD:
                self._enqueue(command.command_type, command.block_id, command.priority)
                await asyncio.sleep(WORKER_IDLE_SLEEP_SEC)
                continue

            block = self.registry.get_block(command.block_id)
            transfer_time_us = min(block.transfer_cost_us + block.decomp_cost_us, SIMULATED_TRANSFER_CAP_US)
            await asyncio.sleep(transfer_time_us / 1_000_000.0)
            with self._lock:
                self._completed_commands.append(command)
                self._transfer_log.append((time.time(), block.size_bytes))
                self._prune_transfer_log()

    def get_queue_depth(self) -> int:
        """Return the current queue depth."""
        return self._queue.qsize()

    def get_bandwidth_utilization(self) -> float:
        """Estimate rolling bandwidth utilization as a fraction of max bandwidth."""
        with self._lock:
            self._prune_transfer_log()
            bytes_in_window = sum(bytes_moved for _, bytes_moved in self._transfer_log)
        return min(1.0, bytes_in_window / max(self.max_bandwidth_bytes_per_sec * DEFAULT_UTILIZATION_WINDOW_SEC, 1.0))

    def shutdown(self) -> None:
        """Stop the background worker."""
        self._stop_event.set()
        self._thread.join(timeout=1.0)

    def _enqueue(self, command_type: str, block_id: str, priority: int) -> None:
        """Insert a command into the priority queue."""
        with self._lock:
            self._counter += 1
            command = TransferCommand(
                priority=priority,
                command_type=command_type,
                block_id=block_id,
                enqueued_at=time.time(),
            )
            self._queue.put((priority, self._counter, command))

    def _run_worker(self) -> None:
        """Run the asyncio worker loop in a background thread."""
        asyncio.run(self.process_queue())

    def _prune_transfer_log(self) -> None:
        """Drop transfer records outside the utilization window."""
        cutoff = time.time() - DEFAULT_UTILIZATION_WINDOW_SEC
        self._transfer_log = [
            entry for entry in self._transfer_log if entry[0] >= cutoff
        ]
