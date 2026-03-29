"""Registry for transformer weight blocks and their residency metadata."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List

from .constants import HBM_RESIDENT_STATES, STATE_HBM_FULL_PRECISION

LOGGER = logging.getLogger(__name__)

DEFAULT_HBM_CAPACITY_BYTES = 4 * 1024**3
DEFAULT_ACCESS_WINDOW = 32
MIN_REUSE_SCORE = 0.0
MAX_REUSE_SCORE = 1.0
DEFAULT_TRANSFER_COST_US = 750.0
DEFAULT_DECOMP_COST_US = 50.0


@dataclass(slots=True)
class WeightBlockMeta:
    """Metadata tracked for a single model weight block.

    Attributes:
        block_id: Unique identifier for the block.
        layer_name: Owning layer name.
        current_state: Current residency state S0-S6.
        precision: Precision or storage encoding for the block.
        size_bytes: Block size in bytes.
        reuse_score: Reuse score rho(b) in [0, 1].
        routing_likelihood: Routing likelihood lambda(b) in [0, 1].
        layer_criticality: Layer criticality kappa(b) in [0, 1].
        quality_sensitivity: Quality sensitivity psi(b) in [0, 1].
        decomp_cost_us: Decompression cost delta(b) in microseconds.
        transfer_cost_us: Transfer cost tau(b) in microseconds.
        last_access_step: Latest observed access step.
        predicted_next_access: Predicted next access step.
        eviction_protected: Guardrail protection flag.
        access_history: Sliding window of prior access steps.
    """

    block_id: str
    layer_name: str
    current_state: int = STATE_HBM_FULL_PRECISION
    precision: str = "fp16"
    size_bytes: int = 0
    reuse_score: float = MIN_REUSE_SCORE
    routing_likelihood: float = 0.0
    layer_criticality: float = 0.0
    quality_sensitivity: float = 0.0
    decomp_cost_us: float = DEFAULT_DECOMP_COST_US
    transfer_cost_us: float = DEFAULT_TRANSFER_COST_US
    last_access_step: int = -1
    predicted_next_access: int = -1
    eviction_protected: bool = False
    access_history: List[int] = field(default_factory=list)


class WeightBlockRegistry:
    """Thread-safe metadata store for tracked weight blocks."""

    def __init__(self, hbm_capacity_bytes: int = DEFAULT_HBM_CAPACITY_BYTES) -> None:
        """Initialize the registry.

        Args:
            hbm_capacity_bytes: Total effective HBM capacity available.
        """
        self.hbm_capacity_bytes = hbm_capacity_bytes
        self._blocks: Dict[str, WeightBlockMeta] = {}
        self._layer_counts: Dict[str, int] = {}
        self._lock = threading.RLock()

    def register_block(
        self,
        layer_name: str,
        size_bytes: int,
        criticality: float,
        sensitivity: float,
    ) -> str:
        """Register a new weight block and return its identifier.

        Args:
            layer_name: Owning layer name.
            size_bytes: Block footprint in bytes.
            criticality: Static layer criticality in [0, 1].
            sensitivity: Quality sensitivity in [0, 1].

        Returns:
            The generated block identifier.

        Raises:
            ValueError: If size or scores are invalid.
        """
        if size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        if not 0.0 <= criticality <= 1.0:
            raise ValueError("criticality must be in [0, 1]")
        if not 0.0 <= sensitivity <= 1.0:
            raise ValueError("sensitivity must be in [0, 1]")

        with self._lock:
            index = self._layer_counts.get(layer_name, 0)
            self._layer_counts[layer_name] = index + 1
            block_id = f"{layer_name}:{index}"
            self._blocks[block_id] = WeightBlockMeta(
                block_id=block_id,
                layer_name=layer_name,
                size_bytes=size_bytes,
                layer_criticality=criticality,
                quality_sensitivity=sensitivity,
            )
            LOGGER.debug("Registered block %s for layer %s", block_id, layer_name)
            return block_id

    def update_access(self, block_id: str, current_step: int) -> None:
        """Record a weight access and update derived access predictions.

        Args:
            block_id: Block identifier.
            current_step: Current scheduler or forward-pass step.
        """
        with self._lock:
            block = self.get_block(block_id)
            block.last_access_step = current_step
            block.access_history.append(current_step)
            if len(block.access_history) > DEFAULT_ACCESS_WINDOW:
                block.access_history = block.access_history[-DEFAULT_ACCESS_WINDOW:]

            if len(block.access_history) >= 2:
                intervals = [
                    later - earlier
                    for earlier, later in zip(
                        block.access_history[:-1],
                        block.access_history[1:],
                    )
                ]
                avg_interval = max(1, round(sum(intervals) / len(intervals)))
                block.predicted_next_access = current_step + avg_interval
            else:
                block.predicted_next_access = current_step + 1
            self.update_reuse_score(block_id, window_size=DEFAULT_ACCESS_WINDOW)

    def update_reuse_score(self, block_id: str, window_size: int = DEFAULT_ACCESS_WINDOW) -> None:
        """Refresh the reuse score using a recent sliding access window.

        Args:
            block_id: Block identifier.
            window_size: Number of recent steps in the scoring window.
        """
        if window_size <= 0:
            raise ValueError("window_size must be positive")

        with self._lock:
            block = self.get_block(block_id)
            if not block.access_history:
                block.reuse_score = MIN_REUSE_SCORE
                return

            history = block.access_history[-window_size:]
            if len(history) == 1:
                block.reuse_score = min(MAX_REUSE_SCORE, 1.0 / float(window_size))
                return

            recency_span = max(1, history[-1] - history[0] + 1)
            density = min(1.0, len(history) / float(window_size))
            frequency = min(1.0, len(history) / float(recency_span))
            block.reuse_score = min(MAX_REUSE_SCORE, max(MIN_REUSE_SCORE, 0.5 * density + 0.5 * frequency))

    def get_block(self, block_id: str) -> WeightBlockMeta:
        """Return metadata for a single block.

        Args:
            block_id: Block identifier.

        Returns:
            The stored block metadata.

        Raises:
            KeyError: If the block is unknown.
        """
        if block_id not in self._blocks:
            raise KeyError(f"unknown block_id: {block_id}")
        return self._blocks[block_id]

    def get_all_blocks(self) -> List[WeightBlockMeta]:
        """Return a snapshot list of all tracked blocks."""
        with self._lock:
            return list(self._blocks.values())

    def get_blocks_by_state(self, state: int) -> List[WeightBlockMeta]:
        """Return all blocks currently assigned to a given state.

        Args:
            state: Residency state integer.
        """
        with self._lock:
            return [block for block in self._blocks.values() if block.current_state == state]

    def get_hbm_pressure(self) -> float:
        """Return the fraction of configured HBM currently occupied."""
        with self._lock:
            resident_bytes = sum(
                block.size_bytes
                for block in self._blocks.values()
                if block.current_state in HBM_RESIDENT_STATES
            )
            if self.hbm_capacity_bytes <= 0:
                return 0.0
            return resident_bytes / float(self.hbm_capacity_bytes)

    def set_eviction_protection(self, block_id: str, protected: bool) -> None:
        """Set the eviction protection flag for a block.

        Args:
            block_id: Block identifier.
            protected: Desired protection state.
        """
        with self._lock:
            block = self.get_block(block_id)
            block.eviction_protected = protected

    def set_state(self, block_id: str, state: int, precision: str | None = None) -> None:
        """Update a block's residency state and optionally its precision.

        Args:
            block_id: Block identifier.
            state: New state S0-S6.
            precision: Optional replacement precision label.
        """
        with self._lock:
            block = self.get_block(block_id)
            block.current_state = state
            if precision is not None:
                block.precision = precision
