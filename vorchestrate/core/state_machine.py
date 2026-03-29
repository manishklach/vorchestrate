"""State machine that governs block residency transitions."""

from __future__ import annotations

import logging
import time
from typing import Protocol

from .constants import (
    HBM_RESIDENT_STATES,
    STATE_HBM_COMPRESSED,
    STATE_HBM_FULL_PRECISION,
    STATE_HBM_LOW_BIT,
    STATE_HOST_DRAM,
    STATE_NVME,
)
from .guardrail import AccuracyGuardrail
from .registry import WeightBlockRegistry
from .scorer import ScoringEngine

LOGGER = logging.getLogger(__name__)

REASON_PROMOTION = "promotion"
REASON_DEMOTION = "demotion"
REASON_TICK = "tick"
DEFAULT_HBM_HEADROOM = 0.95
PROMOTION_TARGETS = {
    6: STATE_NVME,
    5: STATE_HOST_DRAM,
    4: STATE_HOST_DRAM,
    3: STATE_HBM_LOW_BIT,
    2: STATE_HBM_LOW_BIT,
    1: STATE_HBM_FULL_PRECISION,
    0: STATE_HBM_FULL_PRECISION,
}
DEMOTION_TARGETS = {
    0: STATE_HBM_LOW_BIT,
    1: STATE_HOST_DRAM,
    2: STATE_HOST_DRAM,
    3: STATE_NVME,
    4: STATE_NVME,
}

STATE_TO_PRECISION = {
    STATE_HBM_FULL_PRECISION: "fp16",
    STATE_HBM_LOW_BIT: "int8",
    STATE_HBM_COMPRESSED: "compressed",
    STATE_HOST_DRAM: "int8",
    STATE_NVME: "compressed",
}


class WeightStateMachine:
    """Manage block transitions between the seven residency states."""

    def __init__(
        self,
        registry: WeightBlockRegistry,
        scorer: ScoringEngine,
        guardrail: AccuracyGuardrail,
        scheduler: SchedulerProtocol | None = None,
    ) -> None:
        """Initialize the state machine.

        Args:
            registry: Block registry dependency.
            scorer: Scoring engine dependency.
            guardrail: Guardrail dependency.
            scheduler: Optional scheduler for transfer commands.
        """
        self.registry = registry
        self.scorer = scorer
        self.guardrail = guardrail
        self.scheduler = scheduler
        self._transition_history: list[dict[str, object]] = []

    def transition(self, block_id: str, target_state: int) -> bool:
        """Attempt a state transition for a block.

        Args:
            block_id: Block identifier.
            target_state: Desired new state.

        Returns:
            True if the transition executed, otherwise False.
        """
        block = self.registry.get_block(block_id)
        current_state = block.current_state
        if target_state == current_state:
            return True

        if target_state > current_state and not self.guardrail.check_demotion(block, target_state):
            self.guardrail.set_protection(block_id, self.registry)
            self._log_transition(block_id, current_state, current_state, "guardrail_veto")
            return False

        precision = STATE_TO_PRECISION.get(type(block.current_state)(target_state), block.precision)
        self.registry.set_state(block_id, target_state, precision=precision)
        if block.quality_sensitivity > self.guardrail.psi_threshold:
            self.guardrail.set_protection(block_id, self.registry)

        reason = REASON_PROMOTION if target_state < current_state else REASON_DEMOTION
        self._log_transition(block_id, current_state, target_state, reason)
        return True

    def promote(self, block_id: str) -> bool:
        """Promote a block one tier closer to HBM full precision."""
        block = self.registry.get_block(block_id)
        target_state = PROMOTION_TARGETS.get(block.current_state, STATE_HBM_FULL_PRECISION)
        return self.transition(block_id, target_state)

    def demote(self, block_id: str) -> bool:
        """Demote a block one tier farther from HBM full precision."""
        block = self.registry.get_block(block_id)
        target_state = DEMOTION_TARGETS.get(block.current_state, STATE_NVME)
        return self.transition(block_id, target_state)

    def tick(self, current_step: int, hbm_budget_bytes: int) -> None:
        """Run one control-loop iteration.

        Args:
            current_step: Current inference step.
            hbm_budget_bytes: Effective HBM budget in bytes.
        """
        self.registry.hbm_capacity_bytes = hbm_budget_bytes
        hbm_pressure = self.registry.get_hbm_pressure()
        scores = self.scorer.score_all_blocks()
        LOGGER.debug("Tick step=%s pressure=%.3f blocks=%s", current_step, hbm_pressure, len(scores))

        if hbm_pressure > DEFAULT_HBM_HEADROOM:
            for block_id in self.scorer.rank_blocks_for_demotion():
                block = self.registry.get_block(block_id)
                target_state = self.scorer.get_target_state(block_id, self.registry.get_hbm_pressure())
                if target_state <= block.current_state:
                    target_state = DEMOTION_TARGETS.get(block.current_state, STATE_NVME)
                transitioned = self.transition(block_id, target_state)
                if transitioned and self.scheduler is not None:
                    self.scheduler.enqueue_demotion(block_id, priority=10)
                if self.registry.get_hbm_pressure() <= DEFAULT_HBM_HEADROOM:
                    break

        for block_id in self.scorer.rank_blocks_for_promotion():
            block = self.registry.get_block(block_id)
            target_state = self.scorer.get_target_state(block_id, self.registry.get_hbm_pressure())
            if target_state < block.current_state:
                projected_pressure = (
                    self.registry.get_hbm_pressure()
                    + (
                        block.size_bytes / float(max(self.registry.hbm_capacity_bytes, 1))
                        if target_state in HBM_RESIDENT_STATES and block.current_state not in HBM_RESIDENT_STATES
                        else 0.0
                    )
                )
                if projected_pressure > 1.0:
                    continue
                transitioned = self.transition(block_id, target_state)
                if transitioned and self.scheduler is not None:
                    self.scheduler.enqueue_promotion(block_id, priority=1)

            self._log_transition(block_id, block.current_state, self.registry.get_block(block_id).current_state, REASON_TICK, update_only=True)

    def get_transition_history(self) -> list[dict[str, object]]:
        """Return the transition log."""
        return list(self._transition_history)

    def _log_transition(
        self,
        block_id: str,
        from_state: int,
        to_state: int,
        reason: str,
        update_only: bool = False,
    ) -> None:
        """Append an entry to the transition history."""
        if update_only and from_state == to_state:
            return
        self._transition_history.append(
            {
                "block_id": block_id,
                "from_state": from_state,
                "to_state": to_state,
                "reason": reason,
                "timestamp": time.time(),
            }
        )


class SchedulerProtocol(Protocol):
    """Minimal protocol required by the state machine scheduler hook."""

    def enqueue_promotion(self, block_id: str, priority: int) -> None:
        """Queue a promotion command."""

    def enqueue_demotion(self, block_id: str, priority: int) -> None:
        """Queue a demotion command."""
