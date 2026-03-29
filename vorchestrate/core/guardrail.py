"""Accuracy guardrails for sensitive blocks."""

from __future__ import annotations

from typing import List

from .constants import STATE_HBM_LOW_BIT
from .registry import WeightBlockMeta, WeightBlockRegistry

DEFAULT_PSI_THRESHOLD = 0.7


class AccuracyGuardrail:
    """Protect highly quality-sensitive blocks from aggressive demotion."""

    def __init__(self, psi_threshold: float = DEFAULT_PSI_THRESHOLD) -> None:
        """Initialize the guardrail.

        Args:
            psi_threshold: Sensitivity threshold above which blocks are protected.
        """
        self.psi_threshold = psi_threshold

    def check_demotion(self, block: WeightBlockMeta, target_state: int) -> bool:
        """Return whether a demotion target is allowed.

        Args:
            block: Weight block metadata.
            target_state: Proposed target state.

        Returns:
            True when the demotion is allowed, otherwise False.
        """
        return not (
            block.quality_sensitivity > self.psi_threshold
            and target_state > STATE_HBM_LOW_BIT
        )

    def set_protection(self, block_id: str, registry: WeightBlockRegistry) -> None:
        """Enable protection on a registry block."""
        registry.set_eviction_protection(block_id, True)

    def release_protection(self, block_id: str, registry: WeightBlockRegistry) -> None:
        """Disable protection on a registry block."""
        registry.set_eviction_protection(block_id, False)

    def get_protected_blocks(self, registry: WeightBlockRegistry) -> List[str]:
        """Return block ids that are currently protected."""
        return [
            block.block_id
            for block in registry.get_all_blocks()
            if block.eviction_protected
        ]
