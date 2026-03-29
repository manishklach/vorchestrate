"""Scoring engine for predictive weight residency orchestration."""

from __future__ import annotations

import logging

from .constants import (
    STATE_HBM_FULL_PRECISION,
    STATE_HBM_LOW_BIT,
    STATE_HOST_DRAM,
    STATE_NVME,
)
from .registry import WeightBlockMeta, WeightBlockRegistry

LOGGER = logging.getLogger(__name__)

DEFAULT_W1 = 0.25
DEFAULT_W2 = 0.25
DEFAULT_W3 = 0.25
DEFAULT_W4 = 0.25
DEFAULT_ALPHA = 0.5
DEFAULT_BETA = 0.5
MIN_COST_EPSILON = 1e-6
MICROSECONDS_PER_NORMALIZED_UNIT = 1000.0
MIN_NORMALIZED_DENOMINATOR = 1.0
DEFAULT_THETA_HIGH = 0.7
DEFAULT_THETA_DRAM = 0.4
DEFAULT_THETA_SSD = 0.2
DEFAULT_HYSTERESIS_GAP = 0.05
HIGH_PRESSURE_THRESHOLD = 0.9
CRITICAL_PRESSURE_THRESHOLD = 0.98


class ScoringEngine:
    """Compute and rank composite block residency scores."""

    def __init__(
        self,
        registry: WeightBlockRegistry,
        w1: float = DEFAULT_W1,
        w2: float = DEFAULT_W2,
        w3: float = DEFAULT_W3,
        w4: float = DEFAULT_W4,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        theta_high: float = DEFAULT_THETA_HIGH,
        theta_dram: float = DEFAULT_THETA_DRAM,
        theta_ssd: float = DEFAULT_THETA_SSD,
        hysteresis_gap: float = DEFAULT_HYSTERESIS_GAP,
    ) -> None:
        """Initialize the scoring engine.

        Args:
            registry: Weight block registry dependency.
            w1: Weight for reuse score rho.
            w2: Weight for routing likelihood lambda.
            w3: Weight for layer criticality kappa.
            w4: Weight for quality sensitivity psi.
            alpha: Weight for decompression cost in denominator.
            beta: Weight for transfer cost in denominator.
            theta_high: Promotion threshold toward S0.
            theta_dram: Demotion threshold toward S3.
            theta_ssd: Demotion threshold toward S4.
            hysteresis_gap: Gap around thresholds to reduce thrashing.
        """
        total_weight = w1 + w2 + w3 + w4
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError("w1..w4 must sum to 1.0")
        if alpha < 0.0 or beta < 0.0:
            raise ValueError("alpha and beta must be non-negative")

        self.registry = registry
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.alpha = alpha
        self.beta = beta
        self.theta_high = theta_high
        self.theta_dram = theta_dram
        self.theta_ssd = theta_ssd
        self.hysteresis_gap = hysteresis_gap

    def compute_score(self, block: WeightBlockMeta) -> float:
        """Compute the composite score R(b) for a single block.

        Args:
            block: Weight block metadata.

        Returns:
            Composite residency priority score.
        """
        numerator = (
            self.w1 * block.reuse_score
            + self.w2 * block.routing_likelihood
            + self.w3 * block.layer_criticality
            + self.w4 * block.quality_sensitivity
        )
        denominator_us = self.alpha * block.decomp_cost_us + self.beta * block.transfer_cost_us
        normalized_denominator = max(
            MIN_NORMALIZED_DENOMINATOR,
            denominator_us / MICROSECONDS_PER_NORMALIZED_UNIT,
        )
        return numerator / max(normalized_denominator, MIN_COST_EPSILON)

    def score_all_blocks(self) -> dict[str, float]:
        """Compute scores for all blocks in the registry."""
        return {
            block.block_id: self.compute_score(block)
            for block in self.registry.get_all_blocks()
        }

    def rank_blocks_for_demotion(self) -> list[str]:
        """Return block identifiers sorted from lowest to highest score."""
        scores = self.score_all_blocks()
        return sorted(scores, key=lambda block_id: scores[block_id])

    def rank_blocks_for_promotion(self) -> list[str]:
        """Return block identifiers sorted from highest to lowest score."""
        scores = self.score_all_blocks()
        return sorted(scores, key=lambda block_id: scores[block_id], reverse=True)

    def get_target_state(self, block_id: str, hbm_pressure: float) -> int:
        """Determine the desired target state for a block.

        Args:
            block_id: Block identifier.
            hbm_pressure: Current HBM pressure fraction.

        Returns:
            Target residency state.
        """
        block = self.registry.get_block(block_id)
        score = self.compute_score(block)

        if score >= self.theta_high + self.hysteresis_gap and hbm_pressure < HIGH_PRESSURE_THRESHOLD:
            return STATE_HBM_FULL_PRECISION

        if score <= self.theta_ssd - self.hysteresis_gap or hbm_pressure >= CRITICAL_PRESSURE_THRESHOLD:
            return STATE_NVME

        if score <= self.theta_dram - self.hysteresis_gap or hbm_pressure >= HIGH_PRESSURE_THRESHOLD:
            return STATE_HOST_DRAM

        if block.current_state == STATE_HBM_FULL_PRECISION:
            if score < self.theta_high - self.hysteresis_gap or hbm_pressure >= HIGH_PRESSURE_THRESHOLD:
                return STATE_HBM_LOW_BIT
            return block.current_state

        if block.current_state == STATE_HOST_DRAM and score >= self.theta_dram + self.hysteresis_gap:
            return STATE_HBM_LOW_BIT

        if block.current_state == STATE_NVME and score >= self.theta_ssd + self.hysteresis_gap:
            return STATE_HOST_DRAM

        if score >= self.theta_dram + self.hysteresis_gap:
            return STATE_HBM_LOW_BIT

        return block.current_state
