"""Tests for the scoring engine."""

import pytest

from vorchestrate.core import STATE_HBM_FULL_PRECISION, STATE_HOST_DRAM, STATE_NVME, ScoringEngine, WeightBlockRegistry


@pytest.fixture
def registry() -> WeightBlockRegistry:
    """Create a populated registry fixture."""
    registry = WeightBlockRegistry(hbm_capacity_bytes=1000)
    hot = registry.register_block("hot", 100, criticality=0.9, sensitivity=0.8)
    cold = registry.register_block("cold", 100, criticality=0.2, sensitivity=0.1)
    medium = registry.register_block("medium", 100, criticality=0.5, sensitivity=0.5)
    registry.update_access(hot, 1)
    registry.update_access(hot, 2)
    registry.get_block(hot).routing_likelihood = 0.9
    registry.get_block(cold).transfer_cost_us = 4000.0
    registry.get_block(cold).decomp_cost_us = 1200.0
    registry.get_block(medium).routing_likelihood = 0.3
    return registry


def test_compute_score_returns_positive_value(registry: WeightBlockRegistry) -> None:
    scorer = ScoringEngine(registry)
    block = registry.get_block("hot:0")
    assert scorer.compute_score(block) > 0.0


def test_score_all_blocks_returns_all_ids(registry: WeightBlockRegistry) -> None:
    scorer = ScoringEngine(registry)
    scores = scorer.score_all_blocks()
    assert set(scores) == {"hot:0", "cold:0", "medium:0"}


def test_rank_blocks_for_demotion_orders_lowest_first(registry: WeightBlockRegistry) -> None:
    scorer = ScoringEngine(registry)
    ranking = scorer.rank_blocks_for_demotion()
    assert ranking[0] == "cold:0"


def test_rank_blocks_for_promotion_orders_highest_first(registry: WeightBlockRegistry) -> None:
    scorer = ScoringEngine(registry)
    ranking = scorer.rank_blocks_for_promotion()
    assert ranking[0] == "hot:0"


def test_get_target_state_promotes_hot_block_under_low_pressure(registry: WeightBlockRegistry) -> None:
    scorer = ScoringEngine(registry)
    assert scorer.get_target_state("hot:0", hbm_pressure=0.2) == STATE_HBM_FULL_PRECISION


def test_get_target_state_demotes_under_critical_pressure(registry: WeightBlockRegistry) -> None:
    scorer = ScoringEngine(registry)
    assert scorer.get_target_state("hot:0", hbm_pressure=0.99) == STATE_NVME


def test_get_target_state_selects_dram_for_mid_range_pressure(registry: WeightBlockRegistry) -> None:
    scorer = ScoringEngine(registry)
    assert scorer.get_target_state("medium:0", hbm_pressure=0.91) == STATE_HOST_DRAM


def test_constructor_rejects_invalid_weight_sum(registry: WeightBlockRegistry) -> None:
    with pytest.raises(ValueError):
        ScoringEngine(registry, w1=0.3, w2=0.3, w3=0.3, w4=0.3)


def test_compute_score_handles_zero_cost_denominator(registry: WeightBlockRegistry) -> None:
    scorer = ScoringEngine(registry)
    block = registry.get_block("medium:0")
    block.decomp_cost_us = 0.0
    block.transfer_cost_us = 0.0
    assert scorer.compute_score(block) > 0.0


def test_get_target_state_respects_hysteresis_near_threshold(registry: WeightBlockRegistry) -> None:
    scorer = ScoringEngine(registry)
    block = registry.get_block("medium:0")
    block.current_state = STATE_HOST_DRAM
    block.reuse_score = 0.42
    block.routing_likelihood = 0.42
    block.layer_criticality = 0.42
    block.quality_sensitivity = 0.42
    block.decomp_cost_us = 1.0
    block.transfer_cost_us = 1.0
    assert scorer.get_target_state("medium:0", hbm_pressure=0.5) == STATE_HOST_DRAM
