"""Tests for the weight block registry."""

import pytest

from vorchestrate.core import STATE_HBM_LOW_BIT, STATE_HOST_DRAM, WeightBlockRegistry


@pytest.fixture
def registry() -> WeightBlockRegistry:
    """Create a small registry fixture."""
    return WeightBlockRegistry(hbm_capacity_bytes=1000)


def test_register_block_returns_unique_ids(registry: WeightBlockRegistry) -> None:
    first = registry.register_block("layer", 100, criticality=0.5, sensitivity=0.5)
    second = registry.register_block("layer", 100, criticality=0.5, sensitivity=0.5)
    assert first != second


def test_register_block_rejects_non_positive_size(registry: WeightBlockRegistry) -> None:
    with pytest.raises(ValueError):
        registry.register_block("layer", 0, criticality=0.5, sensitivity=0.5)


def test_update_access_updates_last_access_and_prediction(registry: WeightBlockRegistry) -> None:
    block_id = registry.register_block("layer", 100, criticality=0.5, sensitivity=0.5)
    registry.update_access(block_id, 3)
    registry.update_access(block_id, 7)
    block = registry.get_block(block_id)
    assert block.last_access_step == 7
    assert block.predicted_next_access == 11


def test_update_reuse_score_increases_with_dense_accesses(registry: WeightBlockRegistry) -> None:
    block_id = registry.register_block("layer", 100, criticality=0.5, sensitivity=0.5)
    for step in [1, 2, 3, 4]:
        registry.update_access(block_id, step)
    block = registry.get_block(block_id)
    assert block.reuse_score > 0.1


def test_get_block_raises_for_unknown_id(registry: WeightBlockRegistry) -> None:
    with pytest.raises(KeyError):
        registry.get_block("missing")


def test_get_all_blocks_returns_registered_entries(registry: WeightBlockRegistry) -> None:
    registry.register_block("layer1", 100, criticality=0.2, sensitivity=0.3)
    registry.register_block("layer2", 200, criticality=0.4, sensitivity=0.5)
    assert len(registry.get_all_blocks()) == 2


def test_get_blocks_by_state_filters_correctly(registry: WeightBlockRegistry) -> None:
    hot = registry.register_block("hot", 100, criticality=0.6, sensitivity=0.5)
    cold = registry.register_block("cold", 100, criticality=0.6, sensitivity=0.5)
    registry.set_state(cold, STATE_HOST_DRAM)
    ids = [block.block_id for block in registry.get_blocks_by_state(STATE_HOST_DRAM)]
    assert hot not in ids
    assert cold in ids


def test_get_hbm_pressure_only_counts_hbm_states(registry: WeightBlockRegistry) -> None:
    hot = registry.register_block("hot", 200, criticality=0.5, sensitivity=0.5)
    cold = registry.register_block("cold", 300, criticality=0.5, sensitivity=0.5)
    registry.set_state(cold, STATE_HOST_DRAM)
    assert registry.get_hbm_pressure() == pytest.approx(0.2)
    registry.set_state(hot, STATE_HBM_LOW_BIT)
    assert registry.get_hbm_pressure() == pytest.approx(0.2)


def test_set_eviction_protection_flips_flag(registry: WeightBlockRegistry) -> None:
    block_id = registry.register_block("layer", 100, criticality=0.5, sensitivity=0.5)
    registry.set_eviction_protection(block_id, True)
    assert registry.get_block(block_id).eviction_protected is True


def test_update_reuse_score_rejects_invalid_window(registry: WeightBlockRegistry) -> None:
    block_id = registry.register_block("layer", 100, criticality=0.5, sensitivity=0.5)
    with pytest.raises(ValueError):
        registry.update_reuse_score(block_id, window_size=0)


def test_update_access_trims_history_to_window(registry: WeightBlockRegistry) -> None:
    block_id = registry.register_block("layer", 100, criticality=0.5, sensitivity=0.5)
    for step in range(40):
        registry.update_access(block_id, step)
    assert len(registry.get_block(block_id).access_history) == 32
