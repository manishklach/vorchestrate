"""Tests for the weight state machine."""

import pytest

from vorchestrate.core import (
    AccuracyGuardrail,
    ScoringEngine,
    STATE_HBM_FULL_PRECISION,
    STATE_HBM_LOW_BIT,
    STATE_HOST_DRAM,
    STATE_NVME,
    WeightBlockRegistry,
    WeightStateMachine,
)


class DummyScheduler:
    """Collect scheduler commands during tests."""

    def __init__(self) -> None:
        """Initialize empty command lists."""
        self.promotions = []
        self.demotions = []

    def enqueue_promotion(self, block_id: str, priority: int) -> None:
        """Record a promotion request."""
        self.promotions.append((block_id, priority))

    def enqueue_demotion(self, block_id: str, priority: int) -> None:
        """Record a demotion request."""
        self.demotions.append((block_id, priority))


@pytest.fixture
def state_machine() -> WeightStateMachine:
    """Create a state machine with several blocks."""
    registry = WeightBlockRegistry(hbm_capacity_bytes=200)
    registry.register_block("hot", 100, criticality=0.9, sensitivity=0.3)
    registry.register_block("warm", 90, criticality=0.7, sensitivity=0.4)
    registry.register_block("fragile", 80, criticality=0.8, sensitivity=0.95)
    registry.get_block("hot:0").reuse_score = 0.9
    registry.get_block("hot:0").routing_likelihood = 0.9
    registry.get_block("hot:0").transfer_cost_us = 1.0
    registry.get_block("hot:0").decomp_cost_us = 1.0
    registry.get_block("warm:0").reuse_score = 0.1
    registry.get_block("warm:0").transfer_cost_us = 1000.0
    registry.get_block("warm:0").decomp_cost_us = 300.0
    registry.get_block("fragile:0").reuse_score = 0.05
    registry.get_block("fragile:0").transfer_cost_us = 5000.0
    registry.get_block("fragile:0").decomp_cost_us = 1500.0
    scheduler = DummyScheduler()
    return WeightStateMachine(
        registry=registry,
        scorer=ScoringEngine(registry),
        guardrail=AccuracyGuardrail(),
        scheduler=scheduler,
    )


def test_transition_executes_state_change(state_machine: WeightStateMachine) -> None:
    assert state_machine.transition("warm:0", STATE_HOST_DRAM) is True
    assert state_machine.registry.get_block("warm:0").current_state == STATE_HOST_DRAM


def test_transition_vetoes_guardrailed_demotion(state_machine: WeightStateMachine) -> None:
    assert state_machine.transition("fragile:0", STATE_HOST_DRAM) is False
    assert state_machine.registry.get_block("fragile:0").current_state == STATE_HBM_FULL_PRECISION


def test_promote_moves_state_up_one_level(state_machine: WeightStateMachine) -> None:
    state_machine.registry.set_state("warm:0", STATE_HOST_DRAM)
    assert state_machine.promote("warm:0") is True
    assert state_machine.registry.get_block("warm:0").current_state == STATE_HBM_LOW_BIT


def test_demote_moves_state_down_one_level(state_machine: WeightStateMachine) -> None:
    assert state_machine.demote("warm:0") is True
    assert state_machine.registry.get_block("warm:0").current_state == STATE_HBM_LOW_BIT


def test_tick_demotes_low_value_blocks_under_pressure(state_machine: WeightStateMachine) -> None:
    state_machine.tick(current_step=10, hbm_budget_bytes=150)
    assert state_machine.registry.get_hbm_pressure() <= 0.95
    assert state_machine.scheduler.demotions


def test_tick_enqueues_promotions_for_high_value_blocks(state_machine: WeightStateMachine) -> None:
    state_machine.registry.set_state("hot:0", STATE_HOST_DRAM)
    state_machine.tick(current_step=10, hbm_budget_bytes=300)
    assert state_machine.scheduler.promotions
    assert state_machine.registry.get_block("hot:0").current_state in {STATE_HBM_LOW_BIT, STATE_HBM_FULL_PRECISION}


def test_get_transition_history_records_entries(state_machine: WeightStateMachine) -> None:
    state_machine.transition("warm:0", STATE_HOST_DRAM)
    history = state_machine.get_transition_history()
    assert history
    assert history[-1]["block_id"] == "warm:0"


def test_transition_is_noop_when_target_matches_current(state_machine: WeightStateMachine) -> None:
    assert state_machine.transition("hot:0", STATE_HBM_FULL_PRECISION) is True


def test_transition_updates_precision_mapping(state_machine: WeightStateMachine) -> None:
    state_machine.transition("warm:0", STATE_NVME)
    assert state_machine.registry.get_block("warm:0").precision == "compressed"


def test_sensitive_block_gets_protection_flag_after_veto(state_machine: WeightStateMachine) -> None:
    state_machine.transition("fragile:0", STATE_NVME)
    assert state_machine.registry.get_block("fragile:0").eviction_protected is True
