"""Tests for synthetic controller simulation."""

from __future__ import annotations

from vorchestrate.utils.simulation import SimulationConfig, run_controller_simulation


def test_simulation_produces_events_and_metrics() -> None:
    result = run_controller_simulation(config=SimulationConfig(steps=3))
    assert result.events
    assert result.metrics.to_dict()["transition_counts"] is not None


def test_simulation_records_valid_state_values() -> None:
    result = run_controller_simulation(config=SimulationConfig(steps=2))
    for event in result.events:
        assert 0 <= event.old_state <= 6
        assert 0 <= event.new_state <= 6
        assert 0.0 <= event.hbm_pressure


def test_simulation_accounts_for_guardrail_veto() -> None:
    result = run_controller_simulation(config=SimulationConfig(steps=4, hbm_budget_bytes=10 * 1024 * 1024))
    assert result.metrics.guardrail_vetoes >= 1
    assert any(event.guardrail_veto for event in result.events)
