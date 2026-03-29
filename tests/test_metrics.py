"""Tests for controller metrics accumulation."""

from __future__ import annotations

from vorchestrate.core import ControllerMetrics


def test_metrics_accumulate_transitions_and_bytes() -> None:
    metrics = ControllerMetrics()
    metrics.record_transition(old_state=3, new_state=1, size_bytes=4096)
    metrics.record_transition(old_state=1, new_state=3, size_bytes=2048)
    assert metrics.promotions == 1
    assert metrics.demotions == 1
    assert metrics.bytes_promoted == 4096
    assert metrics.bytes_demoted == 2048


def test_metrics_record_guardrail_and_prefetch_counts() -> None:
    metrics = ControllerMetrics()
    metrics.record_prefetch()
    metrics.record_stage()
    metrics.record_guardrail_veto()
    snapshot = metrics.to_dict()
    assert snapshot["prefetches"] == 1
    assert snapshot["stages"] == 1
    assert snapshot["guardrail_vetoes"] == 1
