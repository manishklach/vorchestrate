"""Tests for synthetic trace visualization helpers."""

from __future__ import annotations

from vorchestrate.utils.trace import TraceEvent, write_trace_json
from vorchestrate.utils.visualization import load_trace_events, render_trace_report, summarize_trace


def _sample_events() -> list[TraceEvent]:
    return [
        TraceEvent(
            step=0,
            block_id="block_a",
            score=0.8,
            psi=0.7,
            old_state=3,
            new_state=1,
            old_tier="DRAM",
            new_tier="HBM",
            action="promote",
            guardrail_veto=False,
            bytes_moved=1024,
        ),
        TraceEvent(
            step=1,
            block_id="block_b",
            score=0.2,
            psi=0.9,
            old_state=1,
            new_state=1,
            old_tier="HBM",
            new_tier="HBM",
            action="guardrail_veto",
            guardrail_veto=True,
            bytes_moved=0,
        ),
    ]


def test_summarize_trace_counts_actions_and_vetoes() -> None:
    summary = summarize_trace(_sample_events())
    assert summary["action_counts"]["promote"] == 1
    assert summary["guardrail_vetoes"] == 1


def test_render_trace_report_creates_pngs(tmp_path) -> None:
    trace_path = write_trace_json(tmp_path / "trace.json", _sample_events())
    outputs = render_trace_report(trace_path, tmp_path / "report")
    assert len(outputs) == 2
    assert all(path.exists() for path in outputs)


def test_load_trace_events_round_trips_json(tmp_path) -> None:
    trace_path = write_trace_json(tmp_path / "trace.json", _sample_events())
    loaded = load_trace_events(trace_path)
    assert loaded[0].block_id == "block_a"
