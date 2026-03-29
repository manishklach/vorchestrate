"""Tests for synthetic trace visualization helpers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from vorchestrate.utils.trace import TraceEvent, write_trace_csv, write_trace_json
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
            hbm_pressure=0.35,
        ),
        TraceEvent(
            step=0,
            block_id="block_b",
            score=0.4,
            psi=0.2,
            old_state=1,
            new_state=3,
            old_tier="HBM",
            new_tier="DRAM",
            action="stage",
            guardrail_veto=False,
            bytes_moved=2048,
            hbm_pressure=0.62,
        ),
        TraceEvent(
            step=1,
            block_id="block_a",
            score=0.7,
            psi=0.7,
            old_state=1,
            new_state=1,
            old_tier="HBM",
            new_tier="HBM",
            action="keep",
            guardrail_veto=False,
            bytes_moved=0,
            hbm_pressure=0.41,
        ),
        TraceEvent(
            step=1,
            block_id="block_b",
            score=0.2,
            psi=0.9,
            old_state=3,
            new_state=3,
            old_tier="DRAM",
            new_tier="DRAM",
            action="guardrail_veto",
            guardrail_veto=True,
            bytes_moved=0,
            hbm_pressure=0.58,
        ),
    ]


def test_summarize_trace_counts_actions_vetoes_and_pressure() -> None:
    summary = summarize_trace(_sample_events())
    assert summary["action_counts"]["promote"] == 1
    assert summary["guardrail_vetoes"] == 1
    assert summary["mean_pressure_by_step"][0] > 0.0


def test_render_trace_report_creates_requested_pngs_from_json(tmp_path: Path) -> None:
    trace_path = write_trace_json(tmp_path / "trace.json", _sample_events())
    outputs = render_trace_report(trace_path, tmp_path / "report")
    output_names = {path.name for path in outputs}
    assert output_names == {
        "action_counts.png",
        "score_over_time.png",
        "state_timeline.png",
        "pressure_over_time.png",
    }
    assert all(path.exists() for path in outputs)


def test_render_trace_report_creates_requested_pngs_from_csv(tmp_path: Path) -> None:
    trace_path = write_trace_csv(tmp_path / "trace.csv", _sample_events())
    outputs = render_trace_report(trace_path, tmp_path / "report")
    assert len(outputs) == 4
    assert all(path.exists() for path in outputs)


def test_load_trace_events_round_trips_json(tmp_path: Path) -> None:
    trace_path = write_trace_json(tmp_path / "trace.json", _sample_events())
    loaded = load_trace_events(trace_path)
    assert loaded[0].block_id == "block_a"
    assert loaded[0].hbm_pressure == 0.35


def test_load_trace_events_round_trips_csv(tmp_path: Path) -> None:
    trace_path = write_trace_csv(tmp_path / "trace.csv", _sample_events())
    loaded = load_trace_events(trace_path)
    assert loaded[1].block_id == "block_b"
    assert loaded[1].new_state == 3


def test_render_trace_report_script_creates_outputs(tmp_path: Path) -> None:
    trace_path = write_trace_json(tmp_path / "trace.json", _sample_events())
    output_dir = tmp_path / "report"
    script_path = Path(__file__).resolve().parents[1] / "examples" / "render_trace_report.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--trace",
            str(trace_path),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "action_counts.png" in result.stdout
    assert (output_dir / "pressure_over_time.png").exists()
