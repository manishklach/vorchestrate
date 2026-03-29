"""Tests for trace output helpers."""

from __future__ import annotations

import json

from vorchestrate.utils.trace import TraceEvent, write_trace_csv, write_trace_json


def test_write_trace_json_preserves_event_shape(tmp_path) -> None:
    event = TraceEvent(
        step=1,
        block_id="block0",
        score=0.5,
        psi=0.7,
        old_state=1,
        new_state=3,
        old_tier="HBM",
        new_tier="DRAM",
        action="stage",
        guardrail_veto=False,
        bytes_moved=1024,
    )
    path = write_trace_json(tmp_path / "trace.json", [event])
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded[0]["block_id"] == "block0"
    assert loaded[0]["new_tier"] == "DRAM"


def test_write_trace_csv_writes_header_and_row(tmp_path) -> None:
    event = TraceEvent(
        step=2,
        block_id="block1",
        score=0.2,
        psi=0.1,
        old_state=3,
        new_state=1,
        old_tier="DRAM",
        new_tier="HBM",
        action="promote",
        guardrail_veto=False,
        bytes_moved=2048,
    )
    path = write_trace_csv(tmp_path / "trace.csv", [event])
    content = path.read_text(encoding="utf-8")
    assert "block_id" in content
    assert "block1" in content
