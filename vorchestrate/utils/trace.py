"""Structured trace utilities for synthetic controller runs."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class TraceEvent:
    """A single controller trace record."""

    step: int
    block_id: str
    score: float
    psi: float
    old_state: int
    new_state: int
    old_tier: str
    new_tier: str
    action: str
    guardrail_veto: bool
    bytes_moved: int

    def to_dict(self) -> dict:
        """Return a JSON-serializable dictionary."""
        return asdict(self)


def write_trace_json(path: str | Path, events: Iterable[TraceEvent]) -> Path:
    """Write trace events to JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = [event.to_dict() for event in events]
    output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return output_path


def write_trace_csv(path: str | Path, events: Iterable[TraceEvent]) -> Path:
    """Write trace events to CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    event_list: list[TraceEvent] = list(events)
    fieldnames = list(TraceEvent.__dataclass_fields__.keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for event in event_list:
            writer.writerow(event.to_dict())
    return output_path
