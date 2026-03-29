"""Visualization helpers for synthetic controller traces."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

from .trace import TraceEvent


def load_trace_events(path: str | Path) -> list[TraceEvent]:
    """Load trace events from a JSON trace file."""
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return [TraceEvent(**record) for record in records]


def summarize_trace(events: Iterable[TraceEvent]) -> dict[str, object]:
    """Return simple aggregate summaries for a trace."""
    event_list = list(events)
    action_counts = Counter(event.action for event in event_list)
    guardrail_vetoes = sum(1 for event in event_list if event.guardrail_veto)
    score_by_step = defaultdict(list)
    for event in event_list:
        score_by_step[event.step].append(event.score)
    mean_score_by_step = {
        step: sum(scores) / len(scores)
        for step, scores in sorted(score_by_step.items())
    }
    return {
        "event_count": len(event_list),
        "action_counts": dict(action_counts),
        "guardrail_vetoes": guardrail_vetoes,
        "mean_score_by_step": mean_score_by_step,
    }


def render_trace_report(trace_path: str | Path, output_dir: str | Path) -> list[Path]:
    """Render simple synthetic-trace plots from a JSON trace file."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised via runtime path
        raise RuntimeError(
            "matplotlib is required to render trace reports. Install with `pip install -e .[dev]`."
        ) from exc

    events = load_trace_events(trace_path)
    summary = summarize_trace(events)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    action_counts = summary["action_counts"]
    if not isinstance(action_counts, dict):
        raise TypeError("expected action_counts to be a dictionary")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(list(action_counts.keys()), list(action_counts.values()), color="#3f7ea2")
    ax.set_title("Synthetic Controller Actions")
    ax.set_xlabel("Action")
    ax.set_ylabel("Count")
    fig.tight_layout()
    action_path = output_root / "action_counts.png"
    fig.savefig(action_path, dpi=160)
    plt.close(fig)
    output_paths.append(action_path)

    fig, ax = plt.subplots(figsize=(9, 5))
    block_names = sorted({event.block_id for event in events})
    block_index = {name: idx for idx, name in enumerate(block_names)}
    for event in events:
        ax.scatter(
            event.step,
            block_index[event.block_id],
            c=event.new_state,
            cmap="viridis",
            vmin=0,
            vmax=6,
            s=40,
        )
    ax.set_title("Residency State Transitions Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Residency-Managed Block")
    ax.set_yticks(list(block_index.values()), list(block_index.keys()))
    fig.tight_layout()
    state_path = output_root / "state_timeline.png"
    fig.savefig(state_path, dpi=160)
    plt.close(fig)
    output_paths.append(state_path)

    return output_paths
