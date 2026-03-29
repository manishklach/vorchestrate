"""Visualization helpers for synthetic controller traces."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

from .trace import TraceEvent

ACTION_ORDER = ["keep", "promote", "demote", "prefetch", "stage", "guardrail_veto"]
STATE_MIN = 0
STATE_MAX = 6


def load_trace_events(path: str | Path) -> list[TraceEvent]:
    """Load trace events from a JSON or CSV trace file."""
    trace_path = Path(path)
    if trace_path.suffix.lower() == ".json":
        records = json.loads(trace_path.read_text(encoding="utf-8"))
        return [TraceEvent(**record) for record in records]
    if trace_path.suffix.lower() == ".csv":
        with trace_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return [_trace_event_from_row(row) for row in reader]
    raise ValueError(f"unsupported trace format: {trace_path.suffix}")


def summarize_trace(events: Iterable[TraceEvent]) -> dict[str, object]:
    """Return simple aggregate summaries for a trace."""
    event_list = list(events)
    action_counts = Counter(event.action for event in event_list)
    guardrail_vetoes = sum(1 for event in event_list if event.guardrail_veto)
    score_by_step = defaultdict(list)
    pressure_by_step = defaultdict(list)
    for event in event_list:
        score_by_step[event.step].append(event.score)
        pressure_by_step[event.step].append(event.hbm_pressure)
    mean_score_by_step = {
        step: sum(scores) / len(scores) for step, scores in sorted(score_by_step.items())
    }
    mean_pressure_by_step = {
        step: sum(values) / len(values) for step, values in sorted(pressure_by_step.items())
    }
    return {
        "event_count": len(event_list),
        "action_counts": dict(action_counts),
        "guardrail_vetoes": guardrail_vetoes,
        "mean_score_by_step": mean_score_by_step,
        "mean_pressure_by_step": mean_pressure_by_step,
    }


def render_trace_report(trace_path: str | Path, output_dir: str | Path) -> list[Path]:
    """Render synthetic controller trace plots from a JSON or CSV trace file."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except ImportError as exc:  # pragma: no cover - exercised via runtime path
        raise RuntimeError(
            "matplotlib is required to render trace reports. Install with `pip install -e .[dev]`."
        ) from exc

    events = load_trace_events(trace_path)
    summary = summarize_trace(events)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    output_paths.append(_render_action_counts(plt, summary, output_root))
    output_paths.append(_render_score_over_time(plt, summary, output_root))
    output_paths.append(_render_state_timeline(plt, Normalize, events, output_root))
    output_paths.append(_render_pressure_over_time(plt, summary, output_root))
    return output_paths


def _render_action_counts(plt: object, summary: dict[str, object], output_root: Path) -> Path:
    action_counts = summary["action_counts"]
    if not isinstance(action_counts, dict):
        raise TypeError("expected action_counts to be a dictionary")
    labels = [action for action in ACTION_ORDER if action in action_counts or action == "keep"]
    values = [int(action_counts.get(action, 0)) for action in labels]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color="#3f7ea2")
    ax.set_title("Synthetic Controller Action Counts")
    ax.set_xlabel("Action")
    ax.set_ylabel("Count")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = output_root / "action_counts.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _render_score_over_time(plt: object, summary: dict[str, object], output_root: Path) -> Path:
    mean_score_by_step = summary["mean_score_by_step"]
    if not isinstance(mean_score_by_step, dict):
        raise TypeError("expected mean_score_by_step to be a dictionary")
    steps = list(mean_score_by_step.keys())
    values = [float(mean_score_by_step[step]) for step in steps]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(steps, values, marker="o", color="#2b7a78", linewidth=2)
    ax.set_title("Average Controller Score Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean score across residency-managed units")
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = output_root / "score_over_time.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _render_state_timeline(
    plt: object,
    normalize_cls: object,
    events: list[TraceEvent],
    output_root: Path,
) -> Path:
    block_names = sorted({event.block_id for event in events})
    steps = sorted({event.step for event in events})
    block_index = {name: idx for idx, name in enumerate(block_names)}
    step_index = {step: idx for idx, step in enumerate(steps)}
    matrix = [[float("nan") for _ in steps] for _ in block_names]

    for event in events:
        matrix[block_index[event.block_id]][step_index[event.step]] = float(event.new_state)

    fig, ax = plt.subplots(figsize=(9, max(4.5, len(block_names) * 0.7)))
    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        norm=normalize_cls(vmin=STATE_MIN, vmax=STATE_MAX),
    )
    ax.set_title("Synthetic Residency State Timeline")
    ax.set_xlabel("Step")
    ax.set_ylabel("Residency-managed block")
    ax.set_xticks(range(len(steps)), [str(step) for step in steps])
    ax.set_yticks(range(len(block_names)), block_names)
    cbar = fig.colorbar(image, ax=ax, ticks=list(range(STATE_MIN, STATE_MAX + 1)))
    cbar.set_label("State (S0-S6)")
    fig.tight_layout()
    path = output_root / "state_timeline.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _render_pressure_over_time(plt: object, summary: dict[str, object], output_root: Path) -> Path:
    mean_pressure_by_step = summary["mean_pressure_by_step"]
    if not isinstance(mean_pressure_by_step, dict):
        raise TypeError("expected mean_pressure_by_step to be a dictionary")
    steps = list(mean_pressure_by_step.keys())
    values = [float(mean_pressure_by_step[step]) for step in steps]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(steps, values, marker="o", color="#d97706", linewidth=2)
    ax.set_title("Synthetic HBM Pressure Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean HBM pressure fraction")
    ax.set_ylim(0.0, max(1.0, max(values, default=0.0) * 1.1))
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = output_root / "pressure_over_time.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _trace_event_from_row(row: dict[str, str]) -> TraceEvent:
    """Build a trace event from a CSV row."""
    return TraceEvent(
        step=int(row["step"]),
        block_id=row["block_id"],
        score=float(row["score"]),
        psi=float(row["psi"]),
        old_state=int(row["old_state"]),
        new_state=int(row["new_state"]),
        old_tier=row["old_tier"],
        new_tier=row["new_tier"],
        action=row["action"],
        guardrail_veto=row["guardrail_veto"].strip().lower() == "true",
        bytes_moved=int(row["bytes_moved"]),
        hbm_pressure=float(row.get("hbm_pressure", "0.0")),
    )
