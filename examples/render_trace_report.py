"""Render simple visual reports from synthetic controller traces."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vorchestrate.utils.visualization import render_trace_report

DEFAULT_TRACE_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "results"
    / "simulated_trace"
    / "simulation_trace.json"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "results" / "simulated_trace_report"
)


def main() -> None:
    """Render a synthetic controller trace report."""
    output_paths = render_trace_report(DEFAULT_TRACE_PATH, DEFAULT_OUTPUT_DIR)
    print("Rendered synthetic controller trace report:")
    for path in output_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
