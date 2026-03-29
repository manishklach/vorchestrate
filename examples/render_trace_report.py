"""Render visual reports from synthetic controller trace output."""

from __future__ import annotations

import argparse
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
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "simulated_trace_report"


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser for synthetic trace rendering."""
    parser = argparse.ArgumentParser(
        description="Render plots from synthetic controller-simulation traces."
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=DEFAULT_TRACE_PATH,
        help="Path to a JSON or CSV trace file produced by examples/simulated_trace.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where PNG plots will be written",
    )
    return parser


def main() -> None:
    """Render a synthetic controller trace report."""
    args = build_parser().parse_args()
    output_paths = render_trace_report(args.trace, args.output_dir)
    print("Rendered synthetic controller trace report from synthetic block descriptors:")
    for path in output_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
