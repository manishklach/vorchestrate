"""Run the narrow real-model validation path for a small decoder-only model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vorchestrate.benchmarks.real_model import RealModelBenchmarkConfig, run_real_model_benchmark


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser for the real-model benchmark."""
    parser = argparse.ArgumentParser(
        description="Run a narrow real-model validation benchmark for a small decoder-only model."
    )
    parser.add_argument("--model-name", default="distilgpt2", help="Model name to load from Hugging Face")
    parser.add_argument("--device", default="auto", help="Device to use: auto, cpu, or cuda")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for the prompt")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Number of warmup forward passes")
    parser.add_argument("--measured-runs", type=int, default=2, help="Number of measured forward passes")
    parser.add_argument(
        "--hbm-budget-mb",
        type=int,
        default=64,
        help="Synthetic controller HBM budget in MiB used for controller decisions",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "real_model",
        help="Directory where benchmark artifacts will be written",
    )
    return parser


def main() -> None:
    """Run the narrow real-model benchmark and print artifact locations."""
    args = build_parser().parse_args()
    config = RealModelBenchmarkConfig(
        model_name=args.model_name,
        batch_size=args.batch_size,
        warmup_runs=args.warmup_runs,
        measured_runs=args.measured_runs,
        hbm_budget_bytes=args.hbm_budget_mb * 1024 * 1024,
        device=args.device,
        output_dir=str(args.output_dir),
    )
    result = run_real_model_benchmark(config)
    print("Completed narrow real-model benchmark.")
    print(f"output_dir: {result.output_dir}")
    print(f"summary: {result.output_dir / 'benchmark_summary.json'}")
    print(f"trace: {result.output_dir / 'benchmark_trace.json'}")
    if result.summary["artifacts"]["plots"]:
        print("plots:")
        for plot in result.summary["artifacts"]["plots"]:
            print(f"  {plot}")


if __name__ == "__main__":
    main()
