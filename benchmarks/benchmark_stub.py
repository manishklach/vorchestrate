"""Synthetic benchmark scaffold for early controller-side measurements."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vorchestrate.utils.simulation import SimulationConfig, run_controller_simulation

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass(slots=True)
class BenchmarkConfig:
    """Configuration for the synthetic benchmark scaffold."""

    name: str = "synthetic_controller_trace"
    steps: int = 8
    hbm_budget_bytes: int = 20 * 1024 * 1024


def run_benchmark(config: BenchmarkConfig) -> Path:
    """Run the synthetic benchmark scaffold and persist outputs."""
    output_dir = DEFAULT_RESULTS_DIR / config.name
    simulation = run_controller_simulation(
        config=SimulationConfig(
            steps=config.steps,
            hbm_budget_bytes=config.hbm_budget_bytes,
            output_dir=str(output_dir),
        )
    )
    summary = {
        "benchmark": asdict(config),
        "metrics": simulation.metrics.to_dict(),
        "events": len(simulation.events),
        "score_steps": len(simulation.scores_by_step),
        "note": "Synthetic controller benchmark only. Does not represent real-model performance.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def main() -> None:
    """Run the synthetic benchmark stub and print output locations."""
    summary_path = run_benchmark(BenchmarkConfig())
    print("Synthetic benchmark scaffold complete.")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
