"""Controller simulation using synthetic block descriptors."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vorchestrate.utils.simulation import SimulationConfig, run_controller_simulation

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "results" / "simulated_trace"


def main() -> None:
    """Run a deterministic controller simulation and print a summary."""
    config = SimulationConfig(output_dir=str(DEFAULT_OUTPUT_DIR))
    result = run_controller_simulation(config=config)

    print("vOrchestrate controller simulation using synthetic block descriptors")
    print(f"steps: {config.steps}")
    print(f"events: {len(result.events)}")
    print("metrics:")
    print(json.dumps(result.metrics.to_dict(), indent=2))
    print(f"trace_json: {DEFAULT_OUTPUT_DIR / 'simulation_trace.json'}")
    print(f"trace_csv: {DEFAULT_OUTPUT_DIR / 'simulation_trace.csv'}")
    print("sample actions:")
    for event in result.events[:8]:
        print(
            f"  step={event.step} block={event.block_id} action={event.action} "
            f"old=S{event.old_state} new=S{event.new_state} score={event.score:.3f}"
        )


if __name__ == "__main__":
    main()
