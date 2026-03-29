# Visualization

The visualization path in this repository is designed for synthetic controller traces.

## What It Shows

The current report renderer can generate simple artifacts such as:

- action counts over a synthetic run
- residency state transitions over time by block

These figures are useful for inspecting controller behavior, validating trace output, and communicating how the policy scaffold responds to synthetic access patterns.

## How To Generate It

1. Run the controller simulation:

```bash
python examples/simulated_trace.py
```

2. Render the report:

```bash
python examples/render_trace_report.py
```

The images are written under `benchmarks/results/simulated_trace_report/`.

## What You Can And Cannot Conclude

You can use these visualizations to:

- inspect transition patterns
- check whether traces and actions look plausible
- compare controller behavior between synthetic scenarios

You should not use them to claim:

- real-model memory savings
- real-model throughput improvements
- production-quality telemetry coverage

They are a visibility layer for the controller simulation, not a substitute for real-model benchmarking.
