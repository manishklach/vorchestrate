# Examples

The examples in this repository are designed to make the current controller scaffold easy to inspect. They exercise the logic honestly without implying broader runtime maturity than the code supports.

## Start Here

If you want the clearest picture of controller behavior today, start with `simulated_trace.py`. It shows how synthetic block descriptors move through scoring, guardrail-aware demotion, staging, and trace capture.

To turn that trace into simple figures, run `render_trace_report.py` after the simulation completes.

## `basic_usage.py`

- label: runnable prototype example
- purpose: wraps a small toy transformer-like module with `VOrchestrate`
- demonstrates: block registration, access tracking, and wrapper control flow

## `moe_usage.py`

- label: controller policy demonstration
- purpose: shows how routing likelihood affects scoring in a mixture-of-experts flavored scenario
- demonstrates: registry setup, metadata assignment, and ranking behavior

## `simulated_trace.py`

- label: controller simulation
- purpose: runs deterministic synthetic block descriptors through the existing controller path
- demonstrates: scoring, guardrails, stage or promote decisions, prefetch hints, trace writing, and metrics accumulation

## Scope Note

These examples are prototype exercises:

- `basic_usage.py` is a small runnable wrapper path
- `moe_usage.py` is a metadata-level policy demonstration
- `simulated_trace.py` is the clearest truthful example of the current controller behavior

They are not a substitute for broader integration validation or benchmark-backed claims.

## Visualizing The Controller Simulation

```bash
python examples/simulated_trace.py
python examples/render_trace_report.py
```

This produces a synthetic trace under `benchmarks/results/simulated_trace/` and PNG plots under `examples/output/simulated_trace_report/`.

Generated files include:

- `action_counts.png`
- `score_over_time.png`
- `state_timeline.png`
- `pressure_over_time.png`

What you will learn from these plots:

- which actions dominate a given synthetic scenario
- how average controller scores evolve over time
- how residency-managed units move across `S0`–`S6`
- how the synthetic HBM pressure signal relates to controller activity

These figures are synthetic controller-simulation output. They are meant to make policy behavior easier to inspect, not to stand in for real-model telemetry or benchmark evidence.
