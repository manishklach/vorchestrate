# Visualization

The visualization pipeline in this repository is designed for synthetic controller traces. It turns the output of the controller simulation into a small set of plots that make policy behavior easier to inspect.

These figures are useful for understanding the current prototype scaffold. They are not real-model telemetry and they are not hardware benchmark results.

## What The Pipeline Does

The renderer consumes the JSON or CSV trace emitted by `examples/simulated_trace.py` and writes PNG plots to an output directory.

```bash
python examples/simulated_trace.py
python examples/render_trace_report.py
```

By default this produces:

- `benchmarks/results/simulated_trace/simulation_trace.json`
- `benchmarks/results/simulated_trace/simulation_trace.csv`
- `examples/output/simulated_trace_report/action_counts.png`
- `examples/output/simulated_trace_report/score_over_time.png`
- `examples/output/simulated_trace_report/state_timeline.png`
- `examples/output/simulated_trace_report/pressure_over_time.png`

You can also point the renderer at a specific trace file:

```bash
python examples/render_trace_report.py --trace benchmarks/results/simulated_trace/simulation_trace.csv
```

## What Each Plot Means

### `action_counts.png`

Shows how often the controller simulation emitted actions such as `keep`, `promote`, `prefetch`, `stage`, and `guardrail_veto`.

This is useful for spotting broad policy shape:

- whether the run is mostly stable or highly reactive
- whether prefetch hints are occurring at all
- whether guardrail vetoes are rare or frequent

### `score_over_time.png`

Shows the average controller score over time across the synthetic residency-managed units in the trace.

This is useful for:

- seeing whether the workload is becoming hotter or colder on average
- comparing synthetic scenarios after policy changes

### `state_timeline.png`

Shows a heatmap-style timeline of block states over steps, with blocks on one axis and steps on the other. The color encodes states `S0` through `S6`.

This is useful for:

- understanding how the controller moves units across the residency ladder
- spotting oscillation or stable placement regions
- checking whether expected hot blocks stay closer to HBM

### `pressure_over_time.png`

Shows the average synthetic HBM pressure signal over time. In the current trace format this is derived from the registry's HBM occupancy fraction at the moment each trace event is recorded.

This is useful for:

- relating controller actions to pressure conditions in the simulation
- checking whether colder-tier staging and promotions cluster around pressure spikes

## What You Can Conclude

You can use these plots to:

- inspect the internal behavior of the controller scaffold
- compare policy changes on the same synthetic scenario
- validate that trace fields, scoring, and transition logic are behaving plausibly

## What You Cannot Conclude

You should not use these figures to claim:

- real-model memory savings
- measured throughput gains on hardware
- validated quality preservation
- production runtime behavior under live inference load

Those conclusions require real-model experiments, instrumentation, and reproducible benchmark runs. See [docs/benchmark_plan.md](./benchmark_plan.md) for that validation path.
