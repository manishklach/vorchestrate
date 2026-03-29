# Benchmark Work In Progress

This directory is reserved for reproducible benchmark entry points and run notes.

The current repository does not yet publish benchmark results as settled facts. Instead, the goal is to build a careful path toward credible measurements.

## First Benchmark Flow

1. Validate the controller on the toy example path.
2. Validate the wrapper on one small Hugging Face model.
3. Capture baseline memory and latency traces.
4. Compare against the orchestrated run under the same prompt and hardware settings.
5. Record both policy settings and observed trace outputs.

See [../docs/benchmark_plan.md](../docs/benchmark_plan.md) for the broader methodology.
