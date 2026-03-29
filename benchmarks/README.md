# Benchmarks

This directory currently contains an early benchmark scaffold, not a published benchmark suite.

## What The Scaffold Does Today

- runs a synthetic controller simulation
- writes structured trace files under `benchmarks/results/`
- records a summary JSON with controller counters and event counts
- provides a place to standardize run configuration before real-model benchmarking is added

## What It Does Not Yet Do

- it does not measure real-model GPU performance
- it does not publish authoritative throughput or latency numbers
- it does not prove quality preservation on large production models
- it does not replace the future need for baseline comparisons against static quantization or naive offload

## Future Real-Model Benchmarking Should Measure

- peak GPU memory
- host memory
- offload traffic
- prefetch count
- eviction count
- latency or throughput
- a quality proxy such as perplexity delta

See [../docs/benchmark_plan.md](../docs/benchmark_plan.md) for the staged methodology.
