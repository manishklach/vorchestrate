# Benchmarks

This directory now contains two benchmark paths:

- a synthetic controller benchmark scaffold
- a narrow real-model validation path for a small decoder-only model

## Current Benchmark Scope

Today the benchmark path is strongest on controller simulation, but it also includes one real-model validation step that runs a small GPT-2 style model and records observed runtime metrics plus controller-intended actions.

The current scaffold is intended to make the validation path visible and reproducible, not to stand in for production-grade benchmarking.

## Commands

Synthetic benchmark scaffold:

```bash
python benchmarks/benchmark_stub.py
```

Narrow real-model benchmark:

```bash
pip install -e .[dev,real-bench]
python benchmarks/real_model_benchmark.py --model-name distilgpt2
```

## What The Scaffold Does Today

- runs a synthetic controller simulation
- writes structured trace files under `benchmarks/results/`
- records a summary JSON with controller counters and event counts
- provides a narrow `distilgpt2`-style real-model benchmark path
- provides a place to standardize run configuration before real-model benchmarking is added

## What It Does Not Yet Do

- it does not measure real-model GPU performance
- it does not publish authoritative throughput or latency numbers
- it does not prove quality preservation on large production models
- it does not replace the future need for baseline comparisons against static quantization or naive offload

The narrow real-model benchmark is useful because it validates the adapter, instrumentation, and reporting path on a real forward pass. It is still not a broad systems benchmark.

## Future Real-Model Benchmarking Should Measure

- peak GPU memory
- host memory
- offload traffic
- prefetch count
- eviction count
- latency or throughput
- a quality proxy such as perplexity delta

See [../docs/benchmark_plan.md](../docs/benchmark_plan.md) for the staged validation path.
See [../docs/real_model_validation.md](../docs/real_model_validation.md) for the exact supported scope and command for the real-model path.
