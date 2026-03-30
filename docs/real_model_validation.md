# Real-Model Validation

This document describes the first narrow real-model validation path in vOrchestrate.

## What This Benchmark Validates

The current benchmark validates one carefully scoped bridge between the controller scaffold and a real model:

- loading a small decoder-only causal language model
- enumerating residency-managed units through a narrow adapter
- running real forward passes on CPU or CUDA
- recording observed runtime metrics such as latency and memory
- recording controller-intended actions through the registry, scorer, guardrail, and state-machine path

This is meant to improve the repo's credibility by grounding the controller on a real small-model execution path without pretending that the full multi-tier runtime is already complete.

## Supported Scope

The current validation path is intentionally narrow:

- preferred model: `distilgpt2`
- supported family: small GPT-2 style decoder-only models with `transformer.h` blocks
- intended use: single-process prototype benchmarking and instrumentation

This does not imply universal Hugging Face support.

## What It Does Not Validate

This benchmark does not yet validate:

- broad model-family compatibility
- production-quality HBM, DRAM, or NVMe movement backends
- large-model quality preservation claims
- comparative wins against static quantization or naive offload baselines

The controller actions emitted by this benchmark are controller intentions recorded through the prototype abstractions. They are not proof that real memory-tier movement has been executed.

## Commands

Install the optional benchmark dependency:

```bash
pip install -e .[dev,real-bench]
```

Run the benchmark:

```bash
python benchmarks/real_model_benchmark.py --model-name distilgpt2
```

Or with the Makefile:

```bash
make benchmark-real
```

## Artifacts

By default the benchmark writes to `benchmarks/results/real_model/`:

- `benchmark_summary.json`
- `benchmark_trace.json`
- `benchmark_trace.csv`
- `run_rows.csv`
- `plots/`

The summary separates:

- observed runtime metrics
- controller-intended actions and counters

That distinction is important because the current backend remains a controller prototype rather than a full movement runtime.

## Hardware Assumptions

The script runs on CPU or CUDA.

- on CUDA it records peak GPU memory via PyTorch memory statistics
- on CPU it still records latency and controller traces
- host memory is reported when the platform exposes it cleanly enough through standard-library mechanisms

## Interpretation Guidance

This benchmark is useful for:

- validating the narrow adapter path
- checking that the controller records plausible decisions on a real forward pass
- comparing small configuration changes on the same small-model setup

It should not be used to claim:

- general support across arbitrary transformer stacks
- production memory savings
- validated quality parity

It is a first real-model validation step in a broader benchmark path, not the end state.
