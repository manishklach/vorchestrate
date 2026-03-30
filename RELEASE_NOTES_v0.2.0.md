# v0.2.0 Release Notes

## Summary

v0.2.0 marks the first real validation-oriented release of vOrchestrate.

The repository remains an early prototype / research implementation, but this release materially improves its credibility and usefulness by adding one narrow, reproducible real-model benchmark path on top of the existing controller scaffold, simulation path, and documentation base.

## Highlights

### Narrow real-model validation path

- Added a constrained benchmark flow for a small decoder-only model, centered on `distilgpt2`
- Added a narrow adapter path for GPT-2 style decoder-only models
- Recorded observed runtime metrics on a real forward pass
- Recorded controller-intended actions through the current registry, scorer, guardrail, and state-machine path
- Wrote benchmark artifacts under `benchmarks/results/real_model/`

This is intentionally scoped. It does not claim broad Hugging Face compatibility or completed HBM/DRAM/NVMe movement backends.

### Synthetic controller visibility

- Added and refined the synthetic trace visualization path
- Generated plots for:
  - action counts
  - score over time
  - state timeline
  - synthetic HBM pressure over time
- Kept the visualization explicitly labeled as synthetic controller-simulation output

### Tooling and repo hardening

- Added CI with pytest, ruff, and mypy
- Moved packaging to a `pyproject.toml`-first setup
- Added Makefile targets for simulation, synthetic benchmarking, real benchmarking, linting, type checking, and trace rendering
- Added issue templates, PR template, code of conduct, and security policy

### Documentation improvements

- Reworked the README to reflect the repo's true maturity level
- Added architecture, benchmark-plan, limitations, design-principles, API, visualization, and real-model-validation docs
- Clarified the distinction between:
  - observed runtime metrics
  - controller-intended actions
  - future full movement backends

## New and Updated Components

### Benchmarks

- `benchmarks/real_model_benchmark.py`
- `benchmarks/benchmark_stub.py`
- `benchmarks/results/real_model/benchmark_summary.json`
- `benchmarks/results/real_model/run_rows.csv`
- `benchmarks/results/real_model/README.md`

### Integration surface

- `vorchestrate/integrations/decoder_only.py`
- configurable prototype heuristics remain available through `HeuristicProfile`

### Benchmark internals

- `vorchestrate/benchmarks/real_model.py`
- reusable summary and trace writing helpers for the narrow real-model path

### Visualization

- `examples/render_trace_report.py`
- `vorchestrate/utils/visualization.py`

## Validation Snapshot Included In This Release

This release includes a committed sample result snapshot from a narrow CPU benchmark run using:

- model: `distilgpt2`
- device: CPU
- batch size: 1
- measured runs: 1

The included snapshot is useful as a reproducible example of the reporting format. It should not be interpreted as a broad performance claim.

## What This Release Does Not Claim

This release does not claim:

- validated large-model memory savings
- broad quality-parity results
- universal Hugging Face support
- finished real memory-tier movement across HBM, DRAM, and NVMe

Those remain future work.

## Suggested Commands

Install development dependencies:

```bash
pip install -e .[dev]
```

Install the optional real-model benchmark dependency:

```bash
pip install -e .[dev,real-bench]
```

Run the synthetic controller simulation:

```bash
python examples/simulated_trace.py
python examples/render_trace_report.py
```

Run the narrow real-model benchmark:

```bash
python benchmarks/real_model_benchmark.py --model-name distilgpt2
```

## Future Work

The next milestones remain:

- broader adapter-backed experiments
- richer runtime instrumentation
- comparative baselines against static quantization and naive offload
- larger-model validation studies
- backend work toward real multi-tier movement execution
