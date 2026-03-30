# Narrow Real-Model Validation Snapshot

This directory contains one sample artifact from the first narrow real-model validation path in vOrchestrate.

## Scope

- model: `distilgpt2`
- supported path: narrow decoder-only GPT-2 style validation
- intent: validate adapter extraction, controller instrumentation, and reporting on a real forward pass

This is a concrete validation step, not a broad support claim.

## Command Used

```bash
python benchmarks/real_model_benchmark.py --model-name distilgpt2 --device cpu --warmup-runs 1 --measured-runs 1
```

## Environment Snapshot

- device: CPU
- CUDA available during run: false
- batch size: 1
- sequence length: 9 tokens
- prompt: `The controller observes memory pressure and adjusts residency.`

## Included Files

- `benchmark_summary.json`
- `run_rows.csv`

The full trace and plots can be regenerated locally by rerunning the benchmark with the optional `real-bench` dependency installed.

## Interpretation

The summary separates:

- observed runtime metrics such as latency
- controller-intended actions recorded through the prototype registry and state machine

Those controller actions are not evidence of completed HBM/DRAM/NVMe movement backends. They are the current controller decisions emitted on a real small-model forward path.
