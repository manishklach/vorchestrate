# vOrchestrate

> Dynamic weight residency orchestration for LLM inference across HBM, DRAM, and NVMe.

[![Status](https://img.shields.io/badge/status-early%20prototype-blue)](#current-status)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)](setup.py)
[![Patent](https://img.shields.io/badge/patent-IN%20202641039064-orange.svg)]()

vOrchestrate is an early runtime prototype and reference implementation for dynamic multi-tier weight residency and precision control during transformer inference. The repo is aimed at researchers, systems engineers, and infrastructure practitioners who care about memory hierarchy, offload policy, and inference-time control planes.

## Current Status

`Status: early prototype / research implementation`

The current repository demonstrates the controller shape rather than a finished production runtime.

- It implements the core control-plane concepts: block metadata tracking, block scoring, state transitions, guardrail logic, and scheduler scaffolding.
- It includes a CPU-testable reference implementation of the policy logic and small illustrative examples.
- It does not yet provide published large-model benchmark results or rigorous evidence of quality parity under aggressive demotion.
- The current examples are useful for understanding the orchestration model, but they should not be read as definitive end-to-end performance validation.

## Problem

Large language models are constrained by limited high-bandwidth GPU memory. In practice, deployment choices often collapse into a few imperfect options:

- keep more of the model in HBM and pay for larger GPUs
- use static quantization everywhere, including blocks that may be quality-sensitive
- rely on naive CPU or storage offload paths that are simple but latency-heavy

vOrchestrate explores a dynamic alternative: move the right blocks to the right memory tiers at the right time, based on observed reuse, estimated routing behavior, quality sensitivity, and transfer or decompression cost.

## Core Idea

The prototype models each weight block as a policy object with both a residency state and a precision or storage state. The controller can then reason about which blocks deserve scarce HBM capacity and which blocks can be staged in DRAM or NVMe until they are likely to be needed again.

The current implementation uses the following composite score:

```text
R(b) = (w1·ρ(b) + w2·λ(b) + w3·κ(b) + w4·ψ(b))
       ÷ (α·δ(b) + β·τ(b))
```

Where:

- `ρ(b)` is expected reuse score
- `λ(b)` is routing likelihood for MoE-style paths
- `κ(b)` is layer criticality
- `ψ(b)` is quality sensitivity
- `δ(b)` is decompression cost
- `τ(b)` is transfer cost

In the prototype, this score drives a policy-oriented controller model rather than a fully validated data-movement backend. The current implementation captures the policy and control-plane shape: block ranking, target-state selection, hysteresis-aware transitions, and guardrail-aware demotion decisions.

## Residency Model

vOrchestrate currently defines seven conceptual weight states:

| State | Meaning | Notes |
|-------|---------|-------|
| `S0` | Full-precision resident in HBM | Lowest access overhead |
| `S1` | Low-bit resident in HBM | Intended to preserve HBM locality with some precision tradeoff |
| `S2` | Compressed resident in HBM | Policy state for compressed-on-device residency |
| `S3` | Staged in host DRAM / CXL-like memory | Requires host transfer before compute |
| `S4` | Staged on NVMe | Cold tier with larger transfer penalty |
| `S5` | In-flight transfer | Transient bookkeeping state in the conceptual model |
| `S6` | Recomputable / derived | Extreme-pressure fallback state in the conceptual model |

These states are part of the controller model implemented in the prototype. Not every state is backed today by a real kernel path or storage backend.

## What The Repo Implements Today

The current codebase already contains a meaningful amount of scaffolding and policy logic:

- `WeightBlockRegistry` for thread-safe metadata tracking, access history, reuse updates, and HBM pressure accounting
- `ScoringEngine` for composite residency scoring and promotion or demotion ranking
- `AccuracyGuardrail` for sensitivity-aware veto logic that prevents blocks above a threshold from being demoted below low-bit HBM residency
- `WeightStateMachine` for transition decisions, transition logging, and hysteresis-aware policy execution
- `PrefetchScheduler` for async queue management and promotion-priority versus demotion-throttling behavior
- a partial Hugging Face-style wrapper path that instruments leaf modules and calls the control loop at layer intervals
- small benchmark helper utilities for registry memory summaries and perplexity-delta calculations
- illustrative examples and test coverage for the core control-plane modules

## What Is Not Yet Complete

This is where the repo should be read carefully:

- there is no full published benchmark suite yet
- there is no rigorous claim yet of quality parity on large production models
- the examples are illustrative and mostly small-scale
- the Hugging Face integration path should be read as partial and exploratory, not broad production support for arbitrary model families
- the current repository establishes the orchestration skeleton and policy surfaces, but full data-movement backends, CUDA kernels, and large-model validation remain future work

That limitation is intentional to state clearly. The goal is to make the project more trustworthy, not smaller.

## Status Matrix

| Area | Status | Notes |
|------|--------|-------|
| Scoring engine | Present | Composite block scoring and ranking logic are implemented |
| State machine | Present | Transition logic and transition history are implemented |
| Guardrail | Present | Sensitivity threshold veto logic is implemented |
| Scheduler scaffold | Present | Async queue, prioritization, and throttling scaffold are implemented |
| Toy examples | Present | Small illustrative examples are included |
| Hugging Face integration | Partial | Wrapper and layer-hook path exist, but broad compatibility is not yet established |
| Real offload backend | Planned | No production data-movement engine yet |
| Reproducible large-model benchmarks | Planned | Benchmark methodology is being documented and built out |

## Quick Start

### Small runnable example

The repo includes a small CPU-friendly example that demonstrates the wrapper shape on a toy transformer-like model:

```bash
python examples/basic_usage.py
```

That example exercises the orchestration scaffold and registry updates without claiming large-model support.

### Intended wrapper API shape

The Hugging Face integration in this repo is best read as the target API shape for transformer-style model integration:

```python
from transformers import AutoModelForCausalLM
from vorchestrate import VOrchestrate

model = AutoModelForCausalLM.from_pretrained("gpt2")
model = VOrchestrate(
    model,
    hbm_budget_gb=4.0,
    psi_threshold=0.7,
    tick_every_n_layers=4,
    enable_prefetch=True,
)
```

This wrapper path exists in the codebase, but broad end-to-end validation across model families is still future work.

## Architecture

```text
┌───────────────────────────── Inference Engine ─────────────────────────────┐
│ Transformer-style forward path issues layer and block accesses             │
└───────────────────────────────┬────────────────────────────────────────────┘
                                │ access telemetry
                                ▼
┌────────────────────────── Telemetry / Registry ────────────────────────────┐
│ WeightBlockRegistry stores access history, state, criticality, sensitivity │
└───────────────────────────────┬────────────────────────────────────────────┘
                                │ block metadata
                                ▼
┌──────────────────────────── Scoring Engine ────────────────────────────────┐
│ Composite score R(b) ranks promotion and demotion candidates               │
└───────────────┬───────────────────────────────┬────────────────────────────┘
                │                               │
                ▼                               ▼
        ┌───────────────┐               ┌─────────────────┐
        │ Guardrail     │               │ State Machine   │
        │ ψ-threshold   │──────────────▶│ transition logic│
        └───────────────┘               └────────┬────────┘
                                                 │ commands
                                                 ▼
                                  ┌────────────────────────────┐
                                  │ Scheduler / Prefetch Queue │
                                  └─────────────┬──────────────┘
                                                │ residency traffic
                       ┌────────────────────────┼─────────────────────────┐
                       ▼                        ▼                         ▼
                  HBM / GPU                 Host DRAM                 NVMe / SSD
```

For a more detailed controller walkthrough, see [docs/architecture.md](docs/architecture.md) and [docs/architecture.mmd](docs/architecture.mmd).

## Repository Layout

```text
vorchestrate/core          core controller logic: registry, scorer, guardrail, state machine, scheduler
vorchestrate/integrations  model integration scaffolding, including the current Hugging Face wrapper path
vorchestrate/benchmarks    lightweight benchmark helper utilities
examples                   runnable toy examples showing the intended control flow
tests                      pytest coverage for the core policy modules
docs                       architecture, roadmap, benchmark plan, limitations, and design notes
benchmarks                 benchmark planning notes and a stub entry point for future reproducible runs
```

## Benchmarks

Benchmark harness and first reproducible measurements are in progress.

The current repo does not publish authoritative performance numbers yet. Instead, it includes the initial benchmark planning docs and helper utilities needed to structure that work carefully.

| Model | Hardware | HBM Budget | Baseline Peak Memory | Controlled Peak Memory | Throughput Delta | Quality Delta | Status |
|-------|----------|------------|----------------------|------------------------|------------------|---------------|--------|
| `gpt2` | single GPU or CPU emulation | TBD | TBD | TBD | TBD | TBD | planned |
| `Llama-2-13B` | target single-GPU setup | TBD | TBD | TBD | TBD | TBD | planned |
| `Mixtral 8x7B` | target multi-tier offload study | TBD | TBD | TBD | TBD | TBD | planned |

See [docs/benchmark_plan.md](docs/benchmark_plan.md) and [benchmarks/README.md](benchmarks/README.md).

## Roadmap

Immediate next steps:

- validate the wrapper path on one real Hugging Face model with a documented demo flow
- build a reproducible benchmark harness with before-or-after memory traces
- compare policy variants against baseline, static quantization, and naive offload
- improve observability around transitions, queue activity, and residency pressure
- expand the docs and diagrams so the controller model is easier to inspect and critique

The phased roadmap is documented in [docs/roadmap.md](docs/roadmap.md).

## Patent

This repository is related to methods described in Indian Patent Application **IN 202641039064** — *System and Method for Predictive Multi-Tier Weight Residency and Precision Orchestration for Neural-Network Inference* — filed 29 March 2026.

The open-source implementation is released under Apache 2.0. Commercial licensing inquiries can be sent to `manishklach@gmail.com`.

## Contributing

Contributions are welcome, especially around instrumentation, trace collection, benchmark methodology, integration experiments, and careful policy evaluation.

Please read [CONTRIBUTING.md](CONTRIBUTING.md) first. If you share benchmark or integration results, include enough detail that another contributor can reproduce the setup.

## Author

**Manish KL**  
ML Systems Engineer · Bangalore, India  
GitHub: [@manishklach](https://github.com/manishklach)  
Email: `manishklach@gmail.com`

## License

Apache 2.0 — see [LICENSE](LICENSE).
