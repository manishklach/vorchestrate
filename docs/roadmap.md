# vOrchestrate Roadmap

This roadmap is organized around credibility as much as capability. The goal is not just to add features, but to validate the controller against increasingly realistic workloads and measurement standards.

## Phase 1: Prototype Baseline

Current focus:

- establish the controller model
- make the core modules testable without GPU dependencies
- define the state model and scoring inputs clearly
- provide a first integration path and example scaffolding

Repository state today:

- core policy modules are present
- toy examples are present
- partial Hugging Face-style wrapper path is present
- benchmark and observability work are still early

## Phase 2: Validated Benchmark Path

Next milestone:

- build a reproducible benchmark harness
- measure logical HBM pressure versus baseline
- document host-memory and storage-traffic behavior
- compare policy variants under controlled settings
- publish small, clearly reproducible first results

Expected outputs:

- benchmark scripts and configuration docs
- fixed baseline comparisons
- before/after memory traces
- initial quality-proxy evaluation

## Phase 3: Stronger Runtime Backends

Planned work:

- move from policy skeleton to real movement and residency backends
- add explicit transfer or compression pathways
- enrich observability around queue activity and transition timing
- make the scheduler and state machine easier to inspect in live runs

## Phase 4: Broader Integrations

Longer-term direction:

- validate one Hugging Face model family carefully
- explore vLLM or adjacent serving integrations
- add MoE-aware routing history support
- support richer tiering policies, including CXL-like host tiers

## Phase 5: Comparative Studies

Longer-term comparative work:

- compare against baseline no-orchestration runs
- compare against static quantization
- compare against naive offload
- study policy sensitivity and hysteresis tuning
- analyze when quality guardrails meaningfully protect output quality

## Guiding Principle

Each phase should leave behind reproducible artifacts, not just broader claims.
