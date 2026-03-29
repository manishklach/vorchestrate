# v0.1.0

Initial public release of vOrchestrate.

## Highlights

- Introduces predictive multi-tier weight residency orchestration for transformer inference under constrained GPU memory.
- Implements the composite block scoring engine described in the project specification, combining reuse, routing likelihood, layer criticality, quality sensitivity, decompression cost, and transfer cost.
- Ships a seven-state residency model covering HBM full precision, low-bit HBM, compressed HBM, host DRAM, NVMe staging, in-flight transfer, and recomputable fallback.
- Adds an accuracy guardrail that prevents quality-sensitive blocks from being demoted below low-bit HBM residency.
- Includes an asynchronous prefetch scheduler with promotion prioritization and demotion throttling under bandwidth pressure.

## Included Components

- `WeightBlockRegistry` for thread-safe block metadata tracking and HBM pressure accounting
- `ScoringEngine` for computing `R(b)` and ranking promotion or demotion candidates
- `AccuracyGuardrail` for sensitivity-aware eviction vetoes
- `WeightStateMachine` for hysteresis-aware state transitions and transition logging
- `PrefetchScheduler` for background transfer queue management
- Hugging Face integration via `VOrchestrate`
- Benchmark helpers for memory profiling and perplexity delta checks
- CPU-compatible examples, architecture documentation, and production packaging metadata

## Quality and Validation

- 50 automated pytest checks covering the registry, scorer, guardrail, scheduler, and state machine
- CPU-first design so the orchestration logic can be developed and validated without requiring a GPU
- Public package metadata and Apache 2.0 licensing included from the initial release

## Known Gaps

- vLLM integration is currently a stub
- Benchmark result tables are placeholders pending hardware-backed measurements
- Advanced paths such as CXL tiers, INT4-specific flows, MoE history buffering, and multi-GPU orchestration are planned for future releases
