# vOrchestrate Architecture

vOrchestrate is currently best understood as a control-plane prototype for dynamic weight residency orchestration. The repo focuses on how a controller could represent, score, and transition weight blocks across memory tiers before a full production data path exists.

## Controller Overview

The prototype has five central pieces:

1. `WeightBlockRegistry`
2. `ScoringEngine`
3. `AccuracyGuardrail`
4. `WeightStateMachine`
5. `PrefetchScheduler`

Together, they form a policy loop:

1. Track weight-block metadata and access history.
2. Compute a residency priority score per block.
3. Apply guardrail constraints to sensitive blocks.
4. Decide target states under current HBM pressure.
5. Queue promotion or demotion activity.

This is implemented as a CPU-testable reference path in the current repo.

## Score Terms

The prototype uses the following composite score:

```text
R(b) = (w1·ρ(b) + w2·λ(b) + w3·κ(b) + w4·ψ(b))
       ÷ (α·δ(b) + β·τ(b))
```

Meaning of each term:

- `ρ(b)`: expected reuse score derived from recent access history
- `λ(b)`: routing likelihood, intended for MoE-style paths where block usage is sparse and route-dependent
- `κ(b)`: offline or static layer criticality estimate
- `ψ(b)`: quality sensitivity estimate used to avoid collapsing important blocks too aggressively
- `δ(b)`: decompression cost
- `τ(b)`: transfer cost

In the current implementation:

- reuse is updated from a sliding access history
- routing likelihood can be set on a per-block basis
- criticality and sensitivity are stored with each block
- decompression and transfer costs are represented as metadata fields used by the score and scheduler

## Weight States

The residency model currently defines seven states:

| State | Description | Prototype status |
|-------|-------------|------------------|
| `S0` | Full precision in HBM | represented in control logic |
| `S1` | Low-bit in HBM | represented in control logic |
| `S2` | Compressed in HBM | represented in control logic |
| `S3` | Host DRAM / CXL-style staging | represented in control logic |
| `S4` | NVMe staging | represented in control logic |
| `S5` | In-flight transfer | conceptual state, not yet a full transfer backend |
| `S6` | Recomputable / derived fallback | conceptual state, not yet a full recomputation backend |

The important point is that the state model already exists in the controller, even though several states still need real runtime backends behind them.

## Weight Block Lifecycle

The intended lifecycle of a block in the prototype looks like this:

1. A block is registered with metadata such as layer name, size, criticality, sensitivity, and initial state.
2. Accesses update recent history and predicted next-use timing.
3. The scorer recomputes a score from access behavior and cost metadata.
4. The state machine chooses whether the block should remain in place, be promoted, or be demoted.
5. The guardrail can veto demotion below `S1` for sufficiently sensitive blocks.
6. The scheduler can queue promotion or demotion activity and expose queue depth or bandwidth pressure estimates.

In other words, the current repo already captures the lifecycle of control decisions, even though it does not yet move real model weights across actual storage tiers in a production setting.

## Current Hugging Face Path

The `VOrchestrate` wrapper in `vorchestrate/integrations/huggingface.py` currently does the following:

- walks leaf modules with direct parameters
- registers them as weight blocks
- monkey-patches layer forwards to record access timing
- ticks the state machine every `N` layer calls

That makes it a meaningful integration scaffold, but not yet a hardened integration layer for arbitrary transformer implementations.

## What Is Implemented Today

Implemented in the current repo:

- thread-safe registry and per-block metadata model
- score computation and target-state heuristics
- hysteresis-aware transition logic
- sensitivity threshold guardrail
- async queue scaffold for promotions and demotions
- small benchmark helper functions
- tests covering the main control-plane modules

## What Remains Future Work

Still to be built or validated:

- real transfer backends across HBM, host memory, and NVMe
- integration with actual compression or quantization kernels
- measured residency traces on larger models
- published throughput and quality comparisons
- broader model-family compatibility testing
- richer observability and tracing exports

This distinction matters. The repository is already technically interesting, but it is still early enough that the right framing is “serious prototype” rather than “finished runtime.”
