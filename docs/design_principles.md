# Design Principles

These principles describe how the controller architecture is intended to evolve as the project moves from prototype scaffold toward stronger validation.

## Graceful Degradation Before Quality Collapse

When memory pressure increases, the controller should prefer policies that preserve obviously sensitive residency-managed units where possible, even if that means some throughput or staging cost.

## Dynamic Residency Over Static Residency

Residency is treated as a policy decision, not a one-time compile-time choice. The controller should adapt as access behavior and next-use distance change.

## Guardrail-Aware Demotion

Demotion decisions should be guardrail-aware. The sensitivity threshold exists to make that constraint explicit rather than burying it inside opaque heuristics.

## Observability First

Scores, transitions, trace events, queue activity, and controller counters should be visible. Better traces are part of the architecture, not an afterthought.

## Transparent Policy Over Opaque Heuristics

The repository should make policy terms, state transitions, and scoring decisions inspectable. Even when heuristics are used, they should remain legible enough to critique and improve.
