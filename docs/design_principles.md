# Design Principles

These principles guide the project as it grows from controller prototype toward stronger validation.

## Graceful Degradation Before Quality Collapse

When memory pressure increases, the controller should prefer policies that preserve obviously sensitive units where possible, even if that means some throughput or staging cost.

## Dynamic Residency Over Static Residency

Residency is treated as a policy decision, not a one-time compile-time choice. The controller should adapt as access behavior changes.

## Guardrail-Aware Demotion

Demotion decisions should be aware of quality sensitivity. The guardrail exists to make that constraint explicit rather than hiding it inside opaque heuristics.

## Observability First

Scores, transitions, trace events, queue activity, and controller counters should be visible. Better traces are part of the architecture, not just debugging extras.

## Transparent Policy Over Opaque Heuristics

The repo should make policy terms, states, and decisions inspectable. Even when heuristics are used, they should remain legible enough to critique and improve.
