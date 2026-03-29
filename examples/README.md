# Examples

The examples in this repository are deliberately honest about scope. They are useful for exercising the controller logic and illustrating intended integration shape, but they should not be read as proof of production maturity.

## `basic_usage.py`

- label: runnable prototype example
- purpose: wraps a small toy transformer-like module with `VOrchestrate`
- demonstrates: block registration, access tracking, and wrapper control flow

## `moe_usage.py`

- label: controller policy demonstration
- purpose: shows how routing likelihood affects scoring in a mixture-of-experts flavored scenario
- demonstrates: registry setup, metadata assignment, and ranking behavior

## `simulated_trace.py`

- label: controller simulation
- purpose: runs deterministic synthetic block descriptors through the existing controller path
- demonstrates: scoring, guardrails, stage or promote decisions, prefetch hints, trace writing, and metrics accumulation

## Scope Note

These examples are prototype exercises:

- `basic_usage.py` is a small runnable wrapper path
- `moe_usage.py` is a metadata-level policy demonstration
- `simulated_trace.py` is the clearest truthful example of the current controller behavior

They are not a substitute for broader integration validation or benchmark-backed performance claims.
