# Examples

The examples in this repository are designed to make the current controller scaffold easy to inspect. They exercise the logic honestly without implying broader runtime maturity than the code supports.

## Start Here

If you want the clearest picture of controller behavior today, start with `simulated_trace.py`. It shows how synthetic block descriptors move through scoring, guardrail-aware demotion, staging, and trace capture.

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

They are not a substitute for broader integration validation or benchmark-backed claims.
