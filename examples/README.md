# Examples

The examples in this repository are intentionally small and illustrative.

They are designed to show the control-plane shape of vOrchestrate, not to act as proof of large-model production readiness.

## `basic_usage.py`

- type: toy runnable example
- purpose: wraps a very small transformer-like module with `VOrchestrate`
- demonstrates: block registration, access tracking, and the wrapper surface

Run:

```bash
python examples/basic_usage.py
```

## `moe_usage.py`

- type: policy demonstration
- purpose: shows how routing likelihood can affect block scoring
- demonstrates: registry usage, MoE-flavored metadata, and promotion ranking

Run:

```bash
python examples/moe_usage.py
```

## Scope Note

These examples should be read as prototype demonstrations. They help explain the intended orchestration model, but they do not yet establish large-model compatibility or benchmark-backed performance claims.
