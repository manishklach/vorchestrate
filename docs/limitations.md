# Limitations

This repository is intentionally clear about current scope.

## Prototype Scope

vOrchestrate is a prototype control plane. It is not yet a production runtime for arbitrary large-model deployments, but it does establish a concrete architecture for reasoning about dynamic residency decisions.

## No Production Claim

The repository does not claim:

- production readiness
- broad large-model support
- benchmark-proven quality preservation
- benchmark-proven memory or throughput gains

## Limited Integration Surface

- the Hugging Face wrapper is exploratory
- the adapter base class is intentionally abstract
- broader model-family support is still future work

## Illustrative Examples

The examples are designed to exercise the controller honestly:

- small toy wrapper path
- MoE-style metadata example
- synthetic controller simulation

They are useful for inspecting controller behavior, but they are not broad deployment proof.

## Validation Gaps

Still missing:

- published real-model benchmark suite
- stronger memory instrumentation on live model runs
- broader quality-evaluation evidence
- mature backends for actual movement across tiers

These limitations are a boundary on current claims, not a statement that the controller architecture lacks value.
