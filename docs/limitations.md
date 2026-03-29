# Limitations

This repository is intentionally transparent about current scope.

## Prototype Scope

vOrchestrate is a prototype control plane. It is not yet a production runtime for orchestrating arbitrary large-model deployments.

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

They are illustrative and useful, but they are not broad deployment proof.

## Validation Gaps

Still missing:

- published real-model benchmark suite
- stronger memory instrumentation on live model runs
- broader quality-evaluation evidence
- mature backends for actual movement across tiers
