# Current Limitations

vOrchestrate is intentionally published as an early-stage prototype. The main limitations today are listed here so the project can be evaluated honestly.

## Runtime Scope

- the repository currently implements control-plane logic more than full data-plane execution
- there is no production-ready backend for moving real model weights across HBM, DRAM, and NVMe tiers
- several residency states are represented in the controller but not yet backed by a full runtime path

## Integration Scope

- the Hugging Face wrapper is partial and exploratory
- compatibility across arbitrary model architectures is not yet established
- the vLLM integration remains a stub

## Validation Scope

- there is no published large-model benchmark suite yet
- there is no current claim of quality parity on production-scale models
- the examples in the repository are illustrative and mostly toy-sized

## Measurement Scope

- queue and transfer behavior are modeled, but not yet tied to a real production offload backend
- compression, quantization, and decompression pathways are represented at the metadata or policy level rather than through fully integrated kernels

## Why Publish At This Stage

Even with these limitations, the repository is useful because it makes the orchestration model inspectable:

- the state model is explicit
- the scoring policy is explicit
- the guardrail logic is explicit
- the transition loop is testable

That makes the repo a good place to refine the architecture, instrument real experiments, and attract collaborators who care about memory-aware inference systems.
