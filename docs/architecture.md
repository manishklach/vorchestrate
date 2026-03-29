# vOrchestrate Architecture

vOrchestrate implements Predictive Multi-Tier Weight Residency Orchestration for transformer inference under constrained GPU memory.

## Core Flow

1. The `WeightBlockRegistry` tracks metadata for each weight block, including reuse history, routing likelihood, layer criticality, sensitivity, transfer costs, and current residency.
2. The `ScoringEngine` computes the composite score:

   `R(b) = (w1*rho + w2*lambda + w3*kappa + w4*psi) / (alpha*delta + beta*tau)`

3. The `AccuracyGuardrail` prevents sensitive blocks from being demoted below `S1`.
4. The `WeightStateMachine` uses the score and current HBM pressure to apply promotions and demotions.
5. The `PrefetchScheduler` processes promotion and demotion traffic in a background thread, prioritizing promotions and throttling demotions under bandwidth pressure.
6. The Hugging Face integration observes per-layer accesses and periodically calls the control loop during inference.

## State Semantics

- `S0`: FP16/BF16 resident in HBM
- `S1`: Low-bit resident in HBM
- `S2`: Lossless compressed resident in HBM
- `S3`: Staged in host DRAM/CXL
- `S4`: Staged on NVMe
- `S5`: In-flight transfer
- `S6`: Recomputable or derived fallback

## Design Notes

- CPU-first implementation: all logic is testable without a GPU.
- Thread-safe metadata registry for concurrent scheduler access.
- Clear seams for future CUDA kernels, profile-guided calibration, and vLLM-native hooks.
