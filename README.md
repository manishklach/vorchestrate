# vOrchestrate [![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Predictive multi-tier weight residency orchestration for transformer inference under constrained GPU memory.

## Installation

```bash
pip install vorchestrate
```

For local development:

```bash
git clone https://github.com/manishklach/vorchestrate.git
cd vorchestrate
pip install -e .[dev]
```

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from vorchestrate import VOrchestrate

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model = VOrchestrate(model, hbm_budget_gb=4.0)

inputs = tokenizer("Hello from vOrchestrate", return_tensors="pt")
outputs = model(**inputs)
print(outputs.logits.shape)
```

## Architecture

```text
                    +----------------------+
                    |  HuggingFace Model   |
                    |  Layer Access Hooks  |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | WeightBlockRegistry  |
                    | rho, lambda, kappa,  |
                    | psi, delta, tau      |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |   ScoringEngine      |
                    |      R(b) score      |
                    +----------+-----------+
                               |
               +---------------+----------------+
               |                                |
               v                                v
      +------------------+             +------------------+
      | AccuracyGuardrail|             |PrefetchScheduler |
      | psi threshold    |             | async queue      |
      +--------+---------+             +--------+---------+
               |                                |
               +---------------+----------------+
                               v
                    +----------------------+
                    | WeightStateMachine   |
                    | S0 ... S6 transitions|
                    +----------------------+
```

## Core Invention

The library implements the predictive residency score:

```text
R(b) = (w1·rho(b) + w2·lambda(b) + w3·kappa(b) + w4·psi(b))
       / (alpha·delta(b) + beta·tau(b))
```

This jointly decides both precision and memory residency for each weight block across:

- `S0` full precision in HBM
- `S1` low-bit in HBM
- `S2` compressed in HBM
- `S3` staged in host DRAM/CXL
- `S4` staged on NVMe
- `S5` in-flight transfer
- `S6` recomputable fallback

## Benchmark Results

Benchmark harnesses live under `vorchestrate/benchmarks/`. Placeholder sections for published benchmark tables:

- HBM usage versus token throughput
- PCIe and NVMe traffic profiles
- Perplexity delta under aggressive demotion

## Citation

If you build on this work, please cite the invention disclosure:

```text
Manish Lachwani. Predictive Multi-Tier Weight Residency Orchestration
for Neural Network Inference. Patent filing IN 202641039064.
```

## Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a focused branch.
3. Add tests for any behavior change.
4. Run `pytest`.
5. Open a pull request describing the motivation and results.

## License

Apache License 2.0. See [LICENSE](LICENSE).
