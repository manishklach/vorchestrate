# vOrchestrate

> Dynamic weight residency orchestration for LLM inference.  
> Run 70B models on half the GPU memory — without quality loss.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)](setup.py)
[![Patent](https://img.shields.io/badge/patent-IN%20202641039064-orange.svg)]()
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow.svg)]()

---

## The Problem

Running a 70B parameter model on a single A100 (80GB HBM) is already tight. Running it on a 40GB GPU is impossible — unless you sacrifice quality through aggressive static quantization or accept the latency penalty of naive CPU offloading.

The fundamental issue is that **current systems treat all weight blocks equally**. Every layer sits in HBM at full precision, all the time, regardless of whether it will be accessed in the next millisecond or the next minute.

That's wasteful. And it's solvable.

---

## The Solution

vOrchestrate is a **runtime weight residency orchestration controller** that continuously manages which weight blocks live in HBM, which are staged in CPU DRAM, and which are on NVMe — based on live inference telemetry.

It jointly optimizes two dimensions simultaneously:

- **Precision state** — FP16, INT8, INT4, compressed
- **Memory-residency state** — HBM, DRAM, SSD, recomputable

Using a composite scoring function computed at every layer boundary:

```text
R(b) = (w1·ρ(b) + w2·λ(b) + w3·κ(b) + w4·ψ(b))
       ÷ (α·δ(b) + β·τ(b))
```

Where ρ(b) is reuse probability, λ(b) is routing likelihood for MoE models, κ(b) is layer criticality, and ψ(b) is quality sensitivity. High R(b) means keep in HBM. Low R(b) means stage to DRAM or SSD.

---

## Seven Weight States

| State | Location | Precision | Latency |
|-------|----------|-----------|---------|
| S0 | HBM | FP16/BF16 | ~0 μs |
| S1 | HBM | INT4/INT8/FP8 | ~1–5 μs |
| S2 | HBM | Lossless compressed | ~5–50 μs |
| S3 | Host DRAM/CXL | Any | ~0.5–5 ms |
| S4 | NVMe SSD | Any | ~10–100 ms |
| S5 | In-flight | Target prec. | Transient |
| S6 | Recomputable | Derived | Fallback |

---

## Accuracy Guardrail

The most important feature. vOrchestrate maintains a **quality sensitivity score ψ(b)** for each weight block derived from PTQ calibration. Blocks exceeding the sensitivity threshold are **eviction-protected** — they cannot be demoted below INT8 regardless of HBM pressure.

This means quality never collapses under memory pressure. The system degrades gracefully — throughput drops before quality does.

---

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from vorchestrate import VOrchestrate

# Load any HuggingFace model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf")

# Wrap with vOrchestrate — set your HBM budget
model = VOrchestrate(
    model,
    hbm_budget_gb=20.0,
    psi_threshold=0.7,
    tick_every_n_layers=4,
    enable_prefetch=True,
)

# Use exactly as before — zero API changes
inputs = tokenizer("Explain quantum computing", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))
```

---

## Architecture

```text
┌─────────────────────────────────────────────────────┐
│                  INFERENCE ENGINE                    │
│              Transformer forward pass                │
└──────────────────────┬──────────────────────────────┘
                       │ weight requests
┌──────────────────────▼──────────────────────────────┐
│           WEIGHT ORCHESTRATION CONTROLLER            │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │Telemetry │  │ Scoring  │  │  State Machine   │  │
│  │Collector │→ │ Engine   │→ │  7 states S0-S6  │  │
│  │ρ·λ·κ·ψ   │  │R(b) per  │  │                  │  │
│  │          │  │block     │  │  ┌─────────────┐ │  │
│  └──────────┘  └──────────┘  │  │  Accuracy   │ │  │
│                               │  │  Guardrail  │ │  │
│  ┌────────────────────────┐   │  │  ψ(b) veto  │ │  │
│  │   SCHEDULING PIPELINE  │   │  └─────────────┘ │  │
│  │ Prefetch → Queue →     │   └──────────────────┘  │
│  │ Overlap with compute   │                          │
│  └────────────────────────┘                          │
└──────────────────────┬──────────────────────────────┘
                       │ read/write/transfer
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │HBM (GPU) │  │Host DRAM │  │NVMe SSD  │
  │S0/S1/S2  │  │S3 staged │  │S4 staged │
  │FP16-INT4 │  │50-200GB/s│  │6-14 GB/s │
  └──────────┘  └──────────┘  └──────────┘
```

---

## Benchmarks

*Results coming soon — community contributions welcome.*

| Model | GPU | Baseline Memory | vOrchestrate Memory | Throughput Delta |
|-------|-----|----------------|---------------------|------------------|
| Llama-2-70B | A100 80GB | 140GB | ~55GB | TBD |
| Mixtral 8x7B | A100 40GB | 90GB | ~35GB | TBD |
| Llama-2-13B | RTX 3090 | 26GB | ~12GB | TBD |

---

## Installation

```bash
pip install vorchestrate
```

Or from source:

```bash
git clone https://github.com/manishklach/vorchestrate
cd vorchestrate
pip install -e .[dev]
```

---

## Roadmap

- [x] Core scoring engine
- [x] Seven-state machine
- [x] Accuracy guardrail
- [x] Async prefetch scheduler
- [x] HuggingFace integration
- [ ] vLLM integration
- [ ] MoE router history buffer
- [ ] Benchmark suite
- [ ] INT4 precision pathway
- [ ] CXL memory tier support
- [ ] Multi-GPU distributed residency

---

## Patent

This library implements methods described in Indian Patent Application **IN 202641039064** — *System and Method for Predictive Multi-Tier Weight Residency and Precision Orchestration for Neural-Network Inference* — filed 29 March 2026.

The open-source implementation is released under Apache 2.0. Commercial licensing available — contact `manishklach@gmail.com`

---

## Contributing

PRs welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) first. Issues, benchmarks, and integration reports are especially appreciated.

---

## Author

**Manish KL**  
ML Systems Engineer · Bangalore, India  
GitHub: [@manishklach](https://github.com/manishklach)  
Email: `manishklach@gmail.com`

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
