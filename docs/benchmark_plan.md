# Benchmark Plan

This document describes how vOrchestrate should be measured once the benchmark path is mature enough to publish repeatable numbers.

## Goals

The benchmark effort should answer a small number of concrete questions:

- does the controller reduce effective peak HBM usage relative to baseline?
- what extra host-memory pressure or storage traffic does that introduce?
- what is the throughput or latency cost of the policy?
- does the guardrail appear to preserve quality better than more naive demotion paths?

## Metrics

### Memory

Track:

- peak GPU memory
- average GPU memory over a run
- host-memory footprint
- logical bytes placed in HBM versus DRAM versus NVMe

### Data Movement

Track:

- bytes transferred between host and device
- estimated or measured NVMe traffic
- queue depth over time
- number of promotions and demotions
- time spent in transfer or staged states where available

### Performance

Track:

- tokens per second for decode-heavy runs
- end-to-end latency for fixed prompts
- tail latency if batching is involved

### Quality

Track at least one quality proxy:

- perplexity delta on a fixed evaluation slice
- or another reproducible task-level metric tied to a specific model and dataset

The repo already includes a small perplexity-delta helper, but the larger evaluation harness is still future work.

## Baselines

Each benchmark should compare against at least one of the following:

1. baseline model execution with no orchestration
2. static quantization baseline
3. naive offload baseline
4. vOrchestrate policy run

The exact comparison set can vary per experiment, but any published result should name the baselines clearly.

## Measurement Plan

For each run, capture:

- model name and parameter count
- hardware configuration
- PyTorch version and runtime environment
- prompt shape or sequence-length profile
- batch size
- HBM budget target
- policy configuration including thresholds and weights

## Reproducibility Expectations

Each reported result should include enough detail for another contributor to reproduce it:

- model and revision
- hardware setup
- software versions
- benchmark command
- dataset or prompt source
- exact policy settings
- warmup and measurement window

## Suggested Initial Path

Start with a small and inspectable workload:

1. validate bookkeeping on a toy model
2. validate wrapper behavior on a small Hugging Face model
3. collect before/after logical memory traces
4. only then move toward larger-model experiments

This ordering matters because it helps separate controller correctness from hardware-specific performance noise.
