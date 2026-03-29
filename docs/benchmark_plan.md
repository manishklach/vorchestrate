# Benchmark Plan

This document describes how vOrchestrate should be benchmarked in a way that is useful and believable.

## Benchmark Goals

The benchmark path should answer a few concrete questions:

- can the controller reduce peak GPU memory pressure relative to baseline?
- what host-memory or storage traffic does that introduce?
- what prefetch and eviction behavior does the controller generate?
- what latency or throughput cost follows from those decisions?
- does the controller appear to preserve quality better than simpler fallback strategies?

## Metrics To Measure

For real-model benchmarks, the target metrics include:

- peak GPU memory
- host memory
- offload traffic
- prefetch count
- eviction count
- latency or throughput
- quality proxy

Examples of a quality proxy:

- perplexity delta
- task loss delta
- another fixed evaluation metric for a clearly documented workload

## Baseline Comparisons

Every published benchmark should identify its baseline clearly. The intended comparison set is:

1. no orchestration
2. static quantized setup
3. naive offload
4. controller-driven strategy

Not every experiment must include every baseline immediately, but the full benchmark story should eventually cover them.

## Reproducibility Rules

Any reported result should capture:

- model and revision
- hardware setup
- software versions
- prompt or dataset details
- warmup procedure
- measurement window
- controller configuration
- command used to run the benchmark

## Phased Validation Plan

### Phase A: Synthetic Traces

Start with deterministic synthetic traces to validate:

- scoring behavior
- guardrail decisions
- transition counts
- trace output shape

### Phase B: Small Real Models

Move next to one or more small real models to validate:

- instrumentation path
- memory accounting
- latency impact
- first quality proxy comparisons

### Phase C: Larger Models

Only after the above is stable:

- expand to larger-model experiments
- compare multiple policy settings
- publish memory, traffic, throughput, and quality tradeoffs carefully

## Current State

The current repo supports Phase A best. Phase B and Phase C are still future work.
