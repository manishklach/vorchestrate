# Benchmark Plan

This document describes how vOrchestrate should be benchmarked in a way that is methodical, comparable, and publishable.

## Benchmark Goals

The benchmark path should answer a small set of concrete systems questions:

- can the controller reduce peak GPU memory pressure relative to baseline?
- what host-memory or storage traffic does that introduce?
- what prefetch and eviction behavior does the controller generate?
- what latency or throughput cost follows from those decisions?
- does the controller appear to preserve quality better than simpler fallback strategies?

## Metrics To Measure

For real-model benchmarks, the primary metrics should include:

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

Every benchmark report should identify its baseline clearly. The intended comparison set is:

1. no orchestration
2. static quantized setup
3. naive offload
4. controller-driven strategy

Not every experiment needs all four baselines immediately, but the validation path should converge toward that comparison set.

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

Only after the earlier phases are stable:

- expand to larger-model experiments
- compare multiple policy settings
- publish memory, traffic, throughput, and quality tradeoffs carefully

## Current State

The current repository supports Phase A best. Phase B and Phase C define the near-term validation path.
