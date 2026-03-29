# Roadmap

The roadmap is organized around progressive validation rather than isolated feature drops.

## Phase 0: Controller Skeleton

Current repository baseline:

- registry
- scoring engine
- guardrail
- state machine
- scheduler scaffold

## Phase 1: Simulation And Trace Tooling

Current near-term work:

- synthetic controller traces
- JSON and CSV trace output
- metrics accumulation
- deterministic simulation scenarios

## Phase 2: Benchmark Instrumentation

Next step:

- benchmark harness
- better accounting for memory pressure and residency changes
- benchmark result serialization
- clearer comparison to baseline strategies

## Phase 3: Small-Model Validation

Next validation step:

- exercise one or more small real transformer models
- measure latency and memory behavior
- inspect whether guardrail-aware policies behave as intended

## Phase 4: Larger-Model Experiments

Longer-term direction:

- larger-model experiments
- richer tier backends
- policy comparison studies
- reproducible benchmark reports

Each phase is intended to leave behind artifacts that the next phase can build on: traces, counters, benchmark harnesses, validation notes, and eventually real-model studies.
