# API Overview

This document summarizes the main public surfaces exposed by the current prototype scaffold.

## Core Controller Modules

### `WeightBlockRegistry`

Tracks residency-managed blocks, access history, sensitivity, transfer-cost metadata, and HBM pressure.

### `ScoringEngine`

Computes the composite controller score and exposes ranking or target-state helpers.

### `AccuracyGuardrail`

Applies the sensitivity threshold used to veto aggressive demotion.

### `WeightStateMachine`

Executes transitions, records transition history, and coordinates with the scheduler scaffold.

### `PrefetchScheduler`

Maintains a background queue for promotion and demotion commands and exposes simple utilization estimates.

### `ControllerMetrics`

Collects coarse counters for promotions, demotions, prefetches, stages, and guardrail vetoes.

## Trace And Simulation Utilities

### `TraceEvent`

Structured trace record used by the controller simulation.

### `write_trace_json()` / `write_trace_csv()`

Persist synthetic trace output for inspection and visualization.

### `SyntheticBlockDescriptor`

Metadata record used to simulate a residency-managed unit without implying real-model compatibility.

### `run_controller_simulation()`

Runs the synthetic controller simulation and returns events, metrics, and score snapshots.

## Integration Surface

### `ResidencyAdapter`

Minimal abstract interface for future model integrations.

### `VOrchestrate`

Prototype Hugging Face-oriented wrapper path. It currently exposes a useful integration shape, but it is not broadly validated across arbitrary model families.

### `HeuristicProfile`

Prototype heuristic configuration object used by the Hugging Face integration surface. It allows exact-match overrides, keyword-based defaults, and optional callbacks for criticality and sensitivity estimation.
