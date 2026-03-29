# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-03-29

### Added

- Embedded synthetic controller visualization plots in README for immediate visibility
- `CHANGELOG.md` following Keep a Changelog format
- `docs/images/` directory with generated sample plots committed to the repository

### Changed

- README visualization section now shows inline plots (state timeline, score evolution, action distribution, HBM pressure) so visitors can see controller behavior without cloning

## [0.1.0] - 2026-03-29

### Added

- Core controller modules: `WeightBlockRegistry`, `ScoringEngine`, `AccuracyGuardrail`, `WeightStateMachine`, `PrefetchScheduler`, `ControllerMetrics`
- `WeightState` IntEnum for type-safe residency states (`S0`–`S6`)
- Composite scoring formula `R(b)` with configurable weights and hysteresis
- Thread-safe block registry with sliding-window access tracking
- Synthetic controller simulation (`examples/simulated_trace.py`)
- Trace visualization pipeline (`examples/render_trace_report.py`)
- HuggingFace integration wrapper (`VOrchestrate`) with configurable `HeuristicProfile`
- CI/CD pipeline via GitHub Actions (pytest + ruff + mypy across Python 3.10–3.12)
- Documentation: `architecture.md`, `api.md`, `benchmark_plan.md`, `roadmap.md`, `visualization.md`, `limitations.md`
- Community files: `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue/PR templates
- `Makefile` with common developer commands
- `pyproject.toml` with ruff and mypy configuration
