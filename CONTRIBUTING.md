# Contributing to vOrchestrate

Thanks for helping build vOrchestrate.

## Development Setup

```bash
git clone https://github.com/manishklach/vorchestrate.git
cd vorchestrate
pip install -e .[dev]
python -m pytest -q
```

For the narrow real-model benchmark path:

```bash
pip install -e .[dev,real-bench]
```

Common local commands:

```bash
make test
make lint
make typecheck
make simulate
make benchmark-real
make benchmark-stub
make render-trace
```

## Guidelines

- Keep the library CPU-testable by default.
- Add type hints and Google-style docstrings for public APIs.
- Add or update tests for behavior changes.
- Prefer small, focused pull requests.
- Document any new residency state logic, heuristic behavior, or scheduler behavior in `README.md` or `docs/architecture.md`.

## Reporting Results

Benchmark reports, integration notes, and reproducible traces are especially useful. If you share throughput or memory numbers, include:

- model name and size
- GPU, CPU, and storage hardware
- PyTorch version
- batch size and sequence length
- baseline versus vOrchestrate settings

## Pull Requests

1. Fork the repository.
2. Create a feature branch.
3. Run `python -m pytest -q`.
4. Update docs if APIs or behavior changed.
5. Open a pull request with motivation, approach, and validation notes.
