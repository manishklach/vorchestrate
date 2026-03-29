PYTHON ?= python

.PHONY: install test lint typecheck simulate benchmark-stub render-trace-report

install:
	$(PYTHON) -m pip install -e .[dev]

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy vorchestrate

simulate:
	$(PYTHON) examples/simulated_trace.py

benchmark-stub:
	$(PYTHON) benchmarks/benchmark_stub.py

render-trace-report:
	$(PYTHON) examples/render_trace_report.py
