PYTHON ?= python

.PHONY: install test lint typecheck simulate benchmark-sim benchmark-real benchmark-stub render-trace render-trace-report

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

benchmark-sim:
	$(PYTHON) benchmarks/benchmark_stub.py

benchmark-real:
	$(PYTHON) benchmarks/real_model_benchmark.py

benchmark-stub:
	$(PYTHON) benchmarks/benchmark_stub.py

render-trace:
	$(PYTHON) examples/render_trace_report.py

render-trace-report:
	$(PYTHON) examples/render_trace_report.py
