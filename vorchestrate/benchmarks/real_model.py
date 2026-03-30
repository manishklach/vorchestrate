"""Narrow real-model benchmark support for small decoder-only models."""

from __future__ import annotations

import csv
import json
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..core import (
    AccuracyGuardrail,
    ControllerMetrics,
    ScoringEngine,
    WeightBlockRegistry,
    WeightStateMachine,
)
from ..core.constants import HBM_RESIDENT_STATES, STATE_HOST_DRAM, STATE_NVME
from ..integrations import DecoderOnlyTransformerAdapter, HeuristicProfile
from ..utils.trace import TraceEvent, write_trace_csv, write_trace_json
from ..utils.visualization import render_trace_report

DEFAULT_REAL_MODEL_NAME = "distilgpt2"
DEFAULT_PROMPT = "The controller observes memory pressure and adjusts residency."
DEFAULT_BATCH_SIZE = 1
DEFAULT_WARMUP_RUNS = 1
DEFAULT_MEASURED_RUNS = 2
DEFAULT_TICK_EVERY_N_LAYERS = 4
DEFAULT_MAX_INPUT_LENGTH = 32
DEFAULT_HBM_BUDGET_BYTES = 64 * 1024 * 1024
BYTES_PER_GIB = 1024**3
MILLISECONDS_PER_SECOND = 1000.0


@dataclass(slots=True)
class RealModelBenchmarkConfig:
    """Configuration for the narrow real-model validation path."""

    model_name: str = DEFAULT_REAL_MODEL_NAME
    prompt: str = DEFAULT_PROMPT
    batch_size: int = DEFAULT_BATCH_SIZE
    warmup_runs: int = DEFAULT_WARMUP_RUNS
    measured_runs: int = DEFAULT_MEASURED_RUNS
    tick_every_n_layers: int = DEFAULT_TICK_EVERY_N_LAYERS
    max_input_length: int = DEFAULT_MAX_INPUT_LENGTH
    hbm_budget_bytes: int = DEFAULT_HBM_BUDGET_BYTES
    device: str = "auto"
    output_dir: str | None = None
    generate_plots: bool = True


@dataclass(slots=True)
class RealModelBenchmarkResult:
    """Structured result emitted by the narrow real-model benchmark."""

    summary: dict[str, Any]
    events: list[TraceEvent]
    metrics: ControllerMetrics
    output_dir: Path


def load_model_and_tokenizer(model_name: str) -> tuple[Any, Any]:
    """Load a narrow Hugging Face validation model and tokenizer."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - runtime dependency path
        raise RuntimeError(
            "transformers is required for the real-model benchmark. "
            "Install with `pip install -e .[real-bench]`."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


def choose_device(requested_device: str) -> str:
    """Resolve the device string for a benchmark run."""
    if requested_device != "auto":
        return requested_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def prepare_inputs(
    tokenizer: Any,
    prompt: str,
    batch_size: int,
    max_input_length: int,
    device: str,
) -> dict[str, torch.Tensor]:
    """Tokenize and move benchmark inputs to the chosen device."""
    prompts = [prompt] * batch_size
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    )
    return {name: tensor.to(device) for name, tensor in encoded.items()}


def write_benchmark_summary(path: str | Path, summary: dict[str, Any]) -> Path:
    """Write a benchmark summary JSON file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_path


def write_trace_rows_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Write a simple CSV view of real-model benchmark runs."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return output_path
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def run_real_model_benchmark(config: RealModelBenchmarkConfig) -> RealModelBenchmarkResult:
    """Run a narrow real-model validation benchmark for a small decoder-only model."""
    model, tokenizer = load_model_and_tokenizer(config.model_name)
    device = choose_device(config.device)
    model.to(device)
    inputs = prepare_inputs(tokenizer, config.prompt, config.batch_size, config.max_input_length, device)

    adapter = DecoderOnlyTransformerAdapter(model, heuristic_profile=HeuristicProfile())
    registry = WeightBlockRegistry(hbm_capacity_bytes=config.hbm_budget_bytes)
    registry_ids = adapter.register_with_registry(registry)
    scorer = ScoringEngine(registry)
    guardrail = AccuracyGuardrail()
    state_machine = WeightStateMachine(registry, scorer, guardrail, scheduler=None)
    metrics = ControllerMetrics()
    events: list[TraceEvent] = []

    if config.output_dir is None:
        output_dir = Path("benchmarks") / "results" / "real_model"
    else:
        output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_hooks = _install_controller_hooks(
        adapter=adapter,
        registry=registry,
        registry_ids=registry_ids,
        scorer=scorer,
        guardrail=guardrail,
        state_machine=state_machine,
        metrics=metrics,
        events=events,
    )
    run_rows: list[dict[str, Any]] = []
    model_parameter_count = adapter.parameter_count()
    sequence_length = int(inputs["input_ids"].shape[-1])
    total_tokens = config.batch_size * sequence_length

    try:
        for warmup_step in range(config.warmup_runs):
            _run_single_forward(model, inputs, benchmark_hooks, warmup_step, measured=False)

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for measured_step in range(config.measured_runs):
            run_row = _run_single_forward(
                model,
                inputs,
                benchmark_hooks,
                measured_step,
                measured=True,
            )
            run_row["tokens_per_second"] = total_tokens / max(run_row["latency_seconds"], 1e-9)
            run_rows.append(run_row)
    finally:
        for handle in benchmark_hooks.handles:
            handle.remove()

    peak_gpu_memory_bytes = (
        int(torch.cuda.max_memory_allocated(device=device)) if device == "cuda" else None
    )
    host_memory_bytes = _get_host_memory_bytes()
    trace_json_path = write_trace_json(output_dir / "benchmark_trace.json", events)
    trace_csv_path = write_trace_csv(output_dir / "benchmark_trace.csv", events)
    run_rows_path = write_trace_rows_csv(output_dir / "run_rows.csv", run_rows)
    plot_paths = (
        [str(path) for path in render_trace_report(trace_json_path, output_dir / "plots")]
        if config.generate_plots and events
        else []
    )
    benchmark_config = asdict(config)
    benchmark_config["output_dir"] = _portable_path(output_dir)
    summary = {
        "benchmark": benchmark_config,
        "model": {
            "name": config.model_name,
            "parameter_count": model_parameter_count,
            "supported_scope": "narrow decoder-only GPT-2 style validation path",
        },
        "runtime": {
            "device": device,
            "batch_size": config.batch_size,
            "sequence_length": sequence_length,
            "prompt_length_chars": len(config.prompt),
            "measured_runs": config.measured_runs,
            "warmup_runs": config.warmup_runs,
            "latency_seconds_mean": _mean(run_row["latency_seconds"] for run_row in run_rows),
            "latency_seconds_per_run": [run_row["latency_seconds"] for run_row in run_rows],
            "tokens_per_second_mean": _mean(run_row["tokens_per_second"] for run_row in run_rows),
            "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
            "host_memory_bytes": host_memory_bytes,
            "cuda_available": torch.cuda.is_available(),
            "mode": "observed runtime metrics",
        },
        "controller": {
            "metrics": metrics.to_dict(),
            "event_count": len(events),
            "trace_origin": "real_model_benchmark",
            "actions_are_intents": True,
            "note": (
                "Controller actions in this benchmark reflect policy decisions recorded "
                "through the prototype registry/state-machine path. They do not imply "
                "that real HBM/DRAM/NVMe movement backends are active."
            ),
        },
        "artifacts": {
            "trace_json": _portable_path(trace_json_path),
            "trace_csv": _portable_path(trace_csv_path),
            "run_rows_csv": _portable_path(run_rows_path),
            "plots": [_portable_path(Path(path)) for path in plot_paths],
        },
    }
    write_benchmark_summary(output_dir / "benchmark_summary.json", summary)
    return RealModelBenchmarkResult(summary=summary, events=events, metrics=metrics, output_dir=output_dir)


@dataclass(slots=True)
class _BenchmarkHookState:
    """Mutable benchmark hook state."""

    registry: WeightBlockRegistry
    registry_ids: dict[str, str]
    scorer: ScoringEngine
    guardrail: AccuracyGuardrail
    state_machine: WeightStateMachine
    metrics: ControllerMetrics
    events: list[TraceEvent]
    current_step: int = 0
    measured: bool = False
    handles: list[Any] = field(default_factory=list)


def _install_controller_hooks(
    adapter: DecoderOnlyTransformerAdapter,
    registry: WeightBlockRegistry,
    registry_ids: dict[str, str],
    scorer: ScoringEngine,
    guardrail: AccuracyGuardrail,
    state_machine: WeightStateMachine,
    metrics: ControllerMetrics,
    events: list[TraceEvent],
) -> _BenchmarkHookState:
    """Attach narrow benchmark hooks to adapter modules."""
    state = _BenchmarkHookState(
        registry=registry,
        registry_ids=registry_ids,
        scorer=scorer,
        guardrail=guardrail,
        state_machine=state_machine,
        metrics=metrics,
        events=events,
    )

    for block_id, module in adapter.iter_block_modules():
        registry_id = registry_ids[block_id]
        pre_hook = module.register_forward_pre_hook(
            _make_pre_hook(state.registry, registry_id, lambda: state.current_step)
        )
        post_hook = module.register_forward_hook(
            _make_post_hook(
                state.registry,
                state.scorer,
                state.guardrail,
                state.state_machine,
                state.metrics,
                state.events,
                registry_id,
                block_id,
                lambda: state.current_step,
                lambda: state.measured,
            )
        )
        state.handles.extend([pre_hook, post_hook])
    return state


def _make_pre_hook(
    registry: WeightBlockRegistry,
    registry_id: str,
    current_step: Callable[[], int],
) -> Callable[[torch.nn.Module, tuple[Any, ...]], None]:
    """Create a pre-forward hook that updates access metadata."""

    def _pre_hook(module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
        del module, inputs
        registry.update_access(registry_id, current_step())

    return _pre_hook


def _make_post_hook(
    registry: WeightBlockRegistry,
    scorer: ScoringEngine,
    guardrail: AccuracyGuardrail,
    state_machine: WeightStateMachine,
    metrics: ControllerMetrics,
    events: list[TraceEvent],
    registry_id: str,
    block_id: str,
    current_step: Callable[[], int],
    measured: Callable[[], bool],
) -> Callable[[torch.nn.Module, tuple[Any, ...], Any], Any]:
    """Create a post-forward hook that records controller decisions."""

    def _post_hook(module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
        del module, inputs
        block = registry.get_block(registry_id)
        score = scorer.compute_score(block)
        old_state = block.current_state
        target_state = scorer.get_target_state(registry_id, registry.get_hbm_pressure())
        guardrail_veto = target_state > old_state and not guardrail.check_demotion(block, target_state)
        action = _classify_action(old_state, target_state)
        bytes_moved = 0

        if target_state != old_state:
            transitioned = state_machine.transition(registry_id, target_state)
            if transitioned:
                bytes_moved = block.size_bytes
                metrics.record_transition(old_state, target_state, block.size_bytes)
                if target_state in {STATE_HOST_DRAM, STATE_NVME}:
                    metrics.record_stage()
            else:
                metrics.record_guardrail_veto()
                guardrail_veto = True
                action = "guardrail_veto"

        if measured():
            new_state = registry.get_block(registry_id).current_state
            events.append(
                TraceEvent(
                    step=current_step(),
                    block_id=block_id,
                    score=score,
                    psi=block.quality_sensitivity,
                    old_state=old_state,
                    new_state=new_state,
                    old_tier=_state_to_tier(old_state),
                    new_tier=_state_to_tier(new_state),
                    action=action,
                    guardrail_veto=guardrail_veto,
                    bytes_moved=bytes_moved,
                    hbm_pressure=registry.get_hbm_pressure(),
                    trace_origin="real_model_benchmark",
                    action_is_intent=True,
                )
            )
        return output

    return _post_hook


def _run_single_forward(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    benchmark_hooks: _BenchmarkHookState,
    step: int,
    measured: bool,
) -> dict[str, Any]:
    """Run one forward pass and return timing for that run."""
    benchmark_hooks.current_step = step
    benchmark_hooks.measured = measured
    with torch.inference_mode():
        if next(model.parameters()).device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model(**inputs)
        if next(model.parameters()).device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return {"step": step, "latency_seconds": elapsed, "measured": measured}


def _mean(values: Any) -> float | None:
    """Return the mean of an iterable or None when empty."""
    values_list = list(values)
    if not values_list:
        return None
    return float(sum(values_list) / len(values_list))


def _classify_action(old_state: int, target_state: int) -> str:
    """Map a target-state decision to a coarse action label."""
    if target_state == old_state:
        return "keep"
    if target_state < old_state:
        return "promote"
    if target_state in {STATE_HOST_DRAM, STATE_NVME}:
        return "stage"
    return "demote"


def _state_to_tier(state: int) -> str:
    """Map a residency state to a tier label."""
    if state in HBM_RESIDENT_STATES:
        return "HBM"
    if state == STATE_HOST_DRAM:
        return "DRAM"
    if state == STATE_NVME:
        return "NVMe"
    return f"S{state}"


def _get_host_memory_bytes() -> int | None:
    """Return process resident memory when it can be measured portably enough."""
    if sys.platform == "win32":
        return _get_windows_process_memory_bytes()
    try:
        import resource
    except ImportError:
        return None

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(usage)
    return int(usage * 1024)


def _get_windows_process_memory_bytes() -> int | None:
    """Return working-set bytes on Windows without extra dependencies."""
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(ProcessMemoryCounters)
    windll: Any = getattr(ctypes, "windll", None)
    if windll is None:
        return None
    process = windll.kernel32.GetCurrentProcess()
    success = windll.psapi.GetProcessMemoryInfo(
        process,
        ctypes.byref(counters),
        counters.cb,
    )
    if not success:
        return None
    return int(counters.WorkingSetSize)


def _portable_path(path: Path) -> str:
    """Return a repo-relative path when possible for report readability."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)
