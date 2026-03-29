"""Utilities for traces and synthetic controller exercises."""

from .simulation import (
    SimulationConfig,
    SimulationResult,
    SyntheticBlockDescriptor,
    run_controller_simulation,
)
from .trace import TraceEvent, write_trace_csv, write_trace_json
from .visualization import load_trace_events, render_trace_report, summarize_trace

__all__ = [
    "SimulationConfig",
    "SimulationResult",
    "SyntheticBlockDescriptor",
    "TraceEvent",
    "load_trace_events",
    "render_trace_report",
    "run_controller_simulation",
    "summarize_trace",
    "write_trace_csv",
    "write_trace_json",
]
