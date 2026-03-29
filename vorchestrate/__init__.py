"""Public package interface for vOrchestrate."""

import atexit

from .core import (
    AccuracyGuardrail,
    ControllerMetrics,
    PrefetchScheduler,
    ScoringEngine,
    WeightBlockMeta,
    WeightBlockRegistry,
    WeightStateMachine,
)
from .integrations.huggingface import VOrchestrate

__all__ = [
    "AccuracyGuardrail",
    "ControllerMetrics",
    "PrefetchScheduler",
    "ScoringEngine",
    "VOrchestrate",
    "WeightBlockMeta",
    "WeightBlockRegistry",
    "WeightStateMachine",
]

__version__ = "0.1.0"


def _register_scheduler_shutdown() -> None:
    """Install a best-effort interpreter shutdown hook."""

    def _cleanup() -> None:
        return None

    atexit.register(_cleanup)


_register_scheduler_shutdown()
