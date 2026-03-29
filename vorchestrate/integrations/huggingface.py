"""Hugging Face integration for vOrchestrate."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MethodType
from typing import Any

from torch import nn

from ..core import (
    AccuracyGuardrail,
    PrefetchScheduler,
    ScoringEngine,
    WeightBlockRegistry,
    WeightStateMachine,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_HBM_BUDGET_GB = 4.0
DEFAULT_TICK_INTERVAL_LAYERS = 4
BYTES_PER_GB = 1024**3
DEFAULT_BASE_CRITICALITY = 0.6
DEFAULT_BASE_SENSITIVITY = 0.55


@dataclass(slots=True)
class HeuristicProfile:
    """Prototype heuristic profile for criticality and sensitivity estimates.

    The defaults are deliberately simple. They provide a useful integration scaffold,
    not universal model semantics.
    """

    criticality_keywords: dict[str, float] = field(
        default_factory=lambda: {
            "embed": 0.9,
            "lm_head": 0.9,
            "attn": 0.8,
            "attention": 0.8,
            "mlp": 0.7,
            "ffn": 0.7,
        }
    )
    sensitivity_keywords: dict[str, float] = field(
        default_factory=lambda: {
            "norm": 0.85,
            "embed": 0.8,
            "lm_head": 0.8,
        }
    )
    criticality_overrides: dict[str, float] = field(default_factory=dict)
    sensitivity_overrides: dict[str, float] = field(default_factory=dict)
    criticality_callback: Callable[[str, nn.Module], float] | None = None
    sensitivity_callback: Callable[[str, nn.Module], float] | None = None


class VOrchestrate(nn.Module):
    """Wrap a causal language model with predictive residency orchestration."""

    def __init__(
        self,
        model: nn.Module,
        hbm_budget_gb: float = DEFAULT_HBM_BUDGET_GB,
        tick_interval_layers: int = DEFAULT_TICK_INTERVAL_LAYERS,
        psi_threshold: float = 0.7,
        tick_every_n_layers: int | None = None,
        enable_prefetch: bool = True,
        heuristic_profile: HeuristicProfile | None = None,
        criticality_overrides: Mapping[str, float] | None = None,
        sensitivity_overrides: Mapping[str, float] | None = None,
    ) -> None:
        """Initialize the orchestration wrapper.

        Args:
            model: Wrapped Hugging Face style model.
            hbm_budget_gb: Effective HBM budget in gigabytes.
            tick_interval_layers: Number of layer invocations per control tick.
            psi_threshold: Sensitivity threshold for the accuracy guardrail.
            tick_every_n_layers: Backward-compatible alias for tick interval.
            enable_prefetch: Whether to create the async prefetch scheduler.
            heuristic_profile: Prototype heuristic profile for module-name-based scoring hints.
            criticality_overrides: Optional exact-match overrides by module name.
            sensitivity_overrides: Optional exact-match overrides by module name.
        """
        if tick_every_n_layers is not None:
            tick_interval_layers = tick_every_n_layers
        super().__init__()
        if tick_interval_layers <= 0:
            raise ValueError("tick_interval_layers must be positive")

        self.model = model
        self.hbm_budget_bytes = int(hbm_budget_gb * BYTES_PER_GB)
        self.tick_interval_layers = tick_interval_layers
        self.heuristic_profile = heuristic_profile or HeuristicProfile()
        if criticality_overrides:
            self.heuristic_profile.criticality_overrides.update(dict(criticality_overrides))
        if sensitivity_overrides:
            self.heuristic_profile.sensitivity_overrides.update(dict(sensitivity_overrides))
        self.registry = WeightBlockRegistry(hbm_capacity_bytes=self.hbm_budget_bytes)
        self.guardrail = AccuracyGuardrail(psi_threshold=psi_threshold)
        self.scorer = ScoringEngine(self.registry)
        self.scheduler = PrefetchScheduler(self.registry) if enable_prefetch else None
        self.state_machine = WeightStateMachine(
            self.registry,
            self.scorer,
            self.guardrail,
            scheduler=self.scheduler,
        )
        self._step = 0
        self._layer_calls = 0
        self._registered_layers: dict[str, str] = {}
        self._original_forwards: dict[str, Callable[..., Any]] = {}
        self._layer_modules: dict[str, nn.Module] = {}

        self._register_model_layers()
        self._patch_layer_forwards()
        self._patch_model_forward()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate execution to the wrapped model."""
        return self.model(*args, **kwargs)

    def shutdown(self) -> None:
        """Release scheduler resources if prefetching is enabled."""
        if self.scheduler is not None:
            self.scheduler.shutdown()

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def _register_model_layers(self) -> None:
        """Register leaf modules with trainable weights as weight blocks."""
        for module_name, module in self._iter_weighted_layers():
            size_bytes = 0
            for parameter in module.parameters(recurse=False):
                size_bytes += parameter.nelement() * parameter.element_size()
            if size_bytes == 0:
                continue
            block_id = self.registry.register_block(
                layer_name=module_name,
                size_bytes=size_bytes,
                criticality=self._estimate_criticality(module_name, module),
                sensitivity=self._estimate_sensitivity(module_name, module),
            )
            self._registered_layers[module_name] = block_id
            self._layer_modules[module_name] = module

    def _patch_layer_forwards(self) -> None:
        """Monkey-patch layer forwards to track accesses around execution."""
        for module_name, module in self._iter_weighted_layers():
            if module_name not in self._registered_layers:
                continue
            original_forward = module.forward
            self._original_forwards[module_name] = original_forward

            def wrapped_forward(inner_module: nn.Module, *args: Any, __name: str = module_name, **kwargs: Any) -> Any:
                block_id = self._registered_layers[__name]
                self.registry.update_access(block_id, self._step)
                result = self._original_forwards[__name](*args, **kwargs)
                self._layer_calls += 1
                if self._layer_calls % self.tick_interval_layers == 0:
                    self.state_machine.tick(self._step, self.hbm_budget_bytes)
                return result

            module.forward = MethodType(wrapped_forward, module)

    def _patch_model_forward(self) -> None:
        """Monkey-patch the model forward entry point to advance orchestration step."""
        original_forward = self.model.forward

        def wrapped_model_forward(inner_model: nn.Module, *args: Any, **kwargs: Any) -> Any:
            self._step += 1
            self._layer_calls = 0
            return original_forward(*args, **kwargs)

        self.model.forward = MethodType(wrapped_model_forward, self.model)

    def _iter_weighted_layers(self) -> Iterable[tuple[str, nn.Module]]:
        """Yield leaf modules that own direct weight tensors."""
        for module_name, module in self.model.named_modules():
            if not module_name:
                continue
            children = list(module.children())
            if children:
                continue
            if any(parameter.requires_grad or parameter.numel() > 0 for parameter in module.parameters(recurse=False)):
                yield module_name, module

    def _estimate_criticality(self, module_name: str, module: nn.Module) -> float:
        """Estimate static layer criticality using simple prototype heuristics."""
        if module_name in self.heuristic_profile.criticality_overrides:
            return self.heuristic_profile.criticality_overrides[module_name]
        if self.heuristic_profile.criticality_callback is not None:
            return self.heuristic_profile.criticality_callback(module_name, module)
        name = module_name.lower()
        for keyword, value in self.heuristic_profile.criticality_keywords.items():
            if keyword in name:
                return value
        return DEFAULT_BASE_CRITICALITY

    def _estimate_sensitivity(self, module_name: str, module: nn.Module) -> float:
        """Estimate quality sensitivity using simple prototype heuristics."""
        if module_name in self.heuristic_profile.sensitivity_overrides:
            return self.heuristic_profile.sensitivity_overrides[module_name]
        if self.heuristic_profile.sensitivity_callback is not None:
            return self.heuristic_profile.sensitivity_callback(module_name, module)
        name = module_name.lower()
        for keyword, value in self.heuristic_profile.sensitivity_keywords.items():
            if keyword in name:
                return value
        return DEFAULT_BASE_SENSITIVITY
