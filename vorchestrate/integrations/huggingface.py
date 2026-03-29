"""Hugging Face integration for vOrchestrate."""

from __future__ import annotations

import logging
from types import MethodType
from typing import Any, Callable, Dict, Iterable, List, Tuple

import torch
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
    ) -> None:
        """Initialize the orchestration wrapper.

        Args:
            model: Wrapped Hugging Face style model.
            hbm_budget_gb: Effective HBM budget in gigabytes.
            tick_interval_layers: Number of layer invocations per control tick.
            psi_threshold: Sensitivity threshold for the accuracy guardrail.
            tick_every_n_layers: Backward-compatible alias for tick interval.
            enable_prefetch: Whether to create the async prefetch scheduler.
        """
        if tick_every_n_layers is not None:
            tick_interval_layers = tick_every_n_layers
        super().__init__()
        if tick_interval_layers <= 0:
            raise ValueError("tick_interval_layers must be positive")

        self.model = model
        self.hbm_budget_bytes = int(hbm_budget_gb * BYTES_PER_GB)
        self.tick_interval_layers = tick_interval_layers
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
        self._registered_layers: Dict[str, str] = {}
        self._original_forwards: Dict[str, Callable[..., Any]] = {}

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
                criticality=self._estimate_criticality(module_name),
                sensitivity=self._estimate_sensitivity(module_name),
            )
            self._registered_layers[module_name] = block_id

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

    def _iter_weighted_layers(self) -> Iterable[Tuple[str, nn.Module]]:
        """Yield leaf modules that own direct weight tensors."""
        for module_name, module in self.model.named_modules():
            if not module_name:
                continue
            children = list(module.children())
            if children:
                continue
            if any(parameter.requires_grad or parameter.numel() > 0 for parameter in module.parameters(recurse=False)):
                yield module_name, module

    def _estimate_criticality(self, module_name: str) -> float:
        """Heuristically estimate static layer criticality."""
        name = module_name.lower()
        if "embed" in name or "lm_head" in name:
            return 0.9
        if "attn" in name or "attention" in name:
            return 0.8
        if "mlp" in name or "ffn" in name:
            return 0.7
        return 0.6

    def _estimate_sensitivity(self, module_name: str) -> float:
        """Heuristically estimate quality sensitivity."""
        name = module_name.lower()
        if "norm" in name:
            return 0.85
        if "embed" in name or "lm_head" in name:
            return 0.8
        return 0.55
