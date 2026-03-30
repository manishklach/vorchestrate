"""Narrow decoder-only adapter for small Hugging Face-style validation runs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from torch import nn

from ..core import WeightBlockRegistry
from .base import ResidencyAdapter
from .huggingface import HeuristicProfile


@dataclass(slots=True)
class AdapterBlockDescriptor:
    """Metadata for one residency-managed unit discovered from a real model."""

    block_id: str
    module_name: str
    size_bytes: int
    criticality: float
    sensitivity: float


class DecoderOnlyTransformerAdapter(ResidencyAdapter):
    """Adapter for a narrow decoder-only transformer validation path.

    This adapter is intentionally scoped to small GPT-2 style causal language models.
    It discovers leaf modules with direct parameters and maps them into the registry and
    controller abstractions. It does not imply broad Hugging Face model support.
    """

    def __init__(
        self,
        model: nn.Module,
        heuristic_profile: HeuristicProfile | None = None,
    ) -> None:
        """Initialize the adapter and discover supported blocks."""
        self.model = model
        self.heuristic_profile = heuristic_profile or HeuristicProfile()
        self._supported_model = hasattr(model, "transformer") and hasattr(model.transformer, "h")
        if not self._supported_model:
            raise ValueError(
                "DecoderOnlyTransformerAdapter currently supports narrow GPT-2 style "
                "decoder-only models with `transformer.h` blocks."
            )
        self._descriptors: dict[str, AdapterBlockDescriptor] = {}
        self._modules: dict[str, nn.Module] = {}
        self._registry_ids: dict[str, str] = {}
        self._discover_blocks()

    def enumerate_blocks(self) -> Iterable[str]:
        """Return discovered adapter block identifiers."""
        return list(self._descriptors.keys())

    def get_block_metadata(self, block_id: str) -> dict[str, Any]:
        """Return metadata for one residency-managed block."""
        descriptor = self._descriptors[block_id]
        return {
            "block_id": descriptor.block_id,
            "module_name": descriptor.module_name,
            "size_bytes": descriptor.size_bytes,
            "criticality": descriptor.criticality,
            "sensitivity": descriptor.sensitivity,
        }

    def request_block(self, block_id: str) -> nn.Module:
        """Return the underlying module for a given block identifier."""
        return self._modules[block_id]

    def on_inference_step(self, step_ctx: dict[str, Any]) -> None:
        """Accept inference-step context for future adapter extensions."""
        self._last_step_ctx = dict(step_ctx)

    def register_with_registry(self, registry: WeightBlockRegistry) -> dict[str, str]:
        """Register adapter blocks with a controller registry.

        Returns:
            Mapping from adapter block identifier to registry block identifier.
        """
        if self._registry_ids:
            return dict(self._registry_ids)

        for block_id, descriptor in self._descriptors.items():
            registry_id = registry.register_block(
                layer_name=descriptor.module_name,
                size_bytes=descriptor.size_bytes,
                criticality=descriptor.criticality,
                sensitivity=descriptor.sensitivity,
            )
            self._registry_ids[block_id] = registry_id
        return dict(self._registry_ids)

    def iter_block_modules(self) -> Iterable[tuple[str, nn.Module]]:
        """Yield block identifiers paired with the corresponding module."""
        return list(self._modules.items())

    def parameter_count(self) -> int:
        """Return the total parameter count for the wrapped model."""
        return sum(parameter.numel() for parameter in self.model.parameters())

    def _discover_blocks(self) -> None:
        """Discover leaf modules that own direct parameters."""
        for module_name, module in self.model.named_modules():
            if not module_name or any(module.children()):
                continue
            size_bytes = sum(
                parameter.nelement() * parameter.element_size()
                for parameter in module.parameters(recurse=False)
            )
            if size_bytes <= 0:
                continue
            block_id = module_name
            self._modules[block_id] = module
            self._descriptors[block_id] = AdapterBlockDescriptor(
                block_id=block_id,
                module_name=module_name,
                size_bytes=size_bytes,
                criticality=self._estimate_criticality(module_name, module),
                sensitivity=self._estimate_sensitivity(module_name, module),
            )

    def _estimate_criticality(self, module_name: str, module: nn.Module) -> float:
        """Estimate block criticality using configurable prototype heuristics."""
        if module_name in self.heuristic_profile.criticality_overrides:
            return self.heuristic_profile.criticality_overrides[module_name]
        if self.heuristic_profile.criticality_callback is not None:
            return self.heuristic_profile.criticality_callback(module_name, module)
        lowered_name = module_name.lower()
        for keyword, value in self.heuristic_profile.criticality_keywords.items():
            if keyword in lowered_name:
                return value
        return 0.6

    def _estimate_sensitivity(self, module_name: str, module: nn.Module) -> float:
        """Estimate sensitivity using configurable prototype heuristics."""
        if module_name in self.heuristic_profile.sensitivity_overrides:
            return self.heuristic_profile.sensitivity_overrides[module_name]
        if self.heuristic_profile.sensitivity_callback is not None:
            return self.heuristic_profile.sensitivity_callback(module_name, module)
        lowered_name = module_name.lower()
        for keyword, value in self.heuristic_profile.sensitivity_keywords.items():
            if keyword in lowered_name:
                return value
        return 0.55
