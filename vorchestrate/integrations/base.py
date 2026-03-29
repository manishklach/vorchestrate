"""Abstract integration surface for future model adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any


class ResidencyAdapter(ABC):
    """Minimal adapter interface for future real-model integrations."""

    @abstractmethod
    def enumerate_blocks(self) -> Iterable[str]:
        """Return the residency-managed block identifiers."""

    @abstractmethod
    def get_block_metadata(self, block_id: str) -> dict[str, Any]:
        """Return metadata for a residency-managed unit."""

    @abstractmethod
    def request_block(self, block_id: str) -> Any:
        """Request access to a residency-managed block."""

    @abstractmethod
    def on_inference_step(self, step_ctx: dict[str, Any]) -> None:
        """Notify the adapter of an inference-step context."""
