"""vLLM integration stubs for future expansion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class VLLMIntegrationConfig:
    """Configuration placeholder for vLLM integration."""

    enabled: bool = False
    notes: str = "vLLM integration is planned but not yet implemented."


class VLLMIntegration:
    """Stub integration surface for vLLM environments."""

    def __init__(self, config: VLLMIntegrationConfig | None = None) -> None:
        """Initialize the stub integration."""
        self.config = config or VLLMIntegrationConfig()

    def attach(self, engine: Any) -> None:
        """Attach to a vLLM engine.

        Raises:
            NotImplementedError: Always, until integration lands.
        """
        raise NotImplementedError("vLLM integration is not implemented in v0.1.0")
