"""Basic vOrchestrate usage example."""

from __future__ import annotations

import pathlib
import sys

import torch
from torch import nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from vorchestrate import VOrchestrate


class TinyLM(nn.Module):
    """A very small causal-LM-like module for local experimentation."""

    def __init__(self) -> None:
        """Initialize the toy network."""
        super().__init__()
        self.embed = nn.Embedding(32, 16)
        self.proj = nn.Linear(16, 16)
        self.head = nn.Linear(16, 32)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run a simple forward pass."""
        hidden = self.embed(input_ids)
        hidden = self.proj(hidden).relu()
        return self.head(hidden)


if __name__ == "__main__":
    base_model = TinyLM()
    orchestrated_model = VOrchestrate(base_model, hbm_budget_gb=0.25)
    sample = torch.randint(0, 32, (1, 8))
    logits = orchestrated_model(sample)
    print("logits shape:", tuple(logits.shape))
    print("tracked blocks:", len(orchestrated_model.registry.get_all_blocks()))
