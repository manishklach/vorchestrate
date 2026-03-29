"""Tests for the lightweight Hugging Face-style integration scaffold."""

from __future__ import annotations

import torch
from torch import nn

from vorchestrate import VOrchestrate


class TinyModel(nn.Module):
    """Small model used to exercise the wrapper path."""

    def __init__(self) -> None:
        """Initialize the toy network."""
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.proj = nn.Linear(8, 8)
        self.head = nn.Linear(8, 16)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run a simple forward pass."""
        hidden = self.embed(input_ids)
        hidden = self.proj(hidden).relu()
        return self.head(hidden)


def test_wrapper_registers_blocks_for_leaf_modules() -> None:
    model = VOrchestrate(TinyModel(), hbm_budget_gb=0.1, enable_prefetch=False)
    block_ids = {block.block_id for block in model.registry.get_all_blocks()}
    assert "embed:0" in block_ids
    assert "proj:0" in block_ids
    assert "head:0" in block_ids


def test_wrapper_forward_runs_and_records_accesses() -> None:
    model = VOrchestrate(TinyModel(), hbm_budget_gb=0.1, enable_prefetch=False)
    sample = torch.randint(0, 16, (1, 4))
    output = model(sample)
    assert output.shape == (1, 4, 16)
    assert any(block.last_access_step >= 1 for block in model.registry.get_all_blocks())


def test_wrapper_can_disable_prefetch_scheduler() -> None:
    model = VOrchestrate(TinyModel(), hbm_budget_gb=0.1, enable_prefetch=False)
    assert model.scheduler is None
