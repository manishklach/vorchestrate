"""Tests for configurable integration heuristics."""

from __future__ import annotations

from torch import nn

from vorchestrate import HeuristicProfile, VOrchestrate


class TinyModel(nn.Module):
    """Small model used to exercise heuristic overrides."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.proj = nn.Linear(8, 8)

    def forward(self, x):  # type: ignore[override]
        return self.proj(self.embed(x))


def test_exact_match_heuristic_overrides_apply() -> None:
    model = VOrchestrate(
        TinyModel(),
        enable_prefetch=False,
        criticality_overrides={"proj": 0.99},
        sensitivity_overrides={"proj": 0.11},
    )
    proj_block = model.registry.get_block("proj:0")
    assert proj_block.layer_criticality == 0.99
    assert proj_block.quality_sensitivity == 0.11


def test_callback_heuristics_apply() -> None:
    profile = HeuristicProfile(
        criticality_callback=lambda name, module: 0.42,
        sensitivity_callback=lambda name, module: 0.24,
    )
    model = VOrchestrate(TinyModel(), enable_prefetch=False, heuristic_profile=profile)
    embed_block = model.registry.get_block("embed:0")
    assert embed_block.layer_criticality == 0.42
    assert embed_block.quality_sensitivity == 0.24
