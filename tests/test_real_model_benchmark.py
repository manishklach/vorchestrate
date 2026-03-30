"""Tests for the narrow real-model benchmark helpers."""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from vorchestrate.benchmarks.real_model import (
    RealModelBenchmarkConfig,
    choose_device,
    prepare_inputs,
    write_benchmark_summary,
    write_trace_rows_csv,
)
from vorchestrate.integrations import DecoderOnlyTransformerAdapter


class TinyDecoderBlock(nn.Module):
    """Small GPT-2 style block for adapter tests."""

    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Linear(8, 8)
        self.mlp = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.attn(x))


class TinyGPT2StyleModel(nn.Module):
    """Tiny model exposing a GPT-2 style module layout."""

    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.wte = nn.Embedding(16, 8)
        self.transformer.h = nn.ModuleList([TinyDecoderBlock(), TinyDecoderBlock()])
        self.lm_head = nn.Linear(8, 16)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.transformer.wte(input_ids)
        for block in self.transformer.h:
            hidden = block(hidden)
        return self.lm_head(hidden)


class UnsupportedModel(nn.Module):
    """Model missing GPT-2 style decoder blocks."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class TinyTokenizer:
    """Minimal tokenizer stub for benchmark input preparation tests."""

    def __call__(
        self,
        prompts: list[str],
        return_tensors: str,
        padding: bool,
        truncation: bool,
        max_length: int,
    ) -> dict[str, torch.Tensor]:
        del return_tensors, padding, truncation
        token_count = min(max_length, max(len(prompt.split()) for prompt in prompts))
        return {"input_ids": torch.ones(len(prompts), token_count, dtype=torch.long)}


def test_decoder_only_adapter_extracts_blocks() -> None:
    adapter = DecoderOnlyTransformerAdapter(TinyGPT2StyleModel())
    block_ids = set(adapter.enumerate_blocks())
    assert "transformer.wte" in block_ids
    assert "transformer.h.0.attn" in block_ids
    metadata = adapter.get_block_metadata("transformer.h.0.attn")
    assert metadata["criticality"] > 0.0
    assert metadata["size_bytes"] > 0


def test_decoder_only_adapter_rejects_unsupported_model() -> None:
    with pytest.raises(ValueError):
        DecoderOnlyTransformerAdapter(UnsupportedModel())


def test_prepare_inputs_respects_batch_size_and_length() -> None:
    inputs = prepare_inputs(TinyTokenizer(), "one two three", batch_size=2, max_input_length=4, device="cpu")
    assert inputs["input_ids"].shape == (2, 3)


def test_write_benchmark_summary_persists_json(tmp_path) -> None:
    summary_path = write_benchmark_summary(tmp_path / "summary.json", {"model": {"name": "distilgpt2"}})
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["model"]["name"] == "distilgpt2"


def test_write_trace_rows_csv_persists_rows(tmp_path) -> None:
    csv_path = write_trace_rows_csv(
        tmp_path / "rows.csv",
        [{"step": 0, "latency_seconds": 0.1, "measured": True}],
    )
    content = csv_path.read_text(encoding="utf-8")
    assert "latency_seconds" in content


def test_choose_device_honors_explicit_value() -> None:
    assert choose_device("cpu") == "cpu"


def test_real_model_config_defaults_are_small_and_narrow() -> None:
    config = RealModelBenchmarkConfig()
    assert config.model_name == "distilgpt2"
    assert config.batch_size == 1
