"""Smoke tests for the abstract adapter surface."""

from __future__ import annotations

import pytest

from vorchestrate.integrations import ResidencyAdapter


class DummyAdapter(ResidencyAdapter):
    """Concrete adapter used for smoke testing."""

    def enumerate_blocks(self):
        return ["block0"]

    def get_block_metadata(self, block_id: str):
        return {"block_id": block_id}

    def request_block(self, block_id: str):
        return block_id

    def on_inference_step(self, step_ctx: dict[str, object]) -> None:
        self.last_step = step_ctx["step"]


def test_adapter_can_be_subclassed() -> None:
    adapter = DummyAdapter()
    assert list(adapter.enumerate_blocks()) == ["block0"]
    assert adapter.get_block_metadata("block0")["block_id"] == "block0"


def test_abstract_adapter_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        ResidencyAdapter()
