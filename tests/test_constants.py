"""Tests for state constants and enum compatibility."""

from __future__ import annotations

from vorchestrate.core import WeightState, state_label


def test_state_label_formats_enum_and_int() -> None:
    assert state_label(WeightState.S0_HBM_FULL_PRECISION) == "S0"
    assert state_label(4) == "S4"


def test_weight_state_behaves_like_int() -> None:
    assert int(WeightState.S3_HOST_DRAM) == 3
