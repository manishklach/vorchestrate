"""Shared constants and enums for vOrchestrate core modules."""

from __future__ import annotations

from enum import IntEnum


class WeightState(IntEnum):
    """Named residency states used by the controller."""

    S0_HBM_FULL_PRECISION = 0
    S1_HBM_LOW_BIT = 1
    S2_HBM_COMPRESSED = 2
    S3_HOST_DRAM = 3
    S4_NVME = 4
    S5_IN_FLIGHT = 5
    S6_RECOMPUTABLE = 6


STATE_HBM_FULL_PRECISION = WeightState.S0_HBM_FULL_PRECISION
STATE_HBM_LOW_BIT = WeightState.S1_HBM_LOW_BIT
STATE_HBM_COMPRESSED = WeightState.S2_HBM_COMPRESSED
STATE_HOST_DRAM = WeightState.S3_HOST_DRAM
STATE_NVME = WeightState.S4_NVME
STATE_IN_FLIGHT = WeightState.S5_IN_FLIGHT
STATE_RECOMPUTABLE = WeightState.S6_RECOMPUTABLE

HBM_RESIDENT_STATES = {
    STATE_HBM_FULL_PRECISION,
    STATE_HBM_LOW_BIT,
    STATE_HBM_COMPRESSED,
}

ALL_WEIGHT_STATES = {
    STATE_HBM_FULL_PRECISION,
    STATE_HBM_LOW_BIT,
    STATE_HBM_COMPRESSED,
    STATE_HOST_DRAM,
    STATE_NVME,
    STATE_IN_FLIGHT,
    STATE_RECOMPUTABLE,
}


def state_label(state: int | WeightState) -> str:
    """Return the canonical `S0`-style label for a residency state."""
    return f"S{int(state)}"
