"""Tests for accuracy guardrails."""

import pytest

from vorchestrate.core import (
    STATE_HBM_LOW_BIT,
    STATE_HOST_DRAM,
    AccuracyGuardrail,
    WeightBlockRegistry,
)


@pytest.fixture
def registry() -> WeightBlockRegistry:
    """Create a registry with sensitive and insensitive blocks."""
    registry = WeightBlockRegistry()
    registry.register_block("safe", 100, criticality=0.4, sensitivity=0.4)
    registry.register_block("fragile", 100, criticality=0.8, sensitivity=0.9)
    return registry


def test_check_demotion_allows_low_sensitivity_block(registry: WeightBlockRegistry) -> None:
    guardrail = AccuracyGuardrail()
    block = registry.get_block("safe:0")
    assert guardrail.check_demotion(block, STATE_HOST_DRAM) is True


def test_check_demotion_vetoes_sensitive_block(registry: WeightBlockRegistry) -> None:
    guardrail = AccuracyGuardrail()
    block = registry.get_block("fragile:0")
    assert guardrail.check_demotion(block, STATE_HOST_DRAM) is False


def test_check_demotion_allows_sensitive_block_to_stay_at_s1(registry: WeightBlockRegistry) -> None:
    guardrail = AccuracyGuardrail()
    block = registry.get_block("fragile:0")
    assert guardrail.check_demotion(block, STATE_HBM_LOW_BIT) is True


def test_set_protection_marks_registry_entry(registry: WeightBlockRegistry) -> None:
    guardrail = AccuracyGuardrail()
    guardrail.set_protection("fragile:0", registry)
    assert registry.get_block("fragile:0").eviction_protected is True


def test_release_protection_clears_registry_entry(registry: WeightBlockRegistry) -> None:
    guardrail = AccuracyGuardrail()
    guardrail.set_protection("fragile:0", registry)
    guardrail.release_protection("fragile:0", registry)
    assert registry.get_block("fragile:0").eviction_protected is False


def test_get_protected_blocks_lists_only_marked_blocks(registry: WeightBlockRegistry) -> None:
    guardrail = AccuracyGuardrail()
    guardrail.set_protection("fragile:0", registry)
    protected = guardrail.get_protected_blocks(registry)
    assert protected == ["fragile:0"]


def test_custom_threshold_changes_behavior(registry: WeightBlockRegistry) -> None:
    guardrail = AccuracyGuardrail(psi_threshold=0.95)
    block = registry.get_block("fragile:0")
    assert guardrail.check_demotion(block, STATE_HOST_DRAM) is True


def test_setting_protection_for_unknown_block_raises(registry: WeightBlockRegistry) -> None:
    guardrail = AccuracyGuardrail()
    with pytest.raises(KeyError):
        guardrail.set_protection("missing", registry)


def test_releasing_protection_for_unknown_block_raises(registry: WeightBlockRegistry) -> None:
    guardrail = AccuracyGuardrail()
    with pytest.raises(KeyError):
        guardrail.release_protection("missing", registry)


def test_get_protected_blocks_empty_by_default(registry: WeightBlockRegistry) -> None:
    guardrail = AccuracyGuardrail()
    assert guardrail.get_protected_blocks(registry) == []
