"""Simple memory profiling helpers for vOrchestrate benchmarks."""

from __future__ import annotations

from ..core import WeightBlockRegistry


def profile_registry_memory(registry: WeightBlockRegistry) -> dict[str, float]:
    """Summarize logical memory placement statistics for the registry.

    Args:
        registry: Weight registry to summarize.

    Returns:
        Dictionary with bytes tracked and current HBM pressure.
    """
    blocks = registry.get_all_blocks()
    total_bytes = sum(block.size_bytes for block in blocks)
    resident_bytes = sum(
        block.size_bytes
        for block in blocks
        if block.current_state in {0, 1, 2}
    )
    return {
        "total_bytes": float(total_bytes),
        "resident_hbm_bytes": float(resident_bytes),
        "hbm_pressure": registry.get_hbm_pressure(),
    }


def format_memory_profile_rows(registry: WeightBlockRegistry) -> list[dict[str, object]]:
    """Return row-wise placement summaries for external benchmark scripts.

    Args:
        registry: Weight registry to summarize.

    Returns:
        Per-block rows with state and byte count.
    """
    return [
        {
            "block_id": block.block_id,
            "layer_name": block.layer_name,
            "state": block.current_state,
            "size_bytes": block.size_bytes,
        }
        for block in registry.get_all_blocks()
    ]
