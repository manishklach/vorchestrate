"""MoE-flavored usage example for routing-aware scoring."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from vorchestrate.core import ScoringEngine, WeightBlockRegistry


def main() -> None:
    """Create a simple registry and inject routing likelihoods."""
    registry = WeightBlockRegistry(hbm_capacity_bytes=256 * 1024 * 1024)
    hot_expert = registry.register_block("moe.expert0", 8 * 1024 * 1024, criticality=0.8, sensitivity=0.65)
    cold_expert = registry.register_block("moe.expert7", 8 * 1024 * 1024, criticality=0.5, sensitivity=0.4)

    registry.get_block(hot_expert).routing_likelihood = 0.92
    registry.get_block(cold_expert).routing_likelihood = 0.08
    registry.update_access(hot_expert, current_step=100)
    registry.update_access(cold_expert, current_step=12)

    scorer = ScoringEngine(registry)
    print("scores:", scorer.score_all_blocks())
    print("promotion order:", scorer.rank_blocks_for_promotion())


if __name__ == "__main__":
    main()
