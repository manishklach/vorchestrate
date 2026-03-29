"""Synthetic controller simulation helpers for truthful prototype exercises."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from ..core import (
    AccuracyGuardrail,
    ControllerMetrics,
    PrefetchScheduler,
    ScoringEngine,
    WeightBlockRegistry,
    WeightStateMachine,
)
from ..core.constants import HBM_RESIDENT_STATES, STATE_HOST_DRAM, STATE_NVME
from .trace import TraceEvent, write_trace_csv, write_trace_json

DEFAULT_SIMULATION_STEPS = 6
DEFAULT_SIMULATION_HBM_BUDGET_BYTES = 20 * 1024 * 1024
DEFAULT_TRACE_STEPS_PER_SECOND = 2_000.0


@dataclass(slots=True)
class SyntheticBlockDescriptor:
    """Synthetic residency-managed unit used in controller simulation."""

    block_id: str
    size_bytes: int
    state: int
    tier: str
    reuse_score: float
    routing_likelihood: float
    criticality: float
    sensitivity: float
    next_use_distance: int
    access_count: int = 0
    transfer_cost_us: float = 750.0
    decomp_cost_us: float = 50.0


@dataclass(slots=True)
class SimulationConfig:
    """Configuration for a synthetic controller simulation."""

    steps: int = DEFAULT_SIMULATION_STEPS
    hbm_budget_bytes: int = DEFAULT_SIMULATION_HBM_BUDGET_BYTES
    psi_threshold: float = 0.7
    steps_per_second: float = DEFAULT_TRACE_STEPS_PER_SECOND
    output_dir: str | None = None


@dataclass(slots=True)
class SimulationResult:
    """Structured output of a controller simulation."""

    events: List[TraceEvent]
    metrics: ControllerMetrics
    scores_by_step: Dict[int, Dict[str, float]]


def default_synthetic_descriptors() -> List[SyntheticBlockDescriptor]:
    """Return a deterministic set of synthetic block descriptors."""
    return [
        SyntheticBlockDescriptor(
            block_id="embed_hot",
            size_bytes=8 * 1024 * 1024,
            state=0,
            tier="HBM",
            reuse_score=0.92,
            routing_likelihood=0.0,
            criticality=0.90,
            sensitivity=0.85,
            next_use_distance=1,
            transfer_cost_us=300.0,
        ),
        SyntheticBlockDescriptor(
            block_id="attn_mid",
            size_bytes=7 * 1024 * 1024,
            state=1,
            tier="HBM",
            reuse_score=0.55,
            routing_likelihood=0.0,
            criticality=0.80,
            sensitivity=0.60,
            next_use_distance=2,
            transfer_cost_us=700.0,
        ),
        SyntheticBlockDescriptor(
            block_id="mlp_cold",
            size_bytes=9 * 1024 * 1024,
            state=3,
            tier="DRAM",
            reuse_score=0.18,
            routing_likelihood=0.0,
            criticality=0.50,
            sensitivity=0.35,
            next_use_distance=5,
            transfer_cost_us=1800.0,
        ),
        SyntheticBlockDescriptor(
            block_id="moe_hot_expert",
            size_bytes=6 * 1024 * 1024,
            state=3,
            tier="DRAM",
            reuse_score=0.40,
            routing_likelihood=0.88,
            criticality=0.75,
            sensitivity=0.55,
            next_use_distance=1,
            transfer_cost_us=1400.0,
        ),
        SyntheticBlockDescriptor(
            block_id="archive_expert",
            size_bytes=10 * 1024 * 1024,
            state=4,
            tier="NVMe",
            reuse_score=0.06,
            routing_likelihood=0.05,
            criticality=0.30,
            sensitivity=0.20,
            next_use_distance=8,
            transfer_cost_us=4200.0,
        ),
    ]


def run_controller_simulation(
    descriptors: Sequence[SyntheticBlockDescriptor] | None = None,
    config: SimulationConfig | None = None,
) -> SimulationResult:
    """Run a synthetic controller simulation and return structured results."""
    simulation_config = config or SimulationConfig()
    blocks = list(descriptors or default_synthetic_descriptors())

    registry = WeightBlockRegistry(hbm_capacity_bytes=simulation_config.hbm_budget_bytes)
    descriptor_map: Dict[str, SyntheticBlockDescriptor] = {}
    registry_ids: Dict[str, str] = {}

    for descriptor in blocks:
        registry_id = registry.register_block(
            layer_name=descriptor.block_id,
            size_bytes=descriptor.size_bytes,
            criticality=descriptor.criticality,
            sensitivity=descriptor.sensitivity,
        )
        block = registry.get_block(registry_id)
        block.current_state = descriptor.state
        block.reuse_score = descriptor.reuse_score
        block.routing_likelihood = descriptor.routing_likelihood
        block.transfer_cost_us = descriptor.transfer_cost_us
        block.decomp_cost_us = descriptor.decomp_cost_us
        block.predicted_next_access = descriptor.next_use_distance
        descriptor_map[registry_id] = descriptor
        registry_ids[descriptor.block_id] = registry_id

    guardrail = AccuracyGuardrail(psi_threshold=simulation_config.psi_threshold)
    scorer = ScoringEngine(registry)
    scheduler = PrefetchScheduler(registry)
    state_machine = WeightStateMachine(registry, scorer, guardrail, scheduler=scheduler)
    metrics = ControllerMetrics()
    events: List[TraceEvent] = []
    scores_by_step: Dict[int, Dict[str, float]] = {}

    try:
        for step in range(simulation_config.steps):
            scores = scorer.score_all_blocks()
            scores_by_step[step] = scores

            for registry_id, score in scores.items():
                block = registry.get_block(registry_id)
                descriptor = descriptor_map[registry_id]

                if step % max(descriptor.next_use_distance, 1) == 0:
                    registry.update_access(registry_id, step)
                    descriptor.access_count += 1
                    block.predicted_next_access = step + max(1, descriptor.next_use_distance)

                prefetch = scheduler.should_prefetch(
                    block,
                    current_step=step,
                    steps_per_second=simulation_config.steps_per_second,
                )
                if prefetch and block.current_state not in HBM_RESIDENT_STATES:
                    scheduler.enqueue_promotion(registry_id, priority=1)
                    metrics.record_prefetch()
                    events.append(
                        TraceEvent(
                            step=step,
                            block_id=descriptor.block_id,
                            score=score,
                            psi=block.quality_sensitivity,
                            old_state=block.current_state,
                            new_state=block.current_state,
                            old_tier=_state_to_tier(block.current_state),
                            new_tier=_state_to_tier(block.current_state),
                            action="prefetch",
                            guardrail_veto=False,
                            bytes_moved=0,
                        )
                    )

                target_state = scorer.get_target_state(registry_id, registry.get_hbm_pressure())
                old_state = block.current_state
                guardrail_veto = old_state < target_state and not guardrail.check_demotion(block, target_state)
                action = "keep"
                bytes_moved = 0

                if guardrail_veto:
                    metrics.record_guardrail_veto()
                    state_machine.transition(registry_id, target_state)
                    action = "guardrail_veto"
                elif target_state != old_state:
                    transitioned = state_machine.transition(registry_id, target_state)
                    if transitioned:
                        bytes_moved = block.size_bytes
                        metrics.record_transition(old_state, target_state, block.size_bytes)
                        if target_state in {STATE_HOST_DRAM, STATE_NVME}:
                            metrics.record_stage()
                            action = "stage"
                        elif target_state in HBM_RESIDENT_STATES:
                            action = "promote"
                        else:
                            action = "transition"
                    else:
                        metrics.record_guardrail_veto()
                        action = "guardrail_veto"

                new_state = registry.get_block(registry_id).current_state
                events.append(
                    TraceEvent(
                        step=step,
                        block_id=descriptor.block_id,
                        score=score,
                        psi=block.quality_sensitivity,
                        old_state=old_state,
                        new_state=new_state,
                        old_tier=_state_to_tier(old_state),
                        new_tier=_state_to_tier(new_state),
                        action=action,
                        guardrail_veto=guardrail_veto,
                        bytes_moved=bytes_moved,
                    )
                )

            state_machine.tick(step, simulation_config.hbm_budget_bytes)

        result = SimulationResult(events=events, metrics=metrics, scores_by_step=scores_by_step)
        if simulation_config.output_dir:
            output_dir = Path(simulation_config.output_dir)
            write_trace_json(output_dir / "simulation_trace.json", events)
            write_trace_csv(output_dir / "simulation_trace.csv", events)
        return result
    finally:
        scheduler.shutdown()


def _state_to_tier(state: int) -> str:
    """Map a residency state to a coarse tier label."""
    if state in HBM_RESIDENT_STATES:
        return "HBM"
    if state == STATE_HOST_DRAM:
        return "DRAM"
    if state == STATE_NVME:
        return "NVMe"
    return f"S{state}"
