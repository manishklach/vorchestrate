"""Tests for the prefetch scheduler."""

import time

import pytest

from vorchestrate.core import PrefetchScheduler, WeightBlockRegistry


@pytest.fixture
def scheduler() -> PrefetchScheduler:
    """Create a scheduler fixture and tear it down."""
    registry = WeightBlockRegistry(hbm_capacity_bytes=1000)
    block_id = registry.register_block("layer", 128, criticality=0.5, sensitivity=0.5)
    registry.get_block(block_id).predicted_next_access = 10
    sched = PrefetchScheduler(registry, max_bandwidth_bytes_per_sec=1024.0)
    yield sched
    sched.shutdown()


def test_compute_prefetch_window_adds_margin(scheduler: PrefetchScheduler) -> None:
    block = scheduler.registry.get_block("layer:0")
    block.transfer_cost_us = 100.0
    block.decomp_cost_us = 50.0
    assert scheduler.compute_prefetch_window(block) == pytest.approx(650.0)


def test_should_prefetch_true_when_access_is_imminent(scheduler: PrefetchScheduler) -> None:
    block = scheduler.registry.get_block("layer:0")
    block.predicted_next_access = 11
    block.transfer_cost_us = 200_000.0
    assert scheduler.should_prefetch(block, current_step=10, steps_per_second=10.0) is True


def test_should_prefetch_false_when_steps_per_second_invalid(scheduler: PrefetchScheduler) -> None:
    block = scheduler.registry.get_block("layer:0")
    assert scheduler.should_prefetch(block, current_step=10, steps_per_second=0.0) is False


def test_enqueue_promotion_increases_queue_depth(scheduler: PrefetchScheduler) -> None:
    scheduler.enqueue_promotion("layer:0", priority=1)
    assert scheduler.get_queue_depth() >= 1


def test_enqueue_demotion_increases_queue_depth(scheduler: PrefetchScheduler) -> None:
    scheduler.enqueue_demotion("layer:0", priority=10)
    assert scheduler.get_queue_depth() >= 1


def test_worker_processes_commands_in_background(scheduler: PrefetchScheduler) -> None:
    scheduler.enqueue_promotion("layer:0", priority=1)
    time.sleep(0.1)
    assert scheduler.get_queue_depth() == 0


def test_bandwidth_utilization_increases_after_processed_transfer(scheduler: PrefetchScheduler) -> None:
    scheduler.enqueue_promotion("layer:0", priority=1)
    time.sleep(0.1)
    assert scheduler.get_bandwidth_utilization() > 0.0


def test_demotions_can_queue_alongside_promotions(scheduler: PrefetchScheduler) -> None:
    scheduler.enqueue_promotion("layer:0", priority=1)
    scheduler.enqueue_demotion("layer:0", priority=10)
    assert scheduler.get_queue_depth() >= 1


def test_should_prefetch_false_for_past_prediction(scheduler: PrefetchScheduler) -> None:
    block = scheduler.registry.get_block("layer:0")
    block.predicted_next_access = 5
    assert scheduler.should_prefetch(block, current_step=10, steps_per_second=20.0) is False


def test_scheduler_shutdown_is_idempotent(scheduler: PrefetchScheduler) -> None:
    scheduler.shutdown()
    scheduler.shutdown()
