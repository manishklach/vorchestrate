"""Lightweight metrics collection for controller simulations."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ControllerMetrics:
    """Accumulate coarse controller activity counters."""

    promotions: int = 0
    demotions: int = 0
    prefetches: int = 0
    stages: int = 0
    guardrail_vetoes: int = 0
    bytes_promoted: int = 0
    bytes_demoted: int = 0
    transition_counts: dict[str, int] = field(default_factory=dict)

    def record_transition(self, old_state: int, new_state: int, size_bytes: int) -> None:
        """Record a transition and associated byte movement."""
        transition_key = f"S{old_state}->S{new_state}"
        self.transition_counts[transition_key] = self.transition_counts.get(transition_key, 0) + 1
        if new_state < old_state:
            self.promotions += 1
            self.bytes_promoted += size_bytes
        elif new_state > old_state:
            self.demotions += 1
            self.bytes_demoted += size_bytes

    def record_prefetch(self) -> None:
        """Record a prefetch decision."""
        self.prefetches += 1

    def record_stage(self) -> None:
        """Record a stage-to-colder-tier decision."""
        self.stages += 1

    def record_guardrail_veto(self) -> None:
        """Record a guardrail veto."""
        self.guardrail_vetoes += 1

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable metrics snapshot."""
        return {
            "promotions": self.promotions,
            "demotions": self.demotions,
            "prefetches": self.prefetches,
            "stages": self.stages,
            "guardrail_vetoes": self.guardrail_vetoes,
            "bytes_promoted": self.bytes_promoted,
            "bytes_demoted": self.bytes_demoted,
            "transition_counts": dict(self.transition_counts),
        }
