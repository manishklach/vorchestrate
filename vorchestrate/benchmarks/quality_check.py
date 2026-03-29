"""Quality utilities for accuracy regression benchmarking."""

from __future__ import annotations

import math
from typing import Iterable


def compute_perplexity_delta(baseline_losses: Iterable[float], candidate_losses: Iterable[float]) -> float:
    """Compute perplexity delta between baseline and candidate loss traces.

    Args:
        baseline_losses: Baseline negative log-likelihood values.
        candidate_losses: Candidate negative log-likelihood values.

    Returns:
        Candidate perplexity minus baseline perplexity.

    Raises:
        ValueError: If no samples are supplied or list lengths mismatch.
    """
    baseline = list(baseline_losses)
    candidate = list(candidate_losses)
    if not baseline or not candidate:
        raise ValueError("loss traces must be non-empty")
    if len(baseline) != len(candidate):
        raise ValueError("loss traces must have matching lengths")

    baseline_ppl = math.exp(sum(baseline) / len(baseline))
    candidate_ppl = math.exp(sum(candidate) / len(candidate))
    return candidate_ppl - baseline_ppl
