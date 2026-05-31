"""Pure PUCT math — rank-based prior + scale factor + selection score.

Lifted from ``gpu_forecasters.ttt_discover.v1.tinker_utils.sampler.PUCTSampler``
with everything state-mutating / file-backed stripped. Each function
takes plain numpy arrays / mappings and returns plain values; the
``PUCTCandidateArchive`` composes them with its own bookkeeping.

Formula (from v1):

    score(i) = Q(i) + c * scale * P(i) * sqrt(1 + T) / (1 + n[i])

where ``Q(i) = m[i]`` if ``n[i] > 0`` else the candidate's own reward,
``P(i)`` is a rank-based prior (rank 0 = largest reward weight), and
``scale`` is the reward range (clamped to 1e-6).
"""

from __future__ import annotations

import math

import numpy as np


def compute_scale(rewards: np.ndarray) -> float:
    """Reward range, clamped to 1e-6 to avoid zero-bonus pathologies."""
    if rewards.size == 0:
        return 1.0
    return float(max(float(np.max(rewards) - np.min(rewards)), 1e-6))


def compute_prior(rewards: np.ndarray) -> np.ndarray:
    """Rank-based prior over ``rewards``: rank 0 gets the largest weight,
    and weights sum to 1. Empty input → empty output."""
    if rewards.size == 0:
        return np.array([])
    n = len(rewards)
    ranks = np.argsort(np.argsort(-rewards))
    weights = (n - ranks).astype(np.float64)
    return weights / weights.sum()


def compute_scores(
    rewards: np.ndarray,
    priors: np.ndarray,
    n: np.ndarray,
    m: np.ndarray,
    total_visits: int,
    scale: float,
    puct_c: float,
) -> np.ndarray:
    """One selection score per candidate; larger is better."""
    if rewards.size == 0:
        return np.array([])
    q = np.where(n > 0, m, rewards)
    sqrt_t = math.sqrt(1.0 + float(total_visits))
    bonus = puct_c * scale * priors * sqrt_t / (1.0 + n)
    return q + bonus
