import numpy as np

from arid_badger.ttt_discover.v2.archive.puct_math import (
    compute_prior,
    compute_scale,
    compute_scores,
)


def test_prior_sums_to_one_and_gives_largest_weight_to_max() -> None:
    rewards = np.array([0.1, 5.0, 2.0])
    prior = compute_prior(rewards)
    assert abs(prior.sum() - 1.0) < 1e-9
    # Index 1 has the largest reward → largest prior.
    assert np.argmax(prior) == 1


def test_prior_empty() -> None:
    assert compute_prior(np.array([])).size == 0


def test_scale_is_clamped() -> None:
    assert compute_scale(np.array([1.0, 1.0, 1.0])) >= 1e-6
    assert compute_scale(np.array([])) == 1.0


def test_score_prefers_high_q_when_visits_equal() -> None:
    rewards = np.array([0.5, 1.0])
    priors = compute_prior(rewards)
    n = np.array([0.0, 0.0])
    m = np.array([0.0, 0.0])
    scores = compute_scores(
        rewards=rewards,
        priors=priors,
        n=n,
        m=m,
        total_visits=0,
        scale=1.0,
        puct_c=1.0,
    )
    # Q = reward when n=0; reward[1] > reward[0] so scores[1] > scores[0].
    assert scores[1] > scores[0]


def test_score_uses_m_when_visits_positive() -> None:
    rewards = np.array([0.5, 1.0])
    priors = compute_prior(rewards)
    n = np.array([5.0, 0.0])
    m = np.array([10.0, 0.0])  # huge Q on index 0
    scores = compute_scores(
        rewards=rewards,
        priors=priors,
        n=n,
        m=m,
        total_visits=5,
        scale=1.0,
        puct_c=0.1,
    )
    assert scores[0] > scores[1]


def test_score_bonus_shrinks_with_visits() -> None:
    rewards = np.array([1.0])
    priors = compute_prior(rewards)
    score_unvisited = compute_scores(
        rewards=rewards,
        priors=priors,
        n=np.array([0.0]),
        m=np.array([0.0]),
        total_visits=10,
        scale=1.0,
        puct_c=1.0,
    )[0]
    score_visited = compute_scores(
        rewards=rewards,
        priors=priors,
        n=np.array([100.0]),
        m=np.array([1.0]),
        total_visits=10,
        scale=1.0,
        puct_c=1.0,
    )[0]
    # Both land at Q=1.0, but the unvisited one has a larger bonus.
    assert score_unvisited > score_visited
