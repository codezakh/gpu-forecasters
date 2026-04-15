from arid_badger.kernelbench.metrics import compute_fast1_score


def test_empty_returns_zero() -> None:
    assert compute_fast1_score([]) == 0.0


def test_none_and_sub_unit_rewards_do_not_count() -> None:
    # None = failure, 1.0 = no speedup, 0.5 = slowdown → none of these count.
    assert compute_fast1_score([None, 1.0, 0.5]) == 0.0


def test_strictly_greater_than_one() -> None:
    # 3/4 rewards are strictly > 1.0.
    rewards: list[float | None] = [2.0, 1.5, 1.0, 3.0]
    assert compute_fast1_score(rewards) == 0.75


def test_consumes_iterables_not_just_lists() -> None:
    from collections.abc import Iterator

    def gen() -> Iterator[float | None]:
        yield 2.0
        yield None
        yield 1.5

    assert compute_fast1_score(gen()) == 2 / 3
