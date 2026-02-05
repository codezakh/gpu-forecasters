from __future__ import annotations

from typing import Callable, cast

from arid_badger.greedy_search.components import (
    MutationContext,
    MutationFunction,
    MutatedKernel,
)
from arid_badger.greedy_search.search import GreedySearch, GreedySearchConfig
from arid_badger.kernelbench.core import KernelScoringResult
from kernelbench.eval import KernelExecResult


def _fake_scoring_result() -> KernelScoringResult:
    exec_result = KernelExecResult(
        compiled=True,
        correctness=True,
        runtime=1.0,
        ref_runtime=2.0,
    )
    return KernelScoringResult(exec_result=exec_result, speedup=2.0, is_valid=True)


def _scoring_function(_: str, __: str) -> KernelScoringResult:
    return _fake_scoring_result()


def _make_mutation_function() -> (
    tuple[Callable[[MutationContext], MutatedKernel], list[int]]
):
    calls = [0]

    def _mutation(context: MutationContext) -> MutatedKernel:
        calls[0] += 1
        return MutatedKernel(
            kernel_code=f"code_{calls[0]}",
            ancestor_ulid=context.previous_kernel_ulid,
        )

    return _mutation, calls


def _make_search(
    *, max_depth: int, num_mutations: int = 2
) -> tuple[GreedySearch, list[int]]:
    mutation_fn, calls = _make_mutation_function()
    mutation = cast(MutationFunction, mutation_fn)
    config = GreedySearchConfig(
        max_depth=max_depth,
        num_mutations=num_mutations,
        starter_kernel_code="starter",
        reference_kernel_code="reference",
        mutation_function=mutation,
        scoring_function=_scoring_function,
    )
    return GreedySearch(config), calls


def test_is_complete_true_at_depth_0_when_max_depth_0() -> None:
    search, calls = _make_search(max_depth=0)
    checkpoint = search.run()

    assert search.is_complete(checkpoint)

    checkpoint_after_step = search.step(checkpoint)
    assert checkpoint_after_step.cursor.next_depth == 0
    assert calls[0] == 0


def test_step_noop_when_complete_does_not_call_mutation() -> None:
    search, calls = _make_search(max_depth=0)
    checkpoint = search.run()

    assert search.is_complete(checkpoint)

    search.step(checkpoint)
    assert calls[0] == 0


def test_step_advances_exactly_one_round() -> None:
    search, calls = _make_search(max_depth=2, num_mutations=2)
    checkpoint = search._create_initial_checkpoint()

    checkpoint = search.step(checkpoint)
    assert checkpoint.cursor.next_depth == 1
    assert len(checkpoint.rounds) == 1
    assert calls[0] == 2

    checkpoint = search.step(checkpoint)
    assert checkpoint.cursor.next_depth == 2
    assert len(checkpoint.rounds) == 2
    assert calls[0] == 4


def test_resume_idempotent_when_complete() -> None:
    search, calls = _make_search(max_depth=1, num_mutations=3)
    checkpoint = search._create_initial_checkpoint()
    checkpoint = search.step(checkpoint)

    assert search.is_complete(checkpoint)
    calls_before = calls[0]

    search.resume(checkpoint)
    assert calls[0] == calls_before
