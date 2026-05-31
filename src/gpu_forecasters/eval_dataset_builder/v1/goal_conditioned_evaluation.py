"""Goal-conditioned evaluation provider — pack-generic.

Wraps an inner ``AsyncEvaluationProvider`` and substitutes the
``Evaluation.reward`` field with a goal-conditioned score:
``-|log(speedup) - log(target_midpoint_speedup)|`` on the success arm,
``None`` on every failure arm.

The substitution is the load-bearing intervention for goal-directed
search: PUCT prunes by ``evaluation.reward`` (top-K-per-parent and
global archive truncation), so without rewriting the reward, even
in-band kernels get evicted by faster out-of-band siblings. The
observation passes through unchanged so downstream consumers
(eval-set construction, manifest) can still read
``feedback.aggregated_speedup``.
"""

from __future__ import annotations

import math
from concurrent.futures import Future
from typing import Generic, Self

from gpu_forecasters.gpu_mode_kernel.core import (
    CaseSpeedupT,
    GpuModeKernelObservation,
    SuccessFeedback,
)
from gpu_forecasters.hill_climbing.domain import Evaluation
from gpu_forecasters.landscape_map.v1.domain import SpeedupBin
from gpu_forecasters.max_reward_puct.v2.providers import AsyncEvaluationProvider

from .domain import speedup_band_for_bin


def score_evaluation_against_target_bin(
    evaluation: Evaluation[GpuModeKernelObservation[CaseSpeedupT]],
    *,
    target_bin: SpeedupBin,
) -> float | None:
    """Goal-conditioned score: ``-|log(speedup) - log(midpoint)|`` on
    success; ``None`` otherwise."""
    feedback = evaluation.observation.feedback
    if not isinstance(feedback, SuccessFeedback):
        return None
    speedup = feedback.aggregated_speedup
    if speedup <= 0:
        return None
    log_midpoint = math.log(speedup_band_for_bin(target_bin).midpoint)
    return -abs(math.log(speedup) - log_midpoint)


class GoalConditionedEvaluationProvider(Generic[CaseSpeedupT]):
    """Wraps an ``AsyncEvaluationProvider`` for gpu-mode-pack observations,
    rewriting each completed ``Evaluation.reward`` to a goal-conditioned
    score.

    Lifecycle methods are no-ops: the wrapper is a stateless adapter and
    its caller (e.g. ``BinFiller``) owns the inner provider's lifecycle
    explicitly. ``__enter__``/``__exit__`` exist only so the wrapper
    satisfies the ``AsyncEvaluationProvider`` Protocol shape.
    """

    def __init__(
        self,
        *,
        inner_evaluation_provider: AsyncEvaluationProvider[
            GpuModeKernelObservation[CaseSpeedupT]
        ],
        target_bin: SpeedupBin,
    ) -> None:
        if target_bin is SpeedupBin.FAILURE:
            raise ValueError(
                "target_bin=FAILURE is not a meaningful goal — search cannot aim at a non-speedup bin."
            )
        self._inner = inner_evaluation_provider
        self._target_bin = target_bin

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        return None

    def submit(
        self, program_code: str
    ) -> Future[Evaluation[GpuModeKernelObservation[CaseSpeedupT]]]:
        inner_future = self._inner.submit(program_code)
        outer_future: Future[Evaluation[GpuModeKernelObservation[CaseSpeedupT]]] = (
            Future()
        )

        def _remap_completion(
            done: Future[Evaluation[GpuModeKernelObservation[CaseSpeedupT]]],
        ) -> None:
            exc = done.exception()
            if exc is not None:
                outer_future.set_exception(exc)
                return
            inner_evaluation = done.result()
            score = score_evaluation_against_target_bin(
                inner_evaluation, target_bin=self._target_bin
            )
            outer_future.set_result(
                Evaluation[GpuModeKernelObservation[CaseSpeedupT]](
                    observation=inner_evaluation.observation,
                    reward=score,
                )
            )

        inner_future.add_done_callback(_remap_completion)
        return outer_future
