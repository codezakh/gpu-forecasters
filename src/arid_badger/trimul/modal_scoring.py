"""Modal-based TriMul scoring.

Runs the TriMul scoring pipeline on a Modal GPU container. Each container
call scores one candidate against *all* test cases — fan-out over cases
happens sequentially inside the container, fan-out over candidates is
handled by the provider's thread pool.

Usage:

    from arid_badger.trimul.modal_scoring import modal_trimul_scoring_session

    with modal_trimul_scoring_session(gpu="L4") as score:
        results = score(candidate_source, [test_args_1, test_args_2])
        for r in results:
            if is_ok(r):
                print(r.unwrap().runtime_ns)
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable, Generator, cast

import modal

from arid_badger.kernelbench.isolated_scoring import ScoringError
from arid_badger.kernelbench.modal_image import image
from arid_badger.trimul.cases import TriMulTestArgs
from arid_badger.trimul.core import TriMulExecResult
from arid_badger.typing_utils import Err, Option


app = modal.App("arid-badger-trimul")

_DEFAULT_GPU = "A100-80GB"
_GPU_WAIT_TIMEOUT_S = 30
_GPU_WAIT_INITIAL_DELAY_S = 0.5
_GPU_WAIT_MAX_DELAY_S = 8.0


@app.cls(
    image=image,
    gpu=_DEFAULT_GPU,
    timeout=1200,
    max_containers=40,
    retries=modal.Retries(
        max_retries=3,
        backoff_coefficient=2.0,
        initial_delay=1.0,
    ),
)
class ModalTriMulBenchmarker:
    """Runs TriMul scoring inside a Modal GPU container.

    Internal — use ``modal_trimul_scoring_session`` from experiment code.
    """

    @modal.method()
    def evaluate_candidate(
        self,
        mutated_kernel_code: str,
        test_cases: list[dict[str, object]],
        max_repeats: int = 100,
        max_time_ns: float = 10e9,
    ) -> list[Option[TriMulExecResult, ScoringError]]:
        import torch

        from arid_badger.trimul.scoring import score

        start = time.time()
        while not torch.cuda.is_available():
            elapsed = time.time() - start
            if elapsed >= _GPU_WAIT_TIMEOUT_S:
                raise RuntimeError(f"GPU not available after {_GPU_WAIT_TIMEOUT_S}s")
            delay = min(
                _GPU_WAIT_INITIAL_DELAY_S * (2 ** int(elapsed / 2)),
                _GPU_WAIT_MAX_DELAY_S,
            )
            time.sleep(delay)

        results: list[Option[TriMulExecResult, ScoringError]] = []
        for test_case in test_cases:
            try:
                result = score(
                    mutated_kernel_code,
                    cast(TriMulTestArgs, cast(object, test_case)),
                    max_repeats=max_repeats,
                    max_time_ns=max_time_ns,
                )
            except (torch.cuda.CudaError, Exception) as exc:
                exc_type = type(exc).__name__
                if "CudaError" in exc_type or "AcceleratorError" in exc_type:
                    # GPU is poisoned — stop accepting further inputs on
                    # this container but still return partial results.
                    modal.experimental.stop_fetching_inputs()
                    results.append(
                        Err(
                            ScoringError(
                                reason=f"TriMul Modal evaluate raised {exc_type}: {exc}",
                                cause=str(exc),
                            )
                        )
                    )
                    break
                results.append(
                    Err(
                        ScoringError(
                            reason=f"TriMul Modal evaluate raised {exc_type}: {exc}",
                            cause=str(exc),
                        )
                    )
                )
                continue

            torch.cuda.empty_cache()
            results.append(result)

        return results


TriMulScoringFn = Callable[
    [str, list[TriMulTestArgs]], list[Option[TriMulExecResult, ScoringError]]
]


@contextmanager
def modal_trimul_scoring_session(
    gpu: str = _DEFAULT_GPU,
    max_repeats: int = 100,
    max_time_ns: float = 10e9,
) -> Generator[TriMulScoringFn, None, None]:
    """Open a Modal session and yield a ``score(src, test_cases)`` callable."""
    benchmarker = ModalTriMulBenchmarker.with_options(  # ty: ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue]
        gpu=gpu
    )

    with app.run():

        def score_fn(
            mutated_kernel_code: str,
            test_cases: list[TriMulTestArgs],
        ) -> list[Option[TriMulExecResult, ScoringError]]:
            try:
                outcome: list[Option[TriMulExecResult, ScoringError]] = (
                    benchmarker().evaluate_candidate.remote(
                        mutated_kernel_code=mutated_kernel_code,
                        test_cases=[dict(tc) for tc in test_cases],
                        max_repeats=max_repeats,
                        max_time_ns=max_time_ns,
                    )
                )
                return outcome
            except Exception as exc:
                # Container-level failure — return Err for every case
                err = Err(
                    ScoringError(
                        reason=f"Modal call failed: {type(exc).__name__}: {exc}",
                        cause=str(exc),
                    )
                )
                return [err] * len(test_cases)

        yield score_fn
