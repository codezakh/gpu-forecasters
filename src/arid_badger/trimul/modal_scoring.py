"""Modal-based TriMul scoring.

Runs the TriMul scoring pipeline on a Modal GPU container. Uses
``single_use_containers=True`` / ``max_inputs=1`` per the container
lifecycle note in project memory (cold-start between calls is acceptable
for this port — reference-timing cache can be added later).

Usage:

    from arid_badger.trimul.modal_scoring import modal_trimul_scoring_session

    with modal_trimul_scoring_session(gpu="L4") as score:
        result = score(candidate_source, test_args)
        if is_ok(result):
            print(result.unwrap().runtime_ns)
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
    def evaluate(
        self,
        mutated_kernel_code: str,
        test_args: "dict[str, object]",
        max_repeats: int = 100,
        max_time_ns: float = 10e9,
    ) -> Option[TriMulExecResult, ScoringError]:
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

        try:
            result = score(
                mutated_kernel_code,
                cast(TriMulTestArgs, cast(object, test_args)),
                max_repeats=max_repeats,
                max_time_ns=max_time_ns,
            )
        except (torch.cuda.CudaError, Exception) as exc:
            exc_type = type(exc).__name__
            if "CudaError" in exc_type or "AcceleratorError" in exc_type:
                modal.experimental.stop_fetching_inputs()
            return Err(
                ScoringError(
                    reason=f"TriMul Modal evaluate raised {exc_type}: {exc}",
                    cause=str(exc),
                )
            )

        torch.cuda.empty_cache()
        return result


TriMulScoringFn = Callable[
    [str, TriMulTestArgs], Option[TriMulExecResult, ScoringError]
]


@contextmanager
def modal_trimul_scoring_session(
    gpu: str = _DEFAULT_GPU,
    max_repeats: int = 100,
    max_time_ns: float = 10e9,
) -> Generator[TriMulScoringFn, None, None]:
    """Open a Modal session and yield a ``score(src, test_args)`` callable."""
    benchmarker = ModalTriMulBenchmarker.with_options(  # ty: ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue]
        gpu=gpu
    )

    with app.run():

        def score_fn(
            mutated_kernel_code: str,
            test_args: TriMulTestArgs,
        ) -> Option[TriMulExecResult, ScoringError]:
            try:
                outcome: Option[TriMulExecResult, ScoringError] = (
                    benchmarker().evaluate.remote(
                        mutated_kernel_code=mutated_kernel_code,
                        test_args=dict(test_args),
                        max_repeats=max_repeats,
                        max_time_ns=max_time_ns,
                    )
                )
                return outcome
            except Exception as exc:
                return Err(
                    ScoringError(
                        reason=f"Modal call failed: {type(exc).__name__}: {exc}",
                        cause=str(exc),
                    )
                )

        yield score_fn
