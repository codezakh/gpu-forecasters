"""Modal-based causal conv1d scoring.

Runs the causal conv1d scoring pipeline on a Modal GPU container. Each
container call scores one candidate against *all* test cases — fan-out
over cases happens sequentially inside the container, fan-out over
candidates is handled by the provider's thread pool.

Near-duplicate of ``gpu_forecasters.trimul.modal_scoring``; the only
kernel-specific bits are the Modal app name (so this kernel's container
namespace doesn't collide with TriMul's) and the inner ``score``
import. The rest will be lifted in the gh070-A task #3 extraction.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable, Generator, cast

import modal

from gpu_forecasters.causal_conv1d.cases import CausalConv1dTestArgs
from gpu_forecasters.causal_conv1d.core import CausalConv1dExecResult
from gpu_forecasters.kernelbench.isolated_scoring import ScoringError
from gpu_forecasters.kernelbench.modal_image import image
from gpu_forecasters.typing_utils import Err, Option


app = modal.App("arid-badger-causal-conv1d")

_DEFAULT_GPU = "A100-80GB"
_GPU_WAIT_TIMEOUT_S = 30
_GPU_WAIT_INITIAL_DELAY_S = 0.5
_GPU_WAIT_MAX_DELAY_S = 8.0


@app.cls(
    image=image,
    gpu=_DEFAULT_GPU,
    timeout=1200,
    max_containers=40,
    single_use_containers=True,
    scaledown_window=2,
    retries=modal.Retries(
        max_retries=3,
        backoff_coefficient=2.0,
        initial_delay=1.0,
    ),
)
class ModalCausalConv1dBenchmarker:
    """Runs causal conv1d scoring inside a Modal GPU container.

    Internal — use ``modal_causal_conv1d_scoring_session`` from
    experiment code, or import the class directly for v2-style
    ``spawn``-based dispatch.
    """

    @modal.method()
    def evaluate_candidate(
        self,
        mutated_kernel_code: str,
        test_cases: list[dict[str, object]],
        max_repeats: int = 100,
        max_time_ns: float = 10e9,
    ) -> list[Option[CausalConv1dExecResult, ScoringError]]:
        import torch

        from gpu_forecasters.causal_conv1d.scoring import score

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

        results: list[Option[CausalConv1dExecResult, ScoringError]] = []
        for test_case in test_cases:
            try:
                result = score(
                    mutated_kernel_code,
                    cast(CausalConv1dTestArgs, cast(object, test_case)),
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
                                reason=f"Causal conv1d Modal evaluate raised {exc_type}: {exc}",
                                cause=str(exc),
                            )
                        )
                    )
                    break
                results.append(
                    Err(
                        ScoringError(
                            reason=f"Causal conv1d Modal evaluate raised {exc_type}: {exc}",
                            cause=str(exc),
                        )
                    )
                )
                continue

            torch.cuda.empty_cache()
            results.append(result)

        return results


CausalConv1dScoringFn = Callable[
    [str, list[CausalConv1dTestArgs]],
    list[Option[CausalConv1dExecResult, ScoringError]],
]


@contextmanager
def modal_causal_conv1d_scoring_session(
    gpu: str = _DEFAULT_GPU,
    max_repeats: int = 100,
    max_time_ns: float = 10e9,
) -> Generator[CausalConv1dScoringFn, None, None]:
    """Open a Modal session and yield a ``score(src, test_cases)`` callable."""
    benchmarker = ModalCausalConv1dBenchmarker.with_options(  # ty: ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue]
        gpu=gpu
    )

    with app.run():

        def score_fn(
            mutated_kernel_code: str,
            test_cases: list[CausalConv1dTestArgs],
        ) -> list[Option[CausalConv1dExecResult, ScoringError]]:
            try:
                outcome: list[
                    Option[CausalConv1dExecResult, ScoringError]
                ] = benchmarker().evaluate_candidate.remote(
                    mutated_kernel_code=mutated_kernel_code,
                    test_cases=[dict(tc) for tc in test_cases],
                    max_repeats=max_repeats,
                    max_time_ns=max_time_ns,
                )
                return outcome
            except Exception as exc:
                err = Err(
                    ScoringError(
                        reason=f"Modal call failed: {type(exc).__name__}: {exc}",
                        cause=str(exc),
                    )
                )
                return [err] * len(test_cases)

        yield score_fn
