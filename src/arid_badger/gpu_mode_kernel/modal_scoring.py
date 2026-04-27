"""Generic Modal scoring harness.

Each ``KernelPack`` declares its own Modal app + benchmarker cls in its
own module (so Modal sees a stable class at a stable module path —
needed for cloudpickle serialization). The cls body is one line:
delegate to ``run_evaluate_candidate(pack=…, …)``, which holds the
fan-out-over-cases loop, GPU readiness wait, and per-case error
handling that used to live duplicated in ``trimul/modal_scoring.py``
and ``causal_conv1d/modal_scoring.py``.

The shared image (``arid_badger.kernelbench.modal_image.image``) and
default cls decorator settings live here.

Usage from a pack module:

    import modal
    from arid_badger.gpu_mode_kernel.modal_scoring import (
        DEFAULT_CLS_KWARGS, run_evaluate_candidate,
    )
    from arid_badger.kernelbench.modal_image import image
    from .my_pack import MY_PACK

    app = modal.App(MY_PACK.modal_app_name)

    @app.cls(image=image, **DEFAULT_CLS_KWARGS)
    class ModalMyKernelBenchmarker:
        @modal.method()
        def evaluate_candidate(self, mutated_kernel_code, test_cases,
                               max_repeats=100, max_time_ns=10e9):
            return run_evaluate_candidate(
                pack=MY_PACK, mutated_kernel_code=mutated_kernel_code,
                test_cases=test_cases, max_repeats=max_repeats,
                max_time_ns=max_time_ns,
            )
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Generic, cast

import modal

from arid_badger.gpu_mode_kernel.core import KernelExecResult
from arid_badger.gpu_mode_kernel.kernel_pack import (
    CaseSpeedupT,
    KernelPack,
    TestArgsT,
)
from arid_badger.kernelbench.isolated_scoring import ScoringError
from arid_badger.typing_utils import Err, Option


@dataclass(frozen=True)
class PackedModalRuntime(Generic[TestArgsT, CaseSpeedupT]):
    """Bundle of ``(pack, modal app, benchmarker cls)`` for one kernel.

    Each pack module exports one of these as its public runtime entry
    point. The v2 provider consumes one to construct, rather than
    three correlated args.

    Why bundled rather than stored directly on ``KernelPack``: the
    pack lives in a frozen value module that intentionally has no
    Modal dependency (it's portable / testable without the Modal
    SDK). The Modal cls and app are infrastructure artifacts; they
    materialise alongside the pack but live one layer out.
    """

    pack: KernelPack[TestArgsT, CaseSpeedupT]
    app: modal.App
    benchmarker_cls: type


DEFAULT_GPU = "A100-80GB"

DEFAULT_CLS_KWARGS: dict[str, Any] = {
    "gpu": DEFAULT_GPU,
    "timeout": 1200,
    "max_containers": 40,
    "single_use_containers": True,
    "scaledown_window": 2,
    "retries": modal.Retries(
        max_retries=3,
        backoff_coefficient=2.0,
        initial_delay=1.0,
    ),
}


_GPU_WAIT_TIMEOUT_S = 30
_GPU_WAIT_INITIAL_DELAY_S = 0.5
_GPU_WAIT_MAX_DELAY_S = 8.0


def _wait_for_gpu() -> None:
    """Block until ``torch.cuda`` is ready or timeout.

    Containers occasionally need a moment after spinup before the GPU
    is reachable; the per-kernel modal_scoring modules used to embed
    this loop verbatim.
    """
    import torch

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


def run_evaluate_candidate(
    *,
    pack: KernelPack[TestArgsT, CaseSpeedupT],
    mutated_kernel_code: str,
    test_cases: list[dict[str, object]],
    max_repeats: int = 100,
    max_time_ns: float = 10e9,
) -> list[Option[KernelExecResult, ScoringError]]:
    """Score one candidate against all cases on the current container.

    Generic body for the per-pack ``evaluate_candidate`` method. Each
    pack's Modal cls calls this. The function is import-time-safe so
    Modal can serialize the cls without pulling torch / cuda at module
    load.

    Per-case error handling matches the existing per-kernel modal
    scoring: a CudaError stops accepting further inputs on this
    container (poisoned GPU) but still returns whatever results were
    collected; other exceptions become per-case ``Err`` and
    iteration continues.
    """
    import torch

    from arid_badger.gpu_mode_kernel.scoring import score_one_case

    _wait_for_gpu()

    results: list[Option[KernelExecResult, ScoringError]] = []
    for test_case in test_cases:
        try:
            result = score_one_case(
                pack=pack,
                mutated_kernel_code=mutated_kernel_code,
                test_args=cast(TestArgsT, cast(object, test_case)),
                max_repeats=max_repeats,
                max_time_ns=max_time_ns,
            )
        except Exception as exc:
            exc_type = type(exc).__name__
            results.append(
                Err(
                    ScoringError(
                        reason=f"{pack.name} Modal evaluate raised {exc_type}: {exc}",
                        cause=str(exc),
                    )
                )
            )
            # Match legacy substring semantics (``"CudaError" in
            # exc_type``). Both ``AcceleratorError`` (from
            # ``modal.exception``, not importable here) and torch's
            # cuda errors share class-name conventions; isinstance on
            # torch.cuda.CudaError would catch ``OutOfMemoryError`` as
            # well, which the legacy did NOT bail on. Preserving
            # semantics until we're ready to make that a deliberate
            # change.
            is_cuda_failure = (
                "CudaError" in exc_type or "AcceleratorError" in exc_type
            )
            if is_cuda_failure:
                modal.experimental.stop_fetching_inputs()
                break
            continue

        torch.cuda.empty_cache()
        results.append(result)

    return results
