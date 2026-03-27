"""Modal-based kernel scoring.

Provides a low-level scoring interface that compiles and evaluates kernels
on Modal remote GPU containers rather than locally. Mirrors the signature of
`run_scoring_in_subprocess` so callers can swap backends transparently.

Usage (session — preferred for multiple evaluations):

    from arid_badger.kernelbench.modal_scoring import modal_scoring_session

    with modal_scoring_session(gpu="T4") as score:
        result = score(mutated_kernel_code, reference_kernel_code)
        if is_ok(result):
            print(result.unwrap().runtime)

Usage (one-shot convenience):

    from arid_badger.kernelbench.modal_scoring import run_scoring_on_modal

    result = run_scoring_on_modal(mutated_kernel_code, reference_kernel_code, gpu="T4")

Note: `modal_scoring_session` opens a persistent Modal app session for the
duration of the `with` block. Keep the block open for the full duration of
your evaluation loop — opening per-call incurs Modal connection overhead and
loses warm container reuse.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Callable, Optional

import modal

from arid_badger.kernelbench.isolated_scoring import ScoringError
from arid_badger.typing_utils import Ok, Err, Option
from kernelbench.eval import KernelExecResult

from .modal_image import app, image, GPU_ARCH_MAPPING

# ---------------------------------------------------------------------------
# Remote evaluator class (runs inside Modal container)
# ---------------------------------------------------------------------------

_DEFAULT_GPU = "L4"
_GPU_WAIT_TIMEOUT_S = 30
_GPU_WAIT_INITIAL_DELAY_S = 0.5
_GPU_WAIT_MAX_DELAY_S = 8.0


@app.cls(
    image=image,
    gpu=_DEFAULT_GPU,
    retries=modal.Retries(
        max_retries=3,
        backoff_coefficient=2.0,
        initial_delay=1.0,
    ),
)
class ModalKernelEvaluator:
    """Runs kernel compilation and evaluation inside a Modal GPU container.

    This class is internal to this module. Use `modal_scoring_session` or
    `run_scoring_on_modal` from experiment code.
    """

    @modal.method()
    def evaluate(
        self,
        mutated_kernel_code: str,
        reference_kernel_code: str,
        gpu_arch: list[str],
        backend: str = "cuda",
        precision: str = "fp32",
        num_correct_trials: int = 5,
        num_perf_trials: int = 100,
    ) -> KernelExecResult:
        import torch
        from kernelbench.eval import (
            eval_kernel_against_ref,
            get_torch_dtype_from_string,
        )
        from kernelbench.utils import set_gpu_arch

        # Wait for the GPU to become available (containers occasionally start
        # before the GPU is fully attached).
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

        set_gpu_arch(gpu_arch)

        try:
            result = eval_kernel_against_ref(
                original_model_src=reference_kernel_code,
                custom_model_src=mutated_kernel_code,
                num_correct_trials=num_correct_trials,
                num_perf_trials=num_perf_trials,
                measure_performance=True,
                timing_method="cuda_event",
                verbose=False,
                device=torch.device("cuda:0"),
                backend=backend,
                precision=get_torch_dtype_from_string(precision),
            )
        except (torch.cuda.CudaError, Exception) as exc:
            # Detect GPU corruption: retire this container so subsequent
            # evaluations get a fresh one.
            exc_type = type(exc).__name__
            if "CudaError" in exc_type or "AcceleratorError" in exc_type:
                modal.experimental.stop_fetching_inputs()
            return KernelExecResult(
                compiled=False,
                metadata={
                    "compilation_error_name": exc_type,
                    "compilation_error": str(exc),
                },
            )

        torch.cuda.empty_cache()
        return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

ScoringFn = Callable[..., Option[KernelExecResult, ScoringError]]


@contextmanager
def modal_scoring_session(
    gpu: str = _DEFAULT_GPU,
    backend: str = "cuda",
    precision: str = "fp32",
    num_correct_trials: int = 5,
    num_perf_trials: int = 100,
) -> Generator[ScoringFn, None, None]:
    """Context manager that opens a Modal app session for kernel scoring.

    Yields a scoring callable with the same signature as
    `run_scoring_in_subprocess`. Keep the session open for the full duration
    of your evaluation loop — closing and reopening per call incurs overhead
    and loses warm container reuse.

    Args:
        gpu: Modal GPU type (e.g. "T4", "A10G", "H100").
        backend: Kernel backend ("cuda", "triton", etc.).
        precision: Floating-point precision ("fp32", "fp16", "bf16").
        num_correct_trials: Number of correctness trials.
        num_perf_trials: Number of performance timing trials.

    Yields:
        A callable ``score(mutated_kernel_code, reference_kernel_code, ...)``
        that returns ``Option[KernelExecResult, ScoringError]``.
    """
    gpu_arch = GPU_ARCH_MAPPING.get(gpu, ["Ampere"])
    evaluator = ModalKernelEvaluator.with_options(  # ty: ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue]
        gpu=gpu
    )

    with app.run():

        def score(
            mutated_kernel_code: str,
            reference_kernel_code: str,
            *,
            backend: str = backend,
            precision: str = precision,
            num_correct_trials: int = num_correct_trials,
            num_perf_trials: int = num_perf_trials,
            build_dir: Optional[Path] = None,  # accepted for signature compat, unused
        ) -> Option[KernelExecResult, ScoringError]:
            try:
                exec_result: KernelExecResult = evaluator().evaluate.remote(
                    mutated_kernel_code=mutated_kernel_code,
                    reference_kernel_code=reference_kernel_code,
                    gpu_arch=gpu_arch,
                    backend=backend,
                    precision=precision,
                    num_correct_trials=num_correct_trials,
                    num_perf_trials=num_perf_trials,
                )
                return Ok(exec_result)
            except Exception as exc:
                return Err(
                    ScoringError(
                        reason=f"Modal call failed: {type(exc).__name__}: {exc}",
                        cause=str(exc),
                    )
                )

        yield score


def run_scoring_on_modal(
    mutated_kernel_code: str,
    reference_kernel_code: str,
    *,
    gpu: str = _DEFAULT_GPU,
    backend: str = "cuda",
    precision: str = "fp32",
    num_correct_trials: int = 5,
    num_perf_trials: int = 100,
    timeout_seconds: int = 300,  # accepted for signature compat, unused
) -> Option[KernelExecResult, ScoringError]:
    """Score a kernel on Modal in a one-shot session.

    Opens and closes a Modal app session for a single evaluation. For
    evaluating multiple kernels, prefer `modal_scoring_session` to amortise
    session overhead across calls.
    """
    with modal_scoring_session(
        gpu=gpu,
        backend=backend,
        precision=precision,
        num_correct_trials=num_correct_trials,
        num_perf_trials=num_perf_trials,
    ) as score:
        return score(mutated_kernel_code, reference_kernel_code)
