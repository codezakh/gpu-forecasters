"""Run kernel scoring in an isolated subprocess to contain CUDA faults.

When a faulty CUDA kernel triggers ``cudaErrorIllegalAddress``, the CUDA
context for the entire Python process becomes permanently poisoned.  By
running each scoring attempt in a *separate* subprocess (using the ``spawn``
start method so the child gets a fresh interpreter without inherited CUDA
state), we ensure that such faults are contained and do not affect subsequent
evaluations.
"""

import multiprocessing
import traceback
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from arid_badger.typing_utils import Option, Ok, Err

from arid_badger.kernelbench.scoring import _score_kernel_impl
from kernelbench.eval import KernelExecResult

# Default timeout for subprocess scoring (seconds).
DEFAULT_SUBPROCESS_TIMEOUT_SECONDS = 300


class ScoringError(BaseModel):
    """Structured error from a subprocess scoring attempt.

    Uses a Pydantic model rather than a raw Exception so it serialises
    reliably across the subprocess (pickle) boundary.
    """

    reason: str
    exit_code: Optional[int] = None
    cause: Optional[str] = None


def _scoring_worker(
    queue: multiprocessing.Queue,
    mutated_kernel_code: str,
    reference_kernel_code: str,
    backend: str,
    precision: str,
    num_correct_trials: int,
    num_perf_trials: int,
    build_dir: Optional[Path],
) -> None:
    """Entry point executed inside the spawned subprocess.

    Runs the actual scoring logic and puts the result (or error) onto
    *queue* so the parent process can retrieve it.
    """
    try:
        result = _score_kernel_impl(
            mutated_kernel_code=mutated_kernel_code,
            reference_kernel_code=reference_kernel_code,
            backend=backend,
            precision=precision,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials,
            build_dir=build_dir,
        )
        if result is None:
            queue.put(Err(ScoringError(reason="Scoring returned None")))
        else:
            queue.put(Ok(result))
    except Exception as exc:
        queue.put(
            Err(
                ScoringError(
                    reason=str(exc),
                    cause=traceback.format_exc(),
                )
            )
        )


def run_scoring_in_subprocess(
    mutated_kernel_code: str,
    reference_kernel_code: str,
    backend: str = "cuda",
    precision: str = "fp32",
    num_correct_trials: int = 5,
    num_perf_trials: int = 100,
    build_dir: Optional[Path] = None,
    timeout_seconds: int = DEFAULT_SUBPROCESS_TIMEOUT_SECONDS,
) -> Option[KernelExecResult, ScoringError]:
    """Score a kernel in an isolated subprocess.

    Uses ``multiprocessing.get_context('spawn')`` so the child process starts
    a fresh Python interpreter, avoiding inheritance of any poisoned CUDA
    context from the parent.

    Returns ``Ok(KernelExecResult)`` on success or ``Err(ScoringError)`` on
    any failure (timeout, crash, scoring error).
    """
    ctx = multiprocessing.get_context("spawn")
    queue: multiprocessing.Queue[Option[KernelExecResult, ScoringError]] = ctx.Queue()

    process = ctx.Process(
        target=_scoring_worker,
        args=(
            queue,
            mutated_kernel_code,
            reference_kernel_code,
            backend,
            precision,
            num_correct_trials,
            num_perf_trials,
            build_dir,
        ),
    )
    process.start()
    process.join(timeout=timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        return Err(ScoringError(reason=f"Timed out after {timeout_seconds}s"))

    if process.exitcode != 0:
        cause = _drain_error_from_queue(queue)
        return Err(
            ScoringError(
                reason="Subprocess crashed",
                exit_code=process.exitcode,
                cause=cause,
            )
        )

    if queue.empty():
        return Err(ScoringError(reason="Subprocess produced no result"))

    return queue.get_nowait()


def _drain_error_from_queue(
    queue: multiprocessing.Queue,
) -> Optional[str]:
    """Try to retrieve a stringified error cause from the queue, if any."""
    if queue.empty():
        return None
    outcome = queue.get_nowait()
    if isinstance(outcome, Err):
        error: ScoringError = outcome.unwrap_err()
        return error.cause or error.reason
    return None
