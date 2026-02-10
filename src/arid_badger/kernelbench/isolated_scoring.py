"""Run kernel scoring in an isolated subprocess to contain CUDA faults.

When a faulty CUDA kernel triggers ``cudaErrorIllegalAddress``, the CUDA
context for the entire Python process becomes permanently poisoned.  By
running each scoring attempt in a *separate* subprocess (using the ``spawn``
start method so the child gets a fresh interpreter without inherited CUDA
state), we ensure that such faults are contained and do not affect subsequent
evaluations.
"""

import multiprocessing
import multiprocessing.queues
from pathlib import Path
from typing import Optional

from arid_badger.kernelbench.core import KernelScoringResult
from arid_badger.typing_utils import Option, Ok, Err, is_ok

# Default timeout for subprocess scoring (seconds).
DEFAULT_SUBPROCESS_TIMEOUT_SECONDS = 300


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

    Runs the actual scoring logic and puts the result (or exception) onto
    *queue* so the parent process can retrieve it.
    """
    try:
        from arid_badger.kernelbench.scoring import _score_kernel_impl

        result = _score_kernel_impl(
            mutated_kernel_code=mutated_kernel_code,
            reference_kernel_code=reference_kernel_code,
            backend=backend,
            precision=precision,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials,
            build_dir=build_dir,
        )
        queue.put(Ok(result))
    except Exception as exc:
        queue.put(Err(exc))


def run_scoring_in_subprocess(
    mutated_kernel_code: str,
    reference_kernel_code: str,
    backend: str = "cuda",
    precision: str = "fp32",
    num_correct_trials: int = 5,
    num_perf_trials: int = 100,
    build_dir: Optional[Path] = None,
    timeout_seconds: int = DEFAULT_SUBPROCESS_TIMEOUT_SECONDS,
) -> KernelScoringResult:
    """Score a kernel in an isolated subprocess.

    Uses ``multiprocessing.get_context('spawn')`` so the child process starts
    a fresh Python interpreter, avoiding inheritance of any poisoned CUDA
    context from the parent.

    Raises:
        RuntimeError: If the subprocess crashes (e.g. CUDA fault) or times out.
    """
    ctx = multiprocessing.get_context("spawn")
    queue: multiprocessing.Queue[Option[KernelScoringResult, Exception]] = ctx.Queue()

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
        raise RuntimeError(
            f"Kernel scoring subprocess timed out after {timeout_seconds}s"
        )

    if process.exitcode != 0:
        # The subprocess crashed (e.g. a CUDA fault killed it).  Try to
        # retrieve any exception that was placed on the queue before the
        # crash, otherwise report the exit code.
        if not queue.empty():
            outcome: Option[KernelScoringResult, Exception] = queue.get_nowait()
            if outcome.is_err():
                raise RuntimeError(
                    f"Kernel scoring subprocess failed (exit code {process.exitcode})"
                ) from outcome.unwrap_err()
        raise RuntimeError(
            f"Kernel scoring subprocess crashed with exit code {process.exitcode}"
        )

    # Subprocess exited cleanly – retrieve the result from the queue.
    if queue.empty():
        raise RuntimeError(
            "Kernel scoring subprocess exited successfully but produced no result"
        )

    outcome = queue.get_nowait()
    if is_ok(outcome):
        return outcome.unwrap()
    raise outcome.unwrap_err()
