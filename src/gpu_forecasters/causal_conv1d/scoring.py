"""Causal conv1d scoring pipeline.

Accepts a mutated-kernel source string, resolves its ``custom_kernel``
function via a synthetic-module shim, runs the correctness oracle, then
times both candidate and reference in the adaptive cuda.Event loop.
Returns raw nanoseconds — the provider layer computes speedup.

Near-duplicate of ``arid_badger.trimul.scoring``. The candidate
resolution shim and the adaptive timing loop are kernel-agnostic and
will be lifted in the gh070-A task #3 extraction; the only kernel-
specific bits are:
- the ``_TASK_SHIM`` payload (TypeVar bounds reflect this kernel's
  ``input_t = (Tensor, Tensor, Tensor)``);
- the ``_UTILS_SHIM`` re-exports (``DeterministicContext`` rather than
  TriMul's ``DisableCuDNNTF32``);
- the ``score`` body's import of ``generate_input``/``ref_kernel``/
  ``check_implementation`` from the kernel-specific ``reference``
  module, plus the kernel-specific ``test_args`` TypedDict shape.
"""

from __future__ import annotations

import importlib
import math
import os
import sys
import tempfile
import time
import traceback
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Callable

import torch

from arid_badger.causal_conv1d.cases import CausalConv1dTestArgs
from arid_badger.causal_conv1d.comparison import set_seed
from arid_badger.causal_conv1d.core import CausalConv1dExecResult, Stats
from arid_badger.causal_conv1d.reference import (
    check_implementation,
    generate_input,
    ref_kernel,
)
from arid_badger.kernelbench.isolated_scoring import ScoringError
from arid_badger.typing_utils import Err, Ok, Option


# ---------------------------------------------------------------------------
# Candidate resolution (synthetic-module shim)
# ---------------------------------------------------------------------------


_TASK_SHIM = """\
# Synthetic stand-in for upstream task.py so candidate sources containing
# `from task import input_t, output_t` resolve at import time.
from typing import Tuple, TypeVar
import torch

input_t = TypeVar(
    "input_t",
    bound=Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
)
output_t = TypeVar("output_t", bound=torch.Tensor)
"""

_UTILS_SHIM = """\
# Synthetic stand-in for upstream utils.py — re-exports just the helpers
# that candidate sources written against the helion task layout reach for
# (notably DeterministicContext, used by the upstream reference.py).
from arid_badger.causal_conv1d.comparison import (
    DeterministicContext,
    make_match_reference,
    match_reference,
    set_seed,
    verbose_allclose,
)
"""


class CandidateResolutionError(Exception):
    """Candidate source could not be turned into a usable ``custom_kernel``."""


@contextmanager
def _loaded_candidate(
    source: str,
) -> Generator[Callable[..., Any], None, None]:
    """Write ``source`` to a tmp ``submission.py`` and import it.

    Drops ``task.py`` and ``utils.py`` shims into the same tmpdir so
    candidates that mirror the upstream popcorn template resolve without
    modification. The tmpdir is prepended to ``sys.path`` and removed on
    exit; the ``submission``/``task``/``utils`` modules are evicted from
    ``sys.modules`` so a subsequent call gets a fresh import.
    """
    tmpdir = tempfile.mkdtemp(prefix="causal-conv1d-candidate-")
    try:
        with open(os.path.join(tmpdir, "submission.py"), "w") as f:
            f.write(source)
        with open(os.path.join(tmpdir, "task.py"), "w") as f:
            f.write(_TASK_SHIM)
        with open(os.path.join(tmpdir, "utils.py"), "w") as f:
            f.write(_UTILS_SHIM)

        sys.path.insert(0, tmpdir)
        for mod_name in ("submission", "task", "utils"):
            sys.modules.pop(mod_name, None)
        try:
            try:
                module = importlib.import_module("submission")
            except SyntaxError as exc:
                raise CandidateResolutionError(
                    f"Candidate source has a syntax error: {exc}"
                ) from exc
            except CandidateResolutionError:
                raise
            except Exception as exc:
                raise CandidateResolutionError(
                    f"Candidate module import failed: {type(exc).__name__}: {exc}"
                ) from exc

            custom_kernel = getattr(module, "custom_kernel", None)
            if custom_kernel is None:
                raise CandidateResolutionError(
                    "Candidate module does not define `custom_kernel`."
                )
            if not callable(custom_kernel):
                raise CandidateResolutionError(
                    "Candidate module's `custom_kernel` is not callable."
                )

            yield custom_kernel
        finally:
            for mod_name in ("submission", "task", "utils"):
                sys.modules.pop(mod_name, None)
            if tmpdir in sys.path:
                sys.path.remove(tmpdir)
    finally:
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Cloning helper (kernel-agnostic; will be lifted in #3)
# ---------------------------------------------------------------------------


def _clone_data(data: Any) -> Any:
    if isinstance(data, tuple):
        return tuple(_clone_data(x) for x in data)
    if isinstance(data, list):
        return [_clone_data(x) for x in data]
    if isinstance(data, dict):
        return {k: _clone_data(v) for k, v in data.items()}
    if isinstance(data, torch.Tensor):
        return data.clone()
    return data


def _calculate_stats(durations: list[float]) -> Stats:
    runs = len(durations)
    total = sum(durations)
    best = min(durations)
    worst = max(durations)
    avg = total / runs
    variance = sum((x - avg) ** 2 for x in durations)
    std = math.sqrt(variance / (runs - 1)) if runs > 1 else 0.0
    err = std / math.sqrt(runs) if runs > 0 else 0.0
    return Stats(runs=runs, mean=avg, std=std, err=err, best=float(best), worst=float(worst))


# ---------------------------------------------------------------------------
# Adaptive timing loop (kernel-agnostic; will be lifted in #3)
# ---------------------------------------------------------------------------


def _adaptive_time_ns(
    fn: Callable[[Any], torch.Tensor],
    data: Any,
    *,
    max_repeats: int,
    max_time_ns: float,
) -> Stats:
    """Time ``fn(data)`` with an adaptive cuda.Event loop.

    Stops when relative error of the mean drops below 0.1%, total
    measured time exceeds ``max_time_ns``, or we've done ``max_repeats``
    iterations (floor of 3 per upstream).
    """
    durations: list[float] = []
    bm_start = time.perf_counter_ns()
    for i in range(max_repeats):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        output = fn(data)
        end_event.record()
        torch.cuda.synchronize()
        duration_ns = start_event.elapsed_time(end_event) * 1e6  # ms -> ns
        del output
        durations.append(duration_ns)

        if i > 1:
            total_bm_duration = time.perf_counter_ns() - bm_start
            stats = _calculate_stats(durations)
            if (
                stats.err / stats.mean < 0.001
                or stats.mean * stats.runs > max_time_ns
                or total_bm_duration > 120e9
            ):
                break

    return _calculate_stats(durations)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def score(
    mutated_kernel_code: str,
    test_args: CausalConv1dTestArgs,
    *,
    max_repeats: int = 100,
    max_time_ns: float = 10e9,
) -> Option[CausalConv1dExecResult, ScoringError]:
    """Score one candidate against one test case.

    Returns ``Ok(CausalConv1dExecResult)`` on successful execution of the
    scoring pipeline — even if the candidate was wrong. ``Err`` is
    reserved for scoring-infrastructure failures (bad GPU state, etc.).
    """
    try:
        set_seed(42)

        with _loaded_candidate(mutated_kernel_code) as custom_kernel:
            data = generate_input(**test_args)
            check_copy = _clone_data(data)

            # Obligatory correctness pass.
            try:
                output = custom_kernel(_clone_data(data))
                torch.cuda.synchronize()
            except Exception as exc:
                return Ok(
                    CausalConv1dExecResult(
                        correct=False,
                        runtime_ns=0.0,
                        ref_runtime_ns=0.0,
                        failure_kind="runtime_error",
                        runtime_error_name=type(exc).__name__,
                        runtime_error=str(exc),
                        traceback=traceback.format_exc(),
                    )
                )

            good, message = check_implementation(check_copy, output)
            del output
            if not good:
                return Ok(
                    CausalConv1dExecResult(
                        correct=False,
                        runtime_ns=0.0,
                        ref_runtime_ns=0.0,
                        failure_kind="incorrect",
                        error_message=message,
                    )
                )

            candidate_stats = _adaptive_time_ns(
                lambda d: custom_kernel(_clone_data(d)),
                data,
                max_repeats=max_repeats,
                max_time_ns=max_time_ns,
            )
            ref_stats = _adaptive_time_ns(
                lambda d: ref_kernel(_clone_data(d)),
                data,
                max_repeats=max_repeats,
                max_time_ns=max_time_ns,
            )

            return Ok(
                CausalConv1dExecResult(
                    correct=True,
                    runtime_ns=candidate_stats.mean,
                    ref_runtime_ns=ref_stats.mean,
                )
            )

    except CandidateResolutionError as exc:
        return Ok(
            CausalConv1dExecResult(
                correct=False,
                runtime_ns=0.0,
                ref_runtime_ns=0.0,
                failure_kind="compile_failed",
                compilation_error=str(exc),
            )
        )
    except Exception as exc:
        return Err(
            ScoringError(
                reason=f"Causal conv1d scoring harness crashed: {type(exc).__name__}: {exc}",
                cause=traceback.format_exc(),
            )
        )
