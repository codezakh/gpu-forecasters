"""Generic per-case scoring pipeline for gpu-mode-style kernels.

Generalizes ``gpu_forecasters.trimul.scoring`` and
``gpu_forecasters.causal_conv1d.scoring``:

- The ``_TASK_SHIM`` and ``_UTILS_SHIM`` constants migrate onto the
  ``KernelPack`` (their ``TypeVar`` bounds and re-exported symbol names
  are kernel-specific). The candidate-loading machinery itself is
  kernel-agnostic.
- ``_clone_data``, ``_calculate_stats``, and ``_adaptive_time_ns``
  lift verbatim — they were byte-identical across both per-kernel
  scoring modules.
- ``score_one_case(pack, code, test_args)`` replaces the per-kernel
  ``score(...)`` entry points. It runs on the Modal container side
  via the per-pack ``evaluate_candidate`` method.

The candidate-loading shim writes ``submission.py``, ``task.py``, and
``utils.py`` to a tmpdir, prepends the tmpdir to ``sys.path``, and
imports ``submission``. The ``task.py`` and ``utils.py`` shims come
from the pack so candidates that look like upstream
gpu-mode/reference-kernels ``submission.py`` templates resolve their
imports without modification.
"""

from __future__ import annotations

import importlib
import math
import os
import shutil
import sys
import tempfile
import time
import traceback
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Callable

import torch

from gpu_forecasters.gpu_mode_kernel.comparison import set_seed
from gpu_forecasters.gpu_mode_kernel.core import KernelExecResult, Stats
from gpu_forecasters.gpu_mode_kernel.kernel_pack import (
    CaseSpeedupT,
    KernelPack,
    TestArgsT,
)
from gpu_forecasters.kernelbench.isolated_scoring import ScoringError
from gpu_forecasters.typing_utils import Err, Ok, Option


# ---------------------------------------------------------------------------
# Candidate resolution (synthetic-module shim) — generic over the pack.
# ---------------------------------------------------------------------------


class CandidateResolutionError(Exception):
    """Candidate source could not be turned into a usable ``custom_kernel``."""


# Canonical ``task.py`` content. Upstream gpu-mode/reference-kernels
# submissions do ``from task import input_t, output_t`` to type-hint
# their ``custom_kernel`` signature; the bounds are documentation
# only (Python erases TypeVar bounds at runtime), so a bound-free
# alias works for every kernel.
_TASK_SHIM_TEMPLATE = """\
# Synthetic task.py — provides ``input_t``/``output_t`` TypeVars for
# candidate sources that mirror upstream popcorn submission templates.
from typing import TypeVar
input_t = TypeVar("input_t")
output_t = TypeVar("output_t")
"""

# Header re-exports the kernel-agnostic helpers that every pack
# inherits from ``gpu_forecasters.gpu_mode_kernel.comparison``. The
# determinism context manager is appended below if the pack carries
# one.
_UTILS_SHIM_HEADER = """\
# Synthetic utils.py — re-exports helpers that upstream popcorn
# submission templates expect at ``from utils import …``.
from gpu_forecasters.gpu_mode_kernel.comparison import (
    make_match_reference,
    match_reference,
    set_seed,
    verbose_allclose,
)
"""


def _build_utils_shim(determinism_ctx: type | None) -> str:
    if determinism_ctx is None:
        return _UTILS_SHIM_HEADER
    return (
        _UTILS_SHIM_HEADER
        + f"from {determinism_ctx.__module__} import {determinism_ctx.__name__}\n"
    )


@contextmanager
def _loaded_candidate(
    source: str,
    *,
    determinism_ctx: type | None,
    tmpdir_prefix: str,
) -> Generator[Callable[..., Any], None, None]:
    """Write ``source`` to a tmp ``submission.py`` and import it.

    Drops canonical ``task.py`` and ``utils.py`` shims into the same
    tmpdir so candidates that mirror the upstream popcorn template
    (``from task import input_t``, ``from utils import ...``) resolve
    without modification. The kernel-specific surface is just
    ``determinism_ctx`` — when set, an extra import line in
    ``utils.py`` re-exports it.

    The tmpdir is prepended to ``sys.path`` and removed on exit; the
    ``submission``/``task``/``utils`` modules are evicted from
    ``sys.modules`` so a subsequent call gets a fresh import.
    """
    utils_shim = _build_utils_shim(determinism_ctx)
    tmpdir = tempfile.mkdtemp(prefix=tmpdir_prefix)
    try:
        with open(os.path.join(tmpdir, "submission.py"), "w") as f:
            f.write(source)
        with open(os.path.join(tmpdir, "task.py"), "w") as f:
            f.write(_TASK_SHIM_TEMPLATE)
        with open(os.path.join(tmpdir, "utils.py"), "w") as f:
            f.write(utils_shim)

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
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Cloning helper (lifted verbatim from per-kernel scoring modules).
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
    return Stats(
        runs=runs, mean=avg, std=std, err=err, best=float(best), worst=float(worst)
    )


# ---------------------------------------------------------------------------
# Adaptive timing loop (ported from ttt-discover eval.py).
# ---------------------------------------------------------------------------


def _adaptive_time_ns(
    fn: Callable[[Any], Any],
    data: Any,
    *,
    max_repeats: int,
    max_time_ns: float,
) -> Stats:
    """Time ``fn(data)`` with an adaptive cuda.Event loop.

    Stops when relative error of the mean drops below 0.1%, total
    measured time exceeds ``max_time_ns``, or we've done
    ``max_repeats`` iterations. Floor of 3 iterations per upstream.
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
# Public entry point — generic over the pack.
# ---------------------------------------------------------------------------


def score_one_case(
    *,
    pack: KernelPack[TestArgsT, CaseSpeedupT],
    mutated_kernel_code: str,
    test_args: TestArgsT,
    max_repeats: int = 100,
    max_time_ns: float = 10e9,
) -> Option[KernelExecResult, ScoringError]:
    """Score one candidate against one test case.

    Returns ``Ok(KernelExecResult)`` on successful execution of the
    scoring pipeline — even if the candidate was wrong. ``Err`` is
    reserved for scoring-infrastructure failures (bad GPU state,
    etc.).

    Mirrors the body of ``gpu_forecasters.trimul.scoring.score`` and
    ``gpu_forecasters.causal_conv1d.scoring.score`` exactly; the only
    differences are that the reference call, input generator, oracle,
    and shim contents come off the pack.
    """
    try:
        set_seed(42)

        with _loaded_candidate(
            mutated_kernel_code,
            determinism_ctx=pack.determinism_ctx,
            tmpdir_prefix=f"{pack.name}-candidate-",
        ) as custom_kernel:
            data = pack.generate_input(**test_args)  # type: ignore[arg-type]
            check_copy = _clone_data(data)

            # Obligatory correctness pass.
            try:
                output = custom_kernel(_clone_data(data))
                torch.cuda.synchronize()
            except Exception as exc:
                return Ok(
                    KernelExecResult(
                        correct=False,
                        runtime_ns=0.0,
                        ref_runtime_ns=0.0,
                        failure_kind="runtime_error",
                        runtime_error_name=type(exc).__name__,
                        runtime_error=str(exc),
                        traceback=traceback.format_exc(),
                    )
                )

            good, message = pack.check_implementation(check_copy, output)
            del output
            if not good:
                return Ok(
                    KernelExecResult(
                        correct=False,
                        runtime_ns=0.0,
                        ref_runtime_ns=0.0,
                        failure_kind="incorrect",
                        error_message=message,
                    )
                )

            # Candidate correct — time it. Fresh data on each invocation
            # in case the kernel mutates inputs in place.
            candidate_stats = _adaptive_time_ns(
                lambda d: custom_kernel(_clone_data(d)),
                data,
                max_repeats=max_repeats,
                max_time_ns=max_time_ns,
            )

            # Reference on the same data, same hardware.
            ref_stats = _adaptive_time_ns(
                lambda d: pack.ref_kernel(_clone_data(d)),
                data,
                max_repeats=max_repeats,
                max_time_ns=max_time_ns,
            )

            return Ok(
                KernelExecResult(
                    correct=True,
                    runtime_ns=candidate_stats.mean,
                    ref_runtime_ns=ref_stats.mean,
                )
            )

    except CandidateResolutionError as exc:
        return Ok(
            KernelExecResult(
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
                reason=f"{pack.name} scoring harness crashed: {type(exc).__name__}: {exc}",
                cause=traceback.format_exc(),
            )
        )
