"""Modal-backed TriMul evaluator mapping raw exec results to typed outcomes.

Wraps ``gpu_forecasters.trimul.modal_scoring.modal_trimul_scoring_session``
as a module-level singleton (same atexit pattern as v1's GpuMode env)
and maps a list of ``Option[TriMulExecResult, ScoringError]`` onto a
single ``TriMulRLOutcome`` per v2's classification rules:

- any ``Err`` (Modal / harness crash) → ``InfrastructureFailureFeedback``
- any passing-execresult but with ``failure_kind != "none"`` →
  the corresponding per-case failure variant (compile / runtime / incorrect)
- all cases ``correct`` → ``SuccessFeedback`` with per-case speedups +
  geomean.

The v1 guards (``@triton.jit`` required, ``identity`` forbidden) surface
as ``CompileFailedFeedback`` with a synthetic message — this keeps the
downstream reward / feedback pipeline on the well-tested variants rather
than introducing yet another outcome variant for the guard.
"""

from __future__ import annotations

import asyncio
import atexit
import math
import threading
import time
from typing import Any

from loguru import logger

from gpu_forecasters.trimul.cases import TriMulTestArgs
from gpu_forecasters.trimul.core import (
    CaseSpeedup,
    CompileFailedFeedback,
    InfrastructureFailureFeedback,
    SuccessFeedback,
    TriMulExecResult,
    failure_feedback_from_exec_result,
)
from gpu_forecasters.trimul.modal_scoring import (
    TriMulScoringFn,
    modal_trimul_scoring_session,
)
from gpu_forecasters.ttt_discover.v2.domain.outcome import TriMulRLOutcome
from gpu_forecasters.ttt_discover.v2.interfaces.evaluator import KernelEvaluator
from gpu_forecasters.typing_utils import Option, implements, is_ok


_session_lock = threading.Lock()
_session_cm: Any = None
_score_fn: TriMulScoringFn | None = None


def _get_score_fn(gpu: str) -> TriMulScoringFn:
    global _session_cm, _score_fn
    if _score_fn is not None:
        return _score_fn
    with _session_lock:
        if _score_fn is not None:
            return _score_fn
        cm = modal_trimul_scoring_session(gpu=gpu)
        score_fn = cm.__enter__()
        _session_cm = cm
        _score_fn = score_fn

        def _close() -> None:
            try:
                _ = cm.__exit__(None, None, None)
            except Exception:
                pass

        _ = atexit.register(_close)
        return score_fn


def map_outcomes_to_rl_outcome(
    outcomes: list[Option[TriMulExecResult, Any]],
    test_cases: list[TriMulTestArgs],
) -> TriMulRLOutcome:
    """Map per-case ``Option[TriMulExecResult]`` → one ``TriMulRLOutcome``.

    - any ``Err`` → ``InfrastructureFailureFeedback``
    - first case with ``failure_kind != 'none'`` → that failure variant
    - all-correct → ``SuccessFeedback``
    """
    if not outcomes:
        return InfrastructureFailureFeedback(reason="Empty outcomes list")
    exec_results: list[TriMulExecResult] = []
    for i, outcome in enumerate(outcomes):
        if not is_ok(outcome):
            err = outcome.unwrap_err()
            return InfrastructureFailureFeedback(
                reason=f"Case {i} infra failure: {err.reason}"
            )
        exec_results.append(outcome.unwrap())

    # Any non-correct case short-circuits to the matching failure variant,
    # mirroring the hill-climbing TriMulObservation feedback semantics.
    for result in exec_results:
        if result.failure_kind != "none":
            return failure_feedback_from_exec_result(result)

    # All-correct path — assemble per-case speedups + geomean.
    per_case: list[CaseSpeedup] = []
    log_runtime_sum = 0.0
    for result, tc in zip(exec_results, test_cases, strict=True):
        if result.runtime_ns <= 0 or result.ref_runtime_ns <= 0:
            return InfrastructureFailureFeedback(
                reason=f"Non-positive runtime: runtime_ns={result.runtime_ns}, "
                f"ref_runtime_ns={result.ref_runtime_ns}"
            )
        speedup = result.ref_runtime_ns / result.runtime_ns
        per_case.append(
            CaseSpeedup(
                seqlen=tc["seqlen"],
                bs=tc["bs"],
                dim=tc["dim"],
                hiddendim=tc["hiddendim"],
                nomask=tc["nomask"],
                distribution=tc["distribution"],
                speedup=speedup,
                runtime_ns=result.runtime_ns,
                ref_runtime_ns=result.ref_runtime_ns,
            )
        )
        log_runtime_sum += math.log(result.runtime_ns)

    geomean_runtime_ns = math.exp(log_runtime_sum / len(exec_results))
    log_ref_sum = sum(math.log(r.ref_runtime_ns) for r in exec_results)
    geomean_ref_ns = math.exp(log_ref_sum / len(exec_results))
    aggregated_speedup = geomean_ref_ns / geomean_runtime_ns
    return SuccessFeedback(
        aggregated_speedup=aggregated_speedup,
        aggregation_method="geomean",
        per_case_speedups=per_case,
    )


class ModalTriMulEvaluator:
    _gpu_name: str
    _test_cases: list[TriMulTestArgs]
    _require_triton_jit: bool
    _forbid_identity: bool

    def __init__(
        self,
        *,
        gpu_name: str,
        test_cases: list[TriMulTestArgs],
        require_triton_jit: bool = True,
        forbid_identity: bool = True,
    ) -> None:
        self._gpu_name = gpu_name
        self._test_cases = list(test_cases)
        self._require_triton_jit = require_triton_jit
        self._forbid_identity = forbid_identity

    async def evaluate(self, code: str) -> TriMulRLOutcome:
        if self._require_triton_jit and "@triton.jit" not in code:
            return CompileFailedFeedback(
                compilation_error="missing @triton.jit: kernel must define a triton.jit kernel"
            )
        if self._forbid_identity and "identity" in code:
            return CompileFailedFeedback(
                compilation_error="identity kernel is not allowed",
            )

        loop = asyncio.get_running_loop()
        start = time.time()
        try:
            score_fn = _get_score_fn(self._gpu_name)
            outcomes = await loop.run_in_executor(
                None, score_fn, code, list(self._test_cases)
            )
        except Exception as exc:
            elapsed = time.time() - start
            logger.warning(
                "ModalTriMulEvaluator: score_fn raised after {elapsed:.1f}s: {exc}",
                elapsed=elapsed,
                exc=exc,
            )
            return InfrastructureFailureFeedback(
                reason=f"score_fn raised: {type(exc).__name__}: {exc}"
            )
        return map_outcomes_to_rl_outcome(outcomes, self._test_cases)


_ = implements(KernelEvaluator)(ModalTriMulEvaluator)
