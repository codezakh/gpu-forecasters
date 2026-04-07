"""Evaluation provider that uses the training-free LLM speedup estimator (kernel world model).

This is the canonical library version of the evaluation provider that wraps
``SpeedupEstimator`` (specifically ``LlmSpeedupEstimator``) to produce
``Evaluation[KernelRuntimeEstimate]`` objects consumable by PUCT search.

It is named "training-free" to distinguish it from future evaluation providers
that wrap learned/trained estimators.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from loguru import logger
from pydantic import BaseModel

from arid_badger.hill_climbing.domain import Evaluation
from arid_badger.invocation_sink import InvocationSink, code_sha256

from .domain import (
    HardwareContext,
    KernelImplementation,
    KernelRuntimeEstimate,
    KernelRuntimeQuery,
    KernelTaskInfo,
    LikertConfidence,
    SpeedupBin,
    SpeedupEstimator,
)
from .llm_estimator import EstimatorParseError


class KwmEvaluationRecord(BaseModel, frozen=True):
    """Invocation record for a single training-free KWM (LLM-based) kernel evaluation."""

    kind: Literal["kwm_evaluation"] = "kwm_evaluation"
    code_sha256: str
    model_slug: str
    input_tokens: int
    output_tokens: int
    predicted_bin: int
    timestamp_utc: str


def _failure_estimate(reasoning: str) -> KernelRuntimeEstimate:
    """Create a FAILURE estimate with a given reasoning string."""
    return KernelRuntimeEstimate(
        predicted_bin=SpeedupBin.FAILURE,
        bin_confidences={
            b: LikertConfidence.VERY_LOW
            for b in SpeedupBin
            if b != SpeedupBin.FAILURE
        },
        reasoning=reasoning,
    )


class KernelWorldModelEvaluationProvider:
    """Evaluates candidate kernels by predicting speedup via the kernel world model.

    Always compares the candidate against the fixed PyTorch reference kernel
    for the problem. The predicted speedup bin (1-8) is used as the reward;
    bin 0 (FAILURE) maps to reward=None.

    Parameters
    ----------
    reference_kernel_code:
        Source code of the PyTorch reference kernel for this task.
    task_info:
        Metadata identifying the KernelBench task.
    estimator:
        A ``SpeedupEstimator`` instance (typically ``LlmSpeedupEstimator``).
    hardware:
        Optional GPU hardware context to condition the estimate on.
    model_slug:
        The model identifier passed to the estimator. Only used for recording;
        it does not affect estimation and defaults to empty string when omitted.
    invocation_sink:
        Optional sink to record LLM usage after each evaluation. When ``None``,
        no tracking occurs. Records are only written when the estimator returns
        non-None usage.
    """

    def __init__(
        self,
        *,
        reference_kernel_code: str,
        task_info: KernelTaskInfo,
        estimator: SpeedupEstimator,
        hardware: HardwareContext | None = None,
        model_slug: str = "",
        invocation_sink: InvocationSink | None = None,
    ) -> None:
        self._reference = KernelImplementation(
            kernel_name="pytorch_reference",
            code=reference_kernel_code,
            runtime_ms=None,
        )
        self._task_info = task_info
        self._estimator = estimator
        self._hardware = hardware
        self._model_slug = model_slug
        self._invocation_sink = invocation_sink

    def evaluate(self, program_code: str) -> Evaluation[KernelRuntimeEstimate]:
        query = KernelRuntimeQuery(
            task=self._task_info,
            reference=self._reference,
            candidate=KernelImplementation(
                kernel_name="candidate",
                code=program_code,
                runtime_ms=None,
            ),
            hardware=self._hardware,
        )

        usage = None
        try:
            estimate, usage = self._estimator.estimate(query)
        except EstimatorParseError as exc:
            logger.warning(
                "Estimator parse error for {op}: {err}",
                op=self._task_info.op_name,
                err=exc,
            )
            estimate = _failure_estimate(f"Estimator parse error: {exc}")

        if usage is not None:
            logger.debug(
                "KWM usage for {op}: {input}in/{output}out tokens",
                op=self._task_info.op_name,
                input=usage.input_tokens,
                output=usage.output_tokens,
            )
            if self._invocation_sink is not None:
                self._invocation_sink.record(
                    KwmEvaluationRecord(
                        code_sha256=code_sha256(program_code),
                        model_slug=self._model_slug,
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        predicted_bin=int(estimate.predicted_bin),
                        timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    )
                )

        reward: float | None
        if estimate.predicted_bin == SpeedupBin.FAILURE:
            reward = None
        else:
            reward = float(estimate.predicted_bin)

        return Evaluation[KernelRuntimeEstimate](observation=estimate, reward=reward)
