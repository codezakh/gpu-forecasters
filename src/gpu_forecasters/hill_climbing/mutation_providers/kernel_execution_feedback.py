"""Kernel mutation provider conditioned on real GPU execution feedback.

Canonical library version of what e0015's `WakePhaseMutationProvider` did
inline. Uses `format_feedback_mutation_prompt` when real execution feedback is
available and falls back to the zero-shot base prompt on infrastructure
failures.

Issues a single `litellm.completion(..., n=num_mutations)` call rather than
`num_mutations` independent calls. For Gemini this maps to
`generationConfig.candidateCount`, which:
  - bills the prompt tokens once instead of `num_mutations` times,
  - counts as a single request against RPM quota, and
  - collapses wall-clock to one request instead of fanning out.

If the single call returns fewer valid code blocks than requested (e.g. some
candidates fail to parse), the shortfall is filled with a single top-up call
for the deficit. One top-up attempt only — we never loop.
"""

from __future__ import annotations

import time
import traceback
from datetime import datetime, timezone
from typing import Literal

import litellm
from loguru import logger
from pydantic import BaseModel

from arid_badger.greedy_search.feedback_mutation import format_feedback_mutation_prompt
from arid_badger.hill_climbing.domain import Evaluation, MutationProvider
from arid_badger.hill_climbing.scoring_providers.kernelbench import (
    KernelBenchObservation,
)
from arid_badger.invocation_sink import InvocationSink, code_sha256
from arid_badger.kernelbench.core import InfrastructureFailureFeedback
from arid_badger.typing_utils import implements

from kernelbench.prompt_constructor_toml import get_prompt_for_backend
from kernelbench.utils import extract_first_code


class KernelExecutionFeedbackMutationRecord(BaseModel, frozen=True):
    """Invocation record for one `generate_mutations` call.

    A single call may issue up to two underlying LLM requests: the primary
    `n=num_mutations` request and an optional top-up for any shortfall. The
    totals aggregate across both.
    """

    kind: Literal["kernel_execution_feedback_mutation"] = (
        "kernel_execution_feedback_mutation"
    )
    parent_code_sha256: str
    child_code_sha256s: list[str]
    model_slug: str
    total_input_tokens: int
    total_output_tokens: int
    num_mutations_requested: int
    num_mutations_produced: int
    num_llm_requests: int
    wall_clock_seconds: float
    timestamp_utc: str


class KernelExecutionFeedbackMutationProvider:
    """Generates kernel mutations conditioned on real GPU execution feedback.

    Implements `MutationProvider[KernelBenchObservation]`.
    """

    _base_prompt: str
    _model_slug: str
    _invocation_sink: InvocationSink | None

    def __init__(
        self,
        *,
        reference_kernel_code: str,
        model_slug: str,
        backend: str = "cuda",
        precision: str = "fp32",
        invocation_sink: InvocationSink | None = None,
    ) -> None:
        self._base_prompt = get_prompt_for_backend(
            ref_arch_src=reference_kernel_code,
            backend=backend,
            option="zero_shot",
            precision=precision,
        )
        self._model_slug = model_slug
        self._invocation_sink = invocation_sink

    def generate_mutations(
        self,
        program_code: str,
        num_mutations: int,
        evaluation: Evaluation[KernelBenchObservation],
    ) -> list[str]:
        feedback = evaluation.observation.feedback
        if isinstance(feedback, InfrastructureFailureFeedback):
            prompt = self._base_prompt
        else:
            prompt = format_feedback_mutation_prompt(
                base_prompt=self._base_prompt,
                previous_kernel_code=program_code,
                feedback=feedback,
            )

        start_time_s = time.perf_counter()
        codes, input_tokens, output_tokens, num_requests = self._request_candidates(
            prompt=prompt, n=num_mutations
        )

        # Single top-up for any shortfall. Do not loop — if even the top-up
        # under-delivers, accept the partial batch. PUCT handles fewer children
        # than requested gracefully.
        if len(codes) < num_mutations:
            deficit = num_mutations - len(codes)
            logger.info(
                "Primary call produced {got}/{want} candidates; topping up {deficit}.",
                got=len(codes),
                want=num_mutations,
                deficit=deficit,
            )
            more_codes, more_in, more_out, more_requests = self._request_candidates(
                prompt=prompt, n=deficit
            )
            codes.extend(more_codes)
            input_tokens += more_in
            output_tokens += more_out
            num_requests += more_requests

        wall_clock_seconds = time.perf_counter() - start_time_s

        if self._invocation_sink is not None:
            self._invocation_sink.record(
                KernelExecutionFeedbackMutationRecord(
                    parent_code_sha256=code_sha256(program_code),
                    child_code_sha256s=[code_sha256(c) for c in codes],
                    model_slug=self._model_slug,
                    total_input_tokens=input_tokens,
                    total_output_tokens=output_tokens,
                    num_mutations_requested=num_mutations,
                    num_mutations_produced=len(codes),
                    num_llm_requests=num_requests,
                    wall_clock_seconds=wall_clock_seconds,
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                )
            )

        return codes

    def _request_candidates(
        self, *, prompt: str, n: int
    ) -> tuple[list[str], int, int, int]:
        """Issue a single `completion(..., n=n)` call and extract code blocks.

        Returns (codes, input_tokens, output_tokens, num_requests). On a
        complete failure of the request itself, returns an empty code list
        with `num_requests=1` so callers can still account for the attempt.
        """
        try:
            response = litellm.completion(
                model=self._model_slug,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                n=n,
            )
        except Exception:
            logger.warning(
                "LLM completion(n={n}) failed:\n{tb}",
                n=n,
                tb=traceback.format_exc(),
            )
            return [], 0, 0, 1

        raw_usage = response.usage  # pyright: ignore[reportAttributeAccessIssue]
        input_tokens = raw_usage.prompt_tokens if raw_usage is not None else 0
        output_tokens = raw_usage.completion_tokens if raw_usage is not None else 0

        codes: list[str] = []
        for index, choice in enumerate(response.choices):  # pyright: ignore[reportAttributeAccessIssue]
            content = choice.message.content  # pyright: ignore[reportAttributeAccessIssue]
            code = extract_first_code(content, code_language_types=["python"])  # pyright: ignore[reportArgumentType]
            if not code:
                logger.warning(
                    "LLM choice {index}: no code block extracted.",
                    index=index,
                )
                continue
            codes.append(code)

        return codes, input_tokens, output_tokens, 1


implements(MutationProvider[KernelBenchObservation])(
    KernelExecutionFeedbackMutationProvider
)
