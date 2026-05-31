"""LiteLLM-backed native-abstain estimator.

Returns a discriminated union :class:`PredictOrDefer` whose two arms
are the existing :class:`KernelRuntimeEstimate` and a new
:class:`Deferral`. The model is given two tools (predict, defer) and
must call exactly one — ``tool_choice="auto"`` lets it pick.

Parsing is strict: we accept exactly one tool call (the LLM's own
choice), validate its arguments against the corresponding wire-format
model, and translate to the domain model. Calling neither tool, or
calling both, raises :class:`EstimatorParseError`.
"""

from __future__ import annotations

from typing import Any

import litellm
from litellm import completion

from arid_badger.landscape_map.v2.abstain_outcome import (
    Deferral,
    Forecast,
    PredictOrDefer,
)
from arid_badger.landscape_map.v2.abstain_prompt_rendering import (
    render_abstain_system_prompt,
    render_abstain_user_prompt,
)
from arid_badger.landscape_map.v2.abstain_tool_spec import (
    DEFER_TOOL_NAME,
    DeferArguments,
    PREDICT_TOOL_NAME,
    both_openai_tool_specs,
)
from arid_badger.landscape_map.v2.domain import (
    KernelRuntimeQuery,
    LlmCallUsage,
)
from arid_badger.landscape_map.v2.parsing import (
    EstimatorParseError,
    parse_tool_call_args,
)


class AbstainingLlmSpeedupEstimator:
    """Two-tool LLM estimator that may predict or defer."""

    def __init__(
        self,
        model_slug: str,
        *,
        temperature: float = 1.0,
        max_tokens: int = 32000,
        request_timeout_s: float = 600.0,
        num_retries: int = 0,
    ) -> None:
        self._model_slug = model_slug
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._request_timeout_s = request_timeout_s
        self._num_retries = num_retries

    def estimate(
        self, query: KernelRuntimeQuery
    ) -> tuple[PredictOrDefer, LlmCallUsage | None]:
        kwargs = self._build_request_kwargs(query)
        response = completion(**kwargs)
        return self._parse_response(response)

    async def aestimate(
        self, query: KernelRuntimeQuery
    ) -> tuple[PredictOrDefer, LlmCallUsage | None]:
        kwargs = self._build_request_kwargs(query)
        response = await litellm.acompletion(**kwargs)
        return self._parse_response(response)

    def _build_request_kwargs(self, query: KernelRuntimeQuery) -> dict[str, Any]:
        return {
            "model": self._model_slug,
            "messages": [
                {"role": "system", "content": render_abstain_system_prompt()},
                {"role": "user", "content": render_abstain_user_prompt(query)},
            ],
            "tools": both_openai_tool_specs(),
            # ``tool_choice="required"`` forces the model to call one of
            # the two tools without pinning *which*. Together gpt-oss
            # silently returns empty content under ``"auto"`` — likely
            # the reasoning channel eats the budget without ever
            # emitting a tool call. ``"required"`` keeps predict-vs-
            # defer as the model's choice but rules out the no-call
            # path that breaks parsing.
            "tool_choice": "required",
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "timeout": self._request_timeout_s,
            "num_retries": self._num_retries,
        }

    def _parse_response(
        self, response: Any
    ) -> tuple[PredictOrDefer, LlmCallUsage | None]:
        choice = response.choices[0]
        message = choice.message
        tool_calls = list(getattr(message, "tool_calls", None) or [])
        if not tool_calls:
            content_preview = (getattr(message, "content", None) or "")[:200]
            raise EstimatorParseError(
                f"model called no tool "
                f"(finish_reason={choice.finish_reason!r}, "
                f"content_preview={content_preview!r})"
            )
        # If the model called both, pick by priority: predict outranks
        # defer only because predict is the more informative outcome.
        # We surface this via the parse error if the model violated
        # the "exactly one" instruction.
        names = [tc.function.name for tc in tool_calls]
        if len(tool_calls) > 1:
            raise EstimatorParseError(
                f"model called multiple tools (names={names!r}); the "
                "abstain estimator requires exactly one of "
                f"{{{PREDICT_TOOL_NAME!r}, {DEFER_TOOL_NAME!r}}}"
            )
        target = tool_calls[0]
        usage = _extract_usage(response)
        if target.function.name == PREDICT_TOOL_NAME:
            estimate = parse_tool_call_args(target.function.arguments)
            return Forecast(estimate=estimate), usage
        if target.function.name == DEFER_TOOL_NAME:
            try:
                args = DeferArguments.model_validate_json(
                    target.function.arguments
                )
            except Exception as exc:
                raise EstimatorParseError(
                    f"defer tool arguments failed schema validation: {exc}"
                ) from exc
            return Deferral(reason=args.reason), usage
        raise EstimatorParseError(
            f"model called unexpected tool {target.function.name!r}"
        )


def _extract_usage(response: Any) -> LlmCallUsage | None:
    raw_usage = getattr(response, "usage", None)
    if raw_usage is None:
        return None
    prompt_tokens = getattr(raw_usage, "prompt_tokens", None)
    completion_tokens = getattr(raw_usage, "completion_tokens", None)
    if prompt_tokens is None or completion_tokens is None:
        return None
    return LlmCallUsage(
        input_tokens=int(prompt_tokens),
        output_tokens=int(completion_tokens),
    )


__all__ = [
    "AbstainingLlmSpeedupEstimator",
    "Deferral",
    "Forecast",
    "PredictOrDefer",
]
