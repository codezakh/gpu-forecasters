"""LiteLLM-backed estimator for frontier and OpenAI-compatible providers.

Targets:
  - frontier: ``gemini/gemini-3-flash-preview`` and similar
  - Together gpt-oss: ``together_ai/openai/gpt-oss-20b`` /
    ``together_ai/openai/gpt-oss-120b``

The estimator forces a single call to ``submit_kernel_runtime_estimate``
via ``tool_choice``, parses the returned JSON with
:func:`parse_tool_call_args`, and surfaces token usage when the
provider returns it.
"""

from __future__ import annotations

from typing import Any

import litellm
from litellm import completion

from arid_badger.typing_utils import implements

from .domain import (
    AsyncSpeedupEstimator,
    KernelRuntimeEstimate,
    KernelRuntimeQuery,
    LlmCallUsage,
    SpeedupEstimator,
)
from .parsing import EstimatorParseError, parse_tool_call_args
from .prompt_rendering import render_system_prompt, render_user_prompt
from .tool_spec import TOOL_NAME, openai_tool_spec


class LlmSpeedupEstimator:
    """Estimate speedup by issuing a single forced tool call via LiteLLM.

    All non-defaulted parameters are kwargs — ``model_slug`` selects
    the provider/model and the rest control the request shape.
    """

    def __init__(
        self,
        model_slug: str,
        *,
        temperature: float = 1.0,
        max_tokens: int = 16384,
        request_timeout_s: float = 600.0,
        num_retries: int = 0,
        force_tool_choice: bool = True,
    ) -> None:
        self._model_slug = model_slug
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._request_timeout_s = request_timeout_s
        self._num_retries = num_retries
        self._force_tool_choice = force_tool_choice

    # -- public API ---------------------------------------------------------

    def estimate(
        self, query: KernelRuntimeQuery
    ) -> tuple[KernelRuntimeEstimate, LlmCallUsage | None]:
        kwargs = self._build_request_kwargs(query)
        response = completion(**kwargs)
        return self._parse_response(response)

    async def aestimate(
        self, query: KernelRuntimeQuery
    ) -> tuple[KernelRuntimeEstimate, LlmCallUsage | None]:
        kwargs = self._build_request_kwargs(query)
        response = await litellm.acompletion(**kwargs)
        return self._parse_response(response)

    # -- internals ----------------------------------------------------------

    def _build_request_kwargs(self, query: KernelRuntimeQuery) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model_slug,
            "messages": [
                {"role": "system", "content": render_system_prompt()},
                {"role": "user", "content": render_user_prompt(query)},
            ],
            "tools": [openai_tool_spec()],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "timeout": self._request_timeout_s,
            "num_retries": self._num_retries,
        }
        if self._force_tool_choice:
            kwargs["tool_choice"] = {
                "type": "function",
                "function": {"name": TOOL_NAME},
            }
        return kwargs

    def _parse_response(
        self, response: Any
    ) -> tuple[KernelRuntimeEstimate, LlmCallUsage | None]:
        choice = response.choices[0]
        message = choice.message
        tool_calls = list(getattr(message, "tool_calls", None) or [])
        if not tool_calls:
            content_preview = (getattr(message, "content", None) or "")[:200]
            raise EstimatorParseError(
                f"model did not call any tool "
                f"(finish_reason={choice.finish_reason!r}, "
                f"content_preview={content_preview!r})"
            )
        target = next(
            (tc for tc in tool_calls if tc.function.name == TOOL_NAME),
            tool_calls[0],
        )
        if target.function.name != TOOL_NAME:
            raise EstimatorParseError(
                f"model called unexpected tool {target.function.name!r}"
            )

        estimate = parse_tool_call_args(target.function.arguments)
        usage = _extract_usage(response)
        return estimate, usage


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


implements(SpeedupEstimator)(LlmSpeedupEstimator)
implements(AsyncSpeedupEstimator)(LlmSpeedupEstimator)
