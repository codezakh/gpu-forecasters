"""Tinker-backed estimator for the abstain-or-forecast surrogate.

Sibling of :class:`TinkerSamplingClientEstimator` — same path
(prompt build → ``SamplingClient.sample_async`` → renderer parse)
but registers both abstain tool specs on the prefix, renders the
abstain system + user prompts, and returns a
:class:`PredictOrDefer` outcome instead of a bare
:class:`KernelRuntimeEstimate`.

Used by the e0169 scoring runner to evaluate the trained-correctness-
abstain checkpoint on the canonical eval-set registry. The model can
emit either the predict or defer tool — whichever it called is what
the parser yields, exactly as during RL training.
"""

from __future__ import annotations

from typing import Any, Protocol

import tinker
from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

from .abstain_outcome import Deferral, Forecast, PredictOrDefer
from .abstain_prompt_rendering import (
    render_abstain_system_prompt,
    render_abstain_user_prompt,
)
from .abstain_tool_spec import (
    DEFER_TOOL_NAME,
    DeferArguments,
    PREDICT_TOOL_NAME,
    both_cookbook_tool_specs,
)
from .domain import KernelRuntimeQuery, LlmCallUsage
from .parsing import EstimatorParseError, parse_tool_call_args


DEFAULT_RENDERER_NAME = "gpt_oss_medium_reasoning"


class AsyncAbstainSpeedupEstimator(Protocol):
    """Protocol for asynchronously emitting a predict-or-defer outcome."""

    async def aestimate(
        self, query: KernelRuntimeQuery
    ) -> tuple[PredictOrDefer, LlmCallUsage | None]: ...


class TinkerSamplingClientAbstainingEstimator:
    """Two-tool Tinker sampler — yields either a forecast or a deferral.

    The contract matches the abstain RL env: the model is given two
    tools and must call exactly one. Zero, multiple, or unknown tool
    calls raise :class:`EstimatorParseError`. Per-arm validation
    (``parse_tool_call_args`` for predict, :class:`DeferArguments`
    schema for defer) raises :class:`EstimatorParseError` on failure.
    """

    def __init__(
        self,
        *,
        base_model: str = "openai/gpt-oss-20b",
        model_path: str | None = None,
        renderer_name: str = DEFAULT_RENDERER_NAME,
        temperature: float = 1.0,
        max_tokens: int = 16384,
        service_client: tinker.ServiceClient | None = None,
        sampling_client: tinker.SamplingClient | None = None,
        renderer: Renderer | None = None,
    ) -> None:
        self._base_model = base_model
        self._model_path = model_path
        self._renderer_name = renderer_name
        self._temperature = temperature
        self._max_tokens = max_tokens

        if renderer is None:
            tokenizer = get_tokenizer(base_model)
            renderer = get_renderer(renderer_name, tokenizer)
        self._renderer: Renderer = renderer

        if sampling_client is None:
            sc = service_client or tinker.ServiceClient()
            sampling_client = sc.create_sampling_client(
                base_model=base_model, model_path=model_path
            )
        self._sampling_client: tinker.SamplingClient = sampling_client

    # -- public API ---------------------------------------------------------

    async def aestimate(
        self, query: KernelRuntimeQuery
    ) -> tuple[PredictOrDefer, LlmCallUsage | None]:
        prompt = self._build_prompt(query)
        response = await self._sampling_client.sample_async(
            prompt=prompt,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                stop=self._renderer.get_stop_sequences(),
            ),
        )
        tokens = response.sequences[0].tokens
        return self._parse_tokens(tokens, prompt_length=prompt.length)

    def estimate(
        self, query: KernelRuntimeQuery
    ) -> tuple[PredictOrDefer, LlmCallUsage | None]:
        """Synchronous variant — calls into the async API."""
        import asyncio

        return asyncio.run(self.aestimate(query))

    # -- internals ----------------------------------------------------------

    def _build_prompt(self, query: KernelRuntimeQuery) -> tinker.ModelInput:
        prefix = self._renderer.create_conversation_prefix_with_tools(
            tools=both_cookbook_tool_specs(),  # pyright: ignore[reportArgumentType]
            system_prompt=render_abstain_system_prompt(),
        )
        messages: list[Any] = list(prefix) + [
            {"role": "user", "content": render_abstain_user_prompt(query)},
        ]
        return self._renderer.build_generation_prompt(messages)

    def _parse_tokens(
        self, tokens: list[int], prompt_length: int
    ) -> tuple[PredictOrDefer, LlmCallUsage | None]:
        message, parsed_ok = self._renderer.parse_response(tokens)
        tool_calls = list(message.get("tool_calls") or [])
        if len(tool_calls) == 0:
            unparsed = list(message.get("unparsed_tool_calls") or [])
            raise EstimatorParseError(
                f"renderer parsed no tool_calls "
                f"(parsed_ok={parsed_ok}, unparsed={len(unparsed)})"
            )
        if len(tool_calls) > 1:
            names = [tc.function.name for tc in tool_calls]
            raise EstimatorParseError(
                f"model called multiple tools (names={names!r}); the "
                f"abstain estimator requires exactly one of "
                f"{{{PREDICT_TOOL_NAME!r}, {DEFER_TOOL_NAME!r}}}"
            )
        call = tool_calls[0]
        usage = LlmCallUsage(
            input_tokens=prompt_length,
            output_tokens=len(tokens),
        )
        if call.function.name == PREDICT_TOOL_NAME:
            estimate = parse_tool_call_args(call.function.arguments)
            return Forecast(estimate=estimate), usage
        if call.function.name == DEFER_TOOL_NAME:
            try:
                args = DeferArguments.model_validate_json(
                    call.function.arguments
                )
            except Exception as exc:
                raise EstimatorParseError(
                    f"defer tool arguments failed schema validation: {exc}"
                ) from exc
            return Deferral(reason=args.reason), usage
        raise EstimatorParseError(
            f"model called unexpected tool {call.function.name!r}"
        )


__all__ = [
    "AsyncAbstainSpeedupEstimator",
    "TinkerSamplingClientAbstainingEstimator",
]
