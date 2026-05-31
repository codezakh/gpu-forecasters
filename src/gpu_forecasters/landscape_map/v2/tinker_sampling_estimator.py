"""Estimator backed by ``tinker.SamplingClient`` and the cookbook GptOssRenderer.

Used for scoring trained gpt-oss checkpoints (e0125-style calibration
runs) and for the base gpt-oss-20b/120b sample path when we want to
exercise the same parsing pipeline used by RL ``Env.step``.

Path:
  1. Build the conversation prefix with the tool spec via
     ``GptOssRenderer.create_conversation_prefix_with_tools``.
  2. Append the user message and call
     ``renderer.build_generation_prompt`` to get a tokenized
     ``ModelInput``.
  3. Sample with ``SamplingClient.sample_async`` using the renderer's
     stop sequences (``[<|return|>, <|call|>]``).
  4. Parse the returned tokens with ``renderer.parse_response`` to
     recover ``message["tool_calls"]`` and validate the JSON arguments
     via :func:`parse_tool_call_args`.

The default renderer is ``gpt_oss_medium_reasoning`` rather than the
``model_info``-recommended ``gpt_oss_no_sysprompt``: with the latter,
the e0137 smoke run found that the base gpt-oss-20b model often emits
its answer in the ``final`` channel rather than calling the tool. The
medium-reasoning renderer prepends OpenAI's system prompt with
reasoning effort and dramatically improves tool-call adherence,
matching what the RL training experiments
(``e0117``/``e0121``/``e0124``) already use.
"""

from __future__ import annotations

from typing import Any

import tinker
from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

from gpu_forecasters.typing_utils import implements

from .domain import (
    AsyncSpeedupEstimator,
    KernelRuntimeEstimate,
    KernelRuntimeQuery,
    LlmCallUsage,
    SpeedupEstimator,
)
from .parsing import EstimatorParseError, parse_tool_call_args
from .prompt_rendering import render_system_prompt, render_user_prompt
from .tool_spec import TOOL_NAME, cookbook_tool_spec


DEFAULT_RENDERER_NAME = "gpt_oss_medium_reasoning"


class TinkerSamplingClientEstimator:
    """Estimator that drives a Tinker ``SamplingClient``.

    Pass either a base-model name (sampler reads the bare base model) or
    a ``model_path`` pointing at a trained sampler checkpoint, plus a
    ``ServiceClient`` if you already have one. This object is a deep
    wrapper around ``(SamplingClient, Renderer)`` and is the seam that
    makes test-time inference look identical to RL ``Env.step``.
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

    @classmethod
    def for_base_model(
        cls,
        base_model: str = "openai/gpt-oss-20b",
        *,
        renderer_name: str = DEFAULT_RENDERER_NAME,
        temperature: float = 1.0,
        max_tokens: int = 16384,
    ) -> TinkerSamplingClientEstimator:
        """Convenience constructor for sampling the bare base model.

        Equivalent to ``TinkerSamplingClientEstimator(base_model=...)``,
        kept as a named alternate per the deep-objects pattern.
        """
        return cls(
            base_model=base_model,
            renderer_name=renderer_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # -- public API ---------------------------------------------------------

    async def aestimate(
        self, query: KernelRuntimeQuery
    ) -> tuple[KernelRuntimeEstimate, LlmCallUsage | None]:
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
    ) -> tuple[KernelRuntimeEstimate, LlmCallUsage | None]:
        """Synchronous variant — calls into the async API.

        Tinker's sampling API is async-first; we expose ``estimate``
        for parity with :class:`SpeedupEstimator` but recommend
        :meth:`aestimate` for any real workload.
        """
        import asyncio

        return asyncio.run(self.aestimate(query))

    # -- internals ----------------------------------------------------------

    def _build_prompt(self, query: KernelRuntimeQuery) -> tinker.ModelInput:
        prefix = self._renderer.create_conversation_prefix_with_tools(
            tools=[cookbook_tool_spec()],  # pyright: ignore[reportArgumentType]
            system_prompt=render_system_prompt(),
        )
        messages: list[Any] = list(prefix) + [
            {"role": "user", "content": render_user_prompt(query)},
        ]
        return self._renderer.build_generation_prompt(messages)

    def _parse_tokens(
        self, tokens: list[int], prompt_length: int
    ) -> tuple[KernelRuntimeEstimate, LlmCallUsage | None]:
        message, parsed_ok = self._renderer.parse_response(tokens)
        tool_calls = list(message.get("tool_calls") or [])
        if not tool_calls:
            unparsed = list(message.get("unparsed_tool_calls") or [])
            raise EstimatorParseError(
                f"renderer parsed no tool_calls "
                f"(parsed_ok={parsed_ok}, unparsed={len(unparsed)})"
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

        # Tinker's sampling API does not split prompt vs completion
        # tokens for us, so we approximate: the Tinker sampler reports
        # ``len(tokens)`` as the *generated* tokens, and we already know
        # the prompt length from the input.
        usage = LlmCallUsage(
            input_tokens=prompt_length,
            output_tokens=len(tokens),
        )
        return estimate, usage


implements(AsyncSpeedupEstimator)(TinkerSamplingClientEstimator)
implements(SpeedupEstimator)(TinkerSamplingClientEstimator)
