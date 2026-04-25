"""LLM-backed ``WorldModelBuilder``.

Single frozen LLM, three internal seams (renderers, LLM client,
parser) — all swappable for tests.

The retry loop is a small agentic conversation: on apply / parse
failure, the builder appends the failed attempt + the error to the
chat as a synthetic user turn and re-prompts. Bounded by ``max_retries``;
on exhaustion the builder raises ``BuilderError`` and the driver
records ``HypothesisFailed`` / ``ExplanationFailed`` and moves on.

Async LiteLLM is wrapped in a thin sync surface (using a per-call
``asyncio.run``) because the outer loop is single-threaded and the
overhead of spinning a fresh event loop per call is negligible
compared to the API latency.
"""

from __future__ import annotations

import asyncio
from typing import Any, Generic, Self

import litellm
from loguru import logger

from arid_badger.hill_climbing.domain import ObservationT
from arid_badger.theory_builder.v1.diff import (
    DiffApplyError,
    apply_diffs,
)
from arid_badger.theory_builder.v1.domain import (
    Explanation,
    ExperimentResult,
    Hypothesis,
    WorldModel,
)
from arid_badger.theory_builder.v1.parser import (
    ParseError,
    parse_explanation,
    parse_hypothesis_into_domain,
)
from arid_badger.theory_builder.v1.prompts import (
    EXPLANATION_SYSTEM_PROMPT,
    HYPOTHESIS_SYSTEM_PROMPT,
    explanation_user_prompt,
    hypothesis_user_prompt,
)
from arid_badger.theory_builder.v1.renderers import (
    ExperimentResultRenderer,
    MarkdownWorldModelRenderer,
    WorldModelRenderer,
)


class BuilderError(RuntimeError):
    """Raised when the builder exhausts its retry budget. The driver
    catches this and emits the appropriate ``*Failed`` event."""


class LLMWorldModelBuilder(Generic[ObservationT]):
    """LLM-backed builder.

    Construction args separate the parts the spec expects to iterate
    on (renderers, retry limits) from the parts that are basically
    static (model slug, decoding params).

    Implements ``WorldModelBuilder[ObservationT]``.
    """

    def __init__(
        self,
        *,
        model_slug: str,
        result_renderer: ExperimentResultRenderer[ObservationT],
        world_model_renderer: WorldModelRenderer | None = None,
        max_retries: int = 3,
        request_timeout_s: float = 600.0,
        num_retries: int = 4,
        temperature: float = 1.0,
        max_tokens: int | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self._model_slug = model_slug
        self._result_renderer = result_renderer
        self._world_model_renderer = (
            world_model_renderer or MarkdownWorldModelRenderer()
        )
        self._max_retries = max_retries
        self._request_timeout_s = request_timeout_s
        self._num_retries = num_retries
        self._temperature = temperature
        self._max_tokens = max_tokens

    # --- Lifecycle (no-op for now; LiteLLM is stateless) ---------------

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        return None

    # --- propose_hypothesis -------------------------------------------

    def propose_hypothesis(self, world_model: WorldModel) -> Hypothesis:
        wm_str = self._world_model_renderer.render(world_model)
        user = hypothesis_user_prompt(wm_str)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": HYPOTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

        last_error: str | None = None
        for attempt in range(self._max_retries + 1):
            response = self._chat(messages)
            try:
                return parse_hypothesis_into_domain(response)
            except ParseError as exc:
                last_error = str(exc)
                logger.warning(
                    "Hypothesis parse failed (attempt {a}/{n}): {err}",
                    a=attempt + 1,
                    n=self._max_retries + 1,
                    err=exc,
                )
                if attempt == self._max_retries:
                    break
                # Append the failed attempt + error and re-prompt.
                messages.append({"role": "assistant", "content": response})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous response could not be parsed: {exc}.\n"
                            "Please try again. Make sure every required tag is "
                            "present and non-empty."
                        ),
                    }
                )
        raise BuilderError(
            f"hypothesis proposal failed after {self._max_retries + 1} "
            f"attempt(s): {last_error}"
        )

    # --- propose_explanation ------------------------------------------

    def propose_explanation(
        self,
        world_model: WorldModel,
        hypothesis: Hypothesis,
        result: ExperimentResult[ObservationT],
    ) -> tuple[Explanation, str]:
        wm_str = self._world_model_renderer.render(world_model)
        result_str = self._result_renderer.render(result)
        user = explanation_user_prompt(wm_str, hypothesis, result_str)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

        last_error: str | None = None
        for attempt in range(self._max_retries + 1):
            response = self._chat(messages)
            try:
                explanation = parse_explanation(response, hypothesis.id)
                new_text = apply_diffs(world_model.text, explanation.diffs)
                return explanation, new_text
            except (ParseError, DiffApplyError) as exc:
                last_error = str(exc)
                logger.warning(
                    "Explanation produce failed (attempt {a}/{n}): {err}",
                    a=attempt + 1,
                    n=self._max_retries + 1,
                    err=exc,
                )
                if attempt == self._max_retries:
                    break
                messages.append({"role": "assistant", "content": response})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response could not be applied: "
                            f"{exc}.\n"
                            "Please try again. Make sure all required tags "
                            "are present, and that each SEARCH block matches "
                            "exactly one location in the world model "
                            "(use enough surrounding context to make it "
                            "unique)."
                        ),
                    }
                )
        raise BuilderError(
            f"explanation production failed after {self._max_retries + 1} "
            f"attempt(s): {last_error}"
        )

    # --- LLM client ---------------------------------------------------

    def _chat(self, messages: list[dict[str, str]]) -> str:
        """Single ``litellm.acompletion`` call. Returns the assistant
        response content. Raises on infrastructure failure (handled
        by the driver as a per-step failure)."""
        try:
            content = asyncio.run(self._chat_async(messages))
        except Exception as exc:
            raise BuilderError(f"litellm call failed: {exc}") from exc
        if not content:
            raise BuilderError("LLM returned empty content")
        return content

    async def _chat_async(self, messages: list[dict[str, str]]) -> str:
        kwargs: dict[str, Any] = {
            "model": self._model_slug,
            "messages": messages,
            "temperature": self._temperature,
            "num_retries": self._num_retries,
            "timeout": self._request_timeout_s,
        }
        if self._max_tokens is not None:
            kwargs["max_tokens"] = self._max_tokens
        response = await litellm.acompletion(**kwargs)
        content = response.choices[0].message.content  # pyright: ignore[reportAttributeAccessIssue]
        return content or ""


__all__ = [
    "BuilderError",
    "LLMWorldModelBuilder",
]
