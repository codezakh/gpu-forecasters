"""LLM-backed ``WorldModelBuilder``.

Single frozen LLM, three internal seams (renderers, LLM client,
parser) — all swappable for tests.

Two distinct loops:

* ``propose_hypothesis`` is a one-shot call with parse-error retry:
  the LLM sees the world model and returns a single tagged response.
  Bounded by ``max_retries``.
* ``propose_explanation`` is an iterative tool-use loop: the LLM
  emits the three explanation tags + ONE SEARCH/REPLACE edit per turn,
  sees the updated world model after each apply, and signals
  ``<done/>`` to terminate. Bounded by ``max_turns`` total turns and
  ``max_apply_failures`` consecutive bad-edit retries. Single-edit-
  per-turn is the deliberate fix for the v1-shipping multi-block
  failure mode where later blocks referenced text that earlier
  blocks already rewrote.

On exhaustion either loop raises ``BuilderError`` and the driver
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
from ulid import ULID

from arid_badger.hill_climbing.domain import ObservationT
from arid_badger.theory_builder.v1.diff import (
    DiffApplyError,
    apply_diffs,
    parse_diff_blocks,
)
from arid_badger.theory_builder.v1.domain import (
    Explanation,
    ExperimentResult,
    Hypothesis,
    WorldModel,
    WorldModelDiff,
)
from arid_badger.theory_builder.v1.parser import (
    ParseError,
    ParsedExplanationTags,
    has_done_signal,
    parse_explanation_tags,
    parse_hypothesis_into_domain,
)
from arid_badger.theory_builder.v1.prompts import (
    EXPLANATION_SYSTEM_PROMPT,
    HYPOTHESIS_SYSTEM_PROMPT,
    apply_failed_message,
    explanation_user_prompt,
    hypothesis_user_prompt,
    missing_diff_message,
    missing_explanation_tags_message,
    next_edit_message,
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
        hypothesis_system_prompt: str = HYPOTHESIS_SYSTEM_PROMPT,
        explanation_system_prompt: str = EXPLANATION_SYSTEM_PROMPT,
        max_retries: int = 3,
        max_turns: int = 12,
        max_apply_failures: int = 3,
        request_timeout_s: float = 600.0,
        num_retries: int = 4,
        temperature: float = 1.0,
        max_tokens: int | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        if max_apply_failures < 1:
            raise ValueError("max_apply_failures must be >= 1")
        self._model_slug = model_slug
        self._result_renderer = result_renderer
        self._world_model_renderer = (
            world_model_renderer or MarkdownWorldModelRenderer()
        )
        self._hypothesis_system_prompt = hypothesis_system_prompt
        self._explanation_system_prompt = explanation_system_prompt
        self._max_retries = max_retries
        self._max_turns = max_turns
        self._max_apply_failures = max_apply_failures
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
            {"role": "system", "content": self._hypothesis_system_prompt},
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
        """Iterative tool-use loop.

        Turn 1: LLM emits the three explanation tags + at most one
        SEARCH/REPLACE edit (or ``<done/>``). Tags are extracted once
        and reused across the rest of the loop.

        Subsequent turns: LLM emits one SEARCH/REPLACE edit, applied
        against the *current* (possibly already-edited) document.
        After each successful apply the LLM is shown the new doc.
        ``<done/>`` terminates.

        Failure modes — each handled by feeding the issue back to the
        LLM rather than aborting:

        * Missing explanation tags on turn 1 → re-prompt.
        * No diff and no ``<done/>`` after turn 1 → nudge.
        * ``DiffApplyError`` (SEARCH didn't match uniquely) → show the
          error and the current document; the LLM can either correct
          or give up via ``<done/>``. Bounded by ``max_apply_failures``.

        On total turn exhaustion the builder raises ``BuilderError``.
        """
        wm_str = self._world_model_renderer.render(world_model)
        result_str = self._result_renderer.render(result)
        user = explanation_user_prompt(wm_str, hypothesis, result_str)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._explanation_system_prompt},
            {"role": "user", "content": user},
        ]

        current_text = world_model.text
        tags: ParsedExplanationTags | None = None
        applied_diffs: list[WorldModelDiff] = []
        apply_failures = 0
        last_error: str | None = None

        for turn in range(self._max_turns):
            response = self._chat(messages)
            messages.append({"role": "assistant", "content": response})

            # --- 1. Parse explanation tags on the first turn only.
            if tags is None:
                try:
                    tags = parse_explanation_tags(response)
                except ParseError as exc:
                    last_error = str(exc)
                    logger.warning(
                        "Explanation tag parse failed (turn {t}/{n}): {err}",
                        t=turn + 1,
                        n=self._max_turns,
                        err=exc,
                    )
                    if turn == self._max_turns - 1:
                        break
                    messages.append(
                        {
                            "role": "user",
                            "content": missing_explanation_tags_message(
                                str(exc)
                            ),
                        }
                    )
                    continue

            # --- 2. Look for the termination signal and any edits.
            done = has_done_signal(response)
            diffs = parse_diff_blocks(response)

            # No edit this turn — either we're done, or we need a nudge.
            if not diffs:
                if done:
                    return (
                        _build_explanation(
                            hypothesis.id, tags, applied_diffs
                        ),
                        current_text,
                    )
                if turn == self._max_turns - 1:
                    last_error = (
                        "no SEARCH/REPLACE block and no <done/> signal"
                    )
                    break
                messages.append(
                    {
                        "role": "user",
                        "content": missing_diff_message(),
                    }
                )
                continue

            # --- 3. Apply the first edit. Extra blocks are ignored;
            # the LLM is being taught one-edit-per-turn.
            edit = diffs[0]
            try:
                current_text = apply_diffs(current_text, [edit])
            except DiffApplyError as exc:
                apply_failures += 1
                last_error = str(exc)
                logger.warning(
                    "Diff apply failed (turn {t}/{n}, fail {f}/{m}): {err}",
                    t=turn + 1,
                    n=self._max_turns,
                    f=apply_failures,
                    m=self._max_apply_failures,
                    err=exc,
                )
                if (
                    apply_failures >= self._max_apply_failures
                    or turn == self._max_turns - 1
                ):
                    break
                messages.append(
                    {
                        "role": "user",
                        "content": apply_failed_message(
                            str(exc), current_text
                        ),
                    }
                )
                continue

            # Successful apply.
            applied_diffs.append(edit)

            # If the same response also signalled done, exit cleanly.
            if done:
                return (
                    _build_explanation(
                        hypothesis.id, tags, applied_diffs
                    ),
                    current_text,
                )

            messages.append(
                {
                    "role": "user",
                    "content": next_edit_message(current_text),
                }
            )

        raise BuilderError(
            f"explanation production failed after {self._max_turns} "
            f"turn(s): {last_error}"
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


def _build_explanation(
    hypothesis_id: ULID,
    tags: ParsedExplanationTags,
    applied_diffs: list[WorldModelDiff],
) -> Explanation:
    """Combine the parsed tags from turn 1 with the diffs that
    actually applied across the iterative loop into a final
    ``Explanation`` domain object."""
    return Explanation(
        hypothesis_id=hypothesis_id,
        gap=tags.gap,
        mechanism=tags.mechanism,
        belief_update=tags.belief_update,
        diffs=applied_diffs,
    )


__all__ = [
    "BuilderError",
    "LLMWorldModelBuilder",
]
