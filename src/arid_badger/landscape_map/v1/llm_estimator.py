from __future__ import annotations

import json
import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from litellm import completion
from loguru import logger

from arid_badger.typing_utils import implements

from .domain import (
    KernelRuntimeEstimate,
    KernelRuntimeQuery,
    LikertConfidence,
    LlmCallUsage,
    SpeedupBin,
    SpeedupEstimator,
)

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_JINJA_ENV = Environment(
    loader=FileSystemLoader(str(_PROMPTS_DIR)),
    autoescape=False,
)

# Pre-build lookup for case-insensitive SpeedupBin name matching
_BIN_BY_NAME: dict[str, SpeedupBin] = {b.name.lower(): b for b in SpeedupBin}

# Pre-build lookup for case-insensitive LikertConfidence matching
_CONFIDENCE_BY_VALUE: dict[str, LikertConfidence] = {
    c.value.lower(): c for c in LikertConfidence
}


class EstimatorParseError(Exception):
    """Raised when an LLM response cannot be parsed into a KernelRuntimeEstimate."""


def _extract_json_from_response(content: str) -> dict:
    """Extract JSON dict from LLM response using multiple strategies.

    Tries in order:
    1. ```kernel-runtime-estimate code block
    2. ```json code block
    3. Raw JSON parse of entire content
    """
    # Strategy 1: kernel-runtime-estimate fenced block
    match = re.search(
        r"```kernel-runtime-estimate\s*\n([\s\S]*?)```", content
    )
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 2: generic json fenced block
    match = re.search(r"```json\s*\n([\s\S]*?)```", content)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3: any fenced block
    match = re.search(r"```\w*\s*\n([\s\S]*?)```", content)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 4: raw JSON anywhere in content (find first { ... } block)
    match = re.search(r"\{[\s\S]*\}", content)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise EstimatorParseError(
        f"Could not extract JSON from LLM response. Response starts with: {content[:200]!r}"
    )


def _resolve_bin_key(key: str) -> SpeedupBin:
    """Convert a string key to SpeedupBin, accepting either int or name."""
    # Try as integer index
    try:
        return SpeedupBin(int(key))
    except (ValueError, KeyError):
        pass

    # Try as bin name (case-insensitive)
    normalized = key.strip().lower().replace(" ", "_")
    if normalized in _BIN_BY_NAME:
        return _BIN_BY_NAME[normalized]

    raise EstimatorParseError(f"Unrecognized bin key: {key!r}")


def _resolve_confidence(value: str) -> LikertConfidence:
    """Convert a string to LikertConfidence, case-insensitive."""
    normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
    if normalized in _CONFIDENCE_BY_VALUE:
        return _CONFIDENCE_BY_VALUE[normalized]

    raise EstimatorParseError(f"Unrecognized confidence value: {value!r}")


def _parse_bin_confidences(
    raw: dict,
) -> dict[SpeedupBin, LikertConfidence]:
    """Parse and normalize bin_confidences from the raw JSON dict.

    Accepts int or name keys, normalizes confidence values case-insensitively,
    and fills missing non-FAILURE bins with VERY_LOW.
    """
    parsed: dict[SpeedupBin, LikertConfidence] = {}
    for key, value in raw.items():
        speed_bin = _resolve_bin_key(str(key))
        if speed_bin == SpeedupBin.FAILURE:
            continue  # Skip failure bin in confidences
        parsed[speed_bin] = _resolve_confidence(str(value))

    # Fill missing bins with VERY_LOW
    for b in SpeedupBin:
        if b != SpeedupBin.FAILURE and b not in parsed:
            parsed[b] = LikertConfidence.VERY_LOW

    return parsed


def _parse_llm_response(content: str) -> KernelRuntimeEstimate:
    """Parse a full LLM response into a KernelRuntimeEstimate."""
    data = _extract_json_from_response(content)

    # Extract predicted_bin
    if "predicted_bin" not in data:
        raise EstimatorParseError(
            f"Missing 'predicted_bin' in JSON. Keys found: {list(data.keys())}"
        )
    try:
        predicted_bin = SpeedupBin(int(data["predicted_bin"]))
    except (ValueError, KeyError) as exc:
        raise EstimatorParseError(
            f"Invalid predicted_bin value: {data['predicted_bin']!r}"
        ) from exc

    # Extract bin_confidences
    raw_confidences = data.get("bin_confidences", {})
    if not isinstance(raw_confidences, dict):
        raise EstimatorParseError(
            f"bin_confidences must be a dict, got {type(raw_confidences).__name__}"
        )
    bin_confidences = _parse_bin_confidences(raw_confidences)

    # Extract reasoning — use JSON field, fall back to <runtimeCalculation> XML
    reasoning = str(data.get("reasoning", ""))
    if not reasoning.strip():
        xml_match = re.search(
            r"<runtimeCalculation>([\s\S]*?)</runtimeCalculation>", content
        )
        if xml_match:
            reasoning = xml_match.group(1).strip()

    return KernelRuntimeEstimate(
        predicted_bin=predicted_bin,
        bin_confidences=bin_confidences,
        reasoning=reasoning,
    )


class LlmSpeedupEstimator:
    """Estimates kernel speedup by prompting a frontier LLM via LiteLLM."""

    def __init__(
        self,
        model_slug: str = "gemini/gemini-3.1-pro-preview",
        temperature: float = 1.0,
    ) -> None:
        self._model_slug = model_slug
        self._temperature = temperature

    def estimate(
        self, query: KernelRuntimeQuery
    ) -> tuple[KernelRuntimeEstimate, LlmCallUsage | None]:
        system_prompt = _JINJA_ENV.get_template("system.j2").render()
        user_prompt = _JINJA_ENV.get_template("user.j2").render(
            task=query.task,
            reference=query.reference,
            candidate=query.candidate,
            hardware=query.hardware,
        )

        logger.debug(
            "Calling {model} for {op}...",
            model=self._model_slug,
            op=query.task.op_name,
        )

        response = completion(
            model=self._model_slug,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self._temperature,
        )

        content = response.choices[  # pyright: ignore[reportAttributeAccessIssue]
            0
        ].message.content  # pyright: ignore[reportAttributeAccessIssue]
        if content is None:
            raise EstimatorParseError("LLM returned empty content")

        raw_usage = response.usage  # pyright: ignore[reportAttributeAccessIssue]
        llm_usage: LlmCallUsage | None = None
        if raw_usage is not None:
            llm_usage = LlmCallUsage(
                input_tokens=raw_usage.prompt_tokens,
                output_tokens=raw_usage.completion_tokens,
            )

        return _parse_llm_response(content), llm_usage


implements(SpeedupEstimator)(LlmSpeedupEstimator)
