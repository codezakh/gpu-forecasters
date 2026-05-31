"""Parse a tool call's JSON arguments into a domain :class:`KernelRuntimeEstimate`.

Shared between every backend that talks to the surrogate (LiteLLM,
Tinker SamplingClient, RL ``Env`` step). The function is intentionally
narrow:

  - it accepts the tool call's ``arguments`` JSON string as the model
    emitted it;
  - validates against :class:`SubmitEstimateArguments` (the wire-format
    flat model — eight ``p_*`` fields plus ``predicted_bin`` and
    ``reasoning``);
  - **renormalizes the per-bin probabilities unconditionally** so the
    resulting domain :class:`KernelRuntimeEstimate` is a true simplex;
  - surfaces the original (pre-renormalization) sum on the domain
    model as ``raw_probability_sum`` for calibration scoring.

We renormalize unconditionally because the e0137 smoke run showed
that even with a rich, v1-faithful prompt and an explicit "must sum
to 1" instruction, models routinely return sums in ``[0.75, 1.10]``.
A strict-tolerance validator drops ~3% of otherwise-usable
predictions; a renormalize-and-keep-going policy lets calibration
scoring see them all and surface the raw-sum number as a separate
calibration-health signal.
"""

from __future__ import annotations

import json

from pydantic import ValidationError

from .domain import KernelRuntimeEstimate, SpeedupBin, renormalize
from .tool_spec import SubmitEstimateArguments


class EstimatorParseError(Exception):
    """Raised when a tool call's arguments cannot be parsed into a domain estimate."""


def parse_tool_call_args(arguments_json: str) -> KernelRuntimeEstimate:
    """Parse a tool call's ``arguments`` JSON string into a domain estimate.

    Raises :class:`EstimatorParseError` when:
      - the string is not valid JSON;
      - the JSON does not validate against
        :class:`SubmitEstimateArguments` (missing fields, out-of-range
        probabilities, etc.);
      - the eight per-bin probabilities are all zero (cannot renormalize).
    """
    try:
        raw = json.loads(arguments_json)
    except json.JSONDecodeError as exc:
        raise EstimatorParseError(
            f"tool arguments are not valid JSON: {exc}"
        ) from exc

    try:
        wire = SubmitEstimateArguments.model_validate(raw)
    except ValidationError as exc:
        raise EstimatorParseError(
            f"tool arguments failed schema validation: {exc}"
        ) from exc

    raw_probabilities = wire.per_bin_dict()
    try:
        renormalized, raw_sum = renormalize(raw_probabilities)
    except ValueError as exc:
        raise EstimatorParseError(
            f"cannot renormalize per-bin probabilities: {exc}"
        ) from exc

    return KernelRuntimeEstimate(
        predicted_bin=SpeedupBin(int(wire.predicted_bin)),
        bin_probabilities=renormalized,
        reasoning=wire.reasoning,
        raw_probability_sum=raw_sum,
    )
