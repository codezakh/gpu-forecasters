"""Domain types for the theory builder.

Pure data — no LLM, no Modal, no I/O. The shapes here form the contract
between the builder, the worker, and the event-sourced driver.

Design choices:

* **Markdown world model.** ``WorldModel.text`` is a single markdown
  string. Per the spec's open question, option (1) ("markdown + standard
  diff format") is what's implemented in v1; the structure (Established
  Beliefs / Working Hypotheses / Open Questions / Anomalies) is a
  *prompt convention*, not a typed invariant. Diffs are ``SEARCH``/
  ``REPLACE`` blocks applied by ``diff.py``.
* **Hypotheses are ULID-keyed.** The id is what ties an experiment
  result and an explanation back to the hypothesis that drove them; it
  is also what the LLM uses to reference an entry inside the world
  model markdown.
* **Status lifecycle.** ``open`` is the bootstrap state. The builder
  may flip a hypothesis to ``under_investigation`` when it picks it up,
  to ``closed`` (rejected by evidence) or ``established`` (confirmed)
  after the explanation step. The reducer trusts the builder's status
  string; it's also encoded in the prompt convention.
* **Inner-search atom.** ``ExperimentTrial`` is one ``(code, runtime,
  reward, observation)`` tuple. ``ExperimentResult`` is the bag of
  trials a single inner search produced for one hypothesis, plus the
  best trial (for cheap downstream summarisation).
"""

from __future__ import annotations

from typing import Generic, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field
from ulid import ULID

from arid_badger.hill_climbing.domain import Evaluation, ObservationT


HypothesisStatus = Literal[
    "open",
    "under_investigation",
    "closed",
    "established",
]
"""Lifecycle of a hypothesis. The builder transitions between these
via world-model diffs; the reducer doesn't enforce the transitions
(option (1) — see the spec). String form is what the LLM emits."""


class Hypothesis(BaseModel):
    """A claim + intervention + prediction the inner search will test.

    The three textual fields are load-bearing — they're what an
    explanation gets compared against. Vague hypotheses produce
    vacuous explanations; that constraint lives in the prompt.

    ``code_references`` is a free-form list of strings the builder uses
    to anchor itself to specific lines / tile sizes / access patterns
    in the kernel under study. Treated as opaque text by the reducer.
    """

    model_config = ConfigDict(frozen=True)

    id: ULID = Field(default_factory=ULID)

    bottleneck: str
    """One-paragraph claim about what is currently bottlenecking the kernel."""

    intervention: str
    """The proposed change + the mechanism by which it should help."""

    prediction: str
    """A quantitative or directional prediction about what the
    benchmark will show if the claim is right."""

    code_references: list[str] = Field(default_factory=list)
    """Specific code anchors (line numbers, tile sizes, access
    patterns). Free-form strings."""

    status: HypothesisStatus = "open"


class ExperimentTrial(BaseModel, Generic[ObservationT]):
    """One ``(code, evaluation)`` pair the inner search produced.

    ``runtime_ns`` is a denormalised view onto the inner search's
    benchmark output that the builder's renderer can consume without
    re-deriving from ``observation``. ``None`` if the trial failed
    (eval error, incorrect, infrastructure failure)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    code: str
    evaluation: Evaluation[ObservationT]


class ExperimentResult(BaseModel, Generic[ObservationT]):
    """The bag of trials a single inner search produced for one
    hypothesis, plus a denormalised pointer at the best trial."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    hypothesis_id: ULID
    trials: list[ExperimentTrial[ObservationT]]

    @property
    def best_trial(self) -> Optional[ExperimentTrial[ObservationT]]:
        """Highest-reward valid trial, or ``None`` if none scored."""
        valid = [t for t in self.trials if t.evaluation.reward is not None]
        if not valid:
            return None
        return max(valid, key=lambda t: t.evaluation.reward or float("-inf"))

    @property
    def num_trials(self) -> int:
        return len(self.trials)

    @property
    def num_valid_trials(self) -> int:
        return sum(1 for t in self.trials if t.evaluation.reward is not None)


class WorldModelDiff(BaseModel):
    """One ``SEARCH``/``REPLACE`` block.

    The applier (``diff.py``) requires that ``search`` matches exactly
    one location in the document — no fuzzy matching. ``search`` may
    be empty, in which case ``replace`` is appended to the end of the
    document (the convention used to seed brand-new sections)."""

    model_config = ConfigDict(frozen=True)

    search: str
    replace: str


class Explanation(BaseModel):
    """The builder's read of an experiment result.

    Carries everything needed to update the world model: a textual
    explanation of the gap between prediction and observation, plus the
    diff(s) the builder wants applied to ``WorldModel.text``."""

    model_config = ConfigDict(frozen=True)

    hypothesis_id: ULID

    gap: str
    """Where the prediction and the observation diverged (or
    converged). One short paragraph."""

    mechanism: str
    """Proposed mechanism that explains the gap. May be 'no gap; the
    prediction was confirmed'."""

    belief_update: str
    """A specific belief update committed back to the world model.
    Should be reflected in ``diffs`` below — this field is the prose
    version, the diffs are the structural version."""

    diffs: list[WorldModelDiff]
    """Concrete edits to apply to ``WorldModel.text``. Empty is allowed
    when the experiment confirms the existing world model with no
    update needed."""


class WorldModel(BaseModel):
    """The kernel under study + the prose world model.

    ``text`` is markdown. The four spec sections (Established Beliefs,
    Working Hypotheses, Open Questions, Anomalies) are conventions that
    the prompt teaches and the diff applier preserves — they're not
    typed properties of this object.

    ``kernel_description`` is a short prose description of the kernel,
    used to seed the builder's prompt. It does not change over a run.
    """

    model_config = ConfigDict(frozen=True)

    kernel_description: str
    text: str = ""

    def with_text(self, new_text: str) -> "WorldModel":
        return WorldModel(
            kernel_description=self.kernel_description, text=new_text
        )


__all__ = [
    "Hypothesis",
    "HypothesisStatus",
    "ExperimentTrial",
    "ExperimentResult",
    "Explanation",
    "WorldModel",
    "WorldModelDiff",
]
