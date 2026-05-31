"""Archive-admission action DSL.

The env consults an ``AdmissionPolicy`` after every rollout to decide
what to do with the resulting ``Candidate``. The policy returns one of
these typed actions; the env (or a thin applier) translates it into a
single ``archive.credit_rollout(...)`` call.

Keeping the decision as a typed value rather than a direct archive
mutation means the policy is a pure function of ``(candidate, parent)``
and can be unit-tested without any archive at all.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class AdmitChild(BaseModel):
    """Register the candidate as a node in the archive and credit the
    rollout to its parent's subtree."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["admit"] = "admit"


class CreditOnly(BaseModel):
    """Credit the rollout to the parent's subtree (visit-count bookkeeping)
    but do not register the candidate as a node. The candidate is still
    serialized into ``rollouts.jsonl`` by the sink — only its presence in
    the live search tree is elided."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["credit_only"] = "credit_only"


AdmissionDecision = Annotated[
    Union[AdmitChild, CreditOnly],
    Field(discriminator="kind"),
]
