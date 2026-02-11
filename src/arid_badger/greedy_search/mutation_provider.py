from __future__ import annotations

import traceback
from typing import List

from loguru import logger

from arid_badger.typing_utils import implements

from .domain import (
    KernelCandidate,
    MutationAttempt,
    MutationContext,
    MutationError,
    MutationFailure,
    MutationFunction,
    MutationProvider,
    MutationSuccess,
)


def _short_ulid(ulid: object | None) -> str:
    if ulid is None:
        return "none"
    return str(ulid)[:6]


class SerialMutationProvider:
    """Generates kernel mutations serially using a provided mutation function."""

    def __init__(self, mutation_function: MutationFunction) -> None:
        self._mutation_function = mutation_function

    def generate_mutations(
        self,
        context: MutationContext,
    ) -> tuple[List[MutationAttempt], List[KernelCandidate]]:
        attempts: List[MutationAttempt] = []
        generated: List[KernelCandidate] = []
        logger.info(
            "Generating {num_mutations} candidate(s) from parent {parent_short}",
            num_mutations=context.num_mutations,
            parent_short=_short_ulid(context.previous_kernel_ulid),
        )
        for attempt_idx in range(context.num_mutations):
            try:
                mutated = self._mutation_function(context)
                candidate = KernelCandidate(
                    ulid=mutated.ulid,
                    code=mutated.kernel_code,
                    parent_ulid=mutated.ancestor_ulid,
                    evaluation=None,
                )
                attempts.append(
                    MutationSuccess(
                        attempt_idx=attempt_idx, candidate_ulid=candidate.ulid
                    )
                )
                logger.success(
                    "Generated candidate {attempt_idx} ulid={candidate_short}",
                    attempt_idx=attempt_idx,
                    candidate_short=_short_ulid(candidate.ulid),
                )
                generated.append(candidate)
            except Exception as e:
                logger.error(
                    "Mutation failed attempt_idx={attempt_idx} "
                    "parent_ulid={parent_ulid} error={error}",
                    attempt_idx=attempt_idx,
                    parent_ulid=str(context.previous_kernel_ulid),
                    error=repr(e),
                )
                logger.debug(
                    "Mutation failure traceback:\n{traceback}",
                    traceback=traceback.format_exc(),
                )
                attempts.append(
                    MutationFailure(
                        attempt_idx=attempt_idx,
                        error=MutationError(
                            message="Mutation function raised an exception",
                            exception_repr=repr(e),
                            traceback=traceback.format_exc(),
                        ),
                    )
                )

        return attempts, generated


implements(MutationProvider)(SerialMutationProvider)
