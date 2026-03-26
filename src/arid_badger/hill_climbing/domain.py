"""
Depth-First Greedy Search (Hill Climbing) for program optimization.

This module implements a simple baseline search algorithm that:
1. Generates mutations of the current program
2. Picks the best mutation (greedy choice)
3. Continues from that best mutation if it improves (depth-first)
4. Otherwise continues sampling from current position
5. Stops when reaching max steps

Uses the same provider-based architecture as max_reward_puct for consistency.
"""

from typing import List, Set, cast, Generic, Union, Annotated, Any
from pydantic import BaseModel, ConfigDict, Field
from typing import TypeVar, TypeGuard
from typing import Literal, Protocol, Optional
from typing import Self
from ulid import ULID


class NoFeedback(BaseModel):
    value: None = None


ObservationT = TypeVar("ObservationT", bound=BaseModel, default=NoFeedback)


class Evaluation(BaseModel, Generic[ObservationT]):
    observation: ObservationT
    reward: Optional[float] = None
    model_config = ConfigDict(frozen=True)


class EvaluationProvider(Protocol[ObservationT]):
    """Evaluates a program and returns its reward.

    Providers must be used as context managers to ensure proper lifecycle
    management. Local providers implement no-op enter/exit; remote providers
    (e.g. Modal) use them to open and close the remote session:

        with Provider(reference_kernel_code=ref_code) as provider:
            best = search(..., evaluation_provider=provider)
    """

    def evaluate(self, program_code: str) -> Evaluation[ObservationT]:
        """Returns reward (or None if evaluation failed)."""
        ...

    def __enter__(self) -> Self: ...

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None: ...


class MutationProvider(Protocol[ObservationT]):
    def generate_mutations(
        self,
        program_code: str,
        num_mutations: int,
        evaluation: Evaluation[ObservationT],
    ) -> List[str]: ...


class Node(BaseModel, Generic[ObservationT]):
    """
    Represents a specific program/kernel solution.
    """

    program_code: str

    # Ancestor chain: list of {"id": str, "timestep": int} dicts,
    # most recent parent first.  Matches State.parents.
    ancestors: List[ULID]

    # Evaluation of the program code.
    evaluation: Evaluation[ObservationT]

    ulid: ULID = Field(default_factory=ULID)
    is_seed: bool = False


class Checkpoint(BaseModel, Generic[ObservationT]):
    archive: List[Node[ObservationT]]
    visited: Set[str]
    current_step: int


class CheckpointProvider(Protocol[ObservationT]):
    def save(self, checkpoint: Checkpoint[ObservationT]): ...
    def load(self) -> Optional[Checkpoint[ObservationT]]: ...
