"""
Checkpoint infrastructure for Max-Reward PUCT search.

Provides protocols and implementations for saving/loading search state,
allowing interrupted searches to be resumed.
"""

from pathlib import Path
from typing import Dict, List, Set, Optional, Protocol, Generic

from pydantic import BaseModel, TypeAdapter
from ulid import ULID

from arid_badger.hill_climbing.domain import Node, ObservationT
from arid_badger.typing_utils import implements


class PuctCheckpoint(BaseModel, Generic[ObservationT]):
    archive: List[Node[ObservationT]]
    seed_ids: Set[ULID]
    visit_counts: Dict[ULID, int]
    best_child_rewards: Dict[ULID, float]
    global_expansion_count: int
    current_step: int


class PuctCheckpointProvider(Protocol[ObservationT]):
    def save(self, checkpoint: PuctCheckpoint[ObservationT]) -> None: ...
    def load(self) -> Optional[PuctCheckpoint[ObservationT]]: ...


class NoOpPuctCheckpointProvider(PuctCheckpointProvider[ObservationT]):
    """
    Provider that doesn't save or load anything.

    This is the default provider when no checkpointing is needed.
    """

    def save(self, checkpoint: PuctCheckpoint[ObservationT]) -> None:
        """No-op save."""
        pass

    def load(self) -> Optional[PuctCheckpoint[ObservationT]]:
        """Always returns None."""
        return None


implements(PuctCheckpointProvider)(NoOpPuctCheckpointProvider)


class FilePuctCheckpointProvider(PuctCheckpointProvider[ObservationT]):
    """
    Provider that saves/loads checkpoints to/from a file.

    Uses JSON via Pydantic's model_dump_json()/model_validate_json() for
    human-readable, safe serialization.
    """

    def __init__(
        self,
        path: Path,
        checkpoint_type: type[PuctCheckpoint[ObservationT]] = PuctCheckpoint,
    ):
        """
        Initialize provider with a file path.

        Args:
            path: Path where checkpoint will be saved/loaded
            checkpoint_type: Concrete generic type used for deserialization,
                e.g. PuctCheckpoint[KernelBenchObservation]
        """
        self.path = path
        self._checkpoint_type = checkpoint_type
        # Use TypeAdapter rather than checkpoint.model_dump_json() so that
        # serialization uses the parametrized schema from checkpoint_type.
        # Without this, an unparametrized PuctCheckpoint(...) construction
        # (as in _search_impl) causes Pydantic to fall back to the TypeVar
        # default schema (NoFeedback), silently dropping all observation fields.
        self._type_adapter: TypeAdapter[PuctCheckpoint[ObservationT]] = TypeAdapter(
            checkpoint_type
        )

    def save(self, checkpoint: PuctCheckpoint[ObservationT]) -> None:
        """
        Save checkpoint to file as JSON.

        Creates parent directories if they don't exist.
        Overwrites existing checkpoint file.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            f.write(self._type_adapter.dump_json(checkpoint))

    def load(self) -> Optional[PuctCheckpoint[ObservationT]]:
        """
        Load checkpoint from file.

        Returns:
            Checkpoint if file exists, None otherwise.
        """
        if not self.path.exists():
            return None
        with open(self.path, "rb") as f:
            return self._type_adapter.validate_json(f.read())


implements(PuctCheckpointProvider)(FilePuctCheckpointProvider)
