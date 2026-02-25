"""
Checkpoint infrastructure for Max-Reward PUCT search.

Provides protocols and implementations for saving/loading search state,
allowing interrupted searches to be resumed.
"""

from pathlib import Path
from typing import Dict, List, Set, Optional, Protocol, Generic
import pickle

from pydantic import BaseModel, ConfigDict
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

    model_config = ConfigDict(arbitrary_types_allowed=True)


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

    Uses pickle for serialization, which efficiently handles
    the nested Pydantic model structure.
    """

    def __init__(self, path: Path):
        """
        Initialize provider with a file path.

        Args:
            path: Path where checkpoint will be saved/loaded
        """
        self.path = path

    def save(self, checkpoint: PuctCheckpoint[ObservationT]) -> None:
        """
        Save checkpoint to file using pickle.

        Creates parent directories if they don't exist.
        Overwrites existing checkpoint file.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump(checkpoint, f)

    def load(self) -> Optional[PuctCheckpoint[ObservationT]]:
        """
        Load checkpoint from file.

        Returns:
            Checkpoint if file exists, None otherwise.
        """
        if not self.path.exists():
            return None
        with open(self.path, "rb") as f:
            return pickle.load(f)


implements(PuctCheckpointProvider)(FilePuctCheckpointProvider)
