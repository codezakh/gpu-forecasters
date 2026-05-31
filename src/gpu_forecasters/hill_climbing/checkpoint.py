"""
Checkpoint infrastructure for hill climbing search.

Provides protocols and implementations for saving/loading search state,
allowing interrupted searches to be resumed.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Optional, Protocol, Generic
import pickle
from .domain import Checkpoint, CheckpointProvider, ObservationT
from arid_badger.typing_utils import implements


class NoOpCheckpointProvider(CheckpointProvider[ObservationT]):
    """
    Provider that doesn't save or load anything.

    This is the default provider when no checkpointing is needed.
    """

    def save(self, checkpoint: Checkpoint[ObservationT]) -> None:
        """No-op save."""
        pass

    def load(self) -> Optional[Checkpoint[ObservationT]]:
        """Always returns None."""
        return None


implements(CheckpointProvider)(NoOpCheckpointProvider)


class FileCheckpointProvider(CheckpointProvider[ObservationT]):
    """
    Provider that saves/loads checkpoints to/from a file.

    Uses pickle for serialization, which efficiently handles
    the nested dataclass structure and Set[str].
    """

    def __init__(self, path: Path):
        """
        Initialize provider with a file path.

        Args:
            path: Path where checkpoint will be saved/loaded
        """
        self.path = path

    def save(self, checkpoint: Checkpoint[ObservationT]) -> None:
        """
        Save checkpoint to file using pickle.

        Creates parent directories if they don't exist.
        Overwrites existing checkpoint file.
        """
        # Ensure parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Write checkpoint using pickle
        with open(self.path, "wb") as f:
            pickle.dump(checkpoint, f)

    def load(self) -> Optional[Checkpoint[ObservationT]]:
        """
        Load checkpoint from file.

        Returns:
            Checkpoint if file exists, None otherwise.
        """
        if not self.path.exists():
            return None

        with open(self.path, "rb") as f:
            return pickle.load(f)


implements(CheckpointProvider)(FileCheckpointProvider)
