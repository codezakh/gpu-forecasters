"""
Checkpoint infrastructure for hill climbing search.

Provides protocols and implementations for saving/loading search state,
allowing interrupted searches to be resumed.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Optional, Protocol
import pickle

from arid_badger.max_reward_puct.domain import Node


@dataclass
class Checkpoint:
    """
    Minimal state needed to resume a hill climbing search.

    Contains only the evolving state during search:
    - current_node: The current node being explored
    - best_node: The best node found so far
    - archive: List of all nodes explored
    - visited: Set of content keys for deduplication
    - current_step: Current iteration step
    """

    current_node: Node
    best_node: Node
    archive: List[Node]
    visited: Set[str]
    current_step: int


class CheckpointProvider(Protocol):
    """
    Protocol for checkpoint providers.

    Implementations can use different serialization strategies
    (pickle, JSON, binary formats, etc.) and storage backends
    (files, databases, cloud storage, etc.).
    """

    def save(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint to the provider's configured location."""
        ...

    def load(self) -> Optional[Checkpoint]:
        """
        Load checkpoint from the provider's configured location.

        Returns:
            Checkpoint if one exists, None otherwise.
        """
        ...


class NoOpCheckpointProvider:
    """
    Provider that doesn't save or load anything.

    This is the default provider when no checkpointing is needed.
    """

    def save(self, checkpoint: Checkpoint) -> None:
        """No-op save."""
        pass

    def load(self) -> Optional[Checkpoint]:
        """Always returns None."""
        return None


class FileCheckpointProvider:
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

    def save(self, checkpoint: Checkpoint) -> None:
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

    def load(self) -> Optional[Checkpoint]:
        """
        Load checkpoint from file.

        Returns:
            Checkpoint if file exists, None otherwise.
        """
        if not self.path.exists():
            return None

        with open(self.path, "rb") as f:
            return pickle.load(f)
