"""Depth-First Greedy Search (Hill Climbing) for program optimization."""

from arid_badger.hill_climbing.domain import (
    search,
    resume_search,
    get_archive_statistics,
)
from arid_badger.hill_climbing.checkpoint import (
    Checkpoint,
    CheckpointProvider,
    NoOpCheckpointProvider,
    FileCheckpointProvider,
)

__all__ = [
    "search",
    "resume_search",
    "get_archive_statistics",
    "Checkpoint",
    "CheckpointProvider",
    "NoOpCheckpointProvider",
    "FileCheckpointProvider",
]
