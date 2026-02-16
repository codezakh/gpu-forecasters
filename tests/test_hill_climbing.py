"""
Tests for Depth-First Greedy Search (Hill Climbing) algorithm.

Uses Binary String test environment from test_max_reward_puct.py:
- BinaryStringMutationProvider: Flips single bits in binary string
- BinaryStringEvaluationProvider: Returns decimal value as reward
- FailingEvaluationProvider: Always returns None to test failure handling
"""

import random
import tempfile
from pathlib import Path
from typing import List, Optional

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
from arid_badger.max_reward_puct.domain import Node, get_content_key, set_parent_info


# Test Providers (reused from PUCT tests)


class BinaryStringMutationProvider:
    """Toy mutation provider: flips single bits in binary string."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def generate_mutations(
        self, program_code: str, num_mutations: int
    ) -> List[str]:
        """Generate mutations by flipping bit positions."""
        mutations = []
        n = len(program_code)
        # Sample positions without replacement, up to the string length
        num_to_sample = min(num_mutations, n)
        positions = self.rng.sample(range(n), num_to_sample)

        for pos in positions:
            mutated = list(program_code)
            mutated[pos] = "1" if mutated[pos] == "0" else "0"
            mutations.append("".join(mutated))

        return mutations


class BinaryStringEvaluationProvider:
    """Toy evaluation provider: returns decimal value of binary string."""

    def evaluate(self, program_code: str) -> Optional[float]:
        """Return decimal value of binary string."""
        try:
            return float(int(program_code, 2))
        except ValueError:
            return None


class FailingEvaluationProvider:
    """Provider that always returns None to test failed evaluation handling."""

    def evaluate(self, program_code: str) -> Optional[float]:
        return None


# Tests


def test_search_converges_to_maximum():
    """Test that greedy search finds optimal binary string."""
    mutation_provider = BinaryStringMutationProvider(seed=42)
    evaluation_provider = BinaryStringEvaluationProvider()

    result = search(
        initial_program="0000",
        max_depth=20,
        samples_per_node=4,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
    )

    # Should find the maximum (all 1s) = 15.0
    assert result.program_code == "1111"
    assert result.reward == 15.0


def test_greedy_selection_picks_best():
    """Test that greedy choice always picks highest reward at each step."""
    # Use a deterministic mutation provider that generates all single-bit flips
    class DeterministicMutationProvider:
        def generate_mutations(self, program_code: str, num_mutations: int) -> List[str]:
            """Generate all single-bit flips."""
            mutations = []
            for i in range(len(program_code)):
                mutated = list(program_code)
                mutated[i] = "1" if mutated[i] == "0" else "0"
                mutations.append("".join(mutated))
            return mutations[:num_mutations]

    mutation_provider = DeterministicMutationProvider()
    evaluation_provider = BinaryStringEvaluationProvider()

    result = search(
        initial_program="0000",  # reward = 0
        max_depth=2,
        samples_per_node=4,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
    )

    # After step 1: should pick "1000" (8) over "0100" (4), "0010" (2), "0001" (1)
    # After step 2: should pick "1100" (12) or "1010" (10) or "1001" (9) - all > 8
    assert result.reward is not None
    assert result.reward >= 8.0  # Should improve from initial


def test_stops_at_local_maximum():
    """Test that search stops when no improvement is found."""
    # Create a provider that can only generate worse or equal mutations
    class LocalMaximumMutationProvider:
        def __init__(self):
            self.call_count = 0

        def generate_mutations(self, program_code: str, num_mutations: int) -> List[str]:
            self.call_count += 1
            if self.call_count == 1:
                # First call: generate better mutations
                return ["0001", "0010", "0011"]
            else:
                # Subsequent calls: only generate worse mutations
                return ["0000", "0001"]

    mutation_provider = LocalMaximumMutationProvider()
    evaluation_provider = BinaryStringEvaluationProvider()

    result = search(
        initial_program="0000",
        max_depth=10,
        samples_per_node=3,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
    )

    # Should stop after finding local maximum at "0011" (3)
    assert result.program_code == "0011"
    assert result.reward == 3.0
    # Should not make all 10 iterations
    assert mutation_provider.call_count <= 3


def test_deduplication_works():
    """Test that duplicate programs are skipped."""
    # Provider that generates duplicate mutations
    class DuplicateMutationProvider:
        def generate_mutations(self, program_code: str, num_mutations: int) -> List[str]:
            # Always return the same mutation multiple times
            return ["0001"] * num_mutations

    mutation_provider = DuplicateMutationProvider()
    evaluation_provider = BinaryStringEvaluationProvider()

    result = search(
        initial_program="0000",
        max_depth=5,
        samples_per_node=10,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
    )

    # Should only explore "0000" -> "0001", then stop (no new mutations)
    assert result.program_code == "0001"
    assert result.reward == 1.0


def test_handles_failed_evaluations():
    """Test that None rewards are filtered out correctly."""
    # Provider that generates mix of valid and invalid mutations
    class MixedMutationProvider:
        def generate_mutations(self, program_code: str, num_mutations: int) -> List[str]:
            # Return mix of valid binary strings and invalid strings
            return ["0001", "invalid", "0010", "also_invalid"]

    mutation_provider = MixedMutationProvider()
    evaluation_provider = BinaryStringEvaluationProvider()

    result = search(
        initial_program="0000",
        max_depth=5,
        samples_per_node=4,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
    )

    # Should pick best valid mutation ("0010" = 2) and ignore invalid ones
    assert result.reward is not None
    assert result.reward >= 1.0  # Should find at least "0001" or "0010"


def test_tracks_parent_child_relationships():
    """Test that ancestor chain is set correctly."""
    mutation_provider = BinaryStringMutationProvider(seed=42)
    evaluation_provider = BinaryStringEvaluationProvider()

    result = search(
        initial_program="0000",
        max_depth=3,
        samples_per_node=4,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
    )

    # Result should have ancestors (at least the initial seed)
    # After 3 steps, should have ancestors: [parent_2, parent_1, initial_seed]
    assert len(result.ancestors) > 0
    # Verify ancestors are properly linked
    assert result.is_seed is False  # Result should not be the seed


def test_archive_contains_explored_nodes():
    """Test that all valid children are tracked (not used for selection like PUCT)."""
    # We don't expose archive in the API, but we can verify behavior indirectly
    # by checking that the search doesn't revisit the same nodes
    visited_programs = []

    class TrackingMutationProvider:
        def generate_mutations(self, program_code: str, num_mutations: int) -> List[str]:
            visited_programs.append(program_code)
            mutations = []
            n = len(program_code)
            for i in range(min(num_mutations, n)):
                mutated = list(program_code)
                mutated[i] = "1" if mutated[i] == "0" else "0"
                mutations.append("".join(mutated))
            return mutations

    mutation_provider = TrackingMutationProvider()
    evaluation_provider = BinaryStringEvaluationProvider()

    search(
        initial_program="0000",
        max_depth=5,
        samples_per_node=4,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
    )

    # Should not mutate the same program twice
    assert len(visited_programs) == len(set(visited_programs))


def test_returns_global_best_not_just_final():
    """Test that search returns best node from entire search, not just final node."""
    # Create a scenario where we climb up then down
    class PeakMutationProvider:
        def __init__(self):
            self.step = 0

        def generate_mutations(self, program_code: str, num_mutations: int) -> List[str]:
            self.step += 1
            if self.step == 1:
                # First step: go to high value
                return ["1111"]  # 15
            elif self.step == 2:
                # Second step: go to lower value
                return ["0111"]  # 7
            else:
                # Continue going down
                return ["0011"]  # 3

    mutation_provider = PeakMutationProvider()
    evaluation_provider = BinaryStringEvaluationProvider()

    result = search(
        initial_program="0000",
        max_depth=10,
        samples_per_node=1,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
    )

    # Should return the peak (1111 = 15), not the final node
    assert result.program_code == "1111"
    assert result.reward == 15.0


def test_stops_at_max_depth():
    """Test that max_depth parameter is respected."""
    mutation_provider = BinaryStringMutationProvider(seed=42)
    evaluation_provider = BinaryStringEvaluationProvider()

    call_count = 0

    class CountingMutationProvider:
        def generate_mutations(self, program_code: str, num_mutations: int) -> List[str]:
            nonlocal call_count
            call_count += 1
            # Always generate valid mutations so we don't stop early
            return BinaryStringMutationProvider(seed=42).generate_mutations(
                program_code, num_mutations
            )

    mutation_provider = CountingMutationProvider()

    search(
        initial_program="0000",
        max_depth=5,
        samples_per_node=4,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
    )

    # Should call generate_mutations at most max_depth times
    assert call_count <= 5


def test_handles_all_failures():
    """Test that search handles gracefully when all evaluations fail."""
    mutation_provider = BinaryStringMutationProvider(seed=42)
    failing_provider = FailingEvaluationProvider()

    result = search(
        initial_program="0000",
        max_depth=5,
        samples_per_node=4,
        mutation_provider=mutation_provider,
        evaluation_provider=failing_provider,
    )

    # Should return the initial program (with None reward)
    assert result.program_code == "0000"
    assert result.reward is None
    assert result.is_seed is True


def test_handles_initial_program_failure():
    """Test that search handles when even the initial program fails."""
    mutation_provider = BinaryStringMutationProvider(seed=42)
    failing_provider = FailingEvaluationProvider()

    result = search(
        initial_program="invalid",
        max_depth=5,
        samples_per_node=4,
        mutation_provider=mutation_provider,
        evaluation_provider=failing_provider,
    )

    # Should return the initial program with None reward
    assert result.program_code == "invalid"
    assert result.reward is None


def test_single_iteration():
    """Test that search works with max_depth=1."""
    mutation_provider = BinaryStringMutationProvider(seed=42)
    evaluation_provider = BinaryStringEvaluationProvider()

    result = search(
        initial_program="0000",
        max_depth=1,
        samples_per_node=4,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
    )

    # Should make one improvement step
    assert result.reward is not None
    assert result.reward > 0.0  # Should be better than initial


def test_get_archive_statistics():
    """Test the helper function for analyzing archive."""
    # Create a simple mock archive
    node1 = Node(program_code="0000", reward=0.0, ancestors=[])
    node2 = Node(program_code="0001", reward=1.0, ancestors=[])
    node3 = Node(program_code="invalid", reward=None, ancestors=[])
    node4 = Node(program_code="0001", reward=1.0, ancestors=[])  # duplicate code

    archive = [node1, node2, node3, node4]

    stats = get_archive_statistics(archive)

    assert stats["total_nodes"] == 4
    assert stats["valid_nodes"] == 3
    assert stats["failed_nodes"] == 1
    assert stats["best_reward"] == 1.0
    assert stats["unique_programs"] == 3  # "0000", "0001", "invalid"


def test_zero_max_depth():
    """Test edge case with max_depth=0."""
    mutation_provider = BinaryStringMutationProvider(seed=42)
    evaluation_provider = BinaryStringEvaluationProvider()

    result = search(
        initial_program="0000",
        max_depth=0,
        samples_per_node=4,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
    )

    # Should just return the initial program
    assert result.program_code == "0000"
    assert result.reward == 0.0
    assert result.is_seed is True


def test_content_key_deduplication():
    """Test that get_content_key works correctly for deduplication."""
    node1 = Node(program_code="0001", reward=1.0, ancestors=[])
    node2 = Node(program_code="0001", reward=2.0, ancestors=[])  # same code, different reward
    node3 = Node(program_code="0010", reward=1.0, ancestors=[])

    key1 = get_content_key(node1)
    key2 = get_content_key(node2)
    key3 = get_content_key(node3)

    # Same code should produce same key
    assert key1 == key2
    # Different code should produce different key
    assert key1 != key3


def test_set_parent_info_creates_ancestor_chain():
    """Test that set_parent_info correctly builds ancestor chain."""
    grandparent = Node(program_code="0000", reward=0.0, ancestors=[], is_seed=True)
    parent = Node(program_code="0001", reward=1.0, ancestors=[])
    child = Node(program_code="0011", reward=3.0, ancestors=[])

    set_parent_info(parent, grandparent)
    set_parent_info(child, parent)

    # Parent should have grandparent as ancestor
    assert len(parent.ancestors) == 1
    assert parent.ancestors[0].ulid == grandparent.ulid

    # Child should have parent and grandparent as ancestors
    assert len(child.ancestors) == 2
    assert child.ancestors[0].ulid == parent.ulid
    assert child.ancestors[1].ulid == grandparent.ulid


# Checkpoint Tests


def test_no_op_provider_default():
    """Test that search works without checkpointing (default behavior)."""
    mutation_provider = BinaryStringMutationProvider(seed=42)
    evaluation_provider = BinaryStringEvaluationProvider()

    # Should work without explicit checkpoint provider
    result = search(
        initial_program="0000",
        max_depth=5,
        samples_per_node=4,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
    )

    assert result.reward is not None
    assert result.reward > 0.0


def test_checkpoint_save_called():
    """Test that checkpoint provider.save() is called after each iteration."""
    # Create a mock provider that tracks save calls
    class MockCheckpointProvider:
        def __init__(self):
            self.save_count = 0
            self.saved_checkpoints: List[Checkpoint] = []

        def save(self, checkpoint: Checkpoint) -> None:
            self.save_count += 1
            self.saved_checkpoints.append(checkpoint)

        def load(self) -> Optional[Checkpoint]:
            return None

    mutation_provider = BinaryStringMutationProvider(seed=42)
    evaluation_provider = BinaryStringEvaluationProvider()
    mock_provider = MockCheckpointProvider()

    result = search(
        initial_program="0000",
        max_depth=3,
        samples_per_node=4,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
        checkpoint_provider=mock_provider,
    )

    # Should save after each iteration (depth 0->1, 1->2, 2->3)
    assert mock_provider.save_count >= 1
    assert len(mock_provider.saved_checkpoints) >= 1

    # Verify last checkpoint has correct depth
    last_checkpoint = mock_provider.saved_checkpoints[-1]
    assert last_checkpoint.current_depth > 0


def test_resume_search_continues():
    """Test that resume_search continues from checkpoint state."""
    # Create a mock provider that captures the checkpoint
    class CapturingCheckpointProvider:
        def __init__(self):
            self.captured_checkpoint: Optional[Checkpoint] = None

        def save(self, checkpoint: Checkpoint) -> None:
            # Capture checkpoint after first iteration
            if checkpoint.current_depth == 1:
                self.captured_checkpoint = checkpoint

        def load(self) -> Optional[Checkpoint]:
            return None

    mutation_provider = BinaryStringMutationProvider(seed=42)
    evaluation_provider = BinaryStringEvaluationProvider()
    capturing_provider = CapturingCheckpointProvider()

    # Run initial search and capture checkpoint
    result1 = search(
        initial_program="0000",
        max_depth=2,
        samples_per_node=4,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
        checkpoint_provider=capturing_provider,
    )

    # Verify we captured a checkpoint
    assert capturing_provider.captured_checkpoint is not None
    checkpoint = capturing_provider.captured_checkpoint
    assert checkpoint.current_depth == 1

    # Reset mutation provider with same seed for reproducibility
    mutation_provider = BinaryStringMutationProvider(seed=42)

    # Resume from checkpoint
    result2 = resume_search(
        checkpoint=checkpoint,
        max_depth=3,  # Continue for 2 more iterations (depth 1 -> 3)
        samples_per_node=4,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
    )

    # Result should have improved or stayed same (depending on local maximum)
    assert result2.reward is not None
    assert checkpoint.best_node.reward is not None
    assert result2.reward >= checkpoint.best_node.reward


def test_checkpoint_serialization():
    """Test FileCheckpointProvider save/load roundtrip."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "checkpoint.pkl"
        provider = FileCheckpointProvider(checkpoint_path)

        mutation_provider = BinaryStringMutationProvider(seed=42)
        evaluation_provider = BinaryStringEvaluationProvider()

        # Run search with file checkpoint provider
        result = search(
            initial_program="0000",
            max_depth=2,
            samples_per_node=4,
            mutation_provider=mutation_provider,
            evaluation_provider=evaluation_provider,
            checkpoint_provider=provider,
        )

        # Verify checkpoint file was created
        assert checkpoint_path.exists()

        # Load checkpoint
        loaded_checkpoint = provider.load()
        assert loaded_checkpoint is not None
        assert loaded_checkpoint.current_depth == 2
        assert loaded_checkpoint.best_node.reward == result.reward

        # Verify visited set was preserved
        assert len(loaded_checkpoint.visited) > 0

        # Verify archive was preserved
        assert len(loaded_checkpoint.archive) > 0


def test_resume_skips_initial_evaluation():
    """Test that resume doesn't re-evaluate the current node."""
    # Create a tracking evaluation provider
    class TrackingEvaluationProvider:
        def __init__(self):
            self.evaluated_programs: List[str] = []

        def evaluate(self, program_code: str) -> Optional[float]:
            self.evaluated_programs.append(program_code)
            try:
                return float(int(program_code, 2))
            except ValueError:
                return None

    # Run initial search to depth 1
    mutation_provider = BinaryStringMutationProvider(seed=42)
    tracking_provider = TrackingEvaluationProvider()

    result1 = search(
        initial_program="0000",
        max_depth=1,
        samples_per_node=4,
        mutation_provider=mutation_provider,
        evaluation_provider=tracking_provider,
    )

    # Create checkpoint from result
    checkpoint = Checkpoint(
        current_node=result1,
        best_node=result1,
        archive=[result1],
        visited={get_content_key(result1)},
        current_depth=1,
    )

    # Reset tracking
    tracking_provider2 = TrackingEvaluationProvider()
    mutation_provider2 = BinaryStringMutationProvider(seed=100)

    # Resume search
    resume_search(
        checkpoint=checkpoint,
        max_depth=2,
        samples_per_node=4,
        mutation_provider=mutation_provider2,
        evaluation_provider=tracking_provider2,
    )

    # The current node from checkpoint should NOT be re-evaluated
    # Only new mutations should be evaluated
    assert result1.program_code not in tracking_provider2.evaluated_programs


def test_file_checkpoint_provider_no_file():
    """Test that FileCheckpointProvider returns None when file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "nonexistent.pkl"
        provider = FileCheckpointProvider(checkpoint_path)

        loaded = provider.load()
        assert loaded is None


def test_file_checkpoint_provider_creates_directory():
    """Test that FileCheckpointProvider creates parent directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "nested" / "dir" / "checkpoint.pkl"
        provider = FileCheckpointProvider(checkpoint_path)

        # Create a simple checkpoint
        node = Node(program_code="0000", reward=0.0, ancestors=[], is_seed=True)
        checkpoint = Checkpoint(
            current_node=node,
            best_node=node,
            archive=[node],
            visited={"0000"},
            current_depth=0,
        )

        # Save should create nested directories
        provider.save(checkpoint)
        assert checkpoint_path.exists()
        assert checkpoint_path.parent.exists()


def test_checkpoint_preserves_archive_and_visited():
    """Test that checkpoint correctly preserves archive and visited state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "checkpoint.pkl"
        provider = FileCheckpointProvider(checkpoint_path)

        mutation_provider = BinaryStringMutationProvider(seed=42)
        evaluation_provider = BinaryStringEvaluationProvider()

        # Run search to build up archive and visited
        result = search(
            initial_program="0000",
            max_depth=3,
            samples_per_node=4,
            mutation_provider=mutation_provider,
            evaluation_provider=evaluation_provider,
            checkpoint_provider=provider,
        )

        # Load checkpoint
        loaded_checkpoint = provider.load()
        assert loaded_checkpoint is not None

        # Archive should contain multiple nodes
        assert len(loaded_checkpoint.archive) > 1

        # Visited should contain multiple programs
        assert len(loaded_checkpoint.visited) > 1

        # Reset and resume - should not re-evaluate visited programs
        mutation_provider2 = BinaryStringMutationProvider(seed=42)
        evaluation_provider2 = BinaryStringEvaluationProvider()

        result2 = resume_search(
            checkpoint=loaded_checkpoint,
            max_depth=5,
            samples_per_node=4,
            mutation_provider=mutation_provider2,
            evaluation_provider=evaluation_provider2,
            checkpoint_provider=provider,
        )

        # Should continue from checkpoint
        assert result2.reward is not None
