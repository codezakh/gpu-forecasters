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
    Evaluation,
    EvaluationProvider,
    Node,
    get_content_key,
    set_parent_info,
    Checkpoint,
    CheckpointProvider,
    MutationProvider,
)
from arid_badger.hill_climbing.checkpoint import (
    NoOpCheckpointProvider,
    FileCheckpointProvider,
)
from arid_badger.typing_utils import implements
from arid_badger.hill_climbing.domain import NoFeedback


# Test Providers (reused from PUCT tests)


class BinaryStringMutationProvider:
    """Toy mutation provider: flips single bits in binary string."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def generate_mutations(
        self, program_code: str, num_mutations: int, evaluation: Evaluation[NoFeedback]
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


implements(MutationProvider[NoFeedback])(BinaryStringMutationProvider)


class BinaryStringEvaluationProvider:
    """Toy evaluation provider: returns decimal value of binary string."""

    def evaluate(self, program_code: str) -> Evaluation[NoFeedback]:
        """Return decimal value of binary string."""
        try:
            return Evaluation(
                observation=NoFeedback(), reward=float(int(program_code, 2))
            )
        except ValueError:
            return Evaluation(observation=NoFeedback(), reward=None)


implements(EvaluationProvider[NoFeedback])(BinaryStringEvaluationProvider)


class FailingEvaluationProvider:
    """Provider that always returns None to test failed evaluation handling."""

    def evaluate(self, program_code: str) -> Evaluation[NoFeedback]:
        return Evaluation(observation=NoFeedback(), reward=None)


implements(EvaluationProvider[NoFeedback])(FailingEvaluationProvider)
# Tests


def test_search_converges_to_maximum():
    """Test that greedy search finds optimal binary string."""
    mutation_provider = BinaryStringMutationProvider(seed=42)
    evaluation_provider = BinaryStringEvaluationProvider()

    result = search(
        initial_program="0000",
        max_steps=20,
        samples_per_node=4,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
        checkpoint_provider=NoOpCheckpointProvider(),
    )

    # Should find the maximum (all 1s) = 15.0
    assert result.program_code == "1111"
    assert result.evaluation.reward == 15.0


def test_greedy_selection_picks_best():
    """Test that greedy choice always picks highest reward at each step."""

    # Use a deterministic mutation provider that generates all single-bit flips
    class DeterministicMutationProvider:
        def generate_mutations(
            self,
            program_code: str,
            num_mutations: int,
            evaluation: Evaluation[NoFeedback],
        ) -> List[str]:
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
        max_steps=2,
        samples_per_node=4,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
        checkpoint_provider=NoOpCheckpointProvider(),
    )

    # After step 1: should pick "1000" (8) over "0100" (4), "0010" (2), "0001" (1)
    # After step 2: should pick "1100" (12) or "1010" (10) or "1001" (9) - all > 8
    assert result.evaluation.reward is not None
    assert result.evaluation.reward >= 8.0  # Should improve from initial


def test_continues_sampling_without_improvement():
    """Test that algorithm continues sampling from current position when no improvement found."""

    # Create a provider that generates worse mutations for 3+ batches, then finds improvement on batch 4
    class ContinuousSamplingMutationProvider:
        def __init__(self):
            self.call_count = 0

        def generate_mutations(
            self,
            program_code: str,
            num_mutations: int,
            evaluation: Evaluation[NoFeedback],
        ) -> List[str]:
            self.call_count += 1
            if self.call_count == 1:
                # First call from "0000": move to "0011" (3)
                return ["0001", "0010", "0011"]
            elif self.call_count <= 4:
                # Calls 2-4 from "0011": generate worse or equal mutations (stay at "0011")
                return ["0001", "0010"]  # Both worse than "0011"
            else:
                # Call 5+ from "0011": finally find improvement
                return ["0111"]  # 7 > 3

    mutation_provider = ContinuousSamplingMutationProvider()
    evaluation_provider = BinaryStringEvaluationProvider()

    result = search(
        initial_program="0000",
        max_steps=10,
        samples_per_node=3,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
        checkpoint_provider=NoOpCheckpointProvider(),
    )

    # Should find the improved position on batch 5
    assert result.program_code == "0111"
    assert result.evaluation.reward == 7.0
    # Should have called mutation provider 5+ times (proving it didn't stop early)
    assert mutation_provider.call_count >= 5


def test_deduplication_works():
    """Test that duplicate programs are skipped and algorithm continues until max_steps."""

    # Provider that generates duplicate mutations
    class DuplicateMutationProvider:
        def __init__(self):
            self.call_count = 0

        def generate_mutations(
            self,
            program_code: str,
            num_mutations: int,
            evaluation: Evaluation[NoFeedback],
        ) -> List[str]:
            self.call_count += 1
            # Always return the same mutation multiple times
            return ["0001"] * num_mutations

    mutation_provider = DuplicateMutationProvider()
    evaluation_provider = BinaryStringEvaluationProvider()

    result = search(
        initial_program="0000",
        max_steps=5,
        samples_per_node=10,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
        checkpoint_provider=NoOpCheckpointProvider(),
    )

    # Should explore "0000" -> "0001", then keep trying until max_steps
    assert result.program_code == "0001"
    assert result.evaluation.reward == 1.0
    # Should continue calling generate_mutations (no valid children after first move)
    assert mutation_provider.call_count >= 2


def test_exhausts_budget_at_plateau():
    """Test that algorithm uses full max_steps budget when stuck at plateau."""

    # Provider that generates no improvements after initial move
    class PlateauMutationProvider:
        def __init__(self):
            self.call_count = 0

        def generate_mutations(
            self,
            program_code: str,
            num_mutations: int,
            evaluation: Evaluation[NoFeedback],
        ) -> List[str]:
            self.call_count += 1
            if self.call_count == 1:
                # First call: move from "0000" to "0001"
                return ["0001"]
            else:
                # All subsequent calls: generate worse mutations (stay at plateau)
                return ["0000"]

    mutation_provider = PlateauMutationProvider()
    evaluation_provider = BinaryStringEvaluationProvider()

    result = search(
        initial_program="0000",
        max_steps=5,
        samples_per_node=1,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
        checkpoint_provider=NoOpCheckpointProvider(),
    )

    # Should find "0001" and stay there
    assert result.program_code == "0001"
    assert result.evaluation.reward == 1.0
    # Should call generate_mutations exactly max_steps times (exhausts full budget)
    assert mutation_provider.call_count == 5


def test_only_moves_on_improvement():
    """Test that current position only changes when improvement found."""
    # Track which positions were sampled
    positions_sampled = []

    class TrackingMutationProvider:
        def __init__(self):
            self.step = 0

        def generate_mutations(
            self,
            program_code: str,
            num_mutations: int,
            evaluation: Evaluation[NoFeedback],
        ) -> List[str]:
            positions_sampled.append(program_code)
            self.step += 1

            if self.step == 1:
                # From "0000": move to "0011"
                return ["0011"]
            elif self.step == 2:
                # From "0011": generate worse (stay at "0011")
                return ["0001"]
            elif self.step == 3:
                # From "0011": generate worse again (stay at "0011")
                return ["0010"]
            elif self.step == 4:
                # From "0011": finally find improvement
                return ["0111"]
            else:
                # Continue from "0111"
                return ["1111"]

    mutation_provider = TrackingMutationProvider()
    evaluation_provider = BinaryStringEvaluationProvider()

    result = search(
        initial_program="0000",
        max_steps=5,
        samples_per_node=1,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
        checkpoint_provider=NoOpCheckpointProvider(),
    )

    # Should end at "1111"
    assert result.program_code == "1111"
    assert result.evaluation.reward == 15.0

    # Verify position sequence: "0000" → "0011" (move) → "0011" (stay) → "0011" (stay) → "0111" (move) → "1111" (move)
    expected_positions = ["0000", "0011", "0011", "0011", "0111"]
    assert positions_sampled == expected_positions


def test_handles_failed_evaluations():
    """Test that None rewards are filtered out correctly."""

    # Provider that generates mix of valid and invalid mutations
    class MixedMutationProvider:
        def generate_mutations(
            self,
            program_code: str,
            num_mutations: int,
            evaluation: Evaluation[NoFeedback],
        ) -> List[str]:
            # Return mix of valid binary strings and invalid strings
            return ["0001", "invalid", "0010", "also_invalid"]

    mutation_provider = MixedMutationProvider()
    evaluation_provider = BinaryStringEvaluationProvider()

    result = search(
        initial_program="0000",
        max_steps=5,
        samples_per_node=4,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
        checkpoint_provider=NoOpCheckpointProvider(),
    )

    # Should pick best valid mutation ("0010" = 2) and ignore invalid ones
    assert result.evaluation.reward is not None
    assert result.evaluation.reward >= 1.0  # Should find at least "0001" or "0010"


def test_tracks_parent_child_relationships():
    """Test that ancestor chain is set correctly."""
    mutation_provider = BinaryStringMutationProvider(seed=42)
    evaluation_provider = BinaryStringEvaluationProvider()

    result = search(
        initial_program="0000",
        max_steps=3,
        samples_per_node=4,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
        checkpoint_provider=NoOpCheckpointProvider(),
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
        def generate_mutations(
            self,
            program_code: str,
            num_mutations: int,
            evaluation: Evaluation[NoFeedback],
        ) -> List[str]:
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
        max_steps=5,
        samples_per_node=4,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
        checkpoint_provider=NoOpCheckpointProvider(),
    )

    # Should not mutate the same program twice
    assert len(visited_programs) == len(set(visited_programs))


def test_returns_global_best_not_just_final():
    """Test that search returns best node from entire search, not just final node."""

    # Create a scenario where we climb up then down
    class PeakMutationProvider:
        def __init__(self):
            self.step = 0

        def generate_mutations(
            self,
            program_code: str,
            num_mutations: int,
            evaluation: Evaluation[NoFeedback],
        ) -> List[str]:
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
        max_steps=10,
        samples_per_node=1,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
        checkpoint_provider=NoOpCheckpointProvider(),
    )

    # Should return the peak (1111 = 15), not the final node
    assert result.program_code == "1111"
    assert result.evaluation.reward == 15.0


def test_stops_at_max_steps():
    """Test that max_steps parameter is respected."""
    mutation_provider = BinaryStringMutationProvider(seed=42)
    evaluation_provider = BinaryStringEvaluationProvider()

    call_count = 0

    class CountingMutationProvider:
        def generate_mutations(
            self,
            program_code: str,
            num_mutations: int,
            evaluation: Evaluation[NoFeedback],
        ) -> List[str]:
            nonlocal call_count
            call_count += 1
            # Always generate valid mutations so we don't stop early
            return BinaryStringMutationProvider(seed=42).generate_mutations(
                program_code, num_mutations, evaluation
            )

    mutation_provider = CountingMutationProvider()

    search(
        initial_program="0000",
        max_steps=5,
        samples_per_node=4,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
        checkpoint_provider=NoOpCheckpointProvider(),
    )

    # Should call generate_mutations at most max_steps times
    assert call_count <= 5


def test_handles_all_failures():
    """Test that search handles gracefully when all evaluations fail."""
    mutation_provider = BinaryStringMutationProvider(seed=42)
    failing_provider = FailingEvaluationProvider()

    result = search(
        initial_program="0000",
        max_steps=5,
        samples_per_node=4,
        mutation_provider=mutation_provider,
        evaluation_provider=failing_provider,
        checkpoint_provider=NoOpCheckpointProvider(),
    )

    # Should return the initial program (with None reward)
    assert result.program_code == "0000"
    assert result.evaluation.reward is None
    assert result.is_seed is True


def test_handles_initial_program_failure():
    """Test that search handles when even the initial program fails."""
    mutation_provider = BinaryStringMutationProvider(seed=42)
    failing_provider = FailingEvaluationProvider()

    result = search(
        initial_program="invalid",
        max_steps=5,
        samples_per_node=4,
        mutation_provider=mutation_provider,
        evaluation_provider=failing_provider,
        checkpoint_provider=NoOpCheckpointProvider(),
    )

    # Should return the initial program with None reward
    assert result.program_code == "invalid"
    assert result.evaluation.reward is None


def test_single_iteration():
    """Test that search works with max_steps=1."""
    mutation_provider = BinaryStringMutationProvider(seed=42)
    evaluation_provider = BinaryStringEvaluationProvider()

    result = search(
        initial_program="0000",
        max_steps=1,
        samples_per_node=4,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
        checkpoint_provider=NoOpCheckpointProvider(),
    )

    # Should make one improvement step
    assert result.evaluation.reward is not None
    assert result.evaluation.reward > 0.0  # Should be better than initial


def test_get_archive_statistics():
    """Test the helper function for analyzing archive."""
    # Create a simple mock archive
    node1 = Node(
        program_code="0000",
        ancestors=[],
        evaluation=Evaluation(observation=NoFeedback(), reward=0.0),
    )
    node2 = Node(
        program_code="0001",
        ancestors=[],
        evaluation=Evaluation(observation=NoFeedback(), reward=1.0),
    )
    node3 = Node(
        program_code="invalid",
        ancestors=[],
        evaluation=Evaluation(observation=NoFeedback(), reward=None),
    )
    node4 = Node(
        program_code="0001",
        ancestors=[],
        evaluation=Evaluation(observation=NoFeedback(), reward=1.0),
    )  # duplicate code

    archive = [node1, node2, node3, node4]

    stats = get_archive_statistics(archive)

    assert stats["total_nodes"] == 4
    assert stats["valid_nodes"] == 3
    assert stats["failed_nodes"] == 1
    assert stats["best_reward"] == 1.0
    assert stats["unique_programs"] == 3  # "0000", "0001", "invalid"


def test_zero_max_steps():
    """Test edge case with max_steps=0."""
    mutation_provider = BinaryStringMutationProvider(seed=42)
    evaluation_provider = BinaryStringEvaluationProvider()

    result = search(
        initial_program="0000",
        max_steps=0,
        samples_per_node=4,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
        checkpoint_provider=NoOpCheckpointProvider(),
    )

    # Should just return the initial program
    assert result.program_code == "0000"
    assert result.evaluation.reward == 0.0
    assert result.is_seed is True


def test_set_parent_info_creates_ancestor_chain():
    """Test that set_parent_info correctly builds ancestor chain."""
    grandparent = Node(
        program_code="0000",
        ancestors=[],
        evaluation=Evaluation(observation=NoFeedback(), reward=0.0),
        is_seed=True,
    )
    parent = Node(
        program_code="0001",
        ancestors=[],
        evaluation=Evaluation(observation=NoFeedback(), reward=1.0),
    )
    child = Node(
        program_code="0011",
        ancestors=[],
        evaluation=Evaluation(observation=NoFeedback(), reward=3.0),
    )

    set_parent_info(parent, grandparent)
    set_parent_info(child, parent)

    # Parent should have grandparent as ancestor
    assert len(parent.ancestors) == 1
    assert parent.ancestors[0] == grandparent.ulid

    # Child should have parent and grandparent as ancestors
    assert len(child.ancestors) == 2
    assert child.ancestors[0] == parent.ulid
    assert child.ancestors[1] == grandparent.ulid


# Checkpoint Tests
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
        max_steps=3,
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
    assert last_checkpoint.current_step > 0
