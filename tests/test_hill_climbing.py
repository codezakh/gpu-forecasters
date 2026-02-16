"""
Tests for Depth-First Greedy Search (Hill Climbing) algorithm.

Uses Binary String test environment from test_max_reward_puct.py:
- BinaryStringMutationProvider: Flips single bits in binary string
- BinaryStringEvaluationProvider: Returns decimal value as reward
- FailingEvaluationProvider: Always returns None to test failure handling
"""

import random
from typing import List, Optional

from arid_badger.hill_climbing.domain import (
    search,
    get_archive_statistics,
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
