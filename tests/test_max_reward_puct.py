"""
Tests for Max-Reward PUCT search algorithm.

Uses Binary Graph Traversal as a toy example: binary strings where mutations
flip single bits and the reward is the decimal value. This provides a simple,
deterministic way to test the PUCT algorithm's selection, exploration/exploitation,
and convergence behavior.
"""

import random
from typing import List, Optional
import pytest

from arid_badger.hill_climbing.domain import Evaluation, NoFeedback, Node
from arid_badger.max_reward_puct.search import (
    search,
    resume_search,
    select_batch_of_parents,
    expand_and_evaluate,
    update_archive,
    backpropagate,
    record_failed_rollout,
    calculate_puct_scores,
    set_parent_info,
)
from arid_badger.max_reward_puct.checkpoint import PuctCheckpoint


def _eval(reward: float | None) -> Evaluation[NoFeedback]:
    return Evaluation(observation=NoFeedback(), reward=reward)


class BinaryStringMutationProvider:
    """Toy mutation provider: flips single bits in binary string."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def generate_mutations(
        self,
        program_code: str,
        num_mutations: int,
        evaluation: Evaluation[NoFeedback],
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

    def evaluate(self, program_code: str) -> Evaluation[NoFeedback]:
        """Return decimal value of binary string."""
        try:
            reward: float | None = float(int(program_code, 2))
        except ValueError:
            reward = None
        return _eval(reward)


class FailingEvaluationProvider:
    """Provider that always returns reward=None to test failed rollout handling."""

    def evaluate(self, program_code: str) -> Evaluation[NoFeedback]:
        return _eval(None)


def test_binary_string_mutation_provider():
    """Test that mutation provider generates valid mutations."""
    provider = BinaryStringMutationProvider(seed=42)
    mutations = provider.generate_mutations("0000", 4, _eval(0.0))

    assert len(mutations) == 4
    # Each mutation should differ by exactly one bit
    for mutation in mutations:
        assert len(mutation) == 4
        assert all(c in "01" for c in mutation)
        hamming_distance = sum(a != b for a, b in zip("0000", mutation))
        assert hamming_distance == 1


def test_binary_string_evaluation_provider():
    """Test that evaluation provider returns correct decimal values."""
    provider = BinaryStringEvaluationProvider()

    assert provider.evaluate("0000").reward == 0.0
    assert provider.evaluate("0001").reward == 1.0
    assert provider.evaluate("0010").reward == 2.0
    assert provider.evaluate("1111").reward == 15.0
    assert provider.evaluate("invalid").reward is None


def test_search_converges_to_maximum():
    """Test that PUCT search finds optimal binary string."""
    mutation_provider = BinaryStringMutationProvider(seed=42)
    evaluation_provider = BinaryStringEvaluationProvider()

    result = search(
        initial_program="0000",
        total_budget_steps=20,
        batch_size=1,
        samples_per_parent=4,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
    )

    # Should find the maximum (all 1s)
    assert result.program_code == "1111"
    assert result.evaluation.reward == 15.0


def test_search_batch_size_greater_than_one():
    """Test that search works with batch_size > 1."""
    mutation_provider = BinaryStringMutationProvider(seed=42)
    evaluation_provider = BinaryStringEvaluationProvider()

    result = search(
        initial_program="0000",
        total_budget_steps=20,
        batch_size=2,
        samples_per_parent=4,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
    )

    # Should still find a high-reward solution
    assert result.evaluation.reward is not None
    assert result.evaluation.reward >= 10.0  # Should get reasonably close to maximum


def test_puct_selection_prioritizes_high_reward():
    """Test that PUCT scoring prioritizes high-reward states."""
    # Create a simple archive with different rewards
    node_low = Node(program_code="0001", evaluation=_eval(1.0), ancestors=[], is_seed=True)
    node_mid = Node(program_code="0011", evaluation=_eval(3.0), ancestors=[], is_seed=True)
    node_high = Node(program_code="0111", evaluation=_eval(7.0), ancestors=[], is_seed=True)

    archive = [node_low, node_mid, node_high]
    seed_ids = {node_low.ulid, node_mid.ulid, node_high.ulid}

    # With no visit counts, highest reward should score highest
    scored = calculate_puct_scores(
        archive=archive,
        visit_counts={},
        best_child_rewards={},
        global_expansion_count=0,
        seed_ids=seed_ids,
    )

    # Extract nodes in score order
    nodes_by_score = [node for _, _, node in scored]
    # Highest reward should be first
    assert nodes_by_score[0].ulid == node_high.ulid


def test_lineage_blocking_in_batch_selection():
    """Test that lineage blocking prevents parent-child pairs in same batch."""
    # Create a parent and its children
    parent = Node(program_code="0000", evaluation=_eval(0.0), ancestors=[], is_seed=True)
    child1 = Node(program_code="0001", evaluation=_eval(1.0), ancestors=[])
    child2 = Node(program_code="0010", evaluation=_eval(2.0), ancestors=[])

    # Manually set parent info
    set_parent_info(child1, parent)
    set_parent_info(child2, parent)

    archive = [parent, child1, child2]
    seed_ids = {parent.ulid}

    # Select batch of size 2
    selected = select_batch_of_parents(
        archive=archive,
        batch_size=2,
        visit_counts={},
        best_child_rewards={},
        global_expansion_count=0,
        seed_ids=seed_ids,
    )

    # Should not select both parent and child
    selected_ids = {node.ulid for node in selected}
    if parent.ulid in selected_ids:
        assert child1.ulid not in selected_ids
        assert child2.ulid not in selected_ids


def test_archive_update_enforces_top_k_per_parent():
    """Test that update_archive keeps only top-k children per parent."""
    mutation_provider = BinaryStringMutationProvider(seed=42)
    evaluation_provider = BinaryStringEvaluationProvider()

    parent = Node(program_code="0000", evaluation=_eval(0.0), ancestors=[], is_seed=True)

    # Generate many children
    children, parent_states = expand_and_evaluate(
        parents=[parent],
        samples_per_parent=4,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
    )

    archive = [parent]
    seed_ids = {parent.ulid}

    # Update with k_per_parent=2
    update_archive(
        archive=archive,
        children=children,
        parent_states=parent_states,
        seed_ids=seed_ids,
        k_per_parent=2,
    )

    # Should have parent + 2 children = 3 nodes
    assert len(archive) <= 3


def test_archive_update_skips_none_rewards():
    """Test that update_archive skips children with None rewards."""
    failing_provider = FailingEvaluationProvider()
    mutation_provider = BinaryStringMutationProvider(seed=42)

    parent = Node(program_code="0000", evaluation=_eval(0.0), ancestors=[], is_seed=True)

    children, parent_states = expand_and_evaluate(
        parents=[parent],
        samples_per_parent=4,
        mutation_provider=mutation_provider,
        evaluation_provider=failing_provider,
    )

    archive = [parent]
    seed_ids = {parent.ulid}

    update_archive(
        archive=archive,
        children=children,
        parent_states=parent_states,
        seed_ids=seed_ids,
    )

    # Should only have the parent (no children added)
    assert len(archive) == 1
    assert archive[0].ulid == parent.ulid


def test_archive_deduplication():
    """Test that update_archive deduplicates by program content."""
    parent = Node(program_code="0000", evaluation=_eval(0.0), ancestors=[], is_seed=True)

    # Create duplicate children manually
    child1 = Node(program_code="0001", evaluation=_eval(1.0), ancestors=[])
    child2 = Node(program_code="0001", evaluation=_eval(1.0), ancestors=[])  # duplicate

    archive = [parent]
    seed_ids = {parent.ulid}

    update_archive(
        archive=archive,
        children=[child1, child2],
        parent_states=[parent, parent],
        seed_ids=seed_ids,
    )

    # Should only have parent + 1 child (duplicate removed)
    assert len(archive) == 2


def test_backpropagate_updates_visit_counts():
    """Test that backpropagate correctly updates n and m."""
    parent = Node(program_code="0000", evaluation=_eval(0.0), ancestors=[], is_seed=True)
    child = Node(program_code="0001", evaluation=_eval(1.0), ancestors=[])

    set_parent_info(child, parent)

    n = {}
    m = {}
    T = 0

    T_new = backpropagate(
        children=[child], parent_states=[parent], n=n, m=m, T=T
    )

    # Visit count should be incremented for parent
    assert n[parent.ulid] == 1
    # Best child reward should be set
    assert m[parent.ulid] == 1.0
    # Global expansion counter should be incremented
    assert T_new == 1


def test_backpropagate_skips_none_rewards():
    """Test that backpropagate skips children with None rewards."""
    parent = Node(program_code="0000", evaluation=_eval(0.0), ancestors=[], is_seed=True)
    child = Node(program_code="invalid", evaluation=_eval(None), ancestors=[])

    set_parent_info(child, parent)

    n = {}
    m = {}
    T = 0

    T_new = backpropagate(
        children=[child], parent_states=[parent], n=n, m=m, T=T
    )

    # No updates should occur
    assert parent.ulid not in n
    assert parent.ulid not in m
    assert T_new == 0


def test_record_failed_rollout():
    """Test that failed rollouts still increment visit counts."""
    parent = Node(program_code="0000", evaluation=_eval(0.0), ancestors=[], is_seed=True)

    n = {}
    T = 0

    T_new = record_failed_rollout(parent=parent, n=n, T=T)

    # Visit count should be incremented
    assert n[parent.ulid] == 1
    # Global expansion counter should be incremented
    assert T_new == 1


def test_search_handles_all_failed_evaluations():
    """Test that search handles case where all evaluations fail."""
    mutation_provider = BinaryStringMutationProvider(seed=42)
    failing_provider = FailingEvaluationProvider()

    # This should not crash, even though all mutations fail
    result = search(
        initial_program="0000",
        total_budget_steps=5,
        batch_size=1,
        samples_per_parent=4,
        mutation_provider=mutation_provider,
        evaluation_provider=failing_provider,
    )

    # Should return the initial program (no valid children found)
    assert result.program_code == "0000"
    assert result.evaluation.reward is None


def test_exploration_bonus_decays_with_visits():
    """Test that nodes with more visits get lower exploration bonus."""
    node1 = Node(program_code="0001", evaluation=_eval(1.0), ancestors=[], is_seed=True)
    node2 = Node(program_code="0010", evaluation=_eval(1.0), ancestors=[], is_seed=True)

    archive = [node1, node2]
    seed_ids = {node1.ulid, node2.ulid}

    # Give node1 many visits, node2 few visits
    visit_counts = {node1.ulid: 10, node2.ulid: 1}

    scored = calculate_puct_scores(
        archive=archive,
        visit_counts=visit_counts,
        best_child_rewards={},
        global_expansion_count=10,
        seed_ids=seed_ids,
    )

    # Extract nodes and their scores
    scores_dict = {node.ulid: score for score, _, node in scored}

    # Node2 should have higher score due to exploration bonus
    assert scores_dict[node2.ulid] > scores_dict[node1.ulid]


# Checkpoint Tests


def test_checkpoint_save_called():
    """Test that checkpoint provider.save() is called after each step."""

    class MockCheckpointProvider:
        def __init__(self):
            self.save_count = 0
            self.saved_checkpoints: List[PuctCheckpoint] = []

        def save(self, checkpoint: PuctCheckpoint) -> None:
            self.save_count += 1
            self.saved_checkpoints.append(checkpoint)

        def load(self) -> Optional[PuctCheckpoint]:
            return None

    mutation_provider = BinaryStringMutationProvider(seed=42)
    evaluation_provider = BinaryStringEvaluationProvider()
    mock_provider = MockCheckpointProvider()

    search(
        initial_program="0000",
        total_budget_steps=3,
        batch_size=1,
        samples_per_parent=4,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
        checkpoint_provider=mock_provider,
    )

    assert mock_provider.save_count >= 1
    assert len(mock_provider.saved_checkpoints) >= 1

    last_checkpoint = mock_provider.saved_checkpoints[-1]
    assert last_checkpoint.current_step > 0


def test_checkpoint_resume_produces_same_result():
    """Test that resuming from a checkpoint finds an equally good or better result."""
    N = 20
    N_half = N // 2

    evaluation_provider = BinaryStringEvaluationProvider()

    # Run partial search to N//2, capturing final checkpoint
    saved_checkpoints: List[PuctCheckpoint] = []

    class CapturingProvider:
        def save(self, checkpoint: PuctCheckpoint) -> None:
            saved_checkpoints.append(checkpoint)

        def load(self) -> Optional[PuctCheckpoint]:
            return None

    mutation_provider_partial = BinaryStringMutationProvider(seed=42)
    search(
        initial_program="0000",
        total_budget_steps=N_half,
        batch_size=1,
        samples_per_parent=4,
        mutation_provider=mutation_provider_partial,
        evaluation_provider=evaluation_provider,
        checkpoint_provider=CapturingProvider(),
    )

    assert saved_checkpoints, "Expected at least one checkpoint to be saved"
    checkpoint = saved_checkpoints[-1]

    # Get best reward in the checkpoint's archive
    best_in_checkpoint = max(
        (n.evaluation.reward for n in checkpoint.archive if n.evaluation.reward is not None),
        default=None,
    )

    # Resume from checkpoint for the remaining steps
    mutation_provider_resume = BinaryStringMutationProvider(seed=42)
    result_resumed = resume_search(
        checkpoint=checkpoint,
        total_budget_steps=N,
        batch_size=1,
        samples_per_parent=4,
        mutation_provider=mutation_provider_resume,
        evaluation_provider=evaluation_provider,
    )

    assert result_resumed.evaluation.reward is not None
    if best_in_checkpoint is not None:
        assert result_resumed.evaluation.reward >= best_in_checkpoint


def test_resume_idempotent_when_complete():
    """Test that resuming a completed checkpoint performs no additional mutations."""
    N = 5

    evaluation_provider = BinaryStringEvaluationProvider()
    saved_checkpoints: List[PuctCheckpoint] = []

    class CapturingProvider:
        def save(self, checkpoint: PuctCheckpoint) -> None:
            saved_checkpoints.append(checkpoint)

        def load(self) -> Optional[PuctCheckpoint]:
            return None

    mutation_provider = BinaryStringMutationProvider(seed=42)
    search(
        initial_program="0000",
        total_budget_steps=N,
        batch_size=1,
        samples_per_parent=4,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
        checkpoint_provider=CapturingProvider(),
    )

    assert saved_checkpoints, "Expected checkpoints to be saved during search"
    final_checkpoint = saved_checkpoints[-1]
    assert final_checkpoint.current_step == N

    # Resume with same total_budget_steps — loop range(N, N) is empty, no mutations
    mutation_calls: List[str] = []

    class TrackingMutationProvider:
        def generate_mutations(
            self,
            program_code: str,
            num_mutations: int,
            evaluation: Evaluation[NoFeedback],
        ) -> List[str]:
            mutation_calls.append(program_code)
            return []

    resume_search(
        checkpoint=final_checkpoint,
        total_budget_steps=N,
        batch_size=1,
        samples_per_parent=4,
        mutation_provider=TrackingMutationProvider(),
        evaluation_provider=evaluation_provider,
    )

    assert len(mutation_calls) == 0, "No mutations should occur when resuming a completed search"
