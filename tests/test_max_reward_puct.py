"""
Tests for Max-Reward PUCT search algorithm.

Uses Binary Graph Traversal as a toy example: binary strings where mutations
flip single bits and the reward is the decimal value. This provides a simple,
deterministic way to test the PUCT algorithm's selection, exploration/exploitation,
and convergence behavior.
"""

import json
import random
from pathlib import Path
from typing import List, Optional
from ulid import ULID

from arid_badger.hill_climbing.domain import Evaluation, NoFeedback, Node
from arid_badger.hill_climbing.scoring_providers.kernelbench import KernelBenchObservation
from arid_badger.kernelbench.core import (
    CompileFailedFeedback,
    RuntimeErrorFeedback,
    IncorrectFeedback,
    SuccessFeedback,
    InfrastructureFailureFeedback,
)
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
from arid_badger.max_reward_puct.checkpoint import FilePuctCheckpointProvider, PuctCheckpoint
from arid_badger.max_reward_puct.trajectory import FileTrajectoryProvider, TrajectoryRecord


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

    def batch_evaluate(
        self, program_codes: list[str]
    ) -> list[Evaluation[NoFeedback]]:
        return [self.evaluate(code) for code in program_codes]


class FailingEvaluationProvider:
    """Provider that always returns reward=None to test failed rollout handling."""

    def evaluate(self, program_code: str) -> Evaluation[NoFeedback]:
        return _eval(None)

    def batch_evaluate(
        self, program_codes: list[str]
    ) -> list[Evaluation[NoFeedback]]:
        return [self.evaluate(code) for code in program_codes]


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


# FilePuctCheckpointProvider JSON round-trip tests


def test_file_checkpoint_json_round_trip_no_feedback(tmp_path: Path) -> None:
    """Checkpoint with NoFeedback nodes serializes and deserializes via JSON correctly."""
    node_a = Node(program_code="0000", evaluation=_eval(0.0), ancestors=[], is_seed=True)
    node_b = Node(program_code="0001", evaluation=_eval(1.0), ancestors=[node_a.ulid])

    checkpoint = PuctCheckpoint[NoFeedback](
        archive=[node_a, node_b],
        seed_ids={node_a.ulid},
        visit_counts={node_a.ulid: 3},
        best_child_rewards={node_a.ulid: 1.0},
        global_expansion_count=5,
        current_step=2,
    )

    provider: FilePuctCheckpointProvider[NoFeedback] = FilePuctCheckpointProvider(
        path=tmp_path / "checkpoint.json",
        checkpoint_type=PuctCheckpoint[NoFeedback],
    )
    provider.save(checkpoint)
    loaded = provider.load()

    assert loaded is not None
    assert loaded.current_step == 2
    assert loaded.global_expansion_count == 5
    assert len(loaded.archive) == 2

    loaded_ulids = {n.ulid for n in loaded.archive}
    assert node_a.ulid in loaded_ulids
    assert node_b.ulid in loaded_ulids

    assert node_a.ulid in loaded.seed_ids
    assert loaded.visit_counts[node_a.ulid] == 3
    assert loaded.best_child_rewards[node_a.ulid] == 1.0

    loaded_b = next(n for n in loaded.archive if n.ulid == node_b.ulid)
    assert loaded_b.ancestors == [node_a.ulid]


def test_file_checkpoint_load_returns_none_when_missing(tmp_path: Path) -> None:
    """load() returns None when the checkpoint file does not exist."""
    provider: FilePuctCheckpointProvider[NoFeedback] = FilePuctCheckpointProvider(
        path=tmp_path / "nonexistent.json",
        checkpoint_type=PuctCheckpoint[NoFeedback],
    )
    assert provider.load() is None


def test_file_checkpoint_json_round_trip_kernelbench_all_feedback_variants(
    tmp_path: Path,
) -> None:
    """All five KernelBench feedback variants survive a JSON checkpoint round-trip."""

    def _kb_eval(feedback: object, reward: float | None) -> Evaluation[KernelBenchObservation]:
        return Evaluation[KernelBenchObservation](
            observation=KernelBenchObservation(feedback=feedback),  # type: ignore[arg-type]
            reward=reward,
        )

    node_compile = Node(
        program_code="kernel_compile_fail",
        evaluation=_kb_eval(
            CompileFailedFeedback(
                compilation_error_name="SyntaxError",
                compilation_error="unexpected token",
            ),
            reward=None,
        ),
        ancestors=[],
        is_seed=True,
    )
    node_runtime = Node(
        program_code="kernel_runtime_error",
        evaluation=_kb_eval(
            RuntimeErrorFeedback(
                runtime_error_name="RuntimeError",
                runtime_error="segfault",
                runtime_error_traceback="traceback here",
            ),
            reward=None,
        ),
        ancestors=[node_compile.ulid],
    )
    node_incorrect = Node(
        program_code="kernel_incorrect",
        evaluation=_kb_eval(
            IncorrectFeedback(
                correctness_issue="wrong output",
                max_difference=["0.5"],
                avg_difference=["0.1"],
            ),
            reward=None,
        ),
        ancestors=[node_compile.ulid],
    )
    node_success = Node(
        program_code="kernel_success",
        evaluation=_kb_eval(
            SuccessFeedback(runtime_us=100.0, ref_runtime_us=200.0, speedup=2.0),
            reward=2.0,
        ),
        ancestors=[node_compile.ulid],
    )
    node_infra = Node(
        program_code="kernel_infra_fail",
        evaluation=_kb_eval(
            InfrastructureFailureFeedback(reason="subprocess timeout"),
            reward=None,
        ),
        ancestors=[node_compile.ulid],
    )

    all_nodes = [node_compile, node_runtime, node_incorrect, node_success, node_infra]

    checkpoint = PuctCheckpoint[KernelBenchObservation](
        archive=all_nodes,
        seed_ids={node_compile.ulid},
        visit_counts={node_compile.ulid: 4},
        best_child_rewards={node_compile.ulid: 2.0},
        global_expansion_count=4,
        current_step=4,
    )

    provider: FilePuctCheckpointProvider[KernelBenchObservation] = FilePuctCheckpointProvider(
        path=tmp_path / "kb_checkpoint.json",
        checkpoint_type=PuctCheckpoint[KernelBenchObservation],
    )
    provider.save(checkpoint)
    loaded = provider.load()

    assert loaded is not None
    assert len(loaded.archive) == 5

    by_ulid = {n.ulid: n for n in loaded.archive}

    fb_compile = by_ulid[node_compile.ulid].evaluation.observation.feedback
    assert fb_compile.kind == "compile_failed"
    assert fb_compile.compilation_error_name == "SyntaxError"

    fb_runtime = by_ulid[node_runtime.ulid].evaluation.observation.feedback
    assert fb_runtime.kind == "runtime_error"
    assert fb_runtime.runtime_error == "segfault"
    assert by_ulid[node_runtime.ulid].ancestors == [node_compile.ulid]

    fb_incorrect = by_ulid[node_incorrect.ulid].evaluation.observation.feedback
    assert fb_incorrect.kind == "incorrect"
    assert fb_incorrect.correctness_issue == "wrong output"

    fb_success = by_ulid[node_success.ulid].evaluation.observation.feedback
    assert fb_success.kind == "success"
    assert fb_success.speedup == 2.0
    assert by_ulid[node_success.ulid].evaluation.reward == 2.0

    fb_infra = by_ulid[node_infra.ulid].evaluation.observation.feedback
    assert fb_infra.kind == "infrastructure_failure"
    assert fb_infra.reason == "subprocess timeout"


def test_file_checkpoint_produces_valid_json(tmp_path: Path) -> None:
    """Saved checkpoint file contains valid JSON with expected top-level keys."""
    node = Node(program_code="0000", evaluation=_eval(0.0), ancestors=[], is_seed=True)
    checkpoint = PuctCheckpoint[NoFeedback](
        archive=[node],
        seed_ids={node.ulid},
        visit_counts={},
        best_child_rewards={},
        global_expansion_count=0,
        current_step=0,
    )

    checkpoint_path = tmp_path / "checkpoint.json"
    provider: FilePuctCheckpointProvider[NoFeedback] = FilePuctCheckpointProvider(
        path=checkpoint_path,
        checkpoint_type=PuctCheckpoint[NoFeedback],
    )
    provider.save(checkpoint)

    raw = checkpoint_path.read_text()
    data = json.loads(raw)

    assert isinstance(data, dict)
    assert "archive" in data
    assert "seed_ids" in data
    assert "visit_counts" in data
    assert "best_child_rewards" in data
    assert "global_expansion_count" in data
    assert "current_step" in data


def test_unparametrized_checkpoint_loses_kernelbench_observation(tmp_path: Path) -> None:
    """Regression test for the e0007 serialization bug.

    _search_impl creates PuctCheckpoint(archive=...) without the concrete type
    parameter.  Previously, Pydantic serialized KernelBenchObservation using
    NoFeedback's schema (the TypeVar default), silently dropping all observation
    fields and causing a ValidationError on load.

    FilePuctCheckpointProvider now uses a TypeAdapter built from checkpoint_type
    for both save and load, so the correct parametrized schema is always used
    regardless of how the PuctCheckpoint object was constructed.
    """
    node = Node(
        program_code="kernel_code",
        evaluation=Evaluation[KernelBenchObservation](
            observation=KernelBenchObservation(
                feedback=SuccessFeedback(runtime_us=100.0, ref_runtime_us=200.0, speedup=2.0)
            ),
            reward=2.0,
        ),
        ancestors=[],
        is_seed=True,
    )

    # Simulate what _search_impl does: PuctCheckpoint without the type parameter.
    checkpoint = PuctCheckpoint(
        archive=[node],
        seed_ids={node.ulid},
        visit_counts={},
        best_child_rewards={},
        global_expansion_count=0,
        current_step=1,
    )

    provider = FilePuctCheckpointProvider(
        path=tmp_path / "checkpoint.json",
        checkpoint_type=PuctCheckpoint[KernelBenchObservation],
    )
    provider.save(checkpoint)

    loaded = provider.load()
    assert loaded is not None
    assert loaded.current_step == 1
    assert len(loaded.archive) == 1
    loaded_node = loaded.archive[0]
    assert loaded_node.program_code == "kernel_code"
    feedback = loaded_node.evaluation.observation.feedback
    assert isinstance(feedback, SuccessFeedback)
    assert feedback.speedup == 2.0
    assert feedback.runtime_us == 100.0
    assert feedback.ref_runtime_us == 200.0


# Trajectory Tests


def test_trajectory_records_one_per_step() -> None:
    """trajectory_provider.record() is called exactly once per step with correct data."""

    class CapturingTrajectoryProvider:
        def __init__(self) -> None:
            self.records: list[tuple[int, ULID, float | None, int]] = []

        def record(self, step: int, best_node: Node, archive_size: int) -> None:
            self.records.append((step, best_node.ulid, best_node.evaluation.reward, archive_size))

    captured = CapturingTrajectoryProvider()
    search(
        initial_program="0000",
        total_budget_steps=5,
        batch_size=1,
        samples_per_parent=4,
        mutation_provider=BinaryStringMutationProvider(seed=42),
        evaluation_provider=BinaryStringEvaluationProvider(),
        trajectory_provider=captured,
    )

    assert len(captured.records) == 5
    assert [r[0] for r in captured.records] == [0, 1, 2, 3, 4]
    # BinaryString evaluator always succeeds — reward is always non-None
    assert all(r[2] is not None for r in captured.records)
    # archive_size is monotonically non-decreasing
    sizes = [r[3] for r in captured.records]
    assert sizes == sorted(sizes)


def test_file_trajectory_provider_writes_valid_jsonl(tmp_path: Path) -> None:
    """FileTrajectoryProvider appends valid JSONL that round-trips via TrajectoryRecord."""
    path = tmp_path / "trajectory.jsonl"
    provider = FileTrajectoryProvider(path)

    node_a = Node(program_code="0000", evaluation=_eval(0.5), ancestors=[], is_seed=True)
    node_b = Node(program_code="0001", evaluation=_eval(0.9), ancestors=[node_a.ulid])

    provider.record(step=0, best_node=node_a, archive_size=1)
    provider.record(step=1, best_node=node_b, archive_size=2)

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2

    rec0 = TrajectoryRecord.model_validate_json(lines[0])
    rec1 = TrajectoryRecord.model_validate_json(lines[1])

    assert rec0.step == 0
    assert rec0.best_ulid == node_a.ulid
    assert rec0.best_reward == 0.5
    assert rec0.archive_size == 1

    assert rec1.step == 1
    assert rec1.best_ulid == node_b.ulid
    assert rec1.best_reward == 0.9
    assert rec1.archive_size == 2
