"""
Depth-First Greedy Search (Hill Climbing) for program optimization.

This module implements a simple baseline search algorithm that:
1. Generates mutations of the current program
2. Picks the best mutation (greedy choice)
3. Continues from that best mutation if it improves (depth-first)
4. Otherwise continues sampling from current position
5. Stops when reaching max steps

Uses the same provider-based architecture as max_reward_puct for consistency.
"""

from typing import List, Set, cast, Generic, Union, Annotated
from arid_badger.max_reward_puct.domain import (
    MutationProvider,
    Node,
    get_content_key,
    set_parent_info,
)
from arid_badger.hill_climbing.checkpoint import (
    Checkpoint,
    CheckpointProvider,
    NoOpCheckpointProvider,
)
from pydantic import BaseModel, ConfigDict, Field
from typing import TypeVar, TypeGuard
from typing import Literal, Protocol, Optional


class NoFeedback(BaseModel):
    value: None = None


ObservationT = TypeVar("ObservationT", bound=BaseModel, default=NoFeedback)


class Evaluation(BaseModel, Generic[ObservationT]):
    observation: ObservationT
    reward: Optional[float] = None
    model_config = ConfigDict(frozen=True)


class EvaluationProvider(Protocol[ObservationT]):
    """Evaluates a program and returns its reward."""

    def evaluate(self, program_code: str) -> Evaluation[ObservationT]:
        """Returns reward (or None if evaluation failed)."""
        ...


# Helper functions for search phases (extracted for clarity and testability)


def generate_mutations_from_current(
    current: Node,
    samples_per_node: int,
    mutation_provider: MutationProvider,
) -> List[str]:
    """
    Phase A: Generate mutations from the current node.

    Args:
        current: Current node being explored
        samples_per_node: Number of mutations to generate
        mutation_provider: Provider for generating mutations

    Returns:
        List of mutation codes
    """
    return mutation_provider.generate_mutations(current.program_code, samples_per_node)


def evaluate_mutations(
    mutation_codes: List[str],
    current: Node,
    visited: Set[str],
    evaluation_provider: EvaluationProvider,
) -> List[Node]:
    """
    Phase B: Evaluate mutations, filter duplicates and failures, create child nodes.

    Args:
        mutation_codes: List of mutation codes to evaluate
        current: Current node (parent of mutations)
        visited: Set of content keys for deduplication
        evaluation_provider: Provider for evaluating programs

    Returns:
        List of valid child nodes (excludes duplicates and failed evaluations)
    """
    children: List[Node] = []
    for code in mutation_codes:
        # Skip duplicates
        content_key = code  # For binary strings, code is the key
        if content_key in visited:
            continue

        # Evaluate
        evaluation = evaluation_provider.evaluate(code)
        if evaluation.reward is None:
            # Skip failed evaluations
            continue

        # Create child node
        child = Node(program_code=code, reward=evaluation.reward, ancestors=[])
        set_parent_info(child, current)
        children.append(child)
        visited.add(content_key)

    return children


def select_best_child(children: List[Node]) -> Node:
    """
    Phase C: Select the child with the highest reward (greedy selection).

    Args:
        children: List of child nodes (must be non-empty)

    Returns:
        Child node with highest reward
    """
    return max(
        children, key=lambda c: c.reward if c.reward is not None else float("-inf")
    )


def update_global_best(candidate: Node, current_best: Node) -> Node:
    """
    Phase D: Update global best if candidate is better.

    Args:
        candidate: Node to consider as new best
        current_best: Current global best node

    Returns:
        Updated best node (either candidate or current_best)
    """
    if candidate.reward is not None and (
        current_best.reward is None or candidate.reward > current_best.reward
    ):
        return candidate
    return current_best


def search(
    initial_program: str,
    max_steps: int,
    samples_per_node: int,
    mutation_provider: MutationProvider,
    evaluation_provider: EvaluationProvider,
    checkpoint_provider: CheckpointProvider = NoOpCheckpointProvider(),
) -> Node:
    """
    Performs depth-first greedy search (hill climbing) from an initial program.

    Algorithm:
    1. Start with initial program and evaluate it
    2. For each step (up to max_steps):
       - Generate samples_per_node mutations of current program
       - Evaluate all mutations
       - Filter out failed evaluations (reward is None)
       - Skip duplicates using content-based deduplication
       - Pick the mutation with highest reward (greedy)
       - If best mutation improves over current → move to it (depth-first)
       - Otherwise stay at current position and continue sampling
    3. Return the best node found during entire search
    4. Stops when reaching max steps

    Args:
        initial_program: Starting program code
        max_steps: Maximum number of iterations
        samples_per_node: Number of mutations to generate at each step
        mutation_provider: Provider for generating mutations
        evaluation_provider: Provider for evaluating programs
        checkpoint_provider: Provider for saving/loading checkpoints (default: no-op)

    Returns:
        Best node found during search (global best, not just final node)
    """
    # Initialize state
    initial_evaluation = evaluation_provider.evaluate(initial_program)
    current = Node(
        program_code=initial_program,
        reward=initial_evaluation.reward,
        ancestors=[],
        is_seed=True,
    )
    best = current
    archive: List[Node] = [current]
    visited: Set[str] = {get_content_key(current)}

    # Delegate to internal implementation
    return _search_impl(
        current=current,
        best=best,
        archive=archive,
        visited=visited,
        current_step=0,
        max_steps=max_steps,
        samples_per_node=samples_per_node,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
        checkpoint_provider=checkpoint_provider,
    )


def resume_search(
    checkpoint: Checkpoint,
    max_steps: int,
    samples_per_node: int,
    mutation_provider: MutationProvider,
    evaluation_provider: EvaluationProvider,
    checkpoint_provider: CheckpointProvider = NoOpCheckpointProvider(),
) -> Node:
    """
    Resume search from a checkpoint.

    Args:
        checkpoint: Checkpoint containing state to resume from
        max_steps: Maximum number of iterations (total, including already completed)
        samples_per_node: Number of mutations to generate at each step
        mutation_provider: Provider for generating mutations
        evaluation_provider: Provider for evaluating programs
        checkpoint_provider: Provider for saving/loading checkpoints (default: no-op)

    Returns:
        Best node found during search (global best, not just final node)
    """
    return _search_impl(
        current=checkpoint.current_node,
        best=checkpoint.best_node,
        archive=checkpoint.archive,
        visited=checkpoint.visited,
        current_step=checkpoint.current_step,
        max_steps=max_steps,
        samples_per_node=samples_per_node,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
        checkpoint_provider=checkpoint_provider,
    )


def _search_impl(
    current: Node,
    best: Node,
    archive: List[Node],
    visited: Set[str],
    current_step: int,
    max_steps: int,
    samples_per_node: int,
    mutation_provider: MutationProvider,
    evaluation_provider: EvaluationProvider,
    checkpoint_provider: CheckpointProvider,
) -> Node:
    """
    Internal search implementation. Can be called from any state.

    This function is idempotent and doesn't distinguish between "initial" vs "resume" -
    it simply continues from whatever state it receives.

    Args:
        current: Current node being explored
        best: Best node found so far
        archive: List of all nodes explored
        visited: Set of content keys for deduplication
        current_step: Current iteration step
        max_steps: Maximum number of iterations
        samples_per_node: Number of mutations to generate at each step
        mutation_provider: Provider for generating mutations
        evaluation_provider: Provider for evaluating programs
        checkpoint_provider: Provider for saving checkpoints

    Returns:
        Best node found during search
    """
    print(f"Step {current_step}: Current={current.reward}, Best={best.reward}")

    for step in range(current_step, max_steps):
        # A. MUTATION GENERATION
        mutation_codes = generate_mutations_from_current(
            current=current,
            samples_per_node=samples_per_node,
            mutation_provider=mutation_provider,
        )

        # B. EVALUATION
        children = evaluate_mutations(
            mutation_codes=mutation_codes,
            current=current,
            visited=visited,
            evaluation_provider=evaluation_provider,
        )

        # Check for dead end (no valid children)
        if not children:
            print(f"Step {step + 1}: No valid children in this batch, continuing")
            continue

        # C. GREEDY SELECTION
        best_child = select_best_child(children)
        archive.extend(children)

        # D. BEST TRACKING
        best = update_global_best(candidate=best_child, current_best=best)

        # E. MOVEMENT DECISION
        # Only move if best_child improves over current (handle None rewards safely)
        should_move = False
        if current.reward is None or best_child.reward is None:
            # If either reward is None, move to best_child (preserve existing behavior)
            should_move = True
        elif best_child.reward > current.reward:
            should_move = True

        if should_move:
            print(
                f"Step {step + 1}: Moving to improved position (reward={best_child.reward})"
            )
            current = best_child
        else:
            print(
                f"Step {step + 1}: No improvement (best_child={best_child.reward}, current={current.reward}), continuing from current position"
            )

        # F. CHECKPOINT
        checkpoint_provider.save(
            Checkpoint(
                current_node=current,
                best_node=best,
                archive=archive,
                visited=visited,
                current_step=step + 1,
            )
        )

    return best


def get_archive_statistics(archive: List[Node]) -> dict:
    """
    Helper for analyzing explored nodes.

    Returns:
        Dictionary with statistics about the archive:
        - total_nodes: Total number of nodes explored
        - valid_nodes: Number of nodes with non-None rewards
        - failed_nodes: Number of nodes with None rewards
        - best_reward: Best reward found
        - unique_programs: Number of unique program codes
    """
    valid_nodes = [n for n in archive if n.reward is not None]
    failed_nodes = [n for n in archive if n.reward is None]

    return {
        "total_nodes": len(archive),
        "valid_nodes": len(valid_nodes),
        "failed_nodes": len(failed_nodes),
        "best_reward": max(
            cast(List[float], [n.reward for n in valid_nodes]), default=None
        ),
        "unique_programs": len({get_content_key(n) for n in archive}),
    }
