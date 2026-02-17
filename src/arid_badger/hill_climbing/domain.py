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
from pydantic import BaseModel, ConfigDict, Field
from typing import TypeVar, TypeGuard
from typing import Literal, Protocol, Optional
from ulid import ULID


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


class MutationProvider(Protocol[ObservationT]):
    def generate_mutations(
        self,
        program_code: str,
        num_mutations: int,
        evaluation: Evaluation[ObservationT],
    ) -> List[str]: ...


class Node(BaseModel, Generic[ObservationT]):
    """
    Represents a specific program/kernel solution.
    """

    program_code: str

    # Ancestor chain: list of {"id": str, "timestep": int} dicts,
    # most recent parent first.  Matches State.parents.
    ancestors: List[ULID]

    # Evaluation of the program code.
    evaluation: Evaluation[ObservationT]

    ulid: ULID = Field(default_factory=ULID)
    is_seed: bool = False


class Checkpoint(BaseModel, Generic[ObservationT]):
    current_node: Node[ObservationT]
    archive: List[Node[ObservationT]]
    visited: Set[str]
    current_step: int


class CheckpointProvider(Protocol[ObservationT]):
    def save(self, checkpoint: Checkpoint[ObservationT]): ...
    def load(self) -> Optional[Checkpoint[ObservationT]]: ...


def set_parent_info(child: Node[ObservationT], parent: Node[ObservationT]):
    """
    Sets the child's parents list to [parent_ref] + parent.parents.
    Matches _set_parent_info (sampler.py:52-55).
    """
    child.ancestors = [parent.ulid] + list(parent.ancestors)


def get_content_key(node: Node[ObservationT]) -> str:
    """
    Returns the content key for a node.
    """
    return node.program_code


def generate_mutations_from_current(
    current: Node[ObservationT],
    samples_per_node: int,
    mutation_provider: MutationProvider[ObservationT],
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
    return mutation_provider.generate_mutations(
        current.program_code, samples_per_node, current.evaluation
    )


def evaluate_mutations(
    mutation_codes: List[str],
    current: Node[ObservationT],
    visited: Set[str],
    evaluation_provider: EvaluationProvider[ObservationT],
) -> List[Node[ObservationT]]:
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
    children: List[Node[ObservationT]] = []
    for code in mutation_codes:
        # Skip duplicates
        content_key = code  # For binary strings, code is the key
        if content_key in visited:
            continue

        # Evaluate
        evaluation = evaluation_provider.evaluate(code)
        if evaluation.reward is None:
            # Skip failed evaluations
            # NOTE: We explicitly do not store these; we may _want_ to store them simply
            # to provide better feedback (i.e. compilation errors, etc.) but for now this is
            # a choice we are making to keep the code simpler.
            continue

        # Create child node
        child = Node(
            program_code=code,
            ancestors=[],
            evaluation=evaluation,
        )
        set_parent_info(child, current)
        children.append(child)
        visited.add(content_key)

    return children


def expand_and_evaluate(
    current: Node[ObservationT],
    samples_per_node: int,
    mutation_provider: MutationProvider[ObservationT],
    evaluation_provider: EvaluationProvider[ObservationT],
    visited: Set[str],
) -> List[Node[ObservationT]]:
    mutated_programs = mutation_provider.generate_mutations(
        current.program_code, samples_per_node, current.evaluation
    )

    children: List[Node[ObservationT]] = []
    for program in mutated_programs:
        # Skip duplicates
        if program in visited:
            continue

        # Evaluate
        evaluation = evaluation_provider.evaluate(program)
        if evaluation.reward is None:
            # Skip failed evaluations
            # NOTE: We explicitly do not store these; we may _want_ to store them simply
            # to provide better feedback (i.e. compilation errors, etc.) but for now this is
            # a choice we are making to keep the code simpler.
            continue

        # Create child node
        child = Node(
            program_code=program,
            ancestors=[],
            evaluation=evaluation,
        )
        set_parent_info(child, current)
        children.append(child)
        visited.add(program)

    return children


def select_best_child(children: List[Node[ObservationT]]) -> Node[ObservationT]:
    """
    Phase C: Select the child with the highest reward (greedy selection).

    Args:
        children: List of child nodes (must be non-empty)

    Returns:
        Child node with highest reward
    """
    return max(
        children,
        key=lambda c: (
            c.evaluation.reward if c.evaluation.reward is not None else float("-inf")
        ),
    )



def search(
    initial_program: str,
    max_steps: int,
    samples_per_node: int,
    mutation_provider: MutationProvider[ObservationT],
    evaluation_provider: EvaluationProvider[ObservationT],
    checkpoint_provider: CheckpointProvider[ObservationT],
) -> Node[ObservationT]:
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
        evaluation=initial_evaluation,
        ancestors=[],
        is_seed=True,
    )
    archive: List[Node[ObservationT]] = [current]
    visited: Set[str] = {current.program_code}

    # Delegate to internal implementation
    return _search_impl(
        current=current,
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
    checkpoint: Checkpoint[ObservationT],
    max_steps: int,
    samples_per_node: int,
    mutation_provider: MutationProvider[ObservationT],
    evaluation_provider: EvaluationProvider[ObservationT],
    checkpoint_provider: CheckpointProvider[ObservationT],
) -> Node[ObservationT]:
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
    current: Node[ObservationT],
    archive: List[Node[ObservationT]],
    visited: Set[str],
    current_step: int,
    max_steps: int,
    samples_per_node: int,
    mutation_provider: MutationProvider[ObservationT],
    evaluation_provider: EvaluationProvider[ObservationT],
    checkpoint_provider: CheckpointProvider[ObservationT],
) -> Node[ObservationT]:
    """
    Internal search implementation. Can be called from any state.

    This function is idempotent and doesn't distinguish between "initial" vs "resume" -
    it simply continues from whatever state it receives.

    Args:
        current: Current node being explored
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
    print(f"Step {current_step}: Current={current.evaluation.reward}")

    for step in range(current_step, max_steps):
        children = expand_and_evaluate(
            current=current,
            samples_per_node=samples_per_node,
            mutation_provider=mutation_provider,
            evaluation_provider=evaluation_provider,
            visited=visited,
        )

        if not children:
            print(f"Step {step + 1}: No valid children in this batch, continuing")
            continue

        best_child = select_best_child(children)
        archive.extend(children)

        if current.evaluation.reward is None:
            print(
                f"Step {step + 1}: Moving to improved position (reward={best_child.evaluation.reward})"
            )
            current = best_child
        elif best_child.evaluation.reward > current.evaluation.reward:
            print(
                f"Step {step + 1}: Moving to improved position (reward={best_child.evaluation.reward})"
            )
            current = best_child
        else:
            print(
                f"Step {step + 1}: No improvement (best_child={best_child.evaluation.reward}, current={current.evaluation.reward}), continuing from current position"
            )

        checkpoint_provider.save(
            Checkpoint(
                current_node=current,
                archive=archive,
                visited=visited,
                current_step=step + 1,
            )
        )

    return max(archive, key=lambda n: n.evaluation.reward if n.evaluation.reward is not None else float("-inf"))


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
    valid_nodes = [n for n in archive if n.evaluation.reward is not None]
    failed_nodes = [n for n in archive if n.evaluation.reward is None]

    return {
        "total_nodes": len(archive),
        "valid_nodes": len(valid_nodes),
        "failed_nodes": len(failed_nodes),
        "best_reward": max(
            cast(List[float], [n.evaluation.reward for n in valid_nodes]), default=None
        ),
        "unique_programs": len({get_content_key(n) for n in archive}),
    }
