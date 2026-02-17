from .domain import (
    Node,
    ObservationT,
    MutationProvider,
    EvaluationProvider,
    CheckpointProvider,
    Checkpoint,
)
from typing import List, Set, Optional, Any, cast


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
       - Select the best node in the archive as the expansion point
       - Generate samples_per_node mutations from it
       - Evaluate all mutations
       - Filter out failed evaluations (reward is None)
       - Skip duplicates using content-based deduplication
       - Add valid mutations to the archive
    3. Return the best node in the archive
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
    initial_evaluation = evaluation_provider.evaluate(initial_program)
    seed = Node(
        program_code=initial_program,
        evaluation=initial_evaluation,
        ancestors=[],
        is_seed=True,
    )
    return _search_impl(
        archive=[seed],
        visited={seed.program_code},
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
        archive=checkpoint.archive,
        visited=checkpoint.visited,
        current_step=checkpoint.current_step,
        max_steps=max_steps,
        samples_per_node=samples_per_node,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
        checkpoint_provider=checkpoint_provider,
    )


def select_best(archive: List[Node[ObservationT]]) -> Node[ObservationT]:
    return max(
        archive,
        key=lambda n: (
            n.evaluation.reward if n.evaluation.reward is not None else float("-inf")
        ),
    )


def log_changes(
    step: int, prev_best: Node[ObservationT], new_best: Node[ObservationT]
) -> None:
    if prev_best.ulid != new_best.ulid:
        print(
            f"Step {step + 1}: Reward changed from {prev_best.evaluation.reward:.4f} to {new_best.evaluation.reward:.4f}"
        )
    else:
        print(f"Step {step + 1}: Reward unchanged at {new_best.evaluation.reward:.4f}")


class ChangeLogger:
    def __init__(self, step: int, initial: Optional[Node[Any]] = None):
        self.step = step
        self.initial = initial

    @staticmethod
    def _safe_fmt_reward(reward: Optional[float]) -> str:
        return f"{reward:.4f}" if reward is not None else "None"

    def __call__(self, new_best: Node[Any]) -> None:
        if self.initial is None:
            print(
                f"Step {self.step + 1}: Starting search, best reward={self._safe_fmt_reward(new_best.evaluation.reward)}"
            )
        else:
            if new_best.ulid != self.initial.ulid:
                print(
                    f"Step {self.step + 1}: Reward changed from {self._safe_fmt_reward(self.initial.evaluation.reward)} to {self._safe_fmt_reward(new_best.evaluation.reward)}"
                )
            else:
                print(
                    f"Step {self.step + 1}: Reward unchanged at {self._safe_fmt_reward(new_best.evaluation.reward)}"
                )
        self.initial = new_best


def _search_impl(
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
        archive: List of all nodes explored so far
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
    change_logger = ChangeLogger(current_step)
    for step in range(current_step, max_steps):
        to_expand = select_best(archive)
        change_logger(to_expand)
        print(
            f"Step {step}: Expanding best node (reward={to_expand.evaluation.reward})"
        )

        children = expand_and_evaluate(
            current=to_expand,
            samples_per_node=samples_per_node,
            mutation_provider=mutation_provider,
            evaluation_provider=evaluation_provider,
            visited=visited,
        )

        if not children:
            print(f"Step {step + 1}: No valid children in this batch, continuing")
            continue

        archive.extend(children)
        checkpoint_provider.save(
            Checkpoint(
                archive=archive,
                visited=visited,
                current_step=step + 1,
            )
        )

    return select_best(archive)


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
