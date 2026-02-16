"""
Depth-First Greedy Search (Hill Climbing) for program optimization.

This module implements a simple baseline search algorithm that:
1. Generates mutations of the current program
2. Picks the best mutation (greedy choice)
3. Continues from that best mutation (depth-first)
4. Stops when reaching max depth or finding no improvement (local maximum)

Uses the same provider-based architecture as max_reward_puct for consistency.
"""

from typing import List, Set, cast
from arid_badger.max_reward_puct.domain import (
    MutationProvider,
    EvaluationProvider,
    Node,
    get_content_key,
    set_parent_info,
)
from arid_badger.hill_climbing.checkpoint import (
    Checkpoint,
    CheckpointProvider,
    NoOpCheckpointProvider,
)


def search(
    initial_program: str,
    max_depth: int,
    samples_per_node: int,
    mutation_provider: MutationProvider,
    evaluation_provider: EvaluationProvider,
    checkpoint_provider: CheckpointProvider = NoOpCheckpointProvider(),
) -> Node:
    """
    Performs depth-first greedy search (hill climbing) from an initial program.

    Algorithm:
    1. Start with initial program and evaluate it
    2. For each depth step (up to max_depth):
       - Generate samples_per_node mutations of current program
       - Evaluate all mutations
       - Filter out failed evaluations (reward is None)
       - Skip duplicates using content-based deduplication
       - Pick the mutation with highest reward (greedy)
       - If no valid children OR no improvement over current → stop (local maximum)
       - Move to best child and repeat (depth-first)
    3. Return the best node found during entire search

    Args:
        initial_program: Starting program code
        max_depth: Maximum number of iterations
        samples_per_node: Number of mutations to generate at each step
        mutation_provider: Provider for generating mutations
        evaluation_provider: Provider for evaluating programs
        checkpoint_provider: Provider for saving/loading checkpoints (default: no-op)

    Returns:
        Best node found during search (global best, not just final node)
    """
    # Initialize state
    initial_reward = evaluation_provider.evaluate(initial_program)
    current = Node(
        program_code=initial_program, reward=initial_reward, ancestors=[], is_seed=True
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
        current_depth=0,
        max_depth=max_depth,
        samples_per_node=samples_per_node,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
        checkpoint_provider=checkpoint_provider,
    )


def resume_search(
    checkpoint: Checkpoint,
    max_depth: int,
    samples_per_node: int,
    mutation_provider: MutationProvider,
    evaluation_provider: EvaluationProvider,
    checkpoint_provider: CheckpointProvider = NoOpCheckpointProvider(),
) -> Node:
    """
    Resume search from a checkpoint.

    Args:
        checkpoint: Checkpoint containing state to resume from
        max_depth: Maximum number of iterations (total, including already completed)
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
        current_depth=checkpoint.current_depth,
        max_depth=max_depth,
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
    current_depth: int,
    max_depth: int,
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
        current_depth: Current iteration depth
        max_depth: Maximum number of iterations
        samples_per_node: Number of mutations to generate at each step
        mutation_provider: Provider for generating mutations
        evaluation_provider: Provider for evaluating programs
        checkpoint_provider: Provider for saving checkpoints

    Returns:
        Best node found during search
    """
    print(f"Depth {current_depth}: Current={current.reward}, Best={best.reward}")

    for depth in range(current_depth, max_depth):
        # Generate mutations
        mutation_codes = mutation_provider.generate_mutations(
            current.program_code, samples_per_node
        )

        # Evaluate mutations and build valid children
        children: List[Node] = []
        for code in mutation_codes:
            # Skip duplicates
            content_key = code  # For binary strings, code is the key
            if content_key in visited:
                continue

            # Evaluate
            reward = evaluation_provider.evaluate(code)
            if reward is None:
                # Skip failed evaluations
                continue

            # Create child node
            child = Node(program_code=code, reward=reward, ancestors=[])
            set_parent_info(child, current)
            children.append(child)
            visited.add(content_key)

        # Check for dead end (no valid children)
        if not children:
            print(f"Depth {depth + 1}: No valid children found, stopping")
            break

        # Greedy selection: pick best child
        best_child = max(
            children, key=lambda c: c.reward if c.reward is not None else float("-inf")
        )
        archive.extend(children)

        # Update global best
        if best_child.reward is not None and (
            best.reward is None or best_child.reward > best.reward
        ):
            best = best_child

        print(f"Depth {depth + 1}: Current={best_child.reward}, Best={best.reward}")

        # Check for local maximum (no improvement)
        if (
            current.reward is not None
            and best_child.reward is not None
            and best_child.reward <= current.reward
        ):
            print(f"Depth {depth + 1}: Local maximum reached, stopping")
            break

        # Move to best child (depth-first)
        current = best_child

        # Save checkpoint after each iteration
        checkpoint_provider.save(
            Checkpoint(
                current_node=current,
                best_node=best,
                archive=archive,
                visited=visited,
                current_depth=depth + 1,
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
