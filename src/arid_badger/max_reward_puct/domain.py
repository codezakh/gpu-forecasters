from pydantic import BaseModel, Field
from ulid import ULID
from typing import Optional, List, Sequence, Set, Mapping, Tuple, Dict, Protocol
import numpy as np
import numpy.typing as npt
import math
from typing import Callable
from collections import defaultdict


# Provider Protocols
#
# We use dependency injection with Protocol-based providers to enable testing
# with toy examples (e.g., Binary Graph Traversal) before integrating real
# LLM mutation and KernelBench evaluation.
#
# Design notes:
# - Intentionally simpler than greedy_search providers (no context objects,
#   no complex result types) - focus on testability, can evolve later
# - EvaluationProvider is separate (not just a function) because for kernel
#   optimization, scoring the initial/reference program sometimes requires
#   different logic than scoring mutations (e.g., running multiple times for
#   stable baseline, using different compilation flags)
# - Returns Optional[float] to handle evaluation failures gracefully (None
#   signals compilation errors, crashes, etc.)


class MutationProvider(Protocol):
    """Generates mutations of a program."""

    def generate_mutations(self, program_code: str, num_mutations: int) -> List[str]:
        """Returns list of mutated program codes."""
        ...


class EvaluationProvider(Protocol):
    """Evaluates a program and returns its reward."""

    def evaluate(self, program_code: str) -> Optional[float]:
        """Returns reward (or None if evaluation failed)."""
        ...


class AncestorPointer(BaseModel):
    """
    Represents a pointer to an ancestor node.
    """

    ulid: ULID
    timestep: int


class Node(BaseModel):
    """
    Represents a specific program/kernel solution.
    """

    program_code: str

    # R(s): The intrinsic reward of this specific program.
    # Used for calculating the Rank Prior P(s).
    # Can be None if evaluation failed.
    reward: Optional[float]

    # Ancestor chain: list of {"id": str, "timestep": int} dicts,
    # most recent parent first.  Matches State.parents.
    ancestors: List[AncestorPointer]

    ulid: ULID = Field(default_factory=ULID)
    is_seed: bool = False


# --- External PUCT statistics (not stored on nodes) ---
# These persist independently of archive membership: evicted nodes keep their counts.
# n: Dict[ULID, int] = {}  # n[id] = visit count for state id
# m: Dict[ULID, float] = {}  # m[id] = best one-step child reward for state id
# T: int = 0               # Global expansion counter


def get_global_scale(archive: Sequence[Node], seed_ids: Set[ULID]) -> float:
    """
    Calculates (R_max - R_min) from non-seed states in the archive.
    Seed states are excluded to prevent randomly-initialized seeds
    from distorting the scale.
    """
    non_seed_rewards = [
        s.reward for s in archive if s.ulid not in seed_ids and s.reward is not None
    ]
    if not non_seed_rewards:
        # Fall back to full archive if no non-seed states exist yet
        non_seed_rewards = [s.reward for s in archive if s.reward is not None]
    if not non_seed_rewards:
        return 1.0
    return max(1e-6, max(non_seed_rewards) - min(non_seed_rewards))


def get_rank_prior(rewards: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Calculates P(s) for all nodes based on rank.
    Does NOT assume the archive is pre-sorted — computes ranks via argsort.
    Formula: P(s) = (N - rank) / sum(N - rank for all s')
    Matches _compute_prior (sampler.py:434-440).
    """
    if rewards.size == 0:
        return np.array([])
    N = len(rewards)
    # argsort(argsort(-values)) gives rank 0 to the best (highest) value
    ranks = np.argsort(np.argsort(-rewards))
    weights = (N - ranks).astype(np.float64)
    return weights / weights.sum()


def calculate_puct_scores(
    archive: Sequence[Node],
    visit_counts: Mapping[ULID, int],
    best_child_rewards: Mapping[ULID, float],
    global_expansion_count: int,
    seed_ids: Set[ULID],
    c_puct: float = 1.0,
) -> List[Tuple[float, float, Node]]:
    """
    Scores every node in the archive. Returns list of (score, reward, node)
    sorted descending by (score, reward) for tiebreaking.
    """
    # Treat None rewards as -inf for scoring purposes
    rewards: npt.NDArray[np.float64] = np.array(
        [float(s.reward) if s.reward is not None else float("-inf") for s in archive]
    )
    scale = get_global_scale(archive, seed_ids)
    rank_prior = get_rank_prior(rewards)
    sqrt_T = math.sqrt(1.0 + global_expansion_count)

    scored = []
    for i, node in enumerate(archive):
        visit_count = visit_counts.get(node.ulid, 0)
        best_child_reward = best_child_rewards.get(node.ulid, rewards[i])
        Q = best_child_reward if visit_count > 0 else rewards[i]
        bonus = c_puct * scale * rank_prior[i] * sqrt_T / (1.0 + visit_count)
        score = Q + bonus
        scored.append((score, rewards[i], node))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored


def get_ancestor_ids(node: Node) -> Set[ULID]:
    """Read ancestor IDs directly from the parents list."""
    return {p.ulid for p in node.ancestors}


def build_children_map(archive: Sequence[Node]) -> Dict[ULID, Set[ULID]]:
    """
    Scan the archive to build a parent -> children mapping.
    Matches _build_children_map (sampler.py:449-456).
    """
    children: Dict[ULID, Set[ULID]] = {}
    for s in archive:
        for p in s.ancestors:
            pid = p.ulid
            children.setdefault(pid, set()).add(s.ulid)
    return children


def get_full_lineage(node: Node, children_map: Dict[ULID, Set[ULID]]) -> Set[ULID]:
    """
    Returns IDs of the node itself, all ancestors, and all descendants.
    Ancestors come from node.parents; descendants come from BFS over children_map.
    """
    lineage = {node.ulid} | get_ancestor_ids(node)

    # BFS for descendants
    queue = [node.ulid]
    visited = {node.ulid}
    while queue:
        sid = queue.pop(0)
        for child_id in children_map.get(sid, []):
            if child_id not in visited:
                visited.add(child_id)
                lineage.add(child_id)
                queue.append(child_id)

    return lineage


def select_batch_of_parents(
    archive: Sequence[Node],
    batch_size: int,
    visit_counts: Mapping[ULID, int],
    best_child_rewards: Mapping[ULID, float],
    global_expansion_count: int,
    seed_ids: Set[ULID],
    c_puct: float = 1.0,
) -> List[Node]:
    """
    Selects 'batch_size' nodes to expand.
    Enforces LINEAGE BLOCKING: No two nodes in a batch can share an
    ancestor-descendant relationship.
    """
    scored = calculate_puct_scores(
        archive,
        visit_counts,
        best_child_rewards,
        global_expansion_count,
        seed_ids,
        c_puct,
    )

    # batch_size == 1: no blocking needed, just pick the top scorer
    if batch_size == 1:
        return [scored[0][2]] if scored else []

    # batch_size > 1: build children map, greedily pick with lineage blocking
    children_map = build_children_map(archive)
    selected: List[Node] = []
    blocked_ids: Set[ULID] = set()

    for _score, _reward, node in scored:
        if node.ulid in blocked_ids:
            continue
        selected.append(node)
        blocked_ids.update(get_full_lineage(node, children_map))
        if len(selected) >= batch_size:
            break

    return selected


def expand_and_evaluate(
    parents: List[Node],
    samples_per_parent: int,
    mutation_provider: MutationProvider,
    evaluation_provider: EvaluationProvider,
) -> Tuple[List[Node], List[Node]]:
    """
    Generates mutations (via LLM) and evaluates them.
    Returns (children, parent_of_each_child) as parallel lists.

    The mutation and evaluation providers are injected here to allow different
    implementations for testing vs. production (toy examples vs. real kernels).
    """
    all_children: List[Node] = []
    all_parents: List[Node] = []

    for parent in parents:
        # LLM Generation Step
        program_candidates = mutation_provider.generate_mutations(
            parent.program_code, samples_per_parent
        )

        for code in program_candidates:
            # Execution Sandbox Step — reward may be None on failure
            reward = evaluation_provider.evaluate(code)

            child = Node(
                program_code=code,
                reward=reward,
                # parents set later in update_archive via _set_parent_info
                ancestors=[],
            )
            all_children.append(child)
            all_parents.append(parent)

    return all_children, all_parents


def backpropagate(
    children: List[Node],
    parent_states: List[Node],
    n: Dict[ULID, int],
    m: Dict[ULID, float],
    T: int,
) -> int:
    """
    Updates m, n, and T after expansion.
    Skips children with reward is None.
    Returns the updated T.
    Matches update_states (sampler.py:534-550).
    """
    # Find best valid child reward per parent
    parent_max: Dict[ULID, float] = {}
    parent_obj: Dict[ULID, Node] = {}
    for child, parent in zip(children, parent_states):
        if child.reward is None:
            continue
        pid = parent.ulid
        parent_obj[pid] = parent
        parent_max[pid] = max(parent_max.get(pid, float("-inf")), child.reward)

    for pid, y in parent_max.items():
        # 1. Update m(s) for the DIRECT PARENT ONLY
        m[pid] = max(m.get(pid, y), y)

        # 2. Increment visit counts for parent and ALL ancestors
        parent = parent_obj[pid]
        anc_ids = [pid] + [p.ulid for p in parent.ancestors]
        for aid in anc_ids:
            n[aid] = n.get(aid, 0) + 1

        # 3. Increment global expansion counter
        T += 1

    return T


def record_failed_rollout(parent: Node, n: Dict[ULID, int], T: int) -> int:
    """
    Called when an expansion fails to produce any valid children.
    Still increments n(s) and T so the node's exploration bonus decays,
    preventing nodes that always fail from being selected indefinitely.
    Matches record_failed_rollout (sampler.py:619-623).
    """
    anc_ids = [parent.ulid] + [p.ulid for p in parent.ancestors]
    for aid in anc_ids:
        n[aid] = n.get(aid, 0) + 1
    return T + 1


def get_content_key(node: Node) -> str:
    """Returns a hashable key for deduplication (e.g. program code)."""
    return node.program_code


def set_parent_info(child: Node, parent: Node):
    """
    Sets the child's parents list to [parent_ref] + parent.parents.
    Matches _set_parent_info (sampler.py:52-55).
    """
    child.ancestors = [AncestorPointer(ulid=parent.ulid, timestep=0)] + list(
        parent.ancestors
    )


def update_archive(
    archive: List[Node],
    children: List[Node],
    parent_states: List[Node],
    seed_ids: Set[ULID],
    capacity: int = 1000,
    k_per_parent: int = 2,
):
    """
    Updates the global archive with new children, enforcing:
    1. Top-K children per parent (by reward).
    2. Skip children with reward is None.
    3. Deduplication by program content.
    4. Global capacity by intrinsic reward (seed states always kept).
    Matches update_states (sampler.py:555-583).
    """
    # 1. Top-k filter: keep only best k children per parent
    by_parent: Dict[ULID, List[Tuple[Node, Node]]] = defaultdict(list)
    for child, parent in zip(children, parent_states):
        by_parent[parent.ulid].append((child, parent))

    filtered_pairs: List[Tuple[Node, Node]] = []
    for pairs in by_parent.values():
        pairs.sort(
            key=lambda x: x[0].reward if x[0].reward is not None else float("-inf"),
            reverse=True,
        )
        filtered_pairs.extend(pairs[:k_per_parent])

    # 2. Skip None rewards + deduplication
    existing_keys = {get_content_key(s) for s in archive}

    for child, parent in filtered_pairs:
        if child.reward is None:
            continue
        key = get_content_key(child)
        if key is not None and key in existing_keys:
            continue
        set_parent_info(child, parent)
        archive.append(child)
        if key is not None:
            existing_keys.add(key)

    # 3. Global truncation by reward, preserving seed states
    if len(archive) > capacity:
        rewards = [s.reward if s.reward is not None else float("-inf") for s in archive]
        ranked = list(np.argsort(rewards)[::-1])  # indices sorted by reward desc
        keep = {i for i, s in enumerate(archive) if s.ulid in seed_ids}
        for i in ranked:
            if len(keep) >= capacity:
                break
            keep.add(i)
        archive[:] = [archive[i] for i in sorted(keep)]


def flush_archive(
    archive: List[Node],
    seed_ids: Set[ULID],
    capacity: int = 1000,
    k_per_parent: int = 2,
):
    """
    Periodic cleanup: re-applies top-k per parent across the ENTIRE archive,
    then truncates to capacity. This retroactively evicts children that were
    top-k at insertion time but have since been surpassed.
    Matches flush (sampler.py:601-617).
    """
    by_parent: Dict[ULID, List[Node]] = {}
    no_parent: List[Node] = []
    for s in archive:
        pid = s.ancestors[0].ulid if s.ancestors else None
        if pid:
            by_parent.setdefault(pid, []).append(s)
        else:
            no_parent.append(s)

    filtered = list(no_parent)
    for siblings in by_parent.values():
        siblings.sort(
            key=lambda x: x.reward if x.reward is not None else float("-inf"),
            reverse=True,
        )
        filtered.extend(siblings[:k_per_parent])

    archive[:] = filtered

    # Re-apply global truncation
    if len(archive) > capacity:
        values = [s.reward if s.reward is not None else float("-inf") for s in archive]
        ranked = list(np.argsort(values)[::-1])
        keep = {i for i, s in enumerate(archive) if s.ulid in seed_ids}
        for i in ranked:
            if len(keep) >= capacity:
                break
            keep.add(i)
        archive[:] = [archive[i] for i in sorted(keep)]


def search(
    initial_program: str,
    total_budget_steps: int,
    batch_size: int,
    samples_per_parent: int,
    mutation_provider: MutationProvider,
    evaluation_provider: EvaluationProvider,
):
    # Initialization
    r_init = evaluation_provider.evaluate(initial_program)
    root = Node(program_code=initial_program, reward=r_init, is_seed=True, ancestors=[])

    archive = [root]
    seed_ids = {root.ulid}

    # External PUCT statistics
    n: Dict[ULID, int] = {}
    m: Dict[ULID, float] = {}
    T: int = 0

    for step in range(total_budget_steps):
        # A. SELECTION
        # Select parents with lineage blocking
        parents = select_batch_of_parents(archive, batch_size, n, m, T, seed_ids)

        if not parents:
            break

        # B. EXPANSION
        # Generate and evaluate children
        children, parent_states = expand_and_evaluate(
            parents, samples_per_parent, mutation_provider, evaluation_provider
        )

        # C. BACKPROPAGATION
        # Update m for direct parents, n for all ancestors, T += 1 per parent
        T = backpropagate(children, parent_states, n, m, T)

        # C'. FAILED ROLLOUT HANDLING
        # If a parent produced zero valid children, still decay its exploration bonus
        valid_by_parent: Dict[ULID, int] = defaultdict(int)
        for child, parent in zip(children, parent_states):
            if child.reward is not None:
                valid_by_parent[parent.ulid] += 1
        for parent in parents:
            if valid_by_parent.get(parent.ulid, 0) == 0:
                T = record_failed_rollout(parent, n, T)

        # D. ARCHIVE UPDATE
        # Filter (Top-K per parent), skip None rewards, deduplicate, truncate
        update_archive(archive, children, parent_states, seed_ids)

        best = max(
            archive, key=lambda s: s.reward if s.reward is not None else float("-inf")
        )
        print(f"Step {step}: Best Reward = {best.reward}")

    return max(
        archive, key=lambda s: s.reward if s.reward is not None else float("-inf")
    )
