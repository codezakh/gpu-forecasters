from ulid import ULID
from typing import List, Sequence, Set, Mapping, Tuple, Dict
import numpy as np
import numpy.typing as npt
import math

from loguru import logger

from collections import defaultdict

from arid_badger.hill_climbing.domain import (
    Node,
    MutationProvider,
    EvaluationProvider,
    ObservationT,
)
from arid_badger.max_reward_puct.checkpoint import (
    PuctCheckpoint,
    PuctCheckpointProvider,
    NoOpPuctCheckpointProvider,
)


def get_global_scale(archive: Sequence[Node[ObservationT]], seed_ids: Set[ULID]) -> float:
    """
    Calculates (R_max - R_min) from non-seed states in the archive.
    Seed states are excluded to prevent randomly-initialized seeds
    from distorting the scale.
    """
    non_seed_rewards = [
        s.evaluation.reward
        for s in archive
        if s.ulid not in seed_ids and s.evaluation.reward is not None
    ]
    if not non_seed_rewards:
        # Fall back to full archive if no non-seed states exist yet
        non_seed_rewards = [s.evaluation.reward for s in archive if s.evaluation.reward is not None]
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
    archive: Sequence[Node[ObservationT]],
    visit_counts: Mapping[ULID, int],
    best_child_rewards: Mapping[ULID, float],
    global_expansion_count: int,
    seed_ids: Set[ULID],
    c_puct: float = 1.0,
) -> List[Tuple[float, float, Node[ObservationT]]]:
    """
    Scores every node in the archive. Returns list of (score, reward, node)
    sorted descending by (score, reward) for tiebreaking.
    """
    # Treat None rewards as -inf for scoring purposes
    rewards: npt.NDArray[np.float64] = np.array(
        [
            float(s.evaluation.reward) if s.evaluation.reward is not None else float("-inf")
            for s in archive
        ]
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


def get_ancestor_ids(node: Node[ObservationT]) -> Set[ULID]:
    """Read ancestor IDs directly from the ancestors list."""
    return set(node.ancestors)


def build_children_map(archive: Sequence[Node[ObservationT]]) -> Dict[ULID, Set[ULID]]:
    """
    Scan the archive to build a parent -> children mapping.
    Matches _build_children_map (sampler.py:449-456).
    """
    children: Dict[ULID, Set[ULID]] = {}
    for s in archive:
        for p in s.ancestors:
            children.setdefault(p, set()).add(s.ulid)
    return children


def get_full_lineage(
    node: Node[ObservationT], children_map: Dict[ULID, Set[ULID]]
) -> Set[ULID]:
    """
    Returns IDs of the node itself, all ancestors, and all descendants.
    Ancestors come from node.ancestors; descendants come from BFS over children_map.
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
    archive: Sequence[Node[ObservationT]],
    batch_size: int,
    visit_counts: Mapping[ULID, int],
    best_child_rewards: Mapping[ULID, float],
    global_expansion_count: int,
    seed_ids: Set[ULID],
    c_puct: float = 1.0,
) -> List[Node[ObservationT]]:
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
    selected: List[Node[ObservationT]] = []
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
    parents: List[Node[ObservationT]],
    samples_per_parent: int,
    mutation_provider: MutationProvider[ObservationT],
    evaluation_provider: EvaluationProvider[ObservationT],
) -> Tuple[List[Node[ObservationT]], List[Node[ObservationT]]]:
    """
    Generates mutations (via LLM) and evaluates them.
    Returns (children, parent_of_each_child) as parallel lists.

    The mutation and evaluation providers are injected here to allow different
    implementations for testing vs. production (toy examples vs. real kernels).
    """
    # Step 1: collect all (parent, candidate_code) pairs. Mutation stays serial.
    parent_of_code: List[Node[ObservationT]] = []
    candidate_codes: List[str] = []
    for parent in parents:
        for code in mutation_provider.generate_mutations(
            parent.program_code, samples_per_parent, parent.evaluation
        ):
            parent_of_code.append(parent)
            candidate_codes.append(code)

    # Step 2: evaluate the whole batch in one provider call. The provider
    # decides whether that's sequential or parallelised.
    evaluations = evaluation_provider.batch_evaluate(candidate_codes)

    # Step 3: zip back together. Order preservation is guaranteed by the
    # batch_evaluate contract.
    children: List[Node[ObservationT]] = [
        Node(program_code=code, evaluation=ev, ancestors=[])
        for code, ev in zip(candidate_codes, evaluations)
    ]
    return children, parent_of_code


def backpropagate(
    children: List[Node[ObservationT]],
    parent_states: List[Node[ObservationT]],
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
    parent_obj: Dict[ULID, Node[ObservationT]] = {}
    for child, parent in zip(children, parent_states):
        if child.evaluation.reward is None:
            continue
        pid = parent.ulid
        parent_obj[pid] = parent
        parent_max[pid] = max(
            parent_max.get(pid, float("-inf")), child.evaluation.reward
        )

    for pid, y in parent_max.items():
        # 1. Update m(s) for the DIRECT PARENT ONLY
        m[pid] = max(m.get(pid, y), y)

        # 2. Increment visit counts for parent and ALL ancestors
        parent = parent_obj[pid]
        anc_ids = [pid] + list(parent.ancestors)
        for aid in anc_ids:
            n[aid] = n.get(aid, 0) + 1

        # 3. Increment global expansion counter
        T += 1

    return T


def record_failed_rollout(
    parent: Node[ObservationT], n: Dict[ULID, int], T: int
) -> int:
    """
    Called when an expansion fails to produce any valid children.
    Still increments n(s) and T so the node's exploration bonus decays,
    preventing nodes that always fail from being selected indefinitely.
    Matches record_failed_rollout (sampler.py:619-623).
    """
    anc_ids = [parent.ulid] + list(parent.ancestors)
    for aid in anc_ids:
        n[aid] = n.get(aid, 0) + 1
    return T + 1


def get_content_key(node: Node[ObservationT]) -> str:
    """Returns a hashable key for deduplication (e.g. program code)."""
    return node.program_code


def set_parent_info(child: Node[ObservationT], parent: Node[ObservationT]):
    """
    Sets the child's ancestors list to [parent.ulid] + parent.ancestors.
    Matches _set_parent_info (sampler.py:52-55).
    """
    child.ancestors = [parent.ulid] + list(parent.ancestors)


def update_archive(
    archive: List[Node[ObservationT]],
    children: List[Node[ObservationT]],
    parent_states: List[Node[ObservationT]],
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
    by_parent: Dict[ULID, List[Tuple[Node[ObservationT], Node[ObservationT]]]] = defaultdict(list)
    for child, parent in zip(children, parent_states):
        by_parent[parent.ulid].append((child, parent))

    filtered_pairs: List[Tuple[Node[ObservationT], Node[ObservationT]]] = []
    for pairs in by_parent.values():
        pairs.sort(
            key=lambda x: x[0].evaluation.reward
            if x[0].evaluation.reward is not None
            else float("-inf"),
            reverse=True,
        )
        filtered_pairs.extend(pairs[:k_per_parent])

    # 2. Skip None rewards + deduplication
    existing_keys = {get_content_key(s) for s in archive}

    for child, parent in filtered_pairs:
        if child.evaluation.reward is None:
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
        rewards = [
            s.evaluation.reward if s.evaluation.reward is not None else float("-inf")
            for s in archive
        ]
        ranked = list(np.argsort(rewards)[::-1])  # indices sorted by reward desc
        keep = {i for i, s in enumerate(archive) if s.ulid in seed_ids}
        for i in ranked:
            if len(keep) >= capacity:
                break
            keep.add(i)
        archive[:] = [archive[i] for i in sorted(keep)]


def flush_archive(
    archive: List[Node[ObservationT]],
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
    by_parent: Dict[ULID, List[Node[ObservationT]]] = {}
    no_parent: List[Node[ObservationT]] = []
    for s in archive:
        pid = s.ancestors[0] if s.ancestors else None
        if pid:
            by_parent.setdefault(pid, []).append(s)
        else:
            no_parent.append(s)

    filtered = list(no_parent)
    for siblings in by_parent.values():
        siblings.sort(
            key=lambda x: x.evaluation.reward
            if x.evaluation.reward is not None
            else float("-inf"),
            reverse=True,
        )
        filtered.extend(siblings[:k_per_parent])

    archive[:] = filtered

    # Re-apply global truncation
    if len(archive) > capacity:
        values = [
            s.evaluation.reward if s.evaluation.reward is not None else float("-inf")
            for s in archive
        ]
        ranked = list(np.argsort(values)[::-1])
        keep = {i for i, s in enumerate(archive) if s.ulid in seed_ids}
        for i in ranked:
            if len(keep) >= capacity:
                break
            keep.add(i)
        archive[:] = [archive[i] for i in sorted(keep)]


def _search_impl(
    archive: List[Node[ObservationT]],
    seed_ids: Set[ULID],
    n: Dict[ULID, int],
    m: Dict[ULID, float],
    T: int,
    current_step: int,
    total_budget_steps: int,
    batch_size: int,
    samples_per_parent: int,
    mutation_provider: MutationProvider[ObservationT],
    evaluation_provider: EvaluationProvider[ObservationT],
    checkpoint_provider: PuctCheckpointProvider[ObservationT],
) -> Node[ObservationT]:
    """
    Internal search implementation. Can be called from any state.

    This function is idempotent and doesn't distinguish between "initial" vs "resume" —
    it simply continues from whatever state it receives.
    """
    for step in range(current_step, total_budget_steps):
        # A. SELECTION
        # Select parents with lineage blocking
        parents = select_batch_of_parents(archive, batch_size, n, m, T, seed_ids)

        if not parents:
            break

        logger.info(
            "Step {step}/{total}: selecting {nparents} parent(s), launching {nevals} evaluation(s).",
            step=step,
            total=total_budget_steps,
            nparents=len(parents),
            nevals=len(parents) * samples_per_parent,
        )

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
            if child.evaluation.reward is not None:
                valid_by_parent[parent.ulid] += 1
        for parent in parents:
            if valid_by_parent.get(parent.ulid, 0) == 0:
                T = record_failed_rollout(parent, n, T)

        # D. ARCHIVE UPDATE
        # Filter (Top-K per parent), skip None rewards, deduplicate, truncate
        update_archive(archive, children, parent_states, seed_ids)

        best = max(
            archive,
            key=lambda s: s.evaluation.reward
            if s.evaluation.reward is not None
            else float("-inf"),
        )
        logger.info(
            "Step {step}/{total} complete: archive_size={size}, best_reward={reward}.",
            step=step,
            total=total_budget_steps,
            size=len(archive),
            reward=f"{best.evaluation.reward:.4f}" if best.evaluation.reward is not None else "None",
        )

        # E. CHECKPOINT
        checkpoint_provider.save(
            PuctCheckpoint(
                archive=archive,
                seed_ids=seed_ids,
                visit_counts=dict(n),
                best_child_rewards=dict(m),
                global_expansion_count=T,
                current_step=step + 1,
            )
        )

    return max(
        archive,
        key=lambda s: s.evaluation.reward
        if s.evaluation.reward is not None
        else float("-inf"),
    )


def search(
    initial_program: str,
    total_budget_steps: int,
    batch_size: int,
    samples_per_parent: int,
    mutation_provider: MutationProvider[ObservationT],
    evaluation_provider: EvaluationProvider[ObservationT],
    checkpoint_provider: PuctCheckpointProvider[ObservationT] = NoOpPuctCheckpointProvider(),  # type: ignore[assignment]
) -> Node[ObservationT]:
    # Initialization
    eval_result = evaluation_provider.evaluate(initial_program)
    root: Node[ObservationT] = Node(
        program_code=initial_program,
        evaluation=eval_result,
        is_seed=True,
        ancestors=[],
    )

    archive: List[Node[ObservationT]] = [root]
    seed_ids = {root.ulid}

    # External PUCT statistics
    n: Dict[ULID, int] = {}
    m: Dict[ULID, float] = {}
    T: int = 0

    return _search_impl(
        archive=archive,
        seed_ids=seed_ids,
        n=n,
        m=m,
        T=T,
        current_step=0,
        total_budget_steps=total_budget_steps,
        batch_size=batch_size,
        samples_per_parent=samples_per_parent,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
        checkpoint_provider=checkpoint_provider,
    )


def resume_search(
    checkpoint: PuctCheckpoint[ObservationT],
    total_budget_steps: int,
    batch_size: int,
    samples_per_parent: int,
    mutation_provider: MutationProvider[ObservationT],
    evaluation_provider: EvaluationProvider[ObservationT],
    checkpoint_provider: PuctCheckpointProvider[ObservationT] = NoOpPuctCheckpointProvider(),  # type: ignore[assignment]
) -> Node[ObservationT]:
    """
    Resume a PUCT search from a saved checkpoint.

    Args:
        checkpoint: Checkpoint containing the search state to resume from.
        total_budget_steps: Total number of steps (including already completed).
        batch_size: Number of parents to select per step.
        samples_per_parent: Number of mutations per parent per step.
        mutation_provider: Provider for generating mutations.
        evaluation_provider: Provider for evaluating programs.
        checkpoint_provider: Provider for saving checkpoints during the resumed run.

    Returns:
        Best node found during the full search (including steps before the checkpoint).
    """
    return _search_impl(
        archive=list(checkpoint.archive),
        seed_ids=set(checkpoint.seed_ids),
        n=dict(checkpoint.visit_counts),
        m=dict(checkpoint.best_child_rewards),
        T=checkpoint.global_expansion_count,
        current_step=checkpoint.current_step,
        total_budget_steps=total_budget_steps,
        batch_size=batch_size,
        samples_per_parent=samples_per_parent,
        mutation_provider=mutation_provider,
        evaluation_provider=evaluation_provider,
        checkpoint_provider=checkpoint_provider,
    )
