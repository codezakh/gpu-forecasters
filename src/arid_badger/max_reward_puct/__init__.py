"""Max-Reward PUCT search algorithm for program optimization."""

from arid_badger.max_reward_puct.search import (
    search,
    select_batch_of_parents,
    expand_and_evaluate,
    update_archive,
    flush_archive,
    backpropagate,
    record_failed_rollout,
    calculate_puct_scores,
    get_global_scale,
    get_rank_prior,
    get_ancestor_ids,
    build_children_map,
    get_full_lineage,
    get_content_key,
    set_parent_info,
)

__all__ = [
    "search",
    "select_batch_of_parents",
    "expand_and_evaluate",
    "update_archive",
    "flush_archive",
    "backpropagate",
    "record_failed_rollout",
    "calculate_puct_scores",
    "get_global_scale",
    "get_rank_prior",
    "get_ancestor_ids",
    "build_children_map",
    "get_full_lineage",
    "get_content_key",
    "set_parent_info",
]
