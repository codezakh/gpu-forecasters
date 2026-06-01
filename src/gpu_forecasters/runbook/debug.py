"""Debug-mode overrides for runbook configs.

Each runbook script accepts ``--debug``. The flag applies the smallest
override that exercises the same execution path the real run uses,
without making a long, expensive call. Backends are *not* stubbed —
the purpose is end-to-end plumbing verification, not unit testing.

The :func:`apply_debug_overrides` helper dispatches on config type so
every script's ``--debug`` flag goes through one canonical place.
"""

from __future__ import annotations

from typing import TypeVar

from gpu_forecasters.runbook.configs import (
    BaselineScoringConfig,
    DiscoveryScoringConfig,
    FigureConfig,
    KernelSearchConfig,
    TrainedScoringConfig,
    TrainingRunConfig,
    UpstreamPuctConfig,
)


_DEBUG_PUCT_OVERRIDES = dict(
    total_budget_steps=1,
    batch_size=1,
    samples_per_parent=1,
    k_per_parent=1,
    max_llm_concurrency=1,
)


_DEBUG_TRAINING_OVERRIDES = dict(
    num_iters=1,
    group_size=2,
    groups_per_batch=1,
    max_tokens=1024,
    save_every=1,
)


T = TypeVar("T")


def apply_debug_overrides(config: T) -> T:
    """Return a new config with debug-mode overrides applied.

    Returns the original config untouched when the type is not one of
    the runbook configs — useful so a caller can blanket-apply this
    helper without per-type branching.
    """
    if isinstance(config, BaselineScoringConfig):
        return config.model_copy(update=dict(n_repeats=1, max_concurrency=1))  # type: ignore[return-value]
    if isinstance(config, TrainedScoringConfig):
        return config.model_copy(update=dict(n_repeats=1, max_concurrency=1))  # type: ignore[return-value]
    if isinstance(config, DiscoveryScoringConfig):
        return config.model_copy(update=dict(n_repeats=1, max_concurrency=1))  # type: ignore[return-value]
    if isinstance(config, TrainingRunConfig):
        return config.model_copy(update=_DEBUG_TRAINING_OVERRIDES)  # type: ignore[return-value]
    if isinstance(config, UpstreamPuctConfig):
        return config.model_copy(update=_DEBUG_PUCT_OVERRIDES)  # type: ignore[return-value]
    if isinstance(config, KernelSearchConfig):
        return config.model_copy(update=_DEBUG_PUCT_OVERRIDES)  # type: ignore[return-value]
    if isinstance(config, FigureConfig):
        # Figures don't have a meaningful "smaller" mode; debug means
        # render against a tiny pre-shipped fixture, handled in the
        # script itself.
        return config
    raise TypeError(f"no debug override registered for {type(config).__name__}")


# Caps that the scoring scripts apply to dataset rows when ``--debug``
# is set. The config-level override above (n_repeats=1, concurrency=1)
# scopes one axis; these cap the other.
DEBUG_MAX_ROWS_PER_PACK_BASELINE = 1
DEBUG_MAX_ROWS_TRAINED = 1
DEBUG_MAX_ROWS_DISCOVERY = 5
DEBUG_MAX_TRAINING_ROWS = 4


__all__ = [
    "DEBUG_MAX_ROWS_DISCOVERY",
    "DEBUG_MAX_ROWS_PER_PACK_BASELINE",
    "DEBUG_MAX_ROWS_TRAINED",
    "DEBUG_MAX_TRAINING_ROWS",
    "apply_debug_overrides",
]
