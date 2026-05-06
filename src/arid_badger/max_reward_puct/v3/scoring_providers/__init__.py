"""v3 scoring providers.

The v3 ``EvaluationProvider`` protocol (see ``v3/providers.py``) is
intentionally identical to v2's: ``submit(program_code) ->
Future[Evaluation[ObservationT]]`` plus ``__enter__`` / ``__exit__``.
Existing v2 concretes (e.g. ``arid_badger.max_reward_puct.v2.
scoring_providers.trimul_modal``) satisfy the v3 protocol unchanged
and can be passed directly to ``v3.SearchDriver``. No adapter layer
is required for evaluation providers.

The v3 ``SpeedupEstimator`` protocol is also futures-based, but the
surrogate concretes in ``arid_badger.landscape_map.v2`` are async-
native (``async def aestimate``). ``CoroutineSpeedupEstimator``
adapts an ``AsyncSpeedupEstimator`` to the v3 ``SpeedupEstimator``
protocol, owning an asyncio event loop internally.
"""

from arid_badger.max_reward_puct.v3.scoring_providers.coroutine_adapter import (
    CoroutineSpeedupEstimator,
)

__all__ = ["CoroutineSpeedupEstimator"]
