"""v3 scoring providers.

The v3 ``AsyncEvaluationProvider`` protocol (see ``v3/providers.py``)
is intentionally identical to v2's: ``submit(program_code) ->
Future[Evaluation[ObservationT]]`` plus ``__enter__`` / ``__exit__``.
Existing v2 concretes (e.g. ``arid_badger.max_reward_puct.v2.
scoring_providers.trimul_modal``) satisfy the v3 protocol unchanged
and can be passed directly to ``v3.SearchDriver``. No adapter layer
is required.

This directory exists per the spec's acceptance criterion; concrete
v3-specific scoring providers would live here.
"""
