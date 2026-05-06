"""v3 mutation providers.

The v3 ``MutationProvider`` protocol (see ``v3/providers.py``) is
intentionally identical to v2's: ``submit(parent_code, evaluation) ->
Future[str]`` plus ``__enter__`` / ``__exit__``. This means existing v2
concretes (e.g. ``arid_badger.max_reward_puct.v2.mutation_providers.
trimul_feedback_mutation``) satisfy the v3 protocol unchanged and can
be passed directly to ``v3.SearchDriver``. No adapter layer is
required — wrapping v2 concretes in shim classes would just duplicate
their interface.

This directory exists per the spec's acceptance criterion; concrete
v3-specific mutation providers (e.g. ones that need access to v3-only
event-log state, if any are ever added) would live here.
"""
