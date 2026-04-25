"""Theory-builder v1 — training-free natural-language world model.

The package wires up an outer loop on top of an inner search:

* ``Hypothesis`` → ``ExperimentResult`` → ``Explanation`` → ``WorldModel``
  domain types live in ``domain.py``.
* The loop is event-sourced (``events.py`` / ``state.py`` / ``event_log.py``)
  in the same shape as ``arid_badger.max_reward_puct.v2``.
* The builder (``builder.py``) is an LLM that proposes hypotheses and
  writes explanations + world-model diffs back. Diffs are ``SEARCH``/
  ``REPLACE`` blocks applied by ``diff.py``.
* The worker (``worker.py``) is an adapter over ``max_reward_puct.v2``
  that runs a small inner search seeded by a hypothesis.
* The driver (``driver.py``) is the outer loop.
"""
