"""TTT-Discover v2 — rich-observation RL for TriMul.

v1 dropped execution feedback at the reward-dict / State boundary, so the
policy only ever saw a scalar. v2 threads typed feedback (compile errors,
runtime tracebacks, per-case speedups) straight through to the next
rollout's prompt and writes a durable per-rollout event log as the
primary artifact.

See ``docs/theory/th010-ttt-discover-observation-drop.md`` for the
diagnosis, and the package subdirectories for the implementation:
``domain/`` for value types, ``interfaces/`` for the swap seams,
``archive/`` / ``evaluator/`` / ``renderers/`` / ``scalarizers/`` /
``extractors/`` / ``sinks/`` for concrete components, and ``env.py`` /
``rl_integration.py`` / ``discovery.py`` for the wiring into v1's
Tinker training loop.
"""

from gpu_forecasters.ttt_discover.v2.discovery import DiscoverConfig, discover

__all__ = ["DiscoverConfig", "discover"]
