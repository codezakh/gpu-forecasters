"""The static definition of what a rollout is optimising.

Carries the base prompt text, the test cases the evaluator will run
against, the GPU / Triton version (injected into the rules block), and
the target runtime used by ``ScaleByTargetUs`` to normalise ``speedup``
into a reward.
"""

from __future__ import annotations

from dataclasses import dataclass

from arid_badger.trimul.cases import TriMulTestArgs


@dataclass(frozen=True)
class TriMulProblem:
    base_prompt_text: str
    test_cases: tuple[TriMulTestArgs, ...]
    gpu_name: str
    triton_version: str
    target_runtime_us: float
