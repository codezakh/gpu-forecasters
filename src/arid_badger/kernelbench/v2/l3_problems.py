"""Typed loader for KernelBench L3 problems.

Wraps ``kernelbench.dataset.construct_kernelbench_dataset(level=3,
source="local", problem_ids=[id])`` and returns an ``L3ProblemReference``
with the two fields the v2 KernelBench providers actually consume: a
stable ``problem_name`` (used for log labels and per-problem output
directories) and ``reference_kernel_code`` (the seed program for the
search and the prompt's reference module).

``TIER_B_PROBLEMS`` pins the Tier-B subset selected for the gh070 paper
testbed: five compositional single-block L3 problems spanning attention,
SSM, and conv families. Picking by canonical name (``ResNetBasicBlock``)
instead of integer id catches typos at type-check time when used as
``L3_RESNET_BASIC_BLOCK = TIER_B_PROBLEMS[0]``-style constants.

The optional gh070 problems (``L3_25 ShuffleNetUnit``,
``L3_6 GoogleNetInceptionModule``) are not in ``TIER_B_PROBLEMS`` but
remain reachable via ``load_l3_problem(problem_id=...)`` for prompt
spot-checks.
"""

from __future__ import annotations

from functools import lru_cache

from kernelbench.dataset import construct_kernelbench_dataset
from pydantic import BaseModel, ConfigDict


class L3ProblemReference(BaseModel):
    """A KernelBench L3 problem resolved to the fields the v2 providers
    consume. ``name`` matches the canonical KernelBench filename stem
    (e.g. ``8_ResNetBasicBlock``)."""

    model_config = ConfigDict(frozen=True)

    problem_id: int
    name: str
    reference_kernel_code: str


@lru_cache(maxsize=None)
def _l3_dataset():
    """Cache the full L3 dataset on first access. Reading 50 reference
    files from disk is cheap but ``construct_kernelbench_dataset``
    re-walks the filesystem on every call, which is wasteful when a
    multi-cell experiment loop calls ``load_l3_problem`` repeatedly."""
    return construct_kernelbench_dataset(level=3, source="local")


def _canonical_name(raw_name: str) -> str:
    """KernelBench's ``Problem.name`` is the source filename including
    ``.py``. The reference name used for log labels, output paths, and
    the registry below strips that extension so callers don't end up
    with ``8_ResNetBasicBlock.py`` in directory names."""
    return raw_name[:-3] if raw_name.endswith(".py") else raw_name


def load_l3_problem(*, problem_id: int) -> L3ProblemReference:
    """Resolve an L3 problem by its 1-indexed KernelBench id.

    Raises ``KeyError`` (via the underlying dataset) if ``problem_id`` is
    not present in level 3.
    """
    problem = _l3_dataset().get_problem_by_id(problem_id)
    return L3ProblemReference(
        problem_id=problem.problem_id,
        name=_canonical_name(problem.name),
        reference_kernel_code=problem.code,
    )


def load_l3_problem_by_name(name: str) -> L3ProblemReference:
    """Resolve an L3 problem by its canonical filename stem (e.g.
    ``8_ResNetBasicBlock``).

    Linear scan over the 50-problem L3 dataset; cells call this at most
    once per run so the cost is irrelevant.
    """
    for pid in _l3_dataset().get_problem_ids():
        problem = _l3_dataset().get_problem_by_id(pid)
        if _canonical_name(problem.name) == name:
            return L3ProblemReference(
                problem_id=problem.problem_id,
                name=_canonical_name(problem.name),
                reference_kernel_code=problem.code,
            )
    raise KeyError(
        f"No L3 problem with canonical name {name!r}. Names follow the "
        f"pattern '<id>_<PascalCaseModule>' (e.g. '8_ResNetBasicBlock')."
    )


# ---------------------------------------------------------------------------
# Tier-B registry — pinned by docs/specs/gh070-problem-subset-selection.md.
# ---------------------------------------------------------------------------

# Tier-B identifiers, kept as a (problem_id, expected_name) pair so the
# tuple's element order is stable regardless of how the dataset
# enumerates problems internally.
_TIER_B_IDS_AND_NAMES: tuple[tuple[int, str], ...] = (
    (8, "8_ResNetBasicBlock"),
    (21, "21_EfficientNetMBConv"),
    (43, "43_MinGPTCausalAttention"),
    (44, "44_MiniGPTBlock"),
    (48, "48_Mamba2ReturnY"),
)


def _build_tier_b() -> tuple[L3ProblemReference, ...]:
    out: list[L3ProblemReference] = []
    for pid, expected_name in _TIER_B_IDS_AND_NAMES:
        ref = load_l3_problem(problem_id=pid)
        if ref.name != expected_name:
            raise RuntimeError(
                f"Tier-B registry drift: L3_{pid} resolved to name "
                f"{ref.name!r}, expected {expected_name!r}. The KernelBench "
                f"L3 source files may have been renamed; update "
                f"_TIER_B_IDS_AND_NAMES in this module."
            )
        out.append(ref)
    return tuple(out)


TIER_B_PROBLEMS: tuple[L3ProblemReference, ...] = _build_tier_b()


__all__ = [
    "L3ProblemReference",
    "TIER_B_PROBLEMS",
    "load_l3_problem",
    "load_l3_problem_by_name",
]
