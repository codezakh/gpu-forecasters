"""Parity test: vendored cases must match upstream task.yml.

The yml lives in the gpu-mode/reference-kernels checkout under
``/tmp/reference-kernels`` (re-cloned per session per gh070's spec).
If absent, skip — the import-side ``CORRECTNESS_CASES``/
``BENCHMARK_CASES`` are still validated structurally by
``test_case_shapes`` below.

Mirrors ``tests/trimul/test_cases.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from arid_badger.causal_conv1d.cases import BENCHMARK_CASES, CORRECTNESS_CASES


_UPSTREAM_TASK_YML = Path(
    "/tmp/reference-kernels/problems/helion/causal_conv1d_py/task.yml"
)


def _parse_yaml_case_lists(
    yml_text: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Extract ``tests:`` and ``benchmarks:`` list-of-dicts from the yml.

    Each ``-`` entry is inline-JSON-style (e.g.
    ``- {"B": 1, "D": 64, ...}``). ``ast.literal_eval`` parses each row.
    """
    import ast

    sections: dict[str, list[dict[str, object]]] = {"tests": [], "benchmarks": []}
    current: str | None = None
    for line in yml_text.splitlines():
        stripped = line.strip()
        if stripped in ("tests:", "benchmarks:"):
            current = stripped[:-1]
            continue
        if current is None:
            continue
        if not stripped.startswith("- "):
            if stripped and not stripped.startswith("#"):
                current = None
            continue
        body = stripped[2:].strip()
        body = re.sub(r"\s+$", "", body)
        parsed = ast.literal_eval(body)
        assert isinstance(parsed, dict)
        sections[current].append(parsed)
    return sections["tests"], sections["benchmarks"]


def test_cases_match_task_yml() -> None:
    if not _UPSTREAM_TASK_YML.exists():
        pytest.skip(
            f"upstream reference-kernels not available at {_UPSTREAM_TASK_YML}; "
            "re-clone gpu-mode/reference-kernels into /tmp to enable this parity check"
        )
    yml_text = _UPSTREAM_TASK_YML.read_text()
    yml_tests, yml_benchmarks = _parse_yaml_case_lists(yml_text)

    assert yml_tests == [dict(c) for c in CORRECTNESS_CASES]
    assert yml_benchmarks == [dict(c) for c in BENCHMARK_CASES]


def test_case_shapes() -> None:
    assert len(CORRECTNESS_CASES) == 5
    assert len(BENCHMARK_CASES) == 3
    expected_keys = {"B", "D", "S", "W", "seed"}
    for case in CORRECTNESS_CASES + BENCHMARK_CASES:
        assert set(case.keys()) == expected_keys
        assert case["W"] >= 1
        assert case["B"] >= 1
        assert case["D"] >= 1
        assert case["S"] >= case["W"], "sequence length must accommodate filter width"
