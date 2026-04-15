"""Parity test: vendored cases must match ttt-discover's task.yml.

The yml path is repo-relative — skip when ttt-discover is not checked out
at the expected location (e.g. in CI or on a fresh laptop).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from arid_badger.trimul.cases import BENCHMARK_CASES, CORRECTNESS_CASES


_TTT_DISCOVER_TRIMUL = Path(
    "/nas-ssd2/zaidkhan/ttt-discover/examples/gpu_mode/lib/bioml/trimul/task.yml"
)


def _parse_yaml_case_lists(
    yml_text: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Extract the ``tests:`` and ``benchmarks:`` list-of-dicts from the yml.

    The task.yml uses inline-JSON-style dicts on each ``-`` list entry,
    e.g. ``- {"seqlen": 32, ...}``. We parse them with Python literal_eval
    after normalising ``True``/``False`` (which are already Python-capitalised
    in the source — cf. task.yml lines 46-72).
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
    if not _TTT_DISCOVER_TRIMUL.exists():
        pytest.skip(f"ttt-discover not available at {_TTT_DISCOVER_TRIMUL}")
    yml_text = _TTT_DISCOVER_TRIMUL.read_text()
    yml_tests, yml_benchmarks = _parse_yaml_case_lists(yml_text)

    assert yml_tests == [dict(c) for c in CORRECTNESS_CASES]
    assert yml_benchmarks == [dict(c) for c in BENCHMARK_CASES]


def test_case_shapes() -> None:
    assert len(CORRECTNESS_CASES) == 18
    assert len(BENCHMARK_CASES) == 7
    for case in CORRECTNESS_CASES + BENCHMARK_CASES:
        assert set(case.keys()) == {
            "seqlen",
            "bs",
            "dim",
            "hiddendim",
            "seed",
            "nomask",
            "distribution",
        }
        assert case["distribution"] in ("normal", "cauchy")
