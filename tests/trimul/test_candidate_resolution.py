"""Unit tests for the synthetic-module-shim candidate resolver.

No GPU needed: just tests that source strings get turned into callable
``custom_kernel`` objects (or a clean CandidateResolutionError). The
resolver is the load-bearing glue between free-form LLM-generated source
and the scoring pipeline, so bad-source handling is real domain logic.
"""

from __future__ import annotations

import pytest

from gpu_forecasters.trimul.scoring import CandidateResolutionError, _loaded_candidate


def test_resolves_minimal_candidate() -> None:
    src = "def custom_kernel(data):\n    return data[0]\n"
    with _loaded_candidate(src) as fn:
        assert callable(fn)
        assert fn([42]) == 42


def test_resolves_candidate_importing_task_shim() -> None:
    src = (
        "from task import input_t, output_t\n"
        "def custom_kernel(data):\n"
        "    return data[0]\n"
    )
    with _loaded_candidate(src) as fn:
        assert fn([7]) == 7


def test_resolves_candidate_importing_utils_shim() -> None:
    src = (
        "from utils import verbose_allclose\n"
        "def custom_kernel(data):\n"
        "    return data[0]\n"
    )
    with _loaded_candidate(src) as fn:
        assert fn(["ok"]) == "ok"


def test_syntax_error_raises_clean_error() -> None:
    src = "def custom_kernel(data:\n    return data\n"
    with pytest.raises(CandidateResolutionError, match="syntax error"):
        with _loaded_candidate(src):
            pass


def test_missing_custom_kernel_raises_clean_error() -> None:
    src = "def other_function(data):\n    return data\n"
    with pytest.raises(CandidateResolutionError, match="does not define"):
        with _loaded_candidate(src):
            pass


def test_import_error_raises_clean_error() -> None:
    src = "from nonexistent_module import something\n"
    with pytest.raises(CandidateResolutionError, match="import failed"):
        with _loaded_candidate(src):
            pass


def test_subsequent_calls_get_fresh_module() -> None:
    src1 = "MARKER = 1\ndef custom_kernel(data):\n    return MARKER\n"
    src2 = "MARKER = 2\ndef custom_kernel(data):\n    return MARKER\n"
    with _loaded_candidate(src1) as fn1:
        assert fn1(None) == 1
    with _loaded_candidate(src2) as fn2:
        assert fn2(None) == 2
