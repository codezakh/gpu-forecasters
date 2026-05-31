"""TriMul test/benchmark cases.

Vendored verbatim from ``ttt-discover/examples/gpu_mode/lib/bioml/trimul/task.yml``
(lines 45-72 of that file). A parity test in ``tests/trimul/test_cases.py``
loads the yml at test time and asserts these literals match the source.
"""

from __future__ import annotations

from typing import TypedDict


class TriMulTestArgs(TypedDict):
    seqlen: int
    bs: int
    dim: int
    hiddendim: int
    seed: int
    nomask: bool
    distribution: str


CORRECTNESS_CASES: list[TriMulTestArgs] = [
    {"seqlen": 32, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 9371, "nomask": True, "distribution": "normal"},
    {"seqlen": 32, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 1092, "nomask": False, "distribution": "normal"},
    {"seqlen": 64, "bs": 2, "dim": 256, "hiddendim": 128, "seed": 2291, "nomask": True, "distribution": "normal"},
    {"seqlen": 64, "bs": 2, "dim": 256, "hiddendim": 128, "seed": 210284, "nomask": False, "distribution": "normal"},
    {"seqlen": 128, "bs": 1, "dim": 768, "hiddendim": 128, "seed": 81934, "nomask": True, "distribution": "normal"},
    {"seqlen": 256, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 1932, "nomask": True, "distribution": "normal"},
    {"seqlen": 256, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 10432, "nomask": False, "distribution": "normal"},
    {"seqlen": 768, "bs": 2, "dim": 128, "hiddendim": 128, "seed": 731, "nomask": True, "distribution": "normal"},
    {"seqlen": 1024, "bs": 1, "dim": 384, "hiddendim": 128, "seed": 53121, "nomask": False, "distribution": "normal"},
    {"seqlen": 1024, "bs": 1, "dim": 768, "hiddendim": 128, "seed": 31, "nomask": True, "distribution": "normal"},
    {"seqlen": 1024, "bs": 1, "dim": 768, "hiddendim": 128, "seed": 4921, "nomask": False, "distribution": "normal"},
    {"seqlen": 32, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 937321, "nomask": True, "distribution": "cauchy"},
    {"seqlen": 64, "bs": 2, "dim": 256, "hiddendim": 128, "seed": 2291, "nomask": True, "distribution": "cauchy"},
    {"seqlen": 128, "bs": 1, "dim": 768, "hiddendim": 128, "seed": 8134, "nomask": True, "distribution": "cauchy"},
    {"seqlen": 256, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 932, "nomask": True, "distribution": "cauchy"},
    {"seqlen": 768, "bs": 2, "dim": 128, "hiddendim": 128, "seed": 31, "nomask": True, "distribution": "cauchy"},
    {"seqlen": 1024, "bs": 1, "dim": 384, "hiddendim": 128, "seed": 5321, "nomask": False, "distribution": "cauchy"},
    {"seqlen": 1024, "bs": 1, "dim": 768, "hiddendim": 128, "seed": 491, "nomask": False, "distribution": "cauchy"},
]


BENCHMARK_CASES: list[TriMulTestArgs] = [
    {"seqlen": 256, "bs": 2, "dim": 128, "hiddendim": 128, "seed": 9371, "nomask": True, "distribution": "normal"},
    {"seqlen": 768, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 381, "nomask": True, "distribution": "cauchy"},
    {"seqlen": 256, "bs": 2, "dim": 384, "hiddendim": 128, "seed": 2301, "nomask": False, "distribution": "normal"},
    {"seqlen": 512, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 12819, "nomask": True, "distribution": "normal"},
    {"seqlen": 1024, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 381, "nomask": True, "distribution": "cauchy"},
    {"seqlen": 768, "bs": 1, "dim": 384, "hiddendim": 128, "seed": 481, "nomask": False, "distribution": "normal"},
    {"seqlen": 1024, "bs": 1, "dim": 384, "hiddendim": 128, "seed": 23291, "nomask": True, "distribution": "normal"},
]
