"""Causal Conv1d test/benchmark cases.

Vendored from
``gpu-mode/reference-kernels/problems/helion/causal_conv1d_py/task.yml``
(``tests:`` and ``benchmarks:`` blocks). A parity test in
``tests/causal_conv1d/test_cases.py`` loads the yml at test time and
asserts these literals match the source.

Mirrors ``gpu_forecasters.trimul.cases`` — same shape, different fields.
"""

from __future__ import annotations

from typing import TypedDict


class CausalConv1dTestArgs(TypedDict):
    B: int  # batch size
    D: int  # number of channels (depthwise groups)
    S: int  # sequence length
    W: int  # filter width
    seed: int


CORRECTNESS_CASES: list[CausalConv1dTestArgs] = [
    {"B": 1, "D": 64, "S": 64, "W": 4, "seed": 4242},
    {"B": 2, "D": 128, "S": 128, "W": 4, "seed": 5236},
    {"B": 1, "D": 256, "S": 256, "W": 3, "seed": 1001},
    {"B": 1, "D": 128, "S": 64, "W": 8, "seed": 5531},
    {"B": 4, "D": 64, "S": 128, "W": 4, "seed": 9173},
]


BENCHMARK_CASES: list[CausalConv1dTestArgs] = [
    {"B": 1, "D": 1536, "S": 2048, "W": 4, "seed": 2146},
    {"B": 1, "D": 2560, "S": 2048, "W": 4, "seed": 3129},
    {"B": 1, "D": 2560, "S": 4096, "W": 4, "seed": 54352},
]
