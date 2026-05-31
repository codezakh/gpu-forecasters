"""TriMul scoring port from ttt-discover.

A sibling of ``arid_badger.kernelbench`` that vendors just the scoring
pipeline for the CUDA Mode TriMul task: reference implementation, input
generation, correctness oracle, and the adaptive timing loop. The task
uses a ``custom_kernel(data)`` entry point rather than KernelBench's
``ModelNew(Model)``, and times with an adaptive ``cuda.Event`` loop.
"""

from arid_badger.trimul.cases import (
    BENCHMARK_CASES,
    CORRECTNESS_CASES,
    TriMulTestArgs,
)
from arid_badger.trimul.core import (
    CaseSpeedup,
    CompileFailedFeedback,
    IncorrectFeedback,
    InfrastructureFailureFeedback,
    RuntimeErrorFeedback,
    Stats,
    SuccessFeedback,
    TriMulExecResult,
    TriMulFailureFeedback,
    TriMulKernelExecutionFeedback,
    failure_feedback_from_exec_result,
)

__all__ = [
    "BENCHMARK_CASES",
    "CORRECTNESS_CASES",
    "TriMulTestArgs",
    "TriMulExecResult",
    "TriMulKernelExecutionFeedback",
    "TriMulFailureFeedback",
    "CaseSpeedup",
    "SuccessFeedback",
    "IncorrectFeedback",
    "RuntimeErrorFeedback",
    "CompileFailedFeedback",
    "InfrastructureFailureFeedback",
    "Stats",
    "failure_feedback_from_exec_result",
]
