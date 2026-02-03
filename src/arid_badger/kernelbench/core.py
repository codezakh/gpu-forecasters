from kernelbench.eval import KernelExecResult


from dataclasses import dataclass


@dataclass
class KernelScoringResult:
    """Result of scoring a kernel against a reference."""

    exec_result: KernelExecResult
    speedup: float
    is_valid: bool
