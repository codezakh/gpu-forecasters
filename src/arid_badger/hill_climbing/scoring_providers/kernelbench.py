import logging
from pathlib import Path
from typing import Annotated, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from arid_badger.kernelbench.core import (
    InfrastructureFailureFeedback,
    KernelExecutionFeedback,
    execution_feedback_from_exec_result,
)
from arid_badger.kernelbench.isolated_scoring import run_scoring_in_subprocess
from arid_badger.kernelbench.scoring import check_kernel_exec_result_valid
from arid_badger.typing_utils import is_ok
from ..domain import Evaluation

logger = logging.getLogger(__name__)


KernelBenchFeedback = Annotated[
    Union[KernelExecutionFeedback, InfrastructureFailureFeedback],
    Field(discriminator="kind"),
]


class KernelBenchObservation(BaseModel):
    """Observation from a KernelBench scoring attempt.

    Wraps a feedback discriminated union that captures what happened
    during compilation and execution, without leaking KernelBench types.

    This is a single-field wrapper because ``ObservationT`` must be a
    concrete ``BaseModel`` subclass (not a bare ``Union``), and having a
    named type keeps the door open for additional observation fields
    without changing the generic signature of ``Evaluation``.
    """

    model_config = ConfigDict(frozen=True)

    feedback: KernelBenchFeedback


class Provider:
    def __init__(
        self,
        reference_kernel_code: str,
        backend: str = "cuda",
        precision: str = "fp32",
        num_correct_trials: int = 5,
        num_perf_trials: int = 100,
        build_dir: Optional[Path] = None,
    ):
        self.reference_kernel_code = reference_kernel_code
        self.backend = backend
        self.precision = precision
        self.num_correct_trials = num_correct_trials
        self.num_perf_trials = num_perf_trials
        self.build_dir = build_dir

    def __enter__(self) -> "Provider":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        pass

    def evaluate(self, program_code: str) -> Evaluation[KernelBenchObservation]:
        outcome = run_scoring_in_subprocess(
            mutated_kernel_code=program_code,
            reference_kernel_code=self.reference_kernel_code,
            backend=self.backend,
            precision=self.precision,
            num_correct_trials=self.num_correct_trials,
            num_perf_trials=self.num_perf_trials,
            build_dir=self.build_dir,
        )

        if not is_ok(outcome):
            scoring_error = outcome.unwrap_err()
            logger.warning("Kernel scoring failed: %s", scoring_error.reason)
            observation = KernelBenchObservation(
                feedback=InfrastructureFailureFeedback(reason=scoring_error.reason),
            )
            return Evaluation[KernelBenchObservation](observation=observation, reward=None)

        exec_result = outcome.unwrap()
        is_valid = check_kernel_exec_result_valid(exec_result)
        speedup = (
            exec_result.ref_runtime / exec_result.runtime
            if is_valid
            else 0.0
        )

        feedback = execution_feedback_from_exec_result(
            exec_result=exec_result,
            speedup=speedup,
            is_valid=is_valid,
        )
        observation = KernelBenchObservation(feedback=feedback)

        return Evaluation[KernelBenchObservation](
            observation=observation,
            reward=speedup if is_valid else None,
        )
