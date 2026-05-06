"""Toy providers + stub surrogate used by v3 tests.

Same binary-string toy as the v1/v2 search tests. Mutations flip a
single random bit; evaluations interpret the binary string as an
integer reward; the stub surrogate emits a uniform (or fixed-bin)
forecast and is the test's primary lever for steering selection
behavior under v3.
"""

from __future__ import annotations

import random
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Self

from arid_badger.hill_climbing.domain import Evaluation, NoFeedback
from arid_badger.landscape_map.v2 import (
    SUCCESS_BINS,
    HardwareContext,
    KernelRuntimeEstimate,
    KernelRuntimeQuery,
    LlmCallUsage,
    SpeedupBin,
)


# Fake hardware context for forecast queries — the binary-string toy
# has no real hardware story, but ``KernelRuntimeQuery`` requires one.
TEST_HARDWARE = HardwareContext(
    device_name="test-cpu",
    compute_capability=(0, 0),
    total_global_memory_gb=0.0,
    multiprocessor_count=0,
    max_threads_per_multiprocessor=0,
    clock_rate_ghz=0.0,
    memory_clock_rate_ghz=0.0,
    memory_bus_width_bits=0,
)


def _eval(reward: float | None) -> Evaluation[NoFeedback]:
    return Evaluation(observation=NoFeedback(), reward=reward)


class BinaryStringMutationProvider:
    """One submit → one candidate. Picks a random bit, flips it."""

    def __init__(self, seed: int | None = None, max_workers: int = 8) -> None:
        self._rng = random.Random(seed)
        self._rng_lock = threading.Lock()
        self._max_workers = max_workers
        self._executor: ThreadPoolExecutor | None = None
        self.submit_count = 0
        self._submit_lock = threading.Lock()

    def submit(
        self,
        parent_code: str,
        evaluation: Evaluation[NoFeedback],
    ) -> Future[str]:
        del evaluation
        assert self._executor is not None, (
            "BinaryStringMutationProvider must be entered before submit"
        )
        with self._submit_lock:
            self.submit_count += 1
        return self._executor.submit(self._mutate, parent_code)

    def _mutate(self, parent_code: str) -> str:
        with self._rng_lock:
            pos = self._rng.randrange(len(parent_code))
        bits = list(parent_code)
        bits[pos] = "1" if bits[pos] == "0" else "0"
        return "".join(bits)

    def __enter__(self) -> Self:
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None


class BinaryStringEvaluationProvider:
    """One submit → one evaluation. Reward = int(code, 2)."""

    def __init__(self, max_workers: int = 8, sleep_s: float = 0.0) -> None:
        self._max_workers = max_workers
        self._sleep_s = sleep_s
        self._executor: ThreadPoolExecutor | None = None
        self.submit_count = 0
        self._lock = threading.Lock()

    def submit(self, program_code: str) -> Future[Evaluation[NoFeedback]]:
        assert self._executor is not None, (
            "BinaryStringEvaluationProvider must be entered before submit"
        )
        with self._lock:
            self.submit_count += 1
        return self._executor.submit(self._evaluate, program_code)

    def _evaluate(self, program_code: str) -> Evaluation[NoFeedback]:
        if self._sleep_s > 0:
            time.sleep(self._sleep_s)
        try:
            reward: float | None = float(int(program_code, 2))
        except ValueError:
            reward = None
        return _eval(reward)

    def __enter__(self) -> Self:
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None


class UniformAsyncEstimator:
    """Always returns a uniform forecast over the eight success bins.

    With a uniform distribution every candidate gets the same score
    from any monotonic ranking rule, so the only effect is to keep
    or filter candidates by ``k_per_parent`` deterministically (the
    selection breaks ties by ``request_id``). Useful for testing the
    phased-flow plumbing without conflating it with a smart
    surrogate.
    """

    def __init__(self) -> None:
        self.call_count = 0
        self._lock = threading.Lock()

    async def aestimate(
        self, query: KernelRuntimeQuery
    ) -> tuple[KernelRuntimeEstimate, LlmCallUsage | None]:
        del query
        with self._lock:
            self.call_count += 1
        spread = 1.0 / len(SUCCESS_BINS)
        return (
            KernelRuntimeEstimate(
                predicted_bin=SpeedupBin.MINOR_SLOWDOWN,
                bin_probabilities={b: spread for b in SUCCESS_BINS},
                reasoning="uniform stub",
                raw_probability_sum=1.0,
            ),
            None,
        )


class CodeLengthAsyncEstimator:
    """Predicts higher speedup for codes with more ``1`` bits.

    Lets selection prefer high-reward candidates even though the
    surrogate doesn't see the actual reward — the heuristic happens
    to align with the reward function on the binary-string toy. This
    is what lets a ``k_per_parent < samples_per_parent`` run still
    converge on the toy: the surrogate filters in the right
    direction.
    """

    async def aestimate(
        self, query: KernelRuntimeQuery
    ) -> tuple[KernelRuntimeEstimate, LlmCallUsage | None]:
        ones = query.candidate.code.count("1")
        total = max(1, len(query.candidate.code))
        # Map ones-fraction in [0, 1] to a bin index in [1, 8].
        bin_index = max(1, min(8, 1 + int(7 * ones / total)))
        predicted = SpeedupBin(bin_index)
        # Concentrate 0.7 on the predicted bin, spread 0.3 elsewhere.
        spread = 0.3 / (len(SUCCESS_BINS) - 1)
        bin_probabilities = {
            b: (0.7 if b == predicted else spread) for b in SUCCESS_BINS
        }
        return (
            KernelRuntimeEstimate(
                predicted_bin=predicted,
                bin_probabilities=bin_probabilities,
                reasoning="ones-count heuristic",
                raw_probability_sum=1.0,
            ),
            None,
        )
