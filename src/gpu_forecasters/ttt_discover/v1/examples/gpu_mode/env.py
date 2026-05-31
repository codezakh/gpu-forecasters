"""TriMul env for TTT-Discover.

Adapted from ttt-discover's ``examples/gpu_mode/env.py``. The original ran
submissions against the GPU-mode Discord leaderboard via ``libkernelbot``
on Modal. That leaderboard is closed and its launcher drags in a large
dependency tree (DB, profanity filter, Discord reporter). This version
instead scores kernels with our existing ``gpu_forecasters.trimul`` Modal
pipeline, which uses the same adaptive cuda.Event timing loop and a
curated set of benchmark cases.

Reward shape matches the original: the dict returned by ``get_reward``
has the keys TTT-Discover's training loop consumes (``reward``,
``correctness``, ``raw_score``, ``msg``, ``result_construction``,
``stdout``). We convert our geomean speedup into a runtime-in-microseconds
``raw_score`` and a ``score_scale / raw_score`` reward to keep the prompt
context (``to_prompt`` shows "runtime (microseconds)") meaningful.
"""

from __future__ import annotations

import atexit
import math
import threading
from typing import Optional

from gpu_forecasters.trimul.cases import BENCHMARK_CASES
from gpu_forecasters.trimul.modal_scoring import TriMulScoringFn, modal_trimul_scoring_session
from gpu_forecasters.typing_utils import is_ok

from gpu_forecasters.ttt_discover.v1 import (
    BaseRewardEvaluator,
    DiscoverConfig,
    Environment,
    State,
    discover,
)
from gpu_forecasters.ttt_discover.v1.examples.gpu_mode.prompt import TRIMUL_PROMPT

# Paper target for A100 TriMul is ~2198us (TTT-Discover) vs 4531us (best human).
# The ttt-discover original used score_scale=1500 on H100 (best human ~1371us).
# For A100 with published best at ~2198us, 2500 puts reward≈1 at paper-SOTA.
TRIMUL_SCORE_SCALE_US = 2500.0


# Module-level Modal scoring session. All rollouts in a training step run
# concurrently via asyncio + thread pool and Modal rejects nested app.run()
# contexts, so we keep exactly one session open for the lifetime of the
# process. The session is lazily created on first use and torn down via
# atexit.
_SESSION_LOCK = threading.Lock()
_SESSION_CM: Optional[object] = None
_SCORE_FN: Optional[TriMulScoringFn] = None


def _get_score_fn(gpu: str) -> TriMulScoringFn:
    global _SESSION_CM, _SCORE_FN
    if _SCORE_FN is not None:
        return _SCORE_FN
    with _SESSION_LOCK:
        if _SCORE_FN is not None:
            return _SCORE_FN
        cm = modal_trimul_scoring_session(gpu=gpu)
        score_fn = cm.__enter__()
        _SESSION_CM = cm
        _SCORE_FN = score_fn

        def _close() -> None:
            try:
                cm.__exit__(None, None, None)
            except Exception:
                pass

        atexit.register(_close)
        return score_fn


def _gpu_mode_error(msg: str) -> dict:
    return {
        "reward": 0.0,
        "msg": msg,
        "correctness": 0.0,
        "raw_score": -1_000_000,
        "result_construction": [],
        "stdout": "",
    }


class GpuModeRewardEvaluator(BaseRewardEvaluator):
    """Score one TriMul candidate on Modal, return a TTT-Discover reward dict.

    Opens a fresh Modal scoring session per ``get_reward`` call. That's one
    ``app.run()`` per rollout (Modal reuses containers across calls in the
    same Python process, so warm starts are cheap); if end-to-end throughput
    becomes the bottleneck this is the place to introduce a cached session.
    """

    def __init__(self, *args, **kwargs):
        self.problem_type = kwargs.get("problem_type")
        self.log_dir = kwargs.get("log_dir")
        if self.problem_type != "trimul":
            raise ValueError(
                f"GpuModeRewardEvaluator only supports problem_type='trimul', "
                f"got {self.problem_type!r}. (MLA decode path was in the original "
                f"ttt-discover example but is not ported here.)"
            )
        self.score_scale_us = TRIMUL_SCORE_SCALE_US
        self.gpu_type = "A100-80GB"

    def get_reward(self, code: str, state: State) -> dict:
        if "@triton.jit" not in code:
            return _gpu_mode_error("Code must contain @triton.jit.")
        if "identity" in code:
            return _gpu_mode_error("Identity kernel is not allowed.")

        try:
            score_fn = _get_score_fn(self.gpu_type)
            outcomes = score_fn(code, list(BENCHMARK_CASES))
        except Exception as exc:
            return _gpu_mode_error(f"Modal session failed: {type(exc).__name__}: {exc}")

        speedups: list[float] = []
        runtimes_ns: list[float] = []
        for i, outcome in enumerate(outcomes):
            if not is_ok(outcome):
                err = outcome.unwrap_err()
                return _gpu_mode_error(f"Case {i} infra failure: {err.reason}")
            er = outcome.unwrap()
            if not er.correct:
                return _gpu_mode_error(
                    f"Case {i} failed ({er.failure_kind}): "
                    f"{er.error_message or er.runtime_error or er.compilation_error}"
                )
            if er.runtime_ns <= 0:
                return _gpu_mode_error(f"Case {i} runtime_ns <= 0")
            speedups.append(er.ref_runtime_ns / er.runtime_ns)
            runtimes_ns.append(er.runtime_ns)

        # Geomean aggregation matches our TriMulModalProvider default and
        # the GPU-mode leaderboard's ranking_by.geomean.
        geomean_runtime_ns = math.exp(
            sum(math.log(r) for r in runtimes_ns) / len(runtimes_ns)
        )
        raw_score_us = geomean_runtime_ns / 1_000.0
        reward = self.score_scale_us / raw_score_us
        return {
            "reward": float(reward),
            "msg": (
                f"\nOverall geomean runtime: {raw_score_us:.2f} us "
                f"(reward = {self.score_scale_us}/raw = {reward:.4f})"
            ),
            "correctness": 1.0,
            "raw_score": float(raw_score_us),
            "result_construction": [],
            "stdout": "",
        }


class GpuModeEnv(Environment):
    reward_function = GpuModeRewardEvaluator
    state_type = State

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        if problem_type != "trimul":
            raise ValueError(f"Unsupported problem_type: {problem_type}")
        return State(timestep=-1, code="", value=-1_000_000, construction=None)

    def _should_keep_code_separators(self) -> bool:
        return False

    def is_maximize(self) -> bool:
        return False

    def get_question(self) -> str:
        state = self.initial_state
        target = 1000  # TriMul-A100 target (microseconds, lower is better).

        state_ctx = state.to_prompt(
            target, metric_name="runtime (microseconds)", maximize=False, language="python"
        )

        # Rules block from the original example; kept verbatim except the
        # H100→A100 hardware reference to match our Modal GPU.
        return f"""{TRIMUL_PROMPT}

{state_ctx}

Rules:
- The tensors arguments passed in will be already on your cuda device.
- Define all of your code in one final ```python ``` block.
- We will test the correctness of your kernel on multiple input shapes, make sure to support different potential test cases.
- You are allowed to use mixed precision computations, but make sure your final output is in float32.
- You must use trition 3.3.1 and these kernels will be run on an A100.
- You do not have to implement everything in triton, you may choose to have some of the operations done in pytorch. However, you must implement at least part of the operations in a kernel.
- Include a short docstring at the top summarizing your algorithm.
"""


def discover_gpu_mode_trimul():
    """Entry point used by experiment scripts."""
    config = DiscoverConfig(
        env_type=GpuModeEnv,
        problem_type="trimul",
        eval_timeout=530,
        experiment_name="ttt-discover-trimul-debug",
        wandb_project="gpu-mode",
    )
    discover(config)


if __name__ == "__main__":
    discover_gpu_mode_trimul()
