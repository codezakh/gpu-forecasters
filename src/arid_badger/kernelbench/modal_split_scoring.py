"""Split CPU-compile / GPU-benchmark Modal kernel scoring.

Sibling to `modal_scoring.py`. Exposes the same `ScoringFn` contract so
callers can swap `modal_scoring_session` for `modal_split_scoring_session`
with no other changes.

Pipeline per evaluation:

1. A CPU-only Modal container cross-compiles the kernel for the target
   compute capability via `nvcc` (driven by `TORCH_CUDA_ARCH_LIST` and
   `load_custom_model`'s `build_directory` injection), writing the
   resulting `.so` into a shared `modal.Volume`.
2. A fresh GPU container (see `max_inputs=1` on `ModalGpuBenchmarker`)
   runs `eval_kernel_against_ref` with the same `build_dir`, and
   KernelBench's `load_inline` dlopens the just-committed artifact
   instead of shelling out to nvcc.

The volume is a one-shot CPU→GPU transfer channel per call, not a
cross-call cache: reusing a GPU container across evaluations triggers a
`Volume.reload()` / `dlopen` `ConflictError` once `load_inline` has held
an `.so` open (see e0018, ADR-002). Single-input retirement is what
makes the fresh-mount-per-call story work without any mid-lifetime
`reload()`.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional

import modal

from arid_badger.invocation_sink import code_sha256
from arid_badger.kernelbench.isolated_scoring import ScoringError
from arid_badger.kernelbench.modal_image import GPU_ARCH_MAPPING, image
from arid_badger.kernelbench.modal_scoring import ScoringFn, _wrap_exec_result
from arid_badger.typing_utils import Err, Ok, Option
from kernelbench.eval import KernelExecResult

# ---------------------------------------------------------------------------
# App + shared build cache volume
# ---------------------------------------------------------------------------

app = modal.App("arid-badger-kernel-split")

_CACHE_MOUNT = "/cache"

build_cache = modal.Volume.from_name(
    "arid-badger-kernel-build-cache",
    create_if_missing=True,
)

# ---------------------------------------------------------------------------
# Compute capability table (TORCH_CUDA_ARCH_LIST decimal form).
# Matches e0017 POC `ARCH_BY_GPU`.
# ---------------------------------------------------------------------------

COMPUTE_CAPABILITY_BY_GPU: dict[str, str] = {
    "T4": "7.5",  # Turing
    "A10G": "8.6",  # Ampere (AWS G5 variant of A10)
    "A100": "8.0",  # Ampere
    "L4": "8.9",  # Ada
    "L40S": "8.9",  # Ada
    "H100": "9.0",  # Hopper
    "H200": "9.0",  # Hopper
}

_DEFAULT_GPU = "L4"
_GPU_WAIT_TIMEOUT_S = 30
_GPU_WAIT_INITIAL_DELAY_S = 0.5
_GPU_WAIT_MAX_DELAY_S = 8.0


# ---------------------------------------------------------------------------
# CPU compiler class
# ---------------------------------------------------------------------------


@app.cls(
    image=image,
    volumes={_CACHE_MOUNT: build_cache},
    timeout=600,
    retries=modal.Retries(
        max_retries=3,
        backoff_coefficient=2.0,
        initial_delay=1.0,
    ),
)
class ModalCpuCompiler:
    """Cross-compiles a CUDA kernel for a target sm_XX on a CPU-only container.

    Writes the resulting `.so` into the shared build-cache volume under a
    per-arch, content-hashed subdirectory so that (a) artifacts for different
    compute capabilities never collide and (b) identical sources dedupe.
    """

    @modal.method()
    def compile(
        self,
        mutated_kernel_code: str,
        reference_kernel_code: str,
        cc: str,
    ) -> dict[str, Any]:
        import os

        cache_dir = f"{_CACHE_MOUNT}/sm_{cc.replace('.', '')}/{code_sha256(mutated_kernel_code)[:16]}"
        os.makedirs(cache_dir, exist_ok=True)

        # Must be set before importing torch so torch's cpp_extension machinery
        # doesn't try to probe a non-existent CUDA runtime. `kernelbench.utils.
        # set_gpu_arch` is not usable here because the production path runs it
        # *after* importing torch and after a `torch.cuda.is_available()` wait
        # loop — both of which fail on a CPU-only container.
        os.environ["TORCH_EXTENSIONS_DIR"] = cache_dir
        os.environ["TORCH_CUDA_ARCH_LIST"] = cc

        try:
            from kernelbench.eval import (
                load_custom_model,
                load_original_model_and_inputs,
            )

            # Resolve `Model` into the exec context so `class ModelNew(Model)`
            # in the mutated source can bind its base class.
            context: dict[str, Any] = {}
            load_original_model_and_inputs(reference_kernel_code, context)

            # `load_custom_model` prepends an env-var assignment and execs the
            # source; the exec fires `torch.utils.cpp_extension.load_inline`
            # which runs nvcc and drops a `.so` into `cache_dir`. The ModelNew
            # class it returns is discarded — we only want the compile side
            # effect.
            _ = load_custom_model(
                mutated_kernel_code,
                context,
                build_directory=cache_dir,
            )
        except Exception as exc:
            return {
                "cache_dir": cache_dir,
                "error": f"{type(exc).__name__}: {exc}",
            }

        build_cache.commit()
        return {"cache_dir": cache_dir, "error": None}


# ---------------------------------------------------------------------------
# GPU benchmarker class
# ---------------------------------------------------------------------------


@app.cls(
    image=image,
    gpu=_DEFAULT_GPU,
    volumes={_CACHE_MOUNT: build_cache},
    timeout=120,
    # Retire the container after one input: KernelBench's `load_inline`
    # dlopens a `.so` from the volume, and the resulting open file handle
    # makes `Volume.reload()` fail on subsequent calls (`ConflictError:
    # there are open files preventing the operation`, observed in e0018).
    # A fresh container per call mounts the volume at its latest committed
    # state, so no mid-lifetime reload is needed.
    single_use_containers=True,
    scaledown_window=2,
    retries=modal.Retries(
        max_retries=3,
        backoff_coefficient=2.0,
        initial_delay=1.0,
    ),
)
class ModalGpuBenchmarker:
    """Benchmarks a pre-compiled kernel on a GPU container.

    Expects `ModalCpuCompiler.compile` to have already populated `cache_dir`.
    `eval_kernel_against_ref(build_dir=cache_dir, ...)` forwards to
    `load_custom_model`, which sets `TORCH_EXTENSIONS_DIR` so KernelBench's
    `load_inline` dlopens the cached `.so` rather than invoking nvcc.
    """

    @modal.method()
    def evaluate(
        self,
        mutated_kernel_code: str,
        reference_kernel_code: str,
        cache_dir: str,
        gpu_arch: list[str],
        backend: str = "cuda",
        precision: str = "fp32",
        num_correct_trials: int = 5,
        num_perf_trials: int = 100,
    ) -> KernelExecResult | None:
        # No `build_cache.reload()` here: `max_inputs=1` means this container
        # is brand new, so the volume mount already reflects the latest
        # commit from the CPU compiler. Reloading mid-lifetime would also
        # conflict with any `.so` already dlopen'd by `load_inline`.
        import torch
        from kernelbench.eval import (
            eval_kernel_against_ref,
            get_torch_dtype_from_string,
        )
        from kernelbench.utils import set_gpu_arch

        start = time.time()
        while not torch.cuda.is_available():
            elapsed = time.time() - start
            if elapsed >= _GPU_WAIT_TIMEOUT_S:
                raise RuntimeError(f"GPU not available after {_GPU_WAIT_TIMEOUT_S}s")
            delay = min(
                _GPU_WAIT_INITIAL_DELAY_S * (2 ** int(elapsed / 2)),
                _GPU_WAIT_MAX_DELAY_S,
            )
            time.sleep(delay)

        set_gpu_arch(gpu_arch)

        try:
            result = eval_kernel_against_ref(
                original_model_src=reference_kernel_code,
                custom_model_src=mutated_kernel_code,
                num_correct_trials=num_correct_trials,
                num_perf_trials=num_perf_trials,
                measure_performance=True,
                timing_method="cuda_event",
                verbose=False,
                build_dir=Path(cache_dir),
                device=torch.device("cuda:0"),
                backend=backend,
                precision=get_torch_dtype_from_string(precision),
            )
        except (torch.cuda.CudaError, Exception) as exc:
            exc_type = type(exc).__name__
            if "CudaError" in exc_type or "AcceleratorError" in exc_type:
                modal.experimental.stop_fetching_inputs()
            return KernelExecResult(
                compiled=False,
                metadata={
                    "compilation_error_name": exc_type,
                    "compilation_error": str(exc),
                },
            )

        torch.cuda.empty_cache()
        return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@contextmanager
def modal_split_scoring_session(
    gpu: str = _DEFAULT_GPU,
    backend: str = "cuda",
    precision: str = "fp32",
    num_correct_trials: int = 5,
    num_perf_trials: int = 100,
) -> Generator[ScoringFn, None, None]:
    """Open a split-pipeline Modal scoring session.

    Drop-in compatible with `modal_scoring_session`. Each `score()` call
    routes compilation to a CPU container and benchmarking to a GPU
    container, sharing artifacts via a persistent `modal.Volume`.

    Args:
        gpu: Modal GPU type (e.g. "L4", "H100").
        backend: Kernel backend ("cuda", "triton", ...).
        precision: Floating-point precision ("fp32", "fp16", "bf16").
        num_correct_trials: Correctness trial count.
        num_perf_trials: Performance trial count.

    Yields:
        A callable with the `ScoringFn` contract.
    """
    if gpu not in COMPUTE_CAPABILITY_BY_GPU:
        raise ValueError(
            f"Unknown GPU {gpu!r} — add it to COMPUTE_CAPABILITY_BY_GPU "
            f"before calling modal_split_scoring_session."
        )
    gpu_arch = GPU_ARCH_MAPPING.get(gpu, ["Ampere"])
    cc = COMPUTE_CAPABILITY_BY_GPU[gpu]

    compiler = ModalCpuCompiler()
    benchmarker_cls = ModalGpuBenchmarker.with_options(  # ty: ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue]
        gpu=gpu
    )

    with app.run():

        def score(
            mutated_kernel_code: str,
            reference_kernel_code: str,
            *,
            backend: str = backend,
            precision: str = precision,
            num_correct_trials: int = num_correct_trials,
            num_perf_trials: int = num_perf_trials,
            build_dir: Optional[Path] = None,  # accepted for signature compat, unused
        ) -> Option[KernelExecResult, ScoringError]:
            try:
                compile_result: dict[str, Any] = compiler.compile.remote(
                    mutated_kernel_code,
                    reference_kernel_code,
                    cc,
                )
            except Exception as exc:
                return Err(
                    ScoringError(
                        reason=f"Modal CPU compile failed: {type(exc).__name__}: {exc}",
                        cause=str(exc),
                    )
                )

            if compile_result.get("error"):
                return Ok(
                    KernelExecResult(
                        compiled=False,
                        metadata={
                            "compilation_error_name": "CpuCompileError",
                            "compilation_error": compile_result["error"],
                        },
                    )
                )

            cache_dir: str = compile_result["cache_dir"]

            try:
                exec_result: (
                    KernelExecResult | None
                ) = benchmarker_cls().evaluate.remote(
                    mutated_kernel_code=mutated_kernel_code,
                    reference_kernel_code=reference_kernel_code,
                    cache_dir=cache_dir,
                    gpu_arch=gpu_arch,
                    backend=backend,
                    precision=precision,
                    num_correct_trials=num_correct_trials,
                    num_perf_trials=num_perf_trials,
                )
            except Exception as exc:
                return Err(
                    ScoringError(
                        reason=f"Modal GPU benchmark failed: {type(exc).__name__}: {exc}",
                        cause=str(exc),
                    )
                )

            return _wrap_exec_result(exec_result)

        yield score


def run_split_scoring_on_modal(
    mutated_kernel_code: str,
    reference_kernel_code: str,
    *,
    gpu: str = _DEFAULT_GPU,
    backend: str = "cuda",
    precision: str = "fp32",
    num_correct_trials: int = 5,
    num_perf_trials: int = 100,
    timeout_seconds: int = 300,  # accepted for signature compat, unused
) -> Option[KernelExecResult, ScoringError]:
    """One-shot split-pipeline scoring. Prefer `modal_split_scoring_session`
    when evaluating multiple kernels so that session overhead amortises."""
    with modal_split_scoring_session(
        gpu=gpu,
        backend=backend,
        precision=precision,
        num_correct_trials=num_correct_trials,
        num_perf_trials=num_perf_trials,
    ) as score:
        return score(mutated_kernel_code, reference_kernel_code)


__all__ = [
    "ModalCpuCompiler",
    "ModalGpuBenchmarker",
    "app",
    "build_cache",
    "modal_split_scoring_session",
    "run_split_scoring_on_modal",
    "COMPUTE_CAPABILITY_BY_GPU",
]
