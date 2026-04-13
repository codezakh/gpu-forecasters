"""Incremental smoke tests for Modal container setup.

Uses a separate Modal app from the production evaluator so these functions
don't pollute the real app (and don't force GPU allocation when only testing
imports, etc.).

Run with -x to stop at first failure and isolate which layer is broken.
Run with -s to see modal.enable_output() image build logs (pytest captures
output by default, which suppresses them):

    uv run --env-file .env pytest -m modal tests/test_modal_smoke.py -v -x -s
"""

import modal
import pytest

modal.enable_output()

from arid_badger.kernelbench.modal_image import image

# Separate app so smoke test functions don't register on the production app.
_smoke_app = modal.App("arid-badger-smoke-test")


@_smoke_app.function(image=image)
def _smoke_hello():
    return "hello from modal"


@_smoke_app.function(image=image, timeout=600)
def _check_imports() -> dict[str, str]:
    import sys
    print(f"Python: {sys.version}", flush=True)

    print("Importing torch...", flush=True)
    import torch
    print(f"torch {torch.__version__}", flush=True)

    print("Importing kernelbench.eval...", flush=True)
    import kernelbench.eval as kb_eval
    print("Importing kernelbench.utils...", flush=True)
    import kernelbench.utils as kb_utils
    print("All imports done", flush=True)

    return {
        "torch": torch.__version__,
        "eval_fn": type(kb_eval.eval_kernel_against_ref).__name__,
        "set_gpu_arch": type(kb_utils.set_gpu_arch).__name__,
    }


REFERENCE_KERNEL = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)

N = 128

def get_inputs():
    a = torch.rand(N, N)
    b = torch.rand(N, N)
    return [a, b]

def get_init_inputs():
    return []
"""

CORRECT_KERNEL = """
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.mm(a, b)
"""


@_smoke_app.function(image=image, gpu="H100")
def _run_eval(reference_code: str, kernel_code: str) -> dict:
    import torch
    from kernelbench.eval import eval_kernel_against_ref, get_torch_dtype_from_string
    from kernelbench.utils import set_gpu_arch

    set_gpu_arch(["Hopper"])

    result = eval_kernel_against_ref(
        original_model_src=reference_code,
        custom_model_src=kernel_code,
        num_correct_trials=1,
        num_perf_trials=5,
        measure_performance=True,
        timing_method="cuda_event",
        verbose=False,
        device=torch.device("cuda:0"),
        backend="cuda",
        precision=get_torch_dtype_from_string("fp32"),
    )

    return {
        "compiled": result.compiled,
        "correctness": result.correctness,
        "runtime": result.runtime,
        "ref_runtime": result.ref_runtime,
    }


# ---------------------------------------------------------------------------
# Tests — run with -x to stop at first failure
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.modal
def test_step1_container_starts() -> None:
    """Container starts and returns a value."""
    with _smoke_app.run():
        result = _smoke_hello.remote()
    assert result == "hello from modal"


@pytest.mark.integration
@pytest.mark.modal
def test_step2_imports_work() -> None:
    """kernelbench and torch are importable in the container."""
    with _smoke_app.run():
        info = _check_imports.remote()
    assert "torch" in info
    assert info["eval_fn"] == "function"
    assert info["set_gpu_arch"] == "function"


@pytest.mark.integration
@pytest.mark.modal
def test_step3_eval_kernel_on_gpu() -> None:
    """eval_kernel_against_ref runs successfully on a Modal GPU."""
    with _smoke_app.run():
        result = _run_eval.remote(REFERENCE_KERNEL, CORRECT_KERNEL)
    assert result["compiled"], f"Kernel failed to compile: {result}"
    assert result["correctness"], f"Kernel produced incorrect output: {result}"
    assert result["runtime"] > 0
    assert result["ref_runtime"] > 0
