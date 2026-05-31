"""Minimal Modal debug script. Run directly to see full container logs:

    cd 15-arid-badger
    uv run --env-file .env python scripts/debug_modal.py
"""

import modal

from gpu_forecasters.kernelbench.modal_image import app, image, GPU_ARCH_MAPPING


# ---------------------------------------------------------------------------
# Step 1: Can we start a container?
# ---------------------------------------------------------------------------

@app.function(image=image)
def step1_hello() -> str:
    print("step1: hello from container")
    return "ok"


# ---------------------------------------------------------------------------
# Step 2: Can we import kernelbench + torch?
# ---------------------------------------------------------------------------

@app.function(image=image)
def step2_imports() -> str:
    import sys
    print(f"Python: {sys.version}")
    print(f"sys.path: {sys.path}")

    print("Importing torch...")
    import torch
    print(f"torch {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    print("Importing kernelbench...")
    from kernelbench.eval import eval_kernel_against_ref
    from kernelbench.utils import set_gpu_arch
    print("All imports OK")
    return "ok"


# ---------------------------------------------------------------------------
# Step 3: Can we run eval_kernel_against_ref on a GPU?
# ---------------------------------------------------------------------------

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


@app.function(image=image, gpu="L4")
def step3_eval(reference_code: str, kernel_code: str) -> dict:
    import torch
    from kernelbench.eval import eval_kernel_against_ref, get_torch_dtype_from_string
    from kernelbench.utils import set_gpu_arch

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    set_gpu_arch(["Ada"])

    print("Running eval_kernel_against_ref...")
    result = eval_kernel_against_ref(
        original_model_src=reference_code,
        custom_model_src=kernel_code,
        num_correct_trials=1,
        num_perf_trials=5,
        measure_performance=True,
        timing_method="cuda_event",
        verbose=True,
        device=torch.device("cuda:0"),
        backend="cuda",
        precision=get_torch_dtype_from_string("fp32"),
    )

    print(f"compiled={result.compiled}")
    print(f"correctness={result.correctness}")
    print(f"runtime={result.runtime}")
    print(f"ref_runtime={result.ref_runtime}")
    return {
        "compiled": result.compiled,
        "correctness": result.correctness,
        "runtime": result.runtime,
        "ref_runtime": result.ref_runtime,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    modal.enable_output()
    with app.run():
        print("\n=== Step 1: Container starts ===")
        t0 = time.time()
        r = step1_hello.remote()
        print(f"Result: {r} ({time.time() - t0:.1f}s)\n")

        # Step 2 (import-only) skipped: kernelbench import hangs without a GPU.
        # Step 3 covers imports implicitly.

        print("=== Step 3: GPU eval ===")
        t0 = time.time()
        r = step3_eval.remote(REFERENCE_KERNEL, CORRECT_KERNEL)
        print(f"Result: {r} ({time.time() - t0:.1f}s)\n")

    print("All steps passed!")
