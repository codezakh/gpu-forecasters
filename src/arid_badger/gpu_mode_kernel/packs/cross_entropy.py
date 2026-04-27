"""Cross-entropy KernelPack — fused forward + backward.

Vendored from
``gpu-mode/reference-kernels/problems/princeton/cross_entropy_py``. The
upstream submission exports two top-level functions
(``cross_entropy_forward``, ``cross_entropy_backward``); we collapse
that into one ``custom_kernel(data) -> (losses, grad_logits)`` so the
abstraction's single-entry-point contract still applies. This framing
matches fused-cross-entropy implementations (Liger-Kernel etc.) where
the same softmax computation drives both outputs.

Test-case shape: ``B`` is fixed at 4096 by upstream; ``vocab_size``
varies. Tolerance: rtol=1e-2, atol=1e-3 (loose, per upstream).

No determinism context manager: cross-entropy goes through
softmax+gather, not cuDNN's conv autotuner, so timing is stable
across Modal containers without explicit pinning.
"""

from __future__ import annotations

from typing import Any, Mapping, Tuple, TypedDict

import modal
import torch
import torch.nn.functional as F
from pydantic import ConfigDict

from arid_badger.gpu_mode_kernel.core import CaseSpeedupBase, KernelExecResult
from arid_badger.gpu_mode_kernel.kernel_pack import KernelPack
from arid_badger.gpu_mode_kernel.modal_scoring import (
    DEFAULT_CLS_KWARGS,
    PackedModalRuntime,
    run_evaluate_candidate,
)
from arid_badger.kernelbench.modal_image import image


# ---------------------------------------------------------------------------
# Test args + cases
# ---------------------------------------------------------------------------

# Upstream fixes B=4096 (see eval.py). We keep the same constant here
# rather than threading it through every test args dict — it isn't a
# variable the search would benefit from sweeping.
B: int = 4096


class CrossEntropyTestArgs(TypedDict):
    vocab_size: int
    seed: int


# Upstream uses the same three shapes for tests and benchmarks. We do
# the same: correctness on these is the contract, and the timing
# numbers come from the same shapes.
CORRECTNESS_CASES: list[CrossEntropyTestArgs] = [
    {"vocab_size": 32000, "seed": 4242},
    {"vocab_size": 50264, "seed": 5236},
    {"vocab_size": 128256, "seed": 1001},
]

BENCHMARK_CASES: list[CrossEntropyTestArgs] = [
    {"vocab_size": 32000, "seed": 2146},
    {"vocab_size": 50264, "seed": 3129},
    {"vocab_size": 128256, "seed": 54352},
]


# ---------------------------------------------------------------------------
# CaseSpeedup with vocab-size shape field
# ---------------------------------------------------------------------------


class CrossEntropyCaseSpeedup(CaseSpeedupBase):
    model_config = ConfigDict(frozen=True)

    vocab_size: int

    @classmethod
    def from_exec_result(
        cls,
        test_args: Mapping[str, Any],
        exec_result: KernelExecResult,
    ) -> "CrossEntropyCaseSpeedup":
        speedup = (
            exec_result.ref_runtime_ns / exec_result.runtime_ns
            if exec_result.runtime_ns > 0
            else 0.0
        )
        return cls(
            vocab_size=test_args["vocab_size"],
            speedup=speedup,
            runtime_ns=exec_result.runtime_ns,
            ref_runtime_ns=exec_result.ref_runtime_ns,
        )

    def format_for_prompt(self) -> str:
        ref_us = self.ref_runtime_ns / 1_000.0
        candidate_us = self.runtime_ns / 1_000.0
        return (
            f"B={B}, V={self.vocab_size}: "
            f"{self.speedup:.3f}x "
            f"(ref: {ref_us:.1f}μs, candidate: {candidate_us:.1f}μs)"
        )


# ---------------------------------------------------------------------------
# Reference + input generator + correctness oracle
# ---------------------------------------------------------------------------

_DTYPE: torch.dtype = torch.bfloat16
_RTOL: float = 1e-2
_ATOL: float = 1e-3

CrossEntropyData = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
CrossEntropyOutput = Tuple[torch.Tensor, torch.Tensor]


def generate_input(vocab_size: int, seed: int) -> CrossEntropyData:
    """``(logits, targets, grad_output)`` deterministic under ``seed``."""
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    logits = torch.randn(
        B, vocab_size, dtype=_DTYPE, device="cuda", generator=gen
    ).contiguous()
    targets = torch.randint(
        0, vocab_size, (B,), device="cuda", generator=gen
    )
    grad_output = torch.randn(
        B, dtype=torch.float32, device="cuda", generator=gen
    ).contiguous()
    return logits, targets, grad_output


def _reference_forward(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.float(), targets, reduction="none")


def _reference_backward(
    logits: torch.Tensor, targets: torch.Tensor, grad_output: torch.Tensor
) -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=-1)
    grad = probs
    grad[torch.arange(logits.shape[0], device=logits.device), targets] -= 1.0
    grad = grad * grad_output.unsqueeze(1)
    return grad.to(logits.dtype)


def ref_kernel(data: CrossEntropyData) -> CrossEntropyOutput:
    """Reference fused fwd+bwd: returns ``(losses, grad_logits)``."""
    logits, targets, grad_output = data
    losses = _reference_forward(logits, targets)
    grad_logits = _reference_backward(logits, targets, grad_output)
    return losses, grad_logits


def check_implementation(
    data: CrossEntropyData, output: Any
) -> Tuple[bool, str]:
    """Validate the candidate's ``(losses, grad_logits)`` output.

    Mirrors upstream eval.py's correctness checks: shape + dtype +
    allclose for both outputs. Tolerances rtol=1e-2, atol=1e-3 are
    upstream defaults.
    """
    if not isinstance(output, tuple) or len(output) != 2:
        return (
            False,
            f"custom_kernel must return a 2-tuple (losses, grad_logits); "
            f"got {type(output).__name__} of length "
            f"{len(output) if hasattr(output, '__len__') else 'n/a'}",
        )
    sub_loss, sub_grad = output
    logits, targets, grad_output = data

    ref_loss = _reference_forward(logits, targets)
    ref_grad = _reference_backward(logits, targets, grad_output)

    if sub_loss.shape != ref_loss.shape:
        return (
            False,
            f"Forward shape mismatch: expected {tuple(ref_loss.shape)}, "
            f"got {tuple(sub_loss.shape)}",
        )
    if sub_loss.dtype != torch.float32:
        return (
            False,
            f"Forward dtype mismatch: expected float32, got {sub_loss.dtype}",
        )
    if sub_grad.shape != ref_grad.shape:
        return (
            False,
            f"Backward shape mismatch: expected {tuple(ref_grad.shape)}, "
            f"got {tuple(sub_grad.shape)}",
        )
    if sub_grad.dtype != _DTYPE:
        return (
            False,
            f"Backward dtype mismatch: expected {_DTYPE}, got {sub_grad.dtype}",
        )

    fwd_close = torch.allclose(sub_loss, ref_loss, atol=_ATOL, rtol=_RTOL)
    bwd_close = torch.allclose(sub_grad, ref_grad, atol=_ATOL, rtol=_RTOL)
    if fwd_close and bwd_close:
        return True, ""

    fwd_err = (sub_loss - ref_loss).abs().max().item()
    bwd_err = (sub_grad.float() - ref_grad.float()).abs().max().item()
    return (
        False,
        f"forward max err={fwd_err:.3e} {'OK' if fwd_close else 'FAIL'}; "
        f"backward max err={bwd_err:.3e} {'OK' if bwd_close else 'FAIL'} "
        f"(rtol={_RTOL}, atol={_ATOL})",
    )


# ---------------------------------------------------------------------------
# Seed kernel — fused fwd+bwd PyTorch baseline.
# ---------------------------------------------------------------------------

# The upstream submission.py is the same baseline split across two
# top-level functions. Here we collapse to one ``custom_kernel`` to
# fit the abstraction's single-symbol contract; the LLM is free to
# refactor internally.
_SEED_KERNEL_CODE: str = '''import torch
import torch.nn.functional as F


def custom_kernel(data):
    """Fused cross-entropy fwd+bwd, returning ``(losses, grad_logits)``.

    Args:
        data: tuple ``(logits, targets, grad_output)`` where
            - logits: (B, V) bfloat16 on CUDA
            - targets: (B,) int64 on CUDA
            - grad_output: (B,) float32 on CUDA

    Returns:
        Tuple ``(losses, grad_logits)`` where
            - losses: (B,) float32
            - grad_logits: (B, V) bfloat16
    """
    logits, targets, grad_output = data

    # Forward: per-row cross-entropy in float32 accumulation.
    losses = F.cross_entropy(logits.float(), targets, reduction="none")

    # Backward: dL/dlogit = (softmax(logits) - one_hot(targets)) * grad_output.
    probs = torch.softmax(logits.float(), dim=-1)
    grad = probs
    grad[torch.arange(logits.shape[0], device=logits.device), targets] -= 1.0
    grad = grad * grad_output.unsqueeze(1)
    grad_logits = grad.to(logits.dtype)

    return losses, grad_logits
'''


# ---------------------------------------------------------------------------
# Mutation-prompt body.
# ---------------------------------------------------------------------------

_PROMPT_BODY = r'''You are an expert Triton engineer tasked with writing a fused cross-entropy forward+backward kernel.

The operator is the standard categorical cross-entropy loss with mean-zero softmax gradients. It computes, per row of a (B, V) logits matrix:

  forward:    losses[b]      = -log_softmax(logits[b])[targets[b]]
  backward:   grad_logits[b] = (softmax(logits[b]) - one_hot(targets[b])) * grad_output[b]

The forward and backward share the softmax computation, so a fused implementation that computes the row's softmax once and emits both outputs is strictly better than two passes. This is the structure used by Liger-Kernel and similar fused implementations.

Your task:
- Implement a single ``custom_kernel`` that returns both outputs.
- No autograd; the candidate is called as a plain function.

Input:
- ``data``: tuple ``(logits, targets, grad_output)`` where
    - ``logits``     : ``torch.Tensor``, shape ``[B, V]``, dtype ``bfloat16``, on CUDA
    - ``targets``    : ``torch.Tensor``, shape ``[B]``,    dtype ``int64``,    on CUDA
    - ``grad_output``: ``torch.Tensor``, shape ``[B]``,    dtype ``float32``,  on CUDA

Output:
- Tuple ``(losses, grad_logits)`` where
    - ``losses``     : ``torch.Tensor``, shape ``[B]``,    dtype ``float32``
    - ``grad_logits``: ``torch.Tensor``, shape ``[B, V]``, dtype ``bfloat16``

**Problem Constraints:**
- B = 4096 (fixed)
- V ∈ {32000, 50264, 128256}
- V is guaranteed divisible by 8.
- Logits are finite real numbers (no -inf masking).
- Tolerance: rtol=1e-2, atol=1e-3 against fp32 reference.

**Remarks.** This kernel is bandwidth-bound: each row reads V bf16 logits (and writes V bf16 grads), and the row's reduction (max + sum-exp) lives in registers. The win comes from (1) computing softmax once and reusing it for both outputs, (2) processing each row in one program instance so the reduction stays in shared memory or registers, and (3) using fp32 accumulation for the softmax reduction even though storage is bf16.

Here is a PyTorch implementation. You should implement a kernel for the operations in ``ref_kernel``:

```python
import torch
import torch.nn.functional as F


def ref_kernel(data):
    logits, targets, grad_output = data

    # Forward: per-row cross-entropy in float32 accumulation.
    losses = F.cross_entropy(logits.float(), targets, reduction="none")

    # Backward: dL/dlogit = (softmax(logits) - one_hot(targets)) * grad_output.
    probs = torch.softmax(logits.float(), dim=-1)
    grad = probs
    grad[torch.arange(logits.shape[0], device=logits.device), targets] -= 1.0
    grad = grad * grad_output.unsqueeze(1)
    grad_logits = grad.to(logits.dtype)

    return losses, grad_logits
```

Here is some example skeleton code of the entrypoint function you will create:
```python
def custom_kernel(data):
    logits, targets, grad_output = data
    B, V = logits.shape

    # ... your kernel here ...

    return losses, grad_logits  # losses: [B] float32, grad_logits: [B, V] bfloat16
```

To help you understand which Triton version we are using, here is some example Triton code for an unrelated task:
```python
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = acc.to(tl.float16)
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    tl.store(c_ptrs, c)
```

A few general triton tips:
- ``tl.arange`` only takes constexpr arguments (static or ``tl.constexpr``)
- You cannot use ``continue`` in your kernel code
- ``tl.dot`` can only take in two input tensors
- There is no ``tl.mean``

Test cases for correctness:
'''


def _format_cases_block() -> str:
    lines: list[str] = []
    for c in CORRECTNESS_CASES:
        lines.append(f'  - {{"B": {B}, "V": {c["vocab_size"]}}}')
    lines.append("")
    lines.append("Benchmark cases for runtime (optimize runtime for these):")
    for c in BENCHMARK_CASES:
        lines.append(f'  - {{"B": {B}, "V": {c["vocab_size"]}}}')
    return "\n".join(lines) + "\n"


_KERNEL_DESCRIPTION_BODY: str = (
    _PROMPT_BODY.rstrip() + "\n" + _format_cases_block()
)


# ---------------------------------------------------------------------------
# Pack
# ---------------------------------------------------------------------------


CROSS_ENTROPY_PACK: KernelPack[CrossEntropyTestArgs, CrossEntropyCaseSpeedup] = (
    KernelPack(
        name="cross_entropy",
        modal_app_name="arid-badger-cross-entropy",
        correctness_cases=CORRECTNESS_CASES,
        benchmark_cases=BENCHMARK_CASES,
        ref_kernel=ref_kernel,
        generate_input=generate_input,
        check_implementation=check_implementation,
        seed_kernel_code=_SEED_KERNEL_CODE,
        determinism_ctx=None,
        case_speedup_type=CrossEntropyCaseSpeedup,
        kernel_description_body=_KERNEL_DESCRIPTION_BODY,
    )
)


# ---------------------------------------------------------------------------
# Modal app + benchmarker cls (stable module path so cloudpickle is happy).
# ---------------------------------------------------------------------------


app = modal.App(CROSS_ENTROPY_PACK.modal_app_name)


@app.cls(image=image, **DEFAULT_CLS_KWARGS)  # ty: ignore[unresolved-attribute]  # pyright: ignore[reportArgumentType]
class ModalCrossEntropyBenchmarker:
    """Modal benchmarker for the cross-entropy kernel pack."""

    @modal.method()
    def evaluate_candidate(
        self,
        mutated_kernel_code: str,
        test_cases: list[dict[str, object]],
        max_repeats: int = 100,
        max_time_ns: float = 10e9,
    ) -> list[Any]:
        return run_evaluate_candidate(
            pack=CROSS_ENTROPY_PACK,
            mutated_kernel_code=mutated_kernel_code,
            test_cases=test_cases,
            max_repeats=max_repeats,
            max_time_ns=max_time_ns,
        )


CROSS_ENTROPY_RUNTIME: PackedModalRuntime[
    CrossEntropyTestArgs, CrossEntropyCaseSpeedup
] = PackedModalRuntime(
    pack=CROSS_ENTROPY_PACK,
    app=app,
    benchmarker_cls=ModalCrossEntropyBenchmarker,
)
