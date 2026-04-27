"""Block-32 FP8 (E4M3) per-token-group quantization KernelPack.

Vendored from
``gpu-mode/reference-kernels/problems/helion/fp8_quant_py``.
This is the standard activation-quantization kernel used in production
LLM inference (DeepSeek-V3, Llama 3, Qwen3) for W8A8 quantized
inference: per-group absmax → per-group scale → clamped-quantize.

Per gh070 v1, this is a quantization-flavored Triton entry — slight
tonal mismatch with the paper's "compositional" framing (this kernel
is a flat reduce + elementwise) but a clean, small kernel that
broadens the testbed beyond the SSM/attention/recurrent cluster.

Output is float32 clamped to FP8 range, not actual fp8 storage —
upstream chose that for broad GPU compatibility (Ampere does not have
fp8 instructions). The candidate writes into the pre-allocated x_q
and x_s tensors and returns them (mirroring upstream's signature).

Tolerance: rtol=1e-3, atol=1e-3 — upstream values.
"""

from __future__ import annotations

from typing import Any, Mapping, Tuple, TypedDict

import modal
import torch
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
# Test args + cases — vendored from upstream task.yml.
# ---------------------------------------------------------------------------

FP8_MAX: float = 448.0
FP8_MIN: float = -448.0
FP8_EPS: float = 1e-10


class Fp8QuantTestArgs(TypedDict):
    num_tokens: int
    hidden_dim: int
    group_size: int
    seed: int


CORRECTNESS_CASES: list[Fp8QuantTestArgs] = [
    {"num_tokens": 1, "hidden_dim": 256, "group_size": 64, "seed": 4242},
    {"num_tokens": 4, "hidden_dim": 512, "group_size": 128, "seed": 5236},
    {"num_tokens": 16, "hidden_dim": 1024, "group_size": 64, "seed": 1001},
    {"num_tokens": 1, "hidden_dim": 4096, "group_size": 128, "seed": 5531},
    {"num_tokens": 8, "hidden_dim": 4096, "group_size": 128, "seed": 9173},
]

BENCHMARK_CASES: list[Fp8QuantTestArgs] = [
    {"num_tokens": 256, "hidden_dim": 4096, "group_size": 128, "seed": 2146},
    {"num_tokens": 256, "hidden_dim": 8192, "group_size": 128, "seed": 3129},
    {"num_tokens": 4096, "hidden_dim": 7168, "group_size": 128, "seed": 54352},
]


# ---------------------------------------------------------------------------
# CaseSpeedup with shape fields
# ---------------------------------------------------------------------------


class Fp8QuantCaseSpeedup(CaseSpeedupBase):
    model_config = ConfigDict(frozen=True)

    num_tokens: int
    hidden_dim: int
    group_size: int

    @classmethod
    def from_exec_result(
        cls,
        test_args: Mapping[str, Any],
        exec_result: KernelExecResult,
    ) -> "Fp8QuantCaseSpeedup":
        speedup = (
            exec_result.ref_runtime_ns / exec_result.runtime_ns
            if exec_result.runtime_ns > 0
            else 0.0
        )
        return cls(
            num_tokens=test_args["num_tokens"],
            hidden_dim=test_args["hidden_dim"],
            group_size=test_args["group_size"],
            speedup=speedup,
            runtime_ns=exec_result.runtime_ns,
            ref_runtime_ns=exec_result.ref_runtime_ns,
        )

    def format_for_prompt(self) -> str:
        ref_us = self.ref_runtime_ns / 1_000.0
        candidate_us = self.runtime_ns / 1_000.0
        return (
            f"num_tokens={self.num_tokens}, hidden_dim={self.hidden_dim}, "
            f"group_size={self.group_size}: "
            f"{self.speedup:.3f}x "
            f"(ref: {ref_us:.1f}μs, candidate: {candidate_us:.1f}μs)"
        )


# ---------------------------------------------------------------------------
# Reference + input generator + correctness oracle.
# Vendored from upstream reference.py.
# ---------------------------------------------------------------------------

_RTOL: float = 1e-3
_ATOL: float = 1e-3

Fp8QuantData = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
Fp8QuantOutput = Tuple[torch.Tensor, torch.Tensor]


def generate_input(
    num_tokens: int, hidden_dim: int, group_size: int, seed: int
) -> Fp8QuantData:
    """``(x, x_q, x_s)`` deterministic under ``seed``.

    ``x_q`` and ``x_s`` are pre-allocated empty buffers the candidate
    writes into — upstream's contract is that the candidate fills both
    in-place and also returns them.
    """
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    x = torch.randn(
        num_tokens,
        hidden_dim,
        dtype=torch.float32,
        device="cuda",
        generator=gen,
    ).contiguous()
    x_q = torch.empty(
        num_tokens, hidden_dim, dtype=torch.float32, device="cuda"
    ).contiguous()
    x_s = torch.empty(
        num_tokens, hidden_dim // group_size, dtype=torch.float32, device="cuda"
    ).contiguous()
    return x, x_q, x_s


def ref_kernel(data: Fp8QuantData) -> Fp8QuantOutput:
    """Per-group absmax → scale → clamped-quantize.

    For each group of ``group_size`` contiguous elements:
      absmax = max(|x_group|)
      scale  = max(absmax, eps) / FP8_MAX
      x_q    = clamp(x / scale, FP8_MIN, FP8_MAX)
    """
    x, x_q, x_s = data
    num_tokens, hidden_dim = x.shape
    num_groups = x_s.shape[1]
    group_size = hidden_dim // num_groups

    x_f32 = x.float()
    x_grouped = x_f32.reshape(num_tokens, num_groups, group_size)

    absmax = x_grouped.abs().amax(dim=-1).clamp(min=FP8_EPS)
    scale = absmax / FP8_MAX

    quantized = (x_grouped / scale.unsqueeze(-1)).clamp(FP8_MIN, FP8_MAX)
    quantized = quantized.reshape(num_tokens, hidden_dim)

    x_q[...] = quantized
    x_s[...] = scale
    return x_q, x_s


def check_implementation(
    data: Fp8QuantData, output: Any
) -> Tuple[bool, str]:
    """Validate the candidate's ``(x_q, x_s)`` output.

    Per-output shape + verbose-allclose on both quantized values and
    per-group scales. Both must be within tolerance.
    """
    if not isinstance(output, tuple) or len(output) != 2:
        return (
            False,
            f"custom_kernel must return a 2-tuple (x_q, x_s); got "
            f"{type(output).__name__} of length "
            f"{len(output) if hasattr(output, '__len__') else 'n/a'}",
        )
    sub_q, sub_s = output

    # Re-derive expectations from a fresh copy of the input — the
    # in-place writes the candidate may have already done to data[1]
    # and data[2] don't pollute this check.
    x = data[0]
    num_tokens, hidden_dim = x.shape
    num_groups = data[2].shape[1]
    group_size = hidden_dim // num_groups
    x_grouped = x.float().reshape(num_tokens, num_groups, group_size)
    absmax = x_grouped.abs().amax(dim=-1).clamp(min=FP8_EPS)
    expected_s = absmax / FP8_MAX
    expected_q = (x_grouped / expected_s.unsqueeze(-1)).clamp(FP8_MIN, FP8_MAX)
    expected_q = expected_q.reshape(num_tokens, hidden_dim)

    if sub_q.shape != expected_q.shape:
        return (
            False,
            f"x_q shape mismatch: expected {tuple(expected_q.shape)}, "
            f"got {tuple(sub_q.shape)}",
        )
    if sub_s.shape != expected_s.shape:
        return (
            False,
            f"x_s shape mismatch: expected {tuple(expected_s.shape)}, "
            f"got {tuple(sub_s.shape)}",
        )

    q_close = torch.allclose(
        sub_q.float(), expected_q.float(), atol=_ATOL, rtol=_RTOL
    )
    s_close = torch.allclose(
        sub_s.float(), expected_s.float(), atol=_ATOL, rtol=_RTOL
    )
    if q_close and s_close:
        return True, ""

    q_err = (sub_q.float() - expected_q.float()).abs().max().item()
    s_err = (sub_s.float() - expected_s.float()).abs().max().item()
    return (
        False,
        f"x_q max err={q_err:.3e} {'OK' if q_close else 'FAIL'}; "
        f"x_s max err={s_err:.3e} {'OK' if s_close else 'FAIL'} "
        f"(rtol={_RTOL}, atol={_ATOL})",
    )


# ---------------------------------------------------------------------------
# Seed kernel — pure-PyTorch baseline mirroring ref_kernel.
# Same shape as the reference; the LLM is asked to fuse the per-group
# absmax + clamped-divide into a single Triton kernel.
# ---------------------------------------------------------------------------

_SEED_KERNEL_CODE: str = '''import torch


FP8_MAX = 448.0
FP8_MIN = -448.0
FP8_EPS = 1e-10


def custom_kernel(data):
    """Pure-PyTorch per-token-group FP8 quantization.

    Args:
        data: tuple ``(x, x_q, x_s)`` where
            - x:   [num_tokens, hidden_dim]                float32 on CUDA — input
            - x_q: [num_tokens, hidden_dim]                float32 on CUDA — pre-allocated output buffer
            - x_s: [num_tokens, hidden_dim // group_size]  float32 on CUDA — pre-allocated scale buffer

    Returns:
        Tuple ``(x_q, x_s)`` with the same buffers populated in-place.
    """
    x, x_q, x_s = data
    num_tokens, hidden_dim = x.shape
    num_groups = x_s.shape[1]
    group_size = hidden_dim // num_groups

    x_f32 = x.float()
    x_grouped = x_f32.reshape(num_tokens, num_groups, group_size)

    absmax = x_grouped.abs().amax(dim=-1).clamp(min=FP8_EPS)
    scale = absmax / FP8_MAX

    quantized = (x_grouped / scale.unsqueeze(-1)).clamp(FP8_MIN, FP8_MAX)
    quantized = quantized.reshape(num_tokens, hidden_dim)

    x_q[...] = quantized
    x_s[...] = scale
    return x_q, x_s
'''


# ---------------------------------------------------------------------------
# Mutation-prompt body. Auto-generates the test/benchmark cases listing
# from the python literals so the prompt can never go out of sync with
# the cases the candidate is actually scored against.
# ---------------------------------------------------------------------------


_PROMPT_BODY = r'''You are an expert Triton engineer tasked with writing a fused per-token-group FP8 (E4M3) quantization kernel.

This is the standard activation-quantization kernel used in production LLM inference (DeepSeek-V3, Llama 3, Qwen3) for W8A8 quantized inference. For each token row, the hidden dimension is split into contiguous groups of ``group_size`` elements; each group gets its own per-group absmax and scale.

For each group of ``group_size`` contiguous elements:
```
absmax = max(|x_group|)
scale  = max(absmax, 1e-10) / 448.0
x_q    = clamp(x / scale, -448.0, 448.0)
```
where ``448.0`` is the max representable value in FP8 E4M3 format.

NOTE: The output ``x_q`` is float32 *clamped* to the FP8 range — not actual fp8 storage. This makes the kernel runnable on broadly compatible hardware (Ampere lacks fp8 instructions).

Your task:
- Implement a single ``custom_kernel`` that fills the pre-allocated ``x_q`` and ``x_s`` buffers and returns them as a tuple.
- No autograd; the candidate is called as a plain function.

Input:
- ``data``: tuple ``(x, x_q, x_s)`` where
    - ``x``:   ``torch.Tensor``, shape ``[num_tokens, hidden_dim]``,                    dtype ``float32``, on CUDA — input activations
    - ``x_q``: ``torch.Tensor``, shape ``[num_tokens, hidden_dim]``,                    dtype ``float32``, on CUDA — pre-allocated output buffer for quantized values (clamped to FP8 range)
    - ``x_s``: ``torch.Tensor``, shape ``[num_tokens, hidden_dim // group_size]``,      dtype ``float32``, on CUDA — pre-allocated output buffer for per-group scale factors

Output:
- Tuple ``(x_q, x_s)`` (the same buffers, written in-place).

**Problem Constraints:**
- ``hidden_dim`` is always divisible by ``group_size``.
- ``group_size`` is always a power-of-2 in ``{64, 128}`` for the cases here.
- Tolerance: rtol=1e-3, atol=1e-3 against the fp32 reference.

**Remarks.** The kernel is bandwidth-bound: every element of ``x`` is read once and written once, with a single per-group reduction in between. The win comes from (1) doing the absmax reduction and the clamped-divide in a single program per ``(token, group)`` so the group's elements are loaded into registers once, (2) parallelizing across ``(num_tokens, num_groups)`` since groups are independent, and (3) avoiding any HBM round-trip for the per-group scale.

Here is a PyTorch reference implementation. You should implement a kernel for the operations in ``ref_kernel``:

```python
import torch


FP8_MAX = 448.0
FP8_MIN = -448.0
FP8_EPS = 1e-10


def ref_kernel(data):
    x, x_q, x_s = data
    num_tokens, hidden_dim = x.shape
    num_groups = x_s.shape[1]
    group_size = hidden_dim // num_groups

    x_f32 = x.float()
    x_grouped = x_f32.reshape(num_tokens, num_groups, group_size)

    absmax = x_grouped.abs().amax(dim=-1).clamp(min=FP8_EPS)
    scale = absmax / FP8_MAX

    quantized = (x_grouped / scale.unsqueeze(-1)).clamp(FP8_MIN, FP8_MAX)
    quantized = quantized.reshape(num_tokens, hidden_dim)

    x_q[...] = quantized
    x_s[...] = scale
    return x_q, x_s
```

Here is some example skeleton code of the entrypoint function you will create:
```python
def custom_kernel(data):
    x, x_q, x_s = data

    # ... your kernel here, writing into x_q and x_s ...

    return x_q, x_s  # x_q: [num_tokens, hidden_dim] float32 (clamped to FP8 range), x_s: [num_tokens, hidden_dim // group_size] float32
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
        lines.append(
            f'  - {{"num_tokens": {c["num_tokens"]}, '
            f'"hidden_dim": {c["hidden_dim"]}, '
            f'"group_size": {c["group_size"]}}}'
        )
    lines.append("")
    lines.append("Benchmark cases for runtime (optimize runtime for these):")
    for c in BENCHMARK_CASES:
        lines.append(
            f'  - {{"num_tokens": {c["num_tokens"]}, '
            f'"hidden_dim": {c["hidden_dim"]}, '
            f'"group_size": {c["group_size"]}}}'
        )
    return "\n".join(lines) + "\n"


_KERNEL_DESCRIPTION_BODY: str = (
    _PROMPT_BODY.rstrip() + "\n" + _format_cases_block()
)


# ---------------------------------------------------------------------------
# Pack
# ---------------------------------------------------------------------------


FP8_QUANT_PACK: KernelPack[Fp8QuantTestArgs, Fp8QuantCaseSpeedup] = KernelPack(
    name="fp8_quant",
    modal_app_name="arid-badger-fp8-quant",
    correctness_cases=CORRECTNESS_CASES,
    benchmark_cases=BENCHMARK_CASES,
    ref_kernel=ref_kernel,
    generate_input=generate_input,
    check_implementation=check_implementation,
    seed_kernel_code=_SEED_KERNEL_CODE,
    determinism_ctx=None,
    case_speedup_type=Fp8QuantCaseSpeedup,
    kernel_description_body=_KERNEL_DESCRIPTION_BODY,
)


# ---------------------------------------------------------------------------
# Modal app + benchmarker cls (stable module path so cloudpickle is happy).
# ---------------------------------------------------------------------------


app = modal.App(FP8_QUANT_PACK.modal_app_name)


@app.cls(image=image, **DEFAULT_CLS_KWARGS)  # ty: ignore[unresolved-attribute]  # pyright: ignore[reportArgumentType]
class ModalFp8QuantBenchmarker:
    """Modal benchmarker for the FP8 per-token-group quantization pack."""

    @modal.method()
    def evaluate_candidate(
        self,
        mutated_kernel_code: str,
        test_cases: list[dict[str, object]],
        max_repeats: int = 100,
        max_time_ns: float = 10e9,
    ) -> list[Any]:
        return run_evaluate_candidate(
            pack=FP8_QUANT_PACK,
            mutated_kernel_code=mutated_kernel_code,
            test_cases=test_cases,
            max_repeats=max_repeats,
            max_time_ns=max_time_ns,
        )


FP8_QUANT_RUNTIME: PackedModalRuntime[
    Fp8QuantTestArgs, Fp8QuantCaseSpeedup
] = PackedModalRuntime(
    pack=FP8_QUANT_PACK,
    app=app,
    benchmarker_cls=ModalFp8QuantBenchmarker,
)
