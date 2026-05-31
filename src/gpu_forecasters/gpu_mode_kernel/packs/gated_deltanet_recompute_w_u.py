"""Gated DeltaNet recompute-w-u KernelPack.

Vendored from
``gpu-mode/reference-kernels/problems/helion/gated_deltanet_recompute_w_u_py``.
This is the WY-transform forward — for each chunk, compute
``u = A @ (v * beta)`` and ``w = A @ (k * beta * exp(g))`` — that
produces the chunk-fwd-h kernel's inputs (``w``, ``u``) from the raw
``(k, v, beta, A, g)``. Together with chunk-fwd-h and chunk-fwd-o this
is the third per-chunk kernel in the chunkwise parallel forward pass of
Gated DeltaNet (arXiv:2412.06464, ICLR 2025).

Per gh070 v1, this sweetens the GDN trio into an in-architecture
multi-kernel story. Same B/T/H/K/V test/benchmark shapes as GDN h/o so
cross-pack comparisons stay clean.

Like GDN o (and unlike GDN h), every chunk here is independent — no
Python loop in the reference, just batched matmuls. The seed kernel is
the reference verbatim; the LLM's job is to fuse the per-chunk matmul
chain into a single Triton kernel parallelizing across ``(B, NT, H)``.

Tolerance: rtol=1e-3, atol=1e-3 — upstream values, looser than TriMul
because both the reference and the candidate live in fp32.
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
# Test args + cases — vendored from upstream task.yml. Match GDN h/o
# verbatim so cross-pack comparisons stay clean.
# ---------------------------------------------------------------------------

CHUNK_SIZE: int = 64


class GdnRecomputeWUTestArgs(TypedDict):
    B: int
    T: int
    H: int
    K: int
    V: int
    seed: int


CORRECTNESS_CASES: list[GdnRecomputeWUTestArgs] = [
    {"B": 1, "T": 64, "H": 2, "K": 64, "V": 64, "seed": 4242},
    {"B": 2, "T": 128, "H": 4, "K": 64, "V": 64, "seed": 5236},
    {"B": 1, "T": 256, "H": 4, "K": 64, "V": 128, "seed": 1001},
]

BENCHMARK_CASES: list[GdnRecomputeWUTestArgs] = [
    {"B": 1, "T": 64, "H": 1, "K": 64, "V": 64, "seed": 31232},
    {"B": 2, "T": 512, "H": 3, "K": 64, "V": 64, "seed": 4052},
    {"B": 2, "T": 1024, "H": 3, "K": 64, "V": 64, "seed": 2146},
]


# ---------------------------------------------------------------------------
# CaseSpeedup with shape fields (identical schema to GDN h/o)
# ---------------------------------------------------------------------------


class GdnRecomputeWUCaseSpeedup(CaseSpeedupBase):
    model_config = ConfigDict(frozen=True)

    B: int
    T: int
    H: int
    K: int
    V: int

    @classmethod
    def from_exec_result(
        cls,
        test_args: Mapping[str, Any],
        exec_result: KernelExecResult,
    ) -> "GdnRecomputeWUCaseSpeedup":
        speedup = (
            exec_result.ref_runtime_ns / exec_result.runtime_ns
            if exec_result.runtime_ns > 0
            else 0.0
        )
        return cls(
            B=test_args["B"],
            T=test_args["T"],
            H=test_args["H"],
            K=test_args["K"],
            V=test_args["V"],
            speedup=speedup,
            runtime_ns=exec_result.runtime_ns,
            ref_runtime_ns=exec_result.ref_runtime_ns,
        )

    def format_for_prompt(self) -> str:
        ref_us = self.ref_runtime_ns / 1_000.0
        candidate_us = self.runtime_ns / 1_000.0
        return (
            f"B={self.B}, T={self.T}, H={self.H}, K={self.K}, V={self.V}: "
            f"{self.speedup:.3f}x "
            f"(ref: {ref_us:.1f}μs, candidate: {candidate_us:.1f}μs)"
        )


# ---------------------------------------------------------------------------
# Reference + input generator + correctness oracle.
# Vendored from upstream reference.py. The eager helpers
# (_chunk_local_cumsum_eager etc.) are pre-computation for input
# generation; ref_kernel itself is the per-chunk WY transform the LLM
# is asked to optimize.
# ---------------------------------------------------------------------------

_RTOL: float = 1e-3
_ATOL: float = 1e-3

GdnRecomputeWUData = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]
GdnRecomputeWUOutput = Tuple[torch.Tensor, torch.Tensor]


def _chunk_local_cumsum_eager(g: torch.Tensor, chunk_size: int) -> torch.Tensor:
    B, T, H = g.shape
    C = chunk_size
    return g.float().reshape(B, T // C, C, H).cumsum(dim=2).reshape(B, T, H)


def _chunk_scaled_dot_kkt_fwd_eager(
    k: torch.Tensor, g_cumsum: torch.Tensor, beta: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    B, T, H, K = k.shape
    C = chunk_size
    NT = T // C
    k_c = k.float().reshape(B, NT, C, H, K).permute(0, 1, 3, 2, 4)
    g_c = g_cumsum.float().reshape(B, NT, C, H).permute(0, 1, 3, 2)
    beta_c = beta.float().reshape(B, NT, C, H).permute(0, 1, 3, 2)
    kkt = k_c @ k_c.transpose(-1, -2)
    strict_lower = torch.tril(torch.ones(C, C, device=k.device), diagonal=-1)
    g_diff = g_c.unsqueeze(-1) - g_c.unsqueeze(-2)
    g_diff = g_diff * strict_lower
    A = kkt * beta_c.unsqueeze(-1) * torch.exp(g_diff) * strict_lower
    return A.permute(0, 1, 3, 2, 4).reshape(B, T, H, C).to(torch.float32)


def _solve_tril_eager(A: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
    B, T, H, C = A.shape
    NT = T // C
    A_mat = A.float().reshape(B, NT, C, H, C).permute(0, 1, 3, 2, 4)
    eye = torch.eye(C, device=A.device).expand_as(A_mat)
    result = torch.linalg.solve_triangular(eye + A_mat, eye, upper=False)
    return result.permute(0, 1, 3, 2, 4).reshape(B, T, H, C).to(output_dtype)


def generate_input(
    B: int, T: int, H: int, K: int, V: int, seed: int
) -> GdnRecomputeWUData:
    """``(k, v, beta, A, g_cumsum)`` deterministic under ``seed``.

    Mirrors upstream's reference.py generate_input exactly: A is the
    solved WY matrix produced from the raw (k, beta, g_cumsum) — the
    candidate must produce w/u given A as a precomputed input.
    """
    torch.manual_seed(seed)
    device = "cuda"
    k = torch.randn(B, T, H, K, dtype=torch.float32, device=device) / K**0.5
    v = torch.randn(B, T, H, V, dtype=torch.float32, device=device)
    beta = torch.sigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device))
    g_inc = -torch.abs(torch.randn(B, T, H, dtype=torch.float32, device=device))
    g = g_inc.cumsum(dim=1)
    g_cumsum = _chunk_local_cumsum_eager(g, chunk_size=CHUNK_SIZE)
    A_kkt = _chunk_scaled_dot_kkt_fwd_eager(
        k=k, g_cumsum=g_cumsum, beta=beta, chunk_size=CHUNK_SIZE
    )
    A_solve = _solve_tril_eager(A=A_kkt, output_dtype=k.dtype)
    return (
        k.contiguous(),
        v.contiguous(),
        beta.contiguous(),
        A_solve.contiguous(),
        g_cumsum.contiguous(),
    )


def ref_kernel(data: GdnRecomputeWUData) -> GdnRecomputeWUOutput:
    """Per-chunk WY transform — the reference the candidate must match
    numerically (and beat in wall-clock).

    Returns ``(w, u)`` where:
        w : [B, T, H, K] (k.dtype) — WY-transformed keys
        u : [B, T, H, V] (v.dtype) — WY-transformed values
    """
    k, v, beta, A, g = data
    B, T, H, K = k.shape
    V = v.shape[-1]
    C = A.shape[-1]
    NT = T // C
    k_c = k.float().reshape(B, NT, C, H, K).permute(0, 1, 3, 2, 4)
    v_c = v.float().reshape(B, NT, C, H, V).permute(0, 1, 3, 2, 4)
    beta_c = beta.float().reshape(B, NT, C, H).permute(0, 1, 3, 2)
    g_c = g.float().reshape(B, NT, C, H).permute(0, 1, 3, 2)
    A_c = A.float().reshape(B, NT, C, H, C).permute(0, 1, 3, 2, 4)
    u_c = A_c @ (v_c * beta_c.unsqueeze(-1))
    w_c = A_c @ (k_c * (beta_c * torch.exp(g_c)).unsqueeze(-1))
    w = w_c.permute(0, 1, 3, 2, 4).reshape(B, T, H, K).to(k.dtype)
    u = u_c.permute(0, 1, 3, 2, 4).reshape(B, T, H, V).to(v.dtype)
    return w, u


def check_implementation(
    data: GdnRecomputeWUData, output: Any
) -> Tuple[bool, str]:
    """Validate the candidate's ``(w, u)`` output.

    Mirrors upstream reference.py's check_implementation: per-output
    shape + dtype + verbose-allclose. Returns the worst-offender error
    string when both sub-checks fail so the LLM gets the most-actionable
    feedback.
    """
    if not isinstance(output, tuple) or len(output) != 2:
        return (
            False,
            f"custom_kernel must return a 2-tuple (w, u); got "
            f"{type(output).__name__} of length "
            f"{len(output) if hasattr(output, '__len__') else 'n/a'}",
        )
    sub_w, sub_u = output
    expected_w, expected_u = ref_kernel(data)

    if sub_w.shape != expected_w.shape:
        return (
            False,
            f"w shape mismatch: expected {tuple(expected_w.shape)}, "
            f"got {tuple(sub_w.shape)}",
        )
    if sub_u.shape != expected_u.shape:
        return (
            False,
            f"u shape mismatch: expected {tuple(expected_u.shape)}, "
            f"got {tuple(sub_u.shape)}",
        )

    w_close = torch.allclose(
        sub_w.float(), expected_w.float(), atol=_ATOL, rtol=_RTOL
    )
    u_close = torch.allclose(
        sub_u.float(), expected_u.float(), atol=_ATOL, rtol=_RTOL
    )
    if w_close and u_close:
        return True, ""

    w_err = (sub_w.float() - expected_w.float()).abs().max().item()
    u_err = (sub_u.float() - expected_u.float()).abs().max().item()
    return (
        False,
        f"w max err={w_err:.3e} {'OK' if w_close else 'FAIL'}; "
        f"u max err={u_err:.3e} {'OK' if u_close else 'FAIL'} "
        f"(rtol={_RTOL}, atol={_ATOL})",
    )


# ---------------------------------------------------------------------------
# Seed kernel — pure-PyTorch chunked WY transform baseline.
# Same shape as ref_kernel; the LLM is asked to fuse the per-chunk
# matmul chain into a Triton kernel that parallelizes across (B, NT, H).
# ---------------------------------------------------------------------------

_SEED_KERNEL_CODE: str = '''import torch


CHUNK_SIZE = 64


def custom_kernel(data):
    """Pure-PyTorch per-chunk WY transform — fully parallel across
    chunks via batched matmuls.

    Args:
        data: tuple ``(k, v, beta, A, g)`` where
            - k:    [B, T, H, K] float32 on CUDA
            - v:    [B, T, H, V] float32 on CUDA
            - beta: [B, T, H]    float32 on CUDA
            - A:    [B, T, H, BT] float32 on CUDA (BT=64; WY matrix)
            - g:    [B, T, H]    float32 on CUDA (chunk-local cumsum'd)

    Returns:
        Tuple ``(w, u)`` where
            - w : [B, T, H, K] (k.dtype) — WY-transformed keys
            - u : [B, T, H, V] (v.dtype) — WY-transformed values
    """
    k, v, beta, A, g = data
    B, T, H, K = k.shape
    V = v.shape[-1]
    C = A.shape[-1]
    NT = T // C
    k_c = k.float().reshape(B, NT, C, H, K).permute(0, 1, 3, 2, 4)
    v_c = v.float().reshape(B, NT, C, H, V).permute(0, 1, 3, 2, 4)
    beta_c = beta.float().reshape(B, NT, C, H).permute(0, 1, 3, 2)
    g_c = g.float().reshape(B, NT, C, H).permute(0, 1, 3, 2)
    A_c = A.float().reshape(B, NT, C, H, C).permute(0, 1, 3, 2, 4)
    u_c = A_c @ (v_c * beta_c.unsqueeze(-1))
    w_c = A_c @ (k_c * (beta_c * torch.exp(g_c)).unsqueeze(-1))
    w = w_c.permute(0, 1, 3, 2, 4).reshape(B, T, H, K).to(k.dtype)
    u = u_c.permute(0, 1, 3, 2, 4).reshape(B, T, H, V).to(v.dtype)
    return w, u
'''


# ---------------------------------------------------------------------------
# Mutation-prompt body. Auto-generates the test/benchmark cases listing
# from the python literals so the prompt can never go out of sync with
# the cases the candidate is actually scored against.
# ---------------------------------------------------------------------------


_PROMPT_BODY = r'''You are an expert Triton engineer tasked with writing a fused per-chunk WY-transform forward kernel for Gated DeltaNet (GDN, arXiv:2412.06464, ICLR 2025).

This kernel produces the WY-transformed keys ``w`` and values ``u`` consumed by the chunkwise parallel forward pass of GDN. It is one of three per-chunk kernels in the GDN forward pipeline (alongside chunk-fwd-h and chunk-fwd-o).

The sequence is divided into non-overlapping chunks of ``BT=64`` timesteps. For each chunk independently:

```
u = A @ diag(beta) @ v             # WY-transformed values
w = A @ diag(beta * exp(g)) @ k    # WY-transformed keys
```

Equivalently:
```
u = A @ (v * beta[:, None])
w = A @ (k * beta[:, None] * exp(g)[:, None])
```

where ``A`` is a ``[BT, BT]`` WY representation matrix per chunk (precomputed and passed in).

Your task:
- Implement a single ``custom_kernel`` that returns both outputs.
- No autograd; the candidate is called as a plain function.

Input:
- ``data``: tuple ``(k, v, beta, A, g)`` where
    - ``k``:    ``torch.Tensor``, shape ``[B, T, H, K]``,  dtype ``float32``, on CUDA — keys
    - ``v``:    ``torch.Tensor``, shape ``[B, T, H, V]``,  dtype ``float32``, on CUDA — values
    - ``beta``: ``torch.Tensor``, shape ``[B, T, H]``,     dtype ``float32``, on CUDA — gating coefficients
    - ``A``:    ``torch.Tensor``, shape ``[B, T, H, 64]``, dtype ``float32``, on CUDA — per-chunk WY matrix (BT=64)
    - ``g``:    ``torch.Tensor``, shape ``[B, T, H]``,     dtype ``float32``, on CUDA — chunk-local cumulative gate

Output:
- Tuple ``(w, u)`` where
    - ``w``: ``torch.Tensor``, shape ``[B, T, H, K]``, dtype ``float32`` — WY-transformed keys
    - ``u``: ``torch.Tensor``, shape ``[B, T, H, V]``, dtype ``float32`` — WY-transformed values

**Problem Constraints:**
- ``T`` is always a multiple of 64. ``NT = T // 64``.
- ``B``, ``H``, ``K``, ``V`` vary across cases (see listing below).
- Tolerance: rtol=1e-3, atol=1e-3 against the fp32 reference.

**Remarks.** Every chunk is independent, so the kernel parallelizes naturally across ``(B, NT, H)``. The win comes from (1) fusing the two matmul chains into a single program per ``(b, nt, h)`` so ``A`` is loaded once and reused for both ``u`` and ``w``, (2) keeping ``A`` (a ``[64, 64]`` block) in SRAM across the two output computations, and (3) avoiding the eager reshape/permute round-trips through HBM.

Here is a PyTorch reference implementation. You should implement a kernel for the operations in ``ref_kernel``:

```python
import torch


CHUNK_SIZE = 64


def ref_kernel(data):
    k, v, beta, A, g = data
    B, T, H, K = k.shape
    V = v.shape[-1]
    C = A.shape[-1]
    NT = T // C
    k_c = k.float().reshape(B, NT, C, H, K).permute(0, 1, 3, 2, 4)
    v_c = v.float().reshape(B, NT, C, H, V).permute(0, 1, 3, 2, 4)
    beta_c = beta.float().reshape(B, NT, C, H).permute(0, 1, 3, 2)
    g_c = g.float().reshape(B, NT, C, H).permute(0, 1, 3, 2)
    A_c = A.float().reshape(B, NT, C, H, C).permute(0, 1, 3, 2, 4)
    u_c = A_c @ (v_c * beta_c.unsqueeze(-1))
    w_c = A_c @ (k_c * (beta_c * torch.exp(g_c)).unsqueeze(-1))
    w = w_c.permute(0, 1, 3, 2, 4).reshape(B, T, H, K).to(k.dtype)
    u = u_c.permute(0, 1, 3, 2, 4).reshape(B, T, H, V).to(v.dtype)
    return w, u
```

Here is some example skeleton code of the entrypoint function you will create:
```python
def custom_kernel(data):
    k, v, beta, A, g = data
    B, T, H, K = k.shape
    V = v.shape[-1]

    # ... your kernel here ...

    return w, u  # w: [B, T, H, K] float32, u: [B, T, H, V] float32
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
            f'  - {{"B": {c["B"]}, "T": {c["T"]}, "H": {c["H"]}, '
            f'"K": {c["K"]}, "V": {c["V"]}}}'
        )
    lines.append("")
    lines.append("Benchmark cases for runtime (optimize runtime for these):")
    for c in BENCHMARK_CASES:
        lines.append(
            f'  - {{"B": {c["B"]}, "T": {c["T"]}, "H": {c["H"]}, '
            f'"K": {c["K"]}, "V": {c["V"]}}}'
        )
    return "\n".join(lines) + "\n"


_KERNEL_DESCRIPTION_BODY: str = (
    _PROMPT_BODY.rstrip() + "\n" + _format_cases_block()
)


# ---------------------------------------------------------------------------
# Pack
# ---------------------------------------------------------------------------


GDN_RECOMPUTE_W_U_PACK: KernelPack[
    GdnRecomputeWUTestArgs, GdnRecomputeWUCaseSpeedup
] = KernelPack(
    name="gdn_recompute_w_u",
    modal_app_name="arid-badger-gdn-recompute-w-u",
    correctness_cases=CORRECTNESS_CASES,
    benchmark_cases=BENCHMARK_CASES,
    ref_kernel=ref_kernel,
    generate_input=generate_input,
    check_implementation=check_implementation,
    seed_kernel_code=_SEED_KERNEL_CODE,
    determinism_ctx=None,
    case_speedup_type=GdnRecomputeWUCaseSpeedup,
    kernel_description_body=_KERNEL_DESCRIPTION_BODY,
)


# ---------------------------------------------------------------------------
# Modal app + benchmarker cls (stable module path so cloudpickle is happy).
# ---------------------------------------------------------------------------


app = modal.App(GDN_RECOMPUTE_W_U_PACK.modal_app_name)


@app.cls(image=image, **DEFAULT_CLS_KWARGS)  # ty: ignore[unresolved-attribute]  # pyright: ignore[reportArgumentType]
class ModalGdnRecomputeWUBenchmarker:
    """Modal benchmarker for the GDN recompute-w-u kernel pack."""

    @modal.method()
    def evaluate_candidate(
        self,
        mutated_kernel_code: str,
        test_cases: list[dict[str, object]],
        max_repeats: int = 100,
        max_time_ns: float = 10e9,
    ) -> list[Any]:
        return run_evaluate_candidate(
            pack=GDN_RECOMPUTE_W_U_PACK,
            mutated_kernel_code=mutated_kernel_code,
            test_cases=test_cases,
            max_repeats=max_repeats,
            max_time_ns=max_time_ns,
        )


GDN_RECOMPUTE_W_U_RUNTIME: PackedModalRuntime[
    GdnRecomputeWUTestArgs, GdnRecomputeWUCaseSpeedup
] = PackedModalRuntime(
    pack=GDN_RECOMPUTE_W_U_PACK,
    app=app,
    benchmarker_cls=ModalGdnRecomputeWUBenchmarker,
)
