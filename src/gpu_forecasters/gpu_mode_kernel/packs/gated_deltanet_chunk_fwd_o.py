"""Gated DeltaNet chunk-fwd-o KernelPack.

Vendored from
``gpu-mode/reference-kernels/problems/helion/gated_deltanet_chunk_fwd_o_py``.
This is the output computation that combines inter-chunk (state-based)
and intra-chunk (causal-attention) contributions in the chunkwise
parallel forward pass of Gated DeltaNet (arXiv:2412.06464, ICLR 2025).

Pairs with ``GDN_CHUNK_FWD_H_PACK`` for an in-architecture multi-kernel
GDN story per gh070 v1: chunk-fwd-h is the sequential bottleneck
producing the per-chunk states, chunk-fwd-o consumes those states to
produce the final output. Same B/T/H/K/V test/benchmark shapes as
chunk-fwd-h.

Unlike GDN h, every chunk here is independent — no Python loop in the
reference. The seed kernel is the reference verbatim; the LLM's job
is to fuse the matmul + gating + causal mask into a single Triton
kernel that parallelizes across (B, NT, H).

Tolerance: rtol=1e-3, atol=1e-3 — upstream values, looser than TriMul
because both the reference and the candidate live in fp32.
"""

from __future__ import annotations

from typing import Any, Mapping, Tuple, TypedDict

import modal
import torch
from pydantic import ConfigDict

from gpu_forecasters.gpu_mode_kernel.core import CaseSpeedupBase, KernelExecResult
from gpu_forecasters.gpu_mode_kernel.kernel_pack import KernelPack
from gpu_forecasters.gpu_mode_kernel.modal_scoring import (
    DEFAULT_CLS_KWARGS,
    PackedModalRuntime,
    run_evaluate_candidate,
)
from gpu_forecasters.kernelbench.modal_image import image


# ---------------------------------------------------------------------------
# Test args + cases — vendored from upstream task.yml. Match GDN h
# verbatim so cross-pack comparisons stay clean.
# ---------------------------------------------------------------------------

CHUNK_SIZE: int = 64


class GdnChunkFwdOTestArgs(TypedDict):
    B: int
    T: int
    H: int
    K: int
    V: int
    seed: int


CORRECTNESS_CASES: list[GdnChunkFwdOTestArgs] = [
    {"B": 1, "T": 64, "H": 2, "K": 64, "V": 64, "seed": 4242},
    {"B": 2, "T": 128, "H": 4, "K": 64, "V": 64, "seed": 5236},
    {"B": 1, "T": 256, "H": 4, "K": 64, "V": 128, "seed": 1001},
]

BENCHMARK_CASES: list[GdnChunkFwdOTestArgs] = [
    {"B": 1, "T": 64, "H": 1, "K": 64, "V": 64, "seed": 31232},
    {"B": 2, "T": 512, "H": 3, "K": 64, "V": 64, "seed": 4052},
    {"B": 2, "T": 1024, "H": 3, "K": 64, "V": 64, "seed": 2146},
]


# ---------------------------------------------------------------------------
# CaseSpeedup with shape fields (identical schema to GDN h)
# ---------------------------------------------------------------------------


class GdnChunkFwdOCaseSpeedup(CaseSpeedupBase):
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
    ) -> "GdnChunkFwdOCaseSpeedup":
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
#
# generate_input must produce a valid (q, k, v_new, h, g) tuple — that
# means running the upstream WY transform + chunk-fwd-h pipeline as a
# preamble. Vendoring the full chain here keeps the pack self-contained
# (no cross-pack import dependency on GDN h).
# ---------------------------------------------------------------------------

_RTOL: float = 1e-3
_ATOL: float = 1e-3

GdnChunkFwdOData = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]
GdnChunkFwdOOutput = torch.Tensor


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


def _recompute_w_u_fwd_eager(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
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


def _chunk_fwd_h_eager(
    k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K = k.shape
    V = u.shape[-1]
    C = CHUNK_SIZE
    NT = T // C
    k_c = k.float().reshape(B, NT, C, H, K).permute(0, 1, 3, 2, 4)
    w_c = w.float().reshape(B, NT, C, H, K).permute(0, 1, 3, 2, 4)
    u_c = u.float().reshape(B, NT, C, H, V).permute(0, 1, 3, 2, 4)
    g_c = g.float().reshape(B, NT, C, H).permute(0, 1, 3, 2)
    h_all = torch.zeros(B, NT, H, K, V, dtype=torch.float32, device=k.device)
    v_new_c = torch.zeros_like(u_c)
    h = torch.zeros(B, H, K, V, dtype=torch.float32, device=k.device)
    for c in range(NT):
        h_all[:, c] = h
        v_new_c[:, c] = u_c[:, c] - w_c[:, c] @ h
        g_last = g_c[:, c, :, -1]
        gate = torch.exp(g_last.unsqueeze(-1) - g_c[:, c])
        v_gated = v_new_c[:, c] * gate.unsqueeze(-1)
        h = h * torch.exp(g_last).unsqueeze(-1).unsqueeze(-1) + k_c[
            :, c
        ].transpose(-1, -2) @ v_gated
    v_new_out = v_new_c.permute(0, 1, 3, 2, 4).reshape(B, T, H, V).to(u.dtype)
    return h_all.to(k.dtype), v_new_out


def generate_input(
    B: int, T: int, H: int, K: int, V: int, seed: int
) -> GdnChunkFwdOData:
    """``(q, k, v_new, h, g)`` deterministic under ``seed``.

    Runs the full WY transform + chunk-fwd-h preamble, then returns
    the inputs the chunk-fwd-o kernel actually consumes. ``q`` is
    sampled fresh; ``k``, ``g_cumsum`` round-trip from the preamble.
    """
    torch.manual_seed(seed)
    device = "cuda"
    q = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
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
    w, u = _recompute_w_u_fwd_eager(k=k, v=v, beta=beta, A=A_solve, g=g_cumsum)
    h, v_new = _chunk_fwd_h_eager(k=k, w=w, u=u, g=g_cumsum)
    return (
        q.contiguous(),
        k.contiguous(),
        v_new.contiguous(),
        h.contiguous(),
        g_cumsum.contiguous(),
    )


def ref_kernel(data: GdnChunkFwdOData) -> GdnChunkFwdOOutput:
    """Per-chunk-independent computation of the GDN output.

    Combines inter-chunk (q @ h, state-based) and intra-chunk
    (causal q @ k^T attention) contributions, then scales by K^-0.5.
    Returns ``[B, T, H, V]`` (q.dtype).
    """
    q, k, v_new, h, g = data
    B, T, H, K = q.shape
    V = v_new.shape[-1]
    C = CHUNK_SIZE
    NT = T // C
    scale = K**-0.5
    q_c = q.float().reshape(B, NT, C, H, K).permute(0, 1, 3, 2, 4)
    k_c = k.float().reshape(B, NT, C, H, K).permute(0, 1, 3, 2, 4)
    v_c = v_new.float().reshape(B, NT, C, H, V).permute(0, 1, 3, 2, 4)
    g_c = g.float().reshape(B, NT, C, H).permute(0, 1, 3, 2)
    o_inter = (q_c @ h.float()) * torch.exp(g_c).unsqueeze(-1)
    causal = torch.tril(torch.ones(C, C, dtype=torch.bool, device=q.device))
    g_diff = g_c.unsqueeze(-1) - g_c.unsqueeze(-2)
    g_diff = torch.where(causal, g_diff, torch.zeros_like(g_diff))
    qk = q_c @ k_c.transpose(-1, -2) * torch.exp(g_diff) * causal
    o = (o_inter + qk @ v_c) * scale
    return o.permute(0, 1, 3, 2, 4).reshape(B, T, H, V).to(q.dtype)


def check_implementation(
    data: GdnChunkFwdOData, output: Any
) -> Tuple[bool, str]:
    """Validate the candidate's ``[B, T, H, V]`` output (single tensor)."""
    expected = ref_kernel(data)

    if not isinstance(output, torch.Tensor):
        return (
            False,
            f"custom_kernel must return a torch.Tensor; got "
            f"{type(output).__name__}",
        )
    if output.shape != expected.shape:
        return (
            False,
            f"output shape mismatch: expected {tuple(expected.shape)}, "
            f"got {tuple(output.shape)}",
        )

    is_close = torch.allclose(
        output.float(), expected.float(), atol=_ATOL, rtol=_RTOL
    )
    if is_close:
        return True, ""

    err = (output.float() - expected.float()).abs().max().item()
    return (
        False,
        f"output max err={err:.3e} FAIL (rtol={_RTOL}, atol={_ATOL})",
    )


# ---------------------------------------------------------------------------
# Seed kernel — pure-PyTorch reference verbatim. Each chunk is
# independent; the LLM is asked to fuse the per-chunk matmul +
# causal-mask + gating into a single Triton kernel and parallelize
# across (B, NT, H).
# ---------------------------------------------------------------------------

_SEED_KERNEL_CODE: str = '''import torch


CHUNK_SIZE = 64


def custom_kernel(data):
    """Per-chunk-independent GDN output computation in pure PyTorch.

    Args:
        data: tuple ``(q, k, v_new, h, g)`` where
            - q     : [B, T, H, K]      float32 on CUDA
            - k     : [B, T, H, K]      float32 on CUDA
            - v_new : [B, T, H, V]      float32 on CUDA
            - h     : [B, NT, H, K, V]  float32 on CUDA
            - g     : [B, T, H]         float32 on CUDA (chunk-local cumsum'd)

    Returns:
        torch.Tensor of shape ``[B, T, H, V]`` (q.dtype).
    """
    q, k, v_new, h, g = data
    B, T, H, K = q.shape
    V = v_new.shape[-1]
    C = CHUNK_SIZE
    NT = T // C
    scale = K**-0.5
    q_c = q.float().reshape(B, NT, C, H, K).permute(0, 1, 3, 2, 4)
    k_c = k.float().reshape(B, NT, C, H, K).permute(0, 1, 3, 2, 4)
    v_c = v_new.float().reshape(B, NT, C, H, V).permute(0, 1, 3, 2, 4)
    g_c = g.float().reshape(B, NT, C, H).permute(0, 1, 3, 2)
    o_inter = (q_c @ h.float()) * torch.exp(g_c).unsqueeze(-1)
    causal = torch.tril(torch.ones(C, C, dtype=torch.bool, device=q.device))
    g_diff = g_c.unsqueeze(-1) - g_c.unsqueeze(-2)
    g_diff = torch.where(causal, g_diff, torch.zeros_like(g_diff))
    qk = q_c @ k_c.transpose(-1, -2) * torch.exp(g_diff) * causal
    o = (o_inter + qk @ v_c) * scale
    return o.permute(0, 1, 3, 2, 4).reshape(B, T, H, V).to(q.dtype)
'''


# ---------------------------------------------------------------------------
# Mutation-prompt body. Auto-generates the test/benchmark cases
# listing so the LLM-facing case list cannot drift from the literals.
# ---------------------------------------------------------------------------


_PROMPT_BODY = r'''You are an expert Triton engineer tasked with writing a fused chunkwise output kernel for Gated DeltaNet (GDN, arXiv:2412.06464, ICLR 2025).

This kernel computes the final per-token output ``o`` by combining inter-chunk (state-based) and intra-chunk (causal-attention) contributions. Each chunk of ``BT=64`` timesteps is independent; the chunks can be processed fully in parallel (unlike the chunk-fwd-h kernel, which is sequential across chunks).

For each chunk ``c``, with ``scale = K^(-0.5)``:

  inter[c] = (q[c] @ h[c]) * exp(g[c])
  intra[c] = causal_mask(q[c] @ k[c].T * exp(g[c, :, None] - g[c, None, :])) @ v_new[c]
  o[c]     = (inter[c] + intra[c]) * scale

where ``causal_mask`` zeros out entries where row-index < col-index within the chunk.

Your task:
- Implement a single ``custom_kernel`` that returns the output tensor.
- No autograd; the candidate is called as a plain function.

Input:
- ``data``: tuple ``(q, k, v_new, h, g)`` where
    - ``q``     : ``torch.Tensor``, shape ``[B, T, H, K]``,     dtype ``float32``, on CUDA — queries
    - ``k``     : ``torch.Tensor``, shape ``[B, T, H, K]``,     dtype ``float32``, on CUDA — keys
    - ``v_new`` : ``torch.Tensor``, shape ``[B, T, H, V]``,     dtype ``float32``, on CUDA — corrected values (from chunk-fwd-h)
    - ``h``     : ``torch.Tensor``, shape ``[B, NT, H, K, V]``, dtype ``float32``, on CUDA — per-chunk hidden states (from chunk-fwd-h)
    - ``g``     : ``torch.Tensor``, shape ``[B, T, H]``,        dtype ``float32``, on CUDA — chunk-local cumulative gate

Output:
- ``torch.Tensor``, shape ``[B, T, H, V]``, dtype ``float32`` — the per-token output.

**Problem Constraints:**
- ``T`` is always a multiple of 64. ``NT = T // 64``.
- ``B``, ``H``, ``K``, ``V`` vary across cases (see listing below).
- ``scale = K^(-0.5)``.
- Tolerance: rtol=1e-3, atol=1e-3 against the fp32 reference.

**Remarks.** This kernel is essentially a fused (q @ h) + small causal-attention block per chunk. The win comes from (1) parallelizing across ``(B, NT, H)`` because the chunk loop is fully independent, (2) keeping the chunk-local matmul accumulators in registers/SRAM rather than HBM round-tripping, and (3) fusing the causal mask, gating, and scale into the same kernel that emits ``o``. The intra-chunk attention is small enough (``C=64``) that the per-chunk Q@K^T can live in a single program's tile.

Here is a PyTorch reference implementation. You should implement a kernel for the operations in ``ref_kernel``:

```python
import torch


CHUNK_SIZE = 64


def ref_kernel(data):
    q, k, v_new, h, g = data
    B, T, H, K = q.shape
    V = v_new.shape[-1]
    C = CHUNK_SIZE
    NT = T // C
    scale = K**-0.5
    q_c = q.float().reshape(B, NT, C, H, K).permute(0, 1, 3, 2, 4)
    k_c = k.float().reshape(B, NT, C, H, K).permute(0, 1, 3, 2, 4)
    v_c = v_new.float().reshape(B, NT, C, H, V).permute(0, 1, 3, 2, 4)
    g_c = g.float().reshape(B, NT, C, H).permute(0, 1, 3, 2)
    o_inter = (q_c @ h.float()) * torch.exp(g_c).unsqueeze(-1)
    causal = torch.tril(torch.ones(C, C, dtype=torch.bool, device=q.device))
    g_diff = g_c.unsqueeze(-1) - g_c.unsqueeze(-2)
    g_diff = torch.where(causal, g_diff, torch.zeros_like(g_diff))
    qk = q_c @ k_c.transpose(-1, -2) * torch.exp(g_diff) * causal
    o = (o_inter + qk @ v_c) * scale
    return o.permute(0, 1, 3, 2, 4).reshape(B, T, H, V).to(q.dtype)
```

Here is some example skeleton code of the entrypoint function you will create:
```python
def custom_kernel(data):
    q, k, v_new, h, g = data
    B, T, H, K = q.shape
    V = v_new.shape[-1]

    # ... your kernel here ...

    return o  # [B, T, H, V] float32
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


GDN_CHUNK_FWD_O_PACK: KernelPack[
    GdnChunkFwdOTestArgs, GdnChunkFwdOCaseSpeedup
] = KernelPack(
    name="gdn_chunk_fwd_o",
    modal_app_name="arid-badger-gdn-chunk-fwd-o",
    correctness_cases=CORRECTNESS_CASES,
    benchmark_cases=BENCHMARK_CASES,
    ref_kernel=ref_kernel,
    generate_input=generate_input,
    check_implementation=check_implementation,
    seed_kernel_code=_SEED_KERNEL_CODE,
    determinism_ctx=None,
    case_speedup_type=GdnChunkFwdOCaseSpeedup,
    kernel_description_body=_KERNEL_DESCRIPTION_BODY,
)


# ---------------------------------------------------------------------------
# Modal app + benchmarker cls (stable module path so cloudpickle is happy).
# ---------------------------------------------------------------------------


app = modal.App(GDN_CHUNK_FWD_O_PACK.modal_app_name)


@app.cls(image=image, **DEFAULT_CLS_KWARGS)  # ty: ignore[unresolved-attribute]  # pyright: ignore[reportArgumentType]
class ModalGdnChunkFwdOBenchmarker:
    """Modal benchmarker for the GDN chunk-fwd-o kernel pack."""

    @modal.method()
    def evaluate_candidate(
        self,
        mutated_kernel_code: str,
        test_cases: list[dict[str, object]],
        max_repeats: int = 100,
        max_time_ns: float = 10e9,
    ) -> list[Any]:
        return run_evaluate_candidate(
            pack=GDN_CHUNK_FWD_O_PACK,
            mutated_kernel_code=mutated_kernel_code,
            test_cases=test_cases,
            max_repeats=max_repeats,
            max_time_ns=max_time_ns,
        )


GDN_CHUNK_FWD_O_RUNTIME: PackedModalRuntime[
    GdnChunkFwdOTestArgs, GdnChunkFwdOCaseSpeedup
] = PackedModalRuntime(
    pack=GDN_CHUNK_FWD_O_PACK,
    app=app,
    benchmarker_cls=ModalGdnChunkFwdOBenchmarker,
)
