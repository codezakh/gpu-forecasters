"""Gated DeltaNet chunk-fwd-h KernelPack.

Vendored from
``gpu-mode/reference-kernels/problems/helion/gated_deltanet_chunk_fwd_h_py``.
This is the inter-chunk state recurrence that is the sequential
bottleneck in the chunkwise parallel forward pass of Gated DeltaNet
(arXiv:2412.06464, ICLR 2025). Per gh070 v1, this is the recommended
third pack — a TriMul-shaped problem with no cuDNN autotuner involvement,
so timing is stable across Modal containers.

Unlike upstream's submission.py (a Helion kernel), the seed here is a
pure-PyTorch chunked recurrence — the same shape as the reference,
relying on torch.compile-style fusion as the LLM's starting point. The
modal image does not ship Helion, and matching upstream's helion seed
would be a baseline we couldn't reproduce on the same A100 anyway.

Test/benchmark cases lifted verbatim from upstream task.yml.
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
# Test args + cases — vendored from upstream task.yml
# ---------------------------------------------------------------------------

CHUNK_SIZE: int = 64


class GdnChunkFwdHTestArgs(TypedDict):
    B: int
    T: int
    H: int
    K: int
    V: int
    seed: int


CORRECTNESS_CASES: list[GdnChunkFwdHTestArgs] = [
    {"B": 1, "T": 64, "H": 2, "K": 64, "V": 64, "seed": 4242},
    {"B": 2, "T": 128, "H": 4, "K": 64, "V": 64, "seed": 5236},
    {"B": 1, "T": 256, "H": 4, "K": 64, "V": 128, "seed": 1001},
]

BENCHMARK_CASES: list[GdnChunkFwdHTestArgs] = [
    {"B": 1, "T": 64, "H": 1, "K": 64, "V": 64, "seed": 31232},
    {"B": 2, "T": 512, "H": 3, "K": 64, "V": 64, "seed": 4052},
    {"B": 2, "T": 1024, "H": 3, "K": 64, "V": 64, "seed": 2146},
]


# ---------------------------------------------------------------------------
# CaseSpeedup with shape fields
# ---------------------------------------------------------------------------


class GdnChunkFwdHCaseSpeedup(CaseSpeedupBase):
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
    ) -> "GdnChunkFwdHCaseSpeedup":
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
# Reference + input generator + correctness oracle
# Vendored from upstream reference.py. The eager helpers
# (_chunk_local_cumsum_eager etc.) are pre-computation for input
# generation; ref_kernel itself is the inner sequential loop the LLM
# is asked to optimize.
# ---------------------------------------------------------------------------

_RTOL: float = 1e-3
_ATOL: float = 1e-3

GdnChunkFwdHData = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
GdnChunkFwdHOutput = Tuple[torch.Tensor, torch.Tensor]


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


def generate_input(
    B: int, T: int, H: int, K: int, V: int, seed: int
) -> GdnChunkFwdHData:
    """``(k, w, u, g)`` deterministic under ``seed``.

    The WY transform (k, v, beta, g) → (w, u) and the chunk-local
    cumsum on g are pre-computed here so the candidate sees the same
    inputs the upstream chunk-fwd-h kernel sees in production.
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
    w, u = _recompute_w_u_fwd_eager(k=k, v=v, beta=beta, A=A_solve, g=g_cumsum)
    return k.contiguous(), w.contiguous(), u.contiguous(), g_cumsum.contiguous()


def ref_kernel(data: GdnChunkFwdHData) -> GdnChunkFwdHOutput:
    """Sequential chunked recurrence — the reference the candidate
    must match numerically (and beat in wall-clock).

    Returns ``(h_all, v_new_out)`` where:
        h_all       : [B, NT, H, K, V] (k.dtype) — per-chunk hidden states
        v_new_out   : [B, T, H, V]     (u.dtype) — corrected values
    """
    k, w, u, g = data
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


def check_implementation(
    data: GdnChunkFwdHData, output: Any
) -> Tuple[bool, str]:
    """Validate the candidate's ``(h, v_new)`` output.

    Mirrors upstream reference.py's check_implementation: per-output
    shape + dtype + verbose-allclose. Returns the worst-offender error
    string when both sub-checks fail so the LLM gets the most-actionable
    feedback.
    """
    if not isinstance(output, tuple) or len(output) != 2:
        return (
            False,
            f"custom_kernel must return a 2-tuple (h, v_new); got "
            f"{type(output).__name__} of length "
            f"{len(output) if hasattr(output, '__len__') else 'n/a'}",
        )
    sub_h, sub_v = output
    expected_h, expected_v = ref_kernel(data)

    if sub_h.shape != expected_h.shape:
        return (
            False,
            f"h shape mismatch: expected {tuple(expected_h.shape)}, "
            f"got {tuple(sub_h.shape)}",
        )
    if sub_v.shape != expected_v.shape:
        return (
            False,
            f"v_new shape mismatch: expected {tuple(expected_v.shape)}, "
            f"got {tuple(sub_v.shape)}",
        )

    h_close = torch.allclose(
        sub_h.float(), expected_h.float(), atol=_ATOL, rtol=_RTOL
    )
    v_close = torch.allclose(
        sub_v.float(), expected_v.float(), atol=_ATOL, rtol=_RTOL
    )
    if h_close and v_close:
        return True, ""

    h_err = (sub_h.float() - expected_h.float()).abs().max().item()
    v_err = (sub_v.float() - expected_v.float()).abs().max().item()
    return (
        False,
        f"h max err={h_err:.3e} {'OK' if h_close else 'FAIL'}; "
        f"v_new max err={v_err:.3e} {'OK' if v_close else 'FAIL'} "
        f"(rtol={_RTOL}, atol={_ATOL})",
    )


# ---------------------------------------------------------------------------
# Seed kernel — pure-PyTorch chunked recurrence baseline.
# Same shape as ref_kernel; the LLM is asked to fuse the chunk loop
# into a Triton kernel that parallelizes across (B, H).
# ---------------------------------------------------------------------------

_SEED_KERNEL_CODE: str = '''import torch


CHUNK_SIZE = 64


def custom_kernel(data):
    """Pure-PyTorch chunked recurrence — sequential over chunks,
    parallel over (B, H) via tensor ops.

    Args:
        data: tuple ``(k, w, u, g)`` where
            - k: [B, T, H, K] float32 on CUDA
            - w: [B, T, H, K] float32 on CUDA
            - u: [B, T, H, V] float32 on CUDA
            - g: [B, T, H]    float32 on CUDA (chunk-local cumsum'd)

    Returns:
        Tuple ``(h_all, v_new_out)`` where
            - h_all     : [B, NT, H, K, V] (k.dtype)
            - v_new_out : [B, T, H, V]     (u.dtype)
    """
    k, w, u, g = data
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
'''


# ---------------------------------------------------------------------------
# Mutation-prompt body. Auto-generates the test/benchmark cases listing
# from the python literals so the prompt can never go out of sync with
# the cases the candidate is actually scored against.
# ---------------------------------------------------------------------------


_PROMPT_BODY = r'''You are an expert Triton engineer tasked with writing a fused chunkwise forward kernel for Gated DeltaNet (GDN, arXiv:2412.06464, ICLR 2025).

This kernel maintains a hidden state ``h`` of shape ``[K, V]`` across chunks and emits per-chunk state plus corrected values for each chunk. It is the sequential bottleneck in the chunkwise parallel forward pass of GDN.

The sequence is divided into chunks of ``BT=64`` timesteps. Processing is sequential across chunks but parallel across ``(B, H)`` and within each chunk:

For each ``(b, h)`` pair, starting with ``h_state = zeros(K, V)``:
  For each chunk ``c = 0, 1, ..., NT-1``:
    1. Store: ``h_out[b, c, h] = h_state``
    2. Compute: ``v_new = u - w @ h_state``
    3. Gate: ``v_gated[t] = v_new[t] * exp(g[last_t] - g[t])``
    4. Decay: ``h_state = h_state * exp(g[last_t])``
    5. Update: ``h_state = h_state + k^T @ v_gated``

Your task:
- Implement a single ``custom_kernel`` that returns both outputs.
- No autograd; the candidate is called as a plain function.

Input:
- ``data``: tuple ``(k, w, u, g)`` where
    - ``k``: ``torch.Tensor``, shape ``[B, T, H, K]``, dtype ``float32``, on CUDA — keys
    - ``w``: ``torch.Tensor``, shape ``[B, T, H, K]``, dtype ``float32``, on CUDA — WY-transformed keys
    - ``u``: ``torch.Tensor``, shape ``[B, T, H, V]``, dtype ``float32``, on CUDA — WY-transformed values
    - ``g``: ``torch.Tensor``, shape ``[B, T, H]``,    dtype ``float32``, on CUDA — chunk-local cumulative gate

Output:
- Tuple ``(h_all, v_new)`` where
    - ``h_all``: ``torch.Tensor``, shape ``[B, NT, H, K, V]``, dtype ``float32`` — per-chunk hidden states
    - ``v_new``: ``torch.Tensor``, shape ``[B, T, H, V]``,     dtype ``float32`` — corrected values

**Problem Constraints:**
- ``T`` is always a multiple of 64. ``NT = T // 64``.
- ``B``, ``H``, ``K``, ``V`` vary across cases (see listing below).
- Tolerance: rtol=1e-3, atol=1e-3 against the fp32 reference.

**Remarks.** The win comes from (1) keeping ``h_state`` in registers/SRAM across the chunk loop so it is never round-tripped through HBM, (2) parallelizing across ``(B, H)`` because the chunk loop is independent across them, and (3) fusing the ``v_new``, gate, and update steps into a single kernel program per ``(b, h)``. The chunk loop itself is intrinsically sequential — a single program instance per ``(b, h)`` walks the chunks.

Here is a PyTorch reference implementation. You should implement a kernel for the operations in ``ref_kernel``:

```python
import torch


CHUNK_SIZE = 64


def ref_kernel(data):
    k, w, u, g = data
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
        h = h * torch.exp(g_last).unsqueeze(-1).unsqueeze(-1) + k_c[:, c].transpose(-1, -2) @ v_gated
    v_new_out = v_new_c.permute(0, 1, 3, 2, 4).reshape(B, T, H, V).to(u.dtype)
    return h_all.to(k.dtype), v_new_out
```

Here is some example skeleton code of the entrypoint function you will create:
```python
def custom_kernel(data):
    k, w, u, g = data
    B, T, H, K = k.shape
    V = u.shape[-1]

    # ... your kernel here ...

    return h_all, v_new  # h_all: [B, NT, H, K, V] float32, v_new: [B, T, H, V] float32
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


GDN_CHUNK_FWD_H_PACK: KernelPack[
    GdnChunkFwdHTestArgs, GdnChunkFwdHCaseSpeedup
] = KernelPack(
    name="gdn_chunk_fwd_h",
    modal_app_name="arid-badger-gdn-chunk-fwd-h",
    correctness_cases=CORRECTNESS_CASES,
    benchmark_cases=BENCHMARK_CASES,
    ref_kernel=ref_kernel,
    generate_input=generate_input,
    check_implementation=check_implementation,
    seed_kernel_code=_SEED_KERNEL_CODE,
    determinism_ctx=None,
    case_speedup_type=GdnChunkFwdHCaseSpeedup,
    kernel_description_body=_KERNEL_DESCRIPTION_BODY,
)


# ---------------------------------------------------------------------------
# Modal app + benchmarker cls (stable module path so cloudpickle is happy).
# ---------------------------------------------------------------------------


app = modal.App(GDN_CHUNK_FWD_H_PACK.modal_app_name)


@app.cls(image=image, **DEFAULT_CLS_KWARGS)  # ty: ignore[unresolved-attribute]  # pyright: ignore[reportArgumentType]
class ModalGdnChunkFwdHBenchmarker:
    """Modal benchmarker for the GDN chunk-fwd-h kernel pack."""

    @modal.method()
    def evaluate_candidate(
        self,
        mutated_kernel_code: str,
        test_cases: list[dict[str, object]],
        max_repeats: int = 100,
        max_time_ns: float = 10e9,
    ) -> list[Any]:
        return run_evaluate_candidate(
            pack=GDN_CHUNK_FWD_H_PACK,
            mutated_kernel_code=mutated_kernel_code,
            test_cases=test_cases,
            max_repeats=max_repeats,
            max_time_ns=max_time_ns,
        )


GDN_CHUNK_FWD_H_RUNTIME: PackedModalRuntime[
    GdnChunkFwdHTestArgs, GdnChunkFwdHCaseSpeedup
] = PackedModalRuntime(
    pack=GDN_CHUNK_FWD_H_PACK,
    app=app,
    benchmarker_cls=ModalGdnChunkFwdHBenchmarker,
)
