"""TriMul KernelPack.

Path-B port of ``arid_badger.trimul/`` into the generic
``gpu_mode_kernel`` abstraction. The vendored cases / reference /
generate_input / check_implementation / determinism context manager
/ seed kernel / mutation prompt are duplicated here (rather than
re-imported from ``arid_badger.trimul``) so the regression check in
``tests/gpu_mode_kernel/test_trimul_pack_vs_legacy.py`` compares two
genuinely independent paths.

The intent is that once Path B is validated (this pack's scoring
output matches ``arid_badger.trimul.scoring.score``'s output on the
same Modal A100 to within determinism tolerance) the legacy
``arid_badger.trimul/`` package can be retired.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Mapping, Tuple, TypedDict

import modal
import torch
from pydantic import ConfigDict
from torch import einsum, nn

from arid_badger.gpu_mode_kernel.comparison import make_match_reference
from arid_badger.gpu_mode_kernel.core import CaseSpeedupBase, KernelExecResult
from arid_badger.gpu_mode_kernel.kernel_pack import KernelPack
from arid_badger.gpu_mode_kernel.modal_scoring import (
    DEFAULT_CLS_KWARGS,
    PackedModalRuntime,
    run_evaluate_candidate,
)
from arid_badger.kernelbench.modal_image import image


# ---------------------------------------------------------------------------
# Determinism context manager — vendored from
# ``arid_badger.trimul.comparison.DisableCuDNNTF32``. Re-exported through
# the candidate-loading utils.py shim below.
# ---------------------------------------------------------------------------


class DisableCuDNNTF32:
    """Disable TF32 + force deterministic cuDNN.

    Wraps the reference pass so that correctness checks are stable
    across runs and GPU architectures.
    """

    def __init__(self) -> None:
        self.allow_tf32 = torch.backends.cudnn.allow_tf32
        self.deterministic = torch.backends.cudnn.deterministic

    def __enter__(self) -> "DisableCuDNNTF32":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        torch.backends.cudnn.allow_tf32 = self.allow_tf32
        torch.backends.cudnn.deterministic = self.deterministic


# ---------------------------------------------------------------------------
# Test args + cases — vendored verbatim from
# ``ttt-discover/examples/gpu_mode/lib/bioml/trimul/task.yml``.
# ---------------------------------------------------------------------------


class TriMulTestArgs(TypedDict):
    seqlen: int
    bs: int
    dim: int
    hiddendim: int
    seed: int
    nomask: bool
    distribution: str


CORRECTNESS_CASES: list[TriMulTestArgs] = [
    {"seqlen": 32, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 9371, "nomask": True, "distribution": "normal"},
    {"seqlen": 32, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 1092, "nomask": False, "distribution": "normal"},
    {"seqlen": 64, "bs": 2, "dim": 256, "hiddendim": 128, "seed": 2291, "nomask": True, "distribution": "normal"},
    {"seqlen": 64, "bs": 2, "dim": 256, "hiddendim": 128, "seed": 210284, "nomask": False, "distribution": "normal"},
    {"seqlen": 128, "bs": 1, "dim": 768, "hiddendim": 128, "seed": 81934, "nomask": True, "distribution": "normal"},
    {"seqlen": 256, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 1932, "nomask": True, "distribution": "normal"},
    {"seqlen": 256, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 10432, "nomask": False, "distribution": "normal"},
    {"seqlen": 768, "bs": 2, "dim": 128, "hiddendim": 128, "seed": 731, "nomask": True, "distribution": "normal"},
    {"seqlen": 1024, "bs": 1, "dim": 384, "hiddendim": 128, "seed": 53121, "nomask": False, "distribution": "normal"},
    {"seqlen": 1024, "bs": 1, "dim": 768, "hiddendim": 128, "seed": 31, "nomask": True, "distribution": "normal"},
    {"seqlen": 1024, "bs": 1, "dim": 768, "hiddendim": 128, "seed": 4921, "nomask": False, "distribution": "normal"},
    {"seqlen": 32, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 937321, "nomask": True, "distribution": "cauchy"},
    {"seqlen": 64, "bs": 2, "dim": 256, "hiddendim": 128, "seed": 2291, "nomask": True, "distribution": "cauchy"},
    {"seqlen": 128, "bs": 1, "dim": 768, "hiddendim": 128, "seed": 8134, "nomask": True, "distribution": "cauchy"},
    {"seqlen": 256, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 932, "nomask": True, "distribution": "cauchy"},
    {"seqlen": 768, "bs": 2, "dim": 128, "hiddendim": 128, "seed": 31, "nomask": True, "distribution": "cauchy"},
    {"seqlen": 1024, "bs": 1, "dim": 384, "hiddendim": 128, "seed": 5321, "nomask": False, "distribution": "cauchy"},
    {"seqlen": 1024, "bs": 1, "dim": 768, "hiddendim": 128, "seed": 491, "nomask": False, "distribution": "cauchy"},
]


BENCHMARK_CASES: list[TriMulTestArgs] = [
    {"seqlen": 256, "bs": 2, "dim": 128, "hiddendim": 128, "seed": 9371, "nomask": True, "distribution": "normal"},
    {"seqlen": 768, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 381, "nomask": True, "distribution": "cauchy"},
    {"seqlen": 256, "bs": 2, "dim": 384, "hiddendim": 128, "seed": 2301, "nomask": False, "distribution": "normal"},
    {"seqlen": 512, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 12819, "nomask": True, "distribution": "normal"},
    {"seqlen": 1024, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 381, "nomask": True, "distribution": "cauchy"},
    {"seqlen": 768, "bs": 1, "dim": 384, "hiddendim": 128, "seed": 481, "nomask": False, "distribution": "normal"},
    {"seqlen": 1024, "bs": 1, "dim": 384, "hiddendim": 128, "seed": 23291, "nomask": True, "distribution": "normal"},
]


class TriMulCaseSpeedup(CaseSpeedupBase):
    model_config = ConfigDict(frozen=True)

    seqlen: int
    bs: int
    dim: int
    hiddendim: int
    nomask: bool
    distribution: str

    @classmethod
    def from_exec_result(
        cls,
        test_args: Mapping[str, Any],
        exec_result: KernelExecResult,
    ) -> "TriMulCaseSpeedup":
        speedup = (
            exec_result.ref_runtime_ns / exec_result.runtime_ns
            if exec_result.runtime_ns > 0
            else 0.0
        )
        return cls(
            seqlen=test_args["seqlen"],
            bs=test_args["bs"],
            dim=test_args["dim"],
            hiddendim=test_args["hiddendim"],
            nomask=test_args["nomask"],
            distribution=test_args["distribution"],
            speedup=speedup,
            runtime_ns=exec_result.runtime_ns,
            ref_runtime_ns=exec_result.ref_runtime_ns,
        )

    def format_for_prompt(self) -> str:
        # Format must match the legacy
        # ``format_trimul_feedback_mutation_prompt`` per-case line
        # exactly — both packs run against the same prompt-engineering
        # baseline, and the regression check would not catch wire
        # drift in the prompt text.
        ref_us = self.ref_runtime_ns / 1_000.0
        candidate_us = self.runtime_ns / 1_000.0
        return (
            f"seqlen={self.seqlen}, bs={self.bs}, dim={self.dim}, "
            f"hiddendim={self.hiddendim}, nomask={self.nomask}, "
            f"dist={self.distribution}: "
            f"{self.speedup:.3f}x "
            f"(ref: {ref_us:.1f}μs, candidate: {candidate_us:.1f}μs)"
        )


# ---------------------------------------------------------------------------
# Reference + input generator + correctness oracle — vendored from
# ``arid_badger.trimul.reference``. The TriMul module body lives here
# rather than being imported from torch hub so that the regression
# check has a self-contained reference path.
# ---------------------------------------------------------------------------

input_t = Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]
output_t = torch.Tensor


class TriMul(nn.Module):
    """Triangle Multiplicative Update (outgoing variant).

    Vendored from ``ttt-discover/examples/gpu_mode/lib/bioml/trimul/reference.py``.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        device: "torch.device | str" = "cuda",
        dtype: str = "",
    ) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, device=device)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, device=device)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, device=device)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, device=device)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, device=device)

        self.to_out_norm = nn.LayerNorm(hidden_dim, device=device)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, device=device)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: [bs, seq_len, seq_len, dim], mask: [bs, seq_len, seq_len]."""
        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum("... i k d, ... j k d -> ... i j d", left, right)

        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def ref_kernel(data: input_t) -> output_t:
    """Reference TriMul forward pass. Wraps in ``DisableCuDNNTF32``."""
    with DisableCuDNNTF32():
        input_tensor, mask, weights, config = data
        trimul = TriMul(
            dim=config["dim"],
            hidden_dim=config["hidden_dim"],
            device=input_tensor.device,
        )

        trimul.norm.weight = nn.Parameter(weights["norm.weight"])
        trimul.norm.bias = nn.Parameter(weights["norm.bias"])
        trimul.left_proj.weight = nn.Parameter(weights["left_proj.weight"])
        trimul.right_proj.weight = nn.Parameter(weights["right_proj.weight"])
        trimul.left_gate.weight = nn.Parameter(weights["left_gate.weight"])
        trimul.right_gate.weight = nn.Parameter(weights["right_gate.weight"])
        trimul.out_gate.weight = nn.Parameter(weights["out_gate.weight"])
        trimul.to_out_norm.weight = nn.Parameter(weights["to_out_norm.weight"])
        trimul.to_out_norm.bias = nn.Parameter(weights["to_out_norm.bias"])
        trimul.to_out.weight = nn.Parameter(weights["to_out.weight"])

        return trimul(input_tensor, mask)


def generate_input(
    seqlen: int,
    bs: int,
    dim: int,
    hiddendim: int,
    seed: int,
    nomask: bool,
    distribution: str,
) -> input_t:
    """Generate a single TriMul test case deterministically from ``seed``."""
    batch_size = bs
    seq_len = seqlen
    hidden_dim = hiddendim
    no_mask = nomask

    config: Dict[str, Any] = {"hidden_dim": hidden_dim, "dim": dim}

    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)

    weights: Dict[str, torch.Tensor] = {}

    if distribution == "cauchy":
        u = torch.empty(
            (batch_size, seq_len, seq_len, dim),
            device="cuda",
            dtype=torch.float32,
        )
        u.uniform_(0.0, 1.0, generator=gen)
        input_tensor = 2.0 * torch.tan(math.pi * (u - 0.5))
    else:
        input_tensor = torch.randn(
            (batch_size, seq_len, seq_len, dim),
            device="cuda",
            dtype=torch.float32,
            generator=gen,
        ).contiguous()

    if no_mask:
        mask = torch.ones(
            batch_size, seq_len, seq_len, device=input_tensor.device
        )
    else:
        mask = torch.randint(
            0,
            2,
            (batch_size, seq_len, seq_len),
            device=input_tensor.device,
            generator=gen,
        )

    weights["norm.weight"] = torch.randn(dim, device="cuda", dtype=torch.float32)
    weights["norm.bias"] = torch.randn(dim, device="cuda", dtype=torch.float32)
    weights["left_proj.weight"] = torch.randn(
        hidden_dim, dim, device="cuda", dtype=torch.float32
    ) / math.sqrt(hidden_dim)
    weights["right_proj.weight"] = torch.randn(
        hidden_dim, dim, device="cuda", dtype=torch.float32
    ) / math.sqrt(hidden_dim)
    weights["left_gate.weight"] = torch.randn(
        hidden_dim, dim, device="cuda", dtype=torch.float32
    ) / math.sqrt(hidden_dim)
    weights["right_gate.weight"] = torch.randn(
        hidden_dim, dim, device="cuda", dtype=torch.float32
    ) / math.sqrt(hidden_dim)
    weights["out_gate.weight"] = torch.randn(
        hidden_dim, dim, device="cuda", dtype=torch.float32
    ) / math.sqrt(hidden_dim)
    weights["to_out_norm.weight"] = torch.randn(
        hidden_dim, device="cuda", dtype=torch.float32
    )
    weights["to_out.weight"] = torch.randn(
        dim, hidden_dim, device="cuda", dtype=torch.float32
    ) / math.sqrt(dim)
    weights["to_out_norm.bias"] = torch.randn(
        hidden_dim, device="cuda", dtype=torch.float32
    )

    return (input_tensor, mask, weights, config)


# Tolerance from upstream: rtol/atol = 2e-2.
check_implementation: Callable[[input_t, torch.Tensor], Tuple[bool, str]] = (
    make_match_reference(ref_kernel, rtol=2e-2, atol=2e-2)
)


# ---------------------------------------------------------------------------
# Seed kernel — vendored verbatim from
# ``arid_badger.trimul.seed_kernel.SEED_KERNEL_CODE``.
# ---------------------------------------------------------------------------

_SEED_KERNEL_CODE: str = '''import torch
from torch import nn, einsum
from task import input_t, output_t

class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: [bs, seq_len, seq_len, dim]
        mask: [bs, seq_len, seq_len]

        Returns:
            output: [bs, seq_len, seq_len, dim]
        """
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)
        x = x.to(torch.float32)

        left = self.left_proj(x.to(torch.float32))
        right = self.right_proj(x.to(torch.float32))

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x.to(torch.float32)).sigmoid()
        right_gate = self.right_gate(x.to(torch.float32)).sigmoid()
        out_gate = self.out_gate(x.to(torch.float32)).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left.to(torch.bfloat16), right.to(torch.bfloat16))

        out = out.to(torch.float32)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def custom_kernel(data: input_t) -> output_t:
    """
    Reference implementation of TriMul using PyTorch.
    """
    input_tensor, mask, weights, config = data
    trimul = TriMul(config["dim"], config["hidden_dim"]).to(input_tensor.device)

    trimul.norm.weight = nn.Parameter(weights['norm.weight'].to(torch.float32))
    trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'].to(torch.float32))
    trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'].to(torch.float32))
    trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'].to(torch.float32))
    trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'].to(torch.float32))
    trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'].to(torch.float32))
    trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'].to(torch.float32))
    trimul.to_out.weight = nn.Parameter(weights['to_out.weight'].to(torch.float32))
    trimul.norm.bias = nn.Parameter(weights['norm.bias'].to(torch.float32))
    trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'].to(torch.float32))

    output = trimul(input_tensor, mask).to(torch.float32)

    return output
'''


# ---------------------------------------------------------------------------
# Mutation-prompt body — vendored verbatim from
# ``arid_badger.hill_climbing.mutation_providers.trimul_feedback_mutation._TRIMUL_BASE_PROMPT``.
# The test cases listing is baked into the prompt text rather than
# generated programmatically, mirroring upstream's prompt.
# ---------------------------------------------------------------------------

_KERNEL_DESCRIPTION_BODY: str = r'''You are an expert Triton engineer tasked with translating PyTorch code into highly optimized Triton kernel code.

You will be implementing a Triangle Multiplicative Update (TriMul) module that is a core operation
for AlphaFold3, Chai, Protenix, and other protein structure prediction models in BioML.

The TriMul operator operates over a 4D tensor of shape [B, N, N, C].

Your task:
- Implement the "outgoing" version of the TriMul operator from the AlphaFold3 paper.
- You will not have to compute or store gradients for this version. You will only need to implement the forward pass.

Your function should be defined as 'custom_kernel' with the following signature:
Input:
- `data`: Tuple of (input: torch.Tensor, weights: Dict[str, torch.Tensor], config: Dict)
    - input: Input tensor of shape [bs, seq_len, seq_len, dim]
    - mask: Mask tensor of shape [bs, seq_len, seq_len]
    - weights: Dictionary containing model weights
    - config: Dictionary containing model configuration parameters

Output:
- output: Processed tensor [bs, seq_len, seq_len, dim]

**Problem Constraints:**
- B ∈ {1,2}, N ∈ {128,256,512,1024}, c ∈ {128}, c_z ∈ {128,384,768}
- The input distribution will be sampled from a standard Normal distribution, or a heavy-tailed Cauchy distribution (gamma = 2).
- There will either be no mask, or a randomly sampled mask over the inputs.

**Remarks.** So why is this problem so annoying? Because you have to choose whether to load / deal with either the channel dimensions c,c_z that the LayerNorms require (otherwise you have to do a synchronize to compute the statistics like mean / variance) or the sequence dimension N.
The sequence dimension is particularly annoying because it's quite large, but also because we compute pair-wise operations at the last operation that sum over another sequence dimension (this is N^3!).
However, I really like this kernel because it only consists of "simple" operations, and is really easy to understand. It is a true test of "fusions" that torch.compile() doesn't do that well.

Here is a pytorch implementation of the TriMul module. You will want to implement a kernel for the operations in the forward call:

```python
import torch
from torch import nn, einsum
import math

# Reference code in PyTorch
class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: [bs, seq_len, seq_len, dim]
        mask: [bs, seq_len, seq_len]

        Returns:
            output: [bs, seq_len, seq_len, dim]
        """
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left, right)
        # This einsum is the same as the following:
        # out = torch.zeros(batch_size, seq_len, seq_len, dim, device=x.device)

        # # Compute using nested loops
        # for b in range(batch_size):
        #     for i in range(seq_len):
        #         for j in range(seq_len):
        #             # Compute each output element
        #             for k in range(seq_len):
        #                 out[b, i, j] += left[b, i, k, :] * right[b, j, k, :]

        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)
```

Here is some example skeleton code of the entrypoint function you will create:
```python
def custom_kernel(data)
    input_tensor, mask, weights, config = data
    dim, hidden_dim = config["dim"], config["hidden_dim"]

    # Access the given weights of the model
    norm_weight = weights["norm.weight"]
    norm_bias = weights["norm.bias"]
    left_proj_weight = weights["left_proj.weight"]
    right_proj_weight = weights["right_proj.weight"]
    left_gate_weight = weights["left_gate.weight"]
    right_gate_weight = weights["right_gate.weight"]
    out_gate_weight = weights["out_gate.weight"]
    to_out_norm_weight = weights["to_out_norm.weight"]
    to_out_norm_bias = weights["to_out_norm.bias"]
    to_out_weight = weights["to_out.weight"]

    # Perform TriMul

    return out
```

To help you understand which triton version we are using, here is some example triton code for an unrelated task:
```python
import triton
import triton.language as tl

@triton.jit
def matmul_persistent_ws_kernel(
   a_ptr, b_ptr, c_ptr, M, N, K,
   stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
   pid = tl.program_id(axis=0) # async_task 0, 1, 2
   num_pid_m = tl.cdiv(M, BLOCK_M) # async_task 0, 1, 2
   num_pid_n = tl.cdiv(N, BLOCK_N) # async_task 0, 1, 2
   pid_m = pid // num_pid_m # async_task 0, 1, 2
   pid_n = pid % num_pid_n # async_task 0, 1, 2
   offs_m_1 = pid_m * BLOCK_M + tl.arange(0, BLOCK_M // 2) # async_task 0, 1, 2
   offs_m_2 = pid_m * BLOCK_M + tl.arange(BLOCK_M // 2, BLOCK_M) # async_task 0, 1, 2
   offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_N) # async_task 0, 1, 2
   offs_k = tl.arange(0, BLOCK_K) # async_task 0
   a_ptrs_1 = a_ptr + (offs_m_1[:, None] * stride_am + offs_k[None, :] * stride_ak) # async_task 0
   a_ptrs_2 = a_ptr + (offs_m_2[:, None] * stride_am + offs_k[None, :] * stride_ak) # async_task 0
   b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn) # async_task 0
   acc_1 = tl.zeros((BLOCK_M // 2, BLOCK_N), dtype=tl.float32) # async_task 1
   acc_1 = tl.zeros((BLOCK_M // 2, BLOCK_N), dtype=tl.float32) # async_task 2
   for k in range(0, tl.cdiv(K, BLOCK_K)): # async_task 0, 1, 2
       a_1 = tl.load(a_ptrs_1)   # async_task 0
       a_2 = tl.load(a_ptrs_2)   # async_task 0
       b = tl.load(b_ptrs)   # async_task 0
       acc_1 += tl.dot(a_1, b)   # async_task 1
       acc_2 += tl.dot(a_2, b)   # async_task 2
       a_ptrs_1 += BLOCK_K * stride_ak # async_task 0
       a_ptrs_2 += BLOCK_K * stride_ak # async_task 0
       b_ptrs += BLOCK_K * stride_bk # async_task 0
   c_1 = acc_1.to(tl.float16) # async_task 1
   c_2 = acc_2.to(tl.float16) # async_task 2
   c_ptrs_1 = c_ptr_1 + stride_cm * offs_m_1[:, None] + stride_cn * offs_n[None, :] # async_task 1
   c_ptrs_2 = c_ptr_2 + stride_cm * offs_m_2[:, None] + stride_cn * offs_n[None, :] # async_task 2
   tl.store(c_ptrs_1, c_1) # async_task 1
   tl.store(c_ptrs_2, c_2) # async_task 2
```

A few general triton tips:
- tl.arange only takes in constexpr arguments (static or tl.constexpr)
- You cannot use continue in your kernel code
- tl.dot can only take in two input tensors
- There is no tl.mean

Here are the different configs that your kernel will be tested on ("nomask" sets whether there will be no mask, or a randomly sampled mask over the inputs):

Test Cases for correctness and runtime (optimize runtime for these):
  - {"seqlen": 256, "bs": 2, "dim": 128, "hidden_dim": 128, "nomask": True, "distribution": "normal"}
  - {"seqlen": 768, "bs": 1, "dim": 128, "hidden_dim": 128, "nomask": True, "distribution": "cauchy"}
  - {"seqlen": 256, "bs": 2, "dim": 384, "hidden_dim": 128, "nomask": False, "distribution": "normal"}
  - {"seqlen": 512, "bs": 1, "dim": 128, "hidden_dim": 128, "nomask": True, "distribution": "normal"}
  - {"seqlen": 1024, "bs": 1, "dim": 128, "hidden_dim": 128, "nomask": True, "distribution": "cauchy"}
  - {"seqlen": 768, "bs": 1, "dim": 384, "hidden_dim": 128, "nomask": False, "distribution": "normal"}
  - {"seqlen": 1024, "bs": 1, "dim": 384, "hidden_dim": 128, "nomask": True, "distribution": "normal"}
'''


# ---------------------------------------------------------------------------
# Pack
# ---------------------------------------------------------------------------


TRIMUL_PACK: KernelPack[TriMulTestArgs, TriMulCaseSpeedup] = KernelPack(
    name="trimul",
    # Distinct from the legacy ``arid-badger-trimul`` namespace so the
    # two paths can be exercised side-by-side from the same Modal client
    # in the regression test without container collisions.
    modal_app_name="arid-badger-trimul-v2pack",
    correctness_cases=CORRECTNESS_CASES,
    benchmark_cases=BENCHMARK_CASES,
    ref_kernel=ref_kernel,
    generate_input=generate_input,
    check_implementation=check_implementation,
    seed_kernel_code=_SEED_KERNEL_CODE,
    determinism_ctx=DisableCuDNNTF32,
    case_speedup_type=TriMulCaseSpeedup,
    kernel_description_body=_KERNEL_DESCRIPTION_BODY,
)


# ---------------------------------------------------------------------------
# Modal app + benchmarker cls.
# ---------------------------------------------------------------------------


app = modal.App(TRIMUL_PACK.modal_app_name)


@app.cls(image=image, **DEFAULT_CLS_KWARGS)  # ty: ignore[unresolved-attribute]  # pyright: ignore[reportArgumentType]
class ModalTrimulPackBenchmarker:
    """Modal benchmarker for the gpu_mode_kernel-based TriMul pack.

    Distinct from ``arid_badger.trimul.modal_scoring.ModalTriMulBenchmarker``
    so the regression test can dispatch to both side-by-side.
    """

    @modal.method()
    def evaluate_candidate(
        self,
        mutated_kernel_code: str,
        test_cases: list[dict[str, object]],
        max_repeats: int = 100,
        max_time_ns: float = 10e9,
    ) -> list[Any]:
        return run_evaluate_candidate(
            pack=TRIMUL_PACK,
            mutated_kernel_code=mutated_kernel_code,
            test_cases=test_cases,
            max_repeats=max_repeats,
            max_time_ns=max_time_ns,
        )


# Public runtime entry point. Experiments import this and pass it to
# ``GpuModeKernelModalProvider(pack_runtime=TRIMUL_RUNTIME, …)``.
TRIMUL_RUNTIME: PackedModalRuntime[TriMulTestArgs, TriMulCaseSpeedup] = (
    PackedModalRuntime(
        pack=TRIMUL_PACK,
        app=app,
        benchmarker_cls=ModalTrimulPackBenchmarker,
    )
)
