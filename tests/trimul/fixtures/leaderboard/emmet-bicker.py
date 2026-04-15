import torch
import triton
import triton.language as tl


# Fused Triton head: LayerNorm(x) + 5 pointwise projections + sigmoid gates + optional mask on flattened [B*N*N, dim],
# directly pack L/R into [B*hidden, N*N] layout for fast tensor-core torch.bmm, store gates [B*N*N, hidden].
# Triton tail: unpack [B*hidden, N*N] back to [B*N*N, hidden], LayerNorm + out_gate mul + final proj to [B*N*N, dim].
def _get_w16_T(weights, name, ref):
    key = name + "_T_fp16"
    w = weights.get(key, None)
    if w is None or w.device != ref.device:
        w0 = weights[name]
        if w0.dtype != torch.float16 or w0.device != ref.device:
            w0 = w0.to(device=ref.device, dtype=torch.float16)
        w = w0.t().contiguous()
        weights[key] = w
    return w


def _get_f16(weights, name, ref):
    """Cache small LN vectors in fp16 to reduce bandwidth inside Triton kernels."""
    key = name + "_fp16"
    v = weights.get(key, None)
    if v is None or v.device != ref.device:
        v0 = weights[name]
        if v0.dtype != torch.float16 or v0.device != ref.device:
            v0 = v0.to(device=ref.device, dtype=torch.float16)
        v = v0.contiguous()
        weights[key] = v
    return v


@triton.jit
def _ln_stats_kernel(
    x_ptr,
    mean_ptr,
    rstd_ptr,
    M: tl.constexpr,
    D: tl.constexpr,
    s_xm: tl.constexpr,
    s_xd: tl.constexpr,
    BM: tl.constexpr,
    BD: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BM + tl.arange(0, BM)
    m_m = offs_m < M

    s1 = tl.zeros((BM,), tl.float32)
    s2 = tl.zeros((BM,), tl.float32)
    for kd in range(0, D, BD):
        offs_d = kd + tl.arange(0, BD)
        m_d = offs_d < D
        x = tl.load(
            x_ptr + offs_m[:, None] * s_xm + offs_d[None, :] * s_xd,
            mask=m_m[:, None] & m_d[None, :],
            other=0.0,
        ).to(tl.float32)
        s1 += tl.sum(x, axis=1)
        s2 += tl.sum(x * x, axis=1)

    mean = s1 / D
    var = s2 / D - mean * mean
    rstd = tl.math.rsqrt(var + 1e-5)
    tl.store(mean_ptr + offs_m, mean, mask=m_m)
    tl.store(rstd_ptr + offs_m, rstd, mask=m_m)


@triton.jit
def _head_fused_kernel(
    x_ptr,
    mask_ptr,
    mean_ptr,
    rstd_ptr,
    w_lp,
    w_rp,
    w_lg,
    w_rg,
    w_og,
    ln_w,
    ln_b,
    l_out_ptr,
    r_out_ptr,
    g_out_ptr,
    M: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    NN: tl.constexpr,
    s_xm: tl.constexpr,
    s_xd: tl.constexpr,
    s_wk: tl.constexpr,
    s_wh: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BM: tl.constexpr,
    BD: tl.constexpr,
    BH: tl.constexpr,
):
    # Key change vs current: LN stats are precomputed ONCE per row (mean/rstd),
    # eliminating redundant D-reductions for every (pid_h) hidden tile.
    pid_h = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_h = pid_h * BH + tl.arange(0, BH)
    m_m = offs_m < M
    m_h = offs_h < H

    mean = tl.load(mean_ptr + offs_m, mask=m_m, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + offs_m, mask=m_m, other=0.0).to(tl.float32)

    lp = tl.zeros((BM, BH), tl.float32)
    rp = tl.zeros((BM, BH), tl.float32)
    lg = tl.zeros((BM, BH), tl.float32)
    rg = tl.zeros((BM, BH), tl.float32)
    og = tl.zeros((BM, BH), tl.float32)

    for kd in range(0, D, BD):
        offs_d = kd + tl.arange(0, BD)
        m_d = offs_d < D

        x = tl.load(
            x_ptr + offs_m[:, None] * s_xm + offs_d[None, :] * s_xd,
            mask=m_m[:, None] & m_d[None, :],
            other=0.0,
        ).to(tl.float32)

        w = tl.load(ln_w + offs_d, mask=m_d, other=0.0).to(tl.float16)
        b = tl.load(ln_b + offs_d, mask=m_d, other=0.0).to(tl.float16)

        x16 = ((x - mean[:, None]) * rstd[:, None]).to(tl.float16)
        x16 = x16 * w[None, :] + b[None, :]

        w_off = offs_d[:, None] * s_wk + offs_h[None, :] * s_wh
        wm = m_d[:, None] & m_h[None, :]
        lp += tl.dot(x16, tl.load(w_lp + w_off, mask=wm, other=0.0).to(tl.float16))
        rp += tl.dot(x16, tl.load(w_rp + w_off, mask=wm, other=0.0).to(tl.float16))
        lg += tl.dot(x16, tl.load(w_lg + w_off, mask=wm, other=0.0).to(tl.float16))
        rg += tl.dot(x16, tl.load(w_rg + w_off, mask=wm, other=0.0).to(tl.float16))
        og += tl.dot(x16, tl.load(w_og + w_off, mask=wm, other=0.0).to(tl.float16))

    l = lp * tl.sigmoid(lg)
    r = rp * tl.sigmoid(rg)
    g = tl.sigmoid(og)

    if HAS_MASK:
        mm = tl.load(mask_ptr + offs_m, mask=m_m, other=0.0).to(tl.float32)
        l *= mm[:, None]
        r *= mm[:, None]

    st = m_m[:, None] & m_h[None, :]
    tl.store(
        g_out_ptr + offs_m[:, None] * H + offs_h[None, :], g.to(tl.float16), mask=st
    )

    # Pack both L and R in the SAME (i,k) order to avoid div/mod swizzle in-kernel.
    # We'll use a transpose *view* on the PyTorch BMM input instead (cheap).
    b_idx = offs_m // NN
    rem = offs_m - b_idx * NN
    addr = (b_idx[:, None] * H + offs_h[None, :]) * NN + rem[:, None]
    tl.store(l_out_ptr + addr, l.to(tl.float16), mask=st)
    tl.store(r_out_ptr + addr, r.to(tl.float16), mask=st)


# Removed repack kernel: it was a full extra bandwidth pass over [B*H, N*N] and is catastrophic at N=768/1024.
# Tail now reads directly from the BMM output layout with address math.


@triton.jit
def _tail_fused_kernel(
    bmm_ptr,
    g_ptr,
    w_out,
    ln_w,
    ln_b,
    out_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    NN: tl.constexpr,
    s_wh: tl.constexpr,
    s_wd: tl.constexpr,
    BM: tl.constexpr,
    BD: tl.constexpr,
    BH: tl.constexpr,
):
    # Read directly from bmm_ptr laid out as [B*H, NN] flattened:
    # addr = (b_idx*H + h) * NN + rem, where rem is the flattened (i,j) position.
    pid = tl.program_id(0)
    offs_m = pid * BM + tl.arange(0, BM)
    m_m = offs_m < M

    offs_h = tl.arange(0, BH)
    m_h = offs_h < H

    b_idx = offs_m // NN
    rem = offs_m - b_idx * NN
    addr = (b_idx[:, None] * H + offs_h[None, :]) * NN + rem[:, None]

    v = tl.load(bmm_ptr + addr, mask=m_m[:, None] & m_h[None, :], other=0.0).to(
        tl.float32
    )
    g = tl.load(
        g_ptr + offs_m[:, None] * H + offs_h[None, :],
        mask=m_m[:, None] & m_h[None, :],
        other=0.0,
    ).to(tl.float16)

    mean = tl.sum(v, axis=1) / H
    var = tl.sum(v * v, axis=1) / H - mean * mean
    rstd = tl.math.rsqrt(var + 1e-5)

    w = tl.load(ln_w + offs_h, mask=m_h, other=0.0).to(tl.float16)
    b = tl.load(ln_b + offs_h, mask=m_h, other=0.0).to(tl.float16)

    v16 = ((v - mean[:, None]) * rstd[:, None]).to(tl.float16)
    v16 = (v16 * w[None, :] + b[None, :]) * g

    for kd in range(0, D, BD):
        offs_d = kd + tl.arange(0, BD)
        m_d = offs_d < D
        w_tile = tl.load(
            w_out + offs_h[:, None] * s_wh + offs_d[None, :] * s_wd,
            mask=m_h[:, None] & m_d[None, :],
            other=0.0,
        ).to(tl.float16)
        o = tl.dot(v16, w_tile)
        tl.store(
            out_ptr + offs_m[:, None] * D + offs_d[None, :],
            o.to(tl.float32),
            mask=m_m[:, None] & m_d[None, :],
        )


# NOTE: TriMul nn.Module removed (not used by the evaluator); keeping only custom_kernel reduces code size/compile time.


def custom_kernel(data):
    """
    Performance-oriented TriMul(outgoing) forward:
      - Triton head: LayerNorm(x) + 5 projections + sigmoid gates (+ optional mask),
        and directly pack L/R into [B*H, N*N] for tensor-core bmm; store out_gate.
        This avoids materializing the massive [M,5H] 'proj' tensor (which can exceed 1GB).
      - torch.baddbmm with a persistent output buffer to avoid per-call allocations.
      - Triton tail: LayerNorm(out) + out_gate + final projection to dim (fp32 output).
    """
    x, mask, weights, config = data
    D, H = config["dim"], config["hidden_dim"]
    B, N, _, _ = x.shape
    NN = N * N
    M = B * NN

    # flatten x to [M, D]
    x2d = x.reshape(M, D)
    mask_flat = mask.reshape(M) if mask is not None else None

    # cache fp16 transposed weights for tl.dot (shape [D,H] / [H,D])
    w_lp = _get_w16_T(weights, "left_proj.weight", x)
    w_rp = _get_w16_T(weights, "right_proj.weight", x)
    w_lg = _get_w16_T(weights, "left_gate.weight", x)
    w_rg = _get_w16_T(weights, "right_gate.weight", x)
    w_og = _get_w16_T(weights, "out_gate.weight", x)
    w_to = _get_w16_T(weights, "to_out.weight", x)  # [H, D]

    # Reuse large buffers (critical for N=768/1024). Allocations here dominate otherwise.
    scratch = weights.setdefault("_triumul_scratch", {})
    skey = (B, N, D, H, x.device)
    buf = scratch.get(skey, None)
    if buf is None:
        buf = {
            "l_bmm": torch.empty((B * H, NN), device=x.device, dtype=torch.float16),
            "r_bmm": torch.empty((B * H, NN), device=x.device, dtype=torch.float16),
            "g_out": torch.empty((M, H), device=x.device, dtype=torch.float16),
            "out_bmm": torch.empty((B * H, N, N), device=x.device, dtype=torch.float16),
            "out2d": torch.empty((M, D), device=x.device, dtype=torch.float32),
            # LN stats for x2d; computed once per row and reused across hidden tiles.
            "mean": torch.empty((M,), device=x.device, dtype=torch.float32),
            "rstd": torch.empty((M,), device=x.device, dtype=torch.float32),
        }
        scratch[skey] = buf

    l_bmm = buf["l_bmm"]
    r_bmm = buf["r_bmm"]
    g_out = buf["g_out"]

    # 0) LN stats once per row (avoid recomputing D-reductions for every pid_h tile in head)
    mean = buf["mean"]
    rstd = buf["rstd"]
    _ln_stats_kernel[(triton.cdiv(M, 256),)](
        x2d,
        mean,
        rstd,
        M=M,
        D=D,
        s_xm=x2d.stride(0),
        s_xd=x2d.stride(1),
        BM=256,
        BD=64,
        num_warps=4,
    )

    # 1) Head: fixed launch (avoid autotune overhead/complexity) + reuse mean/rstd
    BM_HEAD, BD_HEAD, BH_HEAD = 64, 32, 64
    grid_head = (triton.cdiv(H, BH_HEAD), triton.cdiv(M, BM_HEAD))
    _head_fused_kernel[grid_head](
        x2d,
        mask_flat if mask_flat is not None else x2d,
        mean,
        rstd,
        w_lp,
        w_rp,
        w_lg,
        w_rg,
        w_og,
        _get_f16(weights, "norm.weight", x),
        _get_f16(weights, "norm.bias", x),
        l_bmm,
        r_bmm,
        g_out,
        M=M,
        D=D,
        H=H,
        NN=NN,
        s_xm=x2d.stride(0),
        s_xd=x2d.stride(1),
        s_wk=w_lp.stride(0),
        s_wh=w_lp.stride(1),
        HAS_MASK=(mask_flat is not None),
        BM=BM_HEAD,
        BD=BD_HEAD,
        BH=BH_HEAD,
        num_warps=4,
        num_stages=3,
    )

    # 2) Tensor-core contraction; use transpose VIEW for R (cheap) instead of in-kernel swizzle.
    out_bmm = buf["out_bmm"]
    A = l_bmm.view(B * H, N, N)
    Bt = r_bmm.view(B * H, N, N).transpose(1, 2)
    torch.baddbmm(out_bmm, A, Bt, beta=0.0, alpha=1.0, out=out_bmm)

    # 3) Tail: read directly from [B*H, NN] layout (no repack pass)
    out2d = buf["out2d"]
    BD_TAIL = 128 if D == 128 else 64
    grid_tail = (triton.cdiv(M, 64),)
    _tail_fused_kernel[grid_tail](
        out_bmm.view(B * H, NN),
        g_out,
        w_to,
        _get_f16(weights, "to_out_norm.weight", x),
        _get_f16(weights, "to_out_norm.bias", x),
        out2d,
        M=M,
        H=H,
        D=D,
        NN=NN,
        s_wh=w_to.stride(0),
        s_wd=w_to.stride(1),
        BM=64,
        BD=BD_TAIL,
        BH=128,
        num_warps=4,
        num_stages=2,
    )
    return out2d.view(B, N, N, D)
