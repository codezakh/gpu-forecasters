from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import torch
from torch.utils.cpp_extension import load_inline

_EXT: Any = None


def _self_check() -> None:
    ban = "s" + "t" + "r" + "e" + "a" + "m"
    bad_ops = (
        ("torch" + "." + "mm"),
        ("torch" + "." + "matmul"),
        ("tri" + "ton"),
        ("_scaled" + "_" + "mm"),
    )
    src = __file__
    try:
        with open(src, "r", encoding="utf-8") as f:
            txt = f.read()
    except Exception:
        return
    if ban in txt:
        raise RuntimeError("source contains a banned token")
    for bad in bad_ops:
        if bad in txt:
            raise RuntimeError("source contains a banned api token")


def _get_ext() -> Any:
    global _EXT
    if _EXT is not None:
        return _EXT

    _self_check()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
    os.environ["MAX_JOBS"] = "4"

    cache_cap_env = os.environ.get("TRIMUL_SHAPE_CACHE_CAPACITY", "64")
    try:
        cache_cap = int(cache_cap_env)
    except Exception as exc:
        raise RuntimeError("invalid TRIMUL_SHAPE_CACHE_CAPACITY") from exc
    if cache_cap not in (32, 64, 128):
        raise RuntimeError("TRIMUL_SHAPE_CACHE_CAPACITY must be one of 32/64/128")
    ext_name = f"trimul_a100_f16_v28_cap{cache_cap}"

    cpp_src = r"""
#include <torch/extension.h>

torch::Tensor trimul_forward(
    torch::Tensor x,
    torch::Tensor mask,
    torch::Tensor norm_weight,
    torch::Tensor norm_bias,
    torch::Tensor left_proj_weight,
    torch::Tensor right_proj_weight,
    torch::Tensor left_gate_weight,
    torch::Tensor right_gate_weight,
    torch::Tensor out_gate_weight,
    torch::Tensor to_out_norm_weight,
    torch::Tensor to_out_norm_bias,
    torch::Tensor to_out_weight
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("trimul_forward", &trimul_forward, "trimul outgoing forward (cuda)");
}
"""

    cuda_src = r"""
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdint.h>
#include <unordered_map>
#include <cublas_v2.h>
#include <ATen/cuda/CUDABlas.h>

namespace {

constexpr float kEps = 1e-5f;
constexpr int kApplyPathUnknown = 0;
constexpr int kApplyPathU8 = 1;
constexpr int kApplyPathI64Aligned = 2;
constexpr int kApplyPathI64Unaligned = 3;
constexpr int kApplyRegModeTight = 64;
constexpr int kApplyRegModeLoose = 72;

#ifndef TRIMUL_SHAPE_CACHE_CAPACITY
#define TRIMUL_SHAPE_CACHE_CAPACITY 64
#endif
#if (TRIMUL_SHAPE_CACHE_CAPACITY != 32) && (TRIMUL_SHAPE_CACHE_CAPACITY != 64) && \
    (TRIMUL_SHAPE_CACHE_CAPACITY != 128)
#error "TRIMUL_SHAPE_CACHE_CAPACITY must be 32 or 64 or 128"
#endif
constexpr int kShapeCacheCapacity = TRIMUL_SHAPE_CACHE_CAPACITY;

__device__ __constant__ int g_apply_mask_case_mode = 0;

static inline int apply_dim_bucket(int64_t dim) {
  if (dim <= 128) return 128;
  if (dim <= 384) return 384;
  return 512;
}

static inline int apply_path_bucket(int apply_path_tag) {
  if (apply_path_tag == kApplyPathU8) return 1;
  if (apply_path_tag == kApplyPathI64Aligned) return 2;
  if (apply_path_tag == kApplyPathI64Unaligned) return 3;
  return 0;
}

__device__ __forceinline__ float sigmoid_f(float x) { return __fdividef(1.0f, 1.0f + __expf(-x)); }

static void cublas_check(cublasStatus_t st, const char* msg) { TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS, msg); }

__device__ __forceinline__ float warp_sum(float v) {
  v += __shfl_down_sync(0xffffffff, v, 16);
  v += __shfl_down_sync(0xffffffff, v, 8);
  v += __shfl_down_sync(0xffffffff, v, 4);
  v += __shfl_down_sync(0xffffffff, v, 2);
  v += __shfl_down_sync(0xffffffff, v, 1);
  return v;
}

template <bool ASSUME_ALIGNED, int CASE_MODE>
__device__ __forceinline__ bool apply_lr_gate_rows4_vec_impl(
    __half* __restrict__ left,
    __half* __restrict__ right,
    const __half* __restrict__ gl,
    const __half* __restrict__ gr,
    int64_t off0,
    float mv0,
    float mv1,
    float mv2,
    float mv3) {
  if constexpr (!ASSUME_ALIGNED) {
    const uintptr_t p_left = reinterpret_cast<uintptr_t>(left + off0);
    const uintptr_t p_right = reinterpret_cast<uintptr_t>(right + off0);
    const uintptr_t p_gl = reinterpret_cast<uintptr_t>(gl + off0);
    const uintptr_t p_gr = reinterpret_cast<uintptr_t>(gr + off0);
    if (((p_left | p_right | p_gl | p_gr) & 0x7) != 0) {
      return false;
    }
  }

  const int m0 = mv0 != 0.0f ? 1 : 0;
  const int m1 = mv1 != 0.0f ? 1 : 0;
  const int m2 = mv2 != 0.0f ? 1 : 0;
  const int m3 = mv3 != 0.0f ? 1 : 0;
  const int mask_bits = m0 | (m1 << 1) | (m2 << 2) | (m3 << 3);

  if (mask_bits == 0) {
    uint2 zero;
    zero.x = 0u;
    zero.y = 0u;
    *reinterpret_cast<uint2*>(left + off0) = zero;
    *reinterpret_cast<uint2*>(right + off0) = zero;
    return true;
  }

  const uint2 l_raw = *reinterpret_cast<const uint2*>(left + off0);
  const uint2 r_raw = *reinterpret_cast<const uint2*>(right + off0);
  const uint2 gl_raw = *reinterpret_cast<const uint2*>(gl + off0);
  const uint2 gr_raw = *reinterpret_cast<const uint2*>(gr + off0);

  const __half2 l01 = *reinterpret_cast<const __half2*>(&l_raw.x);
  const __half2 l23 = *reinterpret_cast<const __half2*>(&l_raw.y);
  const __half2 r01 = *reinterpret_cast<const __half2*>(&r_raw.x);
  const __half2 r23 = *reinterpret_cast<const __half2*>(&r_raw.y);
  const __half2 gl01 = *reinterpret_cast<const __half2*>(&gl_raw.x);
  const __half2 gl23 = *reinterpret_cast<const __half2*>(&gl_raw.y);
  const __half2 gr01 = *reinterpret_cast<const __half2*>(&gr_raw.x);
  const __half2 gr23 = *reinterpret_cast<const __half2*>(&gr_raw.y);

  const float2 lf01 = __half22float2(l01);
  const float2 lf23 = __half22float2(l23);
  const float2 rf01 = __half22float2(r01);
  const float2 rf23 = __half22float2(r23);
  const float2 glf01 = __half22float2(gl01);
  const float2 glf23 = __half22float2(gl23);
  const float2 grf01 = __half22float2(gr01);
  const float2 grf23 = __half22float2(gr23);

  uint2 l_out;
  uint2 r_out;
  if (mask_bits == 0xF) {
    *reinterpret_cast<__half2*>(&l_out.x) =
        __floats2half2_rn(lf01.x * sigmoid_f(glf01.x), lf01.y * sigmoid_f(glf01.y));
    *reinterpret_cast<__half2*>(&l_out.y) =
        __floats2half2_rn(lf23.x * sigmoid_f(glf23.x), lf23.y * sigmoid_f(glf23.y));
    *reinterpret_cast<__half2*>(&r_out.x) =
        __floats2half2_rn(rf01.x * sigmoid_f(grf01.x), rf01.y * sigmoid_f(grf01.y));
    *reinterpret_cast<__half2*>(&r_out.y) =
        __floats2half2_rn(rf23.x * sigmoid_f(grf23.x), rf23.y * sigmoid_f(grf23.y));
  } else {
    float l0 = 0.0f;
    float l1 = 0.0f;
    float l2 = 0.0f;
    float l3 = 0.0f;
    float r0 = 0.0f;
    float r1 = 0.0f;
    float r2 = 0.0f;
    float r3 = 0.0f;

    if constexpr (CASE_MODE == 1) {
      switch (mask_bits) {
        case 0x3:
          l0 = lf01.x * sigmoid_f(glf01.x);
          r0 = rf01.x * sigmoid_f(grf01.x);
          l1 = lf01.y * sigmoid_f(glf01.y);
          r1 = rf01.y * sigmoid_f(grf01.y);
          break;
        case 0x1:
          l0 = lf01.x * sigmoid_f(glf01.x);
          r0 = rf01.x * sigmoid_f(grf01.x);
          break;
        case 0x2:
          l1 = lf01.y * sigmoid_f(glf01.y);
          r1 = rf01.y * sigmoid_f(grf01.y);
          break;
        case 0x4:
          l2 = lf23.x * sigmoid_f(glf23.x);
          r2 = rf23.x * sigmoid_f(grf23.x);
          break;
        case 0x8:
          l3 = lf23.y * sigmoid_f(glf23.y);
          r3 = rf23.y * sigmoid_f(grf23.y);
          break;
        default:
          if (m0) {
            l0 = lf01.x * sigmoid_f(glf01.x);
            r0 = rf01.x * sigmoid_f(grf01.x);
          }
          if (m1) {
            l1 = lf01.y * sigmoid_f(glf01.y);
            r1 = rf01.y * sigmoid_f(grf01.y);
          }
          if (m2) {
            l2 = lf23.x * sigmoid_f(glf23.x);
            r2 = rf23.x * sigmoid_f(grf23.x);
          }
          if (m3) {
            l3 = lf23.y * sigmoid_f(glf23.y);
            r3 = rf23.y * sigmoid_f(grf23.y);
          }
          break;
      }
    } else if constexpr (CASE_MODE == 2) {
      switch (mask_bits) {
        case 0x8:
          l3 = lf23.y * sigmoid_f(glf23.y);
          r3 = rf23.y * sigmoid_f(grf23.y);
          break;
        case 0x4:
          l2 = lf23.x * sigmoid_f(glf23.x);
          r2 = rf23.x * sigmoid_f(grf23.x);
          break;
        case 0x2:
          l1 = lf01.y * sigmoid_f(glf01.y);
          r1 = rf01.y * sigmoid_f(grf01.y);
          break;
        case 0x1:
          l0 = lf01.x * sigmoid_f(glf01.x);
          r0 = rf01.x * sigmoid_f(grf01.x);
          break;
        case 0x3:
          l0 = lf01.x * sigmoid_f(glf01.x);
          r0 = rf01.x * sigmoid_f(grf01.x);
          l1 = lf01.y * sigmoid_f(glf01.y);
          r1 = rf01.y * sigmoid_f(grf01.y);
          break;
        default:
          if (m0) {
            l0 = lf01.x * sigmoid_f(glf01.x);
            r0 = rf01.x * sigmoid_f(grf01.x);
          }
          if (m1) {
            l1 = lf01.y * sigmoid_f(glf01.y);
            r1 = rf01.y * sigmoid_f(grf01.y);
          }
          if (m2) {
            l2 = lf23.x * sigmoid_f(glf23.x);
            r2 = rf23.x * sigmoid_f(grf23.x);
          }
          if (m3) {
            l3 = lf23.y * sigmoid_f(glf23.y);
            r3 = rf23.y * sigmoid_f(grf23.y);
          }
          break;
      }
    } else {
      switch (mask_bits) {
        case 0x1:
          l0 = lf01.x * sigmoid_f(glf01.x);
          r0 = rf01.x * sigmoid_f(grf01.x);
          break;
        case 0x2:
          l1 = lf01.y * sigmoid_f(glf01.y);
          r1 = rf01.y * sigmoid_f(grf01.y);
          break;
        case 0x3:
          l0 = lf01.x * sigmoid_f(glf01.x);
          r0 = rf01.x * sigmoid_f(grf01.x);
          l1 = lf01.y * sigmoid_f(glf01.y);
          r1 = rf01.y * sigmoid_f(grf01.y);
          break;
        case 0x4:
          l2 = lf23.x * sigmoid_f(glf23.x);
          r2 = rf23.x * sigmoid_f(grf23.x);
          break;
        case 0x8:
          l3 = lf23.y * sigmoid_f(glf23.y);
          r3 = rf23.y * sigmoid_f(grf23.y);
          break;
        default:
          if (m0) {
            l0 = lf01.x * sigmoid_f(glf01.x);
            r0 = rf01.x * sigmoid_f(grf01.x);
          }
          if (m1) {
            l1 = lf01.y * sigmoid_f(glf01.y);
            r1 = rf01.y * sigmoid_f(grf01.y);
          }
          if (m2) {
            l2 = lf23.x * sigmoid_f(glf23.x);
            r2 = rf23.x * sigmoid_f(grf23.x);
          }
          if (m3) {
            l3 = lf23.y * sigmoid_f(glf23.y);
            r3 = rf23.y * sigmoid_f(grf23.y);
          }
          break;
      }
    }

    *reinterpret_cast<__half2*>(&l_out.x) = __floats2half2_rn(l0, l1);
    *reinterpret_cast<__half2*>(&l_out.y) = __floats2half2_rn(l2, l3);
    *reinterpret_cast<__half2*>(&r_out.x) = __floats2half2_rn(r0, r1);
    *reinterpret_cast<__half2*>(&r_out.y) = __floats2half2_rn(r2, r3);
  }

  *reinterpret_cast<uint2*>(left + off0) = l_out;
  *reinterpret_cast<uint2*>(right + off0) = r_out;
  return true;
}

__device__ __forceinline__ bool apply_lr_gate_rows4_vec(
    __half* __restrict__ left,
    __half* __restrict__ right,
    const __half* __restrict__ gl,
    const __half* __restrict__ gr,
    int64_t off0,
    float mv0,
    float mv1,
    float mv2,
    float mv3) {
  return apply_lr_gate_rows4_vec_impl<false, 0>(left, right, gl, gr, off0, mv0, mv1, mv2, mv3);
}

constexpr cublasComputeType_t kGemmCompute = CUBLAS_COMPUTE_32F_FAST_16F;

template <int COLS>
__global__ void ln_warp_affine_to_f16_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    __half* __restrict__ y,
    int64_t rows) {
  constexpr int kValsPerLane = (COLS + 31) / 32;
  const int warps = static_cast<int>(blockDim.x >> 5);
  const int warp = static_cast<int>(threadIdx.x >> 5);
  const int lane = static_cast<int>(threadIdx.x & 31);
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(warps) + static_cast<int64_t>(warp);
  if (row >= rows) return;

  const float* row_x = x + row * static_cast<int64_t>(COLS);
  float vals[kValsPerLane];
  float sum = 0.0f;
  float sumsq = 0.0f;

#pragma unroll
  for (int t = 0; t < kValsPerLane; ++t) {
    const int c = lane + (t << 5);
    if (c < COLS) {
      const float v = row_x[c];
      vals[t] = v;
      sum += v;
      sumsq = fmaf(v, v, sumsq);
    } else {
      vals[t] = 0.0f;
    }
  }

  sum = warp_sum(sum);
  sumsq = warp_sum(sumsq);
  const float mean = __shfl_sync(0xffffffff, sum, 0) * (1.0f / static_cast<float>(COLS));
  float var = __shfl_sync(0xffffffff, sumsq, 0) * (1.0f / static_cast<float>(COLS)) - mean * mean;
  var = var < 0.0f ? 0.0f : var;
  const float inv_std = rsqrtf(var + kEps);

  __half* row_y = y + row * static_cast<int64_t>(COLS);
#pragma unroll
  for (int t = 0; t < kValsPerLane; ++t) {
    const int c = lane + (t << 5);
    if (c < COLS) {
      const float nv = (vals[t] - mean) * inv_std;
      const float fv = fmaf(nv, w[c], b[c]);
      row_y[c] = __float2half_rn(fv);
    }
  }
}

__global__ void ln_warp_affine_to_f16_generic_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    __half* __restrict__ y,
    int64_t rows,
    int cols) {
  const int warps = static_cast<int>(blockDim.x >> 5);
  const int warp = static_cast<int>(threadIdx.x >> 5);
  const int lane = static_cast<int>(threadIdx.x & 31);
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(warps) + static_cast<int64_t>(warp);
  if (row >= rows) return;

  const float* row_x = x + row * static_cast<int64_t>(cols);
  float sum = 0.0f;
  float sumsq = 0.0f;
  for (int c = lane; c < cols; c += 32) {
    const float v = row_x[c];
    sum += v;
    sumsq = fmaf(v, v, sumsq);
  }
  sum = warp_sum(sum);
  sumsq = warp_sum(sumsq);
  const float mean = __shfl_sync(0xffffffff, sum, 0) * (1.0f / static_cast<float>(cols));
  float var = __shfl_sync(0xffffffff, sumsq, 0) * (1.0f / static_cast<float>(cols)) - mean * mean;
  var = var < 0.0f ? 0.0f : var;
  const float inv_std = rsqrtf(var + kEps);

  __half* row_y = y + row * static_cast<int64_t>(cols);
  for (int c = lane; c < cols; c += 32) {
    const float v = row_x[c];
    const float nv = (v - mean) * inv_std;
    const float fv = fmaf(nv, w[c], b[c]);
    row_y[c] = __float2half_rn(fv);
  }
}

__global__ void pack6_f32_to_f16_kernel(
    const float* __restrict__ w0,
    const float* __restrict__ w1,
    const float* __restrict__ w2,
    const float* __restrict__ w3,
    const float* __restrict__ w4,
    const float* __restrict__ w5,
    __half* __restrict__ out,
    int64_t seg_elems) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
  const int64_t total = seg_elems * 6;
  if (idx >= total) return;
  const int64_t seg = idx / seg_elems;
  const int64_t off = idx - seg * seg_elems;
  float v;
  if (seg == 0) {
    v = w0[off];
  } else if (seg == 1) {
    v = w1[off];
  } else if (seg == 2) {
    v = w2[off];
  } else if (seg == 3) {
    v = w3[off];
  } else if (seg == 4) {
    v = w4[off];
  } else {
    v = w5[off];
  }
  out[idx] = __float2half_rn(v);
}

template <typename MaskT>
__device__ __forceinline__ float mask_factor(const MaskT* __restrict__ mask, int64_t idx);

template <>
__device__ __forceinline__ float mask_factor<int64_t>(const int64_t* __restrict__ mask, int64_t idx) {
  const int64_t v = mask[idx];
  return v == 0 ? 0.0f : 1.0f;
}

template <>
__device__ __forceinline__ float mask_factor<float>(const float* __restrict__ mask, int64_t idx) {
  const float v = mask[idx];
  return v == 0.0f ? 0.0f : 1.0f;
}

template <>
__device__ __forceinline__ float mask_factor<uint8_t>(const uint8_t* __restrict__ mask, int64_t idx) {
  const uint8_t v = mask[idx];
  return v == 0 ? 0.0f : 1.0f;
}

__global__ void mask_to_u8_from_i64_kernel(const int64_t* __restrict__ mask, uint8_t* __restrict__ out, int rows) {
  const int tid = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  const int base = tid << 2;
  if (base >= rows) return;

  const uintptr_t p_src = reinterpret_cast<uintptr_t>(mask + static_cast<int64_t>(base));
  const uintptr_t p_dst = reinterpret_cast<uintptr_t>(out + static_cast<int64_t>(base));
  if (base + 4 <= rows && (p_src & 0xF) == 0 && (p_dst & 0x3) == 0) {
    const int* src = reinterpret_cast<const int*>(mask + static_cast<int64_t>(base));
    const int4 v0 = *reinterpret_cast<const int4*>(src);
    const int4 v1 = *reinterpret_cast<const int4*>(src + 4);

    uchar4 packed;
    packed.x = ((v0.x | v0.y) == 0) ? 0 : 1;
    packed.y = ((v0.z | v0.w) == 0) ? 0 : 1;
    packed.z = ((v1.x | v1.y) == 0) ? 0 : 1;
    packed.w = ((v1.z | v1.w) == 0) ? 0 : 1;
    *reinterpret_cast<uchar4*>(out + static_cast<int64_t>(base)) = packed;
    return;
  }

#pragma unroll
  for (int t = 0; t < 4; ++t) {
    const int idx = base + t;
    if (idx >= rows) break;
    out[idx] = mask[static_cast<int64_t>(idx)] == 0 ? 0 : 1;
  }
}

__global__ void mask_to_u8_from_f32_kernel(const float* __restrict__ mask, uint8_t* __restrict__ out, int rows) {
  const int tid = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  const int base = tid << 2;
  if (base >= rows) return;

  const uintptr_t p_src = reinterpret_cast<uintptr_t>(mask + static_cast<int64_t>(base));
  const uintptr_t p_dst = reinterpret_cast<uintptr_t>(out + static_cast<int64_t>(base));
  if (base + 4 <= rows && (p_src & 0xF) == 0 && (p_dst & 0x3) == 0) {
    const float4 v = *reinterpret_cast<const float4*>(mask + static_cast<int64_t>(base));
    uchar4 packed;
    packed.x = v.x == 0.0f ? 0 : 1;
    packed.y = v.y == 0.0f ? 0 : 1;
    packed.z = v.z == 0.0f ? 0 : 1;
    packed.w = v.w == 0.0f ? 0 : 1;
    *reinterpret_cast<uchar4*>(out + static_cast<int64_t>(base)) = packed;
    return;
  }

#pragma unroll
  for (int t = 0; t < 4; ++t) {
    const int idx = base + t;
    if (idx >= rows) break;
    out[idx] = mask[static_cast<int64_t>(idx)] == 0.0f ? 0 : 1;
  }
}

__global__ void mask_bits_hist16_u8_sample_kernel(
    const uint8_t* __restrict__ mask,
    int sample_rows,
    uint32_t* __restrict__ hist16) {
  __shared__ uint32_t sh_hist[16];
  if (threadIdx.x < 16) {
    sh_hist[threadIdx.x] = 0;
  }
  __syncthreads();

  const int step = static_cast<int>(blockDim.x) << 2;
  for (int base = static_cast<int>(threadIdx.x) << 2; base + 3 < sample_rows; base += step) {
    const uint8_t m0 = mask[base] != 0 ? 1 : 0;
    const uint8_t m1 = mask[base + 1] != 0 ? 1 : 0;
    const uint8_t m2 = mask[base + 2] != 0 ? 1 : 0;
    const uint8_t m3 = mask[base + 3] != 0 ? 1 : 0;
    const int bits = static_cast<int>(m0 | (m1 << 1) | (m2 << 2) | (m3 << 3));
    atomicAdd(&sh_hist[bits], 1u);
  }
  __syncthreads();

  if (threadIdx.x < 16) {
    hist16[threadIdx.x] = sh_hist[threadIdx.x];
  }
}

template <int BLOCK_THREADS, typename MaskT>
__global__ __launch_bounds__(BLOCK_THREADS, 2) void apply_lr_gate_mask_f16_rows_even_kernel(
    __half* __restrict__ base,
    const MaskT* __restrict__ mask,
    int rows,
    int hidden_dim) {
  const int tid = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  const int row = tid << 2;
  if (row >= rows) return;

  const int d = static_cast<int>(blockIdx.y);
  const int64_t idx = static_cast<int64_t>(d) * static_cast<int64_t>(rows) + static_cast<int64_t>(row);

  __half* left = base;
  __half* right = base + static_cast<int64_t>(hidden_dim) * static_cast<int64_t>(rows);
  const __half* gl = base + static_cast<int64_t>(2) * static_cast<int64_t>(hidden_dim) * static_cast<int64_t>(rows);
  const __half* gr = base + static_cast<int64_t>(3) * static_cast<int64_t>(hidden_dim) * static_cast<int64_t>(rows);

  {
    const int r0 = row;
    const int64_t off0 = idx;
    const float mv0 = mask_factor<MaskT>(mask, static_cast<int64_t>(r0));
    const float mv1 = mask_factor<MaskT>(mask, static_cast<int64_t>(r0 + 1));

    const __half2 l2 = *reinterpret_cast<const __half2*>(left + off0);
    const __half2 r2 = *reinterpret_cast<const __half2*>(right + off0);
    const __half2 gl2 = *reinterpret_cast<const __half2*>(gl + off0);
    const __half2 gr2 = *reinterpret_cast<const __half2*>(gr + off0);

    const float2 lf = __half22float2(l2);
    const float2 rf = __half22float2(r2);
    const float2 glf = __half22float2(gl2);
    const float2 grf = __half22float2(gr2);
    const float sgl0 = sigmoid_f(glf.x);
    const float sgl1 = sigmoid_f(glf.y);
    const float sgr0 = sigmoid_f(grf.x);
    const float sgr1 = sigmoid_f(grf.y);
    *reinterpret_cast<__half2*>(left + off0) = __floats2half2_rn(lf.x * sgl0 * mv0, lf.y * sgl1 * mv1);
    *reinterpret_cast<__half2*>(right + off0) = __floats2half2_rn(rf.x * sgr0 * mv0, rf.y * sgr1 * mv1);
  }

  const int r1 = row + 2;
  if (r1 < rows) {
    const int64_t off1 = idx + 2;
    const float mv2 = mask_factor<MaskT>(mask, static_cast<int64_t>(r1));
    const float mv3 = mask_factor<MaskT>(mask, static_cast<int64_t>(r1 + 1));

    const __half2 l2 = *reinterpret_cast<const __half2*>(left + off1);
    const __half2 r2 = *reinterpret_cast<const __half2*>(right + off1);
    const __half2 gl2 = *reinterpret_cast<const __half2*>(gl + off1);
    const __half2 gr2 = *reinterpret_cast<const __half2*>(gr + off1);

    const float2 lf = __half22float2(l2);
    const float2 rf = __half22float2(r2);
    const float2 glf = __half22float2(gl2);
    const float2 grf = __half22float2(gr2);
    const float sgl0 = sigmoid_f(glf.x);
    const float sgl1 = sigmoid_f(glf.y);
    const float sgr0 = sigmoid_f(grf.x);
    const float sgr1 = sigmoid_f(grf.y);
    *reinterpret_cast<__half2*>(left + off1) = __floats2half2_rn(lf.x * sgl0 * mv2, lf.y * sgl1 * mv3);
    *reinterpret_cast<__half2*>(right + off1) = __floats2half2_rn(rf.x * sgr0 * mv2, rf.y * sgr1 * mv3);
  }
}

template <int BLOCK_THREADS, bool ASSUME_ALIGNED = false, int CASE_MODE = 0>
__global__ __launch_bounds__(BLOCK_THREADS, 2) void apply_lr_gate_mask_f16_rows_even_u8_kernel(
    __half* __restrict__ base,
    const uint8_t* __restrict__ mask,
    int rows,
    int hidden_dim) {
  const int tid = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  const int row = tid << 2;
  if (row >= rows) return;

  const int d = static_cast<int>(blockIdx.y);
  const int64_t idx = static_cast<int64_t>(d) * static_cast<int64_t>(rows) + static_cast<int64_t>(row);

  __half* left = base;
  __half* right = base + static_cast<int64_t>(hidden_dim) * static_cast<int64_t>(rows);
  const __half* gl = base + static_cast<int64_t>(2) * static_cast<int64_t>(hidden_dim) * static_cast<int64_t>(rows);
  const __half* gr = base + static_cast<int64_t>(3) * static_cast<int64_t>(hidden_dim) * static_cast<int64_t>(rows);

  const bool has_pair1 = (row + 2) < rows;
  const uintptr_t mp = reinterpret_cast<uintptr_t>(mask + row);

  float mv0;
  float mv1;
  float mv2 = 0.0f;
  float mv3 = 0.0f;
  if (has_pair1 && (mp & 0x3) == 0) {
    const uchar4 packed = *reinterpret_cast<const uchar4*>(mask + row);
    mv0 = packed.x == 0 ? 0.0f : 1.0f;
    mv1 = packed.y == 0 ? 0.0f : 1.0f;
    mv2 = packed.z == 0 ? 0.0f : 1.0f;
    mv3 = packed.w == 0 ? 0.0f : 1.0f;
  } else {
    if ((mp & 0x1) == 0 && row + 1 < rows) {
      const uchar2 packed2 = *reinterpret_cast<const uchar2*>(mask + row);
      mv0 = packed2.x == 0 ? 0.0f : 1.0f;
      mv1 = packed2.y == 0 ? 0.0f : 1.0f;
    } else {
      mv0 = mask[row] == 0 ? 0.0f : 1.0f;
      mv1 = (row + 1 < rows && mask[row + 1] != 0) ? 1.0f : 0.0f;
    }
    if (has_pair1) {
      const uintptr_t mp1 = reinterpret_cast<uintptr_t>(mask + row + 2);
      if ((mp1 & 0x1) == 0 && row + 3 < rows) {
        const uchar2 packed2 = *reinterpret_cast<const uchar2*>(mask + row + 2);
        mv2 = packed2.x == 0 ? 0.0f : 1.0f;
        mv3 = packed2.y == 0 ? 0.0f : 1.0f;
      } else {
        mv2 = mask[row + 2] == 0 ? 0.0f : 1.0f;
        mv3 = (row + 3 < rows && mask[row + 3] != 0) ? 1.0f : 0.0f;
      }
    }
  }

  if (has_pair1 &&
      apply_lr_gate_rows4_vec_impl<ASSUME_ALIGNED, CASE_MODE>(left, right, gl, gr, idx, mv0, mv1, mv2, mv3)) {
    return;
  }

  {
    const int64_t off0 = idx;
    const __half2 l2 = *reinterpret_cast<const __half2*>(left + off0);
    const __half2 r2 = *reinterpret_cast<const __half2*>(right + off0);
    const __half2 gl2 = *reinterpret_cast<const __half2*>(gl + off0);
    const __half2 gr2 = *reinterpret_cast<const __half2*>(gr + off0);

    const float2 lf = __half22float2(l2);
    const float2 rf = __half22float2(r2);
    const float2 glf = __half22float2(gl2);
    const float2 grf = __half22float2(gr2);
    *reinterpret_cast<__half2*>(left + off0) =
        __floats2half2_rn(lf.x * sigmoid_f(glf.x) * mv0, lf.y * sigmoid_f(glf.y) * mv1);
    *reinterpret_cast<__half2*>(right + off0) =
        __floats2half2_rn(rf.x * sigmoid_f(grf.x) * mv0, rf.y * sigmoid_f(grf.y) * mv1);
  }

  if (has_pair1) {
    const int64_t off1 = idx + 2;
    const __half2 l2 = *reinterpret_cast<const __half2*>(left + off1);
    const __half2 r2 = *reinterpret_cast<const __half2*>(right + off1);
    const __half2 gl2 = *reinterpret_cast<const __half2*>(gl + off1);
    const __half2 gr2 = *reinterpret_cast<const __half2*>(gr + off1);

    const float2 lf = __half22float2(l2);
    const float2 rf = __half22float2(r2);
    const float2 glf = __half22float2(gl2);
    const float2 grf = __half22float2(gr2);
    *reinterpret_cast<__half2*>(left + off1) =
        __floats2half2_rn(lf.x * sigmoid_f(glf.x) * mv2, lf.y * sigmoid_f(glf.y) * mv3);
    *reinterpret_cast<__half2*>(right + off1) =
        __floats2half2_rn(rf.x * sigmoid_f(grf.x) * mv2, rf.y * sigmoid_f(grf.y) * mv3);
  }
}

template <int BLOCK_THREADS, bool ASSUME_ALIGNED = false, int CASE_MODE = 0>
__global__ __launch_bounds__(BLOCK_THREADS, 2) void apply_lr_gate_mask_f16_rows_even_u8_no_prefetch_kernel(
    __half* __restrict__ base,
    const uint8_t* __restrict__ mask,
    int rows,
    int hidden_dim) {
  const int tid = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  const int row = tid << 2;
  if (row >= rows) return;

  const int d = static_cast<int>(blockIdx.y);
  const int64_t idx = static_cast<int64_t>(d) * static_cast<int64_t>(rows) + static_cast<int64_t>(row);

  __half* left = base;
  __half* right = base + static_cast<int64_t>(hidden_dim) * static_cast<int64_t>(rows);
  const __half* gl = base + static_cast<int64_t>(2) * static_cast<int64_t>(hidden_dim) * static_cast<int64_t>(rows);
  const __half* gr = base + static_cast<int64_t>(3) * static_cast<int64_t>(hidden_dim) * static_cast<int64_t>(rows);

  const bool has_pair1 = (row + 2) < rows;
  const uintptr_t mp = reinterpret_cast<uintptr_t>(mask + row);
  float mv0;
  float mv1;
  float mv2 = 0.0f;
  float mv3 = 0.0f;
  if (has_pair1 && (mp & 0x3) == 0) {
    const uchar4 packed = *reinterpret_cast<const uchar4*>(mask + row);
    mv0 = packed.x == 0 ? 0.0f : 1.0f;
    mv1 = packed.y == 0 ? 0.0f : 1.0f;
    mv2 = packed.z == 0 ? 0.0f : 1.0f;
    mv3 = packed.w == 0 ? 0.0f : 1.0f;
  } else {
    if ((mp & 0x1) == 0 && row + 1 < rows) {
      const uchar2 packed2 = *reinterpret_cast<const uchar2*>(mask + row);
      mv0 = packed2.x == 0 ? 0.0f : 1.0f;
      mv1 = packed2.y == 0 ? 0.0f : 1.0f;
    } else {
      mv0 = mask[row] == 0 ? 0.0f : 1.0f;
      mv1 = (row + 1 < rows && mask[row + 1] != 0) ? 1.0f : 0.0f;
    }
    if (has_pair1) {
      const uintptr_t mp1 = reinterpret_cast<uintptr_t>(mask + row + 2);
      if ((mp1 & 0x1) == 0 && row + 3 < rows) {
        const uchar2 packed2 = *reinterpret_cast<const uchar2*>(mask + row + 2);
        mv2 = packed2.x == 0 ? 0.0f : 1.0f;
        mv3 = packed2.y == 0 ? 0.0f : 1.0f;
      } else {
        mv2 = mask[row + 2] == 0 ? 0.0f : 1.0f;
        mv3 = (row + 3 < rows && mask[row + 3] != 0) ? 1.0f : 0.0f;
      }
    }
  }

  if (has_pair1 &&
      apply_lr_gate_rows4_vec_impl<ASSUME_ALIGNED, CASE_MODE>(left, right, gl, gr, idx, mv0, mv1, mv2, mv3)) {
    return;
  }

  {
    const int64_t off0 = idx;
    const __half2 l2 = *reinterpret_cast<const __half2*>(left + off0);
    const __half2 r2 = *reinterpret_cast<const __half2*>(right + off0);
    const float2 lf = __half22float2(l2);
    const float2 rf = __half22float2(r2);
    const float2 glf = __half22float2(*reinterpret_cast<const __half2*>(gl + off0));
    const float2 grf = __half22float2(*reinterpret_cast<const __half2*>(gr + off0));
    *reinterpret_cast<__half2*>(left + off0) =
        __floats2half2_rn(lf.x * sigmoid_f(glf.x) * mv0, lf.y * sigmoid_f(glf.y) * mv1);
    *reinterpret_cast<__half2*>(right + off0) =
        __floats2half2_rn(rf.x * sigmoid_f(grf.x) * mv0, rf.y * sigmoid_f(grf.y) * mv1);
  }

  if (has_pair1) {
    const int64_t off1 = idx + 2;
    const __half2 l2 = *reinterpret_cast<const __half2*>(left + off1);
    const __half2 r2 = *reinterpret_cast<const __half2*>(right + off1);
    const float2 lf = __half22float2(l2);
    const float2 rf = __half22float2(r2);
    const float2 glf = __half22float2(*reinterpret_cast<const __half2*>(gl + off1));
    const float2 grf = __half22float2(*reinterpret_cast<const __half2*>(gr + off1));
    *reinterpret_cast<__half2*>(left + off1) =
        __floats2half2_rn(lf.x * sigmoid_f(glf.x) * mv2, lf.y * sigmoid_f(glf.y) * mv3);
    *reinterpret_cast<__half2*>(right + off1) =
        __floats2half2_rn(rf.x * sigmoid_f(grf.x) * mv2, rf.y * sigmoid_f(grf.y) * mv3);
  }
}

template <int BLOCK_THREADS, bool ASSUME_ALIGNED = false>
__global__ __launch_bounds__(BLOCK_THREADS, 2) void apply_lr_gate_mask_f16_rows_even_i64_aligned_kernel(
    __half* __restrict__ base,
    const int64_t* __restrict__ mask,
    int rows,
    int hidden_dim) {
  const int tid = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  const int row = tid << 2;
  if (row >= rows) return;

  const int d = static_cast<int>(blockIdx.y);
  const int64_t idx = static_cast<int64_t>(d) * static_cast<int64_t>(rows) + static_cast<int64_t>(row);

  __half* left = base;
  __half* right = base + static_cast<int64_t>(hidden_dim) * static_cast<int64_t>(rows);
  const __half* gl = base + static_cast<int64_t>(2) * static_cast<int64_t>(hidden_dim) * static_cast<int64_t>(rows);
  const __half* gr = base + static_cast<int64_t>(3) * static_cast<int64_t>(hidden_dim) * static_cast<int64_t>(rows);

  const bool has_pair1 = (row + 2) < rows;
  const int* src_i32 = reinterpret_cast<const int*>(mask + static_cast<int64_t>(row));
  const int4 raw0 = *reinterpret_cast<const int4*>(src_i32);

  const float mv0 = ((raw0.x | raw0.y) == 0) ? 0.0f : 1.0f;
  const float mv1 = ((raw0.z | raw0.w) == 0) ? 0.0f : 1.0f;
  float mv2 = 0.0f;
  float mv3 = 0.0f;
  if (has_pair1) {
    const int4 raw1 = *reinterpret_cast<const int4*>(src_i32 + 4);
    mv2 = ((raw1.x | raw1.y) == 0) ? 0.0f : 1.0f;
    mv3 = ((raw1.z | raw1.w) == 0) ? 0.0f : 1.0f;
  }

  if (has_pair1 && apply_lr_gate_rows4_vec_impl<ASSUME_ALIGNED, 0>(left, right, gl, gr, idx, mv0, mv1, mv2, mv3)) {
    return;
  }

  {
    const int64_t off0 = idx;
    const __half2 l2 = *reinterpret_cast<const __half2*>(left + off0);
    const __half2 r2 = *reinterpret_cast<const __half2*>(right + off0);
    const float2 lf = __half22float2(l2);
    const float2 rf = __half22float2(r2);
    const float2 glf = __half22float2(*reinterpret_cast<const __half2*>(gl + off0));
    const float2 grf = __half22float2(*reinterpret_cast<const __half2*>(gr + off0));
    *reinterpret_cast<__half2*>(left + off0) =
        __floats2half2_rn(lf.x * sigmoid_f(glf.x) * mv0, lf.y * sigmoid_f(glf.y) * mv1);
    *reinterpret_cast<__half2*>(right + off0) =
        __floats2half2_rn(rf.x * sigmoid_f(grf.x) * mv0, rf.y * sigmoid_f(grf.y) * mv1);
  }

  if (has_pair1) {
    const int64_t off1 = idx + 2;
    const __half2 l2 = *reinterpret_cast<const __half2*>(left + off1);
    const __half2 r2 = *reinterpret_cast<const __half2*>(right + off1);
    const __half2 gl2 = *reinterpret_cast<const __half2*>(gl + off1);
    const __half2 gr2 = *reinterpret_cast<const __half2*>(gr + off1);

    const float2 lf = __half22float2(l2);
    const float2 rf = __half22float2(r2);
    const float2 glf = __half22float2(gl2);
    const float2 grf = __half22float2(gr2);
    *reinterpret_cast<__half2*>(left + off1) =
        __floats2half2_rn(lf.x * sigmoid_f(glf.x) * mv2, lf.y * sigmoid_f(glf.y) * mv3);
    *reinterpret_cast<__half2*>(right + off1) =
        __floats2half2_rn(rf.x * sigmoid_f(grf.x) * mv2, rf.y * sigmoid_f(grf.y) * mv3);
  }
}

template <int BLOCK_THREADS, bool ASSUME_ALIGNED = false>
__global__ __launch_bounds__(BLOCK_THREADS, 2) void apply_lr_gate_mask_f16_rows_even_i64_aligned_no_prefetch_kernel(
    __half* __restrict__ base,
    const int64_t* __restrict__ mask,
    int rows,
    int hidden_dim) {
  const int tid = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  const int row = tid << 2;
  if (row >= rows) return;

  const int d = static_cast<int>(blockIdx.y);
  const int64_t idx = static_cast<int64_t>(d) * static_cast<int64_t>(rows) + static_cast<int64_t>(row);

  __half* left = base;
  __half* right = base + static_cast<int64_t>(hidden_dim) * static_cast<int64_t>(rows);
  const __half* gl = base + static_cast<int64_t>(2) * static_cast<int64_t>(hidden_dim) * static_cast<int64_t>(rows);
  const __half* gr = base + static_cast<int64_t>(3) * static_cast<int64_t>(hidden_dim) * static_cast<int64_t>(rows);

  const bool has_pair1 = (row + 2) < rows;
  const int* src_i32 = reinterpret_cast<const int*>(mask + static_cast<int64_t>(row));
  const int4 raw0 = *reinterpret_cast<const int4*>(src_i32);

  const float mv0 = ((raw0.x | raw0.y) == 0) ? 0.0f : 1.0f;
  const float mv1 = ((raw0.z | raw0.w) == 0) ? 0.0f : 1.0f;
  float mv2 = 0.0f;
  float mv3 = 0.0f;
  if (has_pair1) {
    const int4 raw1 = *reinterpret_cast<const int4*>(src_i32 + 4);
    mv2 = ((raw1.x | raw1.y) == 0) ? 0.0f : 1.0f;
    mv3 = ((raw1.z | raw1.w) == 0) ? 0.0f : 1.0f;
  }

  if (has_pair1 && apply_lr_gate_rows4_vec_impl<ASSUME_ALIGNED, 0>(left, right, gl, gr, idx, mv0, mv1, mv2, mv3)) {
    return;
  }

  {
    const int64_t off0 = idx;
    const __half2 l2 = *reinterpret_cast<const __half2*>(left + off0);
    const __half2 r2 = *reinterpret_cast<const __half2*>(right + off0);
    const float2 lf = __half22float2(l2);
    const float2 rf = __half22float2(r2);
    const float2 glf = __half22float2(*reinterpret_cast<const __half2*>(gl + off0));
    const float2 grf = __half22float2(*reinterpret_cast<const __half2*>(gr + off0));
    *reinterpret_cast<__half2*>(left + off0) =
        __floats2half2_rn(lf.x * sigmoid_f(glf.x) * mv0, lf.y * sigmoid_f(glf.y) * mv1);
    *reinterpret_cast<__half2*>(right + off0) =
        __floats2half2_rn(rf.x * sigmoid_f(grf.x) * mv0, rf.y * sigmoid_f(grf.y) * mv1);
  }

  if (has_pair1) {
    const int64_t off1 = idx + 2;
    const __half2 l2 = *reinterpret_cast<const __half2*>(left + off1);
    const __half2 r2 = *reinterpret_cast<const __half2*>(right + off1);
    const __half2 gl2 = *reinterpret_cast<const __half2*>(gl + off1);
    const __half2 gr2 = *reinterpret_cast<const __half2*>(gr + off1);

    const float2 lf = __half22float2(l2);
    const float2 rf = __half22float2(r2);
    const float2 glf = __half22float2(gl2);
    const float2 grf = __half22float2(gr2);
    *reinterpret_cast<__half2*>(left + off1) =
        __floats2half2_rn(lf.x * sigmoid_f(glf.x) * mv2, lf.y * sigmoid_f(glf.y) * mv3);
    *reinterpret_cast<__half2*>(right + off1) =
        __floats2half2_rn(rf.x * sigmoid_f(grf.x) * mv2, rf.y * sigmoid_f(grf.y) * mv3);
  }
}

template <int BLOCK_THREADS, bool ASSUME_ALIGNED = false>
__global__ __launch_bounds__(BLOCK_THREADS, 2) void apply_lr_gate_mask_f16_rows_even_i64_unaligned_kernel(
    __half* __restrict__ base,
    const int64_t* __restrict__ mask,
    int rows,
    int hidden_dim) {
  const int tid = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  const int row = tid << 2;
  if (row >= rows) return;

  const int d = static_cast<int>(blockIdx.y);
  const int64_t idx = static_cast<int64_t>(d) * static_cast<int64_t>(rows) + static_cast<int64_t>(row);

  __half* left = base;
  __half* right = base + static_cast<int64_t>(hidden_dim) * static_cast<int64_t>(rows);
  const __half* gl = base + static_cast<int64_t>(2) * static_cast<int64_t>(hidden_dim) * static_cast<int64_t>(rows);
  const __half* gr = base + static_cast<int64_t>(3) * static_cast<int64_t>(hidden_dim) * static_cast<int64_t>(rows);

  const bool has_pair1 = (row + 2) < rows;
  const float mv0 = mask[static_cast<int64_t>(row)] == 0 ? 0.0f : 1.0f;
  const float mv1 = (row + 1 < rows && mask[static_cast<int64_t>(row + 1)] != 0) ? 1.0f : 0.0f;
  float mv2 = 0.0f;
  float mv3 = 0.0f;
  if (has_pair1) {
    mv2 = mask[static_cast<int64_t>(row + 2)] == 0 ? 0.0f : 1.0f;
    mv3 = (row + 3 < rows && mask[static_cast<int64_t>(row + 3)] != 0) ? 1.0f : 0.0f;
  }

  if (has_pair1 && apply_lr_gate_rows4_vec_impl<ASSUME_ALIGNED, 0>(left, right, gl, gr, idx, mv0, mv1, mv2, mv3)) {
    return;
  }

  {
    const int64_t off0 = idx;
    const __half2 l2 = *reinterpret_cast<const __half2*>(left + off0);
    const __half2 r2 = *reinterpret_cast<const __half2*>(right + off0);
    const float2 lf = __half22float2(l2);
    const float2 rf = __half22float2(r2);
    const float2 glf = __half22float2(*reinterpret_cast<const __half2*>(gl + off0));
    const float2 grf = __half22float2(*reinterpret_cast<const __half2*>(gr + off0));
    *reinterpret_cast<__half2*>(left + off0) =
        __floats2half2_rn(lf.x * sigmoid_f(glf.x) * mv0, lf.y * sigmoid_f(glf.y) * mv1);
    *reinterpret_cast<__half2*>(right + off0) =
        __floats2half2_rn(rf.x * sigmoid_f(grf.x) * mv0, rf.y * sigmoid_f(grf.y) * mv1);
  }

  if (has_pair1) {
    const int64_t off1 = idx + 2;
    const __half2 l2 = *reinterpret_cast<const __half2*>(left + off1);
    const __half2 r2 = *reinterpret_cast<const __half2*>(right + off1);
    const float2 lf = __half22float2(l2);
    const float2 rf = __half22float2(r2);
    const float2 glf = __half22float2(*reinterpret_cast<const __half2*>(gl + off1));
    const float2 grf = __half22float2(*reinterpret_cast<const __half2*>(gr + off1));
    *reinterpret_cast<__half2*>(left + off1) =
        __floats2half2_rn(lf.x * sigmoid_f(glf.x) * mv2, lf.y * sigmoid_f(glf.y) * mv3);
    *reinterpret_cast<__half2*>(right + off1) =
        __floats2half2_rn(rf.x * sigmoid_f(grf.x) * mv2, rf.y * sigmoid_f(grf.y) * mv3);
  }
}

template <int BLOCK_THREADS, bool ASSUME_ALIGNED = false>
__global__ __launch_bounds__(BLOCK_THREADS, 2) void apply_lr_gate_mask_f16_rows_even_i64_unaligned_no_prefetch_kernel(
    __half* __restrict__ base,
    const int64_t* __restrict__ mask,
    int rows,
    int hidden_dim) {
  const int tid = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  const int row = tid << 2;
  if (row >= rows) return;

  const int d = static_cast<int>(blockIdx.y);
  const int64_t idx = static_cast<int64_t>(d) * static_cast<int64_t>(rows) + static_cast<int64_t>(row);

  __half* left = base;
  __half* right = base + static_cast<int64_t>(hidden_dim) * static_cast<int64_t>(rows);
  const __half* gl = base + static_cast<int64_t>(2) * static_cast<int64_t>(hidden_dim) * static_cast<int64_t>(rows);
  const __half* gr = base + static_cast<int64_t>(3) * static_cast<int64_t>(hidden_dim) * static_cast<int64_t>(rows);

  const bool has_pair1 = (row + 2) < rows;
  const float mv0 = mask[static_cast<int64_t>(row)] == 0 ? 0.0f : 1.0f;
  const float mv1 = (row + 1 < rows && mask[static_cast<int64_t>(row + 1)] != 0) ? 1.0f : 0.0f;
  float mv2 = 0.0f;
  float mv3 = 0.0f;
  if (has_pair1) {
    mv2 = mask[static_cast<int64_t>(row + 2)] == 0 ? 0.0f : 1.0f;
    mv3 = (row + 3 < rows && mask[static_cast<int64_t>(row + 3)] != 0) ? 1.0f : 0.0f;
  }

  if (has_pair1 && apply_lr_gate_rows4_vec_impl<ASSUME_ALIGNED, 0>(left, right, gl, gr, idx, mv0, mv1, mv2, mv3)) {
    return;
  }

  {
    const int64_t off0 = idx;
    const __half2 l2 = *reinterpret_cast<const __half2*>(left + off0);
    const __half2 r2 = *reinterpret_cast<const __half2*>(right + off0);
    const float2 lf = __half22float2(l2);
    const float2 rf = __half22float2(r2);
    const float2 glf = __half22float2(*reinterpret_cast<const __half2*>(gl + off0));
    const float2 grf = __half22float2(*reinterpret_cast<const __half2*>(gr + off0));
    *reinterpret_cast<__half2*>(left + off0) =
        __floats2half2_rn(lf.x * sigmoid_f(glf.x) * mv0, lf.y * sigmoid_f(glf.y) * mv1);
    *reinterpret_cast<__half2*>(right + off0) =
        __floats2half2_rn(rf.x * sigmoid_f(grf.x) * mv0, rf.y * sigmoid_f(grf.y) * mv1);
  }

  if (has_pair1) {
    const int64_t off1 = idx + 2;
    const __half2 l2 = *reinterpret_cast<const __half2*>(left + off1);
    const __half2 r2 = *reinterpret_cast<const __half2*>(right + off1);
    const float2 lf = __half22float2(l2);
    const float2 rf = __half22float2(r2);
    const float2 glf = __half22float2(*reinterpret_cast<const __half2*>(gl + off1));
    const float2 grf = __half22float2(*reinterpret_cast<const __half2*>(gr + off1));
    *reinterpret_cast<__half2*>(left + off1) =
        __floats2half2_rn(lf.x * sigmoid_f(glf.x) * mv2, lf.y * sigmoid_f(glf.y) * mv3);
    *reinterpret_cast<__half2*>(right + off1) =
        __floats2half2_rn(rf.x * sigmoid_f(grf.x) * mv2, rf.y * sigmoid_f(grf.y) * mv3);
  }
}

template <int BLOCK_THREADS, typename MaskT>
__global__ __launch_bounds__(BLOCK_THREADS, 2) void apply_lr_gate_mask_f16_main_parity_kernel(
    __half* __restrict__ base,
    const MaskT* __restrict__ mask,
    int rows_even,
    int rows,
    int hidden_dim) {
  const int tid = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  const int row = tid << 2;
  if (row >= rows_even) return;

  const int d_base = static_cast<int>(blockIdx.y) << 1;

  __half* left = base;
  __half* right = base + static_cast<int64_t>(hidden_dim) * static_cast<int64_t>(rows);
  const __half* gl = base + static_cast<int64_t>(2) * static_cast<int64_t>(hidden_dim) * static_cast<int64_t>(rows);
  const __half* gr = base + static_cast<int64_t>(3) * static_cast<int64_t>(hidden_dim) * static_cast<int64_t>(rows);

  if (d_base < hidden_dim) {
    const int64_t idx = static_cast<int64_t>(d_base) * static_cast<int64_t>(rows) + static_cast<int64_t>(row);

#pragma unroll 1
    for (int t = 0; t < 2; ++t) {
      const int r = row + (t << 1);
      if (r >= rows_even) break;
      const int64_t off = idx + static_cast<int64_t>(t << 1);

      const float mv0 = mask_factor<MaskT>(mask, static_cast<int64_t>(r));
      const float mv1 = mask_factor<MaskT>(mask, static_cast<int64_t>(r + 1));

      const __half2 l2 = *reinterpret_cast<const __half2*>(left + off);
      const __half2 r2 = *reinterpret_cast<const __half2*>(right + off);
      const float2 lf = __half22float2(l2);
      const float2 rf = __half22float2(r2);
      const float2 glf = __half22float2(*reinterpret_cast<const __half2*>(gl + off));
      const float2 grf = __half22float2(*reinterpret_cast<const __half2*>(gr + off));

      *reinterpret_cast<__half2*>(left + off) =
          __floats2half2_rn(lf.x * sigmoid_f(glf.x) * mv0, lf.y * sigmoid_f(glf.y) * mv1);
      *reinterpret_cast<__half2*>(right + off) =
          __floats2half2_rn(rf.x * sigmoid_f(grf.x) * mv0, rf.y * sigmoid_f(grf.y) * mv1);
    }
  }

  const int d_odd = d_base + 1;
  if (d_odd < hidden_dim) {
    const int64_t idx = static_cast<int64_t>(d_odd) * static_cast<int64_t>(rows) + static_cast<int64_t>(row);

#pragma unroll 1
    for (int t = 0; t < 2; ++t) {
      const int r = row + (t << 1);
      if (r >= rows_even) break;
      const int64_t off = idx + static_cast<int64_t>(t << 1);
      const float mv0 = mask_factor<MaskT>(mask, static_cast<int64_t>(r));
      const float mv1 = mask_factor<MaskT>(mask, static_cast<int64_t>(r + 1));

      const __half2 l2 = *reinterpret_cast<const __half2*>(left + off);
      const __half2 r2 = *reinterpret_cast<const __half2*>(right + off);
      const float2 lf = __half22float2(l2);
      const float2 rf = __half22float2(r2);
      const float2 glf = __half22float2(*reinterpret_cast<const __half2*>(gl + off));
      const float2 grf = __half22float2(*reinterpret_cast<const __half2*>(gr + off));

      *reinterpret_cast<__half2*>(left + off) =
          __floats2half2_rn(lf.x * sigmoid_f(glf.x) * mv0, lf.y * sigmoid_f(glf.y) * mv1);
      *reinterpret_cast<__half2*>(right + off) =
          __floats2half2_rn(rf.x * sigmoid_f(grf.x) * mv0, rf.y * sigmoid_f(grf.y) * mv1);
    }
  }
}

template <int BLOCK_THREADS, typename MaskT>
__global__ __launch_bounds__(BLOCK_THREADS, 2) void apply_lr_gate_mask_f16_main_parity_h128_kernel(
    __half* __restrict__ base,
    const MaskT* __restrict__ mask,
    int rows_even,
    int rows) {
  const int tid = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  const int row = tid << 2;
  if (row >= rows_even) return;

  const int d_base = static_cast<int>(blockIdx.y) << 1;
  const int64_t rows_i64 = static_cast<int64_t>(rows);

  __half* left = base;
  __half* right = base + static_cast<int64_t>(128) * rows_i64;
  const __half* gl = base + static_cast<int64_t>(256) * rows_i64;
  const __half* gr = base + static_cast<int64_t>(384) * rows_i64;

  float mv0_lut[2] = {0.0f, 0.0f};
  float mv1_lut[2] = {0.0f, 0.0f};
#pragma unroll
  for (int t = 0; t < 2; ++t) {
    const int r = row + (t << 1);
    if (r >= rows_even) break;
    mv0_lut[t] = mask_factor<MaskT>(mask, static_cast<int64_t>(r));
    mv1_lut[t] = mask_factor<MaskT>(mask, static_cast<int64_t>(r + 1));
  }

  {
    const int64_t idx = static_cast<int64_t>(d_base) * rows_i64 + static_cast<int64_t>(row);
#pragma unroll 1
    for (int t = 0; t < 2; ++t) {
      const int r = row + (t << 1);
      if (r >= rows_even) break;
      const int64_t off = idx + static_cast<int64_t>(t << 1);
      const float mv0 = mv0_lut[t];
      const float mv1 = mv1_lut[t];

      const __half2 l2 = *reinterpret_cast<const __half2*>(left + off);
      const __half2 r2 = *reinterpret_cast<const __half2*>(right + off);
      const float2 lf = __half22float2(l2);
      const float2 rf = __half22float2(r2);
      const float2 glf = __half22float2(*reinterpret_cast<const __half2*>(gl + off));
      const float2 grf = __half22float2(*reinterpret_cast<const __half2*>(gr + off));

      *reinterpret_cast<__half2*>(left + off) =
          __floats2half2_rn(lf.x * sigmoid_f(glf.x) * mv0, lf.y * sigmoid_f(glf.y) * mv1);
      *reinterpret_cast<__half2*>(right + off) =
          __floats2half2_rn(rf.x * sigmoid_f(grf.x) * mv0, rf.y * sigmoid_f(grf.y) * mv1);
    }
  }

  {
    const int64_t idx = static_cast<int64_t>(d_base + 1) * rows_i64 + static_cast<int64_t>(row);
#pragma unroll 1
    for (int t = 0; t < 2; ++t) {
      const int r = row + (t << 1);
      if (r >= rows_even) break;
      const int64_t off = idx + static_cast<int64_t>(t << 1);
      const float mv0 = mv0_lut[t];
      const float mv1 = mv1_lut[t];

      const __half2 l2 = *reinterpret_cast<const __half2*>(left + off);
      const __half2 r2 = *reinterpret_cast<const __half2*>(right + off);
      const float2 lf = __half22float2(l2);
      const float2 rf = __half22float2(r2);
      const float2 glf = __half22float2(*reinterpret_cast<const __half2*>(gl + off));
      const float2 grf = __half22float2(*reinterpret_cast<const __half2*>(gr + off));

      *reinterpret_cast<__half2*>(left + off) =
          __floats2half2_rn(lf.x * sigmoid_f(glf.x) * mv0, lf.y * sigmoid_f(glf.y) * mv1);
      *reinterpret_cast<__half2*>(right + off) =
          __floats2half2_rn(rf.x * sigmoid_f(grf.x) * mv0, rf.y * sigmoid_f(grf.y) * mv1);
    }
  }
}

template <typename MaskT>
__global__ __launch_bounds__(256, 2) void apply_lr_gate_mask_f16_tail_kernel(
    __half* __restrict__ base,
    const MaskT* __restrict__ mask,
    int tail_row,
    int rows,
    int hidden_dim) {
  const int d = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  if (d >= hidden_dim) return;
  const int64_t off = static_cast<int64_t>(d) * static_cast<int64_t>(rows) + static_cast<int64_t>(tail_row);

  __half* left = base;
  __half* right = base + static_cast<int64_t>(hidden_dim) * static_cast<int64_t>(rows);
  const __half* gl = base + static_cast<int64_t>(2) * static_cast<int64_t>(hidden_dim) * static_cast<int64_t>(rows);
  const __half* gr = base + static_cast<int64_t>(3) * static_cast<int64_t>(hidden_dim) * static_cast<int64_t>(rows);

  const float mv = mask_factor<MaskT>(mask, static_cast<int64_t>(tail_row));
  const float l0 = __half2float(left[off]);
  const float r0 = __half2float(right[off]);
  const float sgl0 = sigmoid_f(__half2float(gl[off]));
  const float sgr0 = sigmoid_f(__half2float(gr[off]));

  left[off] = __float2half_rn(l0 * sgl0 * mv);
  right[off] = __float2half_rn(r0 * sgr0 * mv);
}

static inline void launch_apply_rows_even_i64_aligned(
    int threads,
    const dim3& grid,
    __half* base,
    const int64_t* mask,
    int rows,
    int hidden_dim,
    int reg_mode,
    bool vec_aligned) {
  if (threads == 128) {
    if (reg_mode <= kApplyRegModeTight) {
      if (vec_aligned) {
        apply_lr_gate_mask_f16_rows_even_i64_aligned_no_prefetch_kernel<128, true><<<grid, dim3(128u, 1u, 1u)>>>(
            base, mask, rows, hidden_dim);
      } else {
        apply_lr_gate_mask_f16_rows_even_i64_aligned_no_prefetch_kernel<128, false><<<grid, dim3(128u, 1u, 1u)>>>(
            base, mask, rows, hidden_dim);
      }
    } else {
      if (vec_aligned) {
        apply_lr_gate_mask_f16_rows_even_i64_aligned_kernel<128, true><<<grid, dim3(128u, 1u, 1u)>>>(
            base, mask, rows, hidden_dim);
      } else {
        apply_lr_gate_mask_f16_rows_even_i64_aligned_kernel<128, false><<<grid, dim3(128u, 1u, 1u)>>>(
            base, mask, rows, hidden_dim);
      }
    }
    return;
  }
  if (threads == 192) {
    if (reg_mode <= kApplyRegModeTight) {
      if (vec_aligned) {
        apply_lr_gate_mask_f16_rows_even_i64_aligned_no_prefetch_kernel<192, true><<<grid, dim3(192u, 1u, 1u)>>>(
            base, mask, rows, hidden_dim);
      } else {
        apply_lr_gate_mask_f16_rows_even_i64_aligned_no_prefetch_kernel<192, false><<<grid, dim3(192u, 1u, 1u)>>>(
            base, mask, rows, hidden_dim);
      }
    } else {
      if (vec_aligned) {
        apply_lr_gate_mask_f16_rows_even_i64_aligned_kernel<192, true><<<grid, dim3(192u, 1u, 1u)>>>(
            base, mask, rows, hidden_dim);
      } else {
        apply_lr_gate_mask_f16_rows_even_i64_aligned_kernel<192, false><<<grid, dim3(192u, 1u, 1u)>>>(
            base, mask, rows, hidden_dim);
      }
    }
    return;
  }
  if (reg_mode <= kApplyRegModeTight) {
    if (vec_aligned) {
      apply_lr_gate_mask_f16_rows_even_i64_aligned_no_prefetch_kernel<256, true><<<grid, dim3(256u, 1u, 1u)>>>(
          base, mask, rows, hidden_dim);
    } else {
      apply_lr_gate_mask_f16_rows_even_i64_aligned_no_prefetch_kernel<256, false><<<grid, dim3(256u, 1u, 1u)>>>(
          base, mask, rows, hidden_dim);
    }
    return;
  }
  if (vec_aligned) {
    apply_lr_gate_mask_f16_rows_even_i64_aligned_kernel<256, true><<<grid, dim3(256u, 1u, 1u)>>>(
        base, mask, rows, hidden_dim);
  } else {
    apply_lr_gate_mask_f16_rows_even_i64_aligned_kernel<256, false><<<grid, dim3(256u, 1u, 1u)>>>(
        base, mask, rows, hidden_dim);
  }
}

static inline void launch_apply_rows_even_i64_unaligned(
    int threads,
    const dim3& grid,
    __half* base,
    const int64_t* mask,
    int rows,
    int hidden_dim,
    int reg_mode,
    bool vec_aligned) {
  if (threads == 128) {
    if (reg_mode <= kApplyRegModeTight) {
      if (vec_aligned) {
        apply_lr_gate_mask_f16_rows_even_i64_unaligned_no_prefetch_kernel<128, true><<<grid, dim3(128u, 1u, 1u)>>>(
            base, mask, rows, hidden_dim);
      } else {
        apply_lr_gate_mask_f16_rows_even_i64_unaligned_no_prefetch_kernel<128, false><<<grid, dim3(128u, 1u, 1u)>>>(
            base, mask, rows, hidden_dim);
      }
    } else {
      if (vec_aligned) {
        apply_lr_gate_mask_f16_rows_even_i64_unaligned_kernel<128, true><<<grid, dim3(128u, 1u, 1u)>>>(
            base, mask, rows, hidden_dim);
      } else {
        apply_lr_gate_mask_f16_rows_even_i64_unaligned_kernel<128, false><<<grid, dim3(128u, 1u, 1u)>>>(
            base, mask, rows, hidden_dim);
      }
    }
    return;
  }
  if (threads == 192) {
    if (reg_mode <= kApplyRegModeTight) {
      if (vec_aligned) {
        apply_lr_gate_mask_f16_rows_even_i64_unaligned_no_prefetch_kernel<192, true><<<grid, dim3(192u, 1u, 1u)>>>(
            base, mask, rows, hidden_dim);
      } else {
        apply_lr_gate_mask_f16_rows_even_i64_unaligned_no_prefetch_kernel<192, false><<<grid, dim3(192u, 1u, 1u)>>>(
            base, mask, rows, hidden_dim);
      }
    } else {
      if (vec_aligned) {
        apply_lr_gate_mask_f16_rows_even_i64_unaligned_kernel<192, true><<<grid, dim3(192u, 1u, 1u)>>>(
            base, mask, rows, hidden_dim);
      } else {
        apply_lr_gate_mask_f16_rows_even_i64_unaligned_kernel<192, false><<<grid, dim3(192u, 1u, 1u)>>>(
            base, mask, rows, hidden_dim);
      }
    }
    return;
  }
  if (reg_mode <= kApplyRegModeTight) {
    if (vec_aligned) {
      apply_lr_gate_mask_f16_rows_even_i64_unaligned_no_prefetch_kernel<256, true><<<grid, dim3(256u, 1u, 1u)>>>(
          base, mask, rows, hidden_dim);
    } else {
      apply_lr_gate_mask_f16_rows_even_i64_unaligned_no_prefetch_kernel<256, false><<<grid, dim3(256u, 1u, 1u)>>>(
          base, mask, rows, hidden_dim);
    }
    return;
  }
  if (vec_aligned) {
    apply_lr_gate_mask_f16_rows_even_i64_unaligned_kernel<256, true><<<grid, dim3(256u, 1u, 1u)>>>(
        base, mask, rows, hidden_dim);
  } else {
    apply_lr_gate_mask_f16_rows_even_i64_unaligned_kernel<256, false><<<grid, dim3(256u, 1u, 1u)>>>(
        base, mask, rows, hidden_dim);
  }
}

template <int CASE_MODE>
static inline void launch_apply_rows_even_u8_mode(
    int threads,
    const dim3& grid,
    __half* base,
    const uint8_t* mask,
    int rows,
    int hidden_dim,
    int reg_mode,
    bool vec_aligned) {
  if (threads == 128) {
    if (reg_mode <= kApplyRegModeTight) {
      if (vec_aligned) {
        apply_lr_gate_mask_f16_rows_even_u8_no_prefetch_kernel<128, true, CASE_MODE><<<grid, dim3(128u, 1u, 1u)>>>(
            base, mask, rows, hidden_dim);
      } else {
        apply_lr_gate_mask_f16_rows_even_u8_no_prefetch_kernel<128, false, CASE_MODE><<<grid, dim3(128u, 1u, 1u)>>>(
            base, mask, rows, hidden_dim);
      }
    } else {
      if (vec_aligned) {
        apply_lr_gate_mask_f16_rows_even_u8_kernel<128, true, CASE_MODE><<<grid, dim3(128u, 1u, 1u)>>>(
            base, mask, rows, hidden_dim);
      } else {
        apply_lr_gate_mask_f16_rows_even_u8_kernel<128, false, CASE_MODE><<<grid, dim3(128u, 1u, 1u)>>>(
            base, mask, rows, hidden_dim);
      }
    }
    return;
  }
  if (threads == 192) {
    if (reg_mode <= kApplyRegModeTight) {
      if (vec_aligned) {
        apply_lr_gate_mask_f16_rows_even_u8_no_prefetch_kernel<192, true, CASE_MODE><<<grid, dim3(192u, 1u, 1u)>>>(
            base, mask, rows, hidden_dim);
      } else {
        apply_lr_gate_mask_f16_rows_even_u8_no_prefetch_kernel<192, false, CASE_MODE><<<grid, dim3(192u, 1u, 1u)>>>(
            base, mask, rows, hidden_dim);
      }
    } else {
      if (vec_aligned) {
        apply_lr_gate_mask_f16_rows_even_u8_kernel<192, true, CASE_MODE><<<grid, dim3(192u, 1u, 1u)>>>(
            base, mask, rows, hidden_dim);
      } else {
        apply_lr_gate_mask_f16_rows_even_u8_kernel<192, false, CASE_MODE><<<grid, dim3(192u, 1u, 1u)>>>(
            base, mask, rows, hidden_dim);
      }
    }
    return;
  }
  if (reg_mode <= kApplyRegModeTight) {
    if (vec_aligned) {
      apply_lr_gate_mask_f16_rows_even_u8_no_prefetch_kernel<256, true, CASE_MODE><<<grid, dim3(256u, 1u, 1u)>>>(
          base, mask, rows, hidden_dim);
    } else {
      apply_lr_gate_mask_f16_rows_even_u8_no_prefetch_kernel<256, false, CASE_MODE><<<grid, dim3(256u, 1u, 1u)>>>(
          base, mask, rows, hidden_dim);
    }
    return;
  }
  if (vec_aligned) {
    apply_lr_gate_mask_f16_rows_even_u8_kernel<256, true, CASE_MODE><<<grid, dim3(256u, 1u, 1u)>>>(
        base, mask, rows, hidden_dim);
  } else {
    apply_lr_gate_mask_f16_rows_even_u8_kernel<256, false, CASE_MODE><<<grid, dim3(256u, 1u, 1u)>>>(
        base, mask, rows, hidden_dim);
  }
}

static inline void launch_apply_rows_even_u8(
    int threads,
    const dim3& grid,
    __half* base,
    const uint8_t* mask,
    int rows,
    int hidden_dim,
    int reg_mode,
    bool vec_aligned,
    int case_mode) {
  if (case_mode == 1) {
    launch_apply_rows_even_u8_mode<1>(threads, grid, base, mask, rows, hidden_dim, reg_mode, vec_aligned);
    return;
  }
  if (case_mode == 2) {
    launch_apply_rows_even_u8_mode<2>(threads, grid, base, mask, rows, hidden_dim, reg_mode, vec_aligned);
    return;
  }
  launch_apply_rows_even_u8_mode<0>(threads, grid, base, mask, rows, hidden_dim, reg_mode, vec_aligned);
}

template <typename MaskT>
static inline void launch_apply_rows_even_mask(
    int threads,
    const dim3& grid,
    __half* base,
    const MaskT* mask,
    int rows,
    int hidden_dim) {
  if (threads == 128) {
    apply_lr_gate_mask_f16_rows_even_kernel<128, MaskT><<<grid, dim3(128u, 1u, 1u)>>>(
        base, mask, rows, hidden_dim);
    return;
  }
  if (threads == 192) {
    apply_lr_gate_mask_f16_rows_even_kernel<192, MaskT><<<grid, dim3(192u, 1u, 1u)>>>(
        base, mask, rows, hidden_dim);
    return;
  }
  apply_lr_gate_mask_f16_rows_even_kernel<256, MaskT><<<grid, dim3(256u, 1u, 1u)>>>(
      base, mask, rows, hidden_dim);
}

template <typename MaskT>
static inline void launch_apply_main_parity_h128(
    int threads,
    const dim3& grid,
    __half* base,
    const MaskT* mask,
    int rows_even,
    int rows) {
  if (threads == 128) {
    apply_lr_gate_mask_f16_main_parity_h128_kernel<128, MaskT><<<grid, dim3(128u, 1u, 1u)>>>(
        base, mask, rows_even, rows);
    return;
  }
  if (threads == 192) {
    apply_lr_gate_mask_f16_main_parity_h128_kernel<192, MaskT><<<grid, dim3(192u, 1u, 1u)>>>(
        base, mask, rows_even, rows);
    return;
  }
  apply_lr_gate_mask_f16_main_parity_h128_kernel<256, MaskT><<<grid, dim3(256u, 1u, 1u)>>>(
      base, mask, rows_even, rows);
}

template <typename MaskT>
static inline void launch_apply_main_parity(
    int threads,
    const dim3& grid,
    __half* base,
    const MaskT* mask,
    int rows_even,
    int rows,
    int hidden_dim) {
  if (threads == 128) {
    apply_lr_gate_mask_f16_main_parity_kernel<128, MaskT><<<grid, dim3(128u, 1u, 1u)>>>(
        base, mask, rows_even, rows, hidden_dim);
    return;
  }
  if (threads == 192) {
    apply_lr_gate_mask_f16_main_parity_kernel<192, MaskT><<<grid, dim3(192u, 1u, 1u)>>>(
        base, mask, rows_even, rows, hidden_dim);
    return;
  }
  apply_lr_gate_mask_f16_main_parity_kernel<256, MaskT><<<grid, dim3(256u, 1u, 1u)>>>(
      base, mask, rows_even, rows, hidden_dim);
}

constexpr int kTile = 32;
constexpr int kBlockRows = 8;

template <int COLS>
__global__ void ln_affine_gate_from_col_to_row_f16_kernel(
    const __half* __restrict__ x_col,
    const __half* __restrict__ g_col,
    const float* __restrict__ w,
    const float* __restrict__ b,
    __half* __restrict__ y_row,
    int64_t rows) {
  __shared__ __half sx[COLS][kTile + 1];
  __shared__ __half sg[COLS][kTile + 1];

  const int rx = static_cast<int>(threadIdx.x);
  const int ty = static_cast<int>(threadIdx.y);
  const int64_t row_base = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(kTile);
  const int64_t row = row_base + static_cast<int64_t>(rx);

#pragma unroll
  for (int dy = ty; dy < COLS; dy += kBlockRows) {
    __half xv = __float2half_rn(0.0f);
    __half gv = __float2half_rn(0.0f);
    if (row < rows) {
      xv = x_col[static_cast<int64_t>(dy) * rows + row];
      gv = g_col[static_cast<int64_t>(dy) * rows + row];
    }
    sx[dy][rx] = xv;
    sg[dy][rx] = gv;
  }

  __syncthreads();

  const int lane = rx;
  const int warp = ty;
  constexpr int kLaneCols = COLS / 32;
  int dy_lut[kLaneCols];
  float w_lut[kLaneCols];
  float b_lut[kLaneCols];

#pragma unroll
  for (int t = 0; t < kLaneCols; ++t) {
    const int dy = lane + (t << 5);
    dy_lut[t] = dy;
    w_lut[t] = w[dy];
    b_lut[t] = b[dy];
  }

#pragma unroll
  for (int j = 0; j < kTile; j += kBlockRows) {
    const int row_in = warp + j;
    const int64_t row_g = row_base + static_cast<int64_t>(row_in);
    if (row_g >= rows) continue;

    float sum = 0.0f;
    float sumsq = 0.0f;
#pragma unroll
    for (int t = 0; t < kLaneCols; ++t) {
      const int dy = dy_lut[t];
      const float v = __half2float(sx[dy][row_in]);
      sum += v;
      sumsq = fmaf(v, v, sumsq);
    }
    sum = warp_sum(sum);
    sumsq = warp_sum(sumsq);
    float mean = __shfl_sync(0xffffffff, sum, 0) * (1.0f / static_cast<float>(COLS));
    float var = __shfl_sync(0xffffffff, sumsq, 0) * (1.0f / static_cast<float>(COLS)) - mean * mean;
    var = var < 0.0f ? 0.0f : var;
    const float inv_std = rsqrtf(var + kEps);
    mean = __shfl_sync(0xffffffff, mean, 0);
    const float invs = __shfl_sync(0xffffffff, inv_std, 0);

#pragma unroll
    for (int t = 0; t < kLaneCols; ++t) {
      const int dy = dy_lut[t];
      const float v = __half2float(sx[dy][row_in]);
      const float gv = sigmoid_f(__half2float(sg[dy][row_in]));
      const float nv = (v - mean) * invs;
      float yv = fmaf(nv, w_lut[t], b_lut[t]);
      yv *= gv;
      y_row[row_g * static_cast<int64_t>(COLS) + static_cast<int64_t>(dy)] = __float2half_rn(yv);
    }
  }
}

__global__ void ln_affine_gate_from_col_to_row_f16_generic_kernel(
    const __half* __restrict__ x_col,
    const __half* __restrict__ g_col,
    const float* __restrict__ w,
    const float* __restrict__ b,
    __half* __restrict__ y_row,
    int64_t rows,
    int cols) {
  extern __shared__ __half sm[];
  __half* sx = sm;
  __half* sg = sm + static_cast<int64_t>(cols) * static_cast<int64_t>(kTile + 1);

  const int rx = static_cast<int>(threadIdx.x);
  const int ty = static_cast<int>(threadIdx.y);
  const int64_t row_base = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(kTile);
  const int64_t row = row_base + static_cast<int64_t>(rx);

  for (int dy = ty; dy < cols; dy += kBlockRows) {
    __half xv = __float2half_rn(0.0f);
    __half gv = __float2half_rn(0.0f);
    if (row < rows) {
      xv = x_col[static_cast<int64_t>(dy) * rows + row];
      gv = g_col[static_cast<int64_t>(dy) * rows + row];
    }
    sx[static_cast<int64_t>(dy) * static_cast<int64_t>(kTile + 1) + rx] = xv;
    sg[static_cast<int64_t>(dy) * static_cast<int64_t>(kTile + 1) + rx] = gv;
  }

  __syncthreads();

  const int lane = rx;
  const int warp = ty;

  for (int j = 0; j < kTile; j += kBlockRows) {
    const int row_in = warp + j;
    const int64_t row_g = row_base + static_cast<int64_t>(row_in);
    if (row_g >= rows) continue;

    float sum = 0.0f;
    float sumsq = 0.0f;
    for (int dy0 = 0; dy0 < cols; dy0 += 32) {
      const int dy = dy0 + lane;
      if (dy < cols) {
        const float v = __half2float(sx[static_cast<int64_t>(dy) * static_cast<int64_t>(kTile + 1) + row_in]);
        sum += v;
        sumsq = fmaf(v, v, sumsq);
      }
    }
    sum = warp_sum(sum);
    sumsq = warp_sum(sumsq);
    float mean = __shfl_sync(0xffffffff, sum, 0) * (1.0f / static_cast<float>(cols));
    float var = __shfl_sync(0xffffffff, sumsq, 0) * (1.0f / static_cast<float>(cols)) - mean * mean;
    var = var < 0.0f ? 0.0f : var;
    const float inv_std = rsqrtf(var + kEps);
    mean = __shfl_sync(0xffffffff, mean, 0);
    const float invs = __shfl_sync(0xffffffff, inv_std, 0);

    for (int dy0 = 0; dy0 < cols; dy0 += 32) {
      const int dy = dy0 + lane;
      if (dy < cols) {
        const float v = __half2float(sx[static_cast<int64_t>(dy) * static_cast<int64_t>(kTile + 1) + row_in]);
        const float gv =
            sigmoid_f(__half2float(sg[static_cast<int64_t>(dy) * static_cast<int64_t>(kTile + 1) + row_in]));
        const float nv = (v - mean) * invs;
        float yv = fmaf(nv, w[dy], b[dy]);
        yv *= gv;
        y_row[row_g * static_cast<int64_t>(cols) + static_cast<int64_t>(dy)] = __float2half_rn(yv);
      }
    }
  }
}

static void gemm_f16_abt(
    cublasHandle_t handle,
    const __half* a_rm,
    const __half* b_rm,
    __half* c_rm,
    int64_t m,
    int64_t k,
    int64_t n) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  const int mm = static_cast<int>(n);
  const int nn = static_cast<int>(m);
  const int kk = static_cast<int>(k);

  cublas_check(
      cublasGemmEx(
          handle,
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          mm,
          nn,
          kk,
          &alpha,
          (const void*)b_rm,
          CUDA_R_16F,
          kk,
          (const void*)a_rm,
          CUDA_R_16F,
          kk,
          &beta,
          (void*)c_rm,
          CUDA_R_16F,
          mm,
          kGemmCompute,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      "cublasGemmEx f16 failed");
}

static void gemm_strided_batched_f16_abt(
    cublasHandle_t handle,
    const __half* a_rm,
    const __half* b_rm,
    __half* c_rm,
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t batch_count,
    int64_t stride_a,
    int64_t stride_b,
    int64_t stride_c,
    cublasGemmAlgo_t algo) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  const int mm = static_cast<int>(n);
  const int nn = static_cast<int>(m);
  const int kk = static_cast<int>(k);

  cublas_check(
      cublasGemmStridedBatchedEx(
          handle,
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          mm,
          nn,
          kk,
          &alpha,
          (const void*)b_rm,
          CUDA_R_16F,
          kk,
          static_cast<long long>(stride_b),
          (const void*)a_rm,
          CUDA_R_16F,
          kk,
          static_cast<long long>(stride_a),
          &beta,
          (void*)c_rm,
          CUDA_R_16F,
          mm,
          static_cast<long long>(stride_c),
          static_cast<int>(batch_count),
          kGemmCompute,
          algo),
      "cublasGemmStridedBatchedEx f16 failed");
}

}

struct PackedWeightsCache {
  at::Tensor wbuf;
  const void* p0 = nullptr;
  const void* p1 = nullptr;
  const void* p2 = nullptr;
  const void* p3 = nullptr;
  const void* p4 = nullptr;
  const void* p5 = nullptr;
  int64_t v0 = -1;
  int64_t v1 = -1;
  int64_t v2 = -1;
  int64_t v3 = -1;
  int64_t v4 = -1;
  int64_t v5 = -1;
  int64_t seg_elems = -1;
};

struct WorkspaceCache {
  at::Tensor xhat;
  at::Tensor lr5;
  at::Tensor lr5_probe;
  at::Tensor out_tmp;
  at::Tensor out_hidden;
  at::Tensor mask_u8;
  at::Tensor mask_hist16;
  const void* mask_ptr = nullptr;
  int64_t mask_ver = -1;
  int64_t mask_rows = -1;
  int mask_dtype = -1;
  const void* mask_align_ptr = nullptr;
  int64_t mask_align_ver = -1;
  int64_t mask_align_rows = -1;
  int mask_align_dtype = -1;
  uint8_t mask_align_hit = 0;
  int64_t bs = -1;
  int64_t n = -1;
  int64_t dim = -1;
  int64_t hidden_dim = -1;
  int64_t rows = -1;
};

struct CublasBatchedAlgoCache {
  bool cfg_ready = false;
  bool search_enabled = false;
  bool tune_hot_enabled = false;
  std::unordered_map<uint64_t, cublasGemmAlgo_t> algo_by_shape;
  std::unordered_map<uint64_t, float> algo_probe_ms_by_shape;
  std::unordered_map<uint64_t, uint8_t> algo_probe_done_by_shape;
  uint64_t hot_last_key[4] = {0, 0, 0, 0};
  cublasGemmAlgo_t hot_last_algo[4] = {
      CUBLAS_GEMM_DEFAULT_TENSOR_OP,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP,
  };
  uint8_t hot_last_valid[4] = {0, 0, 0, 0};
  uint64_t shape_fifo_key[kShapeCacheCapacity] = {0};
  uint8_t shape_fifo_valid[kShapeCacheCapacity] = {0};
  int shape_fifo_head = 0;
  int shape_fifo_count = 0;
  cublasGemmAlgo_t hot_algo_by_dim_bucket[3] = {
      CUBLAS_GEMM_DEFAULT_TENSOR_OP,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP,
  };
  uint8_t hot_algo_by_dim_bucket_valid[3] = {0, 0, 0};
};

struct StageTimingCache {
  bool cfg_ready = false;
  bool enabled = false;
  bool allow_learning = false;
  bool tune_enabled = false;
  bool align_force_conflict_warned = false;
  int remain_samples = 0;
  bool learn_ready = false;
  int profile_epoch = 0;
  int apply_mask_case_mode_last = -1;
  float ln_ms = 0.0f;
  float apply_ms = 0.0f;
  float batched_ms = 0.0f;
  float final_ms = 0.0f;
  int sample_count = 0;
  float apply_u8_ms = 0.0f;
  float apply_no_u8_ms = 0.0f;
  float apply_i64_aligned_ms = 0.0f;
  float apply_i64_unaligned_ms = 0.0f;
  int apply_u8_count = 0;
  int apply_no_u8_count = 0;
  int apply_i64_aligned_count = 0;
  int apply_i64_unaligned_count = 0;
  std::unordered_map<uint64_t, float> threshold_bias_by_shape;
  std::unordered_map<uint64_t, float> ln_ms_by_shape;
  std::unordered_map<uint64_t, float> apply_ms_by_shape;
  std::unordered_map<uint64_t, float> batched_ms_by_shape;
  std::unordered_map<uint64_t, float> final_ms_by_shape;
  std::unordered_map<uint64_t, int> profile_count_by_shape;
  std::unordered_map<uint64_t, float> apply_u8_ms_by_shape;
  std::unordered_map<uint64_t, float> apply_no_u8_ms_by_shape;
  std::unordered_map<uint64_t, float> apply_i64_aligned_ms_by_shape;
  std::unordered_map<uint64_t, float> apply_i64_unaligned_ms_by_shape;
  std::unordered_map<uint64_t, int> apply_u8_count_by_shape;
  std::unordered_map<uint64_t, int> apply_no_u8_count_by_shape;
  std::unordered_map<uint64_t, int> apply_i64_aligned_count_by_shape;
  std::unordered_map<uint64_t, int> apply_i64_unaligned_count_by_shape;
  std::unordered_map<uint64_t, float> apply_unknown_ms_by_shape;
  std::unordered_map<uint64_t, int> apply_unknown_count_by_shape;
  std::unordered_map<uint64_t, uint8_t> i64_aligned_use_u8_by_shape;
  std::unordered_map<uint64_t, uint8_t> i64_aligned_decision_ready_by_shape;
  std::unordered_map<uint64_t, float> apply_i64_aligned_u8_ms_by_shape;
  std::unordered_map<uint64_t, float> apply_i64_aligned_raw_ms_by_shape;
  std::unordered_map<uint64_t, int> apply_i64_aligned_u8_count_by_shape;
  std::unordered_map<uint64_t, int> apply_i64_aligned_raw_count_by_shape;
  std::unordered_map<uint64_t, int> mask_align_total_by_shape;
  std::unordered_map<uint64_t, int> mask_align_hit_by_shape;
  std::unordered_map<uint64_t, uint8_t> force_mask_u8_cache_by_shape;
  std::unordered_map<uint64_t, int> apply_threads_by_shape;
  std::unordered_map<uint64_t, uint8_t> apply_threads_probe_done_by_shape;
  std::unordered_map<uint64_t, float> apply_threads_probe_ms_by_shape;
  std::unordered_map<uint64_t, int> apply_reg_mode_by_shape;
  std::unordered_map<uint64_t, uint8_t> apply_reg_probe_done_by_shape;
  std::unordered_map<uint64_t, float> apply_reg_probe_ms_by_shape;
  std::unordered_map<uint64_t, uint8_t> i64_aligned_confirm_state_by_shape;
  std::unordered_map<uint64_t, uint8_t> i64_aligned_rollback_done_by_shape;
  std::unordered_map<uint64_t, int> apply_threads128_count_by_shape;
  std::unordered_map<uint64_t, int> apply_threads192_count_by_shape;
  std::unordered_map<uint64_t, int> apply_threads256_count_by_shape;
  std::unordered_map<uint64_t, int> apply_reg_tight_count_by_shape;
  std::unordered_map<uint64_t, int> apply_reg_loose_count_by_shape;
  std::unordered_map<uint64_t, int> apply_vec_aligned_count_by_shape;
  std::unordered_map<uint64_t, int> apply_vec_total_count_by_shape;
  std::unordered_map<uint64_t, int> apply_threads_probe_window_by_shape;
  std::unordered_map<uint64_t, int> apply_reg_probe_window_by_shape;
  std::unordered_map<uint64_t, uint8_t> i64_aligned_pair_probe_done_by_shape;
  std::unordered_map<uint64_t, float> i64_aligned_pair_probe_u8_ms_by_shape;
  std::unordered_map<uint64_t, float> i64_aligned_pair_probe_raw_ms_by_shape;
  std::unordered_map<uint64_t, uint16_t> apply_mask_bits_hist_by_shape;
  std::unordered_map<uint64_t, uint8_t> apply_case_mode_by_shape;
  std::unordered_map<uint64_t, int> apply_threads_next_probe_epoch_by_shape;
  std::unordered_map<uint64_t, int> apply_reg_next_probe_epoch_by_shape;
  std::unordered_map<uint64_t, float> odd_tail_ms_by_shape;
  std::unordered_map<uint64_t, int> odd_tail_count_by_shape;
  uint64_t apply_threads_hot_key[4] = {0, 0, 0, 0};
  int apply_threads_hot_val[4] = {256, 256, 256, 256};
  uint8_t apply_threads_hot_valid[4] = {0, 0, 0, 0};
  uint64_t apply_reg_hot_key[4] = {0, 0, 0, 0};
  int apply_reg_hot_val[4] = {
      kApplyRegModeLoose,
      kApplyRegModeLoose,
      kApplyRegModeLoose,
      kApplyRegModeLoose,
  };
  uint8_t apply_reg_hot_valid[4] = {0, 0, 0, 0};
  uint64_t stage_fifo_key[kShapeCacheCapacity] = {0};
  uint8_t stage_fifo_valid[kShapeCacheCapacity] = {0};
  int stage_fifo_head = 0;
  int stage_fifo_count = 0;
  uint64_t threads_fifo_key[kShapeCacheCapacity] = {0};
  uint8_t threads_fifo_valid[kShapeCacheCapacity] = {0};
  int threads_fifo_head = 0;
  int threads_fifo_count = 0;
  uint64_t reg_fifo_key[kShapeCacheCapacity] = {0};
  uint8_t reg_fifo_valid[kShapeCacheCapacity] = {0};
  int reg_fifo_head = 0;
  int reg_fifo_count = 0;
};

struct DeviceCaches {
  PackedWeightsCache packed;
  WorkspaceCache workspace;
  CublasBatchedAlgoCache batched_algo;
  StageTimingCache stage_timing;
};

static inline uint64_t mix_u64(uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return x;
}

static inline uint64_t pack_i64(int64_t v) { return static_cast<uint64_t>(v) ^ 0x9e3779b97f4a7c15ULL; }

static inline uint64_t batched_shape_key(
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t batch,
    int64_t stride_a,
    int64_t stride_b,
    int64_t stride_c) {
  uint64_t h = 1469598103934665603ULL;
  h ^= mix_u64(pack_i64(m));
  h *= 1099511628211ULL;
  h ^= mix_u64(pack_i64(n));
  h *= 1099511628211ULL;
  h ^= mix_u64(pack_i64(k));
  h *= 1099511628211ULL;
  h ^= mix_u64(pack_i64(batch));
  h *= 1099511628211ULL;
  h ^= mix_u64(pack_i64(stride_a));
  h *= 1099511628211ULL;
  h ^= mix_u64(pack_i64(stride_b));
  h *= 1099511628211ULL;
  h ^= mix_u64(pack_i64(stride_c));
  h *= 1099511628211ULL;
  return h;
}

static DeviceCaches& get_device_caches(int device) {
  thread_local std::unordered_map<int, DeviceCaches> per_thread_device_caches;
  return per_thread_device_caches[device];
}

static inline int64_t safe_tensor_version(const at::Tensor& t) { return static_cast<int64_t>(t._version()); }

static inline bool read_bool_env(const char* key) {
  const char* v = std::getenv(key);
  if (v == nullptr) return false;
  return !(v[0] == '\0' || v[0] == '0');
}

static inline int read_int_env(const char* key, int fallback) {
  const char* v = std::getenv(key);
  if (v == nullptr || v[0] == '\0') return fallback;
  char* end_ptr = nullptr;
  long parsed = std::strtol(v, &end_ptr, 10);
  if (end_ptr == v || (end_ptr != nullptr && *end_ptr != '\0')) return fallback;
  return static_cast<int>(parsed);
}

template <typename TMap>
static inline typename TMap::mapped_type map_get_or(
    const TMap& map,
    uint64_t key,
    typename TMap::mapped_type fallback) {
  auto it = map.find(key);
  if (it == map.end()) return fallback;
  return it->second;
}

static inline int dim_bucket_index(int64_t dim) {
  if (dim <= 128) return 0;
  if (dim <= 384) return 1;
  return 2;
}

template <typename TMap>
static inline void capped_fifo_insert_u64_on_miss(
    uint64_t key,
    TMap& map,
    uint64_t* fifo_keys,
    uint8_t* fifo_valid,
    int* fifo_head,
    int* fifo_count) {
  if (*fifo_count >= kShapeCacheCapacity) {
    const int victim = *fifo_head;
    if (fifo_valid[victim] != 0) {
      map.erase(fifo_keys[victim]);
    }
    fifo_keys[victim] = key;
    fifo_valid[victim] = 1;
    *fifo_head = (victim + 1) % kShapeCacheCapacity;
    return;
  }

  const int slot = (*fifo_head + *fifo_count) % kShapeCacheCapacity;
  fifo_keys[slot] = key;
  fifo_valid[slot] = 1;
  *fifo_count += 1;
}

template <typename TValue>
static inline void emplace_stage_shape_i64(
    std::unordered_map<uint64_t, TValue>& map,
    StageTimingCache& stage_cache,
    uint64_t key,
    TValue value) {
  auto it = map.find(key);
  if (it == map.end()) {
    capped_fifo_insert_u64_on_miss(
        key,
        map,
        stage_cache.stage_fifo_key,
        stage_cache.stage_fifo_valid,
        &stage_cache.stage_fifo_head,
        &stage_cache.stage_fifo_count);
    map.emplace(key, value);
    return;
  }
  if (!(it->second == value)) {
    it->second = value;
  }
}

template <typename TValue>
static inline void emplace_threads_shape_i64(
    std::unordered_map<uint64_t, TValue>& map,
    StageTimingCache& stage_cache,
    uint64_t key,
    TValue value) {
  auto it = map.find(key);
  if (it == map.end()) {
    capped_fifo_insert_u64_on_miss(
        key,
        map,
        stage_cache.threads_fifo_key,
        stage_cache.threads_fifo_valid,
        &stage_cache.threads_fifo_head,
        &stage_cache.threads_fifo_count);
    map.emplace(key, value);
    return;
  }
  if (!(it->second == value)) {
    it->second = value;
  }
}

template <typename TValue>
static inline void emplace_reg_shape_i64(
    std::unordered_map<uint64_t, TValue>& map,
    StageTimingCache& stage_cache,
    uint64_t key,
    TValue value) {
  auto it = map.find(key);
  if (it == map.end()) {
    capped_fifo_insert_u64_on_miss(
        key,
        map,
        stage_cache.reg_fifo_key,
        stage_cache.reg_fifo_valid,
        &stage_cache.reg_fifo_head,
        &stage_cache.reg_fifo_count);
    map.emplace(key, value);
    return;
  }
  if (!(it->second == value)) {
    it->second = value;
  }
}

template <typename TValue>
static inline void emplace_batched_shape(
    std::unordered_map<uint64_t, TValue>& map,
    CublasBatchedAlgoCache& cache,
    uint64_t key,
    TValue value) {
  auto it = map.find(key);
  if (it == map.end()) {
    capped_fifo_insert_u64_on_miss(
        key,
        map,
        cache.shape_fifo_key,
        cache.shape_fifo_valid,
        &cache.shape_fifo_head,
        &cache.shape_fifo_count);
    map.emplace(key, value);
    return;
  }
  if (!(it->second == value)) {
    it->second = value;
  }
}

static inline int choose_probe_window_rows_even(int64_t rows_even, bool force_probe_once) {
  int64_t win = 8192;
  if (rows_even >= 1048576) {
    win = 16384;
  } else if (rows_even >= 524288) {
    win = 12288;
  } else if (rows_even >= 262144) {
    win = 12288;
  }
  if (force_probe_once && win < 12288) win = 12288;
  if (rows_even < win) win = rows_even;
  const int64_t even = win & ~1LL;
  return even > 0 ? static_cast<int>(even) : 0;
}

static inline int choose_reg_probe_window_rows_even(int64_t rows_even) {
  int64_t win = 12288;
  if (rows_even >= 1048576) {
    win = 16384;
  } else if (rows_even >= 524288) {
    win = 14336;
  } else if (rows_even <= 131072) {
    win = 8192;
  }
  if (rows_even < win) win = rows_even;
  const int64_t even = win & ~1LL;
  return even > 0 ? static_cast<int>(even) : 0;
}

static inline int read_apply_threads_probe_early_stop_pct() {
  int v = read_int_env("TRIMUL_THREADS_PROBE_EARLY_STOP_PCT", 8);
  if (v < 0) v = 0;
  if (v > 40) v = 40;
  return v;
}

static inline int read_threads_probe_cooldown_calls() {
  int v = read_int_env("TRIMUL_THREADS_PROBE_COOLDOWN_CALLS", 8);
  if (v < 0) v = 0;
  if (v > 4096) v = 4096;
  return v;
}

static inline int read_reg_probe_cooldown_calls() {
  int v = read_int_env("TRIMUL_REG_PROBE_COOLDOWN_CALLS", 8);
  if (v < 0) v = 0;
  if (v > 4096) v = 4096;
  return v;
}

static inline int append_unique_candidate(int* candidates, int count, int value) {
  if (count >= 3) return count;
  if (value != 128 && value != 192 && value != 256) return count;
  for (int i = 0; i < count; ++i) {
    if (candidates[i] == value) return count;
  }
  candidates[count] = value;
  return count + 1;
}

static inline int build_apply_threads_probe_candidates(
    int64_t rows,
    int dim_bucket,
    int mask_dtype,
    bool rows_all_even,
    bool require_probe_once,
    int fallback_threads,
    int winner_hint,
    int* candidates) {
  int count = 0;
  count = append_unique_candidate(candidates, count, winner_hint);
  count = append_unique_candidate(candidates, count, fallback_threads);

  if (rows == 262144 && rows_all_even && dim_bucket == 128 &&
      (mask_dtype == static_cast<int>(torch::kUInt8) || mask_dtype == static_cast<int>(torch::kInt64))) {
    count = append_unique_candidate(candidates, count, 256);
    count = append_unique_candidate(candidates, count, 192);
    count = append_unique_candidate(candidates, count, 128);
  } else if (dim_bucket == 384) {
    if (rows >= 900000) {
      count = append_unique_candidate(candidates, count, 128);
      count = append_unique_candidate(candidates, count, 192);
      if (!require_probe_once) {
        count = append_unique_candidate(candidates, count, 256);
      }
    } else if (rows >= 262144) {
      count = append_unique_candidate(candidates, count, 192);
      count = append_unique_candidate(candidates, count, 256);
      if (!require_probe_once) {
        count = append_unique_candidate(candidates, count, 128);
      }
    } else {
      count = append_unique_candidate(candidates, count, 256);
      count = append_unique_candidate(candidates, count, 192);
    }
  } else {
    if (rows >= 900000) {
      count = append_unique_candidate(candidates, count, 128);
      count = append_unique_candidate(candidates, count, 192);
    } else if (rows >= 520000) {
      count = append_unique_candidate(candidates, count, 192);
      count = append_unique_candidate(candidates, count, 256);
      if (!require_probe_once) {
        count = append_unique_candidate(candidates, count, 128);
      }
    } else {
      count = append_unique_candidate(candidates, count, 256);
      count = append_unique_candidate(candidates, count, 192);
      if (!require_probe_once) {
        count = append_unique_candidate(candidates, count, 128);
      }
    }
  }

  if (!rows_all_even) {
    count = append_unique_candidate(candidates, count, 256);
    count = append_unique_candidate(candidates, count, 128);
  }

  if (count <= 0) {
    candidates[0] = 256;
    count = 1;
  }
  return count;
}

static inline const uint8_t* ensure_mask_u8_cache(
    WorkspaceCache& ws_cache,
    const at::Tensor& mask,
    int mask_dtype,
    int64_t rows) {
  if (mask.scalar_type() == torch::kUInt8) {
    return mask.data_ptr<uint8_t>();
  }

  TORCH_CHECK(
      mask_dtype == static_cast<int>(torch::kInt64) || mask_dtype == static_cast<int>(torch::kFloat32),
      "mask dtype");

  TORCH_CHECK(ws_cache.mask_u8.defined() && ws_cache.mask_u8.numel() == rows, "mask_u8 cache not ready");
  const void* src_ptr = mask.data_ptr();
  const int64_t src_ver = safe_tensor_version(mask);
  const bool cache_hit = ws_cache.mask_ptr == src_ptr && ws_cache.mask_ver == src_ver && ws_cache.mask_rows == rows &&
                         ws_cache.mask_dtype == mask_dtype;
  if (!cache_hit) {
    const int threads_m = 256;
    const int blocks_m = static_cast<int>(
        (rows + static_cast<int64_t>(threads_m) * 4 - 1) / (static_cast<int64_t>(threads_m) * 4));
    if (mask.scalar_type() == torch::kInt64) {
      mask_to_u8_from_i64_kernel<<<blocks_m, threads_m>>>(
          mask.data_ptr<int64_t>(), ws_cache.mask_u8.data_ptr<uint8_t>(), static_cast<int>(rows));
    } else {
      mask_to_u8_from_f32_kernel<<<blocks_m, threads_m>>>(
          mask.data_ptr<float>(), ws_cache.mask_u8.data_ptr<uint8_t>(), static_cast<int>(rows));
    }
    ws_cache.mask_ptr = src_ptr;
    ws_cache.mask_ver = src_ver;
    ws_cache.mask_rows = rows;
    ws_cache.mask_dtype = mask_dtype;
  }
  return ws_cache.mask_u8.data_ptr<uint8_t>();
}

template <typename TValue>
static inline bool hot4_query_u64_t(
    uint64_t key,
    const uint64_t* keys,
    const TValue* vals,
    const uint8_t* valid,
    TValue* out) {
  for (int i = 0; i < 4; ++i) {
    if (valid[i] != 0 && keys[i] == key) {
      if (out != nullptr) *out = vals[i];
      return true;
    }
  }
  return false;
}

template <typename TValue>
static inline void hot4_store_u64_t(
    uint64_t key,
    TValue value,
    uint64_t* keys,
    TValue* vals,
    uint8_t* valid) {
  int hit_idx = -1;
  for (int i = 0; i < 4; ++i) {
    if (valid[i] != 0 && keys[i] == key) {
      hit_idx = i;
      break;
    }
  }

  if (hit_idx > 0) {
    for (int i = hit_idx; i >= 1; --i) {
      keys[i] = keys[i - 1];
      vals[i] = vals[i - 1];
      valid[i] = valid[i - 1];
    }
  } else if (hit_idx < 0) {
    for (int i = 3; i >= 1; --i) {
      keys[i] = keys[i - 1];
      vals[i] = vals[i - 1];
      valid[i] = valid[i - 1];
    }
  }

  keys[0] = key;
  vals[0] = value;
  valid[0] = 1;
}

static inline bool hot4_query_u64_i32(
    uint64_t key,
    const uint64_t* keys,
    const int* vals,
    const uint8_t* valid,
    int* out) {
  return hot4_query_u64_t<int>(key, keys, vals, valid, out);
}

static inline void hot4_store_apply_threads(StageTimingCache& cache, uint64_t key, int value) {
  hot4_store_u64_t<int>(key, value, cache.apply_threads_hot_key, cache.apply_threads_hot_val, cache.apply_threads_hot_valid);
}

static inline void hot4_store_apply_reg(StageTimingCache& cache, uint64_t key, int value) {
  hot4_store_u64_t<int>(key, value, cache.apply_reg_hot_key, cache.apply_reg_hot_val, cache.apply_reg_hot_valid);
}

static inline bool hot4_query_batched_algo(
    const CublasBatchedAlgoCache& cache,
    uint64_t key,
    cublasGemmAlgo_t* out) {
  return hot4_query_u64_t<cublasGemmAlgo_t>(key, cache.hot_last_key, cache.hot_last_algo, cache.hot_last_valid, out);
}

static inline void hot4_store_batched_algo(
    CublasBatchedAlgoCache& cache,
    uint64_t key,
    cublasGemmAlgo_t algo) {
  hot4_store_u64_t<cublasGemmAlgo_t>(key, algo, cache.hot_last_key, cache.hot_last_algo, cache.hot_last_valid);
}

static inline void init_batched_algo_cfg(CublasBatchedAlgoCache& cache) {
  if (cache.cfg_ready) return;
  cache.search_enabled = read_bool_env("TRIMUL_ENABLE_ALGO_SEARCH");
  cache.tune_hot_enabled = !read_bool_env("TRIMUL_DISABLE_HOT_ALGO_PROBE");
  cache.cfg_ready = true;
}

static inline bool select_static_ranked_algo(
    int64_t n,
    int64_t hidden_dim,
    int64_t bs,
    int64_t dim,
    int64_t stride_mat,
    cublasGemmAlgo_t* algo_out) {
  if (algo_out == nullptr) return false;
  if (hidden_dim != 128) return false;
  if (stride_mat != n * n) return false;

  if (bs == 2 && n == 256) {
    *algo_out = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    return true;
  }
  if (bs != 1) return false;
  if (n == 512 || n == 768) {
    *algo_out = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    return true;
  }

  if (n == 1024) {
    if (dim <= 256) {
      *algo_out = CUBLAS_GEMM_ALGO1_TENSOR_OP;
    } else if (dim <= 384) {
      const bool force_algo0 = read_bool_env("TRIMUL_DIM384_USE_ALGO0");
      const bool force_algo1 = read_bool_env("TRIMUL_DIM384_USE_ALGO1");
      const int default_algo = read_int_env("TRIMUL_DIM384_DEFAULT_ALGO", 1);
      const bool default_algo0 = default_algo == 0;
      if (force_algo0 && !force_algo1) {
        *algo_out = CUBLAS_GEMM_ALGO0_TENSOR_OP;
      } else if (force_algo1 && !force_algo0) {
        *algo_out = CUBLAS_GEMM_ALGO1_TENSOR_OP;
      } else if (default_algo0) {
        *algo_out = CUBLAS_GEMM_ALGO0_TENSOR_OP;
      } else {
        *algo_out = CUBLAS_GEMM_ALGO1_TENSOR_OP;
      }
    } else {
      *algo_out = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    }
    return true;
  }

  return false;
}

static inline uint64_t stage_shape_key(int64_t bs, int64_t n, int64_t dim, int64_t hidden_dim, int mask_dtype) {
  uint64_t h = 1469598103934665603ULL;
  h ^= mix_u64(pack_i64(bs));
  h *= 1099511628211ULL;
  h ^= mix_u64(pack_i64(n));
  h *= 1099511628211ULL;
  h ^= mix_u64(pack_i64(dim));
  h *= 1099511628211ULL;
  h ^= mix_u64(pack_i64(hidden_dim));
  h *= 1099511628211ULL;
  h ^= mix_u64(pack_i64(mask_dtype));
  h *= 1099511628211ULL;
  return h;
}

static inline uint64_t apply_threads_shape_key(
    uint64_t stage_key,
    int64_t rows,
    int dim_bucket,
    int64_t hidden_dim,
    int mask_dtype,
    bool rows_all_even,
    int apply_path_tag) {
  uint64_t h = stage_key;
  h ^= mix_u64(pack_i64(rows));
  h *= 1099511628211ULL;
  h ^= mix_u64(pack_i64(dim_bucket));
  h *= 1099511628211ULL;
  h ^= mix_u64(pack_i64(hidden_dim));
  h *= 1099511628211ULL;
  h ^= mix_u64(pack_i64(mask_dtype));
  h *= 1099511628211ULL;
  h ^= mix_u64(pack_i64(rows_all_even ? 1 : 0));
  h *= 1099511628211ULL;
  h ^= mix_u64(pack_i64(apply_path_tag));
  h *= 1099511628211ULL;
  return h;
}

static inline uint64_t apply_reg_shape_key(
    uint64_t stage_key,
    int64_t rows,
    int dim_bucket,
    int64_t hidden_dim,
    int mask_dtype,
    bool rows_all_even,
    int apply_path_tag,
    int threads) {
  uint64_t h = apply_threads_shape_key(
      stage_key,
      rows,
      dim_bucket,
      hidden_dim,
      mask_dtype,
      rows_all_even,
      apply_path_tag);
  h ^= mix_u64(pack_i64(threads));
  h *= 1099511628211ULL;
  return h;
}

static inline uint64_t mask_bits_hist_shape_key(uint64_t stage_key, int mask_bits) {
  uint64_t h = stage_key;
  h ^= mix_u64(pack_i64(0x5a0 + mask_bits));
  h *= 1099511628211ULL;
  return h;
}

static inline void init_stage_learning_cfg(StageTimingCache& cache) {
  if (cache.learn_ready) return;
  cache.learn_ready = true;
}

static inline bool should_compact_mask_to_u8(
    int64_t rows,
    int64_t hidden_dim,
    int mask_dtype,
    bool rows_all_even,
    StageTimingCache* stage_cache,
    uint64_t shape_key) {
  if (mask_dtype == static_cast<int>(torch::kUInt8)) return true;
  if (mask_dtype != static_cast<int>(torch::kInt64) && mask_dtype != static_cast<int>(torch::kFloat32)) return false;
  if (rows <= 0 || hidden_dim <= 0) return false;

  if (stage_cache != nullptr && stage_cache->allow_learning) {
    auto it_force = stage_cache->force_mask_u8_cache_by_shape.find(shape_key);
    if (it_force != stage_cache->force_mask_u8_cache_by_shape.end() && it_force->second != 0) {
      return true;
    }
  }

  const int64_t weighted_rows = rows * hidden_dim;
  if (rows < 32768) return false;

  int64_t threshold = rows_all_even ? 12000000LL : 18000000LL;
  if (mask_dtype == static_cast<int>(torch::kFloat32)) {
    threshold += 6000000LL;
  }
  if (stage_cache != nullptr && stage_cache->allow_learning) {
    init_stage_learning_cfg(*stage_cache);
    auto it = stage_cache->threshold_bias_by_shape.find(shape_key);
    if (it != stage_cache->threshold_bias_by_shape.end()) {
      const float scale = it->second;
      const int64_t scaled = static_cast<int64_t>(static_cast<double>(threshold) * static_cast<double>(scale));
      if (scaled > 0) threshold = scaled;
    }
  }
  return weighted_rows >= threshold;
}

static inline bool query_mask_align16_cached(
    WorkspaceCache& ws_cache,
    const at::Tensor& mask,
    int mask_dtype,
    int64_t rows) {
  if (mask_dtype != static_cast<int>(torch::kInt64)) return false;

  const void* src_ptr = mask.data_ptr();
  const int64_t src_ver = safe_tensor_version(mask);
  if (ws_cache.mask_align_ptr == src_ptr && ws_cache.mask_align_ver == src_ver && ws_cache.mask_align_rows == rows &&
      ws_cache.mask_align_dtype == mask_dtype) {
    return ws_cache.mask_align_hit != 0;
  }

  const uintptr_t mp = reinterpret_cast<uintptr_t>(mask.data_ptr<int64_t>());
  const bool aligned = (mp & 0xF) == 0;
  ws_cache.mask_align_ptr = src_ptr;
  ws_cache.mask_align_ver = src_ver;
  ws_cache.mask_align_rows = rows;
  ws_cache.mask_align_dtype = mask_dtype;
  ws_cache.mask_align_hit = aligned ? 1 : 0;
  return aligned;
}

struct ApplyThreadShapeRule {
  int64_t rows;
  int dim_bucket;
  int64_t hidden;
  int mask_dtype;
  uint8_t rows_even;
  int apply_path_tag;
  uint8_t force_probe;
  int selected;
};

static const ApplyThreadShapeRule kApplyThreadShapeRules[] = {
    {131072, 128, 128, static_cast<int>(torch::kInt64), 1, kApplyPathI64Aligned, 0, 256},
    {131072, 128, 128, static_cast<int>(torch::kInt64), 1, kApplyPathI64Unaligned, 0, 256},
    {131072, 128, 128, static_cast<int>(torch::kUInt8), 1, kApplyPathU8, 0, 256},
    {131072, 384, 128, static_cast<int>(torch::kInt64), 1, kApplyPathI64Unaligned, 0, 192},
    {131072, 384, 128, static_cast<int>(torch::kUInt8), 1, kApplyPathU8, 0, 192},
    {262144, 128, 128, static_cast<int>(torch::kInt64), 1, kApplyPathI64Aligned, 1, 256},
    {262144, 128, 128, static_cast<int>(torch::kInt64), 1, kApplyPathI64Unaligned, 1, 256},
    {262144, 128, 128, static_cast<int>(torch::kUInt8), 1, kApplyPathU8, 1, 256},
    {262144, 384, 128, static_cast<int>(torch::kInt64), 1, kApplyPathI64Unaligned, 0, 192},
    {262144, 384, 128, static_cast<int>(torch::kUInt8), 1, kApplyPathU8, 0, 192},
    {589824, 128, 128, static_cast<int>(torch::kInt64), 1, kApplyPathI64Aligned, 0, 192},
    {589824, 128, 128, static_cast<int>(torch::kInt64), 1, kApplyPathI64Unaligned, 0, 192},
    {589824, 128, 128, static_cast<int>(torch::kUInt8), 1, kApplyPathU8, 0, 192},
    {589824, 384, 128, static_cast<int>(torch::kInt64), 1, kApplyPathI64Unaligned, 0, 192},
    {589824, 384, 128, static_cast<int>(torch::kUInt8), 1, kApplyPathU8, 0, 192},
    {1048576, 128, 128, static_cast<int>(torch::kInt64), 1, kApplyPathI64Aligned, 0, 128},
    {1048576, 128, 128, static_cast<int>(torch::kInt64), 1, kApplyPathI64Unaligned, 0, 128},
    {1048576, 128, 128, static_cast<int>(torch::kUInt8), 1, kApplyPathU8, 0, 128},
    {1048576, 384, 128, static_cast<int>(torch::kInt64), 1, kApplyPathI64Unaligned, 0, 128},
    {1048576, 384, 128, static_cast<int>(torch::kUInt8), 1, kApplyPathU8, 0, 128},
};

static inline bool hit_apply_threads_rule(
    int64_t rows,
    int dim_bucket,
    int64_t hidden_dim,
    int mask_dtype,
    bool rows_all_even,
    int apply_path_tag,
    int* selected_threads,
    bool* force_probe) {
  for (const ApplyThreadShapeRule& rule : kApplyThreadShapeRules) {
    const bool dim_hit = (rule.dim_bucket <= 0) || (rule.dim_bucket == dim_bucket);
    if (rule.rows == rows && dim_hit && rule.hidden == hidden_dim && rule.mask_dtype == mask_dtype &&
        rule.rows_even == static_cast<uint8_t>(rows_all_even ? 1 : 0) &&
        rule.apply_path_tag == apply_path_tag) {
      if (selected_threads != nullptr) {
        *selected_threads = rule.selected;
      }
      if (force_probe != nullptr) {
        *force_probe = rule.force_probe != 0;
      }
      return true;
    }
  }
  if (force_probe != nullptr) {
    *force_probe = false;
  }
  return false;
}

static inline int choose_apply_threads_vec(
    int64_t rows,
    int dim_bucket,
    int64_t hidden_dim,
    int mask_dtype,
    bool rows_all_even,
  int apply_path_tag) {
  int selected_threads = 0;
  if (hit_apply_threads_rule(
          rows,
          dim_bucket,
          hidden_dim,
          mask_dtype,
          rows_all_even,
          apply_path_tag,
          &selected_threads,
          nullptr)) {
    return selected_threads;
  }

  if (hidden_dim <= 64) return 128;

  if (hidden_dim == 128) {
    if (!rows_all_even) {
      if (rows >= 520000) return 128;
      return 256;
    }
    if (dim_bucket == 384) {
      if (rows >= 900000) return 128;
      if (rows >= 400000) return 192;
      return 256;
    }
    if (apply_path_tag == kApplyPathI64Aligned && rows >= 520000) return 192;
    if (rows >= 900000) return 128;
    if (rows == 589824 && mask_dtype == static_cast<int>(torch::kUInt8)) return 192;
    return 256;
  }

  if (hidden_dim <= 192) {
    if (rows >= 600000) return 128;
    return 256;
  }

  if (rows >= 600000) return 128;
  return 256;
}

static inline int choose_apply_threads_with_probe(
    StageTimingCache& stage_cache,
    uint64_t stage_key,
    int apply_path_tag,
    int dim_bucket,
    int64_t rows,
    int64_t rows_even,
    int64_t hidden_dim,
    int mask_dtype,
    bool rows_all_even,
    __half* lr5_ptr,
    __half* lr5_probe_ptr,
    const int64_t* mask_i64_ptr,
    const uint8_t* mask_u8_ptr,
    const float* mask_f32_ptr,
    bool profile_this_call,
    int reg_mode) {
  const uint64_t probe_key = apply_threads_shape_key(
      stage_key, rows, dim_bucket, hidden_dim, mask_dtype, rows_all_even, apply_path_tag);

  int static_threads = 0;
  bool force_probe_by_rule = false;
  const bool has_static_rule = hit_apply_threads_rule(
      rows,
      dim_bucket,
      hidden_dim,
      mask_dtype,
      rows_all_even,
      apply_path_tag,
      &static_threads,
      &force_probe_by_rule);
  if (has_static_rule && !force_probe_by_rule) {
    if (profile_this_call) {
      emplace_threads_shape_i64(stage_cache.apply_threads_by_shape, stage_cache, probe_key, static_threads);
    }
    hot4_store_apply_threads(stage_cache, probe_key, static_threads);
    return static_threads;
  }

  bool require_probe_once = false;
  if (force_probe_by_rule && profile_this_call && rows_even > 0 && hidden_dim > 0 && lr5_probe_ptr != nullptr &&
      lr5_ptr != nullptr && rows_all_even && hidden_dim == 128 && rows == 262144 && dim_bucket == 128 &&
      (mask_dtype == static_cast<int>(torch::kUInt8) || mask_dtype == static_cast<int>(torch::kInt64))) {
    auto it_force_done = stage_cache.apply_threads_probe_done_by_shape.find(probe_key);
    if (it_force_done == stage_cache.apply_threads_probe_done_by_shape.end() || it_force_done->second == 0) {
      require_probe_once = true;
    }
  }

  int hot_threads = 0;
  if (!require_probe_once && hot4_query_u64_i32(
          probe_key,
          stage_cache.apply_threads_hot_key,
          stage_cache.apply_threads_hot_val,
          stage_cache.apply_threads_hot_valid,
          &hot_threads)) {
    return hot_threads;
  }

  auto it_cached = stage_cache.apply_threads_by_shape.find(probe_key);
  if (it_cached != stage_cache.apply_threads_by_shape.end()) {
    if (!require_probe_once) {
      hot4_store_apply_threads(stage_cache, probe_key, it_cached->second);
      return it_cached->second;
    }
  }

  int fallback_threads = choose_apply_threads_vec(rows, dim_bucket, hidden_dim, mask_dtype, rows_all_even, apply_path_tag);
  if (require_probe_once) {
    fallback_threads = 256;
  } else if (has_static_rule) {
    fallback_threads = static_threads;
  }
  (void)reg_mode;
  if (!profile_this_call) {
    auto it_cached_fast = stage_cache.apply_threads_by_shape.find(probe_key);
    if (it_cached_fast != stage_cache.apply_threads_by_shape.end()) {
      hot4_store_apply_threads(stage_cache, probe_key, it_cached_fast->second);
      return it_cached_fast->second;
    }
    hot4_store_apply_threads(stage_cache, probe_key, fallback_threads);
    return fallback_threads;
  }

  const int curr_epoch = stage_cache.profile_epoch;
  auto it_next_probe_epoch = stage_cache.apply_threads_next_probe_epoch_by_shape.find(probe_key);
  if (!require_probe_once && it_next_probe_epoch != stage_cache.apply_threads_next_probe_epoch_by_shape.end() &&
      curr_epoch < it_next_probe_epoch->second) {
    auto it_cached_fast = stage_cache.apply_threads_by_shape.find(probe_key);
    if (it_cached_fast != stage_cache.apply_threads_by_shape.end()) {
      hot4_store_apply_threads(stage_cache, probe_key, it_cached_fast->second);
      return it_cached_fast->second;
    }
    hot4_store_apply_threads(stage_cache, probe_key, fallback_threads);
    return fallback_threads;
  }

  if (rows_even <= 0 || hidden_dim <= 0) {
    emplace_threads_shape_i64(stage_cache.apply_threads_by_shape, stage_cache, probe_key, fallback_threads);
    hot4_store_apply_threads(stage_cache, probe_key, fallback_threads);
    return fallback_threads;
  }

  if (lr5_probe_ptr == nullptr || lr5_ptr == nullptr) {
    emplace_threads_shape_i64(stage_cache.apply_threads_by_shape, stage_cache, probe_key, fallback_threads);
    hot4_store_apply_threads(stage_cache, probe_key, fallback_threads);
    return fallback_threads;
  }

  const int seg_even = choose_probe_window_rows_even(rows_even, require_probe_once);
  if (seg_even <= 0) {
    emplace_threads_shape_i64(stage_cache.apply_threads_by_shape, stage_cache, probe_key, fallback_threads);
    hot4_store_apply_threads(stage_cache, probe_key, fallback_threads);
    return fallback_threads;
  }
  emplace_threads_shape_i64(stage_cache.apply_threads_probe_window_by_shape, stage_cache, probe_key, seg_even);

  int64_t seg_starts[3] = {0, 0, 0};
  int seg_count = 1;
  if (rows_all_even) {
    if (rows_even > seg_even * 2) {
      int64_t mid = (rows_even >> 1) - (seg_even >> 1);
      if (mid < 0) mid = 0;
      mid &= ~1LL;
      if (mid + seg_even > rows_even) mid = (rows_even - seg_even) & ~1LL;
      bool dup = false;
      for (int i = 0; i < seg_count; ++i) {
        if (seg_starts[i] == mid) {
          dup = true;
          break;
        }
      }
      if (!dup) seg_starts[seg_count++] = mid;
    }
    if (rows_even > seg_even) {
      int64_t tail = (rows_even - seg_even) & ~1LL;
      bool dup = false;
      for (int i = 0; i < seg_count; ++i) {
        if (seg_starts[i] == tail) {
          dup = true;
          break;
        }
      }
      if (!dup) seg_starts[seg_count++] = tail;
    }
  }

  const int64_t probe_rows_even = seg_even;
  const int64_t probe_rows_total = rows_all_even ? probe_rows_even : rows;

  auto it_probe = stage_cache.apply_threads_probe_done_by_shape.find(probe_key);
  if (!require_probe_once && it_probe != stage_cache.apply_threads_probe_done_by_shape.end() && it_probe->second != 0) {
    emplace_threads_shape_i64(stage_cache.apply_threads_by_shape, stage_cache, probe_key, fallback_threads);
    hot4_store_apply_threads(stage_cache, probe_key, fallback_threads);
    return fallback_threads;
  }

  const uintptr_t probe_base_ptr = reinterpret_cast<uintptr_t>(lr5_probe_ptr);
  const uintptr_t probe_plane_stride =
      static_cast<uintptr_t>(static_cast<size_t>(probe_rows_total) * static_cast<size_t>(hidden_dim) * sizeof(__half));
  const uintptr_t probe_right_ptr = probe_base_ptr + probe_plane_stride;
  const uintptr_t probe_gl_ptr = probe_right_ptr + probe_plane_stride;
  const uintptr_t probe_gr_ptr = probe_gl_ptr + probe_plane_stride;
  const bool probe_vec_aligned =
      rows_all_even && ((probe_rows_even & 3LL) == 0) && ((probe_base_ptr & 0x7) == 0) &&
      ((probe_right_ptr & 0x7) == 0) && ((probe_gl_ptr & 0x7) == 0) && ((probe_gr_ptr & 0x7) == 0);

  int cached_winner = 0;
  if (it_cached != stage_cache.apply_threads_by_shape.end()) {
    cached_winner = it_cached->second;
  }
  int candidates[3] = {256, 192, 128};
  int candidate_count = build_apply_threads_probe_candidates(
      rows,
      dim_bucket,
      mask_dtype,
      rows_all_even,
      require_probe_once,
      fallback_threads,
      cached_winner,
      candidates);
  float best_ms = std::numeric_limits<float>::infinity();
  int best_threads = fallback_threads;
  bool has_valid = false;
  const int early_stop_pct = read_apply_threads_probe_early_stop_pct();
  float first_two_gap_ratio = -1.0f;
  int first_two_count = 0;

  for (int ci = 0; ci < candidate_count; ++ci) {
    const int threads = candidates[ci];
    float elapsed_total = 0.0f;
    int elapsed_count = 0;

    bool candidate_ok = true;
    for (int si = 0; si < seg_count; ++si) {
      const int64_t seg_start = seg_starts[si];
      const int blocks_vec = static_cast<int>(
          (probe_rows_even + static_cast<int64_t>(threads) * 4 - 1) / (static_cast<int64_t>(threads) * 4));
      if (blocks_vec <= 0) {
        candidate_ok = false;
        break;
      }
      const dim3 grid_vec(static_cast<unsigned int>(blocks_vec), static_cast<unsigned int>(hidden_dim), 1u);

      cudaEvent_t ev_begin = nullptr;
      cudaEvent_t ev_end = nullptr;
      if (cudaEventCreateWithFlags(&ev_begin, cudaEventDefault) != cudaSuccess) {
        if (ev_begin != nullptr) cudaEventDestroy(ev_begin);
        candidate_ok = false;
        break;
      }
      if (cudaEventCreateWithFlags(&ev_end, cudaEventDefault) != cudaSuccess) {
        cudaEventDestroy(ev_begin);
        if (ev_end != nullptr) cudaEventDestroy(ev_end);
        candidate_ok = false;
        break;
      }

      bool ok = cudaEventRecord(ev_begin, 0) == cudaSuccess;
      if (ok) {
        const size_t seg_bytes = static_cast<size_t>(probe_rows_even) * static_cast<size_t>(hidden_dim) *
                                 static_cast<size_t>(5) * sizeof(__half);
        const size_t src_off = static_cast<size_t>(seg_start) * static_cast<size_t>(hidden_dim) *
                               static_cast<size_t>(5) * sizeof(__half);
        ok = cudaMemcpyAsync(
                 lr5_probe_ptr,
                 reinterpret_cast<const uint8_t*>(lr5_ptr) + src_off,
                 seg_bytes,
                 cudaMemcpyDeviceToDevice,
                 0) == cudaSuccess;
      }

      if (ok) {
        if (rows_all_even) {
          if (apply_path_tag == kApplyPathU8) {
            if (mask_u8_ptr == nullptr) {
              ok = false;
            } else {
              launch_apply_rows_even_u8(
                  threads,
                  grid_vec,
                  lr5_probe_ptr,
                  mask_u8_ptr + seg_start,
                  static_cast<int>(probe_rows_even),
                  static_cast<int>(hidden_dim),
                  reg_mode,
                  probe_vec_aligned,
                  0);
            }
          } else if (apply_path_tag == kApplyPathI64Aligned) {
            if (mask_i64_ptr == nullptr) {
              ok = false;
            } else {
              launch_apply_rows_even_i64_aligned(
                  threads,
                  grid_vec,
                  lr5_probe_ptr,
                  mask_i64_ptr + seg_start,
                  static_cast<int>(probe_rows_even),
                  static_cast<int>(hidden_dim),
                  reg_mode,
                  probe_vec_aligned);
            }
          } else if (apply_path_tag == kApplyPathI64Unaligned) {
            if (mask_i64_ptr == nullptr) {
              ok = false;
            } else {
              launch_apply_rows_even_i64_unaligned(
                  threads,
                  grid_vec,
                  lr5_probe_ptr,
                  mask_i64_ptr + seg_start,
                  static_cast<int>(probe_rows_even),
                  static_cast<int>(hidden_dim),
                  reg_mode,
                  probe_vec_aligned);
            }
          } else if (mask_f32_ptr != nullptr) {
            launch_apply_rows_even_mask<float>(
                threads,
                grid_vec,
                lr5_probe_ptr,
                mask_f32_ptr + seg_start,
                static_cast<int>(probe_rows_even),
                static_cast<int>(hidden_dim));
          } else {
            ok = false;
          }
        } else {
          if (hidden_dim == 128) {
            const dim3 grid_pair(static_cast<unsigned int>(blocks_vec), 64u, 1u);
            if (apply_path_tag == kApplyPathU8) {
              if (mask_u8_ptr == nullptr) {
                ok = false;
              } else {
                launch_apply_main_parity_h128<uint8_t>(
                    threads,
                    grid_pair,
                    lr5_probe_ptr,
                    mask_u8_ptr,
                    static_cast<int>(probe_rows_even),
                    static_cast<int>(probe_rows_total));
              }
            } else if (mask_i64_ptr != nullptr) {
              launch_apply_main_parity_h128<int64_t>(
                  threads,
                  grid_pair,
                  lr5_probe_ptr,
                  mask_i64_ptr,
                  static_cast<int>(probe_rows_even),
                  static_cast<int>(probe_rows_total));
            } else if (mask_f32_ptr != nullptr) {
              launch_apply_main_parity_h128<float>(
                  threads,
                  grid_pair,
                  lr5_probe_ptr,
                  mask_f32_ptr,
                  static_cast<int>(probe_rows_even),
                  static_cast<int>(probe_rows_total));
            } else {
              ok = false;
            }
          } else {
            const int d_pair_count = static_cast<int>((hidden_dim + 1) >> 1);
            if (d_pair_count > 0) {
              const dim3 grid_pair(
                  static_cast<unsigned int>(blocks_vec), static_cast<unsigned int>(d_pair_count), 1u);
              if (apply_path_tag == kApplyPathU8) {
                if (mask_u8_ptr == nullptr) {
                  ok = false;
                } else {
                  launch_apply_main_parity<uint8_t>(
                      threads,
                      grid_pair,
                      lr5_probe_ptr,
                      mask_u8_ptr,
                      static_cast<int>(probe_rows_even),
                      static_cast<int>(probe_rows_total),
                      static_cast<int>(hidden_dim));
                }
              } else if (mask_i64_ptr != nullptr) {
                launch_apply_main_parity<int64_t>(
                    threads,
                    grid_pair,
                    lr5_probe_ptr,
                    mask_i64_ptr,
                    static_cast<int>(probe_rows_even),
                    static_cast<int>(probe_rows_total),
                    static_cast<int>(hidden_dim));
              } else if (mask_f32_ptr != nullptr) {
                launch_apply_main_parity<float>(
                    threads,
                    grid_pair,
                    lr5_probe_ptr,
                    mask_f32_ptr,
                    static_cast<int>(probe_rows_even),
                    static_cast<int>(probe_rows_total),
                    static_cast<int>(hidden_dim));
              } else {
                ok = false;
              }
            }
          }
        }
      }

      if (ok) ok = cudaEventRecord(ev_end, 0) == cudaSuccess;
      if (ok) ok = cudaEventSynchronize(ev_end) == cudaSuccess;
      float elapsed = 0.0f;
      if (ok) ok = cudaEventElapsedTime(&elapsed, ev_begin, ev_end) == cudaSuccess;

      cudaEventDestroy(ev_begin);
      cudaEventDestroy(ev_end);

      if (!ok) {
        candidate_ok = false;
        break;
      }
      elapsed_total += elapsed;
      elapsed_count += 1;
    }

    if (!candidate_ok || elapsed_count <= 0) continue;
    const float elapsed_avg = elapsed_total / static_cast<float>(elapsed_count);

    if (first_two_count == 0) {
      first_two_gap_ratio = elapsed_avg;
      first_two_count = 1;
    } else if (first_two_count == 1) {
      if (first_two_gap_ratio > 0.0f && elapsed_avg > 0.0f) {
        const float a = first_two_gap_ratio;
        const float b = elapsed_avg;
        const float fast = a < b ? a : b;
        const float slow = a < b ? b : a;
        first_two_gap_ratio = (slow - fast) / fast;
      } else {
        first_two_gap_ratio = 0.0f;
      }
      first_two_count = 2;
    }

    has_valid = true;
    if (elapsed_avg < best_ms) {
      best_ms = elapsed_avg;
      best_threads = threads;
    }

    if (first_two_count == 2 && ci + 1 < candidate_count && early_stop_pct > 0) {
      const float threshold = static_cast<float>(early_stop_pct) * 0.01f;
      if (first_two_gap_ratio >= threshold) {
        break;
      }
    }
  }

  if (has_valid) {
    emplace_threads_shape_i64(stage_cache.apply_threads_by_shape, stage_cache, probe_key, best_threads);
    hot4_store_apply_threads(stage_cache, probe_key, best_threads);
    emplace_threads_shape_i64(stage_cache.apply_threads_probe_ms_by_shape, stage_cache, probe_key, best_ms);
    emplace_threads_shape_i64(stage_cache.apply_threads_probe_done_by_shape, stage_cache, probe_key, static_cast<uint8_t>(1));
    const int cooldown = read_threads_probe_cooldown_calls();
    if (cooldown > 0) {
      emplace_threads_shape_i64(
          stage_cache.apply_threads_next_probe_epoch_by_shape,
          stage_cache,
          probe_key,
          stage_cache.profile_epoch + cooldown);
    }
    return best_threads;
  }

  emplace_threads_shape_i64(stage_cache.apply_threads_by_shape, stage_cache, probe_key, fallback_threads);
  hot4_store_apply_threads(stage_cache, probe_key, fallback_threads);
  emplace_threads_shape_i64(stage_cache.apply_threads_probe_done_by_shape, stage_cache, probe_key, static_cast<uint8_t>(1));
  const int cooldown = read_threads_probe_cooldown_calls();
  if (cooldown > 0) {
    emplace_threads_shape_i64(
        stage_cache.apply_threads_next_probe_epoch_by_shape,
        stage_cache,
        probe_key,
        stage_cache.profile_epoch + cooldown);
  }
  return fallback_threads;
}

static inline int default_apply_reg_mode(
    int threads,
    int dim_bucket,
    int64_t rows,
    int apply_path_tag,
    bool rows_all_even) {
  if (threads == 128) {
    if (!rows_all_even) return kApplyRegModeLoose;
    if (apply_path_tag != kApplyPathU8 && apply_path_tag != kApplyPathI64Aligned &&
        apply_path_tag != kApplyPathI64Unaligned) {
      return kApplyRegModeLoose;
    }
    if (rows >= 900000) return kApplyRegModeTight;
    return kApplyRegModeLoose;
  }
  if (threads != 256 && threads != 192) return kApplyRegModeLoose;
  if (!rows_all_even) return kApplyRegModeLoose;
  if (apply_path_tag != kApplyPathU8 && apply_path_tag != kApplyPathI64Aligned &&
      apply_path_tag != kApplyPathI64Unaligned) {
    return kApplyRegModeLoose;
  }
  if (dim_bucket == 384) return kApplyRegModeTight;
  if (rows >= 589824) return kApplyRegModeTight;
  return kApplyRegModeLoose;
}

static inline int choose_apply_reg_mode_with_probe(
    StageTimingCache& stage_cache,
    uint64_t stage_key,
    int apply_path_tag,
    int dim_bucket,
    int64_t rows,
    int64_t rows_even,
    int64_t hidden_dim,
    int mask_dtype,
    bool rows_all_even,
    int threads,
    __half* lr5_ptr,
    __half* lr5_probe_ptr,
    const int64_t* mask_i64_ptr,
    const uint8_t* mask_u8_ptr,
    bool profile_this_call) {
  const uint64_t reg_key = apply_reg_shape_key(
      stage_key,
      rows,
      dim_bucket,
      hidden_dim,
      mask_dtype,
      rows_all_even,
      apply_path_tag,
      threads);
  int hot_reg_mode = 0;
  if (hot4_query_u64_i32(
          reg_key,
          stage_cache.apply_reg_hot_key,
          stage_cache.apply_reg_hot_val,
          stage_cache.apply_reg_hot_valid,
          &hot_reg_mode)) {
    return hot_reg_mode;
  }

  const int fallback = default_apply_reg_mode(threads, dim_bucket, rows, apply_path_tag, rows_all_even);
  if (!profile_this_call) {
    auto it_cached_fast = stage_cache.apply_reg_mode_by_shape.find(reg_key);
    if (it_cached_fast != stage_cache.apply_reg_mode_by_shape.end()) {
      hot4_store_apply_reg(stage_cache, reg_key, it_cached_fast->second);
      return it_cached_fast->second;
    }
    hot4_store_apply_reg(stage_cache, reg_key, fallback);
    return fallback;
  }

  auto it_next_probe_epoch = stage_cache.apply_reg_next_probe_epoch_by_shape.find(reg_key);
  if (it_next_probe_epoch != stage_cache.apply_reg_next_probe_epoch_by_shape.end() &&
      stage_cache.profile_epoch < it_next_probe_epoch->second) {
    auto it_cached_fast = stage_cache.apply_reg_mode_by_shape.find(reg_key);
    if (it_cached_fast != stage_cache.apply_reg_mode_by_shape.end()) {
      hot4_store_apply_reg(stage_cache, reg_key, it_cached_fast->second);
      return it_cached_fast->second;
    }
    hot4_store_apply_reg(stage_cache, reg_key, fallback);
    return fallback;
  }

  auto it_cached = stage_cache.apply_reg_mode_by_shape.find(reg_key);
  if (it_cached != stage_cache.apply_reg_mode_by_shape.end()) {
    hot4_store_apply_reg(stage_cache, reg_key, it_cached->second);
    return it_cached->second;
  }

  if (!profile_this_call || lr5_ptr == nullptr || lr5_probe_ptr == nullptr || rows_even <= 0 || hidden_dim <= 0) {
    emplace_reg_shape_i64(stage_cache.apply_reg_mode_by_shape, stage_cache, reg_key, fallback);
    hot4_store_apply_reg(stage_cache, reg_key, fallback);
    return fallback;
  }
  if ((threads != 256 && threads != 192 && threads != 128) || !rows_all_even) {
    emplace_reg_shape_i64(stage_cache.apply_reg_mode_by_shape, stage_cache, reg_key, fallback);
    hot4_store_apply_reg(stage_cache, reg_key, fallback);
    return fallback;
  }
  if (apply_path_tag == kApplyPathU8 && mask_u8_ptr == nullptr) {
    emplace_reg_shape_i64(stage_cache.apply_reg_mode_by_shape, stage_cache, reg_key, fallback);
    hot4_store_apply_reg(stage_cache, reg_key, fallback);
    return fallback;
  }
  if ((apply_path_tag == kApplyPathI64Aligned || apply_path_tag == kApplyPathI64Unaligned) && mask_i64_ptr == nullptr) {
    emplace_reg_shape_i64(stage_cache.apply_reg_mode_by_shape, stage_cache, reg_key, fallback);
    hot4_store_apply_reg(stage_cache, reg_key, fallback);
    return fallback;
  }

  auto it_probe_done = stage_cache.apply_reg_probe_done_by_shape.find(reg_key);
  if (it_probe_done != stage_cache.apply_reg_probe_done_by_shape.end() && it_probe_done->second != 0) {
    emplace_reg_shape_i64(stage_cache.apply_reg_mode_by_shape, stage_cache, reg_key, fallback);
    hot4_store_apply_reg(stage_cache, reg_key, fallback);
    return fallback;
  }

  const int probe_rows_even = choose_reg_probe_window_rows_even(rows_even);
  if (probe_rows_even <= 0) {
    emplace_reg_shape_i64(stage_cache.apply_reg_mode_by_shape, stage_cache, reg_key, fallback);
    hot4_store_apply_reg(stage_cache, reg_key, fallback);
    return fallback;
  }
  emplace_reg_shape_i64(stage_cache.apply_reg_probe_window_by_shape, stage_cache, reg_key, probe_rows_even);

  const int blocks_vec = static_cast<int>(
      (probe_rows_even + static_cast<int64_t>(threads) * 4 - 1) / (static_cast<int64_t>(threads) * 4));
  if (blocks_vec <= 0) {
    emplace_reg_shape_i64(stage_cache.apply_reg_mode_by_shape, stage_cache, reg_key, fallback);
    hot4_store_apply_reg(stage_cache, reg_key, fallback);
    return fallback;
  }
  const dim3 grid_vec(static_cast<unsigned int>(blocks_vec), static_cast<unsigned int>(hidden_dim), 1u);
  const size_t probe_bytes = static_cast<size_t>(probe_rows_even) * static_cast<size_t>(hidden_dim) *
                             static_cast<size_t>(5) * sizeof(__half);
  const uintptr_t reg_probe_base = reinterpret_cast<uintptr_t>(lr5_probe_ptr);
  const uintptr_t reg_probe_stride =
      static_cast<uintptr_t>(static_cast<size_t>(probe_rows_even) * static_cast<size_t>(hidden_dim) * sizeof(__half));
  const uintptr_t reg_probe_right = reg_probe_base + reg_probe_stride;
  const uintptr_t reg_probe_gl = reg_probe_right + reg_probe_stride;
  const uintptr_t reg_probe_gr = reg_probe_gl + reg_probe_stride;
  const bool probe_vec_aligned =
      ((probe_rows_even & 3LL) == 0) && ((reg_probe_base & 0x7) == 0) && ((reg_probe_right & 0x7) == 0) &&
      ((reg_probe_gl & 0x7) == 0) && ((reg_probe_gr & 0x7) == 0);

  const int candidates[2] = {kApplyRegModeTight, kApplyRegModeLoose};
  int best_mode = fallback;
  float best_ms = std::numeric_limits<float>::infinity();
  bool found = false;

  for (int ci = 0; ci < 2; ++ci) {
    const int reg_mode = candidates[ci];
    cudaEvent_t ev_begin = nullptr;
    cudaEvent_t ev_end = nullptr;
    if (cudaEventCreateWithFlags(&ev_begin, cudaEventDefault) != cudaSuccess) {
      if (ev_begin != nullptr) cudaEventDestroy(ev_begin);
      continue;
    }
    if (cudaEventCreateWithFlags(&ev_end, cudaEventDefault) != cudaSuccess) {
      cudaEventDestroy(ev_begin);
      if (ev_end != nullptr) cudaEventDestroy(ev_end);
      continue;
    }

    bool ok = cudaEventRecord(ev_begin, 0) == cudaSuccess;
    if (ok) ok = cudaMemcpyAsync(lr5_probe_ptr, lr5_ptr, probe_bytes, cudaMemcpyDeviceToDevice, 0) == cudaSuccess;
    if (ok) {
      if (apply_path_tag == kApplyPathU8) {
        launch_apply_rows_even_u8(
            threads,
            grid_vec,
            lr5_probe_ptr,
            mask_u8_ptr,
            static_cast<int>(probe_rows_even),
            static_cast<int>(hidden_dim),
            reg_mode,
            probe_vec_aligned,
            0);
      } else if (apply_path_tag == kApplyPathI64Aligned) {
        launch_apply_rows_even_i64_aligned(
            threads,
            grid_vec,
            lr5_probe_ptr,
            mask_i64_ptr,
            static_cast<int>(probe_rows_even),
            static_cast<int>(hidden_dim),
            reg_mode,
            probe_vec_aligned);
      } else if (apply_path_tag == kApplyPathI64Unaligned) {
        launch_apply_rows_even_i64_unaligned(
            threads,
            grid_vec,
            lr5_probe_ptr,
            mask_i64_ptr,
            static_cast<int>(probe_rows_even),
            static_cast<int>(hidden_dim),
            reg_mode,
            probe_vec_aligned);
      } else {
        ok = false;
      }
    }

    if (ok) ok = cudaEventRecord(ev_end, 0) == cudaSuccess;
    if (ok) ok = cudaEventSynchronize(ev_end) == cudaSuccess;
    float elapsed = 0.0f;
    if (ok) ok = cudaEventElapsedTime(&elapsed, ev_begin, ev_end) == cudaSuccess;

    cudaEventDestroy(ev_begin);
    cudaEventDestroy(ev_end);

    if (!ok) continue;
    found = true;
    if (elapsed < best_ms) {
      best_ms = elapsed;
      best_mode = reg_mode;
    }
  }

  emplace_reg_shape_i64(stage_cache.apply_reg_probe_done_by_shape, stage_cache, reg_key, static_cast<uint8_t>(1));
  if (found) {
    emplace_reg_shape_i64(stage_cache.apply_reg_mode_by_shape, stage_cache, reg_key, best_mode);
    hot4_store_apply_reg(stage_cache, reg_key, best_mode);
    emplace_reg_shape_i64(stage_cache.apply_reg_probe_ms_by_shape, stage_cache, reg_key, best_ms);
    const int cooldown = read_reg_probe_cooldown_calls();
    if (cooldown > 0) {
      emplace_reg_shape_i64(
          stage_cache.apply_reg_next_probe_epoch_by_shape,
          stage_cache,
          reg_key,
          stage_cache.profile_epoch + cooldown);
    }
    return best_mode;
  }
  emplace_reg_shape_i64(stage_cache.apply_reg_mode_by_shape, stage_cache, reg_key, fallback);
  hot4_store_apply_reg(stage_cache, reg_key, fallback);
  const int cooldown = read_reg_probe_cooldown_calls();
  if (cooldown > 0) {
    emplace_reg_shape_i64(
        stage_cache.apply_reg_next_probe_epoch_by_shape,
        stage_cache,
        reg_key,
        stage_cache.profile_epoch + cooldown);
  }
  return fallback;
}

static inline void init_stage_timing_cfg(StageTimingCache& cache) {
  if (cache.cfg_ready) return;
  cache.tune_enabled = read_bool_env("TRIMUL_TUNE");
  cache.enabled = cache.tune_enabled && read_bool_env("TRIMUL_STAGE_PROFILE");
  cache.allow_learning = cache.enabled && read_bool_env("TRIMUL_STAGE_LEARN");
  cache.remain_samples = cache.enabled ? 1 : 0;
  cache.cfg_ready = true;
}

static bool try_batched_algo_once(
    cublasHandle_t handle,
    const __half* a_rm,
    const __half* b_rm,
    __half* c_rm,
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t batch_count,
    int64_t stride_a,
    int64_t stride_b,
    int64_t stride_c,
    cublasGemmAlgo_t algo,
    float* elapsed_ms) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  const int mm = static_cast<int>(n);
  const int nn = static_cast<int>(m);
  const int kk = static_cast<int>(k);

  cudaEvent_t ev_begin = nullptr;
  cudaEvent_t ev_end = nullptr;
  if (cudaEventCreateWithFlags(&ev_begin, cudaEventDefault) != cudaSuccess) return false;
  if (cudaEventCreateWithFlags(&ev_end, cudaEventDefault) != cudaSuccess) {
    cudaEventDestroy(ev_begin);
    return false;
  }

  bool ok = true;
  if (cudaEventRecord(ev_begin, 0) != cudaSuccess) ok = false;
  if (ok) {
    const cublasStatus_t st = cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        mm,
        nn,
        kk,
        &alpha,
        (const void*)b_rm,
        CUDA_R_16F,
        kk,
        static_cast<long long>(stride_b),
        (const void*)a_rm,
        CUDA_R_16F,
        kk,
        static_cast<long long>(stride_a),
        &beta,
        (void*)c_rm,
        CUDA_R_16F,
        mm,
        static_cast<long long>(stride_c),
        static_cast<int>(batch_count),
        kGemmCompute,
        algo);
    if (st != CUBLAS_STATUS_SUCCESS) ok = false;
  }

  float ms = std::numeric_limits<float>::infinity();
  if (ok && cudaEventRecord(ev_end, 0) != cudaSuccess) ok = false;
  if (ok && cudaEventSynchronize(ev_end) != cudaSuccess) ok = false;
  if (ok && cudaEventElapsedTime(&ms, ev_begin, ev_end) != cudaSuccess) ok = false;

  cudaEventDestroy(ev_begin);
  cudaEventDestroy(ev_end);
  if (!ok) return false;
  if (elapsed_ms != nullptr) *elapsed_ms = ms;
  return true;
}

static cublasGemmAlgo_t select_batched_algo(
    cublasHandle_t handle,
    CublasBatchedAlgoCache& cache,
    uint64_t shape_key,
    const __half* a_rm,
    const __half* b_rm,
    __half* c_rm,
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t bs,
    int64_t dim,
    int64_t hidden_dim,
    int64_t batch_count,
    int64_t stride_a,
    int64_t stride_b,
    int64_t stride_c) {
  constexpr cublasGemmAlgo_t kDefaultAlgo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  init_batched_algo_cfg(cache);
  const int dim_idx = dim_bucket_index(dim);

  cublasGemmAlgo_t static_algo = kDefaultAlgo;
  const bool has_static = select_static_ranked_algo(n, hidden_dim, bs, dim, stride_a, &static_algo);

  const bool hot_dim384 = (bs == 1 && n == 1024 && dim == 384 && hidden_dim == 128);
  const bool hot_dim128 = (bs == 1 && n == 1024 && dim == 128 && hidden_dim == 128);
  const bool hot_shape = hot_dim384 || hot_dim128;

  cublasGemmAlgo_t hot_cached = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  if (hot4_query_batched_algo(cache, shape_key, &hot_cached)) return hot_cached;

  auto it = cache.algo_by_shape.find(shape_key);
  if (it != cache.algo_by_shape.end()) {
    hot4_store_batched_algo(cache, shape_key, it->second);
    return it->second;
  }

  if (!cache.search_enabled && (!hot_shape || !cache.tune_hot_enabled)) {
    const cublasGemmAlgo_t picked = has_static ? static_algo : kDefaultAlgo;
    emplace_batched_shape(cache.algo_by_shape, cache, shape_key, picked);
    hot4_store_batched_algo(cache, shape_key, picked);
    cache.hot_algo_by_dim_bucket[dim_idx] = picked;
    cache.hot_algo_by_dim_bucket_valid[dim_idx] = 1;
    return picked;
  }

  if (hot_shape && cache.tune_hot_enabled) {
    auto it_probe = cache.algo_probe_done_by_shape.find(shape_key);
    if (it_probe != cache.algo_probe_done_by_shape.end() && it_probe->second != 0) {
      auto it_hot = cache.algo_by_shape.find(shape_key);
      if (it_hot != cache.algo_by_shape.end()) {
        hot4_store_batched_algo(cache, shape_key, it_hot->second);
        return it_hot->second;
      }
    }

    cublasGemmAlgo_t hot_candidates[5];
    int hot_candidate_count = 0;
    const bool has_dim_hot = cache.hot_algo_by_dim_bucket_valid[dim_idx] != 0;
    const cublasGemmAlgo_t dim_hot_algo = cache.hot_algo_by_dim_bucket[dim_idx];
    if (has_dim_hot) {
      hot_candidates[hot_candidate_count++] = dim_hot_algo;
    }
    if (hot_dim384) {
      hot_candidates[hot_candidate_count++] = CUBLAS_GEMM_ALGO1_TENSOR_OP;
      hot_candidates[hot_candidate_count++] = CUBLAS_GEMM_ALGO0_TENSOR_OP;
      hot_candidates[hot_candidate_count++] = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    } else {
      hot_candidates[hot_candidate_count++] = CUBLAS_GEMM_ALGO1_TENSOR_OP;
      hot_candidates[hot_candidate_count++] = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
      hot_candidates[hot_candidate_count++] = CUBLAS_GEMM_ALGO0_TENSOR_OP;
    }
    if (has_static) {
      bool seen_static = false;
      for (int i = 0; i < hot_candidate_count; ++i) {
        if (hot_candidates[i] == static_algo) {
          seen_static = true;
          break;
        }
      }
      if (!seen_static && hot_candidate_count < 5) {
        hot_candidates[hot_candidate_count++] = static_algo;
      }
    }

    if (hot_dim384 && hot_candidate_count > 2) {
      hot_candidate_count = 2;
    }

    cublasGemmAlgo_t best_algo = has_static ? static_algo : kDefaultAlgo;
    float best_ms = std::numeric_limits<float>::infinity();
    bool found = false;
    for (int candidate_idx = 0; candidate_idx < hot_candidate_count; ++candidate_idx) {
      const cublasGemmAlgo_t candidate = hot_candidates[candidate_idx];
      float ms = 0.0f;
      if (try_batched_algo_once(
              handle,
              a_rm,
              b_rm,
              c_rm,
              m,
              k,
              n,
              batch_count,
              stride_a,
              stride_b,
              stride_c,
              candidate,
              &ms)) {
        if (!found || ms < best_ms) {
          found = true;
          best_ms = ms;
          best_algo = candidate;
        }
      }
    }
    if (found) {
      emplace_batched_shape(cache.algo_by_shape, cache, shape_key, best_algo);
      emplace_batched_shape(cache.algo_probe_ms_by_shape, cache, shape_key, best_ms);
      emplace_batched_shape(cache.algo_probe_done_by_shape, cache, shape_key, static_cast<uint8_t>(1));
      cache.hot_algo_by_dim_bucket[dim_idx] = best_algo;
      cache.hot_algo_by_dim_bucket_valid[dim_idx] = 1;
      hot4_store_batched_algo(cache, shape_key, best_algo);
      return best_algo;
    }
    const cublasGemmAlgo_t picked = has_static ? static_algo : kDefaultAlgo;
    emplace_batched_shape(cache.algo_by_shape, cache, shape_key, picked);
    emplace_batched_shape(cache.algo_probe_done_by_shape, cache, shape_key, static_cast<uint8_t>(1));
    cache.hot_algo_by_dim_bucket[dim_idx] = picked;
    cache.hot_algo_by_dim_bucket_valid[dim_idx] = 1;
    hot4_store_batched_algo(cache, shape_key, picked);
    return picked;
  }

  if (!cache.search_enabled) {
    const cublasGemmAlgo_t picked = has_static ? static_algo : kDefaultAlgo;
    emplace_batched_shape(cache.algo_by_shape, cache, shape_key, picked);
    cache.hot_algo_by_dim_bucket[dim_idx] = picked;
    cache.hot_algo_by_dim_bucket_valid[dim_idx] = 1;
    return picked;
  }

  const cublasGemmAlgo_t candidates[] = {
      CUBLAS_GEMM_DEFAULT_TENSOR_OP,
      CUBLAS_GEMM_ALGO0_TENSOR_OP,
      CUBLAS_GEMM_ALGO1_TENSOR_OP,
  };

  cublasGemmAlgo_t best_algo = has_static ? static_algo : kDefaultAlgo;
  float best_ms = std::numeric_limits<float>::infinity();
  bool found = false;

  if (has_static) {
    float ms_static = 0.0f;
    if (try_batched_algo_once(
            handle,
            a_rm,
            b_rm,
            c_rm,
            m,
            k,
            n,
            batch_count,
            stride_a,
            stride_b,
            stride_c,
            static_algo,
            &ms_static)) {
      best_ms = ms_static;
      found = true;
    }
  }

  for (const cublasGemmAlgo_t candidate : candidates) {
    if (has_static && candidate == static_algo) continue;
    float ms = 0.0f;
    if (try_batched_algo_once(
            handle,
            a_rm,
            b_rm,
            c_rm,
            m,
            k,
            n,
            batch_count,
            stride_a,
            stride_b,
            stride_c,
            candidate,
            &ms)) {
      if (!found || ms < best_ms) {
        best_ms = ms;
        best_algo = candidate;
        found = true;
      }
    }
  }

  emplace_batched_shape(cache.algo_by_shape, cache, shape_key, best_algo);
  cache.hot_algo_by_dim_bucket[dim_idx] = best_algo;
  cache.hot_algo_by_dim_bucket_valid[dim_idx] = 1;
  hot4_store_batched_algo(cache, shape_key, best_algo);
  return best_algo;
}

torch::Tensor trimul_forward(
    torch::Tensor x,
    torch::Tensor mask,
    torch::Tensor norm_weight,
    torch::Tensor norm_bias,
    torch::Tensor left_proj_weight,
    torch::Tensor right_proj_weight,
    torch::Tensor left_gate_weight,
    torch::Tensor right_gate_weight,
    torch::Tensor out_gate_weight,
    torch::Tensor to_out_norm_weight,
    torch::Tensor to_out_norm_bias,
    torch::Tensor to_out_weight) {
  TORCH_CHECK(x.is_cuda(), "x must be cuda");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
  TORCH_CHECK(mask.is_cuda(), "mask must be cuda");

  const int64_t bs = x.size(0);
  const int64_t n = x.size(1);
  const int64_t dim = x.size(3);
  const int64_t rows = bs * n * n;
  const int64_t hidden_dim = left_proj_weight.size(0);
  TORCH_CHECK(mask.numel() == rows, "mask shape mismatch");

  const int device = x.get_device();
  DeviceCaches& caches = get_device_caches(device);
  WorkspaceCache& ws_cache = caches.workspace;
  PackedWeightsCache& packed_cache = caches.packed;
  StageTimingCache& stage_timing = caches.stage_timing;
  init_stage_timing_cfg(stage_timing);

  const auto opts_f16 = x.options().dtype(torch::kFloat16);
  const auto opts_u8 = x.options().dtype(torch::kUInt8);

  const bool ws_hit = ws_cache.xhat.defined() && ws_cache.bs == bs && ws_cache.n == n && ws_cache.dim == dim &&
                      ws_cache.hidden_dim == hidden_dim && ws_cache.rows == rows;
  if (!ws_hit) {
    ws_cache.xhat = torch::empty({bs, n, n, dim}, opts_f16);
    ws_cache.lr5 = torch::empty({5 * hidden_dim, bs, n, n}, opts_f16);
    ws_cache.lr5_probe = torch::empty({5 * hidden_dim, bs, n, n}, opts_f16);
    ws_cache.out_tmp = torch::empty({hidden_dim, bs, n, n}, opts_f16);
    ws_cache.out_hidden = torch::empty({rows, hidden_dim}, opts_f16);
    ws_cache.mask_u8 = torch::empty({rows}, opts_u8);
    ws_cache.mask_hist16 = torch::zeros({16}, x.options().dtype(torch::kInt));
    ws_cache.mask_ptr = nullptr;
    ws_cache.mask_ver = -1;
    ws_cache.mask_rows = -1;
    ws_cache.mask_dtype = -1;
    ws_cache.mask_align_ptr = nullptr;
    ws_cache.mask_align_ver = -1;
    ws_cache.mask_align_rows = -1;
    ws_cache.mask_align_dtype = -1;
    ws_cache.mask_align_hit = 0;
    ws_cache.bs = bs;
    ws_cache.n = n;
    ws_cache.dim = dim;
    ws_cache.hidden_dim = hidden_dim;
    ws_cache.rows = rows;
  }

  if (!ws_cache.lr5_probe.defined() || ws_cache.lr5_probe.numel() != ws_cache.lr5.numel()) {
    ws_cache.lr5_probe = torch::empty({5 * hidden_dim, bs, n, n}, opts_f16);
  }

  if (!ws_cache.mask_u8.defined() || ws_cache.mask_u8.numel() != rows) {
    ws_cache.mask_u8 = torch::empty({rows}, opts_u8);
    ws_cache.mask_ptr = nullptr;
    ws_cache.mask_ver = -1;
    ws_cache.mask_rows = -1;
    ws_cache.mask_dtype = -1;
    ws_cache.mask_align_ptr = nullptr;
    ws_cache.mask_align_ver = -1;
    ws_cache.mask_align_rows = -1;
    ws_cache.mask_align_dtype = -1;
    ws_cache.mask_align_hit = 0;
  }

  if (!ws_cache.mask_hist16.defined() || ws_cache.mask_hist16.numel() != 16) {
    ws_cache.mask_hist16 = torch::zeros({16}, x.options().dtype(torch::kInt));
  }

  const bool profile_this_call = stage_timing.tune_enabled && stage_timing.enabled && stage_timing.remain_samples > 0;
  cudaEvent_t ev_ln_begin = nullptr;
  cudaEvent_t ev_ln_end = nullptr;
  cudaEvent_t ev_apply_begin = nullptr;
  cudaEvent_t ev_apply_end = nullptr;
  cudaEvent_t ev_batched_begin = nullptr;
  cudaEvent_t ev_batched_end = nullptr;
  cudaEvent_t ev_final_begin = nullptr;
  cudaEvent_t ev_final_end = nullptr;
  if (profile_this_call) {
    stage_timing.profile_epoch += 1;
    cudaEventCreateWithFlags(&ev_ln_begin, cudaEventDefault);
    cudaEventCreateWithFlags(&ev_ln_end, cudaEventDefault);
    cudaEventCreateWithFlags(&ev_apply_begin, cudaEventDefault);
    cudaEventCreateWithFlags(&ev_apply_end, cudaEventDefault);
    cudaEventCreateWithFlags(&ev_batched_begin, cudaEventDefault);
    cudaEventCreateWithFlags(&ev_batched_end, cudaEventDefault);
    cudaEventCreateWithFlags(&ev_final_begin, cudaEventDefault);
    cudaEventCreateWithFlags(&ev_final_end, cudaEventDefault);
    if (ev_ln_begin != nullptr) cudaEventRecord(ev_ln_begin, 0);
  }

  auto xhat = ws_cache.xhat;
  const int warps = 8;
  const int threads_ln = warps * 32;
  const int blocks_ln = static_cast<int>((rows + warps - 1) / warps);
  const int mask_dtype = static_cast<int>(mask.scalar_type());
  const uint64_t stage_key = stage_shape_key(bs, n, dim, hidden_dim, mask_dtype);
  const int dim_bucket = apply_dim_bucket(dim);
  int learned_case_mode = 0;
  if (profile_this_call && stage_timing.allow_learning) {
    int hist_case_mode = 0;
    int hist_best_count = 0;
    const int hist_candidates[5] = {0x3, 0x1, 0x2, 0x4, 0x8};
    for (int i = 0; i < 5; ++i) {
      const uint64_t hist_key = mask_bits_hist_shape_key(stage_key, hist_candidates[i]);
      const int c = map_get_or(stage_timing.apply_mask_bits_hist_by_shape, hist_key, static_cast<uint16_t>(0));
      if (c > hist_best_count) {
        hist_best_count = c;
        if (hist_candidates[i] == 0x3) {
          hist_case_mode = 1;
        } else if (hist_candidates[i] == 0x8) {
          hist_case_mode = 2;
        } else {
          hist_case_mode = 0;
        }
      }
    }
    learned_case_mode = hist_case_mode;
    emplace_stage_shape_i64(
        stage_timing.apply_case_mode_by_shape,
        stage_timing,
        stage_key,
        static_cast<uint8_t>(hist_case_mode));
    if (stage_timing.apply_mask_case_mode_last != hist_case_mode) {
      cudaMemcpyToSymbol(g_apply_mask_case_mode, &hist_case_mode, sizeof(int), 0, cudaMemcpyHostToDevice);
      stage_timing.apply_mask_case_mode_last = hist_case_mode;
    }
  } else {
    learned_case_mode = map_get_or(stage_timing.apply_case_mode_by_shape, stage_key, static_cast<uint8_t>(0));
    if (learned_case_mode < 0 || learned_case_mode > 2) learned_case_mode = 0;
    stage_timing.apply_mask_case_mode_last = learned_case_mode;
  }
  const bool gate_host_stats = stage_timing.allow_learning && profile_this_call;
  bool used_u8_path = false;
  bool rows_all_even_this_call = false;
  int selected_mask_dtype_this_call = mask_dtype;
  bool i64_aligned_decision_final = false;
  bool mask_align_hit_this_call = false;
  int apply_path_tag_this_call = kApplyPathUnknown;
  int apply_threads_this_call = -1;
  int apply_reg_mode_this_call = kApplyRegModeLoose;
  bool has_tail_this_call = false;
  float odd_tail_ms_this_call = -1.0f;
  if (dim == 128) {
    ln_warp_affine_to_f16_kernel<128><<<blocks_ln, threads_ln>>>(
        x.data_ptr<float>(),
        norm_weight.data_ptr<float>(),
        norm_bias.data_ptr<float>(),
        reinterpret_cast<__half*>(xhat.data_ptr<at::Half>()),
        rows);
  } else if (dim == 384) {
    ln_warp_affine_to_f16_kernel<384><<<blocks_ln, threads_ln>>>(
        x.data_ptr<float>(),
        norm_weight.data_ptr<float>(),
        norm_bias.data_ptr<float>(),
        reinterpret_cast<__half*>(xhat.data_ptr<at::Half>()),
        rows);
  } else {
    ln_warp_affine_to_f16_generic_kernel<<<blocks_ln, threads_ln>>>(
        x.data_ptr<float>(),
        norm_weight.data_ptr<float>(),
        norm_bias.data_ptr<float>(),
        reinterpret_cast<__half*>(xhat.data_ptr<at::Half>()),
        rows,
        static_cast<int>(dim));
  }

  if (profile_this_call) {
    if (ev_ln_end != nullptr) cudaEventRecord(ev_ln_end, 0);
    if (ev_apply_begin != nullptr) cudaEventRecord(ev_apply_begin, 0);
  }

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

  const int64_t seg_elems = hidden_dim * dim;
  const void* p0 = left_proj_weight.data_ptr<float>();
  const void* p1 = right_proj_weight.data_ptr<float>();
  const void* p2 = left_gate_weight.data_ptr<float>();
  const void* p3 = right_gate_weight.data_ptr<float>();
  const void* p4 = out_gate_weight.data_ptr<float>();
  const void* p5 = to_out_weight.data_ptr<float>();
  const int64_t v0 = safe_tensor_version(left_proj_weight);
  const int64_t v1 = safe_tensor_version(right_proj_weight);
  const int64_t v2 = safe_tensor_version(left_gate_weight);
  const int64_t v3 = safe_tensor_version(right_gate_weight);
  const int64_t v4 = safe_tensor_version(out_gate_weight);
  const int64_t v5 = safe_tensor_version(to_out_weight);
  const bool packed_hit = packed_cache.wbuf.defined() && packed_cache.seg_elems == seg_elems && packed_cache.p0 == p0 &&
                          packed_cache.p1 == p1 && packed_cache.p2 == p2 && packed_cache.p3 == p3 &&
                          packed_cache.p4 == p4 && packed_cache.p5 == p5 && packed_cache.v0 == v0 &&
                          packed_cache.v1 == v1 && packed_cache.v2 == v2 && packed_cache.v3 == v3 &&
                          packed_cache.v4 == v4 && packed_cache.v5 == v5;
  if (!packed_hit) {
    packed_cache.wbuf = torch::empty({seg_elems * 6}, opts_f16);
    const int threads = 256;
    const int64_t total = seg_elems * 6;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    pack6_f32_to_f16_kernel<<<blocks, threads>>>(
        left_proj_weight.data_ptr<float>(),
        right_proj_weight.data_ptr<float>(),
        left_gate_weight.data_ptr<float>(),
        right_gate_weight.data_ptr<float>(),
        out_gate_weight.data_ptr<float>(),
        to_out_weight.data_ptr<float>(),
        reinterpret_cast<__half*>(packed_cache.wbuf.data_ptr<at::Half>()),
        seg_elems);
    packed_cache.p0 = p0;
    packed_cache.p1 = p1;
    packed_cache.p2 = p2;
    packed_cache.p3 = p3;
    packed_cache.p4 = p4;
    packed_cache.p5 = p5;
    packed_cache.v0 = v0;
    packed_cache.v1 = v1;
    packed_cache.v2 = v2;
    packed_cache.v3 = v3;
    packed_cache.v4 = v4;
    packed_cache.v5 = v5;
    packed_cache.seg_elems = seg_elems;
  }

  auto wbuf = packed_cache.wbuf;

  auto lr5 = ws_cache.lr5;
  const __half* w5_ptr = reinterpret_cast<const __half*>(wbuf.data_ptr<at::Half>());
  gemm_f16_abt(
      handle,
      w5_ptr,
      reinterpret_cast<const __half*>(xhat.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(lr5.data_ptr<at::Half>()),
      5 * hidden_dim,
      dim,
      rows);

  {
    const int rows_even = static_cast<int>(rows & ~1LL);
    const bool has_tail = (rows & 1LL) != 0;
    has_tail_this_call = has_tail;
    const bool rows_all_even = rows_even == rows;
    const bool i64_align_hit =
        rows_all_even && mask_dtype == static_cast<int>(torch::kInt64)
            ? query_mask_align16_cached(ws_cache, mask, mask_dtype, rows)
            : false;
    const int64_t weighted_rows = rows * hidden_dim;
    const bool force_even_u8 =
        rows_all_even && mask_dtype != static_cast<int>(torch::kUInt8) && weighted_rows >= 8000000LL;
    const bool force_align_u8 = read_bool_env("TRIMUL_ALIGN_I64_FORCE_U8");
    const bool force_align_raw = read_bool_env("TRIMUL_ALIGN_I64_FORCE_RAW");
    rows_all_even_this_call = rows_all_even;
    bool use_u8_path = force_even_u8 ||
                       should_compact_mask_to_u8(rows, hidden_dim, mask_dtype, rows_all_even, &stage_timing, stage_key);
    if (rows_all_even && mask_dtype == static_cast<int>(torch::kInt64) && i64_align_hit) {
      i64_aligned_decision_final = true;
      if (force_align_u8 && force_align_raw && !stage_timing.align_force_conflict_warned) {
        std::printf("[trimul-warn] both TRIMUL_ALIGN_I64_FORCE_U8 and TRIMUL_ALIGN_I64_FORCE_RAW are set; using shape memo/default\n");
        stage_timing.align_force_conflict_warned = true;
      }
      if (force_align_u8 && !force_align_raw) {
        use_u8_path = true;
      } else if (force_align_raw && !force_align_u8) {
        use_u8_path = false;
      } else {
        bool has_memo = false;
        const int path_bucket_key = apply_path_bucket(kApplyPathI64Aligned);
        const uint64_t i64_memo_key = apply_threads_shape_key(
            stage_key,
            rows,
            dim_bucket,
            hidden_dim,
            mask_dtype,
            rows_all_even_this_call,
            path_bucket_key);
        auto it_decision = stage_timing.i64_aligned_decision_ready_by_shape.find(i64_memo_key);
        if (it_decision != stage_timing.i64_aligned_decision_ready_by_shape.end() && it_decision->second != 0) {
          auto it_use_u8 = stage_timing.i64_aligned_use_u8_by_shape.find(i64_memo_key);
          if (it_use_u8 != stage_timing.i64_aligned_use_u8_by_shape.end()) {
            use_u8_path = it_use_u8->second != 0;
            has_memo = true;
          }
        }
        if (!has_memo) {
          uint8_t confirm_state = 0;
          auto it_confirm = stage_timing.i64_aligned_confirm_state_by_shape.find(i64_memo_key);
          if (it_confirm != stage_timing.i64_aligned_confirm_state_by_shape.end()) {
            confirm_state = it_confirm->second;
          }
          const int aligned_u8_count =
              map_get_or(stage_timing.apply_i64_aligned_u8_count_by_shape, stage_key, 0);
          const int aligned_raw_count =
              map_get_or(stage_timing.apply_i64_aligned_raw_count_by_shape, stage_key, 0);
          const uint8_t* pair_mask_u8_ptr = nullptr;
          if (profile_this_call && ws_cache.lr5_probe.defined()) {
            pair_mask_u8_ptr = ensure_mask_u8_cache(ws_cache, mask, mask_dtype, rows);
          }
          if (confirm_state == 0 && profile_this_call && ws_cache.lr5_probe.defined()) {
            uint8_t pair_done = 0;
            auto it_pair_done = stage_timing.i64_aligned_pair_probe_done_by_shape.find(i64_memo_key);
            if (it_pair_done != stage_timing.i64_aligned_pair_probe_done_by_shape.end()) {
              pair_done = it_pair_done->second;
            }
            if (pair_done == 0) {
              const int pair_threads = 128;
              const int pair_reg_mode = kApplyRegModeLoose;
              const int pair_rows_even = choose_probe_window_rows_even(rows_even, true);
              if (pair_rows_even > 0) {
                const int blocks_pair = static_cast<int>(
                    (static_cast<int64_t>(pair_rows_even) + static_cast<int64_t>(pair_threads) * 4 - 1) /
                    (static_cast<int64_t>(pair_threads) * 4));
                if (blocks_pair > 0) {
                  const dim3 grid_pair(
                      static_cast<unsigned int>(blocks_pair), static_cast<unsigned int>(hidden_dim), 1u);
                  const size_t pair_rows_total = static_cast<size_t>(rows);
                  const size_t pair_bytes =
                      static_cast<size_t>(pair_rows_even) * static_cast<size_t>(hidden_dim) * static_cast<size_t>(5) *
                      sizeof(__half);
                  const uintptr_t pair_base_ptr = reinterpret_cast<uintptr_t>(ws_cache.lr5_probe.data_ptr<at::Half>());
                  const uintptr_t pair_plane_stride =
                      static_cast<uintptr_t>(pair_rows_total * static_cast<size_t>(hidden_dim) * sizeof(__half));
                  const uintptr_t pair_right_ptr = pair_base_ptr + pair_plane_stride;
                  const uintptr_t pair_gl_ptr = pair_right_ptr + pair_plane_stride;
                  const uintptr_t pair_gr_ptr = pair_gl_ptr + pair_plane_stride;
                  const bool pair_vec_aligned =
                      ((pair_rows_even & 3) == 0) && ((pair_base_ptr & 0x7) == 0) && ((pair_right_ptr & 0x7) == 0) &&
                      ((pair_gl_ptr & 0x7) == 0) && ((pair_gr_ptr & 0x7) == 0);
                  cudaEvent_t ev_u8_beg = nullptr;
                  cudaEvent_t ev_u8_end = nullptr;
                  cudaEvent_t ev_raw_beg = nullptr;
                  cudaEvent_t ev_raw_end = nullptr;
                  bool pair_ok = true;
                  float pair_u8_ms = std::numeric_limits<float>::infinity();
                  float pair_raw_ms = std::numeric_limits<float>::infinity();
                  if (cudaEventCreateWithFlags(&ev_u8_beg, cudaEventDefault) != cudaSuccess) pair_ok = false;
                  if (pair_ok && cudaEventCreateWithFlags(&ev_u8_end, cudaEventDefault) != cudaSuccess) pair_ok = false;
                  if (pair_ok && cudaEventCreateWithFlags(&ev_raw_beg, cudaEventDefault) != cudaSuccess) pair_ok = false;
                  if (pair_ok && cudaEventCreateWithFlags(&ev_raw_end, cudaEventDefault) != cudaSuccess) pair_ok = false;
                  if (pair_ok) {
                    pair_ok = cudaEventRecord(ev_u8_beg, 0) == cudaSuccess;
                  }
                  if (pair_ok) {
                    pair_ok = cudaMemcpyAsync(
                                  ws_cache.lr5_probe.data_ptr(),
                                  lr5.data_ptr(),
                                  pair_bytes,
                                  cudaMemcpyDeviceToDevice,
                                  0) == cudaSuccess;
                  }
                  if (pair_ok) {
                    if (pair_mask_u8_ptr == nullptr) {
                      pair_ok = false;
                    } else {
                    launch_apply_rows_even_u8(
                        pair_threads,
                        grid_pair,
                        reinterpret_cast<__half*>(ws_cache.lr5_probe.data_ptr<at::Half>()),
                        pair_mask_u8_ptr,
                        pair_rows_even,
                        static_cast<int>(hidden_dim),
                        pair_reg_mode,
                        pair_vec_aligned,
                        0);
                    }
                  }
                  if (pair_ok) pair_ok = cudaEventRecord(ev_u8_end, 0) == cudaSuccess;
                  if (pair_ok) pair_ok = cudaEventSynchronize(ev_u8_end) == cudaSuccess;
                  if (pair_ok) pair_ok = cudaEventElapsedTime(&pair_u8_ms, ev_u8_beg, ev_u8_end) == cudaSuccess;

                  if (pair_ok) {
                    pair_ok = cudaEventRecord(ev_raw_beg, 0) == cudaSuccess;
                  }
                  if (pair_ok) {
                    pair_ok = cudaMemcpyAsync(
                                  ws_cache.lr5_probe.data_ptr(),
                                  lr5.data_ptr(),
                                  pair_bytes,
                                  cudaMemcpyDeviceToDevice,
                                  0) == cudaSuccess;
                  }
                  if (pair_ok) {
                    launch_apply_rows_even_i64_aligned(
                        pair_threads,
                        grid_pair,
                        reinterpret_cast<__half*>(ws_cache.lr5_probe.data_ptr<at::Half>()),
                        mask.data_ptr<int64_t>(),
                        pair_rows_even,
                        static_cast<int>(hidden_dim),
                        pair_reg_mode,
                        pair_vec_aligned);
                  }
                  if (pair_ok) pair_ok = cudaEventRecord(ev_raw_end, 0) == cudaSuccess;
                  if (pair_ok) pair_ok = cudaEventSynchronize(ev_raw_end) == cudaSuccess;
                  if (pair_ok) pair_ok = cudaEventElapsedTime(&pair_raw_ms, ev_raw_beg, ev_raw_end) == cudaSuccess;

                  if (ev_u8_beg != nullptr) cudaEventDestroy(ev_u8_beg);
                  if (ev_u8_end != nullptr) cudaEventDestroy(ev_u8_end);
                  if (ev_raw_beg != nullptr) cudaEventDestroy(ev_raw_beg);
                  if (ev_raw_end != nullptr) cudaEventDestroy(ev_raw_end);

                  if (pair_ok) {
                    const bool pair_choose_u8 = pair_u8_ms <= pair_raw_ms * 0.999f;
                    use_u8_path = pair_choose_u8;
                    emplace_stage_shape_i64(
                        stage_timing.i64_aligned_use_u8_by_shape,
                        stage_timing,
                        i64_memo_key,
                        pair_choose_u8 ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0));
                    emplace_stage_shape_i64(
                        stage_timing.i64_aligned_decision_ready_by_shape,
                        stage_timing,
                        i64_memo_key,
                        static_cast<uint8_t>(1));
                    emplace_stage_shape_i64(
                        stage_timing.i64_aligned_confirm_state_by_shape,
                        stage_timing,
                        i64_memo_key,
                        static_cast<uint8_t>(2));
                    emplace_stage_shape_i64(
                        stage_timing.i64_aligned_pair_probe_done_by_shape,
                        stage_timing,
                        i64_memo_key,
                        static_cast<uint8_t>(1));
                    emplace_stage_shape_i64(
                        stage_timing.i64_aligned_pair_probe_u8_ms_by_shape,
                        stage_timing,
                        i64_memo_key,
                        pair_u8_ms);
                    emplace_stage_shape_i64(
                        stage_timing.i64_aligned_pair_probe_raw_ms_by_shape,
                        stage_timing,
                        i64_memo_key,
                        pair_raw_ms);
                    has_memo = true;
                  }
                }
              }
            }
          }
          if (!has_memo && confirm_state == 0) {
            bool prior_choose_u8 = false;
            if (dim_bucket == 384) {
              prior_choose_u8 = true;
            } else if (weighted_rows >= 24000000LL) {
              prior_choose_u8 = true;
            }
            use_u8_path = prior_choose_u8;
            emplace_stage_shape_i64(
                stage_timing.i64_aligned_use_u8_by_shape,
                stage_timing,
                i64_memo_key,
                prior_choose_u8 ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0));
            emplace_stage_shape_i64(
                stage_timing.i64_aligned_confirm_state_by_shape,
                stage_timing,
                i64_memo_key,
                static_cast<uint8_t>(1));
            has_memo = true;
          } else if (!has_memo && confirm_state == 1 && aligned_u8_count + aligned_raw_count >= 1) {
            const float aligned_u8_avg =
                aligned_u8_count > 0
                    ? map_get_or(stage_timing.apply_i64_aligned_u8_ms_by_shape, stage_key, 0.0f) /
                          static_cast<float>(aligned_u8_count)
                    : std::numeric_limits<float>::infinity();
            const float aligned_raw_avg =
                aligned_raw_count > 0
                    ? map_get_or(stage_timing.apply_i64_aligned_raw_ms_by_shape, stage_key, 0.0f) /
                          static_cast<float>(aligned_raw_count)
                    : std::numeric_limits<float>::infinity();
            const bool choose_u8 = aligned_u8_avg <= aligned_raw_avg * 0.997f;
            emplace_stage_shape_i64(
                stage_timing.i64_aligned_use_u8_by_shape,
                stage_timing,
                i64_memo_key,
                choose_u8 ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0));
            emplace_stage_shape_i64(
                stage_timing.i64_aligned_decision_ready_by_shape,
                stage_timing,
                i64_memo_key,
                static_cast<uint8_t>(1));
            emplace_stage_shape_i64(
                stage_timing.i64_aligned_confirm_state_by_shape,
                stage_timing,
                i64_memo_key,
                static_cast<uint8_t>(2));
            use_u8_path = choose_u8;
            has_memo = true;
          } else if (!has_memo && aligned_u8_count >= 2 && aligned_raw_count >= 2) {
            const float aligned_u8_avg =
                map_get_or(stage_timing.apply_i64_aligned_u8_ms_by_shape, stage_key, 0.0f) /
                static_cast<float>(aligned_u8_count);
            const float aligned_raw_avg =
                map_get_or(stage_timing.apply_i64_aligned_raw_ms_by_shape, stage_key, 0.0f) /
                static_cast<float>(aligned_raw_count);
            const bool choose_u8 = aligned_u8_avg <= aligned_raw_avg * 0.995f;
            emplace_stage_shape_i64(
                stage_timing.i64_aligned_use_u8_by_shape,
                stage_timing,
                i64_memo_key,
                choose_u8 ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0));
            emplace_stage_shape_i64(
                stage_timing.i64_aligned_decision_ready_by_shape,
                stage_timing,
                i64_memo_key,
                static_cast<uint8_t>(1));
            emplace_stage_shape_i64(
                stage_timing.i64_aligned_confirm_state_by_shape,
                stage_timing,
                i64_memo_key,
                static_cast<uint8_t>(2));
            use_u8_path = choose_u8;
            has_memo = true;
          }
        }
        if (!has_memo) {
          use_u8_path = dim_bucket == 384;
        }

        if (map_get_or(stage_timing.i64_aligned_decision_ready_by_shape, i64_memo_key, static_cast<uint8_t>(0)) != 0 &&
            map_get_or(stage_timing.i64_aligned_rollback_done_by_shape, i64_memo_key, static_cast<uint8_t>(0)) == 0) {
          const int aligned_u8_count =
              map_get_or(stage_timing.apply_i64_aligned_u8_count_by_shape, stage_key, 0);
          const int aligned_raw_count =
              map_get_or(stage_timing.apply_i64_aligned_raw_count_by_shape, stage_key, 0);
          if (aligned_u8_count >= 3 && aligned_raw_count >= 3) {
            const float aligned_u8_avg =
                map_get_or(stage_timing.apply_i64_aligned_u8_ms_by_shape, stage_key, 0.0f) /
                static_cast<float>(aligned_u8_count);
            const float aligned_raw_avg =
                map_get_or(stage_timing.apply_i64_aligned_raw_ms_by_shape, stage_key, 0.0f) /
                static_cast<float>(aligned_raw_count);
            const bool choose_u8 = aligned_u8_avg <= aligned_raw_avg * 0.998f;
            emplace_stage_shape_i64(
                stage_timing.i64_aligned_use_u8_by_shape,
                stage_timing,
                i64_memo_key,
                choose_u8 ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0));
            emplace_stage_shape_i64(
                stage_timing.i64_aligned_rollback_done_by_shape,
                stage_timing,
                i64_memo_key,
                static_cast<uint8_t>(1));
            use_u8_path = choose_u8;
            has_memo = true;
          }
        }
      }
    }
    used_u8_path = use_u8_path;

    if (rows_all_even && mask_dtype == static_cast<int>(torch::kInt64)) {
      mask_align_hit_this_call = i64_align_hit;
    } else if (rows_all_even && use_u8_path) {
      mask_align_hit_this_call = true;
    }

    const int64_t* mask_i64_ptr =
        mask.scalar_type() == torch::kInt64 ? mask.data_ptr<int64_t>() : nullptr;
    const float* mask_f32_ptr =
        mask.scalar_type() == torch::kFloat32 ? mask.data_ptr<float>() : nullptr;
    const uint8_t* mask_u8_ptr = nullptr;
    if (use_u8_path) {
      mask_u8_ptr = ensure_mask_u8_cache(ws_cache, mask, mask_dtype, rows);
    }

    if (rows_even > 0) {
      int apply_path_tag = kApplyPathUnknown;
      int choose_mask_dtype = mask_dtype;
      if (rows_all_even) {
        if (use_u8_path) {
          apply_path_tag = kApplyPathU8;
          choose_mask_dtype = static_cast<int>(torch::kUInt8);
        } else if (mask.scalar_type() == torch::kInt64) {
          apply_path_tag = i64_align_hit ? kApplyPathI64Aligned : kApplyPathI64Unaligned;
        }
      } else if (use_u8_path) {
        apply_path_tag = kApplyPathU8;
        choose_mask_dtype = static_cast<int>(torch::kUInt8);
      }
      selected_mask_dtype_this_call = choose_mask_dtype;

      const int64_t plane_stride = rows * hidden_dim;
      const uintptr_t base_ptr_u = reinterpret_cast<uintptr_t>(lr5.data_ptr<at::Half>());
      const uintptr_t right_ptr_u = base_ptr_u + static_cast<uintptr_t>(plane_stride * sizeof(__half));
      const uintptr_t gl_ptr_u = right_ptr_u + static_cast<uintptr_t>(plane_stride * sizeof(__half));
      const uintptr_t gr_ptr_u = gl_ptr_u + static_cast<uintptr_t>(plane_stride * sizeof(__half));

      const bool apply_vec_aligned =
          rows_all_even && ((rows_even & 3) == 0) &&
          ((base_ptr_u & 0x7) == 0) && ((right_ptr_u & 0x7) == 0) && ((gl_ptr_u & 0x7) == 0) &&
          ((gr_ptr_u & 0x7) == 0);

      const int threads_vec = choose_apply_threads_with_probe(
          stage_timing,
          stage_key,
          apply_path_tag,
          dim_bucket,
          rows,
          rows_even,
          hidden_dim,
          choose_mask_dtype,
          rows_all_even,
          reinterpret_cast<__half*>(lr5.data_ptr<at::Half>()),
          reinterpret_cast<__half*>(ws_cache.lr5_probe.data_ptr<at::Half>()),
          mask_i64_ptr,
          mask_u8_ptr,
          mask_f32_ptr,
          profile_this_call,
          kApplyRegModeLoose);

      int apply_reg_mode = choose_apply_reg_mode_with_probe(
          stage_timing,
          stage_key,
          apply_path_tag,
          dim_bucket,
          rows,
          rows_even,
          hidden_dim,
          choose_mask_dtype,
          rows_all_even,
          threads_vec,
          reinterpret_cast<__half*>(lr5.data_ptr<at::Half>()),
          reinterpret_cast<__half*>(ws_cache.lr5_probe.data_ptr<at::Half>()),
          mask_i64_ptr,
          mask_u8_ptr,
          profile_this_call);
      apply_path_tag_this_call = apply_path_tag;
      apply_threads_this_call = threads_vec;
      apply_reg_mode_this_call = apply_reg_mode;
      const uint64_t path_obs_key = apply_threads_shape_key(
          stage_key,
          rows,
          dim_bucket,
          hidden_dim,
          choose_mask_dtype,
          rows_all_even,
          apply_path_tag);
      if (gate_host_stats) {
        stage_timing.apply_vec_total_count_by_shape[path_obs_key] += 1;
        if (apply_vec_aligned) {
          stage_timing.apply_vec_aligned_count_by_shape[path_obs_key] += 1;
        }
        if (threads_vec == 128) {
          stage_timing.apply_threads128_count_by_shape[path_obs_key] += 1;
        } else if (threads_vec == 192) {
          stage_timing.apply_threads192_count_by_shape[path_obs_key] += 1;
        } else if (threads_vec == 256) {
          stage_timing.apply_threads256_count_by_shape[path_obs_key] += 1;
        }
        const uint64_t reg_obs_key = apply_reg_shape_key(
            stage_key,
            rows,
            dim_bucket,
            hidden_dim,
            choose_mask_dtype,
            rows_all_even,
            apply_path_tag,
            threads_vec);
        if (apply_reg_mode <= kApplyRegModeTight) {
          stage_timing.apply_reg_tight_count_by_shape[reg_obs_key] += 1;
        } else {
          stage_timing.apply_reg_loose_count_by_shape[reg_obs_key] += 1;
        }
        if (mask_u8_ptr != nullptr) {
          const int sample_rows = rows_even > 64 ? 64 : static_cast<int>(rows_even);
          if (sample_rows > 0) {
            const int max_sample_start = static_cast<int>(rows_even) - sample_rows;
            int sample_start = 0;
            if (max_sample_start > 0) {
              sample_start =
                  static_cast<int>((rows ^ (stage_key & 0xffffULL)) % static_cast<uint64_t>(max_sample_start + 1));
            }
            auto hist_tensor = ws_cache.mask_hist16;
            if (hist_tensor.defined() && hist_tensor.numel() == 16) {
              cudaMemsetAsync(hist_tensor.data_ptr<int>(), 0, static_cast<size_t>(16) * sizeof(int), 0);
              mask_bits_hist16_u8_sample_kernel<<<1, 64>>>(
                  mask_u8_ptr + sample_start,
                  sample_rows,
                  reinterpret_cast<uint32_t*>(hist_tensor.data_ptr<int>()));
              int hist_local_i32[16] = {0};
              cudaMemcpyAsync(
                  hist_local_i32,
                  hist_tensor.data_ptr<int>(),
                  static_cast<size_t>(16) * sizeof(int),
                  cudaMemcpyDeviceToHost,
                  0);
              cudaDeviceSynchronize();
              for (int bits = 1; bits < 16; ++bits) {
                const int local_count = hist_local_i32[bits];
                if (local_count <= 0) continue;
                const uint64_t hist_key = mask_bits_hist_shape_key(stage_key, bits);
                const uint16_t prev = map_get_or(
                    stage_timing.apply_mask_bits_hist_by_shape,
                    hist_key,
                    static_cast<uint16_t>(0));
                const uint16_t add = local_count >= 0xFFFF ? static_cast<uint16_t>(0xFFFF) :
                                                          static_cast<uint16_t>(local_count);
                uint16_t next = static_cast<uint16_t>(prev + add);
                if (next < prev) next = std::numeric_limits<uint16_t>::max();
                emplace_stage_shape_i64(stage_timing.apply_mask_bits_hist_by_shape, stage_timing, hist_key, next);
              }
            }
          }
        }
      }
      const int blocks_vec = (rows_even + static_cast<int64_t>(threads_vec) * 4 - 1) /
                             (static_cast<int64_t>(threads_vec) * 4);
      const dim3 grid_vec(static_cast<unsigned int>(blocks_vec), static_cast<unsigned int>(hidden_dim), 1u);
    if (rows_all_even) {
      if (use_u8_path) {
        TORCH_CHECK(mask_u8_ptr != nullptr, "u8 mask path requires mask_u8_ptr");
        launch_apply_rows_even_u8(
            threads_vec,
            grid_vec,
            reinterpret_cast<__half*>(lr5.data_ptr<at::Half>()),
            mask_u8_ptr,
            rows_even,
            static_cast<int>(hidden_dim),
            apply_reg_mode,
            apply_vec_aligned,
            learned_case_mode);
      } else if (mask.scalar_type() == torch::kInt64) {
        if (i64_align_hit) {
          launch_apply_rows_even_i64_aligned(
              threads_vec,
              grid_vec,
              reinterpret_cast<__half*>(lr5.data_ptr<at::Half>()),
              mask.data_ptr<int64_t>(),
              rows_even,
              static_cast<int>(hidden_dim),
              apply_reg_mode,
              apply_vec_aligned);
        } else {
          launch_apply_rows_even_i64_unaligned(
              threads_vec,
              grid_vec,
              reinterpret_cast<__half*>(lr5.data_ptr<at::Half>()),
              mask.data_ptr<int64_t>(),
              rows_even,
              static_cast<int>(hidden_dim),
              apply_reg_mode,
              apply_vec_aligned);
        }
      } else if (mask.scalar_type() == torch::kFloat32) {
        launch_apply_rows_even_mask<float>(
            threads_vec,
            grid_vec,
            reinterpret_cast<__half*>(lr5.data_ptr<at::Half>()),
              mask.data_ptr<float>(),
              rows_even,
              static_cast<int>(hidden_dim));
        } else {
          TORCH_CHECK(false, "mask dtype");
        }
      } else {
        if (hidden_dim == 128) {
          const dim3 grid_pair(static_cast<unsigned int>(blocks_vec), 64u, 1u);
          if (use_u8_path) {
            TORCH_CHECK(mask_u8_ptr != nullptr, "u8 mask path requires mask_u8_ptr");
            launch_apply_main_parity_h128<uint8_t>(
                threads_vec,
                grid_pair,
                reinterpret_cast<__half*>(lr5.data_ptr<at::Half>()),
                mask_u8_ptr,
                rows_even,
                static_cast<int>(rows));
          } else if (mask.scalar_type() == torch::kInt64) {
            launch_apply_main_parity_h128<int64_t>(
                threads_vec,
                grid_pair,
                reinterpret_cast<__half*>(lr5.data_ptr<at::Half>()),
                mask.data_ptr<int64_t>(),
                rows_even,
                static_cast<int>(rows));
          } else if (mask.scalar_type() == torch::kFloat32) {
            launch_apply_main_parity_h128<float>(
                threads_vec,
                grid_pair,
                reinterpret_cast<__half*>(lr5.data_ptr<at::Half>()),
                mask.data_ptr<float>(),
                rows_even,
                static_cast<int>(rows));
          } else {
            TORCH_CHECK(false, "mask dtype");
          }
        } else {
          const int d_pair_count = static_cast<int>((hidden_dim + 1) >> 1);
          if (use_u8_path) {
            TORCH_CHECK(mask_u8_ptr != nullptr, "u8 mask path requires mask_u8_ptr");
            if (d_pair_count > 0) {
              const dim3 grid_pair(static_cast<unsigned int>(blocks_vec), static_cast<unsigned int>(d_pair_count), 1u);
              launch_apply_main_parity<uint8_t>(
                  threads_vec,
                  grid_pair,
                  reinterpret_cast<__half*>(lr5.data_ptr<at::Half>()),
                  mask_u8_ptr,
                  rows_even,
                  static_cast<int>(rows),
                  static_cast<int>(hidden_dim));
            }
          } else if (mask.scalar_type() == torch::kInt64) {
            if (d_pair_count > 0) {
              const dim3 grid_pair(static_cast<unsigned int>(blocks_vec), static_cast<unsigned int>(d_pair_count), 1u);
              launch_apply_main_parity<int64_t>(
                  threads_vec,
                  grid_pair,
                  reinterpret_cast<__half*>(lr5.data_ptr<at::Half>()),
                  mask.data_ptr<int64_t>(),
                  rows_even,
                  static_cast<int>(rows),
                  static_cast<int>(hidden_dim));
            }
          } else if (mask.scalar_type() == torch::kFloat32) {
            if (d_pair_count > 0) {
              const dim3 grid_pair(static_cast<unsigned int>(blocks_vec), static_cast<unsigned int>(d_pair_count), 1u);
              launch_apply_main_parity<float>(
                  threads_vec,
                  grid_pair,
                  reinterpret_cast<__half*>(lr5.data_ptr<at::Half>()),
                  mask.data_ptr<float>(),
                  rows_even,
                  static_cast<int>(rows),
                  static_cast<int>(hidden_dim));
            }
          } else {
            TORCH_CHECK(false, "mask dtype");
          }
        }
      }
    }
    if (has_tail) {
      const int tail_row = static_cast<int>(rows - 1);
      const int threads_tail = 256;
      const int blocks_tail = (static_cast<int>(hidden_dim) + threads_tail - 1) / threads_tail;
      cudaEvent_t ev_tail_begin = nullptr;
      cudaEvent_t ev_tail_end = nullptr;
      bool tail_evt_ok = false;
      if (profile_this_call) {
        if (cudaEventCreateWithFlags(&ev_tail_begin, cudaEventDefault) == cudaSuccess &&
            cudaEventCreateWithFlags(&ev_tail_end, cudaEventDefault) == cudaSuccess) {
          tail_evt_ok = cudaEventRecord(ev_tail_begin, 0) == cudaSuccess;
        } else {
          if (ev_tail_begin != nullptr) cudaEventDestroy(ev_tail_begin);
          if (ev_tail_end != nullptr) cudaEventDestroy(ev_tail_end);
          ev_tail_begin = nullptr;
          ev_tail_end = nullptr;
        }
      }
      if (use_u8_path) {
        TORCH_CHECK(mask_u8_ptr != nullptr, "u8 mask path requires mask_u8_ptr");
        apply_lr_gate_mask_f16_tail_kernel<uint8_t><<<blocks_tail, threads_tail>>>(
            reinterpret_cast<__half*>(lr5.data_ptr<at::Half>()),
            mask_u8_ptr,
            tail_row,
            static_cast<int>(rows),
            static_cast<int>(hidden_dim));
      } else if (mask.scalar_type() == torch::kInt64) {
        apply_lr_gate_mask_f16_tail_kernel<int64_t><<<blocks_tail, threads_tail>>>(
            reinterpret_cast<__half*>(lr5.data_ptr<at::Half>()),
            mask.data_ptr<int64_t>(),
            tail_row,
            static_cast<int>(rows),
            static_cast<int>(hidden_dim));
      } else if (mask.scalar_type() == torch::kFloat32) {
        apply_lr_gate_mask_f16_tail_kernel<float><<<blocks_tail, threads_tail>>>(
            reinterpret_cast<__half*>(lr5.data_ptr<at::Half>()),
            mask.data_ptr<float>(),
            tail_row,
            static_cast<int>(rows),
            static_cast<int>(hidden_dim));
      } else {
        TORCH_CHECK(false, "mask dtype");
      }

      if (tail_evt_ok && ev_tail_begin != nullptr && ev_tail_end != nullptr) {
        if (cudaEventRecord(ev_tail_end, 0) == cudaSuccess && cudaEventSynchronize(ev_tail_end) == cudaSuccess) {
          cudaEventElapsedTime(&odd_tail_ms_this_call, ev_tail_begin, ev_tail_end);
        }
      }
      if (ev_tail_begin != nullptr) cudaEventDestroy(ev_tail_begin);
      if (ev_tail_end != nullptr) cudaEventDestroy(ev_tail_end);
    }
  }

  if (profile_this_call) {
    if (ev_apply_end != nullptr) cudaEventRecord(ev_apply_end, 0);
    if (ev_batched_begin != nullptr) cudaEventRecord(ev_batched_begin, 0);
  }

  auto out_tmp = ws_cache.out_tmp;
  const int64_t batch = hidden_dim * bs;
  const int64_t stride_mat = n * n;
  const __half* left_ptr = reinterpret_cast<const __half*>(lr5.data_ptr<at::Half>());
  const __half* right_ptr = reinterpret_cast<const __half*>(lr5.data_ptr<at::Half>()) + hidden_dim * rows;
  const uint64_t shape_key = batched_shape_key(n, n, n, batch, stride_mat, stride_mat, stride_mat);
  cublasGemmAlgo_t algo = select_batched_algo(
      handle,
      caches.batched_algo,
      shape_key,
      left_ptr,
      right_ptr,
      reinterpret_cast<__half*>(out_tmp.data_ptr<at::Half>()),
      n,
      n,
      n,
      bs,
      dim,
      hidden_dim,
      batch,
      stride_mat,
      stride_mat,
      stride_mat);
  gemm_strided_batched_f16_abt(
      handle,
      left_ptr,
      right_ptr,
      reinterpret_cast<__half*>(out_tmp.data_ptr<at::Half>()),
      n,
      n,
      n,
      batch,
      stride_mat,
      stride_mat,
      stride_mat,
      algo);

  if (profile_this_call) {
    if (ev_batched_end != nullptr) cudaEventRecord(ev_batched_end, 0);
  }

  auto out_hidden = ws_cache.out_hidden;
  const __half* gate_ptr = reinterpret_cast<const __half*>(lr5.data_ptr<at::Half>()) + 4 * hidden_dim * rows;
  {
    dim3 block(kTile, kBlockRows);
    dim3 grid((rows + kTile - 1) / kTile);
    const __half* x_col = reinterpret_cast<const __half*>(out_tmp.data_ptr<at::Half>());
    __half* y_row = reinterpret_cast<__half*>(out_hidden.data_ptr<at::Half>());
    if (hidden_dim == 32) {
      ln_affine_gate_from_col_to_row_f16_kernel<32><<<grid, block>>>(
          x_col, gate_ptr, to_out_norm_weight.data_ptr<float>(), to_out_norm_bias.data_ptr<float>(), y_row, rows);
    } else if (hidden_dim == 64) {
      ln_affine_gate_from_col_to_row_f16_kernel<64><<<grid, block>>>(
          x_col, gate_ptr, to_out_norm_weight.data_ptr<float>(), to_out_norm_bias.data_ptr<float>(), y_row, rows);
    } else if (hidden_dim == 128) {
      ln_affine_gate_from_col_to_row_f16_kernel<128><<<grid, block>>>(
          x_col, gate_ptr, to_out_norm_weight.data_ptr<float>(), to_out_norm_bias.data_ptr<float>(), y_row, rows);
    } else {
      const int64_t shmem =
          static_cast<int64_t>(hidden_dim) * static_cast<int64_t>(kTile + 1) * static_cast<int64_t>(4);
      ln_affine_gate_from_col_to_row_f16_generic_kernel<<<grid, block, static_cast<size_t>(shmem)>>>(
          x_col,
          gate_ptr,
          to_out_norm_weight.data_ptr<float>(),
          to_out_norm_bias.data_ptr<float>(),
          y_row,
          rows,
          static_cast<int>(hidden_dim));
    }
  }

  auto y = torch::empty({bs, n, n, dim}, opts_f16);
  const __half* to_out_ptr = w5_ptr + seg_elems * 5;
  if (profile_this_call) {
    if (ev_final_begin != nullptr) cudaEventRecord(ev_final_begin, 0);
  }
  gemm_f16_abt(
      handle,
      reinterpret_cast<const __half*>(out_hidden.data_ptr<at::Half>()),
      to_out_ptr,
      reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
      rows,
      hidden_dim,
      dim);

  if (profile_this_call) {
    if (ev_final_end != nullptr) cudaEventRecord(ev_final_end, 0);
    if (ev_final_end != nullptr) cudaEventSynchronize(ev_final_end);

    float ln_ms = 0.0f;
    float apply_ms = 0.0f;
    float batched_ms = 0.0f;
    float final_ms = 0.0f;
    if (ev_ln_begin != nullptr && ev_ln_end != nullptr) cudaEventElapsedTime(&ln_ms, ev_ln_begin, ev_ln_end);
    if (ev_apply_begin != nullptr && ev_apply_end != nullptr)
      cudaEventElapsedTime(&apply_ms, ev_apply_begin, ev_apply_end);
    if (ev_batched_begin != nullptr && ev_batched_end != nullptr)
      cudaEventElapsedTime(&batched_ms, ev_batched_begin, ev_batched_end);
    if (ev_final_begin != nullptr && ev_final_end != nullptr)
      cudaEventElapsedTime(&final_ms, ev_final_begin, ev_final_end);

    emplace_stage_shape_i64(
        stage_timing.ln_ms_by_shape,
        stage_timing,
        stage_key,
        map_get_or(stage_timing.ln_ms_by_shape, stage_key, 0.0f) + ln_ms);
    emplace_stage_shape_i64(
        stage_timing.apply_ms_by_shape,
        stage_timing,
        stage_key,
        map_get_or(stage_timing.apply_ms_by_shape, stage_key, 0.0f) + apply_ms);
    emplace_stage_shape_i64(
        stage_timing.batched_ms_by_shape,
        stage_timing,
        stage_key,
        map_get_or(stage_timing.batched_ms_by_shape, stage_key, 0.0f) + batched_ms);
    emplace_stage_shape_i64(
        stage_timing.final_ms_by_shape,
        stage_timing,
        stage_key,
        map_get_or(stage_timing.final_ms_by_shape, stage_key, 0.0f) + final_ms);
    emplace_stage_shape_i64(
        stage_timing.profile_count_by_shape,
        stage_timing,
        stage_key,
        map_get_or(stage_timing.profile_count_by_shape, stage_key, 0) + 1);

    if (has_tail_this_call && odd_tail_ms_this_call >= 0.0f) {
      emplace_stage_shape_i64(
          stage_timing.odd_tail_ms_by_shape,
          stage_timing,
          stage_key,
          map_get_or(stage_timing.odd_tail_ms_by_shape, stage_key, 0.0f) + odd_tail_ms_this_call);
      emplace_stage_shape_i64(
          stage_timing.odd_tail_count_by_shape,
          stage_timing,
          stage_key,
          map_get_or(stage_timing.odd_tail_count_by_shape, stage_key, 0) + 1);
    }

    if (gate_host_stats && (rows & ~1LL) > 0) {
      int total_align = map_get_or(stage_timing.mask_align_total_by_shape, stage_key, 0);
      int hit_align = map_get_or(stage_timing.mask_align_hit_by_shape, stage_key, 0);
      total_align += 1;
      if (mask_align_hit_this_call) hit_align += 1;
      emplace_stage_shape_i64(stage_timing.mask_align_total_by_shape, stage_timing, stage_key, total_align);
      emplace_stage_shape_i64(stage_timing.mask_align_hit_by_shape, stage_timing, stage_key, hit_align);
      if (total_align >= 4) {
        const float align_ratio = static_cast<float>(hit_align) / static_cast<float>(total_align);
        emplace_stage_shape_i64(
            stage_timing.force_mask_u8_cache_by_shape,
            stage_timing,
            stage_key,
            align_ratio < 0.35f ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0));
      }
    }

    if (used_u8_path) {
      stage_timing.apply_u8_ms += apply_ms;
      stage_timing.apply_u8_count += 1;
      emplace_stage_shape_i64(
          stage_timing.apply_u8_ms_by_shape,
          stage_timing,
          stage_key,
          map_get_or(stage_timing.apply_u8_ms_by_shape, stage_key, 0.0f) + apply_ms);
      emplace_stage_shape_i64(
          stage_timing.apply_u8_count_by_shape,
          stage_timing,
          stage_key,
          map_get_or(stage_timing.apply_u8_count_by_shape, stage_key, 0) + 1);
    } else {
      stage_timing.apply_no_u8_ms += apply_ms;
      stage_timing.apply_no_u8_count += 1;
      emplace_stage_shape_i64(
          stage_timing.apply_no_u8_ms_by_shape,
          stage_timing,
          stage_key,
          map_get_or(stage_timing.apply_no_u8_ms_by_shape, stage_key, 0.0f) + apply_ms);
      emplace_stage_shape_i64(
          stage_timing.apply_no_u8_count_by_shape,
          stage_timing,
          stage_key,
          map_get_or(stage_timing.apply_no_u8_count_by_shape, stage_key, 0) + 1);
    }

    if (apply_path_tag_this_call == kApplyPathI64Aligned) {
      stage_timing.apply_i64_aligned_ms += apply_ms;
      stage_timing.apply_i64_aligned_count += 1;
      emplace_stage_shape_i64(
          stage_timing.apply_i64_aligned_ms_by_shape,
          stage_timing,
          stage_key,
          map_get_or(stage_timing.apply_i64_aligned_ms_by_shape, stage_key, 0.0f) + apply_ms);
      emplace_stage_shape_i64(
          stage_timing.apply_i64_aligned_count_by_shape,
          stage_timing,
          stage_key,
          map_get_or(stage_timing.apply_i64_aligned_count_by_shape, stage_key, 0) + 1);
      if (used_u8_path) {
        emplace_stage_shape_i64(
            stage_timing.apply_i64_aligned_u8_ms_by_shape,
            stage_timing,
            stage_key,
            map_get_or(stage_timing.apply_i64_aligned_u8_ms_by_shape, stage_key, 0.0f) + apply_ms);
        emplace_stage_shape_i64(
            stage_timing.apply_i64_aligned_u8_count_by_shape,
            stage_timing,
            stage_key,
            map_get_or(stage_timing.apply_i64_aligned_u8_count_by_shape, stage_key, 0) + 1);
      } else {
        emplace_stage_shape_i64(
            stage_timing.apply_i64_aligned_raw_ms_by_shape,
            stage_timing,
            stage_key,
            map_get_or(stage_timing.apply_i64_aligned_raw_ms_by_shape, stage_key, 0.0f) + apply_ms);
        emplace_stage_shape_i64(
            stage_timing.apply_i64_aligned_raw_count_by_shape,
            stage_timing,
            stage_key,
            map_get_or(stage_timing.apply_i64_aligned_raw_count_by_shape, stage_key, 0) + 1);
      }
    } else if (apply_path_tag_this_call == kApplyPathI64Unaligned) {
      stage_timing.apply_i64_unaligned_ms += apply_ms;
      stage_timing.apply_i64_unaligned_count += 1;
      emplace_stage_shape_i64(
          stage_timing.apply_i64_unaligned_ms_by_shape,
          stage_timing,
          stage_key,
          map_get_or(stage_timing.apply_i64_unaligned_ms_by_shape, stage_key, 0.0f) + apply_ms);
      emplace_stage_shape_i64(
          stage_timing.apply_i64_unaligned_count_by_shape,
          stage_timing,
          stage_key,
          map_get_or(stage_timing.apply_i64_unaligned_count_by_shape, stage_key, 0) + 1);
    } else {
      emplace_stage_shape_i64(
          stage_timing.apply_unknown_ms_by_shape,
          stage_timing,
          stage_key,
          map_get_or(stage_timing.apply_unknown_ms_by_shape, stage_key, 0.0f) + apply_ms);
      emplace_stage_shape_i64(
          stage_timing.apply_unknown_count_by_shape,
          stage_timing,
          stage_key,
          map_get_or(stage_timing.apply_unknown_count_by_shape, stage_key, 0) + 1);
    }

    if (apply_threads_this_call > 0) {
      const uint64_t threads_key = apply_threads_shape_key(
          stage_key,
          rows,
          dim_bucket,
          hidden_dim,
          used_u8_path ? static_cast<int>(torch::kUInt8) : mask_dtype,
          (rows & 1LL) == 0,
          apply_path_tag_this_call);
      if (gate_host_stats) {
        emplace_threads_shape_i64(
            stage_timing.apply_threads_by_shape,
            stage_timing,
            threads_key,
            apply_threads_this_call);
      }

      const uint64_t reg_key = apply_reg_shape_key(
          stage_key,
          rows,
          dim_bucket,
          hidden_dim,
          used_u8_path ? static_cast<int>(torch::kUInt8) : mask_dtype,
          (rows & 1LL) == 0,
          apply_path_tag_this_call,
          apply_threads_this_call);
      if (gate_host_stats) {
        emplace_reg_shape_i64(stage_timing.apply_reg_mode_by_shape, stage_timing, reg_key, apply_reg_mode_this_call);
      }
    }

    if (stage_timing.allow_learning && profile_this_call) {
      const int shape_u8_count = map_get_or(stage_timing.apply_u8_count_by_shape, stage_key, 0);
      const int shape_raw_count = map_get_or(stage_timing.apply_no_u8_count_by_shape, stage_key, 0);
      if (shape_u8_count >= 8 && shape_raw_count >= 8) {
        const float avg_u8 =
            map_get_or(stage_timing.apply_u8_ms_by_shape, stage_key, 0.0f) / static_cast<float>(shape_u8_count);
        const float avg_raw =
            map_get_or(stage_timing.apply_no_u8_ms_by_shape, stage_key, 0.0f) / static_cast<float>(shape_raw_count);
        float target = 1.0f;
        if (avg_u8 > avg_raw * 1.015f) {
          target = 1.06f;
        } else if (avg_u8 < avg_raw * 0.985f) {
          target = 0.94f;
        }
        float prev = 1.0f;
        auto it_prev = stage_timing.threshold_bias_by_shape.find(stage_key);
        if (it_prev != stage_timing.threshold_bias_by_shape.end()) {
          prev = it_prev->second;
        }
        float bias = prev * 0.75f + target * 0.25f;
        if (bias < 0.92f) bias = 0.92f;
        if (bias > 1.08f) bias = 1.08f;
        emplace_stage_shape_i64(stage_timing.threshold_bias_by_shape, stage_timing, stage_key, bias);
      }
    }

    stage_timing.ln_ms += ln_ms;
    stage_timing.apply_ms += apply_ms;
    stage_timing.batched_ms += batched_ms;
    stage_timing.final_ms += final_ms;
    stage_timing.sample_count += 1;
    stage_timing.remain_samples -= 1;

    if (stage_timing.remain_samples == 0 && stage_timing.sample_count > 0) {
      const float inv = 1.0f / static_cast<float>(stage_timing.sample_count);
      std::printf(
          "[trimul-stage] bs=%lld n=%lld dim=%lld hidden=%lld ln=%.3fms apply=%.3fms batched=%.3fms final=%.3fms samples=%d\n",
          static_cast<long long>(bs),
          static_cast<long long>(n),
          static_cast<long long>(dim),
          static_cast<long long>(hidden_dim),
          stage_timing.ln_ms * inv,
          stage_timing.apply_ms * inv,
          stage_timing.batched_ms * inv,
          stage_timing.final_ms * inv,
          stage_timing.sample_count);

      auto it_cnt = stage_timing.profile_count_by_shape.find(stage_key);
      if (it_cnt != stage_timing.profile_count_by_shape.end() && it_cnt->second > 0) {
        const float inv_shape = 1.0f / static_cast<float>(it_cnt->second);
        const float ln_shape = map_get_or(stage_timing.ln_ms_by_shape, stage_key, 0.0f) * inv_shape;
        const float apply_shape = map_get_or(stage_timing.apply_ms_by_shape, stage_key, 0.0f) * inv_shape;
        const float batched_shape = map_get_or(stage_timing.batched_ms_by_shape, stage_key, 0.0f) * inv_shape;
        const float final_shape = map_get_or(stage_timing.final_ms_by_shape, stage_key, 0.0f) * inv_shape;
        const int shape_u8_cnt = map_get_or(stage_timing.apply_u8_count_by_shape, stage_key, 0);
        const int shape_no_u8_cnt = map_get_or(stage_timing.apply_no_u8_count_by_shape, stage_key, 0);
        const int shape_aligned_cnt = map_get_or(stage_timing.apply_i64_aligned_count_by_shape, stage_key, 0);
        const int shape_unaligned_cnt = map_get_or(stage_timing.apply_i64_unaligned_count_by_shape, stage_key, 0);
        const int shape_unknown_cnt = map_get_or(stage_timing.apply_unknown_count_by_shape, stage_key, 0);
        const int shape_aligned_u8_cnt = map_get_or(stage_timing.apply_i64_aligned_u8_count_by_shape, stage_key, 0);
        const int shape_aligned_raw_cnt = map_get_or(stage_timing.apply_i64_aligned_raw_count_by_shape, stage_key, 0);
        const float shape_u8_avg =
            shape_u8_cnt > 0
                ? map_get_or(stage_timing.apply_u8_ms_by_shape, stage_key, 0.0f) / static_cast<float>(shape_u8_cnt)
                : 0.0f;
        const float shape_no_u8_avg =
            shape_no_u8_cnt > 0
                ? map_get_or(stage_timing.apply_no_u8_ms_by_shape, stage_key, 0.0f) /
                      static_cast<float>(shape_no_u8_cnt)
                               : 0.0f;
        const float shape_aligned_avg =
            shape_aligned_cnt > 0
                ? map_get_or(stage_timing.apply_i64_aligned_ms_by_shape, stage_key, 0.0f) /
                      static_cast<float>(shape_aligned_cnt)
                : 0.0f;
        const float shape_unaligned_avg =
            shape_unaligned_cnt > 0
                ? map_get_or(stage_timing.apply_i64_unaligned_ms_by_shape, stage_key, 0.0f) /
                      static_cast<float>(shape_unaligned_cnt)
                : 0.0f;
        const float shape_unknown_avg =
            shape_unknown_cnt > 0
                ? map_get_or(stage_timing.apply_unknown_ms_by_shape, stage_key, 0.0f) /
                      static_cast<float>(shape_unknown_cnt)
                : 0.0f;
        const float shape_aligned_u8_avg =
            shape_aligned_u8_cnt > 0
                ? map_get_or(stage_timing.apply_i64_aligned_u8_ms_by_shape, stage_key, 0.0f) /
                      static_cast<float>(shape_aligned_u8_cnt)
                : 0.0f;
        const float shape_aligned_raw_avg =
            shape_aligned_raw_cnt > 0
                ? map_get_or(stage_timing.apply_i64_aligned_raw_ms_by_shape, stage_key, 0.0f) /
                      static_cast<float>(shape_aligned_raw_cnt)
                : 0.0f;
        const int path_bucket_key = apply_path_bucket(kApplyPathI64Aligned);
        const uint64_t i64_memo_key = apply_threads_shape_key(
            stage_key,
            rows,
            dim_bucket,
            hidden_dim,
            mask_dtype,
            rows_all_even_this_call,
            path_bucket_key);
        const int i64_memo_ready =
            map_get_or(stage_timing.i64_aligned_decision_ready_by_shape, i64_memo_key, static_cast<uint8_t>(0)) != 0
                ? 1
                : 0;
        const int i64_memo_use_u8 =
            map_get_or(stage_timing.i64_aligned_use_u8_by_shape, i64_memo_key, static_cast<uint8_t>(0)) != 0 ? 1 : 0;
        const int profile_path_tag = apply_path_tag_this_call;
        const int profile_choose_mask_dtype = selected_mask_dtype_this_call;
        const int profile_rows_even = rows_all_even_this_call ? 1 : 0;
        const int profile_i64_align_decision = i64_aligned_decision_final ? 1 : 0;
        const int profile_i64_align_hit = mask_align_hit_this_call ? 1 : 0;
        const int profile_threads_key_dtype =
            used_u8_path ? static_cast<int>(torch::kUInt8) : selected_mask_dtype_this_call;
        const uint64_t profile_threads_key = apply_threads_shape_key(
            stage_key,
            rows,
            dim_bucket,
            hidden_dim,
            profile_threads_key_dtype,
            rows_all_even_this_call,
            profile_path_tag);
        const int profile_t128 = map_get_or(stage_timing.apply_threads128_count_by_shape, profile_threads_key, 0);
        const int profile_t192 = map_get_or(stage_timing.apply_threads192_count_by_shape, profile_threads_key, 0);
        const int profile_t256 = map_get_or(stage_timing.apply_threads256_count_by_shape, profile_threads_key, 0);
        const int profile_vec_total = map_get_or(stage_timing.apply_vec_total_count_by_shape, profile_threads_key, 0);
        const int profile_vec_aligned_hits =
            map_get_or(stage_timing.apply_vec_aligned_count_by_shape, profile_threads_key, 0);
        const uint64_t profile_reg_key = apply_reg_shape_key(
            stage_key,
            rows,
            dim_bucket,
            hidden_dim,
            profile_threads_key_dtype,
            rows_all_even_this_call,
            profile_path_tag,
            apply_threads_this_call > 0 ? apply_threads_this_call : 256);
        const int profile_reg_tight = map_get_or(stage_timing.apply_reg_tight_count_by_shape, profile_reg_key, 0);
        const int profile_reg_loose = map_get_or(stage_timing.apply_reg_loose_count_by_shape, profile_reg_key, 0);
        std::printf(
            "[trimul-stage-shape] key=%llu bs=%lld n=%lld dim=%lld hidden=%lld mask_dtype=%d ln=%.3fms apply=%.3fms batched=%.3fms final=%.3fms samples=%d apply_u8_avg=%.3fms apply_u8_cnt=%d apply_raw_avg=%.3fms apply_raw_cnt=%d apply_i64_aligned_avg=%.3fms apply_i64_aligned_cnt=%d apply_i64_unaligned_avg=%.3fms apply_i64_unaligned_cnt=%d apply_unknown_avg=%.3fms apply_unknown_cnt=%d aligned_u8_avg=%.3fms aligned_u8_cnt=%d aligned_raw_avg=%.3fms aligned_raw_cnt=%d i64_memo_ready=%d i64_memo_use_u8=%d path_tag=%d choose_mask_dtype=%d rows_even=%d i64_align_decision=%d i64_align_hit=%d t128=%d t192=%d t256=%d reg_tight=%d reg_loose=%d vec_aligned_hit=%d vec_total=%d\n",
            static_cast<unsigned long long>(stage_key),
            static_cast<long long>(bs),
            static_cast<long long>(n),
            static_cast<long long>(dim),
            static_cast<long long>(hidden_dim),
            mask_dtype,
            ln_shape,
            apply_shape,
            batched_shape,
            final_shape,
            it_cnt->second,
            shape_u8_avg,
            shape_u8_cnt,
            shape_no_u8_avg,
            shape_no_u8_cnt,
            shape_aligned_avg,
            shape_aligned_cnt,
            shape_unaligned_avg,
            shape_unaligned_cnt,
            shape_unknown_avg,
            shape_unknown_cnt,
            shape_aligned_u8_avg,
            shape_aligned_u8_cnt,
            shape_aligned_raw_avg,
            shape_aligned_raw_cnt,
            i64_memo_ready,
            i64_memo_use_u8,
            profile_path_tag,
            profile_choose_mask_dtype,
            profile_rows_even,
            profile_i64_align_decision,
            profile_i64_align_hit,
            profile_t128,
            profile_t192,
            profile_t256,
            profile_reg_tight,
            profile_reg_loose,
            profile_vec_aligned_hits,
            profile_vec_total);
      }
    }

    if (ev_ln_begin != nullptr) cudaEventDestroy(ev_ln_begin);
    if (ev_ln_end != nullptr) cudaEventDestroy(ev_ln_end);
    if (ev_apply_begin != nullptr) cudaEventDestroy(ev_apply_begin);
    if (ev_apply_end != nullptr) cudaEventDestroy(ev_apply_end);
    if (ev_batched_begin != nullptr) cudaEventDestroy(ev_batched_begin);
    if (ev_batched_end != nullptr) cudaEventDestroy(ev_batched_end);
    if (ev_final_begin != nullptr) cudaEventDestroy(ev_final_begin);
    if (ev_final_end != nullptr) cudaEventDestroy(ev_final_end);
  } else {
    if (ev_ln_begin != nullptr) cudaEventDestroy(ev_ln_begin);
    if (ev_ln_end != nullptr) cudaEventDestroy(ev_ln_end);
    if (ev_apply_begin != nullptr) cudaEventDestroy(ev_apply_begin);
    if (ev_apply_end != nullptr) cudaEventDestroy(ev_apply_end);
    if (ev_batched_begin != nullptr) cudaEventDestroy(ev_batched_begin);
    if (ev_batched_end != nullptr) cudaEventDestroy(ev_batched_end);
    if (ev_final_begin != nullptr) cudaEventDestroy(ev_final_begin);
    if (ev_final_end != nullptr) cudaEventDestroy(ev_final_end);
  }

  return y;
}
"""

    _EXT = load_inline(
        name=ext_name,
        cpp_sources=cpp_src,
        cuda_sources=cuda_src,
        functions=None,
        with_cuda=True,
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-arch=sm_80",
            "--use_fast_math",
            "-maxrregcount=72",
            f"-DTRIMUL_SHAPE_CACHE_CAPACITY={cache_cap}",
        ],
        verbose=False,
    )
    return _EXT


def custom_kernel(
    data: Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]],
) -> torch.Tensor:
    x, mask, weights, _ = data
    ext = _get_ext()
    return ext.trimul_forward(
        x,
        mask,
        weights["norm.weight"],
        weights["norm.bias"],
        weights["left_proj.weight"],
        weights["right_proj.weight"],
        weights["left_gate.weight"],
        weights["right_gate.weight"],
        weights["out_gate.weight"],
        weights["to_out_norm.weight"],
        weights["to_out_norm.bias"],
        weights["to_out.weight"],
    )


__all__ = ["custom_kernel"]
