"""
GPU-coalesced GGUF weight unpacking & fused OpenAI Triton (`@triton.jit`) W4A16 / W8A16
linear, embedding, and sparse Mixture-of-Experts (MoE) kernels (0% llama.cpp).

Optimized specifically for NVIDIA Tesla T4 (`sm_75` FP16 Tensor Cores + GDDR6 bandwidth)
with automatic vectorized PyTorch fallback on CPU or non-Triton environments.
"""

from __future__ import annotations

from contextlib import contextmanager
import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gguf_parser import GGML_BLOCK_INFO, GGMLQuantType, GGUFFile, GGUFTensorInfo

# Sub-5-bit GGUF formats (including Unsloth Dynamic `UD` mixes: IQ4_XS, IQ4_NL, Q3_K, Q2_K, IQ3, IQ2, MXFP4)
# are packed into 4-bit W4A16 (5.0 bits/weight) so 27B-32B models fit comfortably on Dual 15GB T4 GPUs!
SUB_5BIT_W4_QTYPES = frozenset(
    {
        GGMLQuantType.Q4_0,
        GGMLQuantType.Q4_1,
        GGMLQuantType.Q4_K,
        GGMLQuantType.Q2_K,
        GGMLQuantType.Q3_K,
        GGMLQuantType.IQ4_NL,
        GGMLQuantType.IQ4_XS,
        GGMLQuantType.IQ3_XXS,
        GGMLQuantType.IQ3_S,
        GGMLQuantType.IQ2_XXS,
        GGMLQuantType.IQ2_XS,
        GGMLQuantType.IQ2_S,
        GGMLQuantType.IQ1_S,
        GGMLQuantType.IQ1_M,
        GGMLQuantType.TQ1_0,
        GGMLQuantType.TQ2_0,
        GGMLQuantType.MXFP4,
        GGMLQuantType.NVFP4,
    }
)

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    HAS_TRITON = False


# ==============================================================================
# 1. Fused @triton.jit GPU Kernels (W4A16 & W8A16 GEMV + Tensor-Core GEMM)
# ==============================================================================

if HAS_TRITON:

    @triton.jit
    def _w4a16_gemv_kernel(
        x_ptr,
        qw_i32_ptr,
        scales_ptr,
        zeros_ptr,
        bias_ptr,
        out_ptr,
        M,
        N,
        K,
        HAS_BIAS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        row_scale_ptr=None,
        HAS_ROW_SCALE: tl.constexpr = False,
        OUT_FP32: tl.constexpr = False,
        X_PACKED_I32: tl.constexpr = False,
    ):
        """
        Zero-barrier register-accumulated 128-byte `int32` (`ld.global.b32`) W4A16 GEMV kernel (`M == 1`).
        - Accumulates per-lane FP32 registers across the entire K-loop (`0` in-loop `bar.sync` / shared-mem stalls!)
          and reduces once at the epilogue via warp-butterfly shuffles (`shfl.sync.bfly.b32`).
        - Supports `int32`-pair vectorized `x` loads (`X_PACKED_I32=True`, 4x `ld.global.b32` instead of 8x `b16`)
          and fused `row_scale` + `OUT_FP32` epilogue for `down_proj`!
        """
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        offs_n64 = offs_n.to(tl.int64)

        n_groups = K // 32
        k_i32 = K // 8
        lanes32 = tl.arange(0, 32)
        g_local = lanes32 // 4
        j_local = lanes32 % 4

        acc_lane = tl.zeros((BLOCK_N, 32), dtype=tl.float32)
        if X_PACKED_I32:
            x_row_ptr = x_ptr + pid_m.to(tl.int64) * (K // 2)
        else:
            x_row_ptr = x_ptr + pid_m.to(tl.int64) * K

        for g_base in range(0, n_groups, 8):
            offs_i32 = g_base * 4 + lanes32
            mask_k = offs_i32 < k_i32

            w32 = tl.load(
                qw_i32_ptr + offs_n64[:, None] * k_i32 + offs_i32[None, :],
                mask=mask_n[:, None] & mask_k[None, :],
                other=0,
            )

            if X_PACKED_I32:
                x_pair_lo = (g_base + g_local) * 16 + j_local * 2
                xp_lo0 = tl.load(x_row_ptr + x_pair_lo + 0, mask=mask_k, other=0)
                xp_lo1 = tl.load(x_row_ptr + x_pair_lo + 1, mask=mask_k, other=0)
                xp_hi0 = tl.load(x_row_ptr + x_pair_lo + 8, mask=mask_k, other=0)
                xp_hi1 = tl.load(x_row_ptr + x_pair_lo + 9, mask=mask_k, other=0)

                x_lo_0 = (xp_lo0 & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
                x_lo_1 = ((xp_lo0 >> 16) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
                x_lo_2 = (xp_lo1 & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
                x_lo_3 = ((xp_lo1 >> 16) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)

                x_hi_0 = (xp_hi0 & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
                x_hi_1 = ((xp_hi0 >> 16) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
                x_hi_2 = (xp_hi1 & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
                x_hi_3 = ((xp_hi1 >> 16) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
            else:
                x_idx_lo = (g_base + g_local) * 32 + j_local * 4
                x_lo_0 = tl.load(x_row_ptr + x_idx_lo + 0, mask=mask_k, other=0.0).to(tl.float32)
                x_lo_1 = tl.load(x_row_ptr + x_idx_lo + 1, mask=mask_k, other=0.0).to(tl.float32)
                x_lo_2 = tl.load(x_row_ptr + x_idx_lo + 2, mask=mask_k, other=0.0).to(tl.float32)
                x_lo_3 = tl.load(x_row_ptr + x_idx_lo + 3, mask=mask_k, other=0.0).to(tl.float32)

                x_hi_0 = tl.load(x_row_ptr + x_idx_lo + 16, mask=mask_k, other=0.0).to(tl.float32)
                x_hi_1 = tl.load(x_row_ptr + x_idx_lo + 17, mask=mask_k, other=0.0).to(tl.float32)
                x_hi_2 = tl.load(x_row_ptr + x_idx_lo + 18, mask=mask_k, other=0.0).to(tl.float32)
                x_hi_3 = tl.load(x_row_ptr + x_idx_lo + 19, mask=mask_k, other=0.0).to(tl.float32)

            dot32 = (
                ((w32 >> 0) & 0x0F).to(tl.float32) * x_lo_0[None, :]
                + ((w32 >> 8) & 0x0F).to(tl.float32) * x_lo_1[None, :]
                + ((w32 >> 16) & 0x0F).to(tl.float32) * x_lo_2[None, :]
                + ((w32 >> 24) & 0x0F).to(tl.float32) * x_lo_3[None, :]
                + ((w32 >> 4) & 0x0F).to(tl.float32) * x_hi_0[None, :]
                + ((w32 >> 12) & 0x0F).to(tl.float32) * x_hi_1[None, :]
                + ((w32 >> 20) & 0x0F).to(tl.float32) * x_hi_2[None, :]
                + ((w32 >> 28) & 0x0F).to(tl.float32) * x_hi_3[None, :]
            )
            x_lane_sum = x_lo_0 + x_lo_1 + x_lo_2 + x_lo_3 + x_hi_0 + x_hi_1 + x_hi_2 + x_hi_3

            offs_g_lane = g_base + g_local
            mask_gl = offs_g_lane < n_groups
            sg_ptrs = offs_n64[:, None] * n_groups + offs_g_lane[None, :]
            s = tl.load(scales_ptr + sg_ptrs, mask=mask_n[:, None] & mask_gl[None, :], other=0.0).to(tl.float32)
            z = tl.load(zeros_ptr + sg_ptrs, mask=mask_n[:, None] & mask_gl[None, :], other=0.0).to(tl.float32)

            acc_lane += s * dot32 - z * x_lane_sum[None, :]

        acc = tl.sum(acc_lane, axis=1)

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n64, mask=mask_n, other=0.0).to(tl.float32)
            acc += bias
        if HAS_ROW_SCALE:
            r_sc = tl.load(row_scale_ptr + pid_m).to(tl.float32)
            acc = acc * r_sc

        out_ptrs = out_ptr + pid_m.to(tl.int64) * N + offs_n64
        if OUT_FP32:
            tl.store(out_ptrs, acc, mask=mask_n)
        else:
            acc_clamped = tl.minimum(tl.maximum(acc, -65504.0), 65504.0)
            tl.store(out_ptrs, acc_clamped.to(tl.float16), mask=mask_n)

    @triton.jit
    def _w4a16_multirow_gemv_kernel(
        x_ptr,
        qw_i32_ptr,
        scales_ptr,
        zeros_ptr,
        bias_ptr,
        out_ptr,
        M,
        N,
        K,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        row_scale_ptr=None,
        HAS_ROW_SCALE: tl.constexpr = False,
        OUT_FP32: tl.constexpr = False,
        X_PACKED_I32: tl.constexpr = False,
    ):
        """
        Zero-barrier weight-reusing multi-row (`2 <= M <= 8`) W4A16 GEMV kernel for MTP & N-gram verify!
        Eliminates all 4 per-tile `bar.sync` shared-memory stalls by accumulating `(BLOCK_M, BLOCK_N, 32)`
        directly in registers and loading `x` via `int32` pairs (`8` `ld.global.b32` vs `24` `ld.global.b16`).
        """
        pid_n = tl.program_id(0)

        offs_m = tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        offs_m64 = offs_m.to(tl.int64)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        offs_n64 = offs_n.to(tl.int64)

        n_groups = K // 32
        k_i32 = K // 8
        lanes32 = tl.arange(0, 32)
        g_local = lanes32 // 4
        j_local = lanes32 % 4

        acc_lane = tl.zeros((BLOCK_M, BLOCK_N, 32), dtype=tl.float32)

        for g_base in range(0, n_groups, 8):
            offs_i32 = g_base * 4 + lanes32
            mask_k = offs_i32 < k_i32
            mask_mk = mask_m[:, None] & mask_k[None, :]

            w32 = tl.load(
                qw_i32_ptr + offs_n64[:, None] * k_i32 + offs_i32[None, :],
                mask=mask_n[:, None] & mask_k[None, :],
                other=0,
            )
            q_lo_0 = ((w32 >> 0) & 0x0F).to(tl.float32)
            q_lo_1 = ((w32 >> 8) & 0x0F).to(tl.float32)
            q_lo_2 = ((w32 >> 16) & 0x0F).to(tl.float32)
            q_lo_3 = ((w32 >> 24) & 0x0F).to(tl.float32)
            q_hi_0 = ((w32 >> 4) & 0x0F).to(tl.float32)
            q_hi_1 = ((w32 >> 12) & 0x0F).to(tl.float32)
            q_hi_2 = ((w32 >> 20) & 0x0F).to(tl.float32)
            q_hi_3 = ((w32 >> 28) & 0x0F).to(tl.float32)

            if X_PACKED_I32:
                x_pair_lo = (g_base + g_local) * 16 + j_local * 2
                x_base_p = x_ptr + offs_m64[:, None] * (K // 2) + x_pair_lo[None, :]
                xp_lo0 = tl.load(x_base_p + 0, mask=mask_mk, other=0)
                xp_lo1 = tl.load(x_base_p + 1, mask=mask_mk, other=0)
                xp_hi0 = tl.load(x_base_p + 8, mask=mask_mk, other=0)
                xp_hi1 = tl.load(x_base_p + 9, mask=mask_mk, other=0)

                x_lo_0 = (xp_lo0 & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
                x_lo_1 = ((xp_lo0 >> 16) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
                x_lo_2 = (xp_lo1 & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
                x_lo_3 = ((xp_lo1 >> 16) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)

                x_hi_0 = (xp_hi0 & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
                x_hi_1 = ((xp_hi0 >> 16) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
                x_hi_2 = (xp_hi1 & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
                x_hi_3 = ((xp_hi1 >> 16) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
            else:
                x_idx_lo = (g_base + g_local) * 32 + j_local * 4
                x_base_lo = x_ptr + offs_m64[:, None] * K + x_idx_lo[None, :]
                x_lo_0 = tl.load(x_base_lo + 0, mask=mask_mk, other=0.0).to(tl.float32)
                x_lo_1 = tl.load(x_base_lo + 1, mask=mask_mk, other=0.0).to(tl.float32)
                x_lo_2 = tl.load(x_base_lo + 2, mask=mask_mk, other=0.0).to(tl.float32)
                x_lo_3 = tl.load(x_base_lo + 3, mask=mask_mk, other=0.0).to(tl.float32)

                x_hi_0 = tl.load(x_base_lo + 16, mask=mask_mk, other=0.0).to(tl.float32)
                x_hi_1 = tl.load(x_base_lo + 17, mask=mask_mk, other=0.0).to(tl.float32)
                x_hi_2 = tl.load(x_base_lo + 18, mask=mask_mk, other=0.0).to(tl.float32)
                x_hi_3 = tl.load(x_base_lo + 19, mask=mask_mk, other=0.0).to(tl.float32)

            dot32 = (
                q_lo_0[None, :, :] * x_lo_0[:, None, :]
                + q_lo_1[None, :, :] * x_lo_1[:, None, :]
                + q_lo_2[None, :, :] * x_lo_2[:, None, :]
                + q_lo_3[None, :, :] * x_lo_3[:, None, :]
                + q_hi_0[None, :, :] * x_hi_0[:, None, :]
                + q_hi_1[None, :, :] * x_hi_1[:, None, :]
                + q_hi_2[None, :, :] * x_hi_2[:, None, :]
                + q_hi_3[None, :, :] * x_hi_3[:, None, :]
            )
            x_lane_sum = x_lo_0 + x_lo_1 + x_lo_2 + x_lo_3 + x_hi_0 + x_hi_1 + x_hi_2 + x_hi_3

            offs_g_lane = g_base + g_local
            mask_gl = offs_g_lane < n_groups
            sg_ptrs = offs_n64[:, None] * n_groups + offs_g_lane[None, :]
            s = tl.load(scales_ptr + sg_ptrs, mask=mask_n[:, None] & mask_gl[None, :], other=0.0).to(tl.float32)
            z = tl.load(zeros_ptr + sg_ptrs, mask=mask_n[:, None] & mask_gl[None, :], other=0.0).to(tl.float32)

            acc_lane += s[None, :, :] * dot32 - z[None, :, :] * x_lane_sum[:, None, :]

        acc = tl.sum(acc_lane, axis=2)

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n64, mask=mask_n, other=0.0).to(tl.float32)
            acc += bias[None, :]
        if HAS_ROW_SCALE:
            r_sc = tl.load(row_scale_ptr + offs_m64, mask=mask_m, other=1.0).to(tl.float32)
            acc = acc * r_sc[:, None]

        out_ptrs = out_ptr + offs_m64[:, None] * N + offs_n64[None, :]
        mask_mn = mask_m[:, None] & mask_n[None, :]
        if OUT_FP32:
            tl.store(out_ptrs, acc, mask=mask_mn)
        else:
            acc_clamped = tl.minimum(tl.maximum(acc, -65504.0), 65504.0)
            tl.store(out_ptrs, acc_clamped.to(tl.float16), mask=mask_mn)

    @triton.jit
    def _w4a16_gemm_tc_kernel(
        x_ptr,
        qw_i32_ptr,
        scales_ptr,
        zeros_ptr,
        bias_ptr,
        out_ptr,
        M,
        N,
        K,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Coalesced `int32` W4A16 Tensor-Core GEMM kernel (`BLOCK_K=64`) for Ampere+ (`sm_80+`)."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N
        offs_m64 = offs_m.to(tl.int64)
        offs_n64 = offs_n.to(tl.int64)

        n_groups = K // 32
        k_i32 = K // 8
        lanes8 = tl.arange(0, 8)
        lanes64 = tl.arange(0, 64)
        offs_2 = tl.arange(0, 2)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for g_base in range(0, n_groups, 2):
            offs_k = g_base * 32 + lanes64
            mask_k64 = offs_k < K
            a_tile = tl.load(
                x_ptr + offs_m64[:, None] * K + offs_k[None, :],
                mask=mask_m[:, None] & mask_k64[None, :],
                other=0.0,
            ).to(tl.float16)

            offs_i32 = g_base * 4 + lanes8
            mask_i32 = offs_i32 < k_i32
            w32 = tl.load(
                qw_i32_ptr + offs_n64[:, None] * k_i32 + offs_i32[None, :],
                mask=mask_n[:, None] & mask_i32[None, :],
                other=0,
            )

            lo_ev = tl.join((w32 >> 0) & 0x0F, (w32 >> 16) & 0x0F)
            lo_od = tl.join((w32 >> 8) & 0x0F, (w32 >> 24) & 0x0F)
            q_lo = tl.reshape(tl.join(lo_ev, lo_od), (BLOCK_N, 2, 16)).to(tl.float32)

            hi_ev = tl.join((w32 >> 4) & 0x0F, (w32 >> 20) & 0x0F)
            hi_od = tl.join((w32 >> 12) & 0x0F, (w32 >> 28) & 0x0F)
            q_hi = tl.reshape(tl.join(hi_ev, hi_od), (BLOCK_N, 2, 16)).to(tl.float32)

            q_grp = tl.reshape(tl.permute(tl.join(q_lo, q_hi), (0, 1, 3, 2)), (BLOCK_N, 2, 32))

            offs_g = g_base + offs_2
            mask_g = offs_g < n_groups
            sg_ptrs = offs_n64[:, None] * n_groups + offs_g[None, :]
            s = tl.load(scales_ptr + sg_ptrs, mask=mask_n[:, None] & mask_g[None, :], other=0.0).to(tl.float32)[:, :, None]
            z = tl.load(zeros_ptr + sg_ptrs, mask=mask_n[:, None] & mask_g[None, :], other=0.0).to(tl.float32)[:, :, None]

            w_tile = tl.reshape((q_grp * s - z).to(tl.float16), (BLOCK_N, 64))
            w_tile = tl.where(mask_k64[None, :], w_tile, tl.zeros((1, 1), dtype=tl.float16))
            acc += tl.dot(a_tile, tl.trans(w_tile))

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n64, mask=mask_n, other=0.0).to(tl.float32)
            acc += bias[None, :]

        acc_clamped = tl.minimum(tl.maximum(acc, -65504.0), 65504.0)
        out_ptrs = out_ptr + offs_m64[:, None] * N + offs_n64[None, :]
        tl.store(out_ptrs, acc_clamped.to(tl.float16), mask=mask_m[:, None] & mask_n[None, :])

    @triton.jit
    def _w8a16_gemv_kernel(
        x_ptr,
        qw_i32_ptr,
        scales_ptr,
        zeros_ptr,
        bias_ptr,
        out_ptr,
        M,
        N,
        K,
        HAS_ZEROS: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Coalesced 128-byte `int32` (`ld.global.b32`) W8A16 GEMV kernel for decode (M <= 8)."""
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        offs_n64 = offs_n.to(tl.int64)

        n_groups = K // 16
        k_i32 = K // 4
        lanes32 = tl.arange(0, 32)
        offs_8 = tl.arange(0, 8)

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        x_row_ptr = x_ptr + pid_m.to(tl.int64) * K

        for g_base in range(0, n_groups, 8):
            offs_i32 = g_base * 4 + lanes32
            mask_k = offs_i32 < k_i32

            w32 = tl.load(
                qw_i32_ptr + offs_n64[:, None] * k_i32 + offs_i32[None, :],
                mask=mask_n[:, None] & mask_k[None, :],
                other=0,
            )
            q0 = ((w32 << 24) >> 24).to(tl.float32)
            q1 = ((w32 << 16) >> 24).to(tl.float32)
            q2 = ((w32 << 8) >> 24).to(tl.float32)
            q3 = (w32 >> 24).to(tl.float32)

            x_idx = offs_i32 * 4
            x0 = tl.load(x_row_ptr + x_idx + 0, mask=mask_k, other=0.0).to(tl.float32)
            x1 = tl.load(x_row_ptr + x_idx + 1, mask=mask_k, other=0.0).to(tl.float32)
            x2 = tl.load(x_row_ptr + x_idx + 2, mask=mask_k, other=0.0).to(tl.float32)
            x3 = tl.load(x_row_ptr + x_idx + 3, mask=mask_k, other=0.0).to(tl.float32)

            dot32 = q0 * x0[None, :] + q1 * x1[None, :] + q2 * x2[None, :] + q3 * x3[None, :]
            dot_g = tl.sum(tl.reshape(dot32, (BLOCK_N, 8, 4)), axis=2)

            offs_g = g_base + offs_8
            mask_g = offs_g < n_groups
            sg_ptrs = offs_n64[:, None] * n_groups + offs_g[None, :]
            s = tl.load(scales_ptr + sg_ptrs, mask=mask_n[:, None] & mask_g[None, :], other=0.0).to(tl.float32)

            if HAS_ZEROS:
                x_g_sum = tl.sum(tl.reshape(x0 + x1 + x2 + x3, (8, 4)), axis=1)
                z = tl.load(zeros_ptr + sg_ptrs, mask=mask_n[:, None] & mask_g[None, :], other=0.0).to(tl.float32)
                acc += tl.sum(s * dot_g - z * x_g_sum[None, :], axis=1)
            else:
                acc += tl.sum(s * dot_g, axis=1)

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n64, mask=mask_n, other=0.0).to(tl.float32)
            acc += bias

        acc_clamped = tl.minimum(tl.maximum(acc, -65504.0), 65504.0)
        tl.store(out_ptr + pid_m.to(tl.int64) * N + offs_n64, acc_clamped.to(tl.float16), mask=mask_n)

    @triton.jit
    def _w8a16_multirow_gemv_kernel(
        x_ptr,
        qw_i32_ptr,
        scales_ptr,
        zeros_ptr,
        bias_ptr,
        out_ptr,
        M,
        N,
        K,
        HAS_ZEROS: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Weight-reusing multi-row (`2 <= M <= 8`) W8A16 GEMV kernel for MTP & N-gram speculative verify!"""
        pid_n = tl.program_id(0)

        offs_m = tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        offs_m64 = offs_m.to(tl.int64)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        offs_n64 = offs_n.to(tl.int64)

        n_groups = K // 16
        k_i32 = K // 4
        lanes32 = tl.arange(0, 32)
        offs_8 = tl.arange(0, 8)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for g_base in range(0, n_groups, 8):
            offs_i32 = g_base * 4 + lanes32
            mask_k = offs_i32 < k_i32
            mask_mk = mask_m[:, None] & mask_k[None, :]

            w32 = tl.load(
                qw_i32_ptr + offs_n64[:, None] * k_i32 + offs_i32[None, :],
                mask=mask_n[:, None] & mask_k[None, :],
                other=0,
            )
            q0 = ((w32 << 24) >> 24).to(tl.float32)
            q1 = ((w32 << 16) >> 24).to(tl.float32)
            q2 = ((w32 << 8) >> 24).to(tl.float32)
            q3 = (w32 >> 24).to(tl.float32)

            x_idx = offs_i32 * 4
            x_base = x_ptr + offs_m64[:, None] * K + x_idx[None, :]
            x0 = tl.load(x_base + 0, mask=mask_mk, other=0.0).to(tl.float32)
            x1 = tl.load(x_base + 1, mask=mask_mk, other=0.0).to(tl.float32)
            x2 = tl.load(x_base + 2, mask=mask_mk, other=0.0).to(tl.float32)
            x3 = tl.load(x_base + 3, mask=mask_mk, other=0.0).to(tl.float32)

            dot32 = (
                q0[None, :, :] * x0[:, None, :]
                + q1[None, :, :] * x1[:, None, :]
                + q2[None, :, :] * x2[:, None, :]
                + q3[None, :, :] * x3[:, None, :]
            )
            dot_g = tl.sum(tl.reshape(dot32, (BLOCK_M, BLOCK_N, 8, 4)), axis=3)

            offs_g = g_base + offs_8
            mask_g = offs_g < n_groups
            sg_ptrs = offs_n64[:, None] * n_groups + offs_g[None, :]
            s = tl.load(scales_ptr + sg_ptrs, mask=mask_n[:, None] & mask_g[None, :], other=0.0).to(tl.float32)

            if HAS_ZEROS:
                x_g_sum = tl.sum(tl.reshape(x0 + x1 + x2 + x3, (BLOCK_M, 8, 4)), axis=2)
                z = tl.load(zeros_ptr + sg_ptrs, mask=mask_n[:, None] & mask_g[None, :], other=0.0).to(tl.float32)
                acc += tl.sum(s[None, :, :] * dot_g - z[None, :, :] * x_g_sum[:, None, :], axis=2)
            else:
                acc += tl.sum(s[None, :, :] * dot_g, axis=2)

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n64, mask=mask_n, other=0.0).to(tl.float32)
            acc += bias[None, :]

        acc_clamped = tl.minimum(tl.maximum(acc, -65504.0), 65504.0)
        tl.store(
            out_ptr + offs_m64[:, None] * N + offs_n64[None, :],
            acc_clamped.to(tl.float16),
            mask=mask_m[:, None] & mask_n[None, :],
        )

    @triton.jit
    def _w8a16_gemm_tc_kernel(
        x_ptr,
        qw_i32_ptr,
        scales_ptr,
        zeros_ptr,
        bias_ptr,
        out_ptr,
        M,
        N,
        K,
        HAS_ZEROS: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Coalesced `int32` W8A16 Tensor-Core GEMM kernel (`BLOCK_K=64`) for Ampere+ (`sm_80+`)."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N
        offs_m64 = offs_m.to(tl.int64)
        offs_n64 = offs_n.to(tl.int64)

        n_groups = K // 16
        k_i32 = K // 4
        lanes16 = tl.arange(0, 16)
        lanes64 = tl.arange(0, 64)
        offs_4 = tl.arange(0, 4)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for g_base in range(0, n_groups, 4):
            offs_k = g_base * 16 + lanes64
            mask_k64 = offs_k < K
            a_tile = tl.load(
                x_ptr + offs_m64[:, None] * K + offs_k[None, :],
                mask=mask_m[:, None] & mask_k64[None, :],
                other=0.0,
            ).to(tl.float16)

            offs_i32 = g_base * 4 + lanes16
            mask_i32 = offs_i32 < k_i32
            w32 = tl.load(
                qw_i32_ptr + offs_n64[:, None] * k_i32 + offs_i32[None, :],
                mask=mask_n[:, None] & mask_i32[None, :],
                other=0,
            )

            q0 = ((w32 << 24) >> 24).to(tl.float32)
            q1 = ((w32 << 16) >> 24).to(tl.float32)
            q2 = ((w32 << 8) >> 24).to(tl.float32)
            q3 = (w32 >> 24).to(tl.float32)
            q_grp = tl.reshape(tl.join(tl.join(q0, q2), tl.join(q1, q3)), (BLOCK_N, 4, 16))

            offs_g = g_base + offs_4
            mask_g = offs_g < n_groups
            sg_ptrs = offs_n64[:, None] * n_groups + offs_g[None, :]
            s = tl.load(scales_ptr + sg_ptrs, mask=mask_n[:, None] & mask_g[None, :], other=0.0).to(tl.float32)[:, :, None]
            if HAS_ZEROS:
                z = tl.load(zeros_ptr + sg_ptrs, mask=mask_n[:, None] & mask_g[None, :], other=0.0).to(tl.float32)[:, :, None]
                w_grp = (q_grp * s - z).to(tl.float16)
            else:
                w_grp = (q_grp * s).to(tl.float16)

            w_tile = tl.reshape(w_grp, (BLOCK_N, 64))
            w_tile = tl.where(mask_k64[None, :], w_tile, tl.zeros((1, 1), dtype=tl.float16))
            acc += tl.dot(a_tile, tl.trans(w_tile))

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n64, mask=mask_n, other=0.0).to(tl.float32)
            acc += bias[None, :]

        acc_clamped = tl.minimum(tl.maximum(acc, -65504.0), 65504.0)
        out_ptrs = out_ptr + offs_m64[:, None] * N + offs_n64[None, :]
        tl.store(out_ptrs, acc_clamped.to(tl.float16), mask=mask_m[:, None] & mask_n[None, :])

    @triton.jit
    def _dequant_w4_chunk_kernel(
        qw_i32_ptr,
        scales_ptr,
        zeros_ptr,
        out_fp16_ptr,
        rc,
        K,
        BLOCK_R: tl.constexpr,
    ):
        """Fast coalesced W4 -> FP16 chunk unpacker feeding Turing (`sm_75`) cuBLAS Tensor Cores during prefill."""
        pid_r = tl.program_id(0)
        pid_g = tl.program_id(1)

        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        mask_r = offs_r < rc
        offs_r64 = offs_r.to(tl.int64)

        n_groups = K // 32
        k_i32 = K // 8
        lanes4 = tl.arange(0, 4)
        lanes16 = tl.arange(0, 16)

        w32 = tl.load(
            qw_i32_ptr + offs_r64[:, None] * k_i32 + (pid_g * 4 + lanes4)[None, :],
            mask=mask_r[:, None],
            other=0,
        )
        s = tl.load(scales_ptr + offs_r64 * n_groups + pid_g, mask=mask_r, other=0.0).to(tl.float32)[:, None]
        z = tl.load(zeros_ptr + offs_r64 * n_groups + pid_g, mask=mask_r, other=0.0).to(tl.float32)[:, None]

        lo_ev = tl.join((w32 >> 0) & 0x0F, (w32 >> 16) & 0x0F)
        lo_od = tl.join((w32 >> 8) & 0x0F, (w32 >> 24) & 0x0F)
        q_lo = (tl.reshape(tl.join(lo_ev, lo_od), (BLOCK_R, 16)).to(tl.float32) * s - z).to(tl.float16)

        hi_ev = tl.join((w32 >> 4) & 0x0F, (w32 >> 20) & 0x0F)
        hi_od = tl.join((w32 >> 12) & 0x0F, (w32 >> 28) & 0x0F)
        q_hi = (tl.reshape(tl.join(hi_ev, hi_od), (BLOCK_R, 16)).to(tl.float32) * s - z).to(tl.float16)

        base_out = out_fp16_ptr + offs_r64[:, None] * K + pid_g * 32
        tl.store(base_out + lanes16[None, :], q_lo, mask=mask_r[:, None])
        tl.store(base_out + 16 + lanes16[None, :], q_hi, mask=mask_r[:, None])

    @triton.jit
    def _dequant_w8_chunk_kernel(
        qw_i32_ptr,
        scales_ptr,
        zeros_ptr,
        out_fp16_ptr,
        rc,
        K,
        HAS_ZEROS: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        """Fast coalesced W8 -> FP16 chunk unpacker feeding Turing (`sm_75`) cuBLAS Tensor Cores during prefill."""
        pid_r = tl.program_id(0)
        pid_g = tl.program_id(1)

        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        mask_r = offs_r < rc
        offs_r64 = offs_r.to(tl.int64)

        n_groups = K // 16
        k_i32 = K // 4
        lanes4 = tl.arange(0, 4)
        lanes16 = tl.arange(0, 16)

        w32 = tl.load(
            qw_i32_ptr + offs_r64[:, None] * k_i32 + (pid_g * 4 + lanes4)[None, :],
            mask=mask_r[:, None],
            other=0,
        )
        q0 = ((w32 << 24) >> 24).to(tl.float32)
        q1 = ((w32 << 16) >> 24).to(tl.float32)
        q2 = ((w32 << 8) >> 24).to(tl.float32)
        q3 = (w32 >> 24).to(tl.float32)
        q_16 = tl.reshape(tl.join(tl.join(q0, q2), tl.join(q1, q3)), (BLOCK_R, 16))

        s = tl.load(scales_ptr + offs_r64 * n_groups + pid_g, mask=mask_r, other=0.0).to(tl.float32)[:, None]
        if HAS_ZEROS:
            z = tl.load(zeros_ptr + offs_r64 * n_groups + pid_g, mask=mask_r, other=0.0).to(tl.float32)[:, None]
            w_16 = (q_16 * s - z).to(tl.float16)
        else:
            w_16 = (q_16 * s).to(tl.float16)

        base_out = out_fp16_ptr + offs_r64[:, None] * K + pid_g * 16 + lanes16[None, :]
        tl.store(base_out, w_16, mask=mask_r[:, None])

    @triton.jit
    def _fused_act_mul_rowscale_kernel(
        gate_ptr,
        up_ptr,
        out_fp16_ptr,
        row_scale_ptr,
        M,
        I,
        IS_CONCAT: tl.constexpr,
        USE_GELU: tl.constexpr,
        BLOCK_I: tl.constexpr,
    ):
        """
        Fused SwiGLU / GeGLU + dynamic FP16 overflow row-scaling (`max(|inter|)/16384 >= 1.0`) in 1 launch.
        Replaces 9 eager PyTorch launches per MLP layer with a single L1-cached pass.
        """
        pid_m = tl.program_id(0).to(tl.int64)
        g_row_stride = (2 * I) if IS_CONCAT else I
        g_base = gate_ptr + pid_m * g_row_stride
        u_base = (gate_ptr + pid_m * g_row_stride + I) if IS_CONCAT else (up_ptr + pid_m * I)
        out_base = out_fp16_ptr + pid_m * I

        row_max = tl.zeros((), dtype=tl.float32)
        for col_off in range(0, I, BLOCK_I):
            offs = col_off + tl.arange(0, BLOCK_I)
            mask = offs < I
            g = tl.load(g_base + offs, mask=mask, other=0.0).to(tl.float32)
            u = tl.load(u_base + offs, mask=mask, other=0.0).to(tl.float32)
            if USE_GELU:
                inner = 0.7978845608028654 * (g + 0.044715 * g * g * g)
                inner_c = tl.maximum(tl.minimum(inner, 20.0), -20.0)
                th = (2.0 / (1.0 + tl.exp(-2.0 * inner_c))) - 1.0
                act = 0.5 * g * (1.0 + th)
            else:
                act = g / (1.0 + tl.exp(-g))
            inter = act * u
            row_max = tl.maximum(row_max, tl.max(tl.abs(inter), axis=0))

        scale = tl.maximum(row_max * (1.0 / 16384.0), 1.0)
        inv_scale = 1.0 / scale
        tl.store(row_scale_ptr + pid_m, scale)

        for col_off in range(0, I, BLOCK_I):
            offs = col_off + tl.arange(0, BLOCK_I)
            mask = offs < I
            g = tl.load(g_base + offs, mask=mask, other=0.0).to(tl.float32)
            u = tl.load(u_base + offs, mask=mask, other=0.0).to(tl.float32)
            if USE_GELU:
                inner = 0.7978845608028654 * (g + 0.044715 * g * g * g)
                inner_c = tl.maximum(tl.minimum(inner, 20.0), -20.0)
                th = (2.0 / (1.0 + tl.exp(-2.0 * inner_c))) - 1.0
                act = 0.5 * g * (1.0 + th)
            else:
                act = g / (1.0 + tl.exp(-g))
            scaled = (act * u * inv_scale).to(tl.float16)
            tl.store(out_base + offs, scaled, mask=mask)

    @triton.jit
    def _fused_rmsnorm_silu_gate_kernel(
        y_ptr,
        z_ptr,
        w_ptr,
        out_fp16_ptr,
        D,
        eps,
        BLOCK_D: tl.constexpr,
    ):
        """Fused `RMSNorm(y) * SiLU(z) -> fp16` across `S * H_v` head rows in 1 launch."""
        pid_r = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D
        base = pid_r * D + offs

        y = tl.load(y_ptr + base, mask=mask, other=0.0).to(tl.float32)
        z = tl.load(z_ptr + base, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        rms = tl.rsqrt(tl.sum(y * y, axis=0) * (1.0 / D) + eps)
        silu_z = z / (1.0 + tl.exp(-z))
        out = (y * rms * w * silu_z).to(tl.float16)
        tl.store(out_fp16_ptr + base, out, mask=mask)

    @triton.jit
    def _fused_rmsnorm_kernel(
        x_ptr,
        delta_ptr,
        w_ptr,
        res_out_ptr,
        out_ptr,
        D,
        eps,
        HAS_RESIDUAL: tl.constexpr,
        OUT_FP16: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Single-launch FP32-accurate RMSNorm + optional Residual-Add (`h = x + delta -> fp32`,
        `out = RMSNorm(h, w) -> fp16/fp32`). Collapses 3-5 PyTorch elementwise/cast launches
        per norm (~400+ launches per forward pass on 64-layer models) into 1 register-resident kernel.
        """
        pid_r = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D
        base = pid_r * D + offs

        x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            d = tl.load(delta_ptr + base, mask=mask, other=0.0).to(tl.float32)
            x = x + d
            tl.store(res_out_ptr + base, x, mask=mask)

        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        rms = tl.rsqrt(tl.sum(x * x, axis=0) * (1.0 / D) + eps)
        normed = x * rms * w
        if OUT_FP16:
            tl.store(out_ptr + base, normed.to(tl.float16), mask=mask)
        else:
            tl.store(out_ptr + base, normed, mask=mask)


# ==============================================================================
# 2. GGUF Block Decoders -> GPU-Coalesced SoA Buffers (Streaming Tensor-by-Tensor)
# ==============================================================================

def _unpack_k_scales_12b(scales_12b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Unpack 12-byte K-quant scale/min headers (used by Q4_K and Q5_K) into [n_blocks, 8] uint8 pairs."""
    n_blocks = scales_12b.shape[0]
    s = scales_12b.reshape(n_blocks, 3, 4)
    d_b, m_b, md_b = s[:, 0, :], s[:, 1, :], s[:, 2, :]
    sc = np.concatenate([d_b & 0x3F, (md_b & 0x0F) | ((d_b >> 2) & 0x30)], axis=-1)
    mn = np.concatenate([m_b & 0x3F, (md_b >> 4) | ((m_b >> 2) & 0x30)], axis=-1)
    return sc, mn


def repack_gguf_to_w4a16(
    raw_bytes: np.ndarray,
    qtype: int,
    rows: int,
    cols: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Repack Q4_0, Q4_1, or Q4_K raw GGUF bytes into coalesced GPU SoA tensors:
      qweight: uint8[rows, cols // 2]
      scales : float16[rows, cols // 32]
      zeros  : float16[rows, cols // 32]
    Exact formula: w = scale * q - zero.
    """
    if cols % 32 != 0:
        raise ValueError(f"W4A16 requires cols ({cols}) divisible by 32")

    if qtype == GGMLQuantType.Q4_0:
        blocks = raw_bytes.reshape(-1, 18)
        d = np.frombuffer(np.ascontiguousarray(blocks[:, 0:2]), dtype=np.float16).reshape(rows, cols // 32)
        z = (d.astype(np.float32) * 8.0).astype(np.float16)
        qw = np.ascontiguousarray(blocks[:, 2:18]).reshape(rows, cols // 2)

    elif qtype == GGMLQuantType.Q4_1:
        blocks = raw_bytes.reshape(-1, 20)
        d = np.frombuffer(np.ascontiguousarray(blocks[:, 0:2]), dtype=np.float16).reshape(rows, cols // 32)
        m = np.frombuffer(np.ascontiguousarray(blocks[:, 2:4]), dtype=np.float16).reshape(rows, cols // 32)
        z = (-m.astype(np.float32)).astype(np.float16)
        qw = np.ascontiguousarray(blocks[:, 4:20]).reshape(rows, cols // 2)

    elif qtype == GGMLQuantType.Q4_K:
        if cols % 256 != 0:
            raise ValueError(f"Q4_K requires cols ({cols}) divisible by 256")
        blocks = raw_bytes.reshape(-1, 144)
        n_blocks = blocks.shape[0]
        d = np.frombuffer(np.ascontiguousarray(blocks[:, 0:2]), dtype=np.float16).astype(np.float32).reshape(n_blocks, 1)
        dmin = (
            np.frombuffer(np.ascontiguousarray(blocks[:, 2:4]), dtype=np.float16)
            .astype(np.float32)
            .reshape(n_blocks, 1)
        )
        sc, mn = _unpack_k_scales_12b(blocks[:, 4:16])
        d_sub = (d * sc.astype(np.float32)).astype(np.float16).reshape(rows, cols // 32)
        z_sub = (dmin * mn.astype(np.float32)).astype(np.float16).reshape(rows, cols // 32)

        qs = blocks[:, 16:144].reshape(n_blocks, 4, 32)
        q_even = qs & np.uint8(0x0F)
        q_odd = (qs >> np.uint8(4)) & np.uint8(0x0F)
        q_unpacked = np.stack([q_even, q_odd], axis=2).reshape(n_blocks, 8, 32)
        qw = (q_unpacked[:, :, :16] | (q_unpacked[:, :, 16:] << np.uint8(4))).reshape(rows, cols // 2)
        d, z = d_sub, z_sub

    else:
        # Streaming row-chunked W4A16 repacker for sub-5-bit dynamic quants (IQ4_XS, IQ4_NL, Q3_K, Q2_K, IQ3_*, etc.)
        # Streams 4,096 rows at a time (<85 MB CPU RAM peak even on 248K vocab embeddings) directly onto GPU!
        import gguf.quants as gq

        block_elems, block_bytes = GGML_BLOCK_INFO[int(qtype)]
        gq_type = gq.GGMLQuantizationType(int(qtype))
        qw_gpu = torch.empty((rows, cols // 2), device=device, dtype=torch.uint8)
        sc_gpu = torch.empty((rows, cols // 32), device=device, dtype=torch.float16)
        zr_gpu = torch.empty((rows, cols // 32), device=device, dtype=torch.float16)

        chunk_rows = 4096  # Multiple of 256 ensures exact GGML block alignment for every row chunk
        for r0 in range(0, rows, chunk_rows):
            r1 = min(rows, r0 + chunk_rows)
            rc = r1 - r0
            b0 = ((r0 * cols) // block_elems) * block_bytes
            b1 = ((r1 * cols) // block_elems) * block_bytes
            fp32 = gq.dequantize(raw_bytes[b0:b1], gq_type).reshape(rc, cols // 32, 32)

            w_min = np.min(fp32, axis=-1, keepdims=True)
            w_max = np.max(fp32, axis=-1, keepdims=True)
            scale = (w_max - w_min) * (1.0 / 15.0)
            scale_safe = np.where(scale == 0.0, 1.0, scale)
            q_u8 = np.clip(np.round((fp32 - w_min) / scale_safe), 0, 15).astype(np.uint8)
            qw_chunk = (q_u8[:, :, :16] | (q_u8[:, :, 16:] << np.uint8(4))).reshape(rc, cols // 2)

            qw_gpu[r0:r1].copy_(torch.from_numpy(np.ascontiguousarray(qw_chunk)))
            sc_gpu[r0:r1].copy_(torch.from_numpy(np.ascontiguousarray(scale.squeeze(-1).astype(np.float16))))
            zr_gpu[r0:r1].copy_(torch.from_numpy(np.ascontiguousarray((-w_min.squeeze(-1)).astype(np.float16))))

        return qw_gpu, sc_gpu, zr_gpu

    return (
        torch.from_numpy(np.ascontiguousarray(qw)).to(device=device, dtype=torch.uint8),
        torch.from_numpy(np.ascontiguousarray(d)).to(device=device, dtype=torch.float16),
        torch.from_numpy(np.ascontiguousarray(z)).to(device=device, dtype=torch.float16),
    )


def repack_gguf_to_w8a16(
    raw_bytes: np.ndarray,
    qtype: int,
    rows: int,
    cols: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Repack Q8_0, Q6_K, Q5_K (or any fallback GGUF quant) into coalesced W8A16 GPU tensors:
      qweight: int8[rows, cols]
      scales : float16[rows, cols // 16]
      zeros  : Optional[float16[rows, cols // 16]]
    """
    if cols % 16 != 0:
        raise ValueError(f"W8A16 requires cols ({cols}) divisible by 16")

    if qtype == GGMLQuantType.Q8_0:
        blocks = raw_bytes.reshape(-1, 34)
        d32 = np.frombuffer(np.ascontiguousarray(blocks[:, 0:2]), dtype=np.float16).reshape(-1, 1)
        scales = np.repeat(d32, 2, axis=1).reshape(rows, cols // 16)
        qw = np.ascontiguousarray(blocks[:, 2:34]).view(np.int8).reshape(rows, cols)
        zeros = None

    elif qtype == GGMLQuantType.Q6_K:
        blocks = raw_bytes.reshape(-1, 210)
        n_blocks = blocks.shape[0]
        ql = blocks[:, 0:128].reshape(n_blocks, 2, 1, 64)
        qh = blocks[:, 128:192].reshape(n_blocks, 2, 1, 32)
        sc = np.ascontiguousarray(blocks[:, 192:208]).view(np.int8).astype(np.float32)
        d = np.frombuffer(np.ascontiguousarray(blocks[:, 208:210]), dtype=np.float16).astype(np.float32).reshape(n_blocks, 1)

        ql_u = ((ql >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)) & np.uint8(0x0F)).reshape(n_blocks, 8, 32)
        qh_u = ((qh >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1)) & np.uint8(0x03)).reshape(
            n_blocks, 8, 32
        )
        qw = ((ql_u | (qh_u << np.uint8(4))).astype(np.int8) - np.int8(32)).reshape(rows, cols)
        scales = (d * sc).astype(np.float16).reshape(rows, cols // 16)
        zeros = None

    elif qtype == GGMLQuantType.Q5_K:
        blocks = raw_bytes.reshape(-1, 176)
        n_blocks = blocks.shape[0]
        d = np.frombuffer(np.ascontiguousarray(blocks[:, 0:2]), dtype=np.float16).astype(np.float32).reshape(n_blocks, 1)
        dmin = (
            np.frombuffer(np.ascontiguousarray(blocks[:, 2:4]), dtype=np.float16)
            .astype(np.float32)
            .reshape(n_blocks, 1)
        )
        sc, mn = _unpack_k_scales_12b(blocks[:, 4:16])
        qh_raw = blocks[:, 16:48].reshape(n_blocks, 1, 1, 32)
        qs_raw = blocks[:, 48:176].reshape(n_blocks, 4, 1, 32)

        ql_u = ((qs_raw >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)) & np.uint8(0x0F)).reshape(
            n_blocks, 8, 32
        )
        qh_u = ((qh_raw >> np.arange(8, dtype=np.uint8).reshape(1, 1, 8, 1)) & np.uint8(0x01)).reshape(n_blocks, 8, 32)
        qw = (ql_u | (qh_u << np.uint8(4))).astype(np.int8).reshape(rows, cols)

        scales = np.repeat((d * sc.astype(np.float32)).astype(np.float16), 2, axis=1).reshape(rows, cols // 16)
        zeros = np.repeat((dmin * mn.astype(np.float32)).astype(np.float16), 2, axis=1).reshape(rows, cols // 16)

    else:
        # Streaming row-chunked fallback via gguf.quants.dequantize for 5/8-bit formats (Q5_0, Q5_1, Q8_1, Q8_K)
        import gguf.quants as gq

        block_elems, block_bytes = GGML_BLOCK_INFO[int(qtype)]
        gq_type = gq.GGMLQuantizationType(int(qtype))
        qw_gpu = torch.empty((rows, cols), device=device, dtype=torch.int8)
        sc_gpu = torch.empty((rows, cols // 16), device=device, dtype=torch.float16)

        chunk_rows = 4096
        for r0 in range(0, rows, chunk_rows):
            r1 = min(rows, r0 + chunk_rows)
            rc = r1 - r0
            b0 = ((r0 * cols) // block_elems) * block_bytes
            b1 = ((r1 * cols) // block_elems) * block_bytes
            grouped = gq.dequantize(raw_bytes[b0:b1], gq_type).reshape(rc, cols // 16, 16)
            amax = np.max(np.abs(grouped), axis=-1, keepdims=True)
            sc_f32 = np.where(amax == 0, 1.0, amax / 127.0)
            qw_chunk = np.clip(np.round(grouped / sc_f32), -127, 127).astype(np.int8).reshape(rc, cols)
            qw_gpu[r0:r1].copy_(torch.from_numpy(np.ascontiguousarray(qw_chunk)))
            sc_gpu[r0:r1].copy_(torch.from_numpy(np.ascontiguousarray(sc_f32.squeeze(-1).astype(np.float16))))

        return qw_gpu, sc_gpu, None

    return (
        torch.from_numpy(np.ascontiguousarray(qw)).to(device=device, dtype=torch.int8),
        torch.from_numpy(np.ascontiguousarray(scales)).to(device=device, dtype=torch.float16),
        torch.from_numpy(np.ascontiguousarray(zeros)).to(device=device, dtype=torch.float16)
        if zeros is not None
        else None,
    )


# ==============================================================================
# 2b. Parallel row-split repack (GGML blocks never span rows, so any row split is exact)
# ==============================================================================

_ROW_SPLIT_MIN_BYTES = 64 * 1024 * 1024


def _load_workers() -> int:
    """Worker count for parallel weight repack/layer load. Override with `PYGGUF_LOAD_WORKERS` (1..16)."""
    raw = os.environ.get("PYGGUF_LOAD_WORKERS", "").strip()
    if raw:
        try:
            return max(1, min(16, int(raw)))
        except ValueError:
            print(f"[PyGGUF] Ignoring invalid PYGGUF_LOAD_WORKERS={raw!r} (expected an integer 1..16)")
    return max(1, min(4, (os.cpu_count() or 1)))


def _repack_parallel(
    raw_bytes: np.ndarray,
    qtype: int,
    rows: int,
    cols: int,
    device: torch.device,
    workers: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Split a large quantized tensor into contiguous row chunks, repack each chunk on its own
    thread, and concatenate. Bitwise-identical to the serial path (row-independent blocks)."""
    block_elems, block_bytes = GGML_BLOCK_INFO[int(qtype)]
    byte_per_row = (cols // block_elems) * block_bytes
    chunk = (rows + workers - 1) // workers
    bounds = [(r0, min(rows, r0 + chunk)) for r0 in range(0, rows, chunk)]
    jobs = [(raw_bytes[r0 * byte_per_row : r1 * byte_per_row], r1 - r0) for r0, r1 in bounds]

    def _job(pair: Tuple[np.ndarray, int]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        buf, nrows = pair
        if qtype in SUB_5BIT_W4_QTYPES and cols % 32 == 0:
            if qtype != GGMLQuantType.Q4_K or cols % 256 == 0:
                return repack_gguf_to_w4a16(buf, qtype, nrows, cols, device)
        return repack_gguf_to_w8a16(buf, qtype, nrows, cols, device)

    if len(jobs) <= 1:
        return _job(jobs[0])
    with ThreadPoolExecutor(max_workers=len(jobs), thread_name_prefix="pygguf-repack") as pool:
        parts = list(pool.map(_job, jobs))
    with cuda_device_scope(device):
        qw = torch.cat([p[0] for p in parts], dim=0)
        sc = torch.cat([p[1] for p in parts], dim=0)
        zr_parts = [p[2] for p in parts]
        if all(z is not None for z in zr_parts):
            zr: Optional[torch.Tensor] = torch.cat([z for z in zr_parts if z is not None], dim=0)
        elif any(z is not None for z in zr_parts):
            raise RuntimeError("row-split repack produced inconsistent zeros across chunks")
        else:
            zr = None
    return qw, sc, zr


def load_float_tensor(gguf_file: GGUFFile, name: str, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Load a 1D/2D unquantized (or small dequantized) tensor such as RMSNorm weights, biases, conv1d, or router gates."""
    info = gguf_file.tensors[name]
    raw = gguf_file.get_tensor_bytes(name)
    if info.qtype == GGMLQuantType.F32:
        arr = np.frombuffer(raw, dtype=np.float32).reshape(info.shape).copy()
        return torch.from_numpy(arr).to(device=device, dtype=dtype)
    if info.qtype == GGMLQuantType.F16:
        arr = np.frombuffer(raw, dtype=np.float16).reshape(info.shape).copy()
        return torch.from_numpy(arr).to(device=device, dtype=dtype)
    if info.qtype == GGMLQuantType.BF16:
        u16 = np.frombuffer(raw, dtype=np.uint16).reshape(info.shape).copy()
        t_bf16 = torch.from_numpy(u16).view(torch.bfloat16)
        return t_bf16.to(device=device, dtype=dtype)

    # Small quantized tensor (including narrow shapes like conv1d [C, 4]) -> direct dequantize
    import gguf.quants as gq

    fp32 = gq.dequantize(raw, gq.GGMLQuantizationType(int(info.qtype))).reshape(info.shape).copy()
    return torch.from_numpy(fp32).to(device=device, dtype=dtype)


# ==============================================================================
# 3. PyTorch / Triton Execution Helpers (Multi-GPU Safe + sm_75 cuBLAS Prefill)
# ==============================================================================

_CUDA_CC_MAJOR: Dict[torch.device, int] = {}
_DEQUANT_SCRATCH: Dict[torch.device, torch.Tensor] = {}


@contextmanager
def cuda_device_scope(dev: torch.device) -> Iterator[None]:
    """
    Eagerly bind the NVIDIA CUDA driver `CUcontext` (`torch.cuda.set_device`) AND PyTorch's
    device context so OpenAI Triton's `cuPointerGetAttribute` (`driver.c`) always validates
    and launches kernels on `dev`'s primary context — even across Dual T4 GPUs without PCIe P2P
    and inside background FastAPI worker threads!
    """
    if dev.type != "cuda":
        yield
        return
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    prev = torch.cuda.current_device()
    torch.cuda.set_device(idx)
    try:
        with torch.cuda.device(idx):
            yield
    finally:
        if prev != idx:
            torch.cuda.set_device(prev)


def _get_cuda_major(dev: torch.device) -> int:
    if dev not in _CUDA_CC_MAJOR:
        major, _ = torch.cuda.get_device_capability(dev)
        _CUDA_CC_MAJOR[dev] = int(major)
    return _CUDA_CC_MAJOR[dev]


def _get_scratch_fp16(dev: torch.device, rows: int, cols: int) -> torch.Tensor:
    needed = rows * cols
    buf = _DEQUANT_SCRATCH.get(dev)
    if buf is None or buf.numel() < needed:
        buf = torch.empty((needed,), device=dev, dtype=torch.float16)
        _DEQUANT_SCRATCH[dev] = buf
    return buf[:needed].view(rows, cols)


def exec_w4a16_matmul(
    x_2d: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    qw_i32_cached: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Execute x_2d [M, K] @ W^T [K, N] -> [M, N] using Triton on `qweight.device` or PyTorch on CPU."""
    M, K = x_2d.shape
    N = qweight.shape[0]
    dev = qweight.device
    if x_2d.device != dev:
        x_2d = x_2d.to(dev, non_blocking=True)

    if HAS_TRITON and dev.type == "cuda":
        with cuda_device_scope(dev):
            x_fp16 = x_2d if x_2d.dtype == torch.float16 else x_2d.to(torch.float16)
            x_contig = x_fp16 if x_fp16.is_contiguous() else x_fp16.contiguous()
            qw_i32 = qw_i32_cached if qw_i32_cached is not None else qweight.contiguous().view(torch.int32)
            has_bias = bias is not None
            b_ptr = bias if bias is not None else scales

            if M == 1:
                out = torch.empty((1, N), device=dev, dtype=torch.float16)
                BLOCK_N = 4
                _w4a16_gemv_kernel[(triton.cdiv(N, BLOCK_N), 1)](
                    x_contig,
                    qw_i32,
                    scales,
                    zeros,
                    b_ptr,
                    out,
                    1,
                    N,
                    K,
                    HAS_BIAS=has_bias,
                    BLOCK_N=BLOCK_N,
                    num_warps=4,
                )
                return out if x_2d.dtype == torch.float16 else out.to(x_2d.dtype)

            if M <= 8:
                out = torch.empty((M, N), device=dev, dtype=torch.float16)
                BLOCK_M = 2 if M <= 2 else (4 if M <= 4 else 8)
                BLOCK_N = 4
                _w4a16_multirow_gemv_kernel[(triton.cdiv(N, BLOCK_N),)](
                    x_contig,
                    qw_i32,
                    scales,
                    zeros,
                    b_ptr,
                    out,
                    M,
                    N,
                    K,
                    HAS_BIAS=has_bias,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    num_warps=4,
                )
                return out if x_2d.dtype == torch.float16 else out.to(x_2d.dtype)

            # Prompt Prefill (M > 8):
            # On Turing (`sm_75` Tesla T4, major < 8), Triton lowers `tl.dot` to scalar FP32 FMAs;
            # unpacking 4096-row chunks via Triton into a tiny reusable scratch buffer and calling
            # `F.linear` hits real Turing FP16 Tensor Cores (~58 TFLOPS, >10x faster prefill!).
            if _get_cuda_major(dev) < 8:
                out = torch.empty((M, N), device=dev, dtype=torch.float16)
                step = 4096
                n_groups = K // 32
                for r0 in range(0, N, step):
                    r1 = min(N, r0 + step)
                    rc = r1 - r0
                    w_chunk = _get_scratch_fp16(dev, rc, K)
                    _dequant_w4_chunk_kernel[(triton.cdiv(rc, 16), n_groups)](
                        qw_i32[r0:r1],
                        scales[r0:r1],
                        zeros[r0:r1],
                        w_chunk,
                        rc,
                        K,
                        BLOCK_R=16,
                        num_warps=2,
                    )
                    b_sub = bias[r0:r1] if bias is not None else None
                    out[:, r0:r1] = F.linear(x_contig, w_chunk, b_sub)
                return out.to(x_2d.dtype)

            out = torch.empty((M, N), device=dev, dtype=torch.float16)
            BLOCK_M = 32 if M < 64 else 64
            BLOCK_N = 64
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            _w4a16_gemm_tc_kernel[grid](
                x_contig,
                qw_i32,
                scales,
                zeros,
                b_ptr,
                out,
                M,
                N,
                K,
                HAS_BIAS=has_bias,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                num_warps=4,
                num_stages=2,
            )
            return out.to(x_2d.dtype)

    # Vectorized row-chunked CPU / non-Triton fallback (<16 MB peak scratch RAM even on 250K vocabs)
    n_groups = K // 32
    x_f32 = x_2d.float()
    out_cpu = torch.empty((M, N), device=x_2d.device, dtype=x_2d.dtype)
    step = 4096
    for r0 in range(0, N, step):
        r1 = min(N, r0 + step)
        rc = r1 - r0
        packed = qweight[r0:r1].view(rc, n_groups, 16)
        q_lo = (packed & 0x0F).to(torch.float32)
        q_hi = ((packed >> 4) & 0x0F).to(torch.float32)
        q_full = torch.cat([q_lo, q_hi], dim=-1)
        w = (q_full * scales[r0:r1].float().unsqueeze(-1) - zeros[r0:r1].float().unsqueeze(-1)).reshape(rc, K)
        b = bias[r0:r1].float() if bias is not None else None
        out_cpu[:, r0:r1] = F.linear(x_f32, w, b).to(x_2d.dtype)
    return out_cpu


def exec_w8a16_matmul(
    x_2d: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    qw_i32_cached: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Execute x_2d [M, K] @ W^T [K, N] -> [M, N] for W8A16 weights."""
    M, K = x_2d.shape
    N = qweight.shape[0]
    dev = qweight.device
    if x_2d.device != dev:
        x_2d = x_2d.to(dev, non_blocking=True)

    if HAS_TRITON and dev.type == "cuda":
        with cuda_device_scope(dev):
            x_fp16 = x_2d if x_2d.dtype == torch.float16 else x_2d.to(torch.float16)
            x_contig = x_fp16 if x_fp16.is_contiguous() else x_fp16.contiguous()
            qw_i32 = qw_i32_cached if qw_i32_cached is not None else qweight.contiguous().view(torch.int32)
            has_zeros = zeros is not None
            has_bias = bias is not None
            z_ptr = zeros if zeros is not None else scales
            b_ptr = bias if bias is not None else scales

            if M == 1:
                out = torch.empty((1, N), device=dev, dtype=torch.float16)
                BLOCK_N = 4
                _w8a16_gemv_kernel[(triton.cdiv(N, BLOCK_N), 1)](
                    x_contig,
                    qw_i32,
                    scales,
                    z_ptr,
                    b_ptr,
                    out,
                    1,
                    N,
                    K,
                    HAS_ZEROS=has_zeros,
                    HAS_BIAS=has_bias,
                    BLOCK_N=BLOCK_N,
                    num_warps=4,
                )
                return out if x_2d.dtype == torch.float16 else out.to(x_2d.dtype)

            if M <= 8:
                out = torch.empty((M, N), device=dev, dtype=torch.float16)
                BLOCK_M = 2 if M <= 2 else (4 if M <= 4 else 8)
                BLOCK_N = 4
                _w8a16_multirow_gemv_kernel[(triton.cdiv(N, BLOCK_N),)](
                    x_contig,
                    qw_i32,
                    scales,
                    z_ptr,
                    b_ptr,
                    out,
                    M,
                    N,
                    K,
                    HAS_ZEROS=has_zeros,
                    HAS_BIAS=has_bias,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    num_warps=4,
                )
                return out if x_2d.dtype == torch.float16 else out.to(x_2d.dtype)

            if _get_cuda_major(dev) < 8:
                out = torch.empty((M, N), device=dev, dtype=torch.float16)
                step = 4096
                n_groups = K // 16
                for r0 in range(0, N, step):
                    r1 = min(N, r0 + step)
                    rc = r1 - r0
                    w_chunk = _get_scratch_fp16(dev, rc, K)
                    z_sub = zeros[r0:r1] if zeros is not None else scales[r0:r1]
                    _dequant_w8_chunk_kernel[(triton.cdiv(rc, 16), n_groups)](
                        qw_i32[r0:r1],
                        scales[r0:r1],
                        z_sub,
                        w_chunk,
                        rc,
                        K,
                        HAS_ZEROS=has_zeros,
                        BLOCK_R=16,
                        num_warps=2,
                    )
                    b_sub = bias[r0:r1] if bias is not None else None
                    out[:, r0:r1] = F.linear(x_contig, w_chunk, b_sub)
                return out.to(x_2d.dtype)

            out = torch.empty((M, N), device=dev, dtype=torch.float16)
            BLOCK_M = 32 if M < 64 else 64
            BLOCK_N = 64
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            _w8a16_gemm_tc_kernel[grid](
                x_contig,
                qw_i32,
                scales,
                z_ptr,
                b_ptr,
                out,
                M,
                N,
                K,
                HAS_ZEROS=has_zeros,
                HAS_BIAS=has_bias,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                num_warps=4,
                num_stages=2,
            )
            return out.to(x_2d.dtype)

    n_groups = K // 16
    x_f32 = x_2d.float()
    out_cpu = torch.empty((M, N), device=x_2d.device, dtype=x_2d.dtype)
    step = 4096
    for r0 in range(0, N, step):
        r1 = min(N, r0 + step)
        rc = r1 - r0
        w = qweight[r0:r1].float().view(rc, n_groups, 16) * scales[r0:r1].float().unsqueeze(-1)
        if zeros is not None:
            w = w - zeros[r0:r1].float().unsqueeze(-1)
        b = bias[r0:r1].float() if bias is not None else None
        out_cpu[:, r0:r1] = F.linear(x_f32, w.reshape(rc, K), b).to(x_2d.dtype)
    return out_cpu


def exec_fused_act_mul_rowscale(
    gate_or_concat: torch.Tensor,
    up: Optional[torch.Tensor] = None,
    use_gelu: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute `inter = act(gate) * up` + dynamic FP16 overflow row-scaling `(inter / row_scale).to(fp16)`
    in a single Triton kernel when on CUDA (or vectorized PyTorch fallback on CPU).
    - If `up is None`, `gate_or_concat` has shape `[..., 2 * I]`.
    - Returns `(down_in_fp16 [..., I], row_scale_fp32 [..., 1])`.
    """
    orig_prefix = gate_or_concat.shape[:-1]
    if up is None:
        I = gate_or_concat.shape[-1] // 2
        g_2d = gate_or_concat.reshape(-1, 2 * I)
        u_2d = None
    else:
        I = gate_or_concat.shape[-1]
        g_2d = gate_or_concat.reshape(-1, I)
        u_2d = up.reshape(-1, I)
    M = g_2d.shape[0]
    dev = g_2d.device

    if HAS_TRITON and dev.type == "cuda":
        with cuda_device_scope(dev):
            g_c = g_2d if g_2d.is_contiguous() else g_2d.contiguous()
            u_c = g_c if u_2d is None else (u_2d if u_2d.is_contiguous() else u_2d.contiguous())
            out_fp16 = torch.empty((M, I), device=dev, dtype=torch.float16)
            row_scale = torch.empty((M, 1), device=dev, dtype=torch.float32)
            BLOCK_I = 1024 if I >= 1024 else triton.next_power_of_2(max(32, I))
            _fused_act_mul_rowscale_kernel[(M,)](
                g_c,
                u_c,
                out_fp16,
                row_scale,
                M,
                I,
                IS_CONCAT=(u_2d is None),
                USE_GELU=use_gelu,
                BLOCK_I=BLOCK_I,
                num_warps=8 if BLOCK_I >= 1024 else 4,
            )
            return out_fp16.reshape(*orig_prefix, I), row_scale.reshape(*orig_prefix, 1)

    if u_2d is None:
        g_f, u_f = g_2d.float().chunk(2, dim=-1)
    else:
        g_f, u_f = g_2d.float(), u_2d.float()
    act = F.gelu(g_f, approximate="tanh") if use_gelu else F.silu(g_f)
    inter = act * u_f
    r_scale = (inter.abs().amax(dim=-1, keepdim=True) / 16384.0).clamp(min=1.0)
    down_in = (inter / r_scale).to(torch.float16)
    return down_in.reshape(*orig_prefix, I), r_scale.reshape(*orig_prefix, 1)


def exec_fused_rmsnorm_silu_gate(
    y: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute `(RMSNorm(y, norm_weight, eps) * SiLU(z)).to(float16)` in 1 Triton launch."""
    orig_shape = y.shape
    D = orig_shape[-1]
    R = y.numel() // D
    dev = y.device

    if HAS_TRITON and dev.type == "cuda" and D <= 2048:
        with cuda_device_scope(dev):
            y_c = y.reshape(R, D).contiguous()
            z_c = z.reshape(R, D).contiguous()
            out = torch.empty((R, D), device=dev, dtype=torch.float16)
            BLOCK_D = triton.next_power_of_2(max(32, D))
            _fused_rmsnorm_silu_gate_kernel[(R,)](
                y_c,
                z_c,
                norm_weight,
                out,
                D,
                eps,
                BLOCK_D=BLOCK_D,
                num_warps=4 if BLOCK_D >= 256 else 2,
            )
            return out.reshape(orig_shape)

    normed = F.rms_norm(y.float(), (D,), norm_weight, eps)
    return (normed * F.silu(z.float())).to(torch.float16).reshape(orig_shape)


def exec_fused_rmsnorm(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float = 1e-6,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Compute FP32-accurate `RMSNorm(x, norm_weight, eps).to(out_dtype)` in 1 Triton launch."""
    target_dtype = out_dtype if out_dtype is not None else x.dtype
    orig_shape = x.shape
    D = orig_shape[-1]
    R = x.numel() // D
    dev = x.device

    if HAS_TRITON and dev.type == "cuda" and D <= 8192 and target_dtype in (torch.float16, torch.float32):
        with cuda_device_scope(dev):
            x_c = x.reshape(R, D)
            if not x_c.is_contiguous():
                x_c = x_c.contiguous()
            out = torch.empty((R, D), device=dev, dtype=target_dtype)
            BLOCK_D = triton.next_power_of_2(max(32, D))
            num_warps = 2 if BLOCK_D <= 128 else (4 if BLOCK_D <= 2048 else 8)
            _fused_rmsnorm_kernel[(R,)](
                x_c,
                x_c,
                norm_weight,
                x_c,
                out,
                D,
                eps,
                HAS_RESIDUAL=False,
                OUT_FP16=(target_dtype == torch.float16),
                BLOCK_D=BLOCK_D,
                num_warps=num_warps,
            )
            return out.reshape(orig_shape)

    normed = F.rms_norm(x.float(), (D,), norm_weight, eps)
    return normed.to(target_dtype)


def exec_fused_add_rmsnorm(
    x: torch.Tensor,
    delta: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float = 1e-6,
    out_dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute `h = x.float() + delta.float()` AND `normed = RMSNorm(h, norm_weight, eps).to(out_dtype)`
    in a single Triton launch. Returns `(h_fp32, normed)`.
    """
    orig_shape = x.shape
    D = orig_shape[-1]
    R = x.numel() // D
    dev = x.device

    if HAS_TRITON and dev.type == "cuda" and D <= 8192 and out_dtype in (torch.float16, torch.float32):
        with cuda_device_scope(dev):
            x_c = x.reshape(R, D)
            if not x_c.is_contiguous():
                x_c = x_c.contiguous()
            d_c = delta.reshape(R, D)
            if not d_c.is_contiguous():
                d_c = d_c.contiguous()
            res_out = torch.empty((R, D), device=dev, dtype=torch.float32)
            out = torch.empty((R, D), device=dev, dtype=out_dtype)
            BLOCK_D = triton.next_power_of_2(max(32, D))
            num_warps = 2 if BLOCK_D <= 128 else (4 if BLOCK_D <= 2048 else 8)
            _fused_rmsnorm_kernel[(R,)](
                x_c,
                d_c,
                norm_weight,
                res_out,
                out,
                D,
                eps,
                HAS_RESIDUAL=True,
                OUT_FP16=(out_dtype == torch.float16),
                BLOCK_D=BLOCK_D,
                num_warps=num_warps,
            )
            return res_out.reshape(orig_shape), out.reshape(orig_shape)

    h = x.float() + delta.float()
    normed = F.rms_norm(h, (D,), norm_weight, eps).to(out_dtype)
    return h, normed


# ==============================================================================
# 4. Quantized Linear, Horizontal Weight Fusion, TP Slicing, & 3D Sparse MoE
# ==============================================================================

class GGUFQuantLinear(nn.Module):
    """
    Drop-in quantized linear layer directly loaded from any GGUF tensor type
    (W4A16 for 4-bit quants, W8A16 for 5/6/8-bit quants, FP16 for unquantized).
    Supports zero-VRAM horizontal fusion (`fuse`) and group-exact Tensor Parallel slicing!
    """

    def __init__(
        self,
        mode: str,
        in_features: int,
        out_features: int,
        qweight: torch.Tensor,
        scales: Optional[torch.Tensor] = None,
        zeros: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.mode = mode  # "w4", "w8", or "fp16"
        self.in_features = in_features
        self.out_features = out_features
        qw_c = qweight.contiguous()
        self.register_buffer("qweight", qw_c, persistent=False)
        self.register_buffer("scales", scales.contiguous() if scales is not None else None, persistent=False)
        self.register_buffer("zeros", zeros.contiguous() if zeros is not None else None, persistent=False)
        self.register_buffer("bias", bias.contiguous() if bias is not None else None, persistent=False)
        self.qweight_i32: Optional[torch.Tensor] = (
            qw_c.view(torch.int32) if mode in ("w4", "w8") and qw_c.shape[-1] % 4 == 0 else None
        )

    @classmethod
    def fuse(cls, linears: Sequence["GGUFQuantLinear"]) -> "GGUFQuantLinear":
        """Concatenate 2+ compatible `GGUFQuantLinear` modules along `out_features` (`dim=0`)."""
        if len(linears) == 1:
            return linears[0]
        first = linears[0]
        mode = first.mode
        K = first.in_features
        dev = first.qweight.device
        for lin in linears[1:]:
            if lin.mode != mode or lin.in_features != K or lin.qweight.device != dev:
                raise ValueError("Cannot fuse GGUFQuantLinear modules with mismatched mode/in_features/device")

        total_out = sum(lin.out_features for lin in linears)
        qw_cat = torch.cat([lin.qweight for lin in linears], dim=0)

        sc_cat: Optional[torch.Tensor] = None
        zr_cat: Optional[torch.Tensor] = None
        if mode in ("w4", "w8"):
            sc_cat = torch.cat([lin.scales for lin in linears], dim=0)  # type: ignore[arg-type]
            if any(lin.zeros is not None for lin in linears):
                grp = K // (32 if mode == "w4" else 16)
                zr_parts = [
                    lin.zeros
                    if lin.zeros is not None
                    else torch.zeros((lin.out_features, grp), device=dev, dtype=torch.float16)
                    for lin in linears
                ]
                zr_cat = torch.cat(zr_parts, dim=0)

        bias_cat: Optional[torch.Tensor] = None
        if any(lin.bias is not None for lin in linears):
            b_parts = [
                lin.bias
                if lin.bias is not None
                else torch.zeros((lin.out_features,), device=dev, dtype=torch.float16)
                for lin in linears
            ]
            bias_cat = torch.cat(b_parts, dim=0)

        return cls(mode, K, total_out, qw_cat, sc_cat, zr_cat, bias=bias_cat)

    def slice_output_rows(
        self,
        row_sel: Union[slice, Sequence[int], torch.Tensor],
        target_device: torch.device,
    ) -> "GGUFQuantLinear":
        """Extract a subset of output rows (`dim=0`) onto `target_device` with zero re-quantization."""
        if isinstance(row_sel, slice):
            qw = self.qweight[row_sel].to(target_device)
            sc = self.scales[row_sel].to(target_device) if self.scales is not None else None
            zr = self.zeros[row_sel].to(target_device) if self.zeros is not None else None
            bi = self.bias[row_sel].to(target_device) if self.bias is not None else None
        else:
            idx = (
                row_sel.to(self.qweight.device, dtype=torch.long)
                if isinstance(row_sel, torch.Tensor)
                else torch.tensor(row_sel, device=self.qweight.device, dtype=torch.long)
            )
            qw = self.qweight.index_select(0, idx).to(target_device)
            sc = self.scales.index_select(0, idx).to(target_device) if self.scales is not None else None
            zr = self.zeros.index_select(0, idx).to(target_device) if self.zeros is not None else None
            bi = self.bias.index_select(0, idx).to(target_device) if self.bias is not None else None
        return GGUFQuantLinear(self.mode, self.in_features, int(qw.shape[0]), qw, sc, zr, bias=bi)

    def slice_input_blocks(
        self,
        block_indices: Sequence[int],
        block_cols: int,
        target_device: torch.device,
        keep_bias: bool = True,
    ) -> "GGUFQuantLinear":
        """
        Extract column blocks (`dim=1`, where `block_cols % 32 == 0`) onto `target_device`
        for exact Row-Parallel Tensor Parallelism (`o_proj`, `down_proj`, and tiled `ssm_out`).
        """
        if block_cols % 32 != 0 or self.in_features % block_cols != 0:
            raise ValueError(f"block_cols={block_cols} must be divisible by 32 and divide K={self.in_features}")
        n_total_blk = self.in_features // block_cols
        n_sel = len(block_indices)
        new_K = n_sel * block_cols
        N = self.out_features
        # Common TP row-parallel case: one consecutive range. Transfer a strided
        # view directly rather than first materializing an index_select on the host.
        # Keep the original path for same-device slicing (which must own its storage).
        blocks = list(block_indices)
        if (self.qweight.device != target_device and blocks
                and 0 <= blocks[0] and blocks[-1] < n_total_blk
                and blocks == list(range(blocks[0], blocks[0] + n_sel))):
            start = blocks[0] * block_cols
            end = start + new_K
            qdiv = 2 if self.mode == "w4" else 1
            qw = self.qweight[:, start // qdiv:end // qdiv].to(target_device).contiguous()
            group = 32 if self.mode == "w4" else 16
            sc = (self.scales[:, start // group:end // group].to(target_device).contiguous()
                  if self.scales is not None else None)
            zr = (self.zeros[:, start // group:end // group].to(target_device).contiguous()
                  if self.zeros is not None else None)
            bi = self.bias.to(target_device) if keep_bias and self.bias is not None else None
            return GGUFQuantLinear(self.mode, new_K, N, qw, sc, zr, bias=bi)
        idx = torch.tensor(blocks, device=self.qweight.device, dtype=torch.long)

        if self.mode == "w4":
            assert self.scales is not None and self.zeros is not None
            qw = (
                self.qweight.view(N, n_total_blk, block_cols // 2)
                .index_select(1, idx)
                .reshape(N, new_K // 2)
                .to(target_device)
            )
            sc = (
                self.scales.view(N, n_total_blk, block_cols // 32)
                .index_select(1, idx)
                .reshape(N, new_K // 32)
                .to(target_device)
            )
            zr = (
                self.zeros.view(N, n_total_blk, block_cols // 32)
                .index_select(1, idx)
                .reshape(N, new_K // 32)
                .to(target_device)
            )
        elif self.mode == "w8":
            assert self.scales is not None
            qw = (
                self.qweight.view(N, n_total_blk, block_cols)
                .index_select(1, idx)
                .reshape(N, new_K)
                .to(target_device)
            )
            sc = (
                self.scales.view(N, n_total_blk, block_cols // 16)
                .index_select(1, idx)
                .reshape(N, new_K // 16)
                .to(target_device)
            )
            zr = (
                self.zeros.view(N, n_total_blk, block_cols // 16)
                .index_select(1, idx)
                .reshape(N, new_K // 16)
                .to(target_device)
                if self.zeros is not None
                else None
            )
        else:
            qw = (
                self.qweight.view(N, n_total_blk, block_cols)
                .index_select(1, idx)
                .reshape(N, new_K)
                .to(target_device)
            )
            sc, zr = None, None

        bi = self.bias.to(target_device) if (keep_bias and self.bias is not None) else None
        return GGUFQuantLinear(self.mode, new_K, N, qw, sc, zr, bias=bi)

    @classmethod
    def from_gguf(
        cls,
        gguf_file: GGUFFile,
        weight_name: str,
        bias_name: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
        workers: Optional[int] = None,
    ) -> "GGUFQuantLinear":
        info: GGUFTensorInfo = gguf_file.tensors[weight_name]
        out_features, in_features = int(info.shape[0]), int(info.shape[1])
        raw = gguf_file.get_tensor_bytes(weight_name)

        bias_t: Optional[torch.Tensor] = None
        if bias_name and bias_name in gguf_file.tensors:
            bias_t = load_float_tensor(gguf_file, bias_name, device=device, dtype=torch.float16)

        if info.qtype in (GGMLQuantType.F16, GGMLQuantType.F32, GGMLQuantType.BF16):
            w_fp16 = load_float_tensor(gguf_file, weight_name, device=device, dtype=torch.float16)
            return cls("fp16", in_features, out_features, w_fp16, bias=bias_t)

        n_workers = _load_workers() if workers is None else max(1, int(workers))
        use_w4 = info.qtype in SUB_5BIT_W4_QTYPES and in_features % 32 == 0 and (
            info.qtype != GGMLQuantType.Q4_K or in_features % 256 == 0
        )
        if n_workers > 1 and raw.nbytes >= _ROW_SPLIT_MIN_BYTES:
            # Row-split the repack for vocab-sized weights (token_embd / untied output head);
            # bitwise-identical to the serial path.
            qw, sc, zr = _repack_parallel(raw, info.qtype, out_features, in_features, device, n_workers)
        elif use_w4:
            qw, sc, zr = repack_gguf_to_w4a16(raw, info.qtype, out_features, in_features, device)
        else:
            qw, sc, zr = repack_gguf_to_w8a16(raw, info.qtype, out_features, in_features, device)
        mode = "w4" if use_w4 else "w8"
        return cls(mode, in_features, out_features, qw, sc, zr, bias=bias_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)
        if self.mode == "w4":
            assert self.scales is not None and self.zeros is not None
            out = exec_w4a16_matmul(x_2d, self.qweight, self.scales, self.zeros, self.bias, self.qweight_i32)
        elif self.mode == "w8":
            assert self.scales is not None
            out = exec_w8a16_matmul(x_2d, self.qweight, self.scales, self.zeros, self.bias, self.qweight_i32)
        else:
            out = F.linear(x_2d.to(device=self.qweight.device, dtype=torch.float16), self.qweight, self.bias).to(x.dtype)
        return out.reshape(*orig_shape[:-1], self.out_features)

    def embed_lookup(self, token_ids: torch.Tensor, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """
        Dequantizes ONLY the active token rows for embedding lookup (~4x VRAM savings on 152K-256K vocabs),
        allowing the exact same quantized buffer to serve as both `token_embd` AND tied `lm_head`!
        """
        flat_ids = token_ids.reshape(-1)
        K = self.in_features
        if self.mode == "fp16":
            rows = self.qweight.index_select(0, flat_ids).to(dtype)
        elif self.mode == "w4":
            assert self.scales is not None and self.zeros is not None
            packed = self.qweight.index_select(0, flat_ids).view(-1, K // 32, 16)
            sc = self.scales.index_select(0, flat_ids).to(dtype).unsqueeze(-1)
            zr = self.zeros.index_select(0, flat_ids).to(dtype).unsqueeze(-1)
            q_lo = (packed & 0x0F).to(dtype)
            q_hi = ((packed >> 4) & 0x0F).to(dtype)
            rows = (torch.cat([q_lo, q_hi], dim=-1) * sc - zr).reshape(-1, K)
        else:
            assert self.scales is not None
            qw = self.qweight.index_select(0, flat_ids).to(dtype).view(-1, K // 16, 16)
            sc = self.scales.index_select(0, flat_ids).to(dtype).unsqueeze(-1)
            val = qw * sc
            if self.zeros is not None:
                val = val - self.zeros.index_select(0, flat_ids).to(dtype).unsqueeze(-1)
            rows = val.reshape(-1, K)
        return rows.reshape(*token_ids.shape, K)


class FusedLinearPack(nn.Module):
    """
    Automatically fuses sibling `GGUFQuantLinear` projections sharing the same input `x`
    by grouping compatible quant modes (`"w4"`, `"w8"`, `"fp16"`) into minimal GEMV/GEMM launches.
    Consumes 0 bytes of extra VRAM because un-fused tensors are immediately released.
    """

    def __init__(self, linears: Sequence[GGUFQuantLinear]) -> None:
        super().__init__()
        self.num_outputs = len(linears)
        self.out_sizes = [lin.out_features for lin in linears]

        # Group indices by `mode` while preserving deterministic reconstruction order
        mode_to_indices: Dict[str, List[int]] = {}
        for idx, lin in enumerate(linears):
            mode_to_indices.setdefault(lin.mode, []).append(idx)

        self.groups = nn.ModuleList()
        self.group_slots: List[Tuple[List[int], List[int]]] = []
        for _mode, idxs in mode_to_indices.items():
            fused = GGUFQuantLinear.fuse([linears[i] for i in idxs])
            self.groups.append(fused)
            self.group_slots.append((idxs, [linears[i].out_features for i in idxs]))

        self.is_single_contiguous = (
            len(self.groups) == 1 and self.group_slots[0][0] == list(range(self.num_outputs))
        )

    def forward_single_concat(self, x: torch.Tensor) -> torch.Tensor:
        """Fast path when all outputs fused into 1 contiguous matrix (e.g. `[gate | up]`)."""
        assert self.is_single_contiguous
        return self.groups[0](x)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.is_single_contiguous:
            cat_out = self.groups[0](x)
            if self.num_outputs == 1:
                return [cat_out]
            return list(torch.split(cat_out, self.out_sizes, dim=-1))

        results: List[Optional[torch.Tensor]] = [None] * self.num_outputs
        for grp_mod, (idxs, sizes) in zip(self.groups, self.group_slots):
            grp_out = grp_mod(x)
            if len(idxs) == 1:
                results[idxs[0]] = grp_out
            else:
                for i_slot, part in zip(idxs, torch.split(grp_out, sizes, dim=-1)):
                    results[i_slot] = part
        return [r for r in results if r is not None]


class GGUFQuantMoEExperts(nn.Module):
    """
    3D Quantized Routed Experts `[n_experts, out_features, in_features]` for MoE GGUF models
    (e.g., Qwen3-30B-A3B, Qwen2-MoE, Mixtral).
    Reads ONLY the routed `top_k` expert slices from GPU VRAM per token!
    """

    def __init__(
        self,
        mode: str,
        n_experts: int,
        in_features: int,
        out_features: int,
        qweight: torch.Tensor,
        scales: Optional[torch.Tensor] = None,
        zeros: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.n_experts = n_experts
        self.in_features = in_features
        self.out_features = out_features
        qw_c = qweight.contiguous()
        self.register_buffer("qweight", qw_c, persistent=False)
        self.register_buffer("scales", scales.contiguous() if scales is not None else None, persistent=False)
        self.register_buffer("zeros", zeros.contiguous() if zeros is not None else None, persistent=False)
        self.qweight_i32: Optional[torch.Tensor] = (
            qw_c.view(torch.int32) if mode in ("w4", "w8") and qw_c.shape[-1] % 4 == 0 else None
        )

    @classmethod
    def from_gguf(
        cls,
        gguf_file: GGUFFile,
        weight_name: str,
        device: torch.device = torch.device("cpu"),
    ) -> "GGUFQuantMoEExperts":
        info = gguf_file.tensors[weight_name]
        if len(info.shape) != 3:
            raise ValueError(f"Expected 3D MoE tensor for {weight_name}, got shape={info.shape}")
        n_experts, out_features, in_features = int(info.shape[0]), int(info.shape[1]), int(info.shape[2])
        total_rows = n_experts * out_features
        raw = gguf_file.get_tensor_bytes(weight_name)

        if info.qtype in (GGMLQuantType.F16, GGMLQuantType.F32, GGMLQuantType.BF16):
            w_fp16 = load_float_tensor(gguf_file, weight_name, device=device, dtype=torch.float16)
            return cls("fp16", n_experts, in_features, out_features, w_fp16)

        if info.qtype in SUB_5BIT_W4_QTYPES and in_features % 32 == 0:
            if info.qtype != GGMLQuantType.Q4_K or in_features % 256 == 0:
                qw, sc, zr = repack_gguf_to_w4a16(raw, info.qtype, total_rows, in_features, device)
                return cls(
                    "w4",
                    n_experts,
                    in_features,
                    out_features,
                    qw.view(n_experts, out_features, in_features // 2),
                    sc.view(n_experts, out_features, in_features // 32),
                    zr.view(n_experts, out_features, in_features // 32),
                )

        qw, sc, zr = repack_gguf_to_w8a16(raw, info.qtype, total_rows, in_features, device)
        return cls(
            "w8",
            n_experts,
            in_features,
            out_features,
            qw.view(n_experts, out_features, in_features),
            sc.view(n_experts, out_features, in_features // 16),
            zr.view(n_experts, out_features, in_features // 16) if zr is not None else None,
        )

    def forward_expert(self, expert_idx: int, x_tokens: torch.Tensor) -> torch.Tensor:
        """Run `x_tokens` `[M_e, in_features]` through a single routed expert `expert_idx`."""
        qw_i32_e = self.qweight_i32[expert_idx] if self.qweight_i32 is not None else None
        if self.mode == "w4":
            assert self.scales is not None and self.zeros is not None
            return exec_w4a16_matmul(
                x_tokens,
                self.qweight[expert_idx],
                self.scales[expert_idx],
                self.zeros[expert_idx],
                qw_i32_cached=qw_i32_e,
            )
        if self.mode == "w8":
            assert self.scales is not None
            return exec_w8a16_matmul(
                x_tokens,
                self.qweight[expert_idx],
                self.scales[expert_idx],
                self.zeros[expert_idx] if self.zeros is not None else None,
                qw_i32_cached=qw_i32_e,
            )
        return F.linear(x_tokens.to(device=self.qweight.device, dtype=torch.float16), self.qweight[expert_idx]).to(x_tokens.dtype)
