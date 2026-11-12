"""
Big-Context Quantized KV Cache (`int4`, `int8`, & `fp16`) with:
1. Sequential peak-VRAM-minimizing paged capacity growth + Sliding-Window Ring KV (`mtp_kv_window`)
2. `@triton.jit` Fused INT4 & INT8 GQA Split-KV Flash-Decoding kernels (reads 4-bit/8-bit K/V directly in SRAM)
3. `@triton.jit` Fused Multi-Row (`1 <= S <= 16`) Gated-DeltaNet Verify + Causal Conv1D kernels
4. Exact Online-Softmax Chunked Attention for long-context prefill (O(1) peak scratch VRAM at 64K-256K+!)
5. Automatic multi-turn Prefix KV Cache reuse for OpenAI `/v1/chat/completions`
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .quant_kernels import cuda_device_scope

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _int8_gqa_flash_decode_split_kernel(
        q_ptr,          # float16 [n_heads, head_dim]
        k_ptr,          # int8    [n_kv_heads, capacity, head_dim]
        ks_ptr,         # float16 [n_kv_heads, capacity]
        v_ptr,          # int8    [n_kv_heads, capacity, head_dim]
        vs_ptr,         # float16 [n_kv_heads, capacity]
        part_out_ptr,   # float32 [n_heads, num_splits, head_dim]
        part_lse_ptr,   # float32 [n_heads, num_splits]
        total_len,
        capacity,
        win_start,
        sm_scale,
        WIN_RING: tl.constexpr,
        GQA_GROUP: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        BLOCK_S: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        Q_LEN: tl.constexpr = 1,
        WINDOW_SIZE: tl.constexpr = 0,
    ):
        """
        Fused Split-KV Flash-Decoding directly over INT8-quantized K and V caches
        (with zero-copy modulo ring addressing when `WIN_RING=True`).
        """
        head_idx = tl.program_id(0)
        split_idx = tl.program_id(1)
        kv_head = (head_idx // Q_LEN) // GQA_GROUP
        # `total_len` covers the union of all query windows, beginning at
        # win_start. Each query sees only its causal prefix and own window.
        query_end = win_start + total_len - (Q_LEN - 1 - head_idx % Q_LEN)
        if WIN_RING:
            query_start = tl.maximum(win_start, query_end - WINDOW_SIZE) if WINDOW_SIZE > 0 else win_start
        else:
            query_start = 0
        visible_len = query_end - query_start
        tokens_per_split = (visible_len + NUM_SPLITS - 1) // NUM_SPLITS
        split_start = split_idx * tokens_per_split
        split_end = tl.minimum(visible_len, split_start + tokens_per_split)

        offs_d = tl.arange(0, HEAD_DIM)
        lse_offset = head_idx * NUM_SPLITS + split_idx
        out_offset = lse_offset * HEAD_DIM

        neg_inf = tl.zeros((), dtype=tl.float32) - float("inf")
        if split_start >= split_end:
            tl.store(part_lse_ptr + lse_offset, neg_inf)
            tl.store(part_out_ptr + out_offset + offs_d, tl.zeros((HEAD_DIM,), dtype=tl.float32))
            return

        q = tl.load(q_ptr + head_idx * HEAD_DIM + offs_d).to(tl.float32) * sm_scale

        m_i = tl.zeros((), dtype=tl.float32) - float("inf")
        l_i = tl.zeros((), dtype=tl.float32)
        acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)

        kv_h64 = kv_head.to(tl.int64)
        cap64 = capacity.to(tl.int64)
        k_base = k_ptr + kv_h64 * cap64 * HEAD_DIM
        v_base = v_ptr + kv_h64 * cap64 * HEAD_DIM
        ks_base = ks_ptr + kv_h64 * cap64
        vs_base = vs_ptr + kv_h64 * cap64

        for s_off in range(split_start, split_end, BLOCK_S):
            offs_s = s_off + tl.arange(0, BLOCK_S)
            mask_s = offs_s < split_end
            if WIN_RING:
                offs_s64 = ((query_start + offs_s) % capacity).to(tl.int64)
            else:
                offs_s64 = offs_s.to(tl.int64)

            k_q = tl.load(
                k_base + offs_s64[:, None] * HEAD_DIM + offs_d[None, :],
                mask=mask_s[:, None],
                other=0,
            ).to(tl.float32)
            ks = tl.load(ks_base + offs_s64, mask=mask_s, other=0.0).to(tl.float32)

            scores = tl.sum(k_q * q[None, :], axis=1) * ks
            scores = tl.where(mask_s, scores, float("-inf"))

            m_ij = tl.max(scores, axis=0)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new)

            v_q = tl.load(
                v_base + offs_s64[:, None] * HEAD_DIM + offs_d[None, :],
                mask=mask_s[:, None],
                other=0,
            ).to(tl.float32)
            vs = tl.load(vs_base + offs_s64, mask=mask_s, other=0.0).to(tl.float32)

            acc = acc * alpha + tl.sum((p * vs)[:, None] * v_q, axis=0)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            m_i = m_new

        inv_l = 1.0 / tl.maximum(l_i, 1e-20)
        tl.store(part_out_ptr + out_offset + offs_d, acc * inv_l)
        tl.store(part_lse_ptr + lse_offset, m_i + tl.log(tl.maximum(l_i, 1e-20)))

    @triton.jit
    def _int8_gqa_flash_decode_posptr_kernel(
        q_ptr,
        k_ptr,
        ks_ptr,
        v_ptr,
        vs_ptr,
        part_out_ptr,
        part_lse_ptr,
        pos_ptr,        # int32[1] GPU scalar (`cur_pos`, the chunk base position)
        capacity,
        sm_scale,
        GQA_GROUP: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        BLOCK_S: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        Q_LEN: tl.constexpr = 1,
    ):
        """CUDA-Graph-capturable Split-KV INT8 Flash-Decoding reading positions from GPU `pos_ptr`.

        `Q_LEN` queries share one captured graph (`q` rows are laid out `[head, q_row]`); query row
        `r` attends the causal prefix `[0, cur_pos + 1 + r)` (`Q_LEN == 1` reduces to the old path).
        """
        head_idx = tl.program_id(0)
        split_idx = tl.program_id(1)
        q_row = head_idx % Q_LEN
        kv_head = (head_idx // Q_LEN) // GQA_GROUP

        total_len = tl.load(pos_ptr).to(tl.int32) + 1 + q_row
        tokens_per_split = (total_len + NUM_SPLITS - 1) // NUM_SPLITS
        split_start = split_idx * tokens_per_split
        split_end = tl.minimum(total_len, split_start + tokens_per_split)

        offs_d = tl.arange(0, HEAD_DIM)
        lse_offset = head_idx * NUM_SPLITS + split_idx
        out_offset = lse_offset * HEAD_DIM

        neg_inf = tl.zeros((), dtype=tl.float32) - float("inf")
        if split_start >= split_end:
            tl.store(part_lse_ptr + lse_offset, neg_inf)
            tl.store(part_out_ptr + out_offset + offs_d, tl.zeros((HEAD_DIM,), dtype=tl.float32))
            return

        q = tl.load(q_ptr + head_idx * HEAD_DIM + offs_d).to(tl.float32) * sm_scale
        m_i = tl.zeros((), dtype=tl.float32) - float("inf")
        l_i = tl.zeros((), dtype=tl.float32)
        acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)

        kv_h64 = kv_head.to(tl.int64)
        cap64 = capacity.to(tl.int64)
        k_base = k_ptr + kv_h64 * cap64 * HEAD_DIM
        v_base = v_ptr + kv_h64 * cap64 * HEAD_DIM
        ks_base = ks_ptr + kv_h64 * cap64
        vs_base = vs_ptr + kv_h64 * cap64

        for s_off in range(split_start, split_end, BLOCK_S):
            offs_s = s_off + tl.arange(0, BLOCK_S)
            mask_s = offs_s < split_end
            offs_s64 = offs_s.to(tl.int64)

            k_q = tl.load(
                k_base + offs_s64[:, None] * HEAD_DIM + offs_d[None, :],
                mask=mask_s[:, None],
                other=0,
            ).to(tl.float32)
            ks = tl.load(ks_base + offs_s64, mask=mask_s, other=0.0).to(tl.float32)

            scores = tl.sum(k_q * q[None, :], axis=1) * ks
            scores = tl.where(mask_s, scores, float("-inf"))

            m_ij = tl.max(scores, axis=0)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new)

            v_q = tl.load(
                v_base + offs_s64[:, None] * HEAD_DIM + offs_d[None, :],
                mask=mask_s[:, None],
                other=0,
            ).to(tl.float32)
            vs = tl.load(vs_base + offs_s64, mask=mask_s, other=0.0).to(tl.float32)

            acc = acc * alpha + tl.sum((p * vs)[:, None] * v_q, axis=0)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            m_i = m_new

        inv_l = 1.0 / tl.maximum(l_i, 1e-20)
        tl.store(part_out_ptr + out_offset + offs_d, acc * inv_l)
        tl.store(part_lse_ptr + lse_offset, m_i + tl.log(tl.maximum(l_i, 1e-20)))

    @triton.jit
    def _flash_decode_combine_kernel(
        part_out_ptr,   # float32 [n_heads, NUM_SPLITS, HEAD_DIM]
        part_lse_ptr,   # float32 [n_heads, NUM_SPLITS]
        out_ptr,        # float16 [n_heads, HEAD_DIM]
        NUM_SPLITS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        """Fused Split-KV LSE reduction into FP16 output (replaces 7 eager PyTorch kernels per layer!)."""
        pid_h = tl.program_id(0)
        offs_s = tl.arange(0, NUM_SPLITS)
        offs_d = tl.arange(0, HEAD_DIM)

        lse = tl.load(part_lse_ptr + pid_h * NUM_SPLITS + offs_s)
        max_lse = tl.max(lse, axis=0)
        max_safe = tl.where(max_lse == float("-inf"), 0.0, max_lse)
        w = tl.exp(lse - max_safe)
        w = w / tl.maximum(tl.sum(w, axis=0), 1e-20)

        p_ptrs = part_out_ptr + (pid_h * NUM_SPLITS + offs_s[:, None]) * HEAD_DIM + offs_d[None, :]
        part = tl.load(p_ptrs)
        out = tl.sum(part * w[:, None], axis=0)
        tl.store(out_ptr + pid_h * HEAD_DIM + offs_d, out.to(tl.float16))

    @triton.jit
    def _kv_append_int8_kernel(
        k_in_ptr,    # float16 [n_kv_heads, s_new, head_dim]
        v_in_ptr,    # float16 [n_kv_heads, s_new, head_dim]
        k_buf_ptr,   # int8    [n_kv_heads, capacity, head_dim]
        v_buf_ptr,   # int8    [n_kv_heads, capacity, head_dim]
        ks_buf_ptr,  # float16 [n_kv_heads, capacity]
        vs_buf_ptr,  # float16 [n_kv_heads, capacity]
        start_pos,
        s_new,
        capacity,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused single-launch K & V dynamic per-token INT8 quantizer + paged cache writer."""
        pid_h = tl.program_id(0)
        pid_s = tl.program_id(1)

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < HEAD_DIM

        in_off = (pid_h.to(tl.int64) * s_new + pid_s) * HEAD_DIM + offs_d
        k_val = tl.load(k_in_ptr + in_off, mask=mask_d, other=0.0).to(tl.float32)
        v_val = tl.load(v_in_ptr + in_off, mask=mask_d, other=0.0).to(tl.float32)

        k_amax = tl.maximum(tl.max(tl.abs(k_val), axis=0), 1e-6)
        v_amax = tl.maximum(tl.max(tl.abs(v_val), axis=0), 1e-6)

        ks = (k_amax / 127.0).to(tl.float16)
        vs = (v_amax / 127.0).to(tl.float16)

        k_scaled = k_val / ks.to(tl.float32)
        v_scaled = v_val / vs.to(tl.float32)
        kq = tl.floor(tl.abs(k_scaled) + 0.5) * tl.where(k_scaled < 0.0, -1.0, 1.0)
        vq = tl.floor(tl.abs(v_scaled) + 0.5) * tl.where(v_scaled < 0.0, -1.0, 1.0)

        kq_i8 = tl.minimum(tl.maximum(kq, -127.0), 127.0).to(tl.int8)
        vq_i8 = tl.minimum(tl.maximum(vq, -127.0), 127.0).to(tl.int8)

        dst_pos = start_pos + pid_s
        out_off = (pid_h.to(tl.int64) * capacity + dst_pos) * HEAD_DIM + offs_d
        scale_off = pid_h.to(tl.int64) * capacity + dst_pos

        tl.store(k_buf_ptr + out_off, kq_i8, mask=mask_d)
        tl.store(v_buf_ptr + out_off, vq_i8, mask=mask_d)
        tl.store(ks_buf_ptr + scale_off, ks)
        tl.store(vs_buf_ptr + scale_off, vs)

    @triton.jit
    def _kv_append_int8_posptr_kernel(
        k_in_ptr,
        v_in_ptr,
        k_buf_ptr,
        v_buf_ptr,
        ks_buf_ptr,
        vs_buf_ptr,
        pos_ptr,        # int32[1] GPU scalar (`start_pos`, chunk base position)
        capacity,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        S_NEW: tl.constexpr = 1,
    ):
        """CUDA-Graph-capturable INT8 KV cache writer (launch with grid `(n_kv_heads, S_NEW)`).

        Row `pid_s` of the chunk is written at `load(pos_ptr) + pid_s` (`S_NEW == 1` reduces to
        the old single-token writer exactly).
        """
        pid_h = tl.program_id(0)
        pid_s = tl.program_id(1)
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < HEAD_DIM

        in_off = (pid_h.to(tl.int64) * S_NEW + pid_s) * HEAD_DIM + offs_d
        k_val = tl.load(k_in_ptr + in_off, mask=mask_d, other=0.0).to(tl.float32)
        v_val = tl.load(v_in_ptr + in_off, mask=mask_d, other=0.0).to(tl.float32)

        k_amax = tl.maximum(tl.max(tl.abs(k_val), axis=0), 1e-6)
        v_amax = tl.maximum(tl.max(tl.abs(v_val), axis=0), 1e-6)

        ks = (k_amax / 127.0).to(tl.float16)
        vs = (v_amax / 127.0).to(tl.float16)

        k_scaled = k_val / ks.to(tl.float32)
        v_scaled = v_val / vs.to(tl.float32)
        kq = tl.floor(tl.abs(k_scaled) + 0.5) * tl.where(k_scaled < 0.0, -1.0, 1.0)
        vq = tl.floor(tl.abs(v_scaled) + 0.5) * tl.where(v_scaled < 0.0, -1.0, 1.0)

        kq_i8 = tl.minimum(tl.maximum(kq, -127.0), 127.0).to(tl.int8)
        vq_i8 = tl.minimum(tl.maximum(vq, -127.0), 127.0).to(tl.int8)

        dst_pos = tl.load(pos_ptr).to(tl.int64) + pid_s
        out_off = (pid_h.to(tl.int64) * capacity + dst_pos) * HEAD_DIM + offs_d
        scale_off = pid_h.to(tl.int64) * capacity + dst_pos

        tl.store(k_buf_ptr + out_off, kq_i8, mask=mask_d)
        tl.store(v_buf_ptr + out_off, vq_i8, mask=mask_d)
        tl.store(ks_buf_ptr + scale_off, ks)
        tl.store(vs_buf_ptr + scale_off, vs)

    @triton.jit
    def _kv_append_int4_kernel(
        k_in_ptr,    # float16 [n_kv_heads, s_new, 2 * HALF_D]
        v_in_ptr,    # float16 [n_kv_heads, s_new, 2 * HALF_D]
        k_buf_ptr,   # uint8   [n_kv_heads, capacity, HALF_D]
        v_buf_ptr,   # uint8   [n_kv_heads, capacity, HALF_D]
        ks_buf_ptr,  # float16 [n_kv_heads, capacity, 2]
        vs_buf_ptr,  # float16 [n_kv_heads, capacity, 2]
        start_pos,
        s_new,
        capacity,
        HALF_D: tl.constexpr,
        BLOCK_HD: tl.constexpr,
    ):
        """
        Fused single-launch K & V split-half INT4 quantizer + cache writer (`~50%` less VRAM &
        GDDR6 bandwidth than INT8, `~75%` less than FP16; unlocks 256K+ context on Dual T4!).
        """
        pid_h = tl.program_id(0)
        pid_s = tl.program_id(1)

        offs_hd = tl.arange(0, BLOCK_HD)
        mask_hd = offs_hd < HALF_D

        in_base = (pid_h.to(tl.int64) * s_new + pid_s) * (2 * HALF_D)
        k_lo = tl.load(k_in_ptr + in_base + offs_hd, mask=mask_hd, other=0.0).to(tl.float32)
        k_hi = tl.load(k_in_ptr + in_base + HALF_D + offs_hd, mask=mask_hd, other=0.0).to(tl.float32)
        v_lo = tl.load(v_in_ptr + in_base + offs_hd, mask=mask_hd, other=0.0).to(tl.float32)
        v_hi = tl.load(v_in_ptr + in_base + HALF_D + offs_hd, mask=mask_hd, other=0.0).to(tl.float32)

        ks_lo = (tl.maximum(tl.max(tl.abs(k_lo), axis=0), 1e-6) * (1.0 / 7.0)).to(tl.float16)
        ks_hi = (tl.maximum(tl.max(tl.abs(k_hi), axis=0), 1e-6) * (1.0 / 7.0)).to(tl.float16)
        vs_lo = (tl.maximum(tl.max(tl.abs(v_lo), axis=0), 1e-6) * (1.0 / 7.0)).to(tl.float16)
        vs_hi = (tl.maximum(tl.max(tl.abs(v_hi), axis=0), 1e-6) * (1.0 / 7.0)).to(tl.float16)

        k_lo_s = k_lo / ks_lo.to(tl.float32)
        k_hi_s = k_hi / ks_hi.to(tl.float32)
        v_lo_s = v_lo / vs_lo.to(tl.float32)
        v_hi_s = v_hi / vs_hi.to(tl.float32)

        qk_lo = tl.minimum(tl.maximum(tl.floor(tl.abs(k_lo_s) + 0.5) * tl.where(k_lo_s < 0.0, -1.0, 1.0), -7.0), 7.0).to(tl.int32) + 8
        qk_hi = tl.minimum(tl.maximum(tl.floor(tl.abs(k_hi_s) + 0.5) * tl.where(k_hi_s < 0.0, -1.0, 1.0), -7.0), 7.0).to(tl.int32) + 8
        qv_lo = tl.minimum(tl.maximum(tl.floor(tl.abs(v_lo_s) + 0.5) * tl.where(v_lo_s < 0.0, -1.0, 1.0), -7.0), 7.0).to(tl.int32) + 8
        qv_hi = tl.minimum(tl.maximum(tl.floor(tl.abs(v_hi_s) + 0.5) * tl.where(v_hi_s < 0.0, -1.0, 1.0), -7.0), 7.0).to(tl.int32) + 8

        kp_u8 = ((qk_lo & 0x0F) | ((qk_hi & 0x0F) << 4)).to(tl.uint8)
        vp_u8 = ((qv_lo & 0x0F) | ((qv_hi & 0x0F) << 4)).to(tl.uint8)

        dst_pos = start_pos + pid_s
        slot_idx = pid_h.to(tl.int64) * capacity + dst_pos
        out_off = slot_idx * HALF_D + offs_hd
        sc_off = slot_idx * 2

        tl.store(k_buf_ptr + out_off, kp_u8, mask=mask_hd)
        tl.store(v_buf_ptr + out_off, vp_u8, mask=mask_hd)
        tl.store(ks_buf_ptr + sc_off, ks_lo)
        tl.store(ks_buf_ptr + sc_off + 1, ks_hi)
        tl.store(vs_buf_ptr + sc_off, vs_lo)
        tl.store(vs_buf_ptr + sc_off + 1, vs_hi)

    @triton.jit
    def _kv_append_int4_posptr_kernel(
        k_in_ptr,
        v_in_ptr,
        k_buf_ptr,
        v_buf_ptr,
        ks_buf_ptr,
        vs_buf_ptr,
        pos_ptr,        # int32[1] GPU scalar (`start_pos`, chunk base position)
        capacity,
        HALF_D: tl.constexpr,
        BLOCK_HD: tl.constexpr,
        S_NEW: tl.constexpr = 1,
    ):
        """CUDA-Graph-capturable INT4 KV cache writer (launch with grid `(n_kv_heads, S_NEW)`).

        Row `pid_s` of the chunk is written at `load(pos_ptr) + pid_s` (`S_NEW == 1` reduces to
        the old single-token writer exactly).
        """
        pid_h = tl.program_id(0)
        pid_s = tl.program_id(1)
        offs_hd = tl.arange(0, BLOCK_HD)
        mask_hd = offs_hd < HALF_D

        in_base = (pid_h.to(tl.int64) * S_NEW + pid_s) * (2 * HALF_D)
        k_lo = tl.load(k_in_ptr + in_base + offs_hd, mask=mask_hd, other=0.0).to(tl.float32)
        k_hi = tl.load(k_in_ptr + in_base + HALF_D + offs_hd, mask=mask_hd, other=0.0).to(tl.float32)
        v_lo = tl.load(v_in_ptr + in_base + offs_hd, mask=mask_hd, other=0.0).to(tl.float32)
        v_hi = tl.load(v_in_ptr + in_base + HALF_D + offs_hd, mask=mask_hd, other=0.0).to(tl.float32)

        ks_lo = (tl.maximum(tl.max(tl.abs(k_lo), axis=0), 1e-6) * (1.0 / 7.0)).to(tl.float16)
        ks_hi = (tl.maximum(tl.max(tl.abs(k_hi), axis=0), 1e-6) * (1.0 / 7.0)).to(tl.float16)
        vs_lo = (tl.maximum(tl.max(tl.abs(v_lo), axis=0), 1e-6) * (1.0 / 7.0)).to(tl.float16)
        vs_hi = (tl.maximum(tl.max(tl.abs(v_hi), axis=0), 1e-6) * (1.0 / 7.0)).to(tl.float16)

        k_lo_s = k_lo / ks_lo.to(tl.float32)
        k_hi_s = k_hi / ks_hi.to(tl.float32)
        v_lo_s = v_lo / vs_lo.to(tl.float32)
        v_hi_s = v_hi / vs_hi.to(tl.float32)

        qk_lo = tl.minimum(tl.maximum(tl.floor(tl.abs(k_lo_s) + 0.5) * tl.where(k_lo_s < 0.0, -1.0, 1.0), -7.0), 7.0).to(tl.int32) + 8
        qk_hi = tl.minimum(tl.maximum(tl.floor(tl.abs(k_hi_s) + 0.5) * tl.where(k_hi_s < 0.0, -1.0, 1.0), -7.0), 7.0).to(tl.int32) + 8
        qv_lo = tl.minimum(tl.maximum(tl.floor(tl.abs(v_lo_s) + 0.5) * tl.where(v_lo_s < 0.0, -1.0, 1.0), -7.0), 7.0).to(tl.int32) + 8
        qv_hi = tl.minimum(tl.maximum(tl.floor(tl.abs(v_hi_s) + 0.5) * tl.where(v_hi_s < 0.0, -1.0, 1.0), -7.0), 7.0).to(tl.int32) + 8

        kp_u8 = ((qk_lo & 0x0F) | ((qk_hi & 0x0F) << 4)).to(tl.uint8)
        vp_u8 = ((qv_lo & 0x0F) | ((qv_hi & 0x0F) << 4)).to(tl.uint8)

        dst_pos = tl.load(pos_ptr).to(tl.int64) + pid_s
        slot_idx = pid_h.to(tl.int64) * capacity + dst_pos
        out_off = slot_idx * HALF_D + offs_hd
        sc_off = slot_idx * 2

        tl.store(k_buf_ptr + out_off, kp_u8, mask=mask_hd)
        tl.store(v_buf_ptr + out_off, vp_u8, mask=mask_hd)
        tl.store(ks_buf_ptr + sc_off, ks_lo)
        tl.store(ks_buf_ptr + sc_off + 1, ks_hi)
        tl.store(vs_buf_ptr + sc_off, vs_lo)
        tl.store(vs_buf_ptr + sc_off + 1, vs_hi)

    @triton.jit
    def _int4_gqa_flash_decode_split_kernel(
        q_ptr,          # float16 [n_heads, 2 * HALF_D]
        k_ptr,          # uint8   [n_kv_heads, capacity, HALF_D]
        ks_ptr,         # float16 [n_kv_heads, capacity, 2]
        v_ptr,          # uint8   [n_kv_heads, capacity, HALF_D]
        vs_ptr,         # float16 [n_kv_heads, capacity, 2]
        part_out_ptr,   # float32 [n_heads, num_splits, 2 * HALF_D]
        part_lse_ptr,   # float32 [n_heads, num_splits]
        total_len,
        capacity,
        win_start,
        sm_scale,
        WIN_RING: tl.constexpr,
        GQA_GROUP: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        BLOCK_S: tl.constexpr,
        HALF_D: tl.constexpr,
        Q_LEN: tl.constexpr = 1,
        WINDOW_SIZE: tl.constexpr = 0,
    ):
        """
        Fused Split-KV Flash-Decoding directly over packed INT4 K and V caches (50% less GDDR6 traffic
        than INT8, 75% less than FP16, with optional zero-copy modulo ring window `WIN_RING=True`).
        """
        head_idx = tl.program_id(0)
        split_idx = tl.program_id(1)
        kv_head = (head_idx // Q_LEN) // GQA_GROUP
        # `total_len` covers the union of all query windows, beginning at
        # win_start. Each query sees only its causal prefix and own window.
        query_end = win_start + total_len - (Q_LEN - 1 - head_idx % Q_LEN)
        if WIN_RING:
            query_start = tl.maximum(win_start, query_end - WINDOW_SIZE) if WINDOW_SIZE > 0 else win_start
        else:
            query_start = 0
        visible_len = query_end - query_start
        tokens_per_split = (visible_len + NUM_SPLITS - 1) // NUM_SPLITS
        split_start = split_idx * tokens_per_split
        split_end = tl.minimum(visible_len, split_start + tokens_per_split)

        offs_hd = tl.arange(0, HALF_D)
        head_dim = 2 * HALF_D
        lse_offset = head_idx * NUM_SPLITS + split_idx
        out_offset = lse_offset * head_dim

        neg_inf = tl.zeros((), dtype=tl.float32) - float("inf")
        if split_start >= split_end:
            tl.store(part_lse_ptr + lse_offset, neg_inf)
            z_hd = tl.zeros((HALF_D,), dtype=tl.float32)
            tl.store(part_out_ptr + out_offset + offs_hd, z_hd)
            tl.store(part_out_ptr + out_offset + HALF_D + offs_hd, z_hd)
            return

        q_lo = tl.load(q_ptr + head_idx * head_dim + offs_hd).to(tl.float32) * sm_scale
        q_hi = tl.load(q_ptr + head_idx * head_dim + HALF_D + offs_hd).to(tl.float32) * sm_scale

        m_i = tl.zeros((), dtype=tl.float32) - float("inf")
        l_i = tl.zeros((), dtype=tl.float32)
        acc_lo = tl.zeros((HALF_D,), dtype=tl.float32)
        acc_hi = tl.zeros((HALF_D,), dtype=tl.float32)

        kv_h64 = kv_head.to(tl.int64)
        cap64 = capacity.to(tl.int64)
        k_base = k_ptr + kv_h64 * cap64 * HALF_D
        v_base = v_ptr + kv_h64 * cap64 * HALF_D
        ks_base = ks_ptr + kv_h64 * cap64 * 2
        vs_base = vs_ptr + kv_h64 * cap64 * 2

        for s_off in range(split_start, split_end, BLOCK_S):
            offs_s = s_off + tl.arange(0, BLOCK_S)
            mask_s = offs_s < split_end
            if WIN_RING:
                offs_s64 = ((query_start + offs_s) % capacity).to(tl.int64)
            else:
                offs_s64 = offs_s.to(tl.int64)

            kp = tl.load(
                k_base + offs_s64[:, None] * HALF_D + offs_hd[None, :],
                mask=mask_s[:, None],
                other=0x88,
            )
            ks_lo = tl.load(ks_base + offs_s64 * 2, mask=mask_s, other=0.0).to(tl.float32)
            ks_hi = tl.load(ks_base + offs_s64 * 2 + 1, mask=mask_s, other=0.0).to(tl.float32)

            k_lo = (kp & 0x0F).to(tl.float32) - 8.0
            k_hi = ((kp >> 4) & 0x0F).to(tl.float32) - 8.0

            scores = tl.sum(k_lo * q_lo[None, :], axis=1) * ks_lo + tl.sum(k_hi * q_hi[None, :], axis=1) * ks_hi
            scores = tl.where(mask_s, scores, float("-inf"))

            m_ij = tl.max(scores, axis=0)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new)

            vp = tl.load(
                v_base + offs_s64[:, None] * HALF_D + offs_hd[None, :],
                mask=mask_s[:, None],
                other=0x88,
            )
            vs_lo = tl.load(vs_base + offs_s64 * 2, mask=mask_s, other=0.0).to(tl.float32)
            vs_hi = tl.load(vs_base + offs_s64 * 2 + 1, mask=mask_s, other=0.0).to(tl.float32)

            v_lo = (vp & 0x0F).to(tl.float32) - 8.0
            v_hi = ((vp >> 4) & 0x0F).to(tl.float32) - 8.0

            acc_lo = acc_lo * alpha + tl.sum((p * vs_lo)[:, None] * v_lo, axis=0)
            acc_hi = acc_hi * alpha + tl.sum((p * vs_hi)[:, None] * v_hi, axis=0)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            m_i = m_new

        inv_l = 1.0 / tl.maximum(l_i, 1e-20)
        tl.store(part_out_ptr + out_offset + offs_hd, acc_lo * inv_l)
        tl.store(part_out_ptr + out_offset + HALF_D + offs_hd, acc_hi * inv_l)
        tl.store(part_lse_ptr + lse_offset, m_i + tl.log(tl.maximum(l_i, 1e-20)))

    @triton.jit
    def _int4_gqa_flash_decode_posptr_kernel(
        q_ptr,
        k_ptr,
        ks_ptr,
        v_ptr,
        vs_ptr,
        part_out_ptr,
        part_lse_ptr,
        pos_ptr,        # int32[1] GPU scalar (`cur_pos`, the chunk base position)
        capacity,
        sm_scale,
        GQA_GROUP: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        BLOCK_S: tl.constexpr,
        HALF_D: tl.constexpr,
        Q_LEN: tl.constexpr = 1,
    ):
        """CUDA-Graph-capturable Split-KV INT4 Flash-Decoding reading positions from GPU `pos_ptr`.

        `Q_LEN` queries share one captured graph (`q` rows are laid out `[head, q_row]`); query row
        `r` attends the causal prefix `[0, cur_pos + 1 + r)` (`Q_LEN == 1` reduces to the old path).
        """
        head_idx = tl.program_id(0)
        split_idx = tl.program_id(1)
        q_row = head_idx % Q_LEN
        kv_head = (head_idx // Q_LEN) // GQA_GROUP

        total_len = tl.load(pos_ptr).to(tl.int32) + 1 + q_row
        tokens_per_split = (total_len + NUM_SPLITS - 1) // NUM_SPLITS
        split_start = split_idx * tokens_per_split
        split_end = tl.minimum(total_len, split_start + tokens_per_split)

        offs_hd = tl.arange(0, HALF_D)
        head_dim = 2 * HALF_D
        lse_offset = head_idx * NUM_SPLITS + split_idx
        out_offset = lse_offset * head_dim

        neg_inf = tl.zeros((), dtype=tl.float32) - float("inf")
        if split_start >= split_end:
            tl.store(part_lse_ptr + lse_offset, neg_inf)
            z_hd = tl.zeros((HALF_D,), dtype=tl.float32)
            tl.store(part_out_ptr + out_offset + offs_hd, z_hd)
            tl.store(part_out_ptr + out_offset + HALF_D + offs_hd, z_hd)
            return

        q_lo = tl.load(q_ptr + head_idx * head_dim + offs_hd).to(tl.float32) * sm_scale
        q_hi = tl.load(q_ptr + head_idx * head_dim + HALF_D + offs_hd).to(tl.float32) * sm_scale

        m_i = tl.zeros((), dtype=tl.float32) - float("inf")
        l_i = tl.zeros((), dtype=tl.float32)
        acc_lo = tl.zeros((HALF_D,), dtype=tl.float32)
        acc_hi = tl.zeros((HALF_D,), dtype=tl.float32)

        kv_h64 = kv_head.to(tl.int64)
        cap64 = capacity.to(tl.int64)
        k_base = k_ptr + kv_h64 * cap64 * HALF_D
        v_base = v_ptr + kv_h64 * cap64 * HALF_D
        ks_base = ks_ptr + kv_h64 * cap64 * 2
        vs_base = vs_ptr + kv_h64 * cap64 * 2

        for s_off in range(split_start, split_end, BLOCK_S):
            offs_s = s_off + tl.arange(0, BLOCK_S)
            mask_s = offs_s < split_end
            offs_s64 = offs_s.to(tl.int64)

            kp = tl.load(
                k_base + offs_s64[:, None] * HALF_D + offs_hd[None, :],
                mask=mask_s[:, None],
                other=0x88,
            )
            ks_lo = tl.load(ks_base + offs_s64 * 2, mask=mask_s, other=0.0).to(tl.float32)
            ks_hi = tl.load(ks_base + offs_s64 * 2 + 1, mask=mask_s, other=0.0).to(tl.float32)

            k_lo = (kp & 0x0F).to(tl.float32) - 8.0
            k_hi = ((kp >> 4) & 0x0F).to(tl.float32) - 8.0

            scores = tl.sum(k_lo * q_lo[None, :], axis=1) * ks_lo + tl.sum(k_hi * q_hi[None, :], axis=1) * ks_hi
            scores = tl.where(mask_s, scores, float("-inf"))

            m_ij = tl.max(scores, axis=0)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new)

            vp = tl.load(
                v_base + offs_s64[:, None] * HALF_D + offs_hd[None, :],
                mask=mask_s[:, None],
                other=0x88,
            )
            vs_lo = tl.load(vs_base + offs_s64 * 2, mask=mask_s, other=0.0).to(tl.float32)
            vs_hi = tl.load(vs_base + offs_s64 * 2 + 1, mask=mask_s, other=0.0).to(tl.float32)

            v_lo = (vp & 0x0F).to(tl.float32) - 8.0
            v_hi = ((vp >> 4) & 0x0F).to(tl.float32) - 8.0

            acc_lo = acc_lo * alpha + tl.sum((p * vs_lo)[:, None] * v_lo, axis=0)
            acc_hi = acc_hi * alpha + tl.sum((p * vs_hi)[:, None] * v_hi, axis=0)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            m_i = m_new

        inv_l = 1.0 / tl.maximum(l_i, 1e-20)
        tl.store(part_out_ptr + out_offset + offs_hd, acc_lo * inv_l)
        tl.store(part_out_ptr + out_offset + HALF_D + offs_hd, acc_hi * inv_l)
        tl.store(part_lse_ptr + lse_offset, m_i + tl.log(tl.maximum(l_i, 1e-20)))

    @triton.jit
    def _gqa_flash_attn_kernel(
        q_ptr,      # fp16 [n_heads, q_len, head_dim]  (contiguous)
        k_ptr,      # KV_DTYPE == 0: fp16 [n_kv, capacity, head_dim]
                    # KV_DTYPE == 1: int8 [n_kv, capacity, head_dim]
                    # KV_DTYPE == 2: uint8 [n_kv, capacity, head_dim // 2]  (half-split nibble packing)
        ks_ptr,     # fp16 [n_kv, capacity]            (KV_DTYPE == 1)
                    # fp16 [n_kv, capacity, 2]         (KV_DTYPE == 2)
        v_ptr,
        vs_ptr,
        out_ptr,    # element_ty in {fp16, fp32} [n_heads, q_len, head_dim] (contiguous)
        start_pos,  # i32 chunk base position (query row i sits at start_pos + i)
        total_len,  # i32 valid KV entries
        capacity,   # i32 buffer capacity (row stride)
        sm_scale,   # fp32
        q_len,      # i32
        KV_DTYPE: tl.constexpr,   # 0 = fp16, 1 = int8, 2 = int4
        SOFTCAP: tl.constexpr,    # 0.0 = disabled, else attention-logit softcap (Gemma)
        GQA_GROUP: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        """
        Fused single-pass online-softmax GQA flash attention for long contexts (prefill + big-context
        decode/verify on the eager path). Reads INT4/INT8 KV directly (no dequant materialization),
        fp16 tensor-core MMAs via `tl.dot`, FP32 accumulators. Row `i` attends keys `[0, start_pos + i]`.
        Non-ring layouts only (`win_start == 0`); ring layers stay on the torch fallback in `attend()`.
        """
        pid_m = tl.program_id(0)
        head = tl.program_id(1)
        kv_head = head // GQA_GROUP

        HALF_D: tl.constexpr = HEAD_DIM // 2

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < q_len
        offs_d = tl.arange(0, HEAD_DIM)
        offs_hd = tl.arange(0, HALF_D)

        cap64 = capacity.to(tl.int64)
        kv_h64 = kv_head.to(tl.int64)
        if KV_DTYPE == 2:
            k_base = k_ptr + kv_h64 * cap64 * HALF_D
            v_base = v_ptr + kv_h64 * cap64 * HALF_D
            ks_base = ks_ptr + kv_h64 * cap64 * 2
            vs_base = vs_ptr + kv_h64 * cap64 * 2
        else:
            k_base = k_ptr + kv_h64 * cap64 * HEAD_DIM
            v_base = v_ptr + kv_h64 * cap64 * HEAD_DIM
            ks_base = ks_ptr + kv_h64 * cap64
            vs_base = vs_ptr + kv_h64 * cap64

        # Load q rows (fp16); split halves only on the int4 path.
        q_row_base = head.to(tl.int64) * q_len.to(tl.int64) * HEAD_DIM
        q_offs = q_row_base + offs_m[:, None].to(tl.int64) * HEAD_DIM + offs_d[None, :]
        q_h = tl.load(q_ptr + q_offs, mask=mask_m[:, None], other=0.0)
        if KV_DTYPE == 2:
            q_lo = tl.load(q_ptr + q_row_base + offs_m[:, None].to(tl.int64) * HEAD_DIM + offs_hd[None, :],
                           mask=mask_m[:, None], other=0.0)
            q_hi = tl.load(q_ptr + q_row_base + offs_m[:, None].to(tl.int64) * HEAD_DIM + HALF_D + offs_hd[None, :],
                           mask=mask_m[:, None], other=0.0)

        q_abs = start_pos + offs_m  # absolute position per query row

        # Highest KV index this block may read: row (pid_m*BLOCK_M + BLOCK_M - 1) sees abs+1 keys.
        max_visible = tl.minimum(total_len, start_pos + pid_m * BLOCK_M + BLOCK_M)

        neg_inf = tl.zeros((), dtype=tl.float32) - float("inf")
        m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) + neg_inf
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        if KV_DTYPE == 2:
            acc_lo = tl.zeros((BLOCK_M, HALF_D), dtype=tl.float32)
            acc_hi = tl.zeros((BLOCK_M, HALF_D), dtype=tl.float32)
        else:
            acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

        for kv_start in range(0, max_visible, BLOCK_N):
            offs_n = kv_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < max_visible
            offs_n64 = offs_n.to(tl.int64)
            mask_nk = mask_n[:, None]

            if KV_DTYPE == 2:
                kp = tl.load(k_base + offs_n64[:, None] * HALF_D + offs_hd[None, :], mask=mask_nk, other=0x88)
                ks_lo = tl.load(ks_base + offs_n64 * 2, mask=mask_n, other=0.0).to(tl.float32)
                ks_hi = tl.load(ks_base + offs_n64 * 2 + 1, mask=mask_n, other=0.0).to(tl.float32)
                k_lo = ((kp & 0x0F).to(tl.float32) - 8.0).to(tl.float16)
                k_hi = (((kp >> 4) & 0x0F).to(tl.float32) - 8.0).to(tl.float16)
                scores = (
                    tl.dot(q_lo, tl.trans(k_lo), out_dtype=tl.float32) * ks_lo[None, :]
                    + tl.dot(q_hi, tl.trans(k_hi), out_dtype=tl.float32) * ks_hi[None, :]
                )
            elif KV_DTYPE == 1:
                kq = tl.load(k_base + offs_n64[:, None] * HEAD_DIM + offs_d[None, :], mask=mask_nk, other=0)
                ks = tl.load(ks_base + offs_n64, mask=mask_n, other=0.0).to(tl.float32)
                scores = tl.dot(q_h, tl.trans(kq.to(tl.float16)), out_dtype=tl.float32) * ks[None, :]
            else:
                k_blk = tl.load(k_base + offs_n64[:, None] * HEAD_DIM + offs_d[None, :], mask=mask_nk, other=0.0)
                scores = tl.dot(q_h, tl.trans(k_blk), out_dtype=tl.float32)

            scores = scores * sm_scale
            if SOFTCAP > 0.0:
                cap = SOFTCAP
                e = tl.exp(tl.minimum(2.0 * scores / cap, 30.0))
                tanh_s = (e - 1.0) / (e + 1.0)
                scores = tanh_s * cap
            # Causal + validity mask (rows are absolute positions: key visible iff key <= row_pos).
            causal = offs_n[None, :] <= q_abs[:, None]
            scores = tl.where(causal & mask_n[None, :], scores, float("-inf"))

            m_ij = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            m_new_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            m_i_safe = tl.where(m_i == float("-inf"), 0.0, m_i)
            alpha = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i_safe - m_new_safe))
            p = tl.exp(scores - m_new_safe[:, None])

            if KV_DTYPE == 2:
                vp = tl.load(v_base + offs_n64[:, None] * HALF_D + offs_hd[None, :], mask=mask_nk, other=0x88)
                vs_lo = tl.load(vs_base + offs_n64 * 2, mask=mask_n, other=0.0).to(tl.float32)
                vs_hi = tl.load(vs_base + offs_n64 * 2 + 1, mask=mask_n, other=0.0).to(tl.float32)
                v_lo = ((vp & 0x0F).to(tl.float32) - 8.0).to(tl.float16)
                v_hi = (((vp >> 4) & 0x0F).to(tl.float32) - 8.0).to(tl.float16)
                acc_lo = acc_lo * alpha[:, None] + tl.dot((p * vs_lo[None, :]).to(tl.float16), v_lo, out_dtype=tl.float32)
                acc_hi = acc_hi * alpha[:, None] + tl.dot((p * vs_hi[None, :]).to(tl.float16), v_hi, out_dtype=tl.float32)
            elif KV_DTYPE == 1:
                vq = tl.load(v_base + offs_n64[:, None] * HEAD_DIM + offs_d[None, :], mask=mask_nk, other=0)
                vs = tl.load(vs_base + offs_n64, mask=mask_n, other=0.0).to(tl.float32)
                acc = acc * alpha[:, None] + tl.dot((p * vs[None, :]).to(tl.float16), vq.to(tl.float16), out_dtype=tl.float32)
            else:
                v_blk = tl.load(v_base + offs_n64[:, None] * HEAD_DIM + offs_d[None, :], mask=mask_nk, other=0.0)
                acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v_blk, out_dtype=tl.float32)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

        inv_l = 1.0 / tl.maximum(l_i, 1e-20)
        out_row = head.to(tl.int64) * q_len.to(tl.int64) * HEAD_DIM + offs_m.to(tl.int64)[:, None] * HEAD_DIM
        if KV_DTYPE == 2:
            tl.store(out_ptr + out_row + offs_hd[None, :], (acc_lo * inv_l[:, None]).to(out_ptr.dtype.element_ty),
                     mask=mask_m[:, None])
            tl.store(out_ptr + out_row + HALF_D + offs_hd[None, :], (acc_hi * inv_l[:, None]).to(out_ptr.dtype.element_ty),
                     mask=mask_m[:, None])
        else:
            tl.store(out_ptr + out_row + offs_d[None, :], (acc * inv_l[:, None]).to(out_ptr.dtype.element_ty),
                     mask=mask_m[:, None])


    @triton.jit
    def _gated_deltanet_scan_kernel(
        q_ptr,      # float32 [S, H_K, D_k] (pre-scaled by 1 / sqrt(D_k); tiled via pid_h % H_K!)
        k_ptr,      # float32 [S, H_K, D_k]
        v_ptr,      # float32 [S, H_v, D_v]
        decay_ptr,  # float32 [S, H_v]     (exp(g_t))
        beta_ptr,   # float32 [S, H_v]     (sigmoid(beta_t))
        state_ptr,  # float32 [H_v, D_v, D_k] (updated IN-PLACE)
        out_ptr,    # float32 [S, H_v, D_v]
        seq_len,
        H_K: tl.constexpr,
        H_V: tl.constexpr,
        D_K: tl.constexpr,
        D_V: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        """
        Fused Register-Resident Gated-DeltaNet Scan Kernel (for Qwen3.5 / Qwen3.6 / Qwen3.8 / Qwen3Next).
        Each threadblock owns a `[BLOCK_V, D_K]` slice of one V-head's recurrent state `[D_V, D_K]`
        entirely in GPU registers across all `seq_len` steps (zero per-token global state traffic!).
        """
        pid_h = tl.program_id(0)
        pid_v = tl.program_id(1)
        pid_kh = pid_h % H_K

        offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        offs_k = tl.arange(0, BLOCK_K)
        mask_v = offs_v < D_V
        mask_k = offs_k < D_K
        mask_vk = mask_v[:, None] & mask_k[None, :]

        state_base = state_ptr + pid_h * (D_V * D_K)
        s_ptrs = state_base + offs_v[:, None] * D_K + offs_k[None, :]
        state = tl.load(s_ptrs, mask=mask_vk, other=0.0).to(tl.float32)

        for t in range(0, seq_len):
            hb_off = t * H_V + pid_h
            decay_t = tl.load(decay_ptr + hb_off).to(tl.float32)
            beta_t = tl.load(beta_ptr + hb_off).to(tl.float32)

            qk_off = (t * H_K + pid_kh) * D_K + offs_k
            q_t = tl.load(q_ptr + qk_off, mask=mask_k, other=0.0).to(tl.float32)
            k_t = tl.load(k_ptr + qk_off, mask=mask_k, other=0.0).to(tl.float32)

            v_off = hb_off * D_V + offs_v
            v_t = tl.load(v_ptr + v_off, mask=mask_v, other=0.0).to(tl.float32)

            state = state * decay_t
            kv_mem = tl.sum(state * k_t[None, :], axis=1)
            delta = (v_t - kv_mem) * beta_t
            state = state + delta[:, None] * k_t[None, :]
            y_t = tl.sum(state * q_t[None, :], axis=1)

            tl.store(out_ptr + v_off, y_t, mask=mask_v)

        tl.store(s_ptrs, state, mask=mask_vk)

    @triton.jit
    def _causal_conv1d_silu_step_kernel(
        qkv_in_ptr,      # float16/float32 [conv_dim]
        conv_state_ptr,  # float32 [conv_dim, K_HIST] (updated IN-PLACE)
        conv_w_ptr,      # float32 [conv_dim, CONV_K]
        conv_out_ptr,    # float32 [conv_dim]
        conv_dim,
        CONV_K: tl.constexpr,
        K_HIST: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """Fused S=1 causal depthwise Conv1D + in-place state shift + SiLU in 1 launch (~2 us on T4)."""
        pid_c = tl.program_id(0)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < conv_dim

        x_new = tl.load(qkv_in_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
        acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

        for j in tl.static_range(0, K_HIST):
            s_j = tl.load(conv_state_ptr + offs_c * K_HIST + j, mask=mask_c, other=0.0).to(tl.float32)
            w_j = tl.load(conv_w_ptr + offs_c * CONV_K + j, mask=mask_c, other=0.0).to(tl.float32)
            acc += s_j * w_j
            if j > 0:
                tl.store(conv_state_ptr + offs_c * K_HIST + (j - 1), s_j, mask=mask_c)

        if K_HIST > 0:
            tl.store(conv_state_ptr + offs_c * K_HIST + (K_HIST - 1), x_new, mask=mask_c)

        w_last = tl.load(conv_w_ptr + offs_c * CONV_K + K_HIST, mask=mask_c, other=0.0).to(tl.float32)
        acc += x_new * w_last
        silu = acc / (1.0 + tl.exp(-acc))
        tl.store(conv_out_ptr + offs_c, silu, mask=mask_c)

    @triton.jit
    def _gated_deltanet_decode_step_kernel(
        conv_out_ptr,   # float32 [2 * key_dim + d_inner]
        beta_raw_ptr,   # float16/float32 [H_V]
        alpha_raw_ptr,  # float16/float32 [H_V]
        ssm_dt_ptr,     # float32 [H_V]
        ssm_a_ptr,      # float32 [H_V]
        state_ptr,      # float32 [H_V, D_V, D_K] (updated IN-PLACE)
        out_ptr,        # float32 [H_V, D_V]
        key_dim,
        q_scale,
        eps,
        H_K: tl.constexpr,
        H_V: tl.constexpr,
        D_K: tl.constexpr,
        D_V: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        """
        Fused S=1 Gated-DeltaNet Decode Kernel: computes `beta` sigmoid, `alpha+dt` softplus,
        `decay` exp, `Q`/`K` L2-Norm, tiled K-head broadcast, and recurrent state update in 1 launch!
        """
        pid_h = tl.program_id(0)
        pid_v = tl.program_id(1)
        pid_kh = pid_h % H_K

        offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        offs_k = tl.arange(0, BLOCK_K)
        mask_v = offs_v < D_V
        mask_k = offs_k < D_K
        mask_vk = mask_v[:, None] & mask_k[None, :]

        b_raw = tl.load(beta_raw_ptr + pid_h).to(tl.float32)
        beta_t = 1.0 / (1.0 + tl.exp(-b_raw))

        a_val = tl.load(alpha_raw_ptr + pid_h).to(tl.float32) + tl.load(ssm_dt_ptr + pid_h).to(tl.float32)
        sp = tl.where(a_val > 20.0, a_val, tl.log(1.0 + tl.exp(tl.maximum(a_val, -80.0))))
        decay_t = tl.exp(sp * tl.load(ssm_a_ptr + pid_h).to(tl.float32))

        q_raw = tl.load(conv_out_ptr + pid_kh * D_K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        k_raw = tl.load(conv_out_ptr + key_dim + pid_kh * D_K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        q_t = q_raw * tl.rsqrt(tl.sum(q_raw * q_raw, axis=0) + eps) * q_scale
        k_t = k_raw * tl.rsqrt(tl.sum(k_raw * k_raw, axis=0) + eps)

        v_off = pid_h * D_V + offs_v
        v_t = tl.load(conv_out_ptr + 2 * key_dim + v_off, mask=mask_v, other=0.0).to(tl.float32)

        state_base = state_ptr + pid_h * (D_V * D_K)
        s_ptrs = state_base + offs_v[:, None] * D_K + offs_k[None, :]
        state = tl.load(s_ptrs, mask=mask_vk, other=0.0).to(tl.float32)

        state = state * decay_t
        kv_mem = tl.sum(state * k_t[None, :], axis=1)
        delta = (v_t - kv_mem) * beta_t
        state = state + delta[:, None] * k_t[None, :]
        y_t = tl.sum(state * q_t[None, :], axis=1)

        tl.store(out_ptr + v_off, y_t, mask=mask_v)
        tl.store(s_ptrs, state, mask=mask_vk)

    @triton.jit
    def _causal_conv1d_silu_multirow_kernel(
        qkv_in_ptr,      # float16/float32 [S, conv_dim]
        conv_state_ptr,  # float32 [1, conv_dim, K_HIST] (updated IN-PLACE)
        conv_w_ptr,      # float32 [conv_dim, CONV_K]
        conv_out_ptr,    # float32 [S, conv_dim]
        conv_in_ptr,     # float32 [1, conv_dim, K_HIST + S] (written if SAVE_CONV_IN or CONV_K != 4)
        S,
        conv_dim,
        CONV_K: tl.constexpr,
        K_HIST: tl.constexpr,
        SAVE_CONV_IN: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """
        Fused multi-row (`1 <= S <= 16`) causal depthwise Conv1D + in-place state shift + SiLU
        in 1 launch (optionally emitting `conv_in` for zero-recompute speculative verify rollback).
        """
        pid_c = tl.program_id(0)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < conv_dim
        offs_c64 = offs_c.to(tl.int64)
        in_stride = K_HIST + S

        if CONV_K == 4 and K_HIST == 3:
            h0 = tl.load(conv_state_ptr + offs_c64 * 3 + 0, mask=mask_c, other=0.0).to(tl.float32)
            h1 = tl.load(conv_state_ptr + offs_c64 * 3 + 1, mask=mask_c, other=0.0).to(tl.float32)
            h2 = tl.load(conv_state_ptr + offs_c64 * 3 + 2, mask=mask_c, other=0.0).to(tl.float32)

            w0 = tl.load(conv_w_ptr + offs_c64 * 4 + 0, mask=mask_c, other=0.0).to(tl.float32)
            w1 = tl.load(conv_w_ptr + offs_c64 * 4 + 1, mask=mask_c, other=0.0).to(tl.float32)
            w2 = tl.load(conv_w_ptr + offs_c64 * 4 + 2, mask=mask_c, other=0.0).to(tl.float32)
            w3 = tl.load(conv_w_ptr + offs_c64 * 4 + 3, mask=mask_c, other=0.0).to(tl.float32)

            if SAVE_CONV_IN:
                tl.store(conv_in_ptr + offs_c64 * in_stride + 0, h0, mask=mask_c)
                tl.store(conv_in_ptr + offs_c64 * in_stride + 1, h1, mask=mask_c)
                tl.store(conv_in_ptr + offs_c64 * in_stride + 2, h2, mask=mask_c)

            for t in range(0, S):
                x = tl.load(qkv_in_ptr + t * conv_dim + offs_c64, mask=mask_c, other=0.0).to(tl.float32)
                if SAVE_CONV_IN:
                    tl.store(conv_in_ptr + offs_c64 * in_stride + (3 + t), x, mask=mask_c)
                acc = h0 * w0 + h1 * w1 + h2 * w2 + x * w3
                silu = acc / (1.0 + tl.exp(-acc))
                tl.store(conv_out_ptr + t * conv_dim + offs_c64, silu, mask=mask_c)
                h0 = h1
                h1 = h2
                h2 = x

            tl.store(conv_state_ptr + offs_c64 * 3 + 0, h0, mask=mask_c)
            tl.store(conv_state_ptr + offs_c64 * 3 + 1, h1, mask=mask_c)
            tl.store(conv_state_ptr + offs_c64 * 3 + 2, h2, mask=mask_c)
        else:
            for j in tl.static_range(0, K_HIST):
                s_j = tl.load(conv_state_ptr + offs_c64 * K_HIST + j, mask=mask_c, other=0.0).to(tl.float32)
                tl.store(conv_in_ptr + offs_c64 * in_stride + j, s_j, mask=mask_c)
            for t in range(0, S):
                x = tl.load(qkv_in_ptr + t * conv_dim + offs_c64, mask=mask_c, other=0.0).to(tl.float32)
                tl.store(conv_in_ptr + offs_c64 * in_stride + (K_HIST + t), x, mask=mask_c)
                acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
                for j in tl.static_range(0, CONV_K):
                    v_j = tl.load(conv_in_ptr + offs_c64 * in_stride + (t + j), mask=mask_c, other=0.0).to(tl.float32)
                    w_j = tl.load(conv_w_ptr + offs_c64 * CONV_K + j, mask=mask_c, other=0.0).to(tl.float32)
                    acc += v_j * w_j
                silu = acc / (1.0 + tl.exp(-acc))
                tl.store(conv_out_ptr + t * conv_dim + offs_c64, silu, mask=mask_c)
            for j in tl.static_range(0, K_HIST):
                s_new_j = tl.load(conv_in_ptr + offs_c64 * in_stride + (S + j), mask=mask_c, other=0.0).to(tl.float32)
                tl.store(conv_state_ptr + offs_c64 * K_HIST + j, s_new_j, mask=mask_c)

    @triton.jit
    def _gated_deltanet_multirow_verify_kernel(
        conv_out_ptr,    # float32 [S, conv_dim]
        beta_raw_ptr,    # float16/float32 [S, H_V]
        alpha_raw_ptr,   # float16/float32 [S, H_V]
        ssm_dt_ptr,      # float32 [H_V]
        ssm_a_ptr,       # float32 [H_V]
        state_ptr,       # float32 [H_V, D_V, D_K] (updated IN-PLACE)
        out_ptr,         # float32 [S, H_V, D_V]
        pre_ssm_ptr,     # float32 [H_V, D_V, D_K] (written if SAVE_INPUTS)
        q_save_ptr,      # float32 [S, H_K, D_K]   (written if SAVE_INPUTS)
        k_save_ptr,      # float32 [S, H_K, D_K]   (written if SAVE_INPUTS)
        v_save_ptr,      # float32 [S, H_V, D_V]   (written if SAVE_INPUTS)
        decay_save_ptr,  # float32 [S, H_V]        (written if SAVE_INPUTS)
        beta_save_ptr,   # float32 [S, H_V]        (written if SAVE_INPUTS)
        S,
        conv_dim,
        key_dim,
        q_scale,
        eps,
        SAVE_INPUTS: tl.constexpr,
        H_K: tl.constexpr,
        H_V: tl.constexpr,
        D_K: tl.constexpr,
        D_V: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        """
        Fused Multi-Row (`2 <= S <= 16`) Gated-DeltaNet Speculative Verify Kernel:
        Keeps `ssm_state [BLOCK_V, BLOCK_K]` in registers across all `S` tokens, fuses `beta` sigmoid,
        `alpha+dt` softplus, `decay` exp, `Q/K` L2-norm, and `H_K -> H_V` broadcast, and optionally
        snapshots `pre_ssm` + lightweight `(q_scaled, k, v, decay, beta)` for zero-recompute rollback!
        """
        pid_h = tl.program_id(0)
        pid_v = tl.program_id(1)
        pid_kh = pid_h % H_K

        offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        offs_k = tl.arange(0, BLOCK_K)
        mask_v = offs_v < D_V
        mask_k = offs_k < D_K
        mask_vk = mask_v[:, None] & mask_k[None, :]

        dt_h = tl.load(ssm_dt_ptr + pid_h).to(tl.float32)
        a_h = tl.load(ssm_a_ptr + pid_h).to(tl.float32)

        state_off = pid_h.to(tl.int64) * (D_V * D_K) + offs_v[:, None] * D_K + offs_k[None, :]
        state = tl.load(state_ptr + state_off, mask=mask_vk, other=0.0).to(tl.float32)
        if SAVE_INPUTS:
            tl.store(pre_ssm_ptr + state_off, state, mask=mask_vk)

        for t in range(0, S):
            b_raw = tl.load(beta_raw_ptr + t * H_V + pid_h).to(tl.float32)
            beta_t = 1.0 / (1.0 + tl.exp(-b_raw))

            a_val = tl.load(alpha_raw_ptr + t * H_V + pid_h).to(tl.float32) + dt_h
            sp = tl.where(a_val > 20.0, a_val, tl.log(1.0 + tl.exp(tl.maximum(a_val, -80.0))))
            decay_t = tl.exp(sp * a_h)

            c_row = conv_out_ptr + t * conv_dim
            q_raw = tl.load(c_row + pid_kh * D_K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            k_raw = tl.load(c_row + key_dim + pid_kh * D_K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            q_t = q_raw * tl.rsqrt(tl.sum(q_raw * q_raw, axis=0) + eps) * q_scale
            k_t = k_raw * tl.rsqrt(tl.sum(k_raw * k_raw, axis=0) + eps)

            v_off = pid_h * D_V + offs_v
            v_t = tl.load(c_row + 2 * key_dim + v_off, mask=mask_v, other=0.0).to(tl.float32)

            if SAVE_INPUTS:
                tl.store(v_save_ptr + (t * H_V + pid_h) * D_V + offs_v, v_t, mask=mask_v)
                if pid_v == 0:
                    tl.store(decay_save_ptr + t * H_V + pid_h, decay_t)
                    tl.store(beta_save_ptr + t * H_V + pid_h, beta_t)
                    if pid_h < H_K:
                        qk_off = (t * H_K + pid_h) * D_K + offs_k
                        tl.store(q_save_ptr + qk_off, q_t, mask=mask_k)
                        tl.store(k_save_ptr + qk_off, k_t, mask=mask_k)

            state = state * decay_t
            kv_mem = tl.sum(state * k_t[None, :], axis=1)
            delta = (v_t - kv_mem) * beta_t
            state = state + delta[:, None] * k_t[None, :]
            y_t = tl.sum(state * q_t[None, :], axis=1)

            tl.store(out_ptr + (t * H_V + pid_h) * D_V + offs_v, y_t, mask=mask_v)

        tl.store(state_ptr + state_off, state, mask=mask_vk)


def exec_gated_deltanet_scan(
    q_scaled: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    """
    Execute exact Gated-DeltaNet recurrence over `S` tokens, updating `state` `[H_v, D_v, D_k]` in-place.
      q_scaled: float32 `[S, H_k, D_k]` (`H_k` divides `H_v`; tiled automatically without `.repeat`!)
      k       : float32 `[S, H_k, D_k]`
      v       : float32 `[S, H_v, D_v]`
      decay   : float32 `[S, H_v]`
      beta    : float32 `[S, H_v]`
      state   : float32 `[H_v, D_v, D_k]`
    Returns `out` of shape `float32[S, H_v, D_v]`.
    """
    S, H_k, D_k = q_scaled.shape
    H_v = v.shape[1]
    D_v = v.shape[-1]

    if HAS_TRITON and state.is_cuda and D_k <= 256:
        dev = state.device
        with cuda_device_scope(dev):
            q_c = q_scaled.to(dev).contiguous()
            k_c = k.to(dev).contiguous()
            v_c = v.to(dev).contiguous()
            dec_c = decay.to(dev).contiguous()
            bet_c = beta.to(dev).contiguous()
            out = torch.empty((S, H_v, D_v), device=dev, dtype=torch.float32)

            BLOCK_K = triton.next_power_of_2(D_k)
            BLOCK_V = 16 if BLOCK_K >= 128 else 32
            grid = (H_v, triton.cdiv(D_v, BLOCK_V))
            _gated_deltanet_scan_kernel[grid](
                q_c,
                k_c,
                v_c,
                dec_c,
                bet_c,
                state,
                out,
                S,
                H_K=H_k,
                H_V=H_v,
                D_K=D_k,
                D_V=D_v,
                BLOCK_K=BLOCK_K,
                BLOCK_V=BLOCK_V,
                num_warps=2,
            )
            return out

    # Exact FP32 PyTorch fallback (CPU or D_k > 256)
    if H_k != H_v:
        rep = H_v // H_k
        q_scaled = q_scaled.repeat(1, rep, 1)
        k = k.repeat(1, rep, 1)
    out = torch.empty((S, H_v, D_v), device=q_scaled.device, dtype=torch.float32)
    for t in range(S):
        state.mul_(decay[t].view(H_v, 1, 1))
        k_t = k[t]
        kv_mem = (state * k_t.unsqueeze(1)).sum(dim=-1)
        delta = (v[t] - kv_mem) * beta[t].unsqueeze(-1)
        state.add_(delta.unsqueeze(-1) * k_t.unsqueeze(1))
        out[t] = (state * q_scaled[t].unsqueeze(1)).sum(dim=-1)
    return out


def exec_gated_deltanet_decode_step(
    qkv_mix: torch.Tensor,
    beta_raw: torch.Tensor,
    alpha_raw: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    ssm_dt: torch.Tensor,
    ssm_a: torch.Tensor,
    ssm_state: torch.Tensor,
    key_dim: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    q_scale: float,
    eps: float,
) -> Optional[torch.Tensor]:
    """
    Fast CUDA Triton S=1 Gated-DeltaNet path (2 kernel launches instead of 22!).
    Returns `float32[1, H_v, D_v]` on CUDA, or `None` when CPU fallback should be used.
    """
    if not (HAS_TRITON and ssm_state.is_cuda and head_k_dim <= 256):
        return None
    dev = ssm_state.device
    conv_dim = conv_weight.shape[0]
    conv_k = conv_weight.shape[-1]
    k_hist = conv_state.shape[-1]
    if conv_k != k_hist + 1:
        return None

    with cuda_device_scope(dev):
        qkv_c = qkv_mix.reshape(-1).contiguous()
        b_c = beta_raw.reshape(-1).contiguous()
        a_c = alpha_raw.reshape(-1).contiguous()
        conv_out = torch.empty((conv_dim,), device=dev, dtype=torch.float32)
        out = torch.empty((1, num_v_heads, head_v_dim), device=dev, dtype=torch.float32)

        BLOCK_C = 256
        _causal_conv1d_silu_step_kernel[(triton.cdiv(conv_dim, BLOCK_C),)](
            qkv_c,
            conv_state,
            conv_weight,
            conv_out,
            conv_dim,
            CONV_K=conv_k,
            K_HIST=k_hist,
            BLOCK_C=BLOCK_C,
            num_warps=4,
        )

        BLOCK_K = triton.next_power_of_2(head_k_dim)
        BLOCK_V = 16 if BLOCK_K >= 128 else 32
        _gated_deltanet_decode_step_kernel[(num_v_heads, triton.cdiv(head_v_dim, BLOCK_V))](
            conv_out,
            b_c,
            a_c,
            ssm_dt,
            ssm_a,
            ssm_state,
            out,
            key_dim,
            q_scale,
            eps,
            H_K=num_k_heads,
            H_V=num_v_heads,
            D_K=head_k_dim,
            D_V=head_v_dim,
            BLOCK_K=BLOCK_K,
            BLOCK_V=BLOCK_V,
            num_warps=2,
        )
        return out


def exec_gated_deltanet_multirow(
    qkv_mix: torch.Tensor,
    beta_raw: torch.Tensor,
    alpha_raw: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    ssm_dt: torch.Tensor,
    ssm_a: torch.Tensor,
    ssm_state: torch.Tensor,
    key_dim: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    q_scale: float,
    eps: float,
    rec_state: Optional["LayerRecurrentState"] = None,
) -> Optional[torch.Tensor]:
    """
    Fast CUDA Triton Multi-Row (`2 <= S <= 16`) Gated-DeltaNet path (2 kernel launches per layer
    instead of ~20 eager PyTorch ops, saving ~768 launches per speculative verify step on 48-GDN-layer
    models while optionally recording zero-recompute rollback inputs!).
    Returns `float32[S, H_v, D_v]` on CUDA, or `None` when fallback should be used.
    """
    S = int(qkv_mix.shape[1]) if qkv_mix.dim() == 3 else int(qkv_mix.shape[0])
    if not (HAS_TRITON and ssm_state.is_cuda and head_k_dim <= 256 and 1 < S <= 16):
        return None
    dev = ssm_state.device
    conv_dim = int(conv_weight.shape[0])
    conv_k = int(conv_weight.shape[-1])
    k_hist = int(conv_state.shape[-1])
    if conv_k != k_hist + 1:
        return None

    save_inputs = bool(rec_state is not None and rec_state.verify_active)
    need_conv_in = save_inputs or not (conv_k == 4 and k_hist == 3)

    with cuda_device_scope(dev):
        qkv_c = qkv_mix.reshape(S, conv_dim).contiguous()
        b_c = beta_raw.reshape(S, num_v_heads).contiguous()
        a_c = alpha_raw.reshape(S, num_v_heads).contiguous()
        conv_out = torch.empty((S, conv_dim), device=dev, dtype=torch.float32)
        out = torch.empty((S, num_v_heads, head_v_dim), device=dev, dtype=torch.float32)

        conv_in = (
            torch.empty((1, conv_dim, k_hist + S), device=dev, dtype=torch.float32)
            if need_conv_in
            else conv_out
        )

        BLOCK_C = 256
        _causal_conv1d_silu_multirow_kernel[(triton.cdiv(conv_dim, BLOCK_C),)](
            qkv_c,
            conv_state,
            conv_weight,
            conv_out,
            conv_in,
            S,
            conv_dim,
            CONV_K=conv_k,
            K_HIST=k_hist,
            SAVE_CONV_IN=save_inputs,
            BLOCK_C=BLOCK_C,
            num_warps=4,
        )

        if save_inputs:
            pre_ssm = torch.empty_like(ssm_state)
            q_save = torch.empty((S, num_k_heads, head_k_dim), device=dev, dtype=torch.float32)
            k_save = torch.empty((S, num_k_heads, head_k_dim), device=dev, dtype=torch.float32)
            v_save = torch.empty((S, num_v_heads, head_v_dim), device=dev, dtype=torch.float32)
            decay_save = torch.empty((S, num_v_heads), device=dev, dtype=torch.float32)
            beta_save = torch.empty((S, num_v_heads), device=dev, dtype=torch.float32)
        else:
            pre_ssm = ssm_state
            q_save = out
            k_save = out
            v_save = out
            decay_save = conv_out
            beta_save = conv_out

        BLOCK_K = triton.next_power_of_2(head_k_dim)
        BLOCK_V = 16 if BLOCK_K >= 128 else 32
        _gated_deltanet_multirow_verify_kernel[(num_v_heads, triton.cdiv(head_v_dim, BLOCK_V))](
            conv_out,
            b_c,
            a_c,
            ssm_dt,
            ssm_a,
            ssm_state,
            out,
            pre_ssm,
            q_save,
            k_save,
            v_save,
            decay_save,
            beta_save,
            S,
            conv_dim,
            key_dim,
            q_scale,
            eps,
            SAVE_INPUTS=save_inputs,
            H_K=num_k_heads,
            H_V=num_v_heads,
            D_K=head_k_dim,
            D_V=head_v_dim,
            BLOCK_K=BLOCK_K,
            BLOCK_V=BLOCK_V,
            num_warps=2,
        )

        if save_inputs and rec_state is not None:
            rec_state.record_verify_inputs(
                pre_ssm_state=pre_ssm,
                conv_in=conv_in,
                q_scaled=q_save,
                k=k_save,
                v=v_save,
                decay=decay_save,
                beta=beta_save,
                hist_len=k_hist,
            )
        return out


class LayerRecurrentState:
    """
    Fixed-memory recurrent state (`conv_state` + `ssm_state`) for linear-attention / Gated-DeltaNet layers.
    Uses O(1) constant VRAM (~3.1 MiB per layer on Qwen3.8-27B) regardless of whether context is 4K or 128K!
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.conv_state: Optional[torch.Tensor] = None  # float32 [1, conv_dim, conv_kernel - 1]
        self.ssm_state: Optional[torch.Tensor] = None   # float32 [num_v_heads, head_v_dim, head_k_dim]
        self.verify_active: bool = False
        # When an S>1 verify CUDA graph is captured, holds permanent references to the graph-owned
        # verify-cache buffers so `begin_recurrent_verify()` can re-point `_verify_cache` at them.
        self._graph_verify_cache: Optional[Tuple[torch.Tensor, ...]] = None
        self._verify_cache: Optional[
            Tuple[
                torch.Tensor,  # pre_ssm_state [H_v, D_v, D_k]
                torch.Tensor,  # conv_in [1, conv_dim, hist_len + S]
                torch.Tensor,  # q_scaled [S, H_k, D_k]
                torch.Tensor,  # k [S, H_k, D_k]
                torch.Tensor,  # v [S, H_v, D_v]
                torch.Tensor,  # decay [S, H_v]
                torch.Tensor,  # beta [S, H_v]
                int,           # hist_len
            ]
        ] = None

    def ensure_allocated(
        self,
        conv_dim: int,
        conv_kernel: int,
        num_v_heads: int,
        head_v_dim: int,
        head_k_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hist_len = max(1, conv_kernel - 1)
        if self.conv_state is None:
            self.conv_state = torch.zeros((1, conv_dim, hist_len), device=self.device, dtype=torch.float32)
        if self.ssm_state is None:
            self.ssm_state = torch.zeros(
                (num_v_heads, head_v_dim, head_k_dim), device=self.device, dtype=torch.float32
            )
        return self.conv_state, self.ssm_state

    def record_verify_inputs(
        self,
        pre_ssm_state: torch.Tensor,
        conv_in: torch.Tensor,
        q_scaled: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        decay: torch.Tensor,
        beta: torch.Tensor,
        hist_len: int,
    ) -> None:
        self._verify_cache = (pre_ssm_state, conv_in, q_scaled, k, v, decay, beta, hist_len)

    def commit_verify_prefix(self, n_keep: int) -> bool:
        """
        Zero-recompute speculative rollback: settle `conv_state` and `ssm_state` to exact row `n_keep`
        (`0 <= n_keep <= S`) by replaying ONLY the lightweight recurrence scan over `[:n_keep]`,
        avoiding a 64-layer trunk GEMV/MLP/Attention re-forward on draft rejection!
        """
        if self._verify_cache is None or self.conv_state is None or self.ssm_state is None:
            return False
        pre_ssm, conv_in, q_scaled, k, v, decay, beta, hist_len = self._verify_cache
        total_s = int(q_scaled.shape[0])
        if n_keep < 0 or n_keep > total_s:
            self._verify_cache = None
            return False
        with torch.inference_mode():
            if hist_len > 0 and conv_in.shape[-1] >= n_keep + hist_len:
                self.conv_state.copy_(conv_in[:, :, n_keep : n_keep + hist_len])
            if n_keep < total_s:
                self.ssm_state.copy_(pre_ssm)
                if n_keep > 0:
                    exec_gated_deltanet_scan(
                        q_scaled[:n_keep],
                        k[:n_keep],
                        v[:n_keep],
                        decay[:n_keep],
                        beta[:n_keep],
                        self.ssm_state,
                    )
        self._verify_cache = None
        self.verify_active = False
        return True

    def clear_verify_inputs(self) -> None:
        self._verify_cache = None
        self.verify_active = False

    def reset(self) -> None:
        self.clear_verify_inputs()
        with torch.inference_mode():
            if self.conv_state is not None:
                self.conv_state.zero_()
            if self.ssm_state is not None:
                self.ssm_state.zero_()

    def clone_state(self, target_device: Optional[torch.device] = None) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self.conv_state is None or self.ssm_state is None:
            return None
        dev = target_device or self.device
        with torch.inference_mode():
            return (self.conv_state.to(dev).clone(), self.ssm_state.to(dev).clone())

    def restore_state(self, snap: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        if snap is None:
            self.reset()
            return
        c_snap, s_snap = snap
        with torch.inference_mode():
            self.conv_state = c_snap.to(self.device).clone()
            self.ssm_state = s_snap.to(self.device).clone()

    def allocated_bytes(self) -> int:
        total = 0
        if self.conv_state is not None:
            total += self.conv_state.numel() * self.conv_state.element_size()
        if self.ssm_state is not None:
            total += self.ssm_state.numel() * self.ssm_state.element_size()
        return total


class LayerKVCache:
    """
    Single-layer KV cache residing on that layer's target GPU (`cuda:0` or `cuda:1`), supporting:
    - `int4` (~75% VRAM & bandwidth savings vs FP16, ~50% vs INT8; unlocks 256K+ context on Dual T4!),
    - `int8` (~50% VRAM savings vs FP16), or
    - `fp16` storage,
    with sequential peak-VRAM-minimizing paged growth (`page_growth=2048` / 256-token page granularity)
    and optional bounded Sliding-Window Ring Buffer (`sliding_window` for `draft-mtp`).
    """

    def __init__(
        self,
        n_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        device: torch.device,
        kv_dtype: Literal["int4", "int8", "fp16"] = "int8",
        page_growth: int = 2048,
        sliding_window: Optional[int] = None,
    ) -> None:
        if kv_dtype == "int4" and (head_dim % 2 != 0):
            raise ValueError(f"kv_dtype='int4' requires even head_dim, got {head_dim}")
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.kv_dtype: Literal["int4", "int8", "fp16"] = kv_dtype
        self.page_growth = page_growth
        self.sliding_window: Optional[int] = (
            int(sliding_window) if (sliding_window is not None and int(sliding_window) > 0) else None
        )
        # +16 guard slots ensure speculative chaining + rollback never overwrites active window history!
        self._ring_cap: Optional[int] = (
            min(max_seq_len, self.sliding_window + 16) if self.sliding_window is not None else None
        )

        self.capacity = 0
        self.seq_len = 0
        self.logical_seq_len = 0
        self.k_buf: Optional[torch.Tensor] = None
        self.v_buf: Optional[torch.Tensor] = None
        self.k_scale: Optional[torch.Tensor] = None
        self.v_scale: Optional[torch.Tensor] = None
        self._part_out_scratch: Optional[torch.Tensor] = None
        self._graph_scratch_out: Optional[torch.Tensor] = None
        self._graph_scratch_lse: Optional[torch.Tensor] = None
        self._part_lse_scratch: Optional[torch.Tensor] = None

    def _ensure_capacity(self, needed_len: int) -> None:
        if self._ring_cap is not None:
            if self.capacity >= self._ring_cap:
                return
            new_cap = self._ring_cap
        else:
            if needed_len > self.max_seq_len:
                raise ValueError(
                    f"Context length {needed_len} exceeds configured max_seq_len ({self.max_seq_len}). "
                    f"Increase `max_context_len` when starting the engine."
                )
            if needed_len <= self.capacity:
                return
            new_cap = min(
                self.max_seq_len,
                max(self.page_growth, ((needed_len + self.page_growth - 1) // self.page_growth) * self.page_growth),
            )

        copy_len = min(self.seq_len, self.capacity)
        if self.kv_dtype == "int4":
            storage_dtype = torch.uint8
            storage_dim = self.head_dim // 2
        elif self.kv_dtype == "int8":
            storage_dtype = torch.int8
            storage_dim = self.head_dim
        else:
            storage_dtype = torch.float16
            storage_dim = self.head_dim

        # Sequential single-buffer growth: drops old `k_buf` BEFORE allocating `new_v` so PyTorch's
        # CUDA caching allocator immediately recycles the freed block (eliminates 2x VRAM spikes at 128K-256K!).
        new_k = torch.empty((self.n_kv_heads, new_cap, storage_dim), device=self.device, dtype=storage_dtype)
        if copy_len > 0 and self.k_buf is not None:
            new_k[:, :copy_len, :].copy_(self.k_buf[:, :copy_len, :])
        self.k_buf = new_k
        del new_k

        new_v = torch.empty((self.n_kv_heads, new_cap, storage_dim), device=self.device, dtype=storage_dtype)
        if copy_len > 0 and self.v_buf is not None:
            new_v[:, :copy_len, :].copy_(self.v_buf[:, :copy_len, :])
        self.v_buf = new_v
        del new_v

        if self.kv_dtype == "int4":
            new_ks = torch.empty((self.n_kv_heads, new_cap, 2), device=self.device, dtype=torch.float16)
            if copy_len > 0 and self.k_scale is not None:
                new_ks[:, :copy_len, :].copy_(self.k_scale[:, :copy_len, :])
            self.k_scale = new_ks
            del new_ks

            new_vs = torch.empty((self.n_kv_heads, new_cap, 2), device=self.device, dtype=torch.float16)
            if copy_len > 0 and self.v_scale is not None:
                new_vs[:, :copy_len, :].copy_(self.v_scale[:, :copy_len, :])
            self.v_scale = new_vs
            del new_vs
        elif self.kv_dtype == "int8":
            new_ks = torch.empty((self.n_kv_heads, new_cap), device=self.device, dtype=torch.float16)
            if copy_len > 0 and self.k_scale is not None:
                new_ks[:, :copy_len].copy_(self.k_scale[:, :copy_len])
            self.k_scale = new_ks
            del new_ks

            new_vs = torch.empty((self.n_kv_heads, new_cap), device=self.device, dtype=torch.float16)
            if copy_len > 0 and self.v_scale is not None:
                new_vs[:, :copy_len].copy_(self.v_scale[:, :copy_len])
            self.v_scale = new_vs
            del new_vs

        self.capacity = new_cap

    def _write_contiguous_chunk(
        self,
        k_s: torch.Tensor,
        v_s: torch.Tensor,
        dst_pos: int,
        pos_tensor: Optional[torch.Tensor] = None,
    ) -> None:
        """Write `[n_kv_heads, s_chunk, head_dim]` into physical `[dst_pos : dst_pos + s_chunk]`."""
        assert self.k_buf is not None and self.v_buf is not None
        s_chunk = int(k_s.shape[1])
        end_pos = dst_pos + s_chunk

        if self.kv_dtype == "int4":
            assert self.k_scale is not None and self.v_scale is not None
            half_d = self.head_dim // 2
            if HAS_TRITON and self.k_buf.is_cuda and half_d <= 256:
                dev = self.k_buf.device
                with cuda_device_scope(dev):
                    k_c = k_s.to(device=dev, dtype=torch.float16).contiguous()
                    v_c = v_s.to(device=dev, dtype=torch.float16).contiguous()
                    BLOCK_HD = triton.next_power_of_2(max(16, half_d))
                    if pos_tensor is not None:
                        _kv_append_int4_posptr_kernel[(self.n_kv_heads, s_chunk)](
                            k_c,
                            v_c,
                            self.k_buf,
                            self.v_buf,
                            self.k_scale,
                            self.v_scale,
                            pos_tensor,
                            self.capacity,
                            HALF_D=half_d,
                            BLOCK_HD=BLOCK_HD,
                            S_NEW=s_chunk,
                            num_warps=2,
                        )
                    else:
                        _kv_append_int4_kernel[(self.n_kv_heads, s_chunk)](
                            k_c,
                            v_c,
                            self.k_buf,
                            self.v_buf,
                            self.k_scale,
                            self.v_scale,
                            dst_pos,
                            s_chunk,
                            self.capacity,
                            HALF_D=half_d,
                            BLOCK_HD=BLOCK_HD,
                            num_warps=2,
                        )
            else:
                k_lo, k_hi = k_s[..., :half_d].float(), k_s[..., half_d:].float()
                v_lo, v_hi = v_s[..., :half_d].float(), v_s[..., half_d:].float()
                ks_lo = (k_lo.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 7.0).to(torch.float16)
                ks_hi = (k_hi.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 7.0).to(torch.float16)
                vs_lo = (v_lo.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 7.0).to(torch.float16)
                vs_hi = (v_hi.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 7.0).to(torch.float16)

                def _q4_u8(val: torch.Tensor, sc: torch.Tensor) -> torch.Tensor:
                    s_val = val / sc.float()
                    r_val = torch.floor(s_val.abs() + 0.5) * torch.where(s_val < 0.0, -1.0, 1.0)
                    return (r_val.clamp(-7.0, 7.0).to(torch.int32) + 8) & 0x0F

                kp = (_q4_u8(k_lo, ks_lo) | (_q4_u8(k_hi, ks_hi) << 4)).to(torch.uint8)
                vp = (_q4_u8(v_lo, vs_lo) | (_q4_u8(v_hi, vs_hi) << 4)).to(torch.uint8)
                ks_pair = torch.cat([ks_lo, ks_hi], dim=-1)
                vs_pair = torch.cat([vs_lo, vs_hi], dim=-1)
                if pos_tensor is not None:
                    pos_vec = pos_tensor.to(torch.long).view(1) + torch.arange(
                        s_chunk, device=self.k_buf.device, dtype=torch.long
                    )
                    self.k_buf.index_copy_(1, pos_vec, kp)
                    self.v_buf.index_copy_(1, pos_vec, vp)
                    self.k_scale.index_copy_(1, pos_vec, ks_pair)
                    self.v_scale.index_copy_(1, pos_vec, vs_pair)
                else:
                    self.k_buf[:, dst_pos:end_pos, :] = kp
                    self.v_buf[:, dst_pos:end_pos, :] = vp
                    self.k_scale[:, dst_pos:end_pos, :] = ks_pair
                    self.v_scale[:, dst_pos:end_pos, :] = vs_pair
        elif self.kv_dtype == "int8":
            assert self.k_scale is not None and self.v_scale is not None
            if HAS_TRITON and self.k_buf.is_cuda and self.head_dim <= 512:
                dev = self.k_buf.device
                with cuda_device_scope(dev):
                    k_c = k_s.to(device=dev, dtype=torch.float16).contiguous()
                    v_c = v_s.to(device=dev, dtype=torch.float16).contiguous()
                    BLOCK_D = triton.next_power_of_2(self.head_dim)
                    if pos_tensor is not None:
                        _kv_append_int8_posptr_kernel[(self.n_kv_heads, s_chunk)](
                            k_c,
                            v_c,
                            self.k_buf,
                            self.v_buf,
                            self.k_scale,
                            self.v_scale,
                            pos_tensor,
                            self.capacity,
                            HEAD_DIM=self.head_dim,
                            BLOCK_D=BLOCK_D,
                            S_NEW=s_chunk,
                            num_warps=2,
                        )
                    else:
                        _kv_append_int8_kernel[(self.n_kv_heads, s_chunk)](
                            k_c,
                            v_c,
                            self.k_buf,
                            self.v_buf,
                            self.k_scale,
                            self.v_scale,
                            dst_pos,
                            s_chunk,
                            self.capacity,
                            HEAD_DIM=self.head_dim,
                            BLOCK_D=BLOCK_D,
                            num_warps=2,
                        )
            elif pos_tensor is not None:
                k_amax = k_s.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
                v_amax = v_s.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
                ks = (k_amax / 127.0).to(torch.float16)
                vs = (v_amax / 127.0).to(torch.float16)
                kq = torch.round(k_s / ks).clamp(-127, 127).to(torch.int8)
                vq = torch.round(v_s / vs).clamp(-127, 127).to(torch.int8)
                pos_vec = pos_tensor.to(torch.long).view(1) + torch.arange(
                    s_chunk, device=self.k_buf.device, dtype=torch.long
                )
                self.k_buf.index_copy_(1, pos_vec, kq)
                self.v_buf.index_copy_(1, pos_vec, vq)
                self.k_scale.index_copy_(1, pos_vec, ks.squeeze(-1))
                self.v_scale.index_copy_(1, pos_vec, vs.squeeze(-1))
            else:
                k_amax = k_s.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
                v_amax = v_s.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
                ks = (k_amax / 127.0).to(torch.float16)
                vs = (v_amax / 127.0).to(torch.float16)
                self.k_buf[:, dst_pos:end_pos, :] = torch.round(k_s / ks).clamp(-127, 127).to(torch.int8)
                self.v_buf[:, dst_pos:end_pos, :] = torch.round(v_s / vs).clamp(-127, 127).to(torch.int8)
                self.k_scale[:, dst_pos:end_pos] = ks.squeeze(-1)
                self.v_scale[:, dst_pos:end_pos] = vs.squeeze(-1)
        else:
            if pos_tensor is not None:
                pos_vec = pos_tensor.to(torch.long).view(1) + torch.arange(
                    s_chunk, device=self.k_buf.device, dtype=torch.long
                )
                self.k_buf.index_copy_(1, pos_vec, k_s.to(torch.float16))
                self.v_buf.index_copy_(1, pos_vec, v_s.to(torch.float16))
            else:
                self.k_buf[:, dst_pos:end_pos, :] = k_s.to(torch.float16)
                self.v_buf[:, dst_pos:end_pos, :] = v_s.to(torch.float16)

    def append(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        start_pos: int,
        pos_tensor: Optional[torch.Tensor] = None,
    ) -> int:
        """
        Append `k_new`, `v_new` of shape `[1, n_kv_heads, S_new, head_dim]` at `start_pos`.
        When `pos_tensor` (`int32[1]` on GPU) is supplied, the write offsets are derived on-GPU
        from `pos_tensor` (`base + row`), making the path CUDA-Graph-capturable for any `S_new`.
        When `sliding_window` is configured, wraps writes into the bounded ring buffer (`O(W)` VRAM).
        """
        s_new = int(k_new.shape[2])
        end_pos = start_pos + s_new

        if self.sliding_window is not None:
            self._ensure_capacity(1)
            assert self.k_buf is not None and self.v_buf is not None
            cap = self.capacity
            k_s = k_new.squeeze(0)
            v_s = v_new.squeeze(0)
            w_start = start_pos
            if s_new > cap:
                keep = cap
                k_s = k_s[:, -keep:, :]
                v_s = v_s[:, -keep:, :]
                w_start = end_pos - keep
                s_write = keep
            else:
                s_write = s_new

            if w_start + s_write <= cap:
                self._write_contiguous_chunk(k_s, v_s, w_start)
            else:
                r0 = w_start % cap
                first_n = min(s_write, cap - r0)
                self._write_contiguous_chunk(k_s[:, :first_n, :], v_s[:, :first_n, :], r0)
                rem_n = s_write - first_n
                if rem_n > 0:
                    self._write_contiguous_chunk(k_s[:, first_n:, :], v_s[:, first_n:, :], 0)

            self.logical_seq_len = end_pos
            self.seq_len = min(end_pos, self.sliding_window)
            return end_pos

        if pos_tensor is None:
            self._ensure_capacity(end_pos)
        assert self.k_buf is not None and self.v_buf is not None
        self._write_contiguous_chunk(k_new.squeeze(0), v_new.squeeze(0), start_pos, pos_tensor=pos_tensor)
        self.logical_seq_len = end_pos
        self.seq_len = end_pos
        return end_pos

    def rewind_to(self, length: int) -> None:
        """Rewind logical and physical KV cache length cleanly (ring-window aware)."""
        clamped = max(0, int(length))
        if self.sliding_window is not None:
            self.logical_seq_len = min(self.logical_seq_len, clamped)
            self.seq_len = min(self.logical_seq_len, self.sliding_window)
        else:
            self.logical_seq_len = clamped
            self.seq_len = clamped

    def get_dequant_slice(self, start: int, end: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantize only `[start:end]` physical KV chunk -> `[1, n_kv_heads, end-start, head_dim]`."""
        assert self.k_buf is not None and self.v_buf is not None
        if self.kv_dtype == "int4":
            assert self.k_scale is not None and self.v_scale is not None
            kp = self.k_buf[:, start:end, :]
            vp = self.v_buf[:, start:end, :]
            ks = self.k_scale[:, start:end, :].to(dtype)
            vs = self.v_scale[:, start:end, :].to(dtype)
            k_lo = ((kp & 0x0F).to(dtype) - 8.0) * ks[..., 0:1]
            k_hi = (((kp >> 4) & 0x0F).to(dtype) - 8.0) * ks[..., 1:2]
            v_lo = ((vp & 0x0F).to(dtype) - 8.0) * vs[..., 0:1]
            v_hi = (((vp >> 4) & 0x0F).to(dtype) - 8.0) * vs[..., 1:2]
            return torch.cat([k_lo, k_hi], dim=-1).unsqueeze(0), torch.cat([v_lo, v_hi], dim=-1).unsqueeze(0)
        if self.kv_dtype == "int8":
            assert self.k_scale is not None and self.v_scale is not None
            ks = self.k_scale[:, start:end].to(dtype).unsqueeze(-1)
            vs = self.v_scale[:, start:end].to(dtype).unsqueeze(-1)
            k = self.k_buf[:, start:end, :].to(dtype) * ks
            v = self.v_buf[:, start:end, :].to(dtype) * vs
            return k.unsqueeze(0), v.unsqueeze(0)
        return self.k_buf[:, start:end, :].to(dtype).unsqueeze(0), self.v_buf[:, start:end, :].to(dtype).unsqueeze(0)

    def _get_ring_dequant_window(
        self,
        win_start: int,
        end_pos: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return chronological `[win_start : end_pos)` KV window from a ring buffer."""
        cap = self.capacity
        if end_pos <= cap:
            return self.get_dequant_slice(win_start, end_pos, dtype)
        r0 = win_start % cap
        r1 = end_pos % cap
        if r0 < r1:
            return self.get_dequant_slice(r0, r1, dtype)
        k_a, v_a = self.get_dequant_slice(r0, cap, dtype)
        if r1 == 0:
            return k_a, v_a
        k_b, v_b = self.get_dequant_slice(0, r1, dtype)
        return torch.cat([k_a, k_b], dim=2), torch.cat([v_a, v_b], dim=2)

    def attend(
        self,
        q: torch.Tensor,
        start_pos: int,
        sm_scale: float,
        softcap: float = 0.0,
        kv_chunk_size: int = 4096,
        pos_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute GQA causal attention for `q` `[1, n_heads, q_len, head_dim]`.
        Uses:
          1. Fused Split-KV INT4 / INT8 attention for up to 8 queries (GPU position pointers read the chunk base position)
          2. PyTorch SDPA fast path when `eff_len <= kv_chunk_size` and `softcap == 0`
          3. Online-Softmax Chunked Attention for ultra-long contexts (32K-256K+) so VRAM never spikes!
        """
        _, n_heads, q_len, head_dim = q.shape
        gqa_group = n_heads // self.n_kv_heads

        is_ring = self.sliding_window is not None
        if is_ring:
            assert self.sliding_window is not None
            end_pos = self.logical_seq_len
            win_cap = self.sliding_window + q_len - 1
            win_start = max(0, end_pos - win_cap)
            total_len = end_pos - win_start
        else:
            end_pos = self.seq_len
            win_start = 0
            total_len = self.seq_len

        # Path 1: Triton Split-KV INT4 / INT8 attention for short decode/verify batches.
        is_pow2_hd = head_dim in (32, 64, 128, 256)
        if (
            HAS_TRITON
            and q.is_cuda
            and self.kv_dtype in ("int4", "int8")
            and 1 <= q_len <= 8
            and (
                (pos_tensor is not None and not is_ring and q_len <= 8)
                or (
                    pos_tensor is None
                    and (
                        q_len == 1
                        or (start_pos + q_len == end_pos
                            and (not is_ring or self.capacity >= self.sliding_window + q_len - 1))
                    )
                )
            )
            and softcap == 0.0
            and is_pow2_hd
            and (pos_tensor is not None or total_len >= 1)
        ):
            assert self.k_buf is not None and self.v_buf is not None
            assert self.k_scale is not None and self.v_scale is not None
            dev = self.k_buf.device
            with cuda_device_scope(dev):
                if pos_tensor is not None:
                    num_splits = 32
                elif total_len < 128:
                    num_splits = 2
                elif total_len < 1024:
                    num_splits = 8
                elif total_len < 8192:
                    num_splits = 16
                else:
                    num_splits = 32
                block_s = 32 if head_dim >= 256 else 64

                rows = n_heads * q_len
                if pos_tensor is not None:
                    # Graph-reserved scratch: ONE max-size buffer per layer shared by every captured
                    # S=1..8 graph (dim-0 slicing stays contiguous for any row count). Allocated during
                    # capture setup so graph capture never allocates.
                    if self._graph_scratch_out is None or self._graph_scratch_out.shape[0] < rows:
                        self._graph_scratch_out = torch.empty((n_heads * 8, 32, head_dim), device=dev, dtype=torch.float32)
                        self._graph_scratch_lse = torch.empty((n_heads * 8, 32), device=dev, dtype=torch.float32)
                    assert self._graph_scratch_lse is not None
                    part_out = self._graph_scratch_out[:rows]
                    part_lse = self._graph_scratch_lse[:rows]
                else:
                    if (self._part_out_scratch is None
                        or self._part_out_scratch.shape != (rows, num_splits, head_dim)):
                        # Kernels index with NUM_SPLITS as the row stride: sliced
                        # 32-wide buffers are not contiguous for fewer splits.
                        self._part_out_scratch = torch.empty((rows, num_splits, head_dim), device=dev, dtype=torch.float32)
                        self._part_lse_scratch = torch.empty((rows, num_splits), device=dev, dtype=torch.float32)
                    part_out = self._part_out_scratch
                    part_lse = self._part_lse_scratch

                q_contig = q.to(device=dev, dtype=torch.float16).contiguous().view(rows, head_dim)
                grid = (rows, num_splits)
                if self.kv_dtype == "int4":
                    half_d = head_dim // 2
                    if pos_tensor is not None:
                        _int4_gqa_flash_decode_posptr_kernel[grid](
                            q_contig,
                            self.k_buf,
                            self.k_scale,
                            self.v_buf,
                            self.v_scale,
                            part_out,
                            part_lse,
                            pos_tensor,
                            self.capacity,
                            sm_scale,
                            GQA_GROUP=gqa_group,
                            NUM_SPLITS=num_splits,
                            BLOCK_S=block_s,
                            HALF_D=half_d,
                            Q_LEN=q_len,
                            num_warps=4,
                        )
                    else:
                        _int4_gqa_flash_decode_split_kernel[grid](
                            q_contig,
                            self.k_buf,
                            self.k_scale,
                            self.v_buf,
                            self.v_scale,
                            part_out,
                            part_lse,
                            total_len,
                            self.capacity,
                            win_start,
                            sm_scale,
                            WIN_RING=is_ring,
                            Q_LEN=q_len,
                            WINDOW_SIZE=self.sliding_window if is_ring else 0,
                            GQA_GROUP=gqa_group,
                            NUM_SPLITS=num_splits,
                            BLOCK_S=block_s,
                            HALF_D=half_d,
                            num_warps=4,
                        )
                else:
                    if pos_tensor is not None:
                        _int8_gqa_flash_decode_posptr_kernel[grid](
                            q_contig,
                            self.k_buf,
                            self.k_scale,
                            self.v_buf,
                            self.v_scale,
                            part_out,
                            part_lse,
                            pos_tensor,
                            self.capacity,
                            sm_scale,
                            GQA_GROUP=gqa_group,
                            NUM_SPLITS=num_splits,
                            BLOCK_S=block_s,
                            HEAD_DIM=head_dim,
                            Q_LEN=q_len,
                            num_warps=4,
                        )
                    else:
                        _int8_gqa_flash_decode_split_kernel[grid](
                            q_contig,
                            self.k_buf,
                            self.k_scale,
                            self.v_buf,
                            self.v_scale,
                            part_out,
                            part_lse,
                            total_len,
                            self.capacity,
                            win_start,
                            sm_scale,
                            WIN_RING=is_ring,
                            Q_LEN=q_len,
                            WINDOW_SIZE=self.sliding_window if is_ring else 0,
                            GQA_GROUP=gqa_group,
                            NUM_SPLITS=num_splits,
                            BLOCK_S=block_s,
                            HEAD_DIM=head_dim,
                            num_warps=4,
                        )
                out_fp16 = torch.empty((rows, head_dim), device=dev, dtype=torch.float16)
                _flash_decode_combine_kernel[(rows,)](
                    part_out,
                    part_lse,
                    out_fp16,
                    NUM_SPLITS=num_splits,
                    HEAD_DIM=head_dim,
                    num_warps=2,
                )
                return out_fp16.to(q.dtype).view(1, n_heads, q_len, head_dim)

        # Path 2: Short/Medium Context (or Sliding-Window Ring) -> Fused PyTorch SDPA
        if (total_len <= kv_chunk_size or is_ring) and softcap == 0.0:
            if is_ring:
                k_full, v_full = self._get_ring_dequant_window(win_start, end_pos, q.dtype)
            else:
                k_full, v_full = self.get_dequant_slice(0, total_len, q.dtype)

            # Prefer native GQA (`enable_gqa`, torch >= 2.5) so K/V are never expanded in memory;
            # fall back to `repeat_interleave` on unsupported backends/builds.
            sdpa_gqa = gqa_group > 1

            def _sdpa(
                _q: torch.Tensor,
                _k: torch.Tensor,
                _v: torch.Tensor,
                **_kw: Any,
            ) -> torch.Tensor:
                if sdpa_gqa:
                    try:
                        return F.scaled_dot_product_attention(_q, _k, _v, enable_gqa=True, **_kw)
                    except Exception:
                        return F.scaled_dot_product_attention(
                            _q,
                            _k.repeat_interleave(gqa_group, dim=1),
                            _v.repeat_interleave(gqa_group, dim=1),
                            **_kw,
                        )
                return F.scaled_dot_product_attention(_q, _k, _v, **_kw)

            if q_len == 1:
                return _sdpa(q, k_full, v_full, is_causal=False, scale=sm_scale)
            if start_pos == 0 and win_start == 0:
                return _sdpa(q, k_full, v_full, is_causal=True, scale=sm_scale)

            # Speculative / multi-token append (`start_pos > 0` or ring window): explicit causal + window mask
            q_pos = torch.arange(start_pos, start_pos + q_len, device=q.device).unsqueeze(1)
            k_pos = torch.arange(win_start, end_pos, device=q.device).unsqueeze(0)
            if is_ring and self.sliding_window is not None:
                attn_mask = (k_pos <= q_pos) & (k_pos > (q_pos - self.sliding_window))
            else:
                attn_mask = k_pos <= q_pos  # [q_len, total_len] bool mask (True = keep)
            return _sdpa(q, k_full, v_full, attn_mask=attn_mask, scale=sm_scale)

        # Path 3a: Fused Triton GQA flash attention (single pass, tensor-core `tl.dot`, reads quantized
        # KV in-SRAM — no dequant materialization). Used for big contexts and softcap when possible.
        if HAS_TRITON and q.is_cuda and not is_ring and head_dim in (32, 64, 128, 256):
            dev = self.k_buf.device
            with cuda_device_scope(dev):
                assert self.k_buf is not None and self.v_buf is not None
                kv_kind = {"fp16": 0, "int8": 1, "int4": 2}[self.kv_dtype]
                q_c = q.to(device=dev)
                if q_c.dtype != torch.float16:
                    q_c = q_c.to(torch.float16)
                q2 = q_c.squeeze(0).contiguous()  # [n_heads, q_len, head_dim]
                out_c = torch.empty((n_heads, q_len, head_dim), device=dev, dtype=torch.float16)
                if self.kv_dtype == "int4":
                    k_src, v_src, ks_src, vs_src = self.k_buf, self.v_buf, self.k_scale, self.v_scale
                elif self.kv_dtype == "int8":
                    k_src, v_src, ks_src, vs_src = self.k_buf, self.v_buf, self.k_scale, self.v_scale
                else:  # fp16: scales unused (dummy pointers, never loaded)
                    k_src, v_src, ks_src, vs_src = self.k_buf, self.v_buf, self.k_buf, self.v_buf
                block_m = 64 if q_len >= 64 else (32 if q_len >= 32 else 16)
                softcap_val = float(softcap)
                grid = (triton.cdiv(q_len, block_m), n_heads)
                _gqa_flash_attn_kernel[grid](
                    q2,
                    k_src,
                    ks_src,
                    v_src,
                    vs_src,
                    out_c,
                    start_pos,
                    total_len,
                    self.capacity,
                    sm_scale,
                    q_len,
                    KV_DTYPE=kv_kind,
                    SOFTCAP=softcap_val,
                    GQA_GROUP=gqa_group,
                    BLOCK_M=block_m,
                    BLOCK_N=64,
                    HEAD_DIM=head_dim,
                    num_warps=4,
                    num_stages=2,
                )
                return out_c.to(q.dtype).view(1, n_heads, q_len, head_dim)

        # Path 3b: Online-Softmax Chunked Attention fallback (CPU / non-pow2 head_dim / ring+softcap).
        # GQA group is folded into the query-row dimension (`2*qc` rows per kv-head) so K/V chunks are
        # never expanded; tensor-core fp16/bf16 matmuls with FP32 online-softmax bookkeeping.
        n_kv = self.n_kv_heads
        q_chunk_len = 512
        outputs: List[torch.Tensor] = []
        # Prefill matmuls prefer fp16 compute on CUDA tensor cores; CPU stays fp32 for accuracy.
        compute_dtype = torch.float16 if q.is_cuda and head_dim % 8 == 0 else torch.float32

        for q_off in range(0, q_len, q_chunk_len):
            q_end = min(q_len, q_off + q_chunk_len)
            qc_len = q_end - q_off
            q_sub = q[:, :, q_off:q_end, :]  # [1, H, qc_len, D]
            n_elems_g = gqa_group * qc_len
            # [1, H=nkv*g, qc, D] -> [1, nkv, g*qc, D] (row r = j*qc + s covers group head j, position s)
            q_fold = (
                q_sub.to(compute_dtype)
                .view(1, n_kv, gqa_group, qc_len, head_dim)
                .reshape(1, n_kv, n_elems_g, head_dim)
            )
            # Per folded row r: query position = start_pos + q_off + (r % qc_len)
            row_abs_pos = (torch.arange(n_elems_g, device=q.device) % qc_len) + (start_pos + q_off)
            max_visible_k = min(total_len, start_pos + q_end)

            m_i = torch.full((1, n_kv, n_elems_g, 1), float("-inf"), device=q.device, dtype=torch.float32)
            l_i = torch.zeros((1, n_kv, n_elems_g, 1), device=q.device, dtype=torch.float32)
            acc_o = torch.zeros((1, n_kv, n_elems_g, head_dim), device=q.device, dtype=torch.float32)

            for kv_off in range(0, max_visible_k, kv_chunk_size):
                kv_end = min(total_len, kv_off + kv_chunk_size)
                k_chunk, v_chunk = self.get_dequant_slice(kv_off, kv_end, compute_dtype)

                scores = torch.matmul(q_fold, k_chunk.transpose(-2, -1)).float() * sm_scale
                if softcap > 0.0:
                    scores = torch.tanh(scores / softcap) * softcap

                if kv_end > start_pos + q_off:
                    # Only the chunks overlapping the causal boundary need a mask
                    k_abs_pos = torch.arange(kv_off, kv_end, device=q.device).view(1, 1, 1, -1)
                    scores = scores.masked_fill(k_abs_pos > row_abs_pos.view(1, 1, n_elems_g, 1), float("-inf"))

                m_ij = scores.amax(dim=-1, keepdim=True)
                m_new = torch.maximum(m_i, m_ij)
                # Guard against (-inf) - (-inf) = NaN when an entire row is masked
                m_new_safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)
                m_i_safe = torch.where(torch.isinf(m_i), torch.zeros_like(m_i), m_i)

                alpha = torch.where(torch.isinf(m_i), torch.zeros_like(m_i), torch.exp(m_i_safe - m_new_safe))
                p = torch.exp(scores - m_new_safe)

                acc_o = acc_o * alpha + torch.matmul(p.to(compute_dtype), v_chunk).float()
                l_i = l_i * alpha + p.sum(dim=-1, keepdim=True)
                m_i = m_new

            out = (acc_o / l_i.clamp(min=1e-20)).to(q.dtype)
            # Unfold: [1, nkv, g*qc, D] -> [1, nkv, g, qc, D] -> [1, H, qc, D] (head h = kv_head * g + j)
            outputs.append(out.view(1, n_kv, gqa_group, qc_len, head_dim).reshape(1, n_heads, qc_len, head_dim))

        return torch.cat(outputs, dim=2)


class EngineKVCacheManager:
    """
    Coordinates per-layer KV caches (for full-attention layers) AND fixed-size recurrent states
    (for Gated-DeltaNet / SSM hybrid layers) across 1 or 2+ GPUs, while tracking cached token IDs
    for instant Prefix Cache Reuse across multi-turn OpenAI chat conversations.
    """

    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        layer_devices: Sequence[torch.device],
        kv_dtype: Literal["int4", "int8", "fp16"] = "int8",
        recurrent_layers: Optional[Sequence[bool]] = None,
        tp_devices: Optional[Sequence[torch.device]] = None,
    ) -> None:
        rec_flags = list(recurrent_layers) if recurrent_layers is not None else [False] * n_layers
        self.has_recurrent = any(rec_flags)
        self.tp_devices: List[torch.device] = list(tp_devices) if tp_devices and len(tp_devices) > 1 else []
        tp_world = len(self.tp_devices)

        if tp_world > 1:
            self.layers: List[Optional[LayerKVCache]] = [None] * n_layers
            self.recurrent_states: List[Optional[LayerRecurrentState]] = [None] * n_layers
            self.tp_layers: List[Optional[List[LayerKVCache]]] = [
                None
                if rec_flags[i]
                else [
                    LayerKVCache(
                        n_kv_heads=n_kv_heads // tp_world,
                        head_dim=head_dim,
                        max_seq_len=max_seq_len,
                        device=dev,
                        kv_dtype=kv_dtype,
                    )
                    for dev in self.tp_devices
                ]
                for i in range(n_layers)
            ]
            self.tp_recurrent_states: List[Optional[List[LayerRecurrentState]]] = [
                [LayerRecurrentState(device=dev) for dev in self.tp_devices] if rec_flags[i] else None
                for i in range(n_layers)
            ]
            # expose rank-0 caches on `.layers` for backward-compatible inspection
            for i in range(n_layers):
                if self.tp_layers[i] is not None:
                    self.layers[i] = self.tp_layers[i][0]
                if self.tp_recurrent_states[i] is not None:
                    self.recurrent_states[i] = self.tp_recurrent_states[i][0]
        else:
            self.layers = [
                None
                if rec_flags[i]
                else LayerKVCache(
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    max_seq_len=max_seq_len,
                    device=layer_devices[i],
                    kv_dtype=kv_dtype,
                )
                for i in range(n_layers)
            ]
            self.recurrent_states = [
                LayerRecurrentState(device=layer_devices[i]) if rec_flags[i] else None
                for i in range(n_layers)
            ]
            self.tp_layers = [None] * n_layers
            self.tp_recurrent_states = [None] * n_layers

        self.cached_tokens: List[int] = []
        self.recurrent_seq_len: int = 0
        # Lightweight CPU-offloaded prompt checkpoint for hybrid models (0 MB GPU VRAM overhead)
        self._ckpt_len: int = 0
        self._ckpt_states: Dict[Tuple[int, int], Optional[Tuple[torch.Tensor, torch.Tensor]]] = {}

        # Private MTP (`blk.{n_layers}.nextn.*`) full-attention KV cache (supports compact `mtp_cache_start` + ring window)
        self._max_seq_len = max_seq_len
        self._kv_dtype: Literal["int4", "int8", "fp16"] = kv_dtype
        self.has_mtp: bool = False
        self.mtp_kv_window: Optional[int] = None
        self.mtp_kv_cache: Optional[LayerKVCache] = None
        self.mtp_tp_kv_caches: List[LayerKVCache] = []
        self.mtp_cache_start: int = 0
        self.mtp_exact_kv_len: int = 0
        self._mtp_ckpt_len: int = 0
        self._mtp_ckpt_h: Optional[torch.Tensor] = None

    def init_mtp_cache(
        self,
        n_kv_heads: int,
        head_dim: int,
        device: torch.device,
        tp_devices: Optional[Sequence[torch.device]] = None,
        sliding_window: Optional[int] = 4096,
    ) -> None:
        """
        Allocate private full-attention KV cache for the embedded GGUF MTP (`NextN`) draft block.
        Defaults to a `4096`-token Sliding-Window Ring Buffer (`~4 MiB` total VRAM even at 256K trunk context!).
        """
        win = int(sliding_window) if (sliding_window is not None and int(sliding_window) > 0) else None
        self.mtp_kv_window = win
        tp_list = list(tp_devices) if tp_devices and len(tp_devices) > 1 else []
        if tp_list:
            tp_world = len(tp_list)
            self.mtp_tp_kv_caches = [
                LayerKVCache(
                    n_kv_heads=n_kv_heads // tp_world,
                    head_dim=head_dim,
                    max_seq_len=self._max_seq_len,
                    device=dev,
                    kv_dtype=self._kv_dtype,
                    sliding_window=win,
                )
                for dev in tp_list
            ]
            self.mtp_kv_cache = self.mtp_tp_kv_caches[0]
        else:
            self.mtp_tp_kv_caches = []
            self.mtp_kv_cache = LayerKVCache(
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                max_seq_len=self._max_seq_len,
                device=device,
                kv_dtype=self._kv_dtype,
                sliding_window=win,
            )
        self.has_mtp = True
        self.mtp_cache_start = 0
        self.mtp_exact_kv_len = 0

    def _iter_mtp_kv_caches(self) -> List[LayerKVCache]:
        if self.mtp_tp_kv_caches:
            return list(self.mtp_tp_kv_caches)
        if self.mtp_kv_cache is not None:
            return [self.mtp_kv_cache]
        return []

    def reset_mtp_cache(self, cache_start: int = 0) -> None:
        """Rebase private MTP KV cache to start compactly at `cache_start` (`kv_pos = abs_pos - cache_start`)."""
        self.mtp_cache_start = max(0, int(cache_start))
        self.mtp_exact_kv_len = self.mtp_cache_start
        for lkv in self._iter_mtp_kv_caches():
            lkv.rewind_to(0)

    def rollback_mtp_to(self, abs_len: int) -> None:
        """Rewind private MTP KV cache to `abs_len` (absolute token position)."""
        if not self.has_mtp:
            return
        if abs_len < self.mtp_cache_start:
            self.reset_mtp_cache(abs_len)
            return
        rel_len = abs_len - self.mtp_cache_start
        for lkv in self._iter_mtp_kv_caches():
            lkv.rewind_to(rel_len)
        self.mtp_exact_kv_len = min(self.mtp_exact_kv_len, abs_len)

    def _iter_all_kv_caches(self) -> List[LayerKVCache]:
        out: List[LayerKVCache] = []
        if self.tp_devices:
            for grp in self.tp_layers:
                if grp is not None:
                    out.extend(grp)
        else:
            for lkv in self.layers:
                if lkv is not None:
                    out.append(lkv)
        return out

    def _iter_all_recurrent_states(self) -> List[Tuple[Tuple[int, int], LayerRecurrentState]]:
        out: List[Tuple[Tuple[int, int], LayerRecurrentState]] = []
        if self.tp_devices:
            for idx, grp in enumerate(self.tp_recurrent_states):
                if grp is not None:
                    for rank, st in enumerate(grp):
                        out.append(((idx, rank), st))
        else:
            for idx, st in enumerate(self.recurrent_states):
                if st is not None:
                    out.append(((idx, 0), st))
        return out

    def ensure_all_kv_capacity(self, needed_len: int) -> None:
        """Pre-grow all KV cache pages outside CUDA Graph capture so buffer addresses stay static."""
        for lkv in self._iter_all_kv_caches():
            lkv._ensure_capacity(needed_len)
        if self.has_mtp:
            rel_need = max(1, needed_len - self.mtp_cache_start)
            for lkv in self._iter_mtp_kv_caches():
                lkv._ensure_capacity(rel_need)

    def match_prefix_length(self, prompt_tokens: Sequence[int]) -> int:
        """
        Return how many leading tokens of `prompt_tokens` are already materialized in cache.
        Always leaves at least 1 token to forward so logits for the next token are freshly produced!
        """
        if not self.cached_tokens or not prompt_tokens:
            return 0
        limit = min(len(self.cached_tokens), len(prompt_tokens) - 1)
        matched = 0
        while matched < limit and self.cached_tokens[matched] == prompt_tokens[matched]:
            matched += 1

        if not self.has_recurrent:
            return matched

        # Hybrid recurrent models: live GPU state sits at `self.recurrent_seq_len` (instant multi-turn hit!)
        if matched == self.recurrent_seq_len:
            return matched
        if self._ckpt_len > 0 and matched >= self._ckpt_len:
            return self._ckpt_len
        return 0

    def save_prompt_checkpoint(self) -> None:
        """Save CPU-offloaded recurrent state checkpoint at prompt boundary for shared-system-prompt reuse."""
        if not self.has_recurrent:
            return
        cpu_dev = torch.device("cpu")
        self._ckpt_len = self.recurrent_seq_len
        self._ckpt_states = {
            key: st.clone_state(target_device=cpu_dev)
            for key, st in self._iter_all_recurrent_states()
        }

    def begin_recurrent_verify(self) -> None:
        """Arm recurrent layers to cache lightweight scan inputs for zero-recompute rejection rollback."""
        if not self.has_recurrent:
            return
        for _key, st in self._iter_all_recurrent_states():
            st.verify_active = True
            if getattr(st, "_graph_verify_cache", None) is not None:
                # Verify CUDA-graph mode (`forward_verify_step`): the captured graph rewrites the
                # fixed verify-cache buffers on every replay; re-point at them instead of dropping them.
                st._verify_cache = st._graph_verify_cache
            else:
                st._verify_cache = None

    def commit_recurrent_verify(self, start_pos: int, n_keep: int) -> bool:
        """
        Commit `n_keep` verified rows across all recurrent layers in ~0.2 ms (no trunk re-forward!).
        Returns `True` if every recurrent layer settled via cached scan inputs.
        """
        if not self.has_recurrent:
            return True
        all_ok = True
        for _key, st in self._iter_all_recurrent_states():
            if not st.commit_verify_prefix(n_keep):
                all_ok = False
        if all_ok:
            self.recurrent_seq_len = start_pos + n_keep
        return all_ok

    def end_recurrent_verify(self) -> None:
        if not self.has_recurrent:
            return
        for _key, st in self._iter_all_recurrent_states():
            st.clear_verify_inputs()

    def snapshot_recurrent_gpu(self) -> Dict[Tuple[int, int], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Fast on-GPU snapshot of recurrent states prior to speculative verification."""
        if not self.has_recurrent:
            return {}
        return {
            key: st.clone_state()
            for key, st in self._iter_all_recurrent_states()
        }

    def restore_recurrent_gpu(
        self,
        snap: Dict[Tuple[int, int], Optional[Tuple[torch.Tensor, torch.Tensor]]],
        seq_len: int,
    ) -> None:
        if not self.has_recurrent:
            return
        for key, st in self._iter_all_recurrent_states():
            if key in snap:
                st.restore_state(snap[key])
        self.recurrent_seq_len = seq_len

    def rollback_to(self, length: int) -> None:
        """Rollback attention KV caches (O(1)) and synchronize hybrid recurrent states cleanly."""
        for layer_kv in self._iter_all_kv_caches():
            layer_kv.rewind_to(length)
        del self.cached_tokens[length:]
        if self.has_mtp:
            if length == 0:
                self.reset_mtp_cache(0)
            else:
                self.rollback_mtp_to(length)

        if self.has_recurrent:
            if length == 0:
                for _key, st in self._iter_all_recurrent_states():
                    st.reset()
                self.recurrent_seq_len = 0
            elif length == self.recurrent_seq_len:
                pass
            elif length == self._ckpt_len and self._ckpt_states:
                for key, st in self._iter_all_recurrent_states():
                    st.restore_state(self._ckpt_states.get(key))
                self.recurrent_seq_len = length

    def record_tokens(self, start_pos: int, new_tokens: Sequence[int]) -> None:
        del self.cached_tokens[start_pos:]
        self.cached_tokens.extend(int(t) for t in new_tokens)
        if self.has_recurrent:
            self.recurrent_seq_len = len(self.cached_tokens)

    def clear(self) -> None:
        self._ckpt_len = 0
        self._ckpt_states.clear()
        self._mtp_ckpt_len = 0
        self._mtp_ckpt_h = None
        self.rollback_to(0)

    def allocated_vram_bytes(self) -> int:
        total = 0
        for lkv in self._iter_all_kv_caches() + self._iter_mtp_kv_caches():
            if lkv.k_buf is not None:
                total += lkv.k_buf.numel() * lkv.k_buf.element_size() * 2
            if lkv.k_scale is not None:
                total += lkv.k_scale.numel() * lkv.k_scale.element_size() * 2
        for _key, st in self._iter_all_recurrent_states():
            total += st.allocated_bytes()
        return total

    def projected_max_vram_bytes(self) -> int:
        """Exact bytes the KV buffers would occupy at their configured full capacity
        (non-ring layers: `max_seq_len`; sliding-window layers: their ring cap).
        Does NOT allocate anything; recurrent-state bytes are counted as already allocated."""
        total = 0
        for lkv in self._iter_all_kv_caches() + self._iter_mtp_kv_caches():
            cap = lkv._ring_cap if lkv._ring_cap is not None else lkv.max_seq_len
            if lkv.kv_dtype == "int4":
                per_pos_per_head = lkv.head_dim + 8  # K+V int4 packed nibbles + fp16 K/V scales
            elif lkv.kv_dtype == "int8":
                per_pos_per_head = 2 * lkv.head_dim + 4  # K+V int8 + fp16 K/V scales
            else:
                per_pos_per_head = 4 * lkv.head_dim  # K+V fp16
            total += cap * lkv.n_kv_heads * per_pos_per_head
        for _key, st in self._iter_all_recurrent_states():
            total += st.allocated_bytes()
        return total
