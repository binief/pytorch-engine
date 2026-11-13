"""
Tests for the fused Triton GQA flash-attention kernel (`_gqa_flash_attn_kernel`, Path 3a).

CPU-verifiable:
  1. `TRITON_INTERPRET=1` subprocess executes the kernel over fp16/int8/int4 KV (softcap 0 & 4.0,
     q_len=3 verify-style with `start_pos>0`, and q_len=17 prefill-style with causal masking) and
     compares against an independent fp64 online-softmax torch reference.
  2. `attend()` on CPU keeps working via the torch fallback (Path 3b) — shape/dtype contract.

On-GPU (Kaggle) the kernel replaces the torch online-softmax path for `total_len > kv_chunk_size`
or `softcap > 0`; win/loss must be measured there (no T4 numbers are claimed in this file).
"""

from __future__ import annotations

import os
import subprocess
import sys

import torch

from pygguf_engine.kv_cache import (
    HAS_TRITON,
    LayerKVCache,
    _FLASH_ATTN_SMEM_LIMIT,
    _gqa_flash_attn_config,
    _gqa_flash_attn_stage_bytes,
)


def test_gqa_flash_attn_interpreter_parity() -> None:
    interp_code = r"""
import os
os.environ["TRITON_INTERPRET"] = "1"
import torch, triton
from pygguf_engine.kv_cache import _gqa_flash_attn_kernel, _kv_append_int8_kernel, _kv_append_int4_kernel

torch.manual_seed(2026)

def reference_attn(q, k, v, sm_scale, start_pos, softcap):
    # q: [H, Q, D] fp32, k/v: [n_kv, total, D] fp32 -> fp64 reference, GQA folded
    H, Q, D = q.shape
    n_kv = k.shape[0]
    g = H // n_kv
    out = torch.empty((H, Q, D), dtype=torch.float64)
    qf = q.double()
    for h in range(H):
        kv = h // g
        for i in range(Q):
            abs_pos = start_pos + i
            keys = k[kv, : abs_pos + 1].double()
            vals = v[kv, : abs_pos + 1].double()
            scores = keys @ qf[h, i] * sm_scale
            if softcap > 0.0:
                scores = torch.tanh(scores / softcap) * softcap
            p = torch.softmax(scores, dim=-1)
            out[h, i] = p @ vals
    return out

def run_case(kv_kind, softcap, q_len, total_len):
    H, n_kv, D = 4, 2, 64
    g = H // n_kv
    start_pos = total_len - q_len
    cap = total_len + 16
    q = torch.randn(H, q_len, D, dtype=torch.float16)

    sm_scale = 1.0 / (D ** 0.5)
    if kv_kind == "fp16":
        k = torch.randn(n_kv, cap, D, dtype=torch.float16) * 2.0
        v = torch.randn(n_kv, cap, D, dtype=torch.float16) * 2.0
        kf, vf = k.float(), v.float()
        k_src, v_src, ks_src, vs_src = k, v, k, v
        kid = 0
    elif kv_kind == "int8":
        k_full = torch.randn(n_kv, total_len, D, dtype=torch.float16) * 2.0
        v_full = torch.randn(n_kv, total_len, D, dtype=torch.float16) * 2.0
        kb = torch.zeros(n_kv, cap, D, dtype=torch.int8)
        vb = torch.zeros(n_kv, cap, D, dtype=torch.int8)
        ks = torch.zeros(n_kv, cap, dtype=torch.float16)
        vs = torch.zeros(n_kv, cap, dtype=torch.float16)
        _kv_append_int8_kernel[(n_kv, total_len)](k_full, v_full, kb, vb, ks, vs, 0, total_len, cap,
                                                  HEAD_DIM=D, BLOCK_D=D)
        k_src, v_src, ks_src, vs_src = kb, vb, ks, vs
        kid = 1
        kf = kb.float() * ks.float().unsqueeze(-1)
        vf = vb.float() * vs.float().unsqueeze(-1)
    else:
        k_full = torch.randn(n_kv, total_len, D, dtype=torch.float16) * 2.0
        v_full = torch.randn(n_kv, total_len, D, dtype=torch.float16) * 2.0
        HD = D // 2
        kb = torch.zeros(n_kv, cap, HD, dtype=torch.uint8)
        vb = torch.zeros(n_kv, cap, HD, dtype=torch.uint8)
        ks = torch.zeros(n_kv, cap, 2, dtype=torch.float16)
        vs = torch.zeros(n_kv, cap, 2, dtype=torch.float16)
        _kv_append_int4_kernel[(n_kv, total_len)](k_full, v_full, kb, vb, ks, vs, 0, total_len, cap,
                                                  HALF_D=HD, BLOCK_HD=HD)
        k_src, v_src, ks_src, vs_src = kb, vb, ks, vs
        kid = 2
        k_lo = ((kb & 0x0F).float() - 8.0) * ks[..., 0:1].float()
        k_hi = (((kb >> 4) & 0x0F).float() - 8.0) * ks[..., 1:2].float()
        kf = torch.cat([k_lo, k_hi], dim=-1)
        v_lo = ((vb & 0x0F).float() - 8.0) * vs[..., 0:1].float()
        v_hi = (((vb >> 4) & 0x0F).float() - 8.0) * vs[..., 1:2].float()
        vf = torch.cat([v_lo, v_hi], dim=-1)

    out = torch.empty((H, q_len, D), dtype=torch.float16)
    BM = 16
    grid = (triton.cdiv(q_len, BM), H)
    _gqa_flash_attn_kernel[grid](q, k_src, ks_src, v_src, vs_src, out,
                                 start_pos, total_len, cap, sm_scale, q_len,
                                 KV_DTYPE=kid, SOFTCAP=float(softcap), GQA_GROUP=g,
                                 BLOCK_M=BM, BLOCK_N=32, HEAD_DIM=D, num_warps=2, num_stages=1)
    ref = reference_attn(q.float(), kf[:, :total_len].float(), vf[:, :total_len].float(),
                         sm_scale, start_pos, float(softcap))
    tol = {0: 6e-3, 1: 1.5e-2, 2: 3e-2}[kid]
    err = (out.double() - ref).abs().max().item()
    name = {0: "fp16", 1: "int8", 2: "int4"}[kid]
    assert err < tol, f"{name} kv softcap={softcap} q_len={q_len} err={err:.4f}"
    print(f"    [ok] {name} KV, softcap={softcap}, q_len={q_len}, total_len={total_len}: max_err={err:.5f}")

for kv_kind in ("fp16", "int8", "int4"):
    for softcap in (0.0, 4.0):
        run_case(kv_kind, softcap, q_len=3, total_len=40)     # verify-style, start_pos>0
        run_case(kv_kind, softcap, q_len=17, total_len=96)    # short prefill, causal boundary
print("  [OK] TRITON_INTERPRET: GQA flash-attention matches the fp64 reference for fp16/int8/int4 KV!")
"""
    env = dict(os.environ)
    env["TRITON_INTERPRET"] = "1"
    proc = subprocess.run([sys.executable, "-c", interp_code], env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"TRITON_INTERPRET flash-attn check failed:\n{proc.stderr}\n{proc.stdout}"
    print(proc.stdout)


def test_gqa_flash_attn_config_fits_t4_smem() -> None:
    # sm_75 (T4): every supported (kv_dtype, head_dim) must fit the 64 KiB per-block shared memory
    for kv_dtype in ("fp16", "int8", "int4"):
        for head_dim in (32, 64, 128, 256):
            for q_len in (1, 16, 64, 512):
                block_m, block_n, warps, stages = _gqa_flash_attn_config(q_len, head_dim, kv_dtype)
                total = stages * _gqa_flash_attn_stage_bytes(block_n, head_dim, kv_dtype)
                assert total <= _FLASH_ATTN_SMEM_LIMIT, (kv_dtype, head_dim, q_len, total)
                assert block_m == (64 if q_len >= 64 else (32 if q_len >= 32 else 16))
                assert warps == 4
    # Regression: fp16 head_dim=256 must no longer request 131072 B (> 65536 B limit)
    block_m, block_n, warps, stages = _gqa_flash_attn_config(512, 256, "fp16")
    assert stages * _gqa_flash_attn_stage_bytes(block_n, 256, "fp16") <= _FLASH_ATTN_SMEM_LIMIT
    assert (block_n, stages) == (32, 2)
    # Pre-existing fast path is preserved for head_dim <= 128
    for kv_dtype in ("fp16", "int8", "int4"):
        assert _gqa_flash_attn_config(512, 128, kv_dtype)[1] == 64
        assert _gqa_flash_attn_config(512, 128, kv_dtype)[3] == 2
    print("  [OK] flash-attn launch config fits the T4 64 KiB shared-memory budget for all KV dtypes!")


def test_attend_cpu_fallback_contract() -> None:
    # Torch fallback path: attend() must return correctly shaped/typed output on CPU regardless.
    dev = torch.device("cpu")
    cache = LayerKVCache(n_kv_heads=2, head_dim=32, max_seq_len=128, device=dev, kv_dtype="fp16", page_growth=8)
    k = torch.randn(1, 2, 40, 32)
    v = torch.randn(1, 2, 40, 32)
    cache.append(k, v, 0)
    q = torch.randn(1, 4, 5, 32)
    out = cache.attend(q, start_pos=35, sm_scale=0.125, softcap=0.0)
    assert out.shape == (1, 4, 5, 32)
    assert out.dtype == q.dtype
    assert torch.isfinite(out.float()).all()
    print("  [OK] attend() CPU fallback returns finite, correctly shaped output!")


if __name__ == "__main__":
    test_attend_cpu_fallback_contract()
    test_gqa_flash_attn_config_fits_t4_smem()
    test_gqa_flash_attn_interpreter_parity()
    print("=== FLASH ATTENTION TESTS PASSED ===")
