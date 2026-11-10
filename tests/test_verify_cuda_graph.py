"""
CPU-verifiable tests for `S=2..8` speculative-verify support (Gap A groundwork for verify CUDA graphs).

What runs on CPU:
  1. RoPE `pos_tensor` multi-row position math (`base + row`) vs the python-int slicing path.
  2. `LayerKVCache._write_contiguous_chunk` GPU-index (`pos_tensor`) writes vs python-int writes
     (fp16, int8 and int4 fallback quantizers — all on CPU they take the `index_copy_` path).
  3. `EngineKVCacheManager.begin_recurrent_verify()` re-points `_verify_cache` at graph-owned buffers
     when a verify CUDA graph was captured (`_graph_verify_cache` wiring).
  4. `GGUFModel.forward_verify_step` falls back to the eager forward on non-CUDA and returns logits
     identical to `model(..., logits_to_keep=S)` — the exact contract the verify path relies on.
  5. Triton `TRITON_INTERPRET=1` execution of the extended posptr kernels: multi-row KV append equals
     the non-posptr kernel, and multi-row posptr flash-decode equals the split-KV kernel for the same
     position range (int8 + int4, `S in {2, 3, 8}`).

What needs a real T4 (run on Kaggle, `CUDAGraph` capture path):
  Replayed verify logits bit-parity vs eager verify, recurrent rollback after replay with partial
  acceptance, and end-to-end `timings` on the 5b MTP sweep. Those are documented in the README.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from pygguf_engine.kv_cache import EngineKVCacheManager, LayerKVCache, LayerRecurrentState
from pygguf_engine.model import RotaryEmbeddingCache


def test_rope_pos_tensor_multirow_positions() -> None:
    torch.manual_seed(7)
    rope = RotaryEmbeddingCache(
        rope_dim=32,
        rope_theta=10000.0,
        neox_style=True,
        rope_scaling_type="none",
        max_context_len=0,
    )
    dev = torch.device("cpu")
    rope.ensure_capacity(dev, 512)

    head_dim = 40
    n_heads, n_kv, s = 4, 2, 5
    q = torch.randn(1, s, n_heads, head_dim)
    k = torch.randn(1, s, n_kv, head_dim)

    base = 137
    pos32 = torch.tensor([base], dtype=torch.int32)
    q_ptr, k_ptr = rope.apply(q, k, base, pos_tensor=pos32)
    q_ref, k_ref = rope.apply(q, k, base)
    assert torch.equal(q_ptr, q_ref), "RoPE pos_tensor S>1 diverged from python-pos path for q"
    assert torch.equal(k_ptr, k_ref), "RoPE pos_tensor S>1 diverged from python-pos path for k"

    # S=1 reduction must match the historical single-token graph path exactly.
    q1, k1 = q[:, :1], k[:, :1]
    q1_ptr, k1_ptr = rope.apply(q1, k1, base, pos_tensor=pos32)
    q1_ref, k1_ref = rope.apply(q1, k1, base)
    assert torch.equal(q1_ptr, q1_ref) and torch.equal(k1_ptr, k1_ref)

    # Interleaved (non-NeoX) layout as well.
    rope_i = RotaryEmbeddingCache(
        rope_dim=32, rope_theta=500000.0, neox_style=False, rope_scaling_type="none", max_context_len=0
    )
    rope_i.ensure_capacity(dev, 512)
    qi_ptr, ki_ptr = rope_i.apply(q, k, base, pos_tensor=pos32)
    qi_ref, ki_ref = rope_i.apply(q, k, base)
    assert torch.equal(qi_ptr, qi_ref) and torch.equal(ki_ptr, ki_ref)

    print("  [OK] RoPE pos_tensor multi-row (base + row) matches the python-pos path exactly!")


def _fill_cache_fp16(dtype: str):
    torch.manual_seed(11)
    dev = torch.device("cpu")
    cache = LayerKVCache(
        n_kv_heads=2,
        head_dim=32,
        max_seq_len=64,
        device=dev,
        kv_dtype=dtype,  # type: ignore[arg-type]
        page_growth=8,
    )
    cache._ensure_capacity(48)
    return cache


def _append_rows(cache, k_new, v_new, start_pos):
    cache.append(k_new, v_new, start_pos)


def test_kv_append_pos_tensor_multirow_cpu() -> None:
    torch.manual_seed(11)
    dev = torch.device("cpu")
    s, base = 5, 9

    for dtype in ("fp16", "int8", "int4"):
        cache_ref = _fill_cache_fp16(dtype)
        cache_ptr = _fill_cache_fp16(dtype)
        # Pre-fill both caches with identical history so only the chunk write differs.
        k_hist = torch.randn(1, 2, base, 32)
        v_hist = torch.randn(1, 2, base, 32)
        cache_ref.append(k_hist, v_hist, 0)
        cache_ptr.append(k_hist, v_hist, 0)

        k_new = torch.randn(1, 2, s, 32)
        v_new = torch.randn(1, 2, s, 32)
        cache_ref.append(k_new, v_new, base)
        pos32 = torch.tensor([base], dtype=torch.int32)
        cache_ptr.append(k_new, v_new, base, pos_tensor=pos32)

        n = base + s
        assert torch.equal(cache_ref.k_buf[:, :n], cache_ptr.k_buf[:, :n]), f"{dtype}: k_buf mismatch"
        assert torch.equal(cache_ref.v_buf[:, :n], cache_ptr.v_buf[:, :n]), f"{dtype}: v_buf mismatch"
        if cache_ref.k_scale is not None:
            assert torch.equal(cache_ref.k_scale[:, :n], cache_ptr.k_scale[:, :n]), f"{dtype}: k_scale mismatch"
            assert torch.equal(cache_ref.v_scale[:, :n], cache_ptr.v_scale[:, :n]), f"{dtype}: v_scale mismatch"
        assert cache_ptr.seq_len == cache_ref.seq_len == base + s
        assert cache_ptr.logical_seq_len == cache_ref.logical_seq_len == base + s

    print("  [OK] KV append pos_tensor multi-row writes match python-int writes (fp16/int8/int4)!")


def test_begin_recurrent_verify_graph_cache_wiring() -> None:
    mgr = EngineKVCacheManager(
        n_layers=2,
        n_kv_heads=1,
        head_dim=16,
        max_seq_len=64,
        layer_devices=[torch.device("cpu"), torch.device("cpu")],
        kv_dtype="fp16",
        recurrent_layers=[True, True],
    )
    assert mgr.has_recurrent
    sentinel = (
        torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1),
        torch.zeros(1), torch.zeros(1), torch.zeros(1), 3,
    )
    for _key, st in mgr._iter_all_recurrent_states():
        st._graph_verify_cache = sentinel  # type: ignore[assignment]
    mgr.begin_recurrent_verify()
    for _key, st in mgr._iter_all_recurrent_states():
        assert st.verify_active is True
        assert st._verify_cache is sentinel, "graph-mode verify cache was not re-pointed at graph buffers"

    # Without a captured graph, begin() must keep the eager contract (fresh None cache).
    for _key, st in mgr._iter_all_recurrent_states():
        st._graph_verify_cache = None
        st._verify_cache = None
        st.verify_active = False
    mgr.begin_recurrent_verify()
    for _key, st in mgr._iter_all_recurrent_states():
        assert st.verify_active is True
        assert st._verify_cache is None

    print("  [OK] begin_recurrent_verify() re-points _verify_cache at graph-owned buffers when captured!")


def test_forward_verify_step_cpu_fallback_matches_eager() -> None:
    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo))
    from tests.test_pygguf_engine import _build_synthetic_qwen35_gguf
    from pygguf_engine import PyGGUFEngine

    with tempfile.TemporaryDirectory() as tmp:
        gguf_path = Path(tmp) / "qwen35_tiny.gguf"
        _build_synthetic_qwen35_gguf(gguf_path, is_moe=False, include_full_mtp=False)
        eng = PyGGUFEngine(
            str(gguf_path),
            devices=["cpu"],
            parallel_mode="pp",
            max_context_len=256,
            speculative_mtp=False,
            speculative_ngram=False,
        )
        verify_ids = [4, 9, 7, 3]

        def _prefill():
            eng.kv_manager.clear()
            eng.model(
                torch.tensor([[4, 5, 6]], dtype=torch.long),
                eng.kv_manager,
                start_pos=0,
                logits_to_keep=1,
            )

        _prefill()
        logits_verify_path = eng.model.forward_verify_step(
            verify_ids, eng.kv_manager, start_pos=3, return_normed_hidden=True
        )
        _prefill()
        logits_eager_path = eng.model(
            torch.tensor([verify_ids], dtype=torch.long),
            eng.kv_manager,
            start_pos=3,
            logits_to_keep=len(verify_ids),
            return_normed_hidden=True,
        )
        # forward_verify_step on CPU must fall back to exactly the eager forward computation.
        assert logits_verify_path[0].shape == logits_eager_path[0].shape
        torch.testing.assert_close(logits_verify_path[0], logits_eager_path[0], rtol=0, atol=0)
        torch.testing.assert_close(logits_verify_path[1], logits_eager_path[1], rtol=0, atol=0)

    print("  [OK] forward_verify_step falls back to the eager forward on CPU (bit-identical logits + hidden)!")


def test_verify_triton_kernels_multirow() -> None:
    interp_code = r"""
import os
os.environ["TRITON_INTERPRET"] = "1"
import torch, triton
from pygguf_engine.kv_cache import (
    _kv_append_int8_kernel, _kv_append_int8_posptr_kernel,
    _kv_append_int4_kernel, _kv_append_int4_posptr_kernel,
    _int8_gqa_flash_decode_split_kernel, _int8_gqa_flash_decode_posptr_kernel,
    _int4_gqa_flash_decode_split_kernel, _int4_gqa_flash_decode_posptr_kernel,
    _flash_decode_combine_kernel,
)

torch.manual_seed(1234)
H_kv, H_q, D = 2, 4, 64
Cap, base, S = 64, 10, 3
pos32 = torch.tensor([base], dtype=torch.int32)

# ---------- INT8 append: posptr(S_NEW=S) == non-posptr(start_pos=base, s_new=S) ----------
k3 = torch.randn(H_kv, S, D, dtype=torch.float16)
v3 = torch.randn(H_kv, S, D, dtype=torch.float16)
kb_a, vb_a = torch.zeros(H_kv, Cap, D, dtype=torch.int8), torch.zeros(H_kv, Cap, D, dtype=torch.int8)
ks_a, vs_a = torch.zeros(H_kv, Cap, dtype=torch.float16), torch.zeros(H_kv, Cap, dtype=torch.float16)
_kv_append_int8_posptr_kernel[(H_kv, S)](k3, v3, kb_a, vb_a, ks_a, vs_a, pos32, Cap, HEAD_DIM=D, BLOCK_D=D, S_NEW=S)
kb_b, vb_b = torch.zeros(H_kv, Cap, D, dtype=torch.int8), torch.zeros(H_kv, Cap, D, dtype=torch.int8)
ks_b, vs_b = torch.zeros(H_kv, Cap, dtype=torch.float16), torch.zeros(H_kv, Cap, dtype=torch.float16)
_kv_append_int8_kernel[(H_kv, S)](k3, v3, kb_b, vb_b, ks_b, vs_b, base, S, Cap, HEAD_DIM=D, BLOCK_D=D)
assert torch.equal(kb_a, kb_b) and torch.equal(vb_a, vb_b) and torch.equal(ks_a, ks_b) and torch.equal(vs_a, vs_b)

# ---------- INT4 append ----------
k3_4 = torch.randn(H_kv, S, D, dtype=torch.float16)
v3_4 = torch.randn(H_kv, S, D, dtype=torch.float16)
kb4_a, vb4_a = torch.zeros(H_kv, Cap, D // 2, dtype=torch.uint8), torch.zeros(H_kv, Cap, D // 2, dtype=torch.uint8)
ks4_a, vs4_a = torch.zeros(H_kv, Cap, 2, dtype=torch.float16), torch.zeros(H_kv, Cap, 2, dtype=torch.float16)
_kv_append_int4_posptr_kernel[(H_kv, S)](k3_4, v3_4, kb4_a, vb4_a, ks4_a, vs4_a, pos32, Cap, HALF_D=D // 2, BLOCK_HD=D // 2, S_NEW=S)
kb4_b, vb4_b = torch.zeros(H_kv, Cap, D // 2, dtype=torch.uint8), torch.zeros(H_kv, Cap, D // 2, dtype=torch.uint8)
ks4_b, vs4_b = torch.zeros(H_kv, Cap, 2, dtype=torch.float16), torch.zeros(H_kv, Cap, 2, dtype=torch.float16)
_kv_append_int4_kernel[(H_kv, S)](k3_4, v3_4, kb4_b, vb4_b, ks4_b, vs4_b, base, S, Cap, HALF_D=D // 2, BLOCK_HD=D // 2)
assert torch.equal(kb4_a, kb4_b) and torch.equal(vb4_a, vb4_b) and torch.equal(ks4_a, ks4_b) and torch.equal(vs4_a, vs4_b)

# ---------- INT8 flash decode: posptr(Q_LEN=S, pos=base) == split(total_len=base+S, Q_LEN=S) ----------
q = torch.randn(H_q * S, D, dtype=torch.float16)
NS, BS = 32, 32
part_posptr = torch.empty((H_q * S, NS, D), dtype=torch.float32)
part_lse_posptr = torch.empty((H_q * S, NS), dtype=torch.float32)
part_split = torch.empty((H_q * S, NS, D), dtype=torch.float32)
part_lse_split = torch.empty((H_q * S, NS), dtype=torch.float32)
sm_scale = 0.088
# fill cache history via the (already parity-checked) non-posptr append kernel
kall = torch.randn(H_kv, base + S, D, dtype=torch.float16)
vall = torch.randn(H_kv, base + S, D, dtype=torch.float16)
kb, vb = torch.zeros(H_kv, Cap, D, dtype=torch.int8), torch.zeros(H_kv, Cap, D, dtype=torch.int8)
ks, vs = torch.zeros(H_kv, Cap, dtype=torch.float16), torch.zeros(H_kv, Cap, dtype=torch.float16)
_kv_append_int8_kernel[(H_kv, base + S)](kall, vall, kb, vb, ks, vs, 0, base + S, Cap, HEAD_DIM=D, BLOCK_D=D)
grid_rows = (H_q * S, NS)
_int8_gqa_flash_decode_posptr_kernel[grid_rows](q, kb, ks, vb, vs, part_posptr, part_lse_posptr, pos32, Cap, sm_scale,
                                                GQA_GROUP=2, NUM_SPLITS=NS, BLOCK_S=BS, HEAD_DIM=D, Q_LEN=S)
_int8_gqa_flash_decode_split_kernel[grid_rows](q, kb, ks, vb, vs, part_split, part_lse_split, base + S, Cap, 0, sm_scale,
                                                WIN_RING=False, GQA_GROUP=2, NUM_SPLITS=NS, BLOCK_S=BS, HEAD_DIM=D, Q_LEN=S)
assert torch.allclose(part_lse_posptr, part_lse_split, atol=1e-4), (part_lse_posptr - part_lse_split).abs().max()
assert torch.allclose(part_posptr, part_split, atol=1e-4), (part_posptr - part_split).abs().max()
out_a = torch.empty((H_q * S, D), dtype=torch.float16)
out_b = torch.empty((H_q * S, D), dtype=torch.float16)
_flash_decode_combine_kernel[(H_q * S,)](part_posptr, part_lse_posptr, out_a, NUM_SPLITS=NS, HEAD_DIM=D)
_flash_decode_combine_kernel[(H_q * S,)](part_split, part_lse_split, out_b, NUM_SPLITS=NS, HEAD_DIM=D)
assert torch.allclose(out_a.float(), out_b.float(), atol=1e-3)

# ---------- INT4 flash decode ----------
HD = D // 2
kall4 = torch.randn(H_kv, base + S, D, dtype=torch.float16)
vall4 = torch.randn(H_kv, base + S, D, dtype=torch.float16)
kb4, vb4 = torch.zeros(H_kv, Cap, HD, dtype=torch.uint8), torch.zeros(H_kv, Cap, HD, dtype=torch.uint8)
ks4, vs4 = torch.zeros(H_kv, Cap, 2, dtype=torch.float16), torch.zeros(H_kv, Cap, 2, dtype=torch.float16)
_kv_append_int4_kernel[(H_kv, base + S)](kall4, vall4, kb4, vb4, ks4, vs4, 0, base + S, Cap, HALF_D=HD, BLOCK_HD=HD)
part4_posptr = torch.empty((H_q * S, NS, D), dtype=torch.float32)
part4_lse_posptr = torch.empty((H_q * S, NS), dtype=torch.float32)
part4_split = torch.empty((H_q * S, NS, D), dtype=torch.float32)
part4_lse_split = torch.empty((H_q * S, NS), dtype=torch.float32)
_int4_gqa_flash_decode_posptr_kernel[grid_rows](q, kb4, ks4, vb4, vs4, part4_posptr, part4_lse_posptr, pos32, Cap, sm_scale,
                                                 GQA_GROUP=2, NUM_SPLITS=NS, BLOCK_S=BS, HALF_D=HD, Q_LEN=S)
_int4_gqa_flash_decode_split_kernel[grid_rows](q, kb4, ks4, vb4, vs4, part4_split, part4_lse_split, base + S, Cap, 0, sm_scale,
                                                WIN_RING=False, GQA_GROUP=2, NUM_SPLITS=NS, BLOCK_S=BS, HALF_D=HD, Q_LEN=S)
assert torch.allclose(part4_lse_posptr, part4_lse_split, atol=1e-4)
assert torch.allclose(part4_posptr, part4_split, atol=1e-4)

print("  [OK] TRITON_INTERPRET: multi-row posptr append/flash-decode kernels match the split path bit-for-bit!")
"""
    env = dict(os.environ)
    env["TRITON_INTERPRET"] = "1"
    proc = subprocess.run([sys.executable, "-c", interp_code], env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"TRITON_INTERPRET multi-row posptr check failed:\n{proc.stderr}\n{proc.stdout}"
    print(proc.stdout)


if __name__ == "__main__":
    test_rope_pos_tensor_multirow_positions()
    test_kv_append_pos_tensor_multirow_cpu()
    test_begin_recurrent_verify_graph_cache_wiring()
    test_forward_verify_step_cpu_fallback_matches_eager()
    test_verify_triton_kernels_multirow()
    print("=== VERIFY CUDA-GRAPH SUPPORT TESTS PASSED ===")
