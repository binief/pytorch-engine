"""
Tests for the opt-in CPU-RAM KV spill (`kv_cpu_spill=True`).

CPU-verifiable round-trips through `EngineKVCacheManager.spill_kv_to_cpu()` /
`restore_kv_prefix()`:
  1. Non-hybrid round-trip: spill frees the GPU/CPU buffers and restore brings back bitwise-
     identical KV rows for a matching prompt (full and partial prefix).
  2. Hybrid (recurrent) layers: full-boundary anchor restores everything; the mid-sequence
     prompt checkpoint clamps the restore to the checkpoint boundary (matching the live
     prefix-reuse semantics); anything else falls back to full re-prefill (returns 0).
  3. `clear()` drops a pending spill; a non-matching prompt does not consume it.
"""

from __future__ import annotations

import torch

from pygguf_engine.kv_cache import EngineKVCacheManager


def _fill_attention(mgr: EngineKVCacheManager, tokens: list[int], seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    head_dim = 32
    refs = []
    for i, layer in enumerate(mgr.layers):
        if layer is None:
            continue
        layer.append(torch.randn(1, 2, len(tokens), head_dim, generator=g),
                     torch.randn(1, 2, len(tokens), head_dim, generator=g), 0)
        refs.append(layer.get_dequant_slice(0, len(tokens), torch.float32))
    mgr.record_tokens(0, tokens)
    return {"refs": refs}


def _assert_layers_match(mgr: EngineKVCacheManager, refs: list, rows: int) -> None:
    idx = 0
    for layer in mgr.layers:
        if layer is None:
            continue
        ref_k, ref_v = refs[idx]
        got_k, got_v = layer.get_dequant_slice(0, rows, torch.float32)
        assert torch.allclose(ref_k[:, :, :rows], got_k, atol=0, rtol=0)
        assert torch.allclose(ref_v[:, :, :rows], got_v, atol=0, rtol=0)
        assert layer.seq_len == rows and layer.logical_seq_len == rows
        idx += 1


def test_kv_spill_roundtrip_nonhybrid_cpu() -> None:
    dev = torch.device("cpu")
    tokens = list(range(1000, 1040))
    mgr = EngineKVCacheManager(
        n_layers=2, n_kv_heads=2, head_dim=32, max_seq_len=128,
        layer_devices=[dev, dev], kv_dtype="fp16", kv_cpu_spill=True,
    )
    refs = _fill_attention(mgr, tokens, seed=7)["refs"]
    assert all(layer is None or layer.k_buf is not None for layer in mgr.layers)

    info = mgr.spill_kv_to_cpu()
    assert info is not None and info["len"] == len(tokens) and info["bytes"] > 0
    assert mgr.has_spill
    assert all(layer is None or (layer.k_buf is None and layer.capacity == 0) for layer in mgr.layers)
    assert mgr.allocated_vram_bytes() == 0

    # Non-matching prompt does not consume the spill
    assert mgr.restore_kv_prefix(list(range(5000, 5050))) == 0
    assert mgr.has_spill

    # Matching prompt restores the full prefix boundary
    out = mgr.restore_kv_prefix(tokens + [9000, 9001])
    assert out == len(tokens)
    assert not mgr.has_spill
    assert mgr.cached_tokens == tokens
    _assert_layers_match(mgr, refs, len(tokens))

    # clear() drops a pending spill
    mgr.spill_kv_to_cpu()
    assert mgr.has_spill
    mgr.clear()
    assert not mgr.has_spill
    print("  [OK] non-hybrid KV spill round-trip (full restore, no-match, clear)!")


def test_kv_spill_partial_prefix_cpu() -> None:
    dev = torch.device("cpu")
    tokens = list(range(2000, 2040))
    mgr = EngineKVCacheManager(
        n_layers=1, n_kv_heads=2, head_dim=32, max_seq_len=128,
        layer_devices=[dev], kv_dtype="int8", kv_cpu_spill=True,
    )
    refs = _fill_attention(mgr, tokens, seed=11)["refs"]
    mgr.spill_kv_to_cpu()
    # Only the first 18 tokens match the new prompt
    out = mgr.restore_kv_prefix(tokens[:18] + [424242])
    assert out == 18
    assert mgr.cached_tokens == tokens[:18]
    _assert_layers_match(mgr, refs, 18)
    print("  [OK] partial-prefix spill restore serves the common prefix only!")


def test_kv_spill_hybrid_anchors_cpu() -> None:
    dev = torch.device("cpu")
    tokens = list(range(3000, 3040))
    mgr = EngineKVCacheManager(
        n_layers=2, n_kv_heads=2, head_dim=32, max_seq_len=128,
        layer_devices=[dev, dev], kv_dtype="fp16",
        recurrent_layers=[True, False], kv_cpu_spill=True,
    )
    g = torch.Generator().manual_seed(13)
    for st in mgr._iter_all_recurrent_states():
        _key, state = st
        if state is None:
            continue
        state.conv_state = torch.randn(1, 16, 3, generator=g)
        state.ssm_state = torch.randn(2, 4, 4, generator=g)
    refs = _fill_attention(mgr, tokens, seed=17)["refs"]  # also record_tokens(0, tokens) -> recurrent_seq_len=40
    rec_before = mgr._iter_all_recurrent_states()[0][1].ssm_state.clone()
    mgr.spill_kv_to_cpu()
    assert all(st is None or st[1] is None or st[1].ssm_state is None for st in mgr._iter_all_recurrent_states())

    # Full-boundary restore (new prompt extends the spilled conversation)
    out = mgr.restore_kv_prefix(tokens + [777])
    assert out == len(tokens)
    st = mgr._iter_all_recurrent_states()[0][1]
    assert st.ssm_state is not None and torch.allclose(st.ssm_state, rec_before)
    _assert_layers_match(mgr, refs, len(tokens))

    # ckpt anchor: rewind the recurrent checkpoint to token 24, then a partial match clamps to it
    mgr2 = EngineKVCacheManager(
        n_layers=2, n_kv_heads=2, head_dim=32, max_seq_len=128,
        layer_devices=[dev, dev], kv_dtype="fp16",
        recurrent_layers=[True, False], kv_cpu_spill=True,
    )
    g2 = torch.Generator().manual_seed(19)
    _key2, state2 = mgr2._iter_all_recurrent_states()[0]
    assert state2 is not None
    state2.conv_state = torch.randn(1, 16, 3, generator=g2)
    state2.ssm_state = torch.randn(2, 4, 4, generator=g2)
    refs2 = []
    for layer in mgr2.layers:
        if layer is None:
            continue
        layer.append(torch.randn(1, 2, 24, 32, generator=g2), torch.randn(1, 2, 24, 32, generator=g2), 0)
        refs2.append(layer.get_dequant_slice(0, 24, torch.float32))
    mgr2.record_tokens(0, tokens[:24])
    mgr2.save_prompt_checkpoint()  # ckpt anchor at 24
    for layer in mgr2.layers:
        if layer is None:
            continue
        layer.append(torch.randn(1, 2, 16, 32, generator=g2), torch.randn(1, 2, 16, 32, generator=g2), 24)
    mgr2.record_tokens(24, tokens[24:])
    mgr2.spill_kv_to_cpu()
    # Prompt matches 30 tokens; clamped to the 24-token ckpt anchor
    out = mgr2.restore_kv_prefix(tokens[:30] + [555])
    assert out == 24
    assert mgr2.cached_tokens == tokens[:24]
    _assert_layers_match(mgr2, refs2, 24)
    mgr2.rollback_to(24)  # restores recurrent state from the prompt checkpoint
    st2 = mgr2._iter_all_recurrent_states()[0][1]
    assert st2.ssm_state is not None

    # No usable anchor (matched < ckpt boundary) -> 0, spill retained
    mgr2.spill_kv_to_cpu()  # re-spill the restored 24-token prefix
    assert mgr2.has_spill
    out = mgr2.restore_kv_prefix(tokens[:10] + [555])
    assert out == 0 and mgr2.has_spill
    mgr2.clear()
    assert not mgr2.has_spill
    print("  [OK] hybrid spill restore honors full-boundary and ckpt anchors!")


if __name__ == "__main__":
    test_kv_spill_roundtrip_nonhybrid_cpu()
    test_kv_spill_partial_prefix_cpu()
    test_kv_spill_hybrid_anchors_cpu()
    print("=== KV SPILL TESTS PASSED ===")
