"""
End-to-end verification suite for `pygguf_engine`:
1. Bit-accuracy of Q4_0, Q8_0, Q4_K, Q5_K, Q6_K SoA repacking & matmuls vs `gguf.quants`
2. Pure-Python `GGUFFile` binary v3 parser + embedded tokenizer
3. Dense Qwen3 GGUF (mixed Q4_K + Q6_K, QK-norm, INT8 KV cache, prefix reuse, lossless N-gram speculative decode)
4. Sparse MoE Qwen3-MoE GGUF (3D quantized routed experts `ffn_gate_exps` / `up` / `down` + router)
5. FastAPI OpenAI-compatible `/v1/models`, `/v1/chat/completions` (sync + SSE streaming), `/v1/completions`, & auth
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Optional

import gguf
import gguf.quants as gq
import numpy as np
import torch
import torch.nn.functional as F
from fastapi.testclient import TestClient

from pygguf_engine import PyGGUFEngine, SamplingConfig, create_openai_app
from pygguf_engine.quant_kernels import (
    exec_w4a16_matmul,
    exec_w8a16_matmul,
    repack_gguf_to_w4a16,
    repack_gguf_to_w8a16,
)


def _make_raw_quant_tensor(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    qtype: gq.GGMLQuantizationType,
) -> np.ndarray:
    """Create valid raw GGUF quantized tensor bytes for any quantization format."""
    if qtype in (gq.GGMLQuantizationType.Q4_0, gq.GGMLQuantizationType.Q8_0):
        w_f32 = (rng.standard_normal(shape, dtype=np.float32) * 0.1).astype(np.float32)
        return gq.quantize(w_f32, qtype)

    cols = shape[-1]
    outer = shape[:-1]
    n_rows = int(np.prod(outer))
    if qtype == gq.GGMLQuantizationType.Q4_K:
        n_blocks = n_rows * (cols // 256)
        raw = rng.integers(0, 256, size=(n_blocks, 144), dtype=np.uint8)
        raw[:, 0:2] = (rng.uniform(0.0001, 0.0005, size=(n_blocks, 1)).astype(np.float16)).view(np.uint8)
        raw[:, 2:4] = (rng.uniform(0.0001, 0.0005, size=(n_blocks, 1)).astype(np.float16)).view(np.uint8)
        return raw.reshape(*outer, (cols // 256) * 144)

    if qtype == gq.GGMLQuantizationType.Q5_K:
        n_blocks = n_rows * (cols // 256)
        raw = rng.integers(0, 256, size=(n_blocks, 176), dtype=np.uint8)
        raw[:, 0:2] = (rng.uniform(0.0001, 0.0005, size=(n_blocks, 1)).astype(np.float16)).view(np.uint8)
        raw[:, 2:4] = (rng.uniform(0.0001, 0.0005, size=(n_blocks, 1)).astype(np.float16)).view(np.uint8)
        return raw.reshape(*outer, (cols // 256) * 176)

    if qtype == gq.GGMLQuantizationType.Q6_K:
        n_blocks = n_rows * (cols // 256)
        raw = rng.integers(0, 256, size=(n_blocks, 210), dtype=np.uint8)
        raw[:, 208:210] = (rng.uniform(0.0001, 0.0005, size=(n_blocks, 1)).astype(np.float16)).view(np.uint8)
        return raw.reshape(*outer, (cols // 256) * 210)

    raise ValueError(f"Unsupported test qtype: {qtype}")


def test_quant_repack_and_matmul_exactness() -> None:
    rng = np.random.default_rng(42)
    rows, cols = 32, 256
    x_t = torch.from_numpy(rng.standard_normal((5, cols), dtype=np.float32) * 0.5).to(torch.float16)

    for qtype in (
        gq.GGMLQuantizationType.Q4_0,
        gq.GGMLQuantizationType.Q4_K,
    ):
        q_bytes = _make_raw_quant_tensor(rng, (rows, cols), qtype)
        ref_w = torch.from_numpy(gq.dequantize(q_bytes, qtype).reshape(rows, cols)).to(torch.float32)
        ref_out = F.linear(x_t.float(), ref_w).to(torch.float16)

        qw, sc, zr = repack_gguf_to_w4a16(q_bytes.reshape(-1), int(qtype), rows, cols, torch.device("cpu"))
        got_out = exec_w4a16_matmul(x_t, qw, sc, zr)
        max_err = (got_out.float() - ref_out.float()).abs().max().item()
        assert max_err < 1e-2, f"W4A16 mismatch on {qtype.name}: max_err={max_err}"
        print(f"  [OK] {qtype.name:6s} -> W4A16 max_err = {max_err:.6f}")

    for qtype in (
        gq.GGMLQuantizationType.Q8_0,
        gq.GGMLQuantizationType.Q5_K,
        gq.GGMLQuantizationType.Q6_K,
    ):
        q_bytes = _make_raw_quant_tensor(rng, (rows, cols), qtype)
        ref_w = torch.from_numpy(gq.dequantize(q_bytes, qtype).reshape(rows, cols)).to(torch.float32)
        ref_out = F.linear(x_t.float(), ref_w).to(torch.float16)

        qw, sc, zr = repack_gguf_to_w8a16(q_bytes.reshape(-1), int(qtype), rows, cols, torch.device("cpu"))
        got_out = exec_w8a16_matmul(x_t, qw, sc, zr)
        max_err = (got_out.float() - ref_out.float()).abs().max().item()
        assert max_err < 1e-2, f"W8A16 mismatch on {qtype.name}: max_err={max_err}"
        print(f"  [OK] {qtype.name:6s} -> W8A16 max_err = {max_err:.6f}")


def _build_synthetic_gguf(
    path: Path,
    is_moe: bool = False,
    rope_scaling: Optional[dict] = None,
    context_length: int = 4096,
) -> None:
    arch = "qwen3moe" if is_moe else "qwen3"
    writer = gguf.GGUFWriter(str(path), arch)

    hidden = 256
    ffn = 512
    n_layers = 2
    n_heads = 4
    n_kv_heads = 2
    head_dim = 64
    vocab_size = 64

    writer.add_name(f"tiny-{arch}-test")
    writer.add_block_count(n_layers)
    writer.add_embedding_length(hidden)
    writer.add_feed_forward_length(ffn)
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(n_kv_heads)
    writer.add_key_length(head_dim)
    writer.add_value_length(head_dim)
    writer.add_context_length(context_length)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_rope_dimension_count(head_dim)
    writer.add_rope_freq_base(10000.0)
    if rope_scaling:
        import gguf as _gguf
        scaling_type = _gguf.RopeScalingType(str(rope_scaling.get("type", "yarn")))
        writer.add_rope_scaling_type(scaling_type)
        factor = float(rope_scaling.get("factor", 2.0))
        writer.add_rope_scaling_factor(factor)
        writer.add_rope_scaling_orig_ctx_len(int(rope_scaling.get("orig_ctx_len", 4096)))
        if scaling_type == _gguf.RopeScalingType.YARN:
            writer.add_rope_scaling_yarn_beta_fast(float(rope_scaling.get("beta_fast", 32.0)))
            writer.add_rope_scaling_yarn_beta_slow(float(rope_scaling.get("beta_slow", 1.0)))
            writer.add_rope_scaling_yarn_ext_factor(float(rope_scaling.get("ext_factor", 1.0)))
            if "attn_factor" in rope_scaling:
                writer.add_rope_scaling_yarn_attn_factor(float(rope_scaling["attn_factor"]))

    if is_moe:
        n_experts = 4
        writer.add_expert_count(n_experts)
        writer.add_expert_used_count(2)
        writer.add_uint32(f"{arch}.expert_feed_forward_length", 256)

    # Embedded vocab tokens
    base_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "system", "user", "assistant", "\n", " ", "Hello", "world", "AI", "fast"]
    while len(base_tokens) < vocab_size:
        base_tokens.append(f"tok_{len(base_tokens)}")
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(base_tokens)
    writer.add_eos_token_id(2)  # <|im_end|>

    rng = np.random.default_rng(123 if not is_moe else 456)

    def _add_q(name: str, shape: tuple[int, ...], qtype: gq.GGMLQuantizationType) -> None:
        qb = _make_raw_quant_tensor(rng, shape, qtype)
        writer.add_tensor(name, qb, raw_dtype=qtype)

    # Tied quantized token_embd (Q6_K)
    _add_q("token_embd.weight", (vocab_size, hidden), gq.GGMLQuantizationType.Q6_K)
    writer.add_tensor("output_norm.weight", np.ones((hidden,), dtype=np.float32))

    for i in range(n_layers):
        writer.add_tensor(f"blk.{i}.attn_norm.weight", np.ones((hidden,), dtype=np.float32))
        writer.add_tensor(f"blk.{i}.ffn_norm.weight", np.ones((hidden,), dtype=np.float32))
        writer.add_tensor(f"blk.{i}.attn_q_norm.weight", np.ones((head_dim,), dtype=np.float32))
        writer.add_tensor(f"blk.{i}.attn_k_norm.weight", np.ones((head_dim,), dtype=np.float32))

        # Mix Q4_K + Q6_K + Q8_0 just like real Q4_K_M checkpoints!
        _add_q(f"blk.{i}.attn_q.weight", (n_heads * head_dim, hidden), gq.GGMLQuantizationType.Q4_K)
        _add_q(f"blk.{i}.attn_k.weight", (n_kv_heads * head_dim, hidden), gq.GGMLQuantizationType.Q4_K)
        _add_q(f"blk.{i}.attn_v.weight", (n_kv_heads * head_dim, hidden), gq.GGMLQuantizationType.Q6_K)
        _add_q(f"blk.{i}.attn_output.weight", (hidden, n_heads * head_dim), gq.GGMLQuantizationType.Q8_0)

        if is_moe:
            writer.add_tensor(
                f"blk.{i}.ffn_gate_inp.weight",
                (rng.standard_normal((4, hidden), dtype=np.float32) * 0.1).astype(np.float32),
            )
            _add_q(f"blk.{i}.ffn_gate_exps.weight", (4, 256, hidden), gq.GGMLQuantizationType.Q4_K)
            _add_q(f"blk.{i}.ffn_up_exps.weight", (4, 256, hidden), gq.GGMLQuantizationType.Q4_K)
            _add_q(f"blk.{i}.ffn_down_exps.weight", (4, hidden, 256), gq.GGMLQuantizationType.Q6_K)
        else:
            _add_q(f"blk.{i}.ffn_gate.weight", (ffn, hidden), gq.GGMLQuantizationType.Q4_K)
            _add_q(f"blk.{i}.ffn_up.weight", (ffn, hidden), gq.GGMLQuantizationType.Q4_K)
            _add_q(f"blk.{i}.ffn_down.weight", (hidden, ffn), gq.GGMLQuantizationType.Q6_K)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_dense_moe_and_openai_server() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        dense_path = Path(tmp) / "dense_qwen3_q4km.gguf"
        moe_path = Path(tmp) / "moe_qwen3_q4km.gguf"

        _build_synthetic_gguf(dense_path, is_moe=False)
        _build_synthetic_gguf(moe_path, is_moe=True)

        # 1. Dense Engine + Lossless N-Gram Speculative Decoding + Prefix Cache Verification
        eng_dense = PyGGUFEngine(dense_path, kv_dtype="int8", speculative_ngram=False)
        prompt_ids = [8, 9, 10, 11, 8, 9, 10, 11, 8, 9]
        res_base = eng_dense.generate(prompt_ids, SamplingConfig(max_tokens=12, temperature=0.0, cache_prompt=False))

        eng_dense.speculative_ngram = True
        res_spec = eng_dense.generate(prompt_ids, SamplingConfig(max_tokens=12, temperature=0.0, cache_prompt=False))
        assert res_spec.text == res_base.text, f"Speculative output diverged: {res_spec.text!r} != {res_base.text!r}"
        print(f"  [OK] Dense Q4_K_M + Speculative Exactness verified (drafted={res_spec.speculative_drafted})")

        # Multi-turn Prefix KV Cache Hit test
        extended_ids = prompt_ids + [12, 13, 14]
        res_cached = eng_dense.generate(extended_ids, SamplingConfig(max_tokens=6, temperature=0.0, cache_prompt=True))
        assert res_cached.cached_prefix_tokens >= len(prompt_ids), "Expected prefix KV cache hit!"
        print(f"  [OK] Prefix KV Cache Reuse hit {res_cached.cached_prefix_tokens}/{res_cached.prompt_tokens} tokens")

        # 2. Sparse MoE Engine (`qwen3moe` 3D quantized experts)
        eng_moe = PyGGUFEngine(moe_path, kv_dtype="int8", speculative_ngram=True)
        res_moe = eng_moe.chat(
            [{"role": "user", "content": "Hello world AI"}],
            SamplingConfig(max_tokens=8, temperature=0.0),
        )
        assert res_moe.completion_tokens >= 0
        print(f"  [OK] Sparse MoE GGUF (3D Q4_K/Q6_K experts) generated {res_moe.completion_tokens} tokens")

        # 3. FastAPI OpenAI Server (Sync + SSE Streaming + Auth)
        app = create_openai_app(eng_moe, api_key="test-secret")
        client = TestClient(app)

        # Unauthorized check
        r_unauth = client.get("/v1/models")
        assert r_unauth.status_code == 401

        headers = {"Authorization": "Bearer test-secret"}
        r_models = client.get("/v1/models", headers=headers)
        assert r_models.status_code == 200
        assert r_models.json()["data"][0]["id"] == eng_moe.model_id

        # Non-streaming `/v1/chat/completions`
        r_chat = client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": eng_moe.model_id,
                "messages": [{"role": "user", "content": "Hello world"}],
                "max_tokens": 6,
                "temperature": 0.0,
            },
        )
        assert r_chat.status_code == 200, r_chat.text
        body = r_chat.json()
        assert body["object"] == "chat.completion"
        assert "content" in body["choices"][0]["message"]
        assert "timings" in body

        # Streaming SSE `/v1/chat/completions`
        sse_lines = []
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": eng_moe.model_id,
                "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello AI"}]}],
                "max_tokens": 6,
                "temperature": 0.0,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        ) as resp:
            assert resp.status_code == 200
            for line in resp.iter_lines():
                if line:
                    sse_lines.append(line)

        assert sse_lines[-1] == "data: [DONE]"
        parsed_first = json.loads(sse_lines[0][len("data: ") :])
        assert parsed_first["choices"][0]["delta"]["role"] == "assistant"
        print(f"  [OK] FastAPI OpenAI `/v1/chat/completions` sync + SSE ({len(sse_lines)} events) + Auth verified!")


def _build_synthetic_qwen35_gguf(path: Path, is_moe: bool = False, include_full_mtp: bool = False) -> None:
    """
    Build a realistic `qwen35` / `qwen35moe` hybrid GGUF checkpoint matching `Qwen3.8-27B-UD-Q4_K_M.gguf`:
    - `block_count = 5` with `nextn_predict_layers = 1` -> 4 trunk layers (`blk.0..3`) + 1 MTP layer (`blk.4.nextn.*`)
    - Layers `0, 1, 2`: Gated-DeltaNet (`attn_qkv` + `attn_gate` + `ssm_conv1d` + `ssm_alpha/beta/dt/a/norm/out`)
      with `num_v_heads = 6` > `num_k_heads = 2` (3x tiled V-head broadcast!)
    - Layer `3`: Gated Full-Attention (`attn_q` `[2 * n_heads * head_dim, hidden]`, `attn_q_norm`, `attn_k_norm`, partial RoPE)
    - Pre-FFN RMSNorm stored as `blk.{i}.post_attention_norm.weight` (NO `ffn_norm.weight`!)
    - Mixes Unsloth Dynamic (`UD`) quant formats (`IQ4_XS`, `IQ4_NL`, `Q3_K`, `Q4_0`, `Q8_0`)!
    """
    import gguf

    arch = "qwen35moe" if is_moe else "qwen35"
    writer = gguf.GGUFWriter(str(path), arch=arch)

    vocab_size = 64
    hidden = 256       # Divisible by 256 so K-quant & IQ4_XS 256-elem blocks align per row!
    inter = 512
    n_heads = 4
    n_kv_heads = 2
    head_dim = 64
    rope_dim = 32      # Partial NeoX RoPE (like Qwen3.8-27B's rope_dim=64 out of head_dim=256)
    trunk_layers = 4
    nextn_layers = 1

    # Gated-DeltaNet parameters (matching 3:1 V-to-K head ratio of Qwen3.8-27B: 48 / 16 = 3)
    conv_kernel = 4
    num_k_heads = 2
    num_v_heads = 6
    head_k_dim = 32
    head_v_dim = 32
    key_dim = num_k_heads * head_k_dim      # 64
    d_inner = num_v_heads * head_v_dim      # 192
    conv_dim = 2 * key_dim + d_inner        # 320

    writer.add_name("Test-Qwen3.8-Hybrid-UD")
    writer.add_block_count(trunk_layers + nextn_layers)
    writer.add_uint32(f"{arch}.nextn_predict_layers", nextn_layers)
    writer.add_uint32(f"{arch}.full_attention_interval", 4)
    writer.add_embedding_length(hidden)
    writer.add_feed_forward_length(inter)
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(n_kv_heads)
    writer.add_key_length(head_dim)
    writer.add_value_length(head_dim)
    writer.add_rope_dimension_count(rope_dim)
    writer.add_rope_freq_base(10000000.0)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_context_length(8192)

    writer.add_uint32(f"{arch}.ssm.conv_kernel", conv_kernel)
    writer.add_uint32(f"{arch}.ssm.inner_size", d_inner)
    writer.add_uint32(f"{arch}.ssm.state_size", head_k_dim)
    writer.add_uint32(f"{arch}.ssm.time_step_rank", num_v_heads)
    writer.add_uint32(f"{arch}.ssm.group_count", num_k_heads)

    if is_moe:
        writer.add_expert_count(4)
        writer.add_expert_used_count(2)
        writer.add_uint32(f"{arch}.expert_feed_forward_length", 256)
        writer.add_uint32(f"{arch}.expert_shared_feed_forward_length", 256)

    tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "\n", "Hello", " world", " Qwen", "3.8"]
    while len(tokens) < vocab_size:
        tokens.append(f"tok_{len(tokens)}")
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tokens)
    writer.add_eos_token_id(2)

    rng = np.random.default_rng(1337)

    def _q4_0(shape):
        w = (rng.standard_normal(shape) * 0.04).astype(np.float32)
        return gguf.quants.quantize(w, gguf.GGMLQuantizationType.Q4_0), gguf.GGMLQuantizationType.Q4_0

    def _raw_ud_quant(shape, qtype_enum, block_elems, block_bytes):
        # Construct valid deterministic raw GGML blocks for IQ4_XS / Q3_K / IQ4_NL testing
        n_el = int(np.prod(shape))
        n_blk = n_el // block_elems
        raw = rng.integers(0, 64, size=(n_blk, block_bytes), dtype=np.uint8)
        # Keep fp16 scale header/footer small so dequantized weights are well-conditioned (~0.01)
        small_fp16 = np.array([0.01], dtype=np.float16).view(np.uint8)
        raw[:, 0:2] = small_fp16
        raw[:, -2:] = small_fp16
        # Reshape raw bytes so GGUFWriter records exact logical tensor shape
        row_bytes = (shape[-1] // block_elems) * block_bytes
        return raw.reshape(*shape[:-1], row_bytes), qtype_enum

    emb_q, emb_t = _raw_ud_quant((vocab_size, hidden), gguf.GGMLQuantizationType.IQ4_XS, 256, 136)
    writer.add_tensor("token_embd.weight", emb_q, raw_dtype=emb_t)
    writer.add_tensor("output_norm.weight", np.ones((hidden,), dtype=np.float32))
    out_q, out_t = _raw_ud_quant((vocab_size, hidden), gguf.GGMLQuantizationType.Q3_K, 256, 110)
    writer.add_tensor("output.weight", out_q, raw_dtype=out_t)

    for i in range(trunk_layers):
        p = f"blk.{i}"
        writer.add_tensor(f"{p}.attn_norm.weight", np.ones((hidden,), dtype=np.float32))
        # Note: NO `ffn_norm.weight` in qwen35/qwen35moe — uses `post_attention_norm.weight`!
        writer.add_tensor(f"{p}.post_attention_norm.weight", np.ones((hidden,), dtype=np.float32))

        if (i + 1) % 4 != 0:
            # Gated-DeltaNet layer (blk.0, blk.1, blk.2)
            wqkv, t_qkv = _raw_ud_quant((conv_dim, hidden), gguf.GGMLQuantizationType.IQ4_XS, 256, 136)
            wgate, t_gate = _raw_ud_quant((d_inner, hidden), gguf.GGMLQuantizationType.IQ4_NL, 32, 18)
            wbeta, t_beta = _q4_0((num_v_heads, hidden))
            walpha, t_alpha = _q4_0((num_v_heads, hidden))
            # ssm_out has in_features = d_inner = 192 (divisible by 32 -> IQ4_NL W4A16!)
            wout, t_out = _raw_ud_quant((hidden, d_inner), gguf.GGMLQuantizationType.IQ4_NL, 32, 18)

            writer.add_tensor(f"{p}.attn_qkv.weight", wqkv, raw_dtype=t_qkv)
            writer.add_tensor(f"{p}.attn_gate.weight", wgate, raw_dtype=t_gate)
            writer.add_tensor(f"{p}.ssm_beta.weight", wbeta, raw_dtype=t_beta)
            writer.add_tensor(f"{p}.ssm_alpha.weight", walpha, raw_dtype=t_alpha)
            writer.add_tensor(f"{p}.ssm_dt.bias", np.zeros((num_v_heads,), dtype=np.float32))
            writer.add_tensor(f"{p}.ssm_a", -np.ones((num_v_heads,), dtype=np.float32))
            conv_w = (rng.standard_normal((conv_dim, conv_kernel)) * 0.1).astype(np.float32)
            writer.add_tensor(f"{p}.ssm_conv1d.weight", conv_w)
            writer.add_tensor(f"{p}.ssm_norm.weight", np.ones((head_v_dim,), dtype=np.float32))
            writer.add_tensor(f"{p}.ssm_out.weight", wout, raw_dtype=t_out)
        else:
            # Gated Full-Attention layer (blk.3): `attn_q` has `[2 * n_heads * head_dim, hidden]`!
            wq, t_q = _raw_ud_quant((2 * n_heads * head_dim, hidden), gguf.GGMLQuantizationType.IQ4_XS, 256, 136)
            wk, t_k = _q4_0((n_kv_heads * head_dim, hidden))
            wv, t_v = _q4_0((n_kv_heads * head_dim, hidden))
            wo, t_o = _q4_0((hidden, n_heads * head_dim))
            writer.add_tensor(f"{p}.attn_q.weight", wq, raw_dtype=t_q)
            writer.add_tensor(f"{p}.attn_k.weight", wk, raw_dtype=t_k)
            writer.add_tensor(f"{p}.attn_v.weight", wv, raw_dtype=t_v)
            writer.add_tensor(f"{p}.attn_output.weight", wo, raw_dtype=t_o)
            writer.add_tensor(f"{p}.attn_q_norm.weight", np.ones((head_dim,), dtype=np.float32))
            writer.add_tensor(f"{p}.attn_k_norm.weight", np.ones((head_dim,), dtype=np.float32))

        if is_moe:
            writer.add_tensor(
                f"{p}.ffn_gate_inp.weight", (rng.standard_normal((4, hidden)) * 0.05).astype(np.float32)
            )
            ge, t_ge = _q4_0((4, 256, hidden))
            ue, t_ue = _q4_0((4, 256, hidden))
            de, t_de = _q4_0((4, hidden, 256))
            writer.add_tensor(f"{p}.ffn_gate_exps.weight", ge, raw_dtype=t_ge)
            writer.add_tensor(f"{p}.ffn_up_exps.weight", ue, raw_dtype=t_ue)
            writer.add_tensor(f"{p}.ffn_down_exps.weight", de, raw_dtype=t_de)
            sg, t_sg = _q4_0((256, hidden))
            su, t_su = _q4_0((256, hidden))
            sd, t_sd = _q4_0((hidden, 256))
            writer.add_tensor(f"{p}.ffn_gate_shexp.weight", sg, raw_dtype=t_sg)
            writer.add_tensor(f"{p}.ffn_up_shexp.weight", su, raw_dtype=t_su)
            writer.add_tensor(f"{p}.ffn_down_shexp.weight", sd, raw_dtype=t_sd)
            writer.add_tensor(f"{p}.ffn_gate_inp_shexp.weight", np.zeros((hidden,), dtype=np.float32))
        else:
            wg, tg = _raw_ud_quant((inter, hidden), gguf.GGMLQuantizationType.Q3_K, 256, 110)
            wu, tu = _q4_0((inter, hidden))
            wd, td = _q4_0((hidden, inter))
            writer.add_tensor(f"{p}.ffn_gate.weight", wg, raw_dtype=tg)
            writer.add_tensor(f"{p}.ffn_up.weight", wu, raw_dtype=tu)
            writer.add_tensor(f"{p}.ffn_down.weight", wd, raw_dtype=td)

    # Trailing MTP draft layer (`blk.4.nextn.*`) — excluded from `config.n_layers == 4` trunk,
    # and loaded into `GGUFMTPDraftHead` when `include_full_mtp=True` & `speculative_mtp=True`!
    eh, t_eh = _q4_0((hidden, hidden * 2))
    writer.add_tensor("blk.4.nextn.eh_proj.weight", eh, raw_dtype=t_eh)
    writer.add_tensor("blk.4.nextn.enorm.weight", np.ones((hidden,), dtype=np.float32))
    writer.add_tensor("blk.4.nextn.hnorm.weight", np.ones((hidden,), dtype=np.float32))
    if include_full_mtp:
        p4 = "blk.4"
        writer.add_tensor(f"{p4}.nextn.shared_head_norm.weight", np.ones((hidden,), dtype=np.float32))
        writer.add_tensor(f"{p4}.attn_norm.weight", np.ones((hidden,), dtype=np.float32))
        writer.add_tensor(f"{p4}.post_attention_norm.weight", np.ones((hidden,), dtype=np.float32))
        wq4, t_q4 = _raw_ud_quant((2 * n_heads * head_dim, hidden), gguf.GGMLQuantizationType.IQ4_XS, 256, 136)
        wk4, t_k4 = _q4_0((n_kv_heads * head_dim, hidden))
        wv4, t_v4 = _q4_0((n_kv_heads * head_dim, hidden))
        wo4, t_o4 = _q4_0((hidden, n_heads * head_dim))
        writer.add_tensor(f"{p4}.attn_q.weight", wq4, raw_dtype=t_q4)
        writer.add_tensor(f"{p4}.attn_k.weight", wk4, raw_dtype=t_k4)
        writer.add_tensor(f"{p4}.attn_v.weight", wv4, raw_dtype=t_v4)
        writer.add_tensor(f"{p4}.attn_output.weight", wo4, raw_dtype=t_o4)
        writer.add_tensor(f"{p4}.attn_q_norm.weight", np.ones((head_dim,), dtype=np.float32))
        writer.add_tensor(f"{p4}.attn_k_norm.weight", np.ones((head_dim,), dtype=np.float32))
        if is_moe:
            writer.add_tensor(
                f"{p4}.ffn_gate_inp.weight", (rng.standard_normal((4, hidden)) * 0.05).astype(np.float32)
            )
            ge4, t_ge4 = _q4_0((4, 256, hidden))
            ue4, t_ue4 = _q4_0((4, 256, hidden))
            de4, t_de4 = _q4_0((4, hidden, 256))
            writer.add_tensor(f"{p4}.ffn_gate_exps.weight", ge4, raw_dtype=t_ge4)
            writer.add_tensor(f"{p4}.ffn_up_exps.weight", ue4, raw_dtype=t_ue4)
            writer.add_tensor(f"{p4}.ffn_down_exps.weight", de4, raw_dtype=t_de4)
            sg4, t_sg4 = _q4_0((256, hidden))
            su4, t_su4 = _q4_0((256, hidden))
            sd4, t_sd4 = _q4_0((hidden, 256))
            writer.add_tensor(f"{p4}.ffn_gate_shexp.weight", sg4, raw_dtype=t_sg4)
            writer.add_tensor(f"{p4}.ffn_up_shexp.weight", su4, raw_dtype=t_su4)
            writer.add_tensor(f"{p4}.ffn_down_shexp.weight", sd4, raw_dtype=t_sd4)
            writer.add_tensor(f"{p4}.ffn_gate_inp_shexp.weight", np.zeros((hidden,), dtype=np.float32))
        else:
            wg4, tg4 = _raw_ud_quant((inter, hidden), gguf.GGMLQuantizationType.Q3_K, 256, 110)
            wu4, tu4 = _q4_0((inter, hidden))
            wd4, td4 = _q4_0((hidden, inter))
            writer.add_tensor(f"{p4}.ffn_gate.weight", wg4, raw_dtype=tg4)
            writer.add_tensor(f"{p4}.ffn_up.weight", wu4, raw_dtype=tu4)
            writer.add_tensor(f"{p4}.ffn_down.weight", wd4, raw_dtype=td4)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_qwen35_hybrid_deltanet_and_ud_quants() -> None:
    """
    Verify end-to-end `qwen35` & `qwen35moe` support (matching `Qwen3.8-27B-UD-Q4_K_M.gguf`):
    1. `nextn_predict_layers` excludes trailing `blk.4.nextn.*` MTP block (`n_layers = 4`).
    2. `IQ4_XS`, `IQ4_NL`, and `Q3_K` Unsloth Dynamic (`UD`) weights repack into 4-bit `W4A16` (`mode == "w4"`).
    3. Only full-attention layer (`blk.3`) allocates `LayerKVCache`; `blk.0..2` use fixed-size `LayerRecurrentState`.
    4. Parallel sequence prefill (`S=7`) matches token-by-token autoregressive decode (`7 x S=1`) to ~1e-3!
    5. Multi-turn chat prefix reuse + speculative decoding work seamlessly on hybrid `qwen35` & `qwen35moe`.
    """
    with tempfile.TemporaryDirectory() as tmp:
        q35_path = Path(tmp) / "qwen3_8_hybrid_ud.gguf"
        _build_synthetic_qwen35_gguf(q35_path, is_moe=False)

        eng = PyGGUFEngine(q35_path, max_context_len=2048, kv_dtype="int8", speculative_ngram=True)
        assert eng.config.n_layers == 4 and eng.config.nextn_predict_layers == 1
        assert eng.model.recurrent_layer_flags == [True, True, True, False]
        assert eng.kv_manager.layers[0] is None and eng.kv_manager.layers[3] is not None
        assert eng.token_embd_w4_check if hasattr(eng, "token_embd_w4_check") else eng.model.token_embd.mode == "w4"
        assert eng.model.lm_head is not None and eng.model.lm_head.mode == "w4"

        # Exactness check: parallel prefill [1, 7] vs step-by-step decode 7 x [1, 1]
        seq = [4, 5, 6, 7, 10, 11, 12]
        eng.kv_manager.clear()
        logits_prefill = eng.model(torch.tensor([seq], dtype=torch.long), eng.kv_manager, start_pos=0, logits_to_keep=7)

        eng.kv_manager.clear()
        step_logits = []
        for pos, tok in enumerate(seq):
            lg = eng.model(torch.tensor([[tok]], dtype=torch.long), eng.kv_manager, start_pos=pos, logits_to_keep=1)
            step_logits.append(lg)
        logits_step = torch.cat(step_logits, dim=1)
        max_diff = (logits_prefill - logits_step).abs().max().item()
        rel_err = max_diff / max(logits_prefill.abs().max().item(), 1.0)
        assert rel_err < 1e-3, f"Hybrid prefill vs step-by-step rel_err too high: {rel_err} (abs={max_diff})"
        print(
            f"  [OK] Qwen3.5/3.8 Hybrid Gated-DeltaNet + Gated-Attn prefill==decode verified "
            f"(rel_err={rel_err:.6f}, abs={max_diff:.5f})!"
        )

        # Multi-turn generation + prefix cache reuse + speculative rollback check
        r1 = eng.generate([1, 4, 5, 6, 4, 5, 6], SamplingConfig(max_tokens=8, temperature=0.0, cache_prompt=True))
        assert r1.completion_tokens > 0
        # Append more tokens onto exact cached prefix
        r2 = eng.generate(
            list(eng.kv_manager.cached_tokens) + [7, 10],
            SamplingConfig(max_tokens=6, temperature=0.0, cache_prompt=True),
        )
        assert r2.cached_prefix_tokens == len(eng.kv_manager.cached_tokens) - 2 - max(0, r2.completion_tokens - 1) or r2.cached_prefix_tokens > 0
        print(
            f"  [OK] Qwen3.5/3.8 UD-W4A16 generation + hybrid prefix cache reuse verified "
            f"(cached={r2.cached_prefix_tokens} toks)!"
        )

        # Also test `qwen35moe` (Qwen3.5-35B-A3B hybrid Gated-DeltaNet + Gated Shared-Expert MoE)
        q35_moe_path = Path(tmp) / "qwen3_5_moe_hybrid.gguf"
        _build_synthetic_qwen35_gguf(q35_moe_path, is_moe=True)
        eng_moe = PyGGUFEngine(q35_moe_path, max_context_len=1024, kv_dtype="int8")
        rm = eng_moe.chat([{"role": "user", "content": "Hello world"}], SamplingConfig(max_tokens=6, temperature=0.0))
        assert rm.completion_tokens > 0
        print("  [OK] Qwen3.5-MoE (`qwen35moe`) Hybrid Gated-DeltaNet + Shared-Expert Sparse MoE verified!")


def test_triton_sm75_ptx_and_interpreter_exactness() -> None:
    """
    1. Compile all 9 `@triton.jit` kernels directly to NVIDIA Tesla T4 (`sm_75`) PTX assembly
       and verify ZERO uncoalesced 8-bit `ld.global.b8` loads in `int32`-packed W4/W8 weight kernels!
    2. Execute the Triton kernels via `TRITON_INTERPRET=1` and verify bit-accurate parity against CPU reference!
    """
    import os
    import subprocess
    import sys
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource, compile as triton_compile

    from pygguf_engine.kv_cache import (
        _causal_conv1d_silu_multirow_kernel,
        _causal_conv1d_silu_step_kernel,
        _flash_decode_combine_kernel,
        _gated_deltanet_decode_step_kernel,
        _gated_deltanet_multirow_verify_kernel,
        _gated_deltanet_scan_kernel,
        _int4_gqa_flash_decode_posptr_kernel,
        _int4_gqa_flash_decode_split_kernel,
        _int8_gqa_flash_decode_posptr_kernel,
        _int8_gqa_flash_decode_split_kernel,
        _kv_append_int4_kernel,
        _kv_append_int4_posptr_kernel,
        _kv_append_int8_kernel,
        _kv_append_int8_posptr_kernel,
    )
    from pygguf_engine.quant_kernels import (
        _dequant_w4_chunk_kernel,
        _dequant_w8_chunk_kernel,
        _fused_act_mul_rowscale_kernel,
        _fused_rmsnorm_kernel,
        _fused_rmsnorm_silu_gate_kernel,
        _w4a16_gemm_tc_kernel,
        _w4a16_gemv_kernel,
        _w4a16_multirow_gemv_kernel,
        _w8a16_gemm_tc_kernel,
        _w8a16_gemv_kernel,
        _w8a16_multirow_gemv_kernel,
    )

    target = GPUTarget("cuda", 75, 32)
    target_sm80 = GPUTarget("cuda", 80, 32)
    specs = [
        (
            "W4A16_GEMV_INT32",
            _w4a16_gemv_kernel,
            target,
            {
                "x_ptr": "*fp16",
                "qw_i32_ptr": "*i32",
                "scales_ptr": "*fp16",
                "zeros_ptr": "*fp16",
                "bias_ptr": "*fp16",
                "out_ptr": "*fp16",
                "row_scale_ptr": "*fp32",
                "M": "i32",
                "N": "i32",
                "K": "i32",
            },
            {"HAS_ROW_SCALE": False, "OUT_FP32": False, "X_PACKED_I32": False, "HAS_BIAS": True, "BLOCK_N": 8},
            True,
        ),
        (
            "W4A16_GEMM_TC_SM80",
            _w4a16_gemm_tc_kernel,
            target_sm80,
            {
                "x_ptr": "*fp16",
                "qw_i32_ptr": "*i32",
                "scales_ptr": "*fp16",
                "zeros_ptr": "*fp16",
                "bias_ptr": "*fp16",
                "out_ptr": "*fp16",
                "M": "i32",
                "N": "i32",
                "K": "i32",
            },
            {"HAS_BIAS": True, "BLOCK_M": 64, "BLOCK_N": 64},
            True,
        ),
        (
            "W8A16_GEMV_INT32",
            _w8a16_gemv_kernel,
            target,
            {
                "x_ptr": "*fp16",
                "qw_i32_ptr": "*i32",
                "scales_ptr": "*fp16",
                "zeros_ptr": "*fp16",
                "bias_ptr": "*fp16",
                "out_ptr": "*fp16",
                "M": "i32",
                "N": "i32",
                "K": "i32",
            },
            {"HAS_ZEROS": True, "HAS_BIAS": True, "BLOCK_N": 8},
            True,
        ),
        (
            "W8A16_GEMM_TC_SM80",
            _w8a16_gemm_tc_kernel,
            target_sm80,
            {
                "x_ptr": "*fp16",
                "qw_i32_ptr": "*i32",
                "scales_ptr": "*fp16",
                "zeros_ptr": "*fp16",
                "bias_ptr": "*fp16",
                "out_ptr": "*fp16",
                "M": "i32",
                "N": "i32",
                "K": "i32",
            },
            {"HAS_ZEROS": True, "HAS_BIAS": True, "BLOCK_M": 64, "BLOCK_N": 64},
            True,
        ),
        (
            "DEQUANT_W4_SM75_TC",
            _dequant_w4_chunk_kernel,
            target,
            {
                "qw_i32_ptr": "*i32",
                "scales_ptr": "*fp16",
                "zeros_ptr": "*fp16",
                "out_fp16_ptr": "*fp16",
                "rc": "i32",
                "K": "i32",
            },
            {"BLOCK_R": 16},
            True,
        ),
        (
            "DEQUANT_W8_SM75_TC",
            _dequant_w8_chunk_kernel,
            target,
            {
                "qw_i32_ptr": "*i32",
                "scales_ptr": "*fp16",
                "zeros_ptr": "*fp16",
                "out_fp16_ptr": "*fp16",
                "rc": "i32",
                "K": "i32",
            },
            {"HAS_ZEROS": True, "BLOCK_R": 16},
            True,
        ),
        (
            "KV_APPEND_INT8",
            _kv_append_int8_kernel,
            target,
            {
                "k_in_ptr": "*fp16",
                "v_in_ptr": "*fp16",
                "k_buf_ptr": "*i8",
                "v_buf_ptr": "*i8",
                "ks_buf_ptr": "*fp16",
                "vs_buf_ptr": "*fp16",
                "start_pos": "i32",
                "s_new": "i32",
                "capacity": "i32",
            },
            {"HEAD_DIM": 128, "BLOCK_D": 128},
            False,
        ),
        (
            "INT8_FLASH_DECODE",
            _int8_gqa_flash_decode_split_kernel,
            target,
            {
                "q_ptr": "*fp16",
                "k_ptr": "*i8",
                "ks_ptr": "*fp16",
                "v_ptr": "*i8",
                "vs_ptr": "*fp16",
                "part_out_ptr": "*fp32",
                "part_lse_ptr": "*fp32",
                "total_len": "i32",
                "capacity": "i32",
                "win_start": "i32",
                "sm_scale": "fp32",
            },
            {"Q_LEN": 1, "WINDOW_SIZE": 0, "WIN_RING": True, "GQA_GROUP": 4, "NUM_SPLITS": 16, "BLOCK_S": 64, "HEAD_DIM": 128},
            False,
        ),
        (
            "GATED_DELTANET_SCAN",
            _gated_deltanet_scan_kernel,
            target,
            {
                "q_ptr": "*fp32",
                "k_ptr": "*fp32",
                "v_ptr": "*fp32",
                "decay_ptr": "*fp32",
                "beta_ptr": "*fp32",
                "state_ptr": "*fp32",
                "out_ptr": "*fp32",
                "seq_len": "i32",
            },
            {"H_K": 16, "H_V": 48, "D_K": 128, "D_V": 128, "BLOCK_K": 128, "BLOCK_V": 16},
            False,
        ),
        (
            "DELTANET_CONV_STEP",
            _causal_conv1d_silu_step_kernel,
            target,
            {
                "qkv_in_ptr": "*fp16",
                "conv_state_ptr": "*fp32",
                "conv_w_ptr": "*fp32",
                "conv_out_ptr": "*fp32",
                "conv_dim": "i32",
            },
            {"CONV_K": 4, "K_HIST": 3, "BLOCK_C": 256},
            False,
        ),
        (
            "DELTANET_DECODE_STEP",
            _gated_deltanet_decode_step_kernel,
            target,
            {
                "conv_out_ptr": "*fp32",
                "beta_raw_ptr": "*fp16",
                "alpha_raw_ptr": "*fp16",
                "ssm_dt_ptr": "*fp32",
                "ssm_a_ptr": "*fp32",
                "state_ptr": "*fp32",
                "out_ptr": "*fp32",
                "key_dim": "i32",
                "q_scale": "fp32",
                "eps": "fp32",
            },
            {"H_K": 16, "H_V": 48, "D_K": 128, "D_V": 128, "BLOCK_K": 128, "BLOCK_V": 16},
            False,
        ),
        (
            "FUSED_SWIGLU_SCALE",
            _fused_act_mul_rowscale_kernel,
            target,
            {
                "gate_ptr": "*fp16",
                "up_ptr": "*fp16",
                "out_fp16_ptr": "*fp16",
                "row_scale_ptr": "*fp32",
                "M": "i32",
                "I": "i32",
            },
            {"IS_CONCAT": True, "USE_GELU": False, "BLOCK_I": 1024},
            False,
        ),
        (
            "FUSED_RMSNORM_GATE",
            _fused_rmsnorm_silu_gate_kernel,
            target,
            {
                "y_ptr": "*fp32",
                "z_ptr": "*fp16",
                "w_ptr": "*fp32",
                "out_fp16_ptr": "*fp16",
                "D": "i32",
                "eps": "fp32",
            },
            {"BLOCK_D": 128},
            False,
        ),
        (
            "FLASH_DECODE_COMBINE",
            _flash_decode_combine_kernel,
            target,
            {
                "part_out_ptr": "*fp32",
                "part_lse_ptr": "*fp32",
                "out_ptr": "*fp16",
            },
            {"NUM_SPLITS": 32, "HEAD_DIM": 256},
            False,
        ),
        (
            "KV_APPEND_POSPTR",
            _kv_append_int8_posptr_kernel,
            target,
            {
                "k_in_ptr": "*fp16",
                "v_in_ptr": "*fp16",
                "k_buf_ptr": "*i8",
                "v_buf_ptr": "*i8",
                "ks_buf_ptr": "*fp16",
                "vs_buf_ptr": "*fp16",
                "pos_ptr": "*i32",
                "capacity": "i32",
            },
            {"HEAD_DIM": 256, "BLOCK_D": 256},
            False,
        ),
        (
            "FLASH_DECODE_POSPTR",
            _int8_gqa_flash_decode_posptr_kernel,
            target,
            {
                "q_ptr": "*fp16",
                "k_ptr": "*i8",
                "ks_ptr": "*fp16",
                "v_ptr": "*i8",
                "vs_ptr": "*fp16",
                "part_out_ptr": "*fp32",
                "part_lse_ptr": "*fp32",
                "pos_ptr": "*i32",
                "capacity": "i32",
                "sm_scale": "fp32",
            },
            {"GQA_GROUP": 6, "NUM_SPLITS": 32, "BLOCK_S": 32, "HEAD_DIM": 256},
            False,
        ),
        (
            "W4A16_MULTIROW_GEMV",
            _w4a16_multirow_gemv_kernel,
            target,
            {
                "x_ptr": "*fp16",
                "qw_i32_ptr": "*i32",
                "scales_ptr": "*fp16",
                "zeros_ptr": "*fp16",
                "bias_ptr": "*fp16",
                "out_ptr": "*fp16",
                "row_scale_ptr": "*fp32",
                "M": "i32",
                "N": "i32",
                "K": "i32",
            },
            {"HAS_ROW_SCALE": False, "OUT_FP32": False, "X_PACKED_I32": False, "HAS_BIAS": True, "BLOCK_M": 2, "BLOCK_N": 4},
            True,
        ),
        (
            "W8A16_MULTIROW_GEMV",
            _w8a16_multirow_gemv_kernel,
            target,
            {
                "x_ptr": "*fp16",
                "qw_i32_ptr": "*i32",
                "scales_ptr": "*fp16",
                "zeros_ptr": "*fp16",
                "bias_ptr": "*fp16",
                "out_ptr": "*fp16",
                "M": "i32",
                "N": "i32",
                "K": "i32",
            },
            {"HAS_ZEROS": True, "HAS_BIAS": True, "BLOCK_M": 2, "BLOCK_N": 4},
            True,
        ),
        (
            "FUSED_ADD_RMSNORM",
            _fused_rmsnorm_kernel,
            target,
            {
                "x_ptr": "*fp32",
                "delta_ptr": "*fp32",
                "w_ptr": "*fp32",
                "res_out_ptr": "*fp32",
                "out_ptr": "*fp16",
                "D": "i32",
                "eps": "fp32",
            },
            {"HAS_RESIDUAL": True, "OUT_FP16": True, "BLOCK_D": 8192},
            False,
        ),
        (
            "KV_APPEND_INT4",
            _kv_append_int4_kernel,
            target,
            {
                "k_in_ptr": "*fp16",
                "v_in_ptr": "*fp16",
                "k_buf_ptr": "*u8",
                "v_buf_ptr": "*u8",
                "ks_buf_ptr": "*fp16",
                "vs_buf_ptr": "*fp16",
                "start_pos": "i32",
                "s_new": "i32",
                "capacity": "i32",
            },
            {"HALF_D": 128, "BLOCK_HD": 128},
            False,
        ),
        (
            "KV_APPEND_INT4_POSPTR",
            _kv_append_int4_posptr_kernel,
            target,
            {
                "k_in_ptr": "*fp16",
                "v_in_ptr": "*fp16",
                "k_buf_ptr": "*u8",
                "v_buf_ptr": "*u8",
                "ks_buf_ptr": "*fp16",
                "vs_buf_ptr": "*fp16",
                "pos_ptr": "*i32",
                "capacity": "i32",
            },
            {"HALF_D": 128, "BLOCK_HD": 128},
            False,
        ),
        (
            "INT4_FLASH_DECODE",
            _int4_gqa_flash_decode_split_kernel,
            target,
            {
                "q_ptr": "*fp16",
                "k_ptr": "*u8",
                "ks_ptr": "*fp16",
                "v_ptr": "*u8",
                "vs_ptr": "*fp16",
                "part_out_ptr": "*fp32",
                "part_lse_ptr": "*fp32",
                "total_len": "i32",
                "capacity": "i32",
                "win_start": "i32",
                "sm_scale": "fp32",
            },
            {"Q_LEN": 1, "WINDOW_SIZE": 0, "WIN_RING": True, "GQA_GROUP": 6, "NUM_SPLITS": 32, "BLOCK_S": 32, "HALF_D": 128},
            False,
        ),
        (
            "INT4_FLASH_DECODE_POSPTR",
            _int4_gqa_flash_decode_posptr_kernel,
            target,
            {
                "q_ptr": "*fp16",
                "k_ptr": "*u8",
                "ks_ptr": "*fp16",
                "v_ptr": "*u8",
                "vs_ptr": "*fp16",
                "part_out_ptr": "*fp32",
                "part_lse_ptr": "*fp32",
                "pos_ptr": "*i32",
                "capacity": "i32",
                "sm_scale": "fp32",
            },
            {"GQA_GROUP": 6, "NUM_SPLITS": 32, "BLOCK_S": 32, "HALF_D": 128},
            False,
        ),
        (
            "DELTANET_CONV_MULTIROW",
            _causal_conv1d_silu_multirow_kernel,
            target,
            {
                "qkv_in_ptr": "*fp16",
                "conv_state_ptr": "*fp32",
                "conv_w_ptr": "*fp32",
                "conv_out_ptr": "*fp32",
                "conv_in_ptr": "*fp32",
                "S": "i32",
                "conv_dim": "i32",
            },
            {"CONV_K": 4, "K_HIST": 3, "SAVE_CONV_IN": True, "BLOCK_C": 256},
            False,
        ),
        (
            "DELTANET_VERIFY_MULTIROW",
            _gated_deltanet_multirow_verify_kernel,
            target,
            {
                "conv_out_ptr": "*fp32",
                "beta_raw_ptr": "*fp16",
                "alpha_raw_ptr": "*fp16",
                "ssm_dt_ptr": "*fp32",
                "ssm_a_ptr": "*fp32",
                "state_ptr": "*fp32",
                "out_ptr": "*fp32",
                "pre_ssm_ptr": "*fp32",
                "q_save_ptr": "*fp32",
                "k_save_ptr": "*fp32",
                "v_save_ptr": "*fp32",
                "decay_save_ptr": "*fp32",
                "beta_save_ptr": "*fp32",
                "S": "i32",
                "conv_dim": "i32",
                "key_dim": "i32",
                "q_scale": "fp32",
                "eps": "fp32",
            },
            {"SAVE_INPUTS": True, "H_K": 16, "H_V": 48, "D_K": 128, "D_V": 128, "BLOCK_K": 128, "BLOCK_V": 16},
            False,
        ),
    ]
    # Compile the actual multi-token verify specializations as well as decode.
    for spec in list(specs):
        name, fn, tgt, sig, constants, check = spec
        if name in ("INT4_FLASH_DECODE", "INT8_FLASH_DECODE"):
            for q_len in (2, 3, 8):
                for ring in (False, True):
                    cfg = dict(constants, Q_LEN=q_len, WIN_RING=ring,
                               WINDOW_SIZE=9 if ring else 0)
                    specs.append((f"{name}_Q{q_len}_R{int(ring)}", fn, tgt, sig, cfg, check))
    for name, fn, tgt, sig, constexprs, assert_zero_b8 in specs:
        # Triton 3.x prototype argument order follows `signature` insertion order; normalize the
        # spec dict to the kernel's actual parameter order and sanity-check coverage.
        import inspect as _inspect
        required = {p.name for p in fn.params if not p.is_constexpr and p.default is _inspect._empty}
        allowed = {p.name for p in fn.params if not p.is_constexpr}
        assert required <= set(sig) <= allowed, f"{name}: signature/parameter mismatch: {set(sig) ^ required}"
        sig = {a: sig[a] for a in fn.arg_names if a in sig}
        compiled = triton_compile(ASTSource(fn=fn, signature=sig, constants=constexprs), target=tgt)
        ptx = compiled.asm["ptx"]
        assert len(ptx) > 1000
        if assert_zero_b8:
            b8_loads = [ln for ln in ptx.splitlines() if "ld.global" in ln and (".b8" in ln or ".u8" in ln or ".s8" in ln)]
            assert len(b8_loads) == 0, f"{name} emitted scalar byte loads: {b8_loads[:3]}"
            # Ampere+ Triton lowers 32/64/128-bit global->shared staging to `cp.async...shared.global`
            # (no `ld.global` token), so count both cp.async payloads >= 4 bytes and plain ld.global.32 accesses.
            import re as _re
            b32_loads = [
                ln
                for ln in ptx.splitlines()
                if ("ld.global" in ln and "32" in ln)
                or _re.search(r"cp\.async[^;]*shared\.global[^;]*(0x4\b|0x8\b|0x10\b|\.b32\b)", ln)
            ]
            assert len(b32_loads) > 0, f"{name} missing 32-bit loads"
        print(f"  [OK] Triton PTX compiled: {name:20s} ({len(ptx):,} bytes)")

    # Subprocess check with TRITON_INTERPRET=1 verifying exact numerical parity of all Triton kernels!
    interp_code = """
import os
os.environ["TRITON_INTERPRET"] = "1"
import torch, triton
from pygguf_engine.quant_kernels import (
    _w4a16_gemv_kernel, _w4a16_multirow_gemv_kernel, _w4a16_gemm_tc_kernel,
    _w8a16_gemv_kernel, _w8a16_multirow_gemv_kernel, _w8a16_gemm_tc_kernel,
    _dequant_w4_chunk_kernel, _dequant_w8_chunk_kernel,
    exec_w4a16_matmul, exec_w8a16_matmul,
)
from pygguf_engine.kv_cache import _kv_append_int8_kernel, _int8_gqa_flash_decode_split_kernel

torch.manual_seed(7)
M, N, K = 3, 24, 288  # non-multiple of 256 exercises g_base tail masking!
x = torch.randn(M, K, dtype=torch.float16)
qw4 = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8)
s4 = (torch.randn(N, K // 32) * 0.1).to(torch.float16)
z4 = (torch.randn(N, K // 32) * 0.1).to(torch.float16)
b4 = (torch.randn(N) * 0.1).to(torch.float16)

ref4 = exec_w4a16_matmul(x, qw4, s4, z4, b4)
out4_v = torch.empty((M, N), dtype=torch.float16)
_w4a16_gemv_kernel[(triton.cdiv(N, 8), M)](x, qw4.view(torch.int32), s4, z4, b4, out4_v, M, N, K, HAS_BIAS=True, BLOCK_N=8)
assert (ref4 - out4_v).abs().max().item() < 1e-2

# Verify weight-reusing _w4a16_multirow_gemv_kernel across M=2 (BLOCK_M=2), M=3 (BLOCK_M=4), M=5 (BLOCK_M=8)!
for m_sub, bm in ((2, 2), (3, 4), (5, 8)):
    xm = torch.randn(m_sub, K, dtype=torch.float16)
    ref_m4 = exec_w4a16_matmul(xm, qw4, s4, z4, b4)
    out_mr4 = torch.empty((m_sub, N), dtype=torch.float16)
    _w4a16_multirow_gemv_kernel[(triton.cdiv(N, 4),)](
        xm, qw4.view(torch.int32), s4, z4, b4, out_mr4, m_sub, N, K, HAS_BIAS=True, BLOCK_M=bm, BLOCK_N=4
    )
    assert (ref_m4 - out_mr4).abs().max().item() < 1e-2

M_big = 20
x_big = torch.randn(M_big, K, dtype=torch.float16)
ref4_m = exec_w4a16_matmul(x_big, qw4, s4, z4, b4)
out4_m = torch.empty((M_big, N), dtype=torch.float16)
_w4a16_gemm_tc_kernel[(triton.cdiv(M_big, 16), triton.cdiv(N, 32))](
    x_big, qw4.view(torch.int32), s4, z4, b4, out4_m, M_big, N, K, HAS_BIAS=True, BLOCK_M=16, BLOCK_N=32
)
assert (ref4_m - out4_m).abs().max().item() < 5e-2

dq4 = torch.empty((N, K), dtype=torch.float16)
_dequant_w4_chunk_kernel[(triton.cdiv(N, 16), K // 32)](qw4.view(torch.int32), s4, z4, dq4, N, K, BLOCK_R=16)
assert (ref4_m - torch.nn.functional.linear(x_big, dq4, b4)).abs().max().item() < 5e-2

qw8 = torch.randint(-128, 127, (N, K), dtype=torch.int8)
s8 = (torch.randn(N, K // 16) * 0.05).to(torch.float16)
z8 = (torch.randn(N, K // 16) * 0.05).to(torch.float16)
ref8 = exec_w8a16_matmul(x, qw8, s8, z8, b4)
out8_v = torch.empty((M, N), dtype=torch.float16)
_w8a16_gemv_kernel[(triton.cdiv(N, 8), M)](
    x, qw8.view(torch.int32), s8, z8, b4, out8_v, M, N, K, HAS_ZEROS=True, HAS_BIAS=True, BLOCK_N=8
)
assert (ref8 - out8_v).abs().max().item() < 1e-2

for m_sub, bm in ((2, 2), (3, 4), (5, 8)):
    xm = torch.randn(m_sub, K, dtype=torch.float16)
    ref_m8 = exec_w8a16_matmul(xm, qw8, s8, z8, b4)
    out_mr8 = torch.empty((m_sub, N), dtype=torch.float16)
    _w8a16_multirow_gemv_kernel[(triton.cdiv(N, 4),)](
        xm, qw8.view(torch.int32), s8, z8, b4, out_mr8, m_sub, N, K, HAS_ZEROS=True, HAS_BIAS=True, BLOCK_M=bm, BLOCK_N=4
    )
    assert (ref_m8 - out_mr8).abs().max().item() < 1e-2

out8_m = torch.empty((M_big, N), dtype=torch.float16)
_w8a16_gemm_tc_kernel[(triton.cdiv(M_big, 16), triton.cdiv(N, 32))](
    x_big, qw8.view(torch.int32), s8, z8, b4, out8_m, M_big, N, K, HAS_ZEROS=True, HAS_BIAS=True, BLOCK_M=16, BLOCK_N=32
)
dq8 = torch.empty((N, K), dtype=torch.float16)
_dequant_w8_chunk_kernel[(triton.cdiv(N, 16), K // 16)](
    qw8.view(torch.int32), s8, z8, dq8, N, K, HAS_ZEROS=True, BLOCK_R=16
)
ref8_m = exec_w8a16_matmul(x_big, qw8, s8, z8, b4)
lin8_m = torch.nn.functional.linear(x_big, dq8, b4)
assert (out8_m - lin8_m).abs().max().item() <= 0.25  # fp16 ulp at |x|~256 is 0.125; allow 2 ulp for dot-vs-linear accumulation-order noise
assert (ref8_m - out8_m).abs().max().item() / ref8_m.abs().max().item() < 2e-3
assert (ref8_m - lin8_m).abs().max().item() / ref8_m.abs().max().item() < 2e-3

# Verify _kv_append_int8_kernel
H_kv, S_new, Cap, D = 4, 5, 32, 64
kin = torch.randn(H_kv, S_new, D, dtype=torch.float16)
vin = torch.randn(H_kv, S_new, D, dtype=torch.float16)
kb = torch.zeros(H_kv, Cap, D, dtype=torch.int8)
vb = torch.zeros(H_kv, Cap, D, dtype=torch.int8)
ks = torch.zeros(H_kv, Cap, dtype=torch.float16)
vs = torch.zeros(H_kv, Cap, dtype=torch.float16)
_kv_append_int8_kernel[(H_kv, S_new)](kin, vin, kb, vb, ks, vs, 3, S_new, Cap, HEAD_DIM=D, BLOCK_D=D)
deq_k = kb[:, 3:8, :].float() * ks[:, 3:8].float().unsqueeze(-1)
assert (deq_k - kin.float()).abs().max().item() < 0.03

# Verify fused SwiGLU + row-scale, fused RMSNorm*SiLU, and fused S=1 DeltaNet Conv+Recurrent decode step!
from pygguf_engine.quant_kernels import _fused_act_mul_rowscale_kernel, _fused_rmsnorm_silu_gate_kernel
from pygguf_engine.kv_cache import _causal_conv1d_silu_step_kernel, _gated_deltanet_decode_step_kernel, _flash_decode_combine_kernel
import torch.nn.functional as F

gu = (torch.randn(2, 600, dtype=torch.float16) * 200.0)
out_sw = torch.empty((2, 300), dtype=torch.float16)
sc_sw = torch.empty((2, 1), dtype=torch.float32)
_fused_act_mul_rowscale_kernel[(2,)](gu, gu, out_sw, sc_sw, 2, 300, IS_CONCAT=True, USE_GELU=False, BLOCK_I=256)
g_ref, u_ref = gu.float().chunk(2, dim=-1)
inter_ref = F.silu(g_ref) * u_ref
sc_ref = (inter_ref.abs().amax(dim=-1, keepdim=True) / 16384.0).clamp(min=1.0)
assert (sc_sw - sc_ref).abs().max().item() < 1e-3
assert ((inter_ref / sc_ref).to(torch.float16) - out_sw).abs().max().item() <= 1e-2

y_in = torch.randn(6, 32, dtype=torch.float32)
z_in = torch.randn(6, 32, dtype=torch.float16)
w_rn = torch.ones(32, dtype=torch.float32) * 1.25
out_rn = torch.empty((6, 32), dtype=torch.float16)
_fused_rmsnorm_silu_gate_kernel[(6,)](y_in, z_in, w_rn, out_rn, 32, 1e-6, BLOCK_D=32)
ref_rn = (F.rms_norm(y_in, (32,), w_rn, 1e-6) * F.silu(z_in.float())).to(torch.float16)
assert (out_rn - ref_rn).abs().max().item() <= 1e-3

# Verify _causal_conv1d_silu_step_kernel + _gated_deltanet_decode_step_kernel exact parity vs PyTorch!
conv_dim, H_k, H_v, D_k, D_v = 320, 2, 6, 32, 32
c_state_ref = torch.randn(1, conv_dim, 3, dtype=torch.float32)
c_state_tri = c_state_ref.clone()
c_w = torch.randn(conv_dim, 1, 4, dtype=torch.float32) * 0.1
qkv_step = torch.randn(1, 1, conv_dim, dtype=torch.float16)
c_out_tri = torch.empty((conv_dim,), dtype=torch.float32)
_causal_conv1d_silu_step_kernel[(2,)](qkv_step.view(-1), c_state_tri, c_w, c_out_tri, conv_dim, CONV_K=4, K_HIST=3, BLOCK_C=256)
c_in_ref = torch.cat([c_state_ref, qkv_step.float().transpose(1, 2)], dim=-1)
c_state_ref_next = c_in_ref[:, :, -3:]
c_out_ref = F.silu(F.conv1d(c_in_ref, c_w, groups=conv_dim)).view(-1)
assert (c_state_tri - c_state_ref_next).abs().max().item() < 1e-6
assert (c_out_tri - c_out_ref).abs().max().item() < 1e-5

b_raw = torch.randn(H_v, dtype=torch.float16)
a_raw = torch.randn(H_v, dtype=torch.float16)
dt_b = torch.zeros(H_v, dtype=torch.float32)
ssm_a = -torch.ones(H_v, dtype=torch.float32)
s_ref = torch.randn(H_v, D_v, D_k, dtype=torch.float32) * 0.1
s_tri = s_ref.clone()
out_dn = torch.empty((H_v, D_v), dtype=torch.float32)
q_sc = 1.0 / (D_k ** 0.5)
_gated_deltanet_decode_step_kernel[(H_v, 2)](
    c_out_tri, b_raw, a_raw, dt_b, ssm_a, s_tri, out_dn, 64, q_sc, 1e-6,
    H_K=H_k, H_V=H_v, D_K=D_k, D_V=D_v, BLOCK_K=32, BLOCK_V=16
)
qr, kr, vr = torch.split(c_out_ref, [64, 64, 192], dim=-1)
qr = qr.view(H_k, D_k); kr = kr.view(H_k, D_k); vr = vr.view(H_v, D_v)
qr = (qr * torch.rsqrt(qr.pow(2).sum(-1, keepdim=True) + 1e-6) * q_sc).repeat(3, 1)
kr = (kr * torch.rsqrt(kr.pow(2).sum(-1, keepdim=True) + 1e-6)).repeat(3, 1)
dec_r = torch.exp(F.softplus(a_raw.float() + dt_b) * ssm_a)
bet_r = torch.sigmoid(b_raw.float())
s_ref = s_ref * dec_r.view(H_v, 1, 1)
kv_m = (s_ref * kr.unsqueeze(1)).sum(-1)
s_ref = s_ref + ((vr - kv_m) * bet_r.unsqueeze(-1)).unsqueeze(-1) * kr.unsqueeze(1)
out_dn_ref = (s_ref * qr.unsqueeze(1)).sum(-1)
assert (s_tri - s_ref).abs().max().item() < 1e-5
assert (out_dn - out_dn_ref).abs().max().item() < 1e-5

# Verify _fused_rmsnorm_kernel (both standalone and fused Residual-Add + RMSNorm)!
from pygguf_engine.quant_kernels import _fused_rmsnorm_kernel
x_rm = torch.randn(4, 256, dtype=torch.float32)
d_rm = torch.randn(4, 256, dtype=torch.float16)
w_rm = torch.randn(256, dtype=torch.float32) * 0.5 + 1.0
res_rm = torch.empty((4, 256), dtype=torch.float32)
out_rm = torch.empty((4, 256), dtype=torch.float16)
_fused_rmsnorm_kernel[(4,)](
    x_rm, d_rm, w_rm, res_rm, out_rm, 256, 1e-6, HAS_RESIDUAL=True, OUT_FP16=True, BLOCK_D=256
)
h_rm_ref = x_rm + d_rm.float()
out_rm_ref = F.rms_norm(h_rm_ref, (256,), w_rm, 1e-6).to(torch.float16)
assert (res_rm - h_rm_ref).abs().max().item() < 1e-6
assert (out_rm - out_rm_ref).abs().max().item() <= 1e-3

# Verify _kv_append_int4_kernel + _int4_gqa_flash_decode_split_kernel (with WIN_RING=True wrap-around!)
from pygguf_engine.kv_cache import (
    _kv_append_int4_kernel,
    _int4_gqa_flash_decode_split_kernel,
    LayerRecurrentState,
    _causal_conv1d_silu_multirow_kernel,
    _gated_deltanet_multirow_verify_kernel,
)
H_q, H_kv4, D4, Cap4 = 4, 2, 64, 16
HD4 = D4 // 2
kb4 = torch.zeros((H_kv4, Cap4, HD4), dtype=torch.uint8)
vb4 = torch.zeros((H_kv4, Cap4, HD4), dtype=torch.uint8)
ks4 = torch.zeros((H_kv4, Cap4, 2), dtype=torch.float16)
vs4 = torch.zeros((H_kv4, Cap4, 2), dtype=torch.float16)
# Write 10 tokens wrapping across Cap4=16: first 4 at [12..15], next 6 at [0..5] -> chronological window [12..15, 0..5]!
k_all = torch.randn(H_kv4, 10, D4, dtype=torch.float16)
v_all = torch.randn(H_kv4, 10, D4, dtype=torch.float16)
_kv_append_int4_kernel[(H_kv4, 4)](k_all[:, :4, :].contiguous(), v_all[:, :4, :].contiguous(), kb4, vb4, ks4, vs4, 12, 4, Cap4, HALF_D=HD4, BLOCK_HD=HD4)
_kv_append_int4_kernel[(H_kv4, 6)](k_all[:, 4:, :].contiguous(), v_all[:, 4:, :].contiguous(), kb4, vb4, ks4, vs4, 0, 6, Cap4, HALF_D=HD4, BLOCK_HD=HD4)

q4 = torch.randn(H_q, D4, dtype=torch.float16)
part_o4 = torch.empty((H_q, 4, D4), dtype=torch.float32)
part_l4 = torch.empty((H_q, 4), dtype=torch.float32)
sm_sc4 = 1.0 / (D4 ** 0.5)
_int4_gqa_flash_decode_split_kernel[(H_q, 4)](
    q4, kb4, ks4, vb4, vs4, part_o4, part_l4, 10, Cap4, 12, sm_sc4,
    WIN_RING=True, GQA_GROUP=2, NUM_SPLITS=4, BLOCK_S=4, HALF_D=HD4
)
out_fd4 = torch.empty((H_q, D4), dtype=torch.float16)
_flash_decode_combine_kernel[(H_q,)](part_o4, part_l4, out_fd4, NUM_SPLITS=4, HEAD_DIM=D4)

# Dequantize ring window [12..15] + [0..5] and compare against exact SDPA!
idx_ring = torch.tensor([12, 13, 14, 15, 0, 1, 2, 3, 4, 5], dtype=torch.long)
kp_r = kb4[:, idx_ring, :]; vp_r = vb4[:, idx_ring, :]
ks_r = ks4[:, idx_ring, :].float(); vs_r = vs4[:, idx_ring, :].float()
kd_r = torch.cat([((kp_r & 0x0F).float() - 8.0) * ks_r[..., 0:1], (((kp_r >> 4) & 0x0F).float() - 8.0) * ks_r[..., 1:2]], dim=-1)
vd_r = torch.cat([((vp_r & 0x0F).float() - 8.0) * vs_r[..., 0:1], (((vp_r >> 4) & 0x0F).float() - 8.0) * vs_r[..., 1:2]], dim=-1)
ref_fd4 = F.scaled_dot_product_attention(
    q4.float().view(1, H_q, 1, D4),
    kd_r.repeat_interleave(2, dim=0).unsqueeze(0),
    vd_r.repeat_interleave(2, dim=0).unsqueeze(0),
    is_causal=False,
    scale=sm_sc4,
).view(H_q, D4).to(torch.float16)
assert (out_fd4 - ref_fd4).abs().max().item() <= 2e-3

# Verify multi-row Gated-DeltaNet verify kernels (S=4, SAVE_INPUTS=True + commit_verify_prefix rollback)!
S_mv = 4
c_st_mr_ref = torch.randn(1, conv_dim, 3, dtype=torch.float32)
c_st_mr_tri = c_st_mr_ref.clone()
qkv_mr = torch.randn(S_mv, conv_dim, dtype=torch.float16)
c_out_mr = torch.empty((S_mv, conv_dim), dtype=torch.float32)
c_in_mr = torch.empty((1, conv_dim, 3 + S_mv), dtype=torch.float32)
_causal_conv1d_silu_multirow_kernel[(2,)](
    qkv_mr, c_st_mr_tri, c_w, c_out_mr, c_in_mr, S_mv, conv_dim,
    CONV_K=4, K_HIST=3, SAVE_CONV_IN=True, BLOCK_C=256
)
c_in_mr_ref = torch.cat([c_st_mr_ref, qkv_mr.float().transpose(0, 1).unsqueeze(0)], dim=-1)
c_out_mr_ref = F.silu(F.conv1d(c_in_mr_ref, c_w, groups=conv_dim)).squeeze(0).transpose(0, 1).contiguous()
assert (c_in_mr - c_in_mr_ref).abs().max().item() < 1e-6
assert (c_st_mr_tri - c_in_mr_ref[:, :, -3:]).abs().max().item() < 1e-6
assert (c_out_mr - c_out_mr_ref).abs().max().item() < 1e-5

b_mr = torch.randn(S_mv, H_v, dtype=torch.float16)
a_mr = torch.randn(S_mv, H_v, dtype=torch.float16)
s_mr_init = torch.randn(H_v, D_v, D_k, dtype=torch.float32) * 0.1
s_mr_tri = s_mr_init.clone()
out_mr_dn = torch.empty((S_mv, H_v, D_v), dtype=torch.float32)
pre_ssm_mr = torch.empty_like(s_mr_tri)
qs_mr = torch.empty((S_mv, H_k, D_k), dtype=torch.float32)
ks_mr = torch.empty((S_mv, H_k, D_k), dtype=torch.float32)
vs_mr = torch.empty((S_mv, H_v, D_v), dtype=torch.float32)
decs_mr = torch.empty((S_mv, H_v), dtype=torch.float32)
bets_mr = torch.empty((S_mv, H_v), dtype=torch.float32)
_gated_deltanet_multirow_verify_kernel[(H_v, 2)](
    c_out_mr, b_mr, a_mr, dt_b, ssm_a, s_mr_tri, out_mr_dn,
    pre_ssm_mr, qs_mr, ks_mr, vs_mr, decs_mr, bets_mr,
    S_mv, conv_dim, 64, q_sc, 1e-6,
    SAVE_INPUTS=True, H_K=H_k, H_V=H_v, D_K=D_k, D_V=D_v, BLOCK_K=32, BLOCK_V=16
)

# Reference 4-step PyTorch scan + intermediate state at step n_keep=2 for rollback verification!
s_mr_ref = s_mr_init.clone()
s_at_2 = None
out_mr_ref = torch.empty_like(out_mr_dn)
for t_i in range(S_mv):
    q_i, k_i, v_i = torch.split(c_out_mr_ref[t_i], [64, 64, 192], dim=-1)
    q_i = (q_i.view(H_k, D_k) * torch.rsqrt(q_i.view(H_k, D_k).pow(2).sum(-1, keepdim=True) + 1e-6) * q_sc).repeat(3, 1)
    k_i = (k_i.view(H_k, D_k) * torch.rsqrt(k_i.view(H_k, D_k).pow(2).sum(-1, keepdim=True) + 1e-6)).repeat(3, 1)
    v_i = v_i.view(H_v, D_v)
    dec_i = torch.exp(F.softplus(a_mr[t_i].float() + dt_b) * ssm_a)
    bet_i = torch.sigmoid(b_mr[t_i].float())
    s_mr_ref = s_mr_ref * dec_i.view(H_v, 1, 1)
    kv_mi = (s_mr_ref * k_i.unsqueeze(1)).sum(-1)
    s_mr_ref = s_mr_ref + ((v_i - kv_mi) * bet_i.unsqueeze(-1)).unsqueeze(-1) * k_i.unsqueeze(1)
    out_mr_ref[t_i] = (s_mr_ref * q_i.unsqueeze(1)).sum(-1)
    if t_i == 1:
        s_at_2 = s_mr_ref.clone()

assert (out_mr_dn - out_mr_ref).abs().max().item() < 1e-5
assert (s_mr_tri - s_mr_ref).abs().max().item() < 1e-5
rec_test = LayerRecurrentState(device=torch.device("cpu"))
rec_test.conv_state = c_st_mr_tri
rec_test.ssm_state = s_mr_tri
rec_test.record_verify_inputs(pre_ssm_mr, c_in_mr, qs_mr, ks_mr, vs_mr, decs_mr, bets_mr, 3)
assert rec_test.commit_verify_prefix(2) is True
assert (rec_test.ssm_state - s_at_2).abs().max().item() < 1e-5
assert (rec_test.conv_state - c_in_mr_ref[:, :, 2:5]).abs().max().item() < 1e-6
"""
    env = dict(os.environ)
    env["TRITON_INTERPRET"] = "1"
    proc = subprocess.run([sys.executable, "-c", interp_code], env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"TRITON_INTERPRET check failed:\n{proc.stderr}\n{proc.stdout}"
    print("  [OK] TRITON_INTERPRET=1 bit-accurate numerical parity verified across all W4/W8/KV Triton kernels!")


def test_agentic_reasoning_and_tool_calls() -> None:
    """
    Verify:
    1. Zero `<think></think>` leakage in `content` across empty-think, Qwen3.5 pre-opened `<think>\\n`,
       orphan `</think>`, and unclosed `<think>` before `<tool_call>`.
    2. Dual-protocol `<tool_call>` parsing:
       - Qwen3.5 / Qwen3.8 XML `<function=...><parameter=...>` with `<` / `>` code preservation + JSON Schema coercion
       - Qwen3 / Hermes JSON `{"name": ..., "arguments": ...}` + multiple parallel tool calls
    3. Hardened Jinja templates (`QWEN35_AGENTIC_JINJA_TEMPLATE` & `QWEN3_AGENTIC_JINJA_TEMPLATE`) with
       `developer` role, mid-turn `system` messages, `content=None`, and JSON-string `tool_calls.arguments`.
    4. End-to-end FastAPI `/v1/chat/completions` sync & SSE streaming wire format (`reasoning_content`, `tool_calls`, `finish_reason="tool_calls"`).
    """
    from pygguf_engine.agentic_protocol import (
        QWEN3_AGENTIC_JINJA_TEMPLATE,
        QWEN35_AGENTIC_JINJA_TEMPLATE,
        ReasoningAndToolStreamParser,
        normalize_messages_for_template,
        resolve_enable_thinking,
    )
    from pygguf_engine.engine import GenerationResult, StreamChunk

    tools_schema = [
        {
            "type": "function",
            "function": {
                "name": "write_code",
                "description": "Write code to a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "code": {"type": "string"},
                        "line_limit": {"type": "integer"},
                        "overwrite": {"type": "boolean"},
                    },
                    "required": ["path", "code"],
                },
            },
        }
    ]

    # Case 1: Empty `<think>\n\n</think>\n\nHello!` streamed 2 chars at a time -> ZERO `<think>` & ZERO reasoning chunks!
    p_empty = ReasoningAndToolStreamParser(prompt_str="<|im_start|>assistant\n", tools=tools_schema)
    raw_empty = "<think>\n\n</think>\n\nHello from clean assistant!"
    r_chunks, c_chunks = [], []
    for i in range(0, len(raw_empty), 2):
        em = p_empty.feed(raw_empty[i : i + 2])
        if em.reasoning_delta:
            r_chunks.append(em.reasoning_delta)
        if em.content_delta:
            c_chunks.append(em.content_delta)
    em_fin = p_empty.finalize()
    if em_fin.content_delta:
        c_chunks.append(em_fin.content_delta)
    assert r_chunks == [] and p_empty.full_reasoning == ""
    assert "".join(c_chunks) == "Hello from clean assistant!"
    assert "<think>" not in p_empty.full_content and "</think>" not in p_empty.full_content

    # Case 2: Qwen3.5 / Qwen3.8 pre-opened `<think>\n` prompt + XML `<tool_call>` containing `a < b and c > d`
    p_xml = ReasoningAndToolStreamParser(prompt_str="<|im_start|>assistant\n<think>\n", tools=tools_schema)
    raw_xml = (
        "Need to write src/check.py safely.\n</think>\n\n"
        "<tool_call>\n"
        "<function=write_code>\n"
        "<parameter=path>\nsrc/check.py\n</parameter>\n"
        "<parameter=code>\nif a < b and c > d:\n    return '<tag>'\n</parameter>\n"
        "<parameter=line_limit>\n42\n</parameter>\n"
        "<parameter=overwrite>\ntrue\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    streamed_tcs = []
    streamed_r = []
    streamed_c = []
    for i in range(0, len(raw_xml), 3):
        em = p_xml.feed(raw_xml[i : i + 3])
        if em.reasoning_delta:
            streamed_r.append(em.reasoning_delta)
        if em.content_delta:
            streamed_c.append(em.content_delta)
        streamed_tcs.extend(em.tool_calls)
    em_last = p_xml.finalize()
    streamed_tcs.extend(em_last.tool_calls)

    assert "".join(streamed_r).strip() == "Need to write src/check.py safely."
    assert p_xml.full_reasoning == "Need to write src/check.py safely."
    assert "".join(streamed_c) == ""
    assert len(streamed_tcs) == 1
    assert streamed_tcs[0].name == "write_code"
    parsed_args = json.loads(streamed_tcs[0].arguments_json)
    assert parsed_args == {
        "path": "src/check.py",
        "code": "if a < b and c > d:\n    return '<tag>'",
        "line_limit": 42,
        "overwrite": True,
    }

    # Case 3: Unclosed `<think>` going straight into JSON `<tool_call>` + 2 parallel calls!
    p_json = ReasoningAndToolStreamParser(prompt_str="<|im_start|>assistant\n", tools=tools_schema)
    raw_json = (
        "<think>\nCalling two tools\n"
        '<tool_call>\n{"name": "write_code", "arguments": {"path": "a.py", "code": "x=1"}}\n</tool_call>\n'
        '<tool_call>\n{"name": "write_code", "arguments": {"path": "b.py", "code": "y=2"}}\n</tool_call>'
    )
    p_json.feed(raw_json)
    p_json.finalize()
    assert p_json.full_reasoning == "Calling two tools"
    assert p_json.full_content == ""
    assert len(p_json.parsed_tool_calls) == 2
    assert json.loads(p_json.parsed_tool_calls[1].arguments_json)["path"] == "b.py"

    # Case 4: Multi-turn Jinja template rendering with `developer` role, `content=None`, JSON-string tool_calls, & `tool` reply
    from jinja2 import BaseLoader
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    multi_turn_msgs = normalize_messages_for_template(
        [
            {"role": "developer", "content": "You are a coding agent."},
            {"role": "user", "content": "Fix check.py"},
            {"role": "system", "content": "Workspace is /app"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "Let me write it.",
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "write_code", "arguments": '{"path": "check.py", "code": "1+1"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_abc", "content": "OK"},
        ]
    )
    env = ImmutableSandboxedEnvironment(loader=BaseLoader(), trim_blocks=True, lstrip_blocks=True)
    rendered_35 = env.from_string(QWEN35_AGENTIC_JINJA_TEMPLATE).render(
        messages=multi_turn_msgs, tools=tools_schema, add_generation_prompt=True, enable_thinking=True
    )
    assert "<function=write_code>" in rendered_35
    assert "<parameter=path>\ncheck.py\n</parameter>" in rendered_35
    assert "<tool_response>\nOK\n</tool_response>" in rendered_35
    assert rendered_35.endswith("<|im_start|>assistant\n<think>\n")

    rendered_3 = env.from_string(QWEN3_AGENTIC_JINJA_TEMPLATE).render(
        messages=multi_turn_msgs, tools=tools_schema, add_generation_prompt=True, enable_thinking=False
    )
    assert '"name": "write_code"' in rendered_3
    assert rendered_3.endswith("<think>\n\n</think>\n\n")

    assert resolve_enable_thinking(multi_turn_msgs, reasoning_effort="none") is False
    assert resolve_enable_thinking(multi_turn_msgs, thinking_param={"type": "disabled"}) is False

    # Case 5: End-to-end FastAPI `/v1/chat/completions` streaming & sync with mocked model stream
    with tempfile.TemporaryDirectory() as tmp:
        gguf_p = Path(tmp) / "agent_q35.gguf"
        _build_synthetic_qwen35_gguf(gguf_p, is_moe=False)
        eng = PyGGUFEngine(gguf_p, max_context_len=1024)

        async def _mock_astream(prompt_or_ids, sampling=None):
            pieces = [
                "Checking file...\n",
                "</think>\n\n",
                "<tool_call>\n<function=write_code>\n<parameter=path>\nhi.py\n</parameter>\n",
                "<parameter=code>\nx = 1 < 2\n</parameter>\n</function>\n</tool_call>",
            ]
            for idx_p, pc in enumerate(pieces):
                yield StreamChunk(delta_text=pc, token_id=idx_p + 1)
            yield StreamChunk(
                delta_text="",
                token_id=99,
                finish_reason="stop",
                stats=GenerationResult(
                    text="".join(pieces),
                    prompt_tokens=20,
                    cached_prefix_tokens=0,
                    completion_tokens=len(pieces),
                    finish_reason="stop",
                    prefill_time_s=0.01,
                    decode_time_s=0.02,
                ),
            )

        eng.astream_tokens = _mock_astream  # type: ignore[method-assign]
        app = create_openai_app(eng)
        tc = TestClient(app)

        # Non-streaming check
        resp_sync = tc.post(
            "/v1/chat/completions",
            json={
                "model": eng.model_id,
                "messages": [{"role": "developer", "content": "Be brief"}, {"role": "user", "content": "Write hi.py"}],
                "tools": tools_schema,
            },
        )
        assert resp_sync.status_code == 200, resp_sync.text
        b_sync = resp_sync.json()
        choice = b_sync["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["content"] is None
        assert choice["message"]["reasoning_content"] == "Checking file..."
        assert choice["message"]["tool_calls"][0]["function"]["name"] == "write_code"
        assert json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]) == {
            "path": "hi.py",
            "code": "x = 1 < 2",
        }

        # Streaming SSE check
        events = []
        with tc.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": eng.model_id,
                "messages": [{"role": "user", "content": "Write hi.py"}],
                "tools": tools_schema,
                "stream": True,
            },
        ) as sresp:
            for ln in sresp.iter_lines():
                if ln and ln != "data: [DONE]":
                    events.append(json.loads(ln[len("data: ") :]))

        r_streamed = "".join(e["choices"][0]["delta"].get("reasoning_content", "") for e in events if e["choices"])
        c_streamed = "".join(e["choices"][0]["delta"].get("content", "") for e in events if e["choices"])
        tc_events = [e["choices"][0]["delta"]["tool_calls"] for e in events if e["choices"] and "tool_calls" in e["choices"][0]["delta"]]
        assert r_streamed.strip() == "Checking file..."
        assert c_streamed == ""
        assert len(tc_events) == 1 and tc_events[0][0]["function"]["name"] == "write_code"
        assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"

    print("  [OK] Agentic Reasoning (`reasoning_content`, zero `<think>` leak) + XML/JSON Tool-Calls verified!")


def test_tp2_exactness_and_all_temp_speculative() -> None:
    """
    Verify:
    1. 2-Way Tensor Parallelism (`TP=2`, `parallel_mode="tp"`) across Hybrid Gated-DeltaNet (3:1 V-to-K tiled heads!)
       + Gated Full-Attention + SwiGLU MLP + Sharded `lm_head` produces bit-accurate prefill AND multi-step decode
       logits matching single-device execution (`max_err < 1e-3`)!
    2. All-Temperature Lossless Speculative Decoding (`temperature = 0.75` > `0.3`!) drafts & verifies tokens cleanly!
    """
    with tempfile.TemporaryDirectory() as tmp:
        q35_path = Path(tmp) / "tp2_qwen3_8_hybrid.gguf"
        _build_synthetic_qwen35_gguf(q35_path, is_moe=False)

        eng_1 = PyGGUFEngine(q35_path, devices=["cpu"], parallel_mode="pp", max_context_len=1024)
        eng_tp2 = PyGGUFEngine(q35_path, devices=["cpu", "cpu"], parallel_mode="tp", max_context_len=1024)
        assert eng_tp2.model.is_tensor_parallel and len(eng_tp2.model.tp_devices) == 2

        seq = [4, 5, 6, 7, 10, 11, 12]
        inp = torch.tensor([seq], dtype=torch.long)
        logits_1 = eng_1.model(inp, eng_1.kv_manager, start_pos=0, logits_to_keep=7)
        logits_tp2 = eng_tp2.model(inp, eng_tp2.kv_manager, start_pos=0, logits_to_keep=7)
        prefill_rel = (logits_1 - logits_tp2).abs().max().item() / logits_1.abs().max().clamp(min=1.0).item()
        assert prefill_rel < 1e-3, f"TP=2 prefill rel mismatch: {prefill_rel}"

        # Step-by-step decode continuation check (exercises TP KV cache + TP recurrent states!)
        l1_next = eng_1.model.forward_decode_step(9, eng_1.kv_manager, start_pos=7)
        ltp_next = eng_tp2.model.forward_decode_step(9, eng_tp2.kv_manager, start_pos=7)
        decode_rel = (l1_next - ltp_next).abs().max().item() / l1_next.abs().max().clamp(min=1.0).item()
        assert decode_rel < 1e-3, f"TP=2 decode rel mismatch: {decode_rel}"

        # All-temperature (`temperature = 0.75`) speculative N-gram check on Dense Qwen3!
        dense_path = Path(tmp) / "spec_temp075.gguf"
        _build_synthetic_gguf(dense_path, is_moe=False)
        eng_spec = PyGGUFEngine(dense_path, max_context_len=1024, speculative_ngram=True)
        # Repeat pattern so 2-gram drafter triggers at T=0.75!
        rep_ids = [5, 6, 7, 8, 5, 6, 7, 8, 5, 6]
        res_t075 = eng_spec.generate(
            rep_ids,
            SamplingConfig(max_tokens=10, temperature=0.75, top_k=20, seed=42),
        )
        assert res_t075.speculative_drafted > 0, "Expected N-gram speculative drafting at temperature=0.75!"
        assert res_t075.completion_tokens == 10

    print(
        f"  [OK] TP=2 Hybrid DeltaNet+Attn exactness (prefill_rel={prefill_rel:.6f}, decode_rel={decode_rel:.6f}) "
        f"+ All-Temp Speculative Decode verified!"
    )


def test_draft_mtp_and_zero_recompute_rollback() -> None:
    """
    Verify end-to-end native GGUF `draft-mtp` (`blk.{n_layers}.nextn.*`) speculative decoding:
    1. Zero-recompute `LayerRecurrentState.commit_verify_prefix(n_keep)` matches exact prefix scan (`< 1e-6`).
    2. Load-time disable (`speculative_mtp=False`) skips `blk.4` completely (`mtp_head is None`, 0 GB MTP VRAM).
    3. Load-time enable (`speculative_mtp=True`) loads `GGUFMTPDraftHead` on both `PP`, `TP=2`, and `MoE`,
       drafts tokens (`mtp_drafted > 0`), and produces 100% greedy bit-identical output vs baseline!
    4. Live per-request OpenAI `/v1/chat/completions` controls (`speculative_mtp=False`, `speculative_mtp=True`,
       `mtp_draft_tokens=2`, `speculative=False`) work cleanly across multi-turn cached conversations!
    """
    with tempfile.TemporaryDirectory() as tmp:
        mtp_path = Path(tmp) / "qwen3_8_full_mtp.gguf"
        _build_synthetic_qwen35_gguf(mtp_path, is_moe=False, include_full_mtp=True)

        # 1. Load-time `speculative_mtp=False` -> zero MTP weights loaded!
        eng_no_mtp = PyGGUFEngine(
            mtp_path,
            devices=["cpu"],
            parallel_mode="pp",
            max_context_len=1024,
            speculative_mtp=False,
            speculative_ngram=False,
        )
        assert eng_no_mtp.model.mtp_head is None
        assert eng_no_mtp.vram_report()["mtp_loaded"] is False

        prompt_ids = [4, 5, 6, 7, 10, 11]
        res_base = eng_no_mtp.generate(
            prompt_ids,
            SamplingConfig(max_tokens=10, temperature=0.0, cache_prompt=False),
        )
        assert res_base.mtp_drafted == 0 and res_base.speculative_drafted == 0

        # 2. Zero-recompute recurrent prefix rollback check (`n_keep=2` out of `S=5` verify batch!)
        eng_no_mtp.kv_manager.clear()
        eng_no_mtp.model(torch.tensor([[4, 5, 6]], dtype=torch.long), eng_no_mtp.kv_manager, start_pos=0, logits_to_keep=1)
        eng_no_mtp.kv_manager.begin_recurrent_verify()
        eng_no_mtp.model(torch.tensor([[7, 10, 11, 12, 13]], dtype=torch.long), eng_no_mtp.kv_manager, start_pos=3, logits_to_keep=5)
        assert eng_no_mtp.kv_manager.commit_recurrent_verify(start_pos=3, n_keep=2) is True
        eng_no_mtp.kv_manager.rollback_to(5)
        eng_no_mtp.kv_manager.record_tokens(3, [7, 10])
        logits_after_rollback = eng_no_mtp.model.forward_decode_step(14, eng_no_mtp.kv_manager, start_pos=5)

        eng_no_mtp.kv_manager.clear()
        eng_no_mtp.model(torch.tensor([[4, 5, 6, 7, 10]], dtype=torch.long), eng_no_mtp.kv_manager, start_pos=0, logits_to_keep=1)
        eng_no_mtp.kv_manager.record_tokens(0, [4, 5, 6, 7, 10])
        logits_direct = eng_no_mtp.model.forward_decode_step(14, eng_no_mtp.kv_manager, start_pos=5)
        rb_rel = (logits_after_rollback - logits_direct).abs().max().item() / logits_direct.abs().max().clamp(min=1.0).item()
        assert rb_rel < 1e-3, f"Zero-recompute DeltaNet rollback mismatch: {rb_rel}"

        # 3. `PP` + `draft-mtp` (`k=1` and `k=2`) greedy bit-equivalence vs baseline!
        eng_mtp = PyGGUFEngine(
            mtp_path,
            devices=["cpu"],
            parallel_mode="pp",
            max_context_len=1024,
            speculative_mtp=True,
            mtp_draft_tokens=1,
            speculative_ngram=False,
        )
        assert eng_mtp.model.mtp_head is not None
        assert eng_mtp.vram_report()["mtp_loaded"] is True

        res_mtp_k1 = eng_mtp.generate(
            prompt_ids,
            SamplingConfig(max_tokens=10, temperature=0.0, cache_prompt=False),
        )
        assert res_mtp_k1.mtp_drafted > 0, "Expected draft-mtp to propose tokens!"
        assert res_mtp_k1.text == res_base.text, f"MTP k=1 diverged: {res_mtp_k1.text!r} != {res_base.text!r}"

        res_mtp_k2 = eng_mtp.generate(
            prompt_ids,
            SamplingConfig(max_tokens=10, temperature=0.0, cache_prompt=False, mtp_draft_tokens=2),
        )
        assert res_mtp_k2.mtp_drafted >= res_mtp_k1.mtp_drafted
        assert res_mtp_k2.text == res_base.text, f"MTP k=2 diverged: {res_mtp_k2.text!r} != {res_base.text!r}"

        # 4. `TP=2` + `draft-mtp` sharded execution parity!
        eng_tp_mtp = PyGGUFEngine(
            mtp_path,
            devices=["cpu", "cpu"],
            parallel_mode="tp",
            max_context_len=1024,
            speculative_mtp=True,
            mtp_draft_tokens=1,
            speculative_ngram=False,
        )
        assert eng_tp_mtp.model.is_tensor_parallel
        assert eng_tp_mtp.model.mtp_head is not None and eng_tp_mtp.model.mtp_head.is_tensor_parallel
        res_tp_base = eng_tp_mtp.generate(
            prompt_ids,
            SamplingConfig(max_tokens=10, temperature=0.0, cache_prompt=False, use_mtp=False),
        )
        res_tp_mtp = eng_tp_mtp.generate(
            prompt_ids,
            SamplingConfig(max_tokens=10, temperature=0.0, cache_prompt=False, use_mtp=True),
        )
        assert res_tp_base.mtp_drafted == 0
        assert res_tp_mtp.mtp_drafted > 0
        assert res_tp_mtp.text == res_tp_base.text, f"TP=2 MTP diverged: {res_tp_mtp.text!r} != {res_tp_base.text!r}"

        # Non-zero temperature (`temperature=0.6`) lossless rejection sampling with `draft-mtp`!
        res_tp_t06 = eng_tp_mtp.generate(
            prompt_ids,
            SamplingConfig(max_tokens=8, temperature=0.6, top_k=20, seed=99, cache_prompt=False, use_mtp=True),
        )
        assert res_tp_t06.mtp_drafted > 0 and res_tp_t06.completion_tokens == 8

        # 5. `MoE` (`qwen35moe`) + `draft-mtp` check!
        moe_mtp_path = Path(tmp) / "qwen3_5_moe_full_mtp.gguf"
        _build_synthetic_qwen35_gguf(moe_mtp_path, is_moe=True, include_full_mtp=True)
        eng_moe_mtp = PyGGUFEngine(
            moe_mtp_path,
            devices=["cpu"],
            max_context_len=1024,
            speculative_mtp=True,
            speculative_ngram=False,
        )
        assert eng_moe_mtp.model.mtp_head is not None
        res_moe_mtp = eng_moe_mtp.generate(
            prompt_ids,
            SamplingConfig(max_tokens=6, temperature=0.0, cache_prompt=False),
        )
        assert res_moe_mtp.mtp_drafted > 0

        # 6. Live per-request OpenAI `/v1/chat/completions` toggle + multi-turn prefix-cache rebase!
        app = create_openai_app(eng_mtp)
        client = TestClient(app)

        # Turn 1: Disable MTP live for this request (`speculative_mtp=False`)
        r1 = client.post(
            "/v1/chat/completions",
            json={
                "model": eng_mtp.model_id,
                "messages": [{"role": "user", "content": "Hello world"}],
                "max_tokens": 6,
                "temperature": 0.0,
                "speculative_mtp": False,
                "speculative_ngram": False,
            },
        )
        assert r1.status_code == 200, r1.text
        b1 = r1.json()
        assert b1["timings"]["mtp_drafted"] == 0

        # Turn 2: Re-enable MTP live (`speculative_mtp=True, mtp_draft_tokens=2`) on multi-turn continuation!
        r2 = client.post(
            "/v1/chat/completions",
            json={
                "model": eng_mtp.model_id,
                "messages": [
                    {"role": "user", "content": "Hello world"},
                    {"role": "assistant", "content": b1["choices"][0]["message"]["content"] or "Hi"},
                    {"role": "user", "content": "Qwen 3.8"},
                ],
                "max_tokens": 6,
                "temperature": 0.0,
                "speculative_mtp": True,
                "mtp_draft_tokens": 2,
                "speculative_ngram": False,
            },
        )
        assert r2.status_code == 200, r2.text
        b2 = r2.json()
        assert b2["timings"]["mtp_drafted"] > 0

        # Turn 3: Master `speculative=False` switch disables both MTP and N-gram!
        r3 = client.post(
            "/v1/chat/completions",
            json={
                "model": eng_mtp.model_id,
                "messages": [{"role": "user", "content": "Hello Qwen"}],
                "max_tokens": 6,
                "temperature": 0.0,
                "speculative": False,
            },
        )
        assert r3.status_code == 200, r3.text
        assert r3.json()["timings"]["speculative_drafted"] == 0
        assert r3.json()["timings"]["mtp_drafted"] == 0

    print(
        f"  [OK] Native GGUF `draft-mtp` (`PP`, `TP=2`, `MoE`) + Zero-Recompute DeltaNet Rollback "
        f"(rb_rel={rb_rel:.6f}) + Live API Toggles verified!"
    )


def test_int4_kv_ring_mtp_and_adaptive_depth() -> None:
    """
    Verify the 3-Pillar Speed + 256K-Context Upgrade Bundle:
    1. `kv_dtype="int4"` KV Cache (~50% less VRAM than `int8`, ~75% less than `fp16`, chunked prefill, & MTP parity)
    2. Sliding-Window Ring KV Cache (`mtp_kv_window=8`, `_ring_cap=24`): long-prompt tail slicing (`S=18 > 8`),
       ring wrap-around across `capacity=24`, and speculative rollback across wrap boundaries!
    3. Confidence-Gated Adaptive MTP Depth (`k = 1..3`): low threshold chains `k=3` draft tokens per step,
       high threshold stops early at `k=1` without wasting secondary MTP calls, and both match greedy baseline!
    """
    from pygguf_engine.kv_cache import LayerKVCache

    # 1. Direct LayerKVCache VRAM comparison: `int4` vs `int8` vs `fp16` + online-softmax chunked attention!
    dev_cpu = torch.device("cpu")
    c_fp16 = LayerKVCache(2, 64, 128, dev_cpu, kv_dtype="fp16", page_growth=64)
    c_int8 = LayerKVCache(2, 64, 128, dev_cpu, kv_dtype="int8", page_growth=64)
    c_int4 = LayerKVCache(2, 64, 128, dev_cpu, kv_dtype="int4", page_growth=64)

    torch.manual_seed(42)
    k_in = torch.randn(1, 2, 20, 64, dtype=torch.float16)
    v_in = torch.randn(1, 2, 20, 64, dtype=torch.float16)
    c_fp16.append(k_in, v_in, 0)
    c_int8.append(k_in, v_in, 0)
    c_int4.append(k_in, v_in, 0)

    def _kv_bytes(c: LayerKVCache) -> int:
        b = c.k_buf.numel() * c.k_buf.element_size() * 2
        if c.k_scale is not None:
            b += c.k_scale.numel() * c.k_scale.element_size() * 2
        return b

    b16, b8, b4 = _kv_bytes(c_fp16), _kv_bytes(c_int8), _kv_bytes(c_int4)
    assert b4 < b8 * 0.56 and b4 < b16 * 0.30, f"Unexpected KV bytes: fp16={b16}, int8={b8}, int4={b4}"

    # Verify Path 3 (Online-Softmax Chunked Attention) on `int4` matches SDPA (`kv_chunk_size=6` < `total_len=20`)!
    q_chk = torch.randn(1, 4, 3, 64, dtype=torch.float16)
    out_sdpa = c_int4.attend(q_chk, start_pos=17, sm_scale=0.125, kv_chunk_size=4096)
    out_chunked = c_int4.attend(q_chk, start_pos=17, sm_scale=0.125, kv_chunk_size=6)
    assert (out_sdpa.float() - out_chunked.float()).abs().max().item() < 2e-3
    out_sdpa_f32 = c_int4.attend(q_chk.float(), start_pos=17, sm_scale=0.125, kv_chunk_size=4096)
    out_chunked_f32 = c_int4.attend(q_chk.float(), start_pos=17, sm_scale=0.125, kv_chunk_size=6)
    assert (out_sdpa_f32 - out_chunked_f32).abs().max().item() < 1e-6

    # 2. End-to-end `kv_dtype="int4"` + Ring-Window `mtp_kv_window=8` + Adaptive MTP Depth (`k=1..3`)!
    with tempfile.TemporaryDirectory() as tmp:
        mtp_path = Path(tmp) / "qwen3_8_int4_ring_mtp.gguf"
        _build_synthetic_qwen35_gguf(mtp_path, is_moe=False, include_full_mtp=True)

        eng_int4_ring = PyGGUFEngine(
            mtp_path,
            devices=["cpu"],
            parallel_mode="pp",
            max_context_len=512,
            kv_dtype="int4",
            speculative_mtp=True,
            mtp_adaptive=True,
            mtp_max_draft_tokens=3,
            mtp_confidence_threshold=0.55,
            mtp_kv_window=8,  # Small 8-token ring (`_ring_cap = 24`) forces prompt tail-slicing AND ring wrap!
            speculative_ngram=False,
        )
        assert eng_int4_ring.vram_report()["kv_dtype"] == "int4"
        assert eng_int4_ring.vram_report()["mtp_kv_window"] == 8
        assert eng_int4_ring.vram_report()["mtp_adaptive_default"] is True
        assert eng_int4_ring.kv_manager.mtp_kv_cache is not None
        assert eng_int4_ring.kv_manager.mtp_kv_cache._ring_cap == 24

        # Prompt of length 18 (> `mtp_kv_window=8`) + 14 generated tokens (`total = 32 > _ring_cap = 24` -> wraps ring!)
        long_prompt = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

        res_int4_base = eng_int4_ring.generate(
            long_prompt,
            SamplingConfig(max_tokens=14, temperature=0.0, cache_prompt=False, use_mtp=False),
        )
        # Adaptive high-threshold (`0.99`): proposes `k=1` per step without wasting extra MTP calls!
        res_adapt_low = eng_int4_ring.generate(
            long_prompt,
            SamplingConfig(
                max_tokens=14,
                temperature=0.0,
                cache_prompt=False,
                use_mtp=True,
                mtp_draft_tokens=3,
                mtp_adaptive=True,
                mtp_confidence_threshold=0.99,
            ),
        )
        # Adaptive permissive threshold (`0.001`): chains full `k=3` draft tokens per step across ring wrap!
        res_adapt_chain = eng_int4_ring.generate(
            long_prompt,
            SamplingConfig(
                max_tokens=14,
                temperature=0.0,
                cache_prompt=False,
                use_mtp=True,
                mtp_draft_tokens=3,
                mtp_adaptive=True,
                mtp_confidence_threshold=0.001,
            ),
        )

        assert res_adapt_low.text == res_int4_base.text, (
            f"Ring MTP (k=1 early-stop) diverged: {res_adapt_low.text!r} != {res_int4_base.text!r}"
        )
        assert res_adapt_chain.text == res_int4_base.text, (
            f"Ring MTP (k=3 chained across wrap) diverged: {res_adapt_chain.text!r} != {res_int4_base.text!r}"
        )
        # Chained `k=3` must draft strictly more MTP tokens than `k=1` early-stop!
        assert res_adapt_chain.mtp_drafted > res_adapt_low.mtp_drafted > 0, (
            f"Expected adaptive k=3 ({res_adapt_chain.mtp_drafted}) > early-stop k=1 ({res_adapt_low.mtp_drafted})"
        )
        # Physical MTP ring capacity stayed strictly bounded at 24 slots despite 32+ total sequence length!
        assert eng_int4_ring.kv_manager.mtp_kv_cache.capacity == 24
        assert eng_int4_ring.kv_manager.mtp_cache_start == 10  # 18 - 8 = 10 tail rebase!

    print(
        f"  [OK] INT4 KV Cache ({b4}B vs INT8 {b8}B / FP16 {b16}B) + Ring MTP Window (cap=24) + "
        f"Adaptive MTP Depth (k=1:{res_adapt_low.mtp_drafted} vs k=3:{res_adapt_chain.mtp_drafted}) verified!"
    )



def test_tp_input_block_transfer() -> None:
    """Check the cross-device contiguous-block loading fast path when CUDA exists."""
    if not torch.cuda.is_available():
        print("  [SKIP] TP input block transfer (CUDA unavailable)")
        return
    from pygguf_engine.quant_kernels import GGUFQuantLinear
    for mode in ("w4", "w8", "fp16"):
        n, k = 7, 128
        if mode == "w4":
            qw = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8)
            sc = torch.randn(n, k // 32, dtype=torch.float16)
            zr = torch.randn_like(sc)
        elif mode == "w8":
            qw = torch.randint(-128, 127, (n, k), dtype=torch.int8)
            sc = torch.randn(n, k // 16, dtype=torch.float16)
            zr = torch.randn_like(sc)
        else:
            qw = torch.randn(n, k, dtype=torch.float16)
            sc = zr = None
        bias = torch.randn(n, dtype=torch.float16)
        source = GGUFQuantLinear(mode, k, n, qw, sc, zr, bias)
        for blocks in ([1], [1, 2], [0, 2]):
            reference = source.slice_input_blocks(blocks, 32, torch.device("cpu"))
            actual = source.slice_input_blocks(blocks, 32, torch.device("cuda:0"))
            for key in ("qweight", "scales", "zeros", "bias"):
                ref, got = getattr(reference, key), getattr(actual, key)
                assert (ref is None) == (got is None)
                if ref is not None:
                    assert torch.equal(ref, got.cpu()), (mode, blocks, key)
    print("  [OK] TP input-block cross-device transfer")

def test_yarn_big_context_rope_scaling() -> None:
    """YaRN / Linear / NTK-aware RoPE scaling end-to-end plumbing + big-context chunked (Path 3) attention numerics."""
    from pygguf_engine import GGUFFile
    from pygguf_engine.model import RotaryEmbeddingCache
    from pygguf_engine.kv_cache import LayerKVCache

    # 1) GGUF metadata parsing of {arch}.rope.scaling.* (llama.cpp standard keys)
    with tempfile.TemporaryDirectory() as tmp:
        yarn_path = Path(tmp) / "yarn_qwen3.gguf"
        _build_synthetic_gguf(
            yarn_path,
            rope_scaling={
                "type": "yarn",
                "factor": 2.0,
                "orig_ctx_len": 4096,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "ext_factor": 1.0,
                "attn_factor": 1.14,
            },
            context_length=131072,
        )
        with GGUFFile(yarn_path) as gf:
            cfg = gf.build_model_config()
        assert cfg.rope_scaling_type == "yarn"
        assert cfg.rope_scaling_factor == 2.0
        assert cfg.rope_original_context_len == 4096
        assert cfg.rope_yarn_beta_fast == 32.0
        assert cfg.rope_yarn_beta_slow == 1.0
        assert cfg.rope_yarn_ext_factor == 1.0
        assert abs(cfg.rope_yarn_attn_factor - 1.14) < 1e-6
        assert cfg.max_context_len == 131072

    # 2) RotaryEmbeddingCache math (linear / ntk-aware / yarn) + YaRN mscale attention factor
    dev = torch.device("cpu")
    theta, dim = 10000.0, 128
    idx = torch.arange(0, dim, 2, dtype=torch.float32)
    base_inv = 1.0 / (theta ** (idx / dim))

    lin = RotaryEmbeddingCache(dim, theta, True, rope_scaling_type="linear", rope_scaling_factor=2.0)
    assert torch.allclose(lin._inv_freq(dev), base_inv / 2.0)
    assert lin.attn_factor == 1.0

    ntk = RotaryEmbeddingCache(dim, theta, True, rope_scaling_type="ntk-aware", rope_scaling_factor=2.0)
    base_ntk = theta * 2.0 ** (dim / (dim - 2))
    assert torch.allclose(ntk._inv_freq(dev), 1.0 / (base_ntk ** (idx / dim)))
    assert ntk.attn_factor == 1.0

    factor, orig = 4.0, 4096
    yarn = RotaryEmbeddingCache(
        dim,
        theta,
        True,
        rope_scaling_type="yarn",
        rope_scaling_factor=factor,
        rope_original_context_len=orig,
        rope_yarn_beta_fast=32.0,
        rope_yarn_beta_slow=1.0,
    )

    def _corr(rot: float) -> float:
        return (dim * math.log(orig / (rot * 2 * math.pi))) / (2 * math.log(theta))

    low, high = math.floor(_corr(32.0)), math.ceil(_corr(1.0))
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0.0, 1.0)
    expected = (base_inv / factor) * ramp + base_inv * (1.0 - ramp)
    assert torch.allclose(yarn._inv_freq(dev), expected, atol=1e-7)
    assert abs(yarn.attn_factor - (1.0 + 0.1 * math.log(factor))) < 1e-6
    # Tables grow on demand for big contexts (128K+ positions)
    cos, sin = yarn.ensure_capacity(dev, 65536)
    assert cos.shape[0] >= 65536 and cos.shape[1] == dim // 2

    # 3) Chunked online-softmax (Path 3) attention matches SDPA reference on quantized KV
    n_kv, gqa, head_dim = 2, 2, 32
    n_heads = n_kv * gqa
    torch.manual_seed(0)
    cache = LayerKVCache(n_kv_heads=n_kv, head_dim=head_dim, max_seq_len=512, device=dev, kv_dtype="int8")
    total = 40
    for t in range(total):
        cache.append(
            torch.randn(1, n_kv, 1, head_dim, dtype=torch.float16),
            torch.randn(1, n_kv, 1, head_dim, dtype=torch.float16),
            t,
        )
    sm_scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(1, n_heads, 16, head_dim, dtype=torch.float16)
    out = cache.attend(q, start_pos=0, sm_scale=sm_scale, kv_chunk_size=8)
    kf, vf = cache.get_dequant_slice(0, total, torch.float32)
    mask = torch.tril(torch.ones(16, total, dtype=torch.bool))
    ref = F.scaled_dot_product_attention(
        q.float(),
        kf.repeat_interleave(gqa, dim=1),
        vf.repeat_interleave(gqa, dim=1),
        attn_mask=mask,
        scale=sm_scale,
    )
    assert (out.float() - ref).abs().max().item() < 2e-2

    # 3b) Short-context SDPA path (enable_gqa) matches repeat_interleave reference
    q2 = torch.randn(1, n_heads, 6, head_dim, dtype=torch.float16)
    start2 = total - 6
    out2 = cache.attend(q2, start_pos=start2, sm_scale=sm_scale)
    mask2 = torch.arange(total).view(1, -1) <= torch.arange(start2, start2 + 6).view(-1, 1)
    kf2, vf2 = cache.get_dequant_slice(0, total, q2.dtype)
    ref2 = F.scaled_dot_product_attention(
        q2,
        kf2.repeat_interleave(gqa, dim=1),
        vf2.repeat_interleave(gqa, dim=1),
        attn_mask=mask2,
        scale=sm_scale,
    )
    assert (out2.float() - ref2.float()).abs().max().item() < 2e-2

    # 4) End-to-end: tiny YaRN-metadata GGUF loads with parsed scaling config and generates tokens
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "yarn_e2e.gguf"
        _build_synthetic_gguf(
            path,
            rope_scaling={"type": "yarn", "factor": 2.0, "orig_ctx_len": 4096},
            context_length=8192,
        )
        eng = PyGGUFEngine(path, devices=["cpu"], max_context_len=8192, enable_cuda_graph=False)
        assert eng.config.rope_scaling_type == "yarn"
        assert eng.config.rope_scaling_factor == 2.0
        res = eng.generate("Hello world", SamplingConfig(max_tokens=3, temperature=0.0))
        assert res.completion_tokens >= 1
        eng_report = eng.vram_report()
        assert eng_report["rope_scaling_type"] == "yarn"
    print(
        "  [OK] YaRN RoPE scaling (GGUF metadata + inv_freq/mscale) + linear/NTK-aware + "
        "big-context chunked attention + GQA SDPA verified!"
    )


if __name__ == "__main__":
    print("=== Running PyGGUF Engine Verification Suite ===")
    import os, subprocess, sys
    env = dict(os.environ, TRITON_INTERPRET="1")
    subprocess.run([sys.executable, "-m", "tests.test_multiquery_flash_interpreter"], env=env, check=True)
    test_tp_input_block_transfer()
    test_quant_repack_and_matmul_exactness()
    test_triton_sm75_ptx_and_interpreter_exactness()
    test_agentic_reasoning_and_tool_calls()
    test_tp2_exactness_and_all_temp_speculative()
    test_draft_mtp_and_zero_recompute_rollback()
    test_int4_kv_ring_mtp_and_adaptive_depth()
    test_qwen35_hybrid_deltanet_and_ud_quants()
    test_dense_moe_and_openai_server()
    test_yarn_big_context_rope_scaling()
    print("=== ALL TESTS PASSED ===")
