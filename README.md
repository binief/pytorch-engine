# PyGGUF

Python, PyTorch and Triton GGUF inference server. See `PyGGUF_GitHub_Kaggle.ipynb` to clone this repository and run it on a Kaggle GPU, or `PyGGUF_Kaggle_OpenAI_Server.ipynb` for the self-contained version.

## Kaggle

Select a dual-T4 accelerator. Run the GitHub-backed notebook cells in order. By default it fetches `main`, so review the repository or set `GIT_REF` to a trusted full commit SHA in the clone cell. The notebook downloads model weights separately from Hugging Face. A valid GGUF model, sufficient GPU and host memory, and network access are required. The notebook provides a stop/relaunch mechanism; relaunching reloads the model.

## Big context (32K-262K+)

Three mechanisms stack together:

1. **Quantized KV cache** — `kv_dtype="int8"` (default, ~50% VRAM saved) or `kv_dtype="int4"`
   (~75% saved) with paged, sequential 2048-token growth and no 2x peak spikes. A 7B-8B Q4_K_M model
   with `int4` KV fits 128K-256K+ tokens on a single 16 GB T4; GQA head counts keep the cache small.
2. **Online-softmax chunked attention** — for contexts above 4096 (prefill of long prompts, verify
   batches) attention dequantizes KV in 2048/4096-token chunks, folds the GQA group into the
   query-row dimension (K/V are never expanded), and accumulates with tensor-core fp16/bf16 matmuls
   against an FP32 softmax running max/sum — so both VRAM and compute stay flat at 128K+.
3. **RoPE context scaling** — both the checkpoint's own `{arch}.rope.scaling.*` GGUF metadata
   (llama.cpp-compatible `yarn` / `linear` / `ntk-aware` keys) and per-engine overrides are honored:

```python
from pygguf_engine import PyGGUFEngine

engine = PyGGUFEngine(
    "Qwen3-8B-Q4_K_M.gguf",
    max_context_len=131072,          # pre-set or rely on the GGUF context_length metadata
    kv_dtype="int4",                 # int4 halves KV bytes vs int8
    rope_scaling_type="yarn",        # or let the GGUF's rope.scaling.* metadata speak (omit overrides)
    rope_scaling_factor=4.0,
    rope_scaling_original_context_length=32768,
    # llama.cpp YaRN knobs (defaults shown): beta_fast=32.0, beta_slow=1.0, ext_factor=1.0,
    # attn factor auto = 1 + 0.1 * ext_factor * log(factor) unless yarn_attn_factor is pinned.
)
```

YaRN also folds its `mscale` attention factor into the softmax scale, including inside the
`draft-mtp` speculative head (it shares the trunk RoPE tables and scale). `linear` and `ntk-aware`
scaling (no mscale) are supported too, and unknown scaling types raise a loud error instead of
silently corrupting positions. RoPE tables grow on demand (up to the configured `max_context_len`).

`vram_report()` includes `kv_cache_mib` (current) and `kv_cache_max_mib` (projected at full `max_context_len`) — check the latter before picking a context length for your T4.

Equivalent CLI flags on `kaggle_serve.py`: `--rope-scaling-type yarn --rope-scaling-factor 4.0
--rope-scaling-orig-ctx 32768 [--rope-yarn-attn-factor X] [--rope-yarn-log-mul 0.1]`.

## Throughput

- **CUDA Graph replay** for `M=1` decode (per-pipeline-stage capture, re-captured only when KV pages grow).
- **Triton split-KV flash-decode** straight over the INT4/INT8 KV for `q_len <= 8` (single-token decode and short speculative verify), plus multi-row weight-reusing GEMV (`M <= 8`).
- **GQA-native SDPA** for short/medium contexts (`enable_gqa=True`, K/V never materialized expanded; falls back to `repeat_interleave` on older torch).
- **GQA-folded chunked online-softmax** for big contexts/softcapping (see above).
- **Zero-cost N-gram self-speculative decoding** + optional GGUF `draft-mtp` head with adaptive depth.
- **Prefix KV cache reuse** across multi-turn requests and a GPU-resident top-k/top-p/min-p/repetition-penalty sampler (single sync per token).

## Dual-T4 tok/s notes

Measured on dual-T4 with `unsloth/Qwen3.8-27B-UD-Q4_K_M` (hybrid 48× GatedDeltaNet + 16× FullAttention),
`max_context_len=65536`, `kv_dtype="int8"`, thinking on, short notebook test prompt:

- `parallel_mode="auto"` (→ TP=2) + `speculative_mtp=True` (`mtp_draft_tokens=2`) → **4.9 e2e tok/s** (llama.cpp: 20 tok/s).
  Eager hybrid decode is Python-dispatch-bound: TP bypasses the per-stage CUDA-graph capture, and the M=3 verify
  pass bypasses it too, so every token pays thousands of Python kernel launches + cross-GPU activation copies.
- Fastest known PyGGUF recipe for this workload: `parallel_mode="pp"` (per-stage CUDA graphs for M=1) with speculation
  off (`speculative_mtp=False` and `speculative_ngram=False`) — every step replays a captured graph with zero
  all-reduce and no Python dispatch. Trade-off: prefill is kernel-bound ~2x slower than TP on PP.
- With `speculative_mtp=True`, both verify passes (`S >= 2`) run eagerly; on dispatch-bound hardware the draft
  overhead can outweigh its gains. Watch `timings.predicted_per_second` and `timings.mtp_accepted/mtp_drafted`
  (exposed both in sync responses and in the streaming `include_usage` final chunk under `timings`).

A/B from a notebook cell:
`client.chat.completions.create(..., stream=False, extra_body={"speculative_mtp": False, "speculative_ngram": False})`
and compare `resp["timings"]["predicted_per_second"]` against the speculative run. TP remains the right choice for
prefill-heavy / large-batch prompts; it is not dispatch-bound there.

## Tests

Install dependencies including torch, triton, numpy, gguf, fastapi, httpx, jinja2, tokenizers and pydantic, then run:

```sh
python -m tests.test_pygguf_engine
```

The suite covers quant repack parity vs `gguf.quants`, Triton PTX codegen (sm_75) and `TRITON_INTERPRET=1` numerical parity, YaRN/linear/NTK-aware RoPE math and GGUF metadata parsing, chunked big-context attention vs SDPA reference, the FastAPI OpenAI surface, speculative decoding and CUDA-graph/Recurrent-state invariants. T4 performance and CUDA-specific behavior should be benchmarked separately on actual hardware.
