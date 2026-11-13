# PyGGUF

Python, PyTorch and Triton GGUF inference server. See `PyGGUF_GitHub_Kaggle.ipynb` to clone this repository and run it on a Kaggle GPU, or `PyGGUF_Kaggle_OpenAI_Server.ipynb` for the same server with identical benchmark/launch cells (both clone from GitHub).

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
- **Fused single-pass Triton GQA flash attention** for big contexts and softcap (Path 3a): `tl.dot` tensor-core MMAs, FP32 online softmax, reads INT4/INT8/FP16 KV in-SRAM (no dequant materialization). CPU-verified numerics; T4 win must be measured (see below). Falls back to the torch online-softmax path on non-CUDA / non-power-of-two head_dim / ring layouts. Disable with `flash_attn=False` (`launch_kaggle_server` / `PyGGUFEngine` kwarg) or `--no-flash-attn` on the CLI; the loaded-model banner prints `| FlashAttn=off` when disabled.
- **GQA-native SDPA** for short/medium contexts (`enable_gqa=True`, K/V never materialized expanded; falls back to `repeat_interleave` on older torch).
- **GQA-folded chunked online-softmax** for big contexts/softcapping (see above).
- **Zero-cost N-gram self-speculative decoding** + optional GGUF `draft-mtp` head with adaptive depth.
- **Prefix KV cache reuse** across multi-turn requests and a GPU-resident top-k/top-p/min-p/repetition-penalty sampler (single sync per token).
- **Optional CPU-RAM KV spill** (`kv_cpu_spill=True`, default off): at the end of each turn the trunk attention KV and (hybrid) recurrent states are cloned into pinned host RAM and the GPU buffers are freed; on the next request a prompt prefix matching the spill is restored in one host->device pass instead of being re-prefilled. Recurrent anchors are the full spill boundary or the mid-sequence prompt checkpoint (`_ckpt_len`); anything else falls back to full re-prefill. Ring (MTP) caches stay on GPU; MTP rebase is handled by the normal catch-up path. Disable at runtime with `kv_cpu_spill=False` / no CLI flag (`--kv-cpu-spill` to enable). Peak host RAM roughly doubles the in-flight KV footprint between turns — check `free -g` on Kaggle before enabling at 256K.

On-GPU benchmark checklist for this addition (dual T4, Kaggle): prefill tok/s at 32K/64K context for the same prompt before/after (expect the win at long prefill, not at short-context decode, which uses Path 1); verify tok/s unchanged (Path 1 untouched); watch for softcap (Gemma) parity. No T4 speedup is claimed until measured.

## CUDA-graph verify path (S=2..8)

Whole-step CUDA graphs for the speculative verify forward are captured automatically on eligible
single-node runs (1-GPU or PP, non-MoE, non-TP): every attention layer must be on the fused
posptr Split-KV INT4/INT8 path (`head_dim` in 32/64/128/256, `attn_logit_softcapping == 0`, no
sliding-window ring layers) — this forces int4/int8 KV. Recurrent GatedDeltaNet layers are captured
as well and keep their zero-recompute rollback semantics (`begin_recurrent_verify()` re-points at
graph-owned scan-input buffers on every replay). Any ineligibility or capture failure falls back to
the eager verify forward with identical semantics; set `enable_cuda_graph=False` to disable both the
S=1 decode graphs and the S≥2 verify graphs. Eligibility is checked per model; the KV-page capacity
check re-captures automatically when the KV cache grows.

This must be re-benchmarked on dual T4 (Kaggle) — no new T4 numbers are claimed yet. If you run the
"5b. MTP depth sweep" notebooks on this branch, report both before/after `predicted_per_second` for
`mtp only` (pp) and the adaptive rows; compare against the table above.

## Dual-T4 tok/s notes

Measured on dual-T4 with `unsloth/Qwen3.8-27B-UD-Q4_K_M` (hybrid 48× GatedDeltaNet + 16× FullAttention),
`max_context_len=65536`, `kv_dtype="int8"`, `parallel_mode="pp"`, thinking on, short prompt, `seed=42`,
128 new tokens, full `timings` from `/v1/chat/completions`:

| case | e2e tok/s | predicted tok/s | mtp accepted/drafted |
|---|---|---|---|
| Engine default (temp 0.6, MTP k=2 + n-gram) | 8.22 | 8.69 | 79/104 (76.0%) |
| Spec OFF (temp 0.6) | 5.33 | 5.46 | — |
| n-gram only (temp 0.6) | 5.49 | 5.63 | — (n-gram 7/26) |
| **MTP only (temp 0.6)** | **8.76** | **9.14** | **83/110 (75.5%)** |
| Greedy + spec OFF | 5.72 | 5.87 | — |
| Greedy + engine default | 7.64 | 7.92 | 75/117 (64.1%) |
| Spec OFF + thinking off | 5.57 | 5.80 | — |

llama.cpp reference on the same model/ctx/KV (TP split): **~20 tok/s** — the residual gap is kernel
throughput on Turing (Q4_K GEMV etc.), not scheduling. Key lessons from the measurement:

- **Speculation ON wins decisively** — ~2.3–2.5 tokens per verify step at 75–77% acceptance. Do not disable MTP on this workload.
- Spec-off M=1 PP graph replay is *slower*, not faster: per-token cost on T4 is kernel-bound (~173–183 ms), and graphs only remove dispatch, not weight reads.
- n-gram is acceptance-neutral at temperature 0.6 (7/26) but zero-VRAM; keep for repetitive/greedy contexts.
- `enable_thinking=False` cuts wall-clock by halving generated tokens, not by raising tok/s (5.8 pred tok/s).
- Prefill is fast (~0.55s for this prompt; chunked 2048, cache-aware `cache_prompt` reuses prefixes across turns).

Next levers to try (all per-request, no relaunch — see "5b. MTP depth sweep" in the notebooks):
`mtp_draft_tokens=3`, and adaptive chaining (`mtp_adaptive=True` with `mtp_confidence_threshold` 0.55/0.30).
A TP-vs-PP A/B with speculation on requires relaunching with `parallel_mode="tp"` and re-running the sweep —
verify steps are weight-read-bound, so TP should win by the combined-bandwidth delta.

## Tests

Install dependencies including torch, triton, numpy, gguf, fastapi, httpx, jinja2, tokenizers and pydantic, then run:

```sh
python -m tests.test_pygguf_engine
```

The suite covers quant repack parity vs `gguf.quants`, Triton PTX codegen (sm_75) and `TRITON_INTERPRET=1` numerical parity, YaRN/linear/NTK-aware RoPE math and GGUF metadata parsing, chunked big-context attention vs SDPA reference, the FastAPI OpenAI surface, speculative decoding and CUDA-graph/Recurrent-state invariants. T4 performance and CUDA-specific behavior should be benchmarked separately on actual hardware.
