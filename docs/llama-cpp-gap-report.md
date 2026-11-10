# llama.cpp vs PyGGUF — decode-path teardown for Qwen3.8-27B on dual T4

Measured references (same model, same hardware, same 64K context, thinking on):

| engine | config | e2e / predicted tok/s |
|---|---|---|
| llama.cpp build (target of this report) | `--split-mode tensor --tensor-split 1,1`, q8_0 KV, flash-attn, `-b 4096 -ub 512`, draft-mtp n-max 2, jinja | **~20 tok/s** |
| PyGGUF (branch `arena/3189fa99-pytorch-engine`) | PP, int8 KV, MTP k=2 default, n-gram on | 8.22 e2e / 9.14 predicted |
| PyGGUF, PP + spec OFF (M=1 CUDA-graph replay) | per-stage graphs | 5.33 / 5.46 |

llama.cpp was cloned at `3b4d6ab3f79651935fba1dd93732f29022a7f5f5 2026-11-10` (`git clone --depth 1 https://github.com/ggml-org/llama.cpp.git`). All paths cited are relative to that tree.

## TL;DR

PyGGUF's 2.2× loss is **architectural, not cosmetic**, and decomposes into two independent gaps:

1. **Kernel throughput on Turing (~55–65% of the gap).** llama.cpp uses Turing `dp4a` INT8 SIMD dot products for Q4_K (`ggml/src/ggml-cuda/vecdotq.cuh` → `ggml_cuda_dp4a`), tensor-core MMA for prefill/MMQ on Turing (`mmq.cu:442`, `mmq.cuh:193` → `turing_mma_available(cc)` → `mmq-config-pascal-dp4a.cuh`), and Turing MMA for flash attention (`fattn.cu:164,660` → `fattn-mma-f16` instances). PyGGUF dequantizes to fp16 and runs generic Triton/PyTorch ops — no dp4a, and on sm_75 there is no `cp.async` pipeline at all (that's Ampere+), so Triton's latency hiding is structurally weaker than on Ampere/Hopper.
2. **Graph coverage (~35–45% of the gap).** llama.cpp captures the *entire* ggml compute graph as a CUDA graph automatically (`ggml-cuda.cu` `ggml_cuda_graph_check_compability` — the only blocker is `MUL_MAT_ID` needing a sync = MoE routing sync). Speculative verify passes (S = 2–8) and the draft-mtp head run **inside** the captured graph, so per-step CPU cost is microseconds regardless of batch shape. PyGGUF only replays graphs for M=1 non-speculative steps; every verify step pays thousands of Python-dispatched launches.

Everything below quantifies this and ranks remediation options.

## How llama.cpp gets ~20 tok/s (decode path, token by token)

### 1. Decode goes through MMVQ (matrix–vector quantized), not MMQ
- `mmvq.cu` — GEMV for q_len ≤ 8 (`get_mmvq_mmid_max_batch_turing_plus` exists for good reason: Turing gets hand-checked batch limits and separate code paths).
- Q4_K × Q8_1 dot products run on `__dp4a` (`vecdotq.cuh`: `sumi = ggml_cuda_dp4a(vi, u, sumi)`), i.e. INT8×INT8→INT32 IMMA-class integer SIMD at 8 ops/cycle/warp-lane on Turing — no fp16 dequant on the critical path, no fp16 packing overhead.
- MMQ (large batch/prefill) selects the `pascal_dp4a` config on Turing (`mmq.cuh:298`), which is also what sm_75 uses for the MMA path.

### 2. Flash attention on Turing hits the MMA kernels, not the naive path
- `fattn.cu:164/660/669`: `turing_mma_available(cc)` → routes to `fattn-mma-f16` (tensor-core `mma.sync` fp16) with dedicated template instances (`template-instances/fattn-mma-f16-instance-ncols2_*.cu`). The `fattn-vec.cuh` path is only a fallback for head sizes/types without MMA coverage. `sm_75` has no `cp.async` and no wgmma, and llama.cpp never relies on them for this arch.
- PyGGUF: full-attention decode goes through Triton split-KV flash-decode (INT4/INT8 KV) or SDPA/`enable_gqa` — fp16 math, no dp4a, no MMA tuning.

### 3. Speculative draft-mtp shares the target context and the same graph
- `common/speculative.cpp`: `COMMON_SPECULATIVE_TYPE_DRAFT_MTP` is built **against the target model context** (`/root/.cache/.../Qwen3.8-27B-UD-Q4_K_M.gguf` in the user's log: "creating MTP draft context against the target model").
- Draft + verify (S ≥ 1) both run through the same ggml backend scheduler → both are eligible for the same whole-graph CUDA-graph capture. There is no separate "eager verify" mode like PyGGUF's.
- `-ub 512` / `-b 4096` unbatched decode stays MMVQ-shaped.
- Sampler: `--split-mode tensor` forces **CPU sampling** (`backend sampling not supported with SPLIT_MODE_TENSOR`) — and with `--repeat-penalty 1.15 --repeat-last-n 2048` it is still just tight C++ loops at negligible cost. PyGGUF's GPU sampler is not a differentiator here (it helps latency-bound small-vocab cases, not this one).

### 4. NCCL is optional
- The user's build logs `NCCL not compiled in; falling back to internal AllReduce` — implemented in `ggml-cuda/allreduce.cu` with peer copies + an internal allreduce ("in the same process. Used by the tensor-split path alongside NCCL"), which is good enough for TP without NCCL. PyGGUF has no equivalent; its TP path does `normed.to(dev, non_blocking=True)` + Python `.float()` adds per layer per direction (≈256 cross-device ops and dozens of host-roundtrip risk points per token).

### 5. GatedDeltaNet (the hybrid linear-attention blocks)
- `ggml-cuda/gated_delta_net.cu` — fused single-launch recurrent scan for decode (`__launch_bounds__` templated on `S_v`, 4 cols/warp) with in-kernel state update, plus `gated-delta-net-mma.cu` (tensor-core multi-row verify) and `ssm-conv.cu` / `ssm-scan.cu` helpers.
- Architecture registration: `LLM_ARCH_QWEN3NEXT` (`qwen3next`) and siblings (`qwen35`, `qwen35moe`, `clef`, `clover_moe`) in `src/llama-arch.cpp:42` / `src/llama-model.cpp` — the user's model is `arch='qwen35'`, mapped to the same fused implementation.
- PyGGUF has equivalent-functionality Triton kernels (2-launch S=1 decode + multi-row verify, both used and compiled on Kaggle per the warmup log), but they execute inside a CUDA-graph-free eager path for verify and are compiled for a generic arch without Turing-specific scheduling.

## Bandwidth arithmetic (why 20 tok/s is the ceiling-adjacent number to beat)

27B × ~0.59 B/param (UD-Q4_K_M ≈ 16 GB weights) — per decode token, weights must stream:

| path | effective bytes/token | effective bandwidth | floor | achieved |
|---|---|---|---|---|
| TP split (640 GB/s combined) | 16 GB | 640 GB/s | ~25 ms | llama.cpp ~50 ms (≈50% eff.) |
| PP stages (2 × 320 GB/s, sequential) | 16 GB | 320 GB/s | ~50 ms | PyGGUF ~173–183 ms (≈27–29% eff.) |
| PP + MTP k=2 (weights amortized over ~2.3 tokens) | ~7.4 GB/token eff. | 320 GB/s | ~23 ms/token | PyGGUF ~110 ms/token (≈21% eff.) |

Reading: llama.cpp is "only" ~2× off its bandwidth floor (typical for quantized GEMV on Turing: dequant + fp16 FMA/dp4a mix). PyGGUF is **~3.5–5× off** on both the graphed and speculative paths, i.e. kernel-limited, not Python-limited on the graph path — proven by spec-off = 183 ms/token *with* CUDA graphs replaying (Python fully bypassed).

## What to fix, in order of expected payoff

### A. Whole-verify CUDA-graph capture in PyGGUF (highest impact on the spec path)
The entire reason spec-on verification is eager in PyGGUF. Required: position-tensor-driven RoPE + KV append for S>1 (replace python-int `start_pos` baked into captures), and route Path-1 posptr flash-decode kernels (or Path 3) to per-row query positions. Payoff: eliminates the eager dispatch for the MTP path; kernel-bound time unchanged. (Not addressed by language choice — this *is* the "C++ benefit" recreated inside Python.)

### B. dp4a-style INT8 Q4_K GEMV kernels (highest kernel-level payoff)
Triton cannot emit dp4a reliably on sm_75; the warp-per-row W4A16 kernels dequant to fp16 and do fp16 FMA. A hand-written CUDA (or Triton `tl.dot` with int8 inputs) Q4_K×Q8_1 dp4a kernel with `float` accumulation is the single largest kernel win (~1.5–2× on GEMV-dominant layers). Same for Q6_K (unsloth UD has Q6_K in output-ish tensors).

### C. Turing MMA flash-attention parity
For the 16 full-attention layers only. fattn-mma-f16-class mma.sync m16n8k8 kernel via Triton might reach ~70–80% of llama.cpp's MMA path; not urgent until (A) and (B) land.

### D. TP without Python D2D copies
If TP stays on the table, replace the per-layer `.to(dev, non_blocking=True)` + Python sums with a persistent P2P IPC buffer + `torch.ops` allreduce (or `torch.distributed` with NCCL on Kaggle images that ship it), or simply **keep PP** (measured 9.14 > TP's 8.45 — PP is the right topology for PyGGUF on this host).

### E. Accept and use llama.cpp for production decode
Zero engineering. Already validated: 20 tok/s, thinking on, jinja, draft-mtp n-max 2, internal allreduce fallback, CPU sampler — no NCCL required. PyGGUF remains the layer for OpenAI-compat surface, tool/reasoning handling, int4 128K–256K KV, YaRN, and the benchmark harness.

### F. Fork-and-optimize (only if kernel authorship is the goal)
`ggml-org/llama.cpp` is MIT. The surgical targets for *this* model/hardware: (1) UD-Q4_K_M-specific MMVQ row scheduling for sm_75, (2) verify/draft-mtp graph capture coverage (already automatic), (3) internal-allreduce tuning for 2×T4 PCIe. Iterate on Kaggle (nvcc is preinstalled there); nothing here is measurable on the dev sandbox.

## What not to bother with
- **Rewriting in C++ from scratch.** It does not fix (B) or (C) (still kernel work), and after (A) it buys only a small additional scheduling margin on top of CUDA graphs. It also forfeits the Triton/PTX-tested stack already in-tree.
- **BF16/FP8** anything — T4 is fp16/INT8/dp4a silicon.
- **CUDA graphs for MoE MUL_MAT_ID without sync** — llama.cpp itself gates this; MoE decode stays eager there too.
- **CPU sampler → GPU sampler** — already GPU-resident in PyGGUF; irrelevant at this vocab/temperature regime.

## Honest bottom line
On your dual T4, llama.cpp's 20 tok/s is the right production number *today*. PyGGUF at 9.14 tok/s is within the same kernel-family ballpark (Triton vs hand-CUDA on Turing) and **can** approach ~13–15 tok/s with (A) + (B) — roughly llama.cpp-adjacent, not beyond it. Actually *beating* 20 tok/s requires beating ggml's Turing Q4_K GEMV specifically (B done at hand-CUDA level, not Triton), which is a focused kernel project on Kaggle hardware, not a sandbox task — and it is the definition of the honest path to "better than llama.cpp" for this exact model on this exact hardware.
