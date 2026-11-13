"""
High-speed Python GGUF Inference Engine (`PyGGUFEngine`) featuring:
- Zero-copy GGUF loading onto single or multi-GPU setups (Kaggle Dual T4 32GB VRAM)
- Quantized INT8 KV Cache (`kv_dtype="int8"`) for 32K-128K+ context windows
- Automatic Multi-Turn Prefix KV Cache Reuse (`cache_prompt=True`)
- Zero-VRAM N-Gram Self-Speculative Decoding (`speculative_ngram=True`)
- GPU-accelerated Top-K / Top-P / Min-P / Repetition-Penalty sampling & streaming
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, List, Literal, Optional, Sequence, Tuple, Union

import torch

from .gguf_parser import GGUFFile, GGUFModelConfig, GGUFTokenizer
from .kv_cache import EngineKVCacheManager
from .model import GGUFTransformerModel
from .quant_kernels import cuda_device_scope


@dataclass
class SamplingConfig:
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    min_p: float = 0.05
    repetition_penalty: float = 1.0
    stop_strings: List[str] = field(default_factory=list)
    stop_token_ids: List[int] = field(default_factory=list)
    cache_prompt: bool = True
    seed: Optional[int] = None
    # Live per-request speculative decoding overrides (`None` = use engine defaults)
    use_mtp: Optional[bool] = None
    use_ngram: Optional[bool] = None
    mtp_draft_tokens: Optional[int] = None
    mtp_adaptive: Optional[bool] = None
    mtp_confidence_threshold: Optional[float] = None


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    cached_prefix_tokens: int
    completion_tokens: int
    finish_reason: Literal["stop", "length", "tool_calls"]
    prefill_time_s: float
    decode_time_s: float
    speculative_accepted: int = 0
    speculative_drafted: int = 0
    mtp_accepted: int = 0
    mtp_drafted: int = 0
    reasoning_content: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def prompt_tok_per_s(self) -> float:
        evaluated = max(1, self.prompt_tokens - self.cached_prefix_tokens)
        return evaluated / max(self.prefill_time_s, 1e-6)

    @property
    def decode_tok_per_s(self) -> float:
        return self.completion_tokens / max(self.decode_time_s, 1e-6)


@dataclass
class StreamChunk:
    delta_text: str
    token_id: int
    finish_reason: Optional[Literal["stop", "length"]] = None
    stats: Optional[GenerationResult] = None


def _build_filtered_probs_gpu(
    logits_1d: torch.Tensor,
    cfg: SamplingConfig,
    recent_tokens: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Apply repetition penalty, temperature, top-k, min-p, and top-p purely on GPU without intermediate CPU syncs.
    Returns `(probs_1d, cand_indices_or_None, penalized_logits_1d)`.
    """
    logits = logits_1d.float()
    if cfg.repetition_penalty != 1.0 and recent_tokens:
        logits = logits.clone()
        uniq_ids = torch.tensor(list(set(recent_tokens[-256:])), device=logits.device, dtype=torch.long)
        score = logits.index_select(0, uniq_ids)
        score = torch.where(score < 0, score * cfg.repetition_penalty, score / cfg.repetition_penalty)
        logits.scatter_(0, uniq_ids, score)

    if cfg.temperature <= 1e-5:
        return logits, None, logits

    scaled = logits * (1.0 / cfg.temperature)

    if cfg.top_k > 0 and cfg.top_k < scaled.numel():
        cand_logits, cand_indices = torch.topk(scaled, cfg.top_k)
    else:
        cand_logits = scaled
        cand_indices = None

    probs = torch.softmax(cand_logits, dim=-1)

    if cfg.min_p > 0.0:
        p_max = probs.max()
        probs = torch.where(probs < cfg.min_p * p_max, torch.zeros_like(probs), probs)

    if 0.0 < cfg.top_p < 1.0:
        if cand_indices is not None:
            cum_probs = torch.cumsum(probs, dim=-1)
            probs = probs.masked_fill((cum_probs - probs) > cfg.top_p, 0.0)
        else:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_probs = sorted_probs.masked_fill((cum_probs - sorted_probs) > cfg.top_p, 0.0)
            probs = torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_probs)

    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    prob_sum = probs.sum()
    fallback = torch.zeros_like(probs)
    if cand_indices is not None:
        fallback[0] = 1.0
    else:
        fallback.scatter_(0, logits.argmax().unsqueeze(0), 1.0)
    probs = torch.where(prob_sum > 0.0, probs / prob_sum.clamp(min=1e-20), fallback)
    return probs, cand_indices, logits


def sample_token_gpu(
    logits_1d: torch.Tensor,
    cfg: SamplingConfig,
    recent_tokens: Optional[Sequence[int]] = None,
    generator: Optional[torch.Generator] = None,
) -> int:
    """Zero-extra-sync GPU token sampler (single 8-byte `.item()` transfer per token!)."""
    with cuda_device_scope(logits_1d.device):
        probs, cand_indices, logits = _build_filtered_probs_gpu(logits_1d, cfg, recent_tokens)
        if cfg.temperature <= 1e-5:
            return int(torch.argmax(logits).item())

        sampled_local = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
        if cand_indices is not None:
            return int(cand_indices.gather(0, sampled_local).item())
        return int(sampled_local.item())


def verify_or_sample_draft_token_gpu(
    logits_1d: torch.Tensor,
    draft_token: int,
    cfg: SamplingConfig,
    recent_tokens: Optional[Sequence[int]] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[int, bool]:
    """
    Exact lossless speculative verification for deterministic N-gram draft `q(draft_token) = 1`
    at ANY temperature (`T = 0.0 .. 2.0`):
    - Accepts `draft_token` with probability `P_target(draft_token)`.
    - On rejection, samples strictly from `P_res(x != draft_token) = P_target(x) / (1 - P_target(draft_token))`,
      preserving 100% mathematical equivalence to standard autoregressive sampling!
    """
    with cuda_device_scope(logits_1d.device):
        probs, cand_indices, logits = _build_filtered_probs_gpu(logits_1d, cfg, recent_tokens)
        if cfg.temperature <= 1e-5:
            pred = int(torch.argmax(logits).item())
            return pred, (pred == draft_token)

        if cand_indices is not None:
            match_mask = cand_indices == draft_token
            p_draft = (probs * match_mask.to(probs.dtype)).sum()
        else:
            match_mask = None
            p_draft = probs[draft_token]

        u = torch.rand((), device=logits.device, dtype=torch.float32, generator=generator)
        if bool((u < p_draft).item()):
            return draft_token, True

        if cand_indices is not None:
            assert match_mask is not None
            res_probs = probs.masked_fill(match_mask, 0.0)
        else:
            res_probs = probs.clone()
            res_probs[draft_token] = 0.0

        res_sum = res_probs.sum()
        if float(res_sum.item()) <= 1e-20:
            return draft_token, True

        res_probs = res_probs / res_sum
        sampled_local = torch.multinomial(res_probs, num_samples=1, generator=generator).squeeze(-1)
        if cand_indices is not None:
            return int(cand_indices.gather(0, sampled_local).item()), False
        return int(sampled_local.item()), False


def _argmax_with_confidence(logits_1d: torch.Tensor) -> Tuple[int, float]:
    """
    Return `(top_token_id, softmax_probability_of_top_token)` in a SINGLE 16-byte GPU-to-CPU sync
    so confidence-gated adaptive `draft-mtp` (`k = 1..3`) adds zero extra PCIe round-trips!
    """
    with cuda_device_scope(logits_1d.device):
        row = logits_1d.float()
        max_val, top_idx = torch.max(row, dim=-1)
        p_top = torch.exp(max_val - torch.logsumexp(row, dim=-1))
        packed = torch.stack([top_idx.to(torch.float64), p_top.to(torch.float64)]).tolist()
        return int(packed[0]), float(packed[1])


def find_ngram_draft(
    tokens: Sequence[int],
    max_draft: int = 5,
    ngram_min: int = 2,
    ngram_max: int = 4,
) -> List[int]:
    """
    Zero-VRAM Self-Speculative N-Gram Drafter:
    Scans context history from newest to oldest for an `n`-gram matching the current suffix
    and returns up to `max_draft` continuation tokens.
    """
    total = len(tokens)
    if total < ngram_min * 2 or max_draft <= 0:
        return []

    search_limit = max(0, total - 4096)
    for n in range(min(ngram_max, total // 2), ngram_min - 1, -1):
        suffix = tokens[total - n :]
        last_tok = suffix[-1]
        # Reverse scan for fast recency match (avoids per-position slice allocations)
        for pos in range(total - n - 1, search_limit + n - 2, -1):
            if tokens[pos] != last_tok:
                continue
            ok = True
            for off in range(1, n):
                if tokens[pos - off] != suffix[n - 1 - off]:
                    ok = False
                    break
            if not ok:
                continue
            end = min(total - n, pos + 1 + max_draft)
            draft = list(tokens[pos + 1 : end])
            if draft:
                return draft
    return []


class PyGGUFEngine:
    """
    Pure-Python + PyTorch + OpenAI Triton GGUF Engine (`0% llama.cpp`).
    """

    def __init__(
        self,
        gguf_path: Union[str, Path],
        *,
        hf_tokenizer: Optional[str] = None,
        chat_template: Optional[str] = None,
        chat_template_mode: Literal["auto", "gguf", "qwen35_xml", "qwen3_json"] = "auto",
        default_enable_thinking: bool = True,
        max_context_len: Optional[int] = None,
        kv_dtype: Literal["int4", "int8", "fp16"] = "int8",
        devices: Optional[Sequence[str]] = None,
        parallel_mode: Literal["auto", "tp", "pp"] = "auto",
        enable_cuda_graph: bool = True,
        flash_attn: bool = True,
        kv_cpu_spill: bool = False,
        speculative_mtp: bool = True,
        mtp_draft_tokens: Optional[int] = None,
        mtp_adaptive: bool = True,
        mtp_max_draft_tokens: int = 3,
        mtp_confidence_threshold: float = 0.55,
        mtp_kv_window: Optional[int] = 4096,
        speculative_ngram: bool = True,
        speculative_max_draft: int = 5,
        model_alias: Optional[str] = None,
        rope_scaling_type: Optional[Literal["none", "linear", "ntk-aware", "yarn"]] = None,
        rope_scaling_factor: Optional[float] = None,
        rope_scaling_original_context_length: Optional[int] = None,
        rope_yarn_beta_fast: Optional[float] = None,
        rope_yarn_beta_slow: Optional[float] = None,
        rope_yarn_ext_factor: Optional[float] = None,
        rope_yarn_attn_factor: Optional[float] = None,
        rope_yarn_log_mul: Optional[float] = None,
    ) -> None:
        t0 = time.perf_counter()
        self.gguf_path = Path(gguf_path)
        self.default_enable_thinking = default_enable_thinking
        self.speculative_mtp = bool(speculative_mtp)
        self.flash_attn: bool = bool(flash_attn)
        self.kv_cpu_spill: bool = bool(kv_cpu_spill)
        if mtp_draft_tokens is not None:
            # Explicit integer `mtp_draft_tokens` pins exact depth `k` unless overridden per request
            self.mtp_draft_tokens = max(1, min(8, int(mtp_draft_tokens)))
            self.mtp_adaptive = False
        else:
            self.mtp_draft_tokens = 1
            self.mtp_adaptive = bool(mtp_adaptive)
        self.mtp_max_draft_tokens = max(1, min(8, int(mtp_max_draft_tokens)))
        self.mtp_confidence_threshold = float(mtp_confidence_threshold)
        self.mtp_kv_window: Optional[int] = (
            int(mtp_kv_window) if (mtp_kv_window is not None and int(mtp_kv_window) > 0) else None
        )
        print(f"[PyGGUF] Opening GGUF checkpoint: {self.gguf_path}")

        with GGUFFile(self.gguf_path) as gf:
            self.config: GGUFModelConfig = gf.build_model_config()
            if max_context_len is not None:
                self.config.max_context_len = max_context_len
            # Optional RoPE context-extension overrides (otherwise the GGUF's own
            # `{arch}.rope.scaling.*` metadata, if any, is used verbatim).
            if rope_scaling_type is not None:
                self.config.rope_scaling_type = rope_scaling_type
            if rope_scaling_factor is not None:
                self.config.rope_scaling_factor = float(rope_scaling_factor)
            if rope_scaling_original_context_length is not None:
                self.config.rope_original_context_len = int(rope_scaling_original_context_length)
            if rope_yarn_beta_fast is not None:
                self.config.rope_yarn_beta_fast = float(rope_yarn_beta_fast)
            if rope_yarn_beta_slow is not None:
                self.config.rope_yarn_beta_slow = float(rope_yarn_beta_slow)
            if rope_yarn_ext_factor is not None:
                self.config.rope_yarn_ext_factor = float(rope_yarn_ext_factor)
            if rope_yarn_attn_factor is not None:
                self.config.rope_yarn_attn_factor = float(rope_yarn_attn_factor)
            if rope_yarn_log_mul is not None:
                self.config.rope_yarn_log_mul = float(rope_yarn_log_mul)

            self.model_id = model_alias or self.config.name or self.gguf_path.stem
            rope_note = ""
            if getattr(self.config, "rope_scaling_type", "none") != "none" and float(getattr(self.config, "rope_scaling_factor", 1.0) or 1.0) != 1.0:
                rope_note = f" | RoPE={self.config.rope_scaling_type} x{self.config.rope_scaling_factor:g}"
                if self.config.rope_scaling_type == "yarn" and self.config.rope_original_context_len:
                    rope_note += f" (orig {self.config.rope_original_context_len})"
            print(
                f"[PyGGUF] Architecture={self.config.arch} | Layers={self.config.n_layers} | "
                f"Hidden={self.config.hidden_size} | Heads={self.config.n_heads}/{self.config.n_kv_heads} | "
                f"MoE={self.config.is_moe} ({self.config.n_experts}x top-{self.config.n_experts_used}) | "
                f"MaxCtx={self.config.max_context_len} | KV={kv_dtype}{rope_note}"
            )

            self.tokenizer = GGUFTokenizer(
                gf,
                self.config,
                hf_tokenizer=hf_tokenizer,
                chat_template=chat_template,
                chat_template_mode=chat_template_mode,
            )
            self.model = GGUFTransformerModel(
                gf,
                self.config,
                devices=devices,
                compute_dtype=torch.float16,
                parallel_mode=parallel_mode,
                enable_cuda_graph=enable_cuda_graph,
                load_mtp=self.speculative_mtp,
            )
            self.model.eval()

        self.kv_dtype: Literal["int4", "int8", "fp16"] = kv_dtype
        self.kv_manager = EngineKVCacheManager(
            n_layers=self.config.n_layers,
            n_kv_heads=self.config.n_kv_heads,
            head_dim=self.config.head_dim,
            max_seq_len=self.config.max_context_len,
            layer_devices=self.model.layer_devices,
            kv_dtype=kv_dtype,
            recurrent_layers=self.model.recurrent_layer_flags,
            tp_devices=self.model.tp_devices if self.model.is_tensor_parallel else None,
            flash_attn=self.flash_attn,
            kv_cpu_spill=self.kv_cpu_spill,
        )
        if self.model.mtp_head is not None:
            self.kv_manager.init_mtp_cache(
                n_kv_heads=self.config.n_kv_heads,
                head_dim=self.config.head_dim,
                device=self.model.mtp_head.device,
                tp_devices=self.model.mtp_head.tp_devices if self.model.mtp_head.is_tensor_parallel else None,
                sliding_window=self.mtp_kv_window,
            )
        self._mtp_boundary_h: Dict[int, torch.Tensor] = {}

        self.speculative_ngram = speculative_ngram
        self.speculative_max_draft = speculative_max_draft
        self._async_lock = asyncio.Lock()
        # Dedicated persistent GPU worker thread so CUDA driver contexts & Triton module handles stay hot!
        self._gpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pygguf-gpu")

        load_s = time.perf_counter() - t0
        active_devs = self.model.tp_devices if self.model.is_tensor_parallel else self.model.layer_devices
        dev_summary = [str(d) for d in active_devs] if self.model.is_tensor_parallel else sorted({str(d) for d in active_devs})
        par_label = f"TP={len(self.model.tp_devices)}" if self.model.is_tensor_parallel else (
            "PP+CUDAGraph" if self.model.enable_cuda_graph and any(d.type == "cuda" for d in active_devs) else "PP"
        )
        n_rec = sum(self.model.recurrent_layer_flags)
        n_attn = self.config.n_layers - n_rec
        if self.model.mtp_head is not None:
            k_desc = (
                f"adaptive k=1..{max(self.mtp_draft_tokens, self.mtp_max_draft_tokens)}"
                if self.mtp_adaptive
                else f"k={self.mtp_draft_tokens}"
            )
            win_desc = f", win={self.mtp_kv_window}" if self.mtp_kv_window else ""
            mtp_status = f"enabled draft-mtp blk.{self.config.n_layers} ({k_desc}{win_desc})"
        elif self.config.nextn_predict_layers > 0:
            mtp_status = f"disabled {self.config.nextn_predict_layers} MTP (0 GB VRAM)"
        else:
            mtp_status = ""
        hybrid_parts = []
        if n_rec > 0:
            hybrid_parts.append(f"{n_rec} GatedDeltaNet + {n_attn} FullAttn")
        if mtp_status:
            hybrid_parts.append(mtp_status)
        hybrid_note = f" | Hybrid=({', '.join(hybrid_parts)})" if hybrid_parts else ""
        fa_note = ("" if self.flash_attn else " | FlashAttn=off") + (" | KVSpill=on" if self.kv_cpu_spill else "")
        print(f"[PyGGUF] Loaded '{self.model_id}' ({par_label}) across {dev_summary} in {load_s:.2f}s{hybrid_note}{fa_note}.")

        if any(d.type == "cuda" for d in active_devs):
            # Run warmup directly on the persistent GPU worker thread so its CUDA driver contexts are primed!
            self._gpu_executor.submit(self._warmup_cuda_kernels).result()

    @torch.inference_mode()
    def _maybe_spill_kv(self) -> None:
        """Park trunk attention KV + recurrent states in CPU RAM between turns (opt-in `kv_cpu_spill`)."""
        if not self.kv_cpu_spill:
            return
        try:
            info = self.kv_manager.spill_kv_to_cpu()
            if info is not None:
                print(
                    f"[PyGGUF] Spilled {info['bytes'] / 1e9:.2f} GB of KV/recurrent state to CPU RAM "
                    f"({info['len']} tokens); will restore on the next prefix-cache hit."
                )
        except Exception as exc:
            # Spill is purely an optimization for the next turn — never break serving on it.
            print(f"[PyGGUF] KV spill to CPU failed ({exc}); the caches stay GPU-resident for now.")

    def _warmup_cuda_kernels(self) -> None:
        """Pre-compile Triton PTX kernels (including M=2 multi-row GEMV, MTP head, and M=1 CUDA Graphs)."""
        t_w = time.perf_counter()
        dummy_prefill = torch.zeros((1, 12), dtype=torch.long)
        _, h_warm = self.model(dummy_prefill, self.kv_manager, start_pos=0, logits_to_keep=1, return_normed_hidden=True)
        # Warm up M=2 weight-reusing multi-row GEMV verify specialization!
        dummy_verify = torch.zeros((1, 2), dtype=torch.long)
        self.model(dummy_verify, self.kv_manager, start_pos=12, logits_to_keep=2)
        if self.model.mtp_head is not None:
            self.model.mtp_head.prefill_catch_up(
                dummy_prefill, h_warm, self.kv_manager, start_pos=0, trunk_token_embd=self.model.token_embd
            )
            self.model.mtp_head.forward_step(
                torch.zeros((1, 1), dtype=torch.long),
                h_warm[:, -1:, :],
                self.kv_manager,
                start_pos=12,
                trunk_model=self.model,
            )
        self.model.forward_decode_step(0, self.kv_manager, start_pos=14)
        self.kv_manager.clear()
        self._mtp_boundary_h.clear()
        torch.cuda.synchronize()
        cg_note = " + CUDAGraph" if self.model._decode_graphs_ready else ""
        mtp_note = " + draft-mtp" if self.model.mtp_head is not None else ""
        print(f"[PyGGUF] Compiled & warmed up Triton GPU kernels{cg_note}{mtp_note} in {time.perf_counter() - t_w:.2f}s.")

    def vram_report(self) -> Dict[str, Any]:
        gpus = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                alloc = torch.cuda.memory_allocated(i) / (1024**3)
                total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                gpus.append(
                    {
                        "gpu": i,
                        "name": torch.cuda.get_device_name(i),
                        "allocated_gib": round(alloc, 2),
                        "total_gib": round(total, 2),
                    }
                )
        return {
            "model_id": self.model_id,
            "arch": self.config.arch,
            "is_moe": self.config.is_moe,
            "parallel_mode": f"tp={len(self.model.tp_devices)}" if self.model.is_tensor_parallel else "pp",
            "cuda_graph": bool(self.model._decode_graphs_ready),
            "mtp_loaded": self.model.mtp_head is not None,
            "speculative_mtp_default": bool(self.speculative_mtp and self.model.mtp_head is not None),
            "mtp_draft_tokens_default": self.mtp_draft_tokens,
            "mtp_adaptive_default": self.mtp_adaptive,
            "mtp_max_draft_tokens": self.mtp_max_draft_tokens,
            "mtp_confidence_threshold": self.mtp_confidence_threshold,
            "mtp_kv_window": self.mtp_kv_window,
            "speculative_ngram_default": self.speculative_ngram,
            "max_context_len": self.config.max_context_len,
            "rope_scaling_type": getattr(self.config, "rope_scaling_type", "none"),
            "rope_scaling_factor": getattr(self.config, "rope_scaling_factor", 1.0),
            "rope_original_context_len": getattr(self.config, "rope_original_context_len", 0),
            "kv_dtype": self.kv_dtype,
            "cached_prefix_tokens": len(self.kv_manager.cached_tokens),
            "kv_cache_mib": round(self.kv_manager.allocated_vram_bytes() / (1024**2), 2),
            "kv_cache_max_mib": round(self.kv_manager.projected_max_vram_bytes() / (1024**2), 2),
            "gpus": gpus,
        }

    @torch.inference_mode()
    def stream_tokens(
        self,
        prompt_or_ids: Union[str, Sequence[int]],
        sampling: Optional[SamplingConfig] = None,
    ) -> Iterator[StreamChunk]:
        """
        Core streaming token generator with Prefix KV Cache Reuse + N-Gram Self-Speculative Decoding.
        """
        cfg = sampling or SamplingConfig()
        prompt_ids = (
            self.tokenizer.encode(prompt_or_ids, add_special_tokens=True)
            if isinstance(prompt_or_ids, str)
            else list(prompt_or_ids)
        )
        if not prompt_ids:
            prompt_ids = [self.config.bos_token_id or 0]

        max_new = max(1, min(cfg.max_tokens, self.config.max_context_len - len(prompt_ids)))
        stop_ids = set(self.config.eos_token_ids) | set(cfg.stop_token_ids)

        rng: Optional[torch.Generator] = None
        if cfg.seed is not None:
            rng = torch.Generator(device=self.model.last_device if self.model.lm_head else self.model.first_device)
            rng.manual_seed(cfg.seed)

        # Resolve per-request speculative decoding overrides (`SamplingConfig.use_mtp`, `use_ngram`, `mtp_draft_tokens`, `mtp_adaptive`)
        eff_use_mtp = (
            (self.speculative_mtp if cfg.use_mtp is None else bool(cfg.use_mtp))
            and (self.model.mtp_head is not None)
            and (cfg.temperature <= 1.2)
        )
        if cfg.mtp_draft_tokens is None:
            eff_mtp_adaptive = bool(self.mtp_adaptive if cfg.mtp_adaptive is None else cfg.mtp_adaptive)
            base_mtp_k = (
                max(self.mtp_draft_tokens, self.mtp_max_draft_tokens)
                if eff_mtp_adaptive
                else self.mtp_draft_tokens
            )
            eff_mtp_k = max(1, min(8, int(base_mtp_k)))
        else:
            eff_mtp_adaptive = bool(cfg.mtp_adaptive) if cfg.mtp_adaptive is not None else False
            eff_mtp_k = max(1, min(8, int(cfg.mtp_draft_tokens)))
        eff_mtp_conf_thresh = float(
            self.mtp_confidence_threshold
            if cfg.mtp_confidence_threshold is None
            else cfg.mtp_confidence_threshold
        )
        eff_use_ngram = (
            (self.speculative_ngram if cfg.use_ngram is None else bool(cfg.use_ngram))
            and (self.speculative_max_draft > 0)
            and (cfg.temperature <= 1.2)
        )

        # 1. Prefix KV Cache Reuse (restore from CPU RAM first when spilling is enabled)
        if self.kv_cpu_spill and cfg.cache_prompt:
            self.kv_manager.restore_kv_prefix(prompt_ids)
        if cfg.cache_prompt:
            prefix_hit = self.kv_manager.match_prefix_length(prompt_ids)
            self.kv_manager.rollback_to(prefix_hit)
        else:
            prefix_hit = 0
            self.kv_manager.clear()
            self._mtp_boundary_h.clear()

        # 2. Prompt Prefill (only evaluates uncached tail `prompt_ids[prefix_hit:]`!)
        t_prefill_start = time.perf_counter()
        uncached_ids = prompt_ids[prefix_hit:]
        inp = torch.tensor([uncached_ids], dtype=torch.long)
        prefill_h_norm: Optional[torch.Tensor] = None
        if eff_use_mtp:
            logits, prefill_h_norm = self.model(
                inp, self.kv_manager, start_pos=prefix_hit, logits_to_keep=1, return_normed_hidden=True
            )
            assert self.model.mtp_head is not None and prefill_h_norm is not None
            if prefix_hit == 0:
                self.kv_manager.reset_mtp_cache(0)
                self.model.mtp_head.prefill_catch_up(
                    inp,
                    prefill_h_norm,
                    self.kv_manager,
                    start_pos=0,
                    trunk_token_embd=self.model.token_embd,
                    prev_h=None,
                )
            elif self.kv_manager.mtp_exact_kv_len == prefix_hit and prefix_hit in self._mtp_boundary_h:
                self.model.mtp_head.prefill_catch_up(
                    inp,
                    prefill_h_norm,
                    self.kv_manager,
                    start_pos=prefix_hit,
                    trunk_token_embd=self.model.token_embd,
                    prev_h=self._mtp_boundary_h[prefix_hit],
                )
            else:
                # Seamless rebase when a previous turn had MTP disabled via live override!
                self.kv_manager.reset_mtp_cache(prefix_hit)
                self.model.mtp_head.prefill_catch_up(
                    inp,
                    prefill_h_norm,
                    self.kv_manager,
                    start_pos=prefix_hit,
                    trunk_token_embd=self.model.token_embd,
                    prev_h=None,
                )
            prompt_end_h = prefill_h_norm[:, -1, :].detach().clone()
            self._mtp_boundary_h = {len(prompt_ids): prompt_end_h}
        else:
            logits = self.model(inp, self.kv_manager, start_pos=prefix_hit, logits_to_keep=1)

        self.kv_manager.record_tokens(prefix_hit, uncached_ids)
        self.kv_manager.save_prompt_checkpoint()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        prefill_time_s = time.perf_counter() - t_prefill_start

        # Sample first token from last prompt position
        t_decode_start = time.perf_counter()
        all_tokens: List[int] = list(prompt_ids)
        generated_ids: List[int] = []
        decoded_so_far = ""
        spec_accepted = 0
        spec_drafted = 0
        mtp_accepted = 0
        mtp_drafted = 0
        finish_reason: Literal["stop", "length"] = "length"

        first_tok = sample_token_gpu(logits[0, -1, :], cfg, all_tokens, rng)
        if first_tok in stop_ids:
            finish_reason = "stop"
            res = GenerationResult(
                text="",
                prompt_tokens=len(prompt_ids),
                cached_prefix_tokens=prefix_hit,
                completion_tokens=0,
                finish_reason=finish_reason,
                prefill_time_s=prefill_time_s,
                decode_time_s=time.perf_counter() - t_decode_start,
            )
            yield StreamChunk(delta_text="", token_id=first_tok, finish_reason=finish_reason, stats=res)
            self._maybe_spill_kv()
            return

        all_tokens.append(first_tok)
        generated_ids.append(first_tok)
        new_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        delta = new_text[len(decoded_so_far) :]
        decoded_so_far = new_text

        # Check string stop sequences on first token
        stopped_by_str = False
        for s_stop in cfg.stop_strings:
            if s_stop and s_stop in decoded_so_far:
                idx = decoded_so_far.index(s_stop)
                decoded_so_far = decoded_so_far[:idx]
                stopped_by_str = True
                break

        if stopped_by_str or len(generated_ids) >= max_new:
            finish_reason = "stop" if stopped_by_str else "length"
            res = GenerationResult(
                text=decoded_so_far,
                prompt_tokens=len(prompt_ids),
                cached_prefix_tokens=prefix_hit,
                completion_tokens=len(generated_ids),
                finish_reason=finish_reason,
                prefill_time_s=prefill_time_s,
                decode_time_s=time.perf_counter() - t_decode_start,
            )
            yield StreamChunk(delta_text=decoded_so_far, token_id=first_tok, finish_reason=finish_reason, stats=res)
            self._maybe_spill_kv()
            return

        yield StreamChunk(delta_text=delta, token_id=first_tok)

        # 3. Autoregressive Decode Loop (Invariant: `len(all_tokens) == kv_len + 1`)
        # Hybrid Speculative Decoding: Zero-cost N-Gram suffix match first -> Native GGUF `draft-mtp` head second!
        eff_ngram_min = 3 if self.kv_manager.has_recurrent else 2
        mtp_pending_start_pos = len(prompt_ids)
        mtp_pending_tokens: List[int] = [first_tok] if eff_use_mtp else []
        mtp_pending_h: List[torch.Tensor] = (
            [prefill_h_norm[:, -1:, :].detach().clone()] if (eff_use_mtp and prefill_h_norm is not None) else []
        )
        prompt_boundary_entry = dict(self._mtp_boundary_h)

        while len(generated_ids) < max_new:
            kv_pos = len(all_tokens) - 1
            cur_tok = all_tokens[-1]
            remaining = max_new - len(generated_ids)

            draft: List[int] = []
            draft_source: Literal["none", "ngram", "mtp"] = "none"

            if remaining > 1 and eff_use_ngram:
                draft = find_ngram_draft(
                    all_tokens,
                    max_draft=min(self.speculative_max_draft, remaining - 1),
                    ngram_min=eff_ngram_min,
                )
                if draft:
                    draft_source = "ngram"

            if not draft and remaining > 1 and eff_use_mtp and mtp_pending_tokens:
                assert self.model.mtp_head is not None
                max_k = min(eff_mtp_k, remaining - 1)
                self.kv_manager.rollback_mtp_to(self.kv_manager.mtp_exact_kv_len)
                tok_batch = torch.tensor([mtp_pending_tokens], dtype=torch.long)
                h_batch = torch.cat(mtp_pending_h, dim=1)
                d_logits, h_chain = self.model.mtp_head.forward_step(
                    tok_batch,
                    h_batch,
                    self.kv_manager,
                    start_pos=mtp_pending_start_pos,
                    trunk_model=self.model,
                )
                self.kv_manager.mtp_exact_kv_len = mtp_pending_start_pos + len(mtp_pending_tokens)
                mtp_pending_start_pos = self.kv_manager.mtp_exact_kv_len
                mtp_pending_tokens.clear()
                mtp_pending_h.clear()

                if eff_mtp_adaptive and max_k > 1:
                    d0, conf0 = _argmax_with_confidence(d_logits[0, 0])
                    draft = [d0]
                    draft_source = "mtp"
                    cum_conf = conf0
                    if d0 not in stop_ids:
                        for step_i in range(1, max_k):
                            if cum_conf < eff_mtp_conf_thresh:
                                break
                            tok_i = torch.tensor([[draft[-1]]], dtype=torch.long)
                            d_logits_i, h_chain = self.model.mtp_head.forward_step(
                                tok_i,
                                h_chain,
                                self.kv_manager,
                                start_pos=kv_pos + step_i,
                                trunk_model=self.model,
                            )
                            di, confi = _argmax_with_confidence(d_logits_i[0, 0])
                            draft.append(di)
                            cum_conf *= confi
                            if di in stop_ids:
                                break
                else:
                    d0 = int(torch.argmax(d_logits[0, 0]).item())
                    draft = [d0]
                    draft_source = "mtp"
                    if max_k > 1 and d0 not in stop_ids:
                        for step_i in range(1, max_k):
                            tok_i = torch.tensor([[draft[-1]]], dtype=torch.long)
                            d_logits_i, h_chain = self.model.mtp_head.forward_step(
                                tok_i,
                                h_chain,
                                self.kv_manager,
                                start_pos=kv_pos + step_i,
                                trunk_model=self.model,
                            )
                            di = int(torch.argmax(d_logits_i[0, 0]).item())
                            draft.append(di)
                            if di in stop_ids:
                                break

            if not draft:
                if eff_use_mtp:
                    step_logits, step_h_norm = self.model.forward_decode_step(
                        cur_tok, self.kv_manager, start_pos=kv_pos, return_normed_hidden=True
                    )
                else:
                    step_logits = self.model.forward_decode_step(cur_tok, self.kv_manager, start_pos=kv_pos)
                    step_h_norm = None
                self.kv_manager.record_tokens(kv_pos, [cur_tok])
                cand_tokens = [sample_token_gpu(step_logits[0, 0, :], cfg, all_tokens, rng)]
                if eff_use_mtp and step_h_norm is not None:
                    h_cur = step_h_norm[:, 0:1, :].detach().clone()
                    mtp_pending_tokens.append(cand_tokens[0])
                    mtp_pending_h.append(h_cur)
                    self._mtp_boundary_h = {**prompt_boundary_entry, kv_pos + 1: h_cur.squeeze(1)}
            else:
                # Single batched weight-reusing GEMV verification pass over `[cur_tok] + draft`!
                spec_drafted += len(draft)
                if draft_source == "mtp":
                    mtp_drafted += len(draft)
                verify_seq = [cur_tok] + draft
                v_len = len(verify_seq)
                if self.kv_manager.has_recurrent:
                    self.kv_manager.begin_recurrent_verify()
                # Replays a captured whole-step CUDA graph on eligible models (int4/int8 KV, non-MoE,
                # single-node PP/1-GPU); falls back to the eager forward otherwise.
                step_logits_h = self.model.forward_verify_step(
                    verify_seq,
                    self.kv_manager,
                    start_pos=kv_pos,
                    return_normed_hidden=eff_use_mtp,
                )
                if eff_use_mtp:
                    step_logits, step_h_norm = step_logits_h
                else:
                    step_logits = step_logits_h
                    step_h_norm = None

                cand_tokens = []
                accepted_in_step = 0
                if cfg.temperature <= 1e-5 and cfg.repetition_penalty == 1.0:
                    # Fast single-sync greedy batch check across all D+1 positions!
                    greedy_preds = step_logits[0].argmax(dim=-1).tolist()
                    for j, d_tok in enumerate(draft):
                        pred_tok = int(greedy_preds[j])
                        cand_tokens.append(pred_tok)
                        if pred_tok != d_tok or pred_tok in stop_ids:
                            break
                        accepted_in_step += 1
                    else:
                        cand_tokens.append(int(greedy_preds[len(draft)]))
                else:
                    for j, d_tok in enumerate(draft):
                        pred_tok, ok = verify_or_sample_draft_token_gpu(
                            step_logits[0, j, :], d_tok, cfg, all_tokens + cand_tokens, rng
                        )
                        cand_tokens.append(pred_tok)
                        if not ok or pred_tok in stop_ids:
                            break
                        accepted_in_step += 1
                    else:
                        bonus_tok = sample_token_gpu(step_logits[0, len(draft), :], cfg, all_tokens + cand_tokens, rng)
                        cand_tokens.append(bonus_tok)

                spec_accepted += accepted_in_step
                if draft_source == "mtp":
                    mtp_accepted += accepted_in_step

                # Zero-recompute commit of `cur_tok` + `accepted_in_step` verified draft tokens!
                valid_kv_new = verify_seq[: 1 + accepted_in_step]
                n_keep = len(valid_kv_new)
                if self.kv_manager.has_recurrent:
                    if accepted_in_step < len(draft):
                        self.kv_manager.commit_recurrent_verify(kv_pos, n_keep)
                    else:
                        self.kv_manager.end_recurrent_verify()
                self.kv_manager.rollback_to(kv_pos + n_keep)
                self.kv_manager.record_tokens(kv_pos, valid_kv_new)

                if eff_use_mtp and step_h_norm is not None:
                    for j, c_tok in enumerate(cand_tokens):
                        mtp_pending_tokens.append(c_tok)
                        mtp_pending_h.append(step_h_norm[:, j : j + 1, :].detach().clone())
                    self._mtp_boundary_h = {
                        **prompt_boundary_entry,
                        kv_pos + n_keep: step_h_norm[:, n_keep - 1, :].detach().clone(),
                    }

            # Emit newly produced tokens one by one, checking stop conditions cleanly
            hit_end = False
            for idx_c, next_tok in enumerate(cand_tokens):
                if next_tok in stop_ids:
                    finish_reason = "stop"
                    hit_end = True
                    # Roll back any extra speculative KV entries beyond the tokens actually kept
                    self.kv_manager.rollback_to(min(len(self.kv_manager.cached_tokens), len(all_tokens)))
                    break

                all_tokens.append(next_tok)
                generated_ids.append(next_tok)
                # O(1) sliding-window incremental detokenization (preserves multi-byte UTF-8 & spaces)
                if len(generated_ids) <= 24:
                    new_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    piece = new_text[len(decoded_so_far) :]
                    decoded_so_far = new_text
                else:
                    prev_tail = self.tokenizer.decode(generated_ids[-16:-1], skip_special_tokens=True)
                    curr_tail = self.tokenizer.decode(generated_ids[-16:], skip_special_tokens=True)
                    piece = curr_tail[len(prev_tail) :] if curr_tail.startswith(prev_tail) else self.tokenizer.decode([next_tok], skip_special_tokens=True)
                    decoded_so_far += piece

                for s_stop in cfg.stop_strings:
                    if s_stop and s_stop in decoded_so_far:
                        cut = decoded_so_far.index(s_stop)
                        piece = piece[: max(0, len(piece) - (len(decoded_so_far) - cut))]
                        decoded_so_far = decoded_so_far[:cut]
                        finish_reason = "stop"
                        hit_end = True
                        break

                if len(generated_ids) >= max_new and not hit_end:
                    finish_reason = "length"
                    hit_end = True

                if hit_end:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    res = GenerationResult(
                        text=decoded_so_far,
                        prompt_tokens=len(prompt_ids),
                        cached_prefix_tokens=prefix_hit,
                        completion_tokens=len(generated_ids),
                        finish_reason=finish_reason,
                        prefill_time_s=prefill_time_s,
                        decode_time_s=time.perf_counter() - t_decode_start,
                        speculative_accepted=spec_accepted,
                        speculative_drafted=spec_drafted,
                        mtp_accepted=mtp_accepted,
                        mtp_drafted=mtp_drafted,
                    )
                    yield StreamChunk(delta_text=piece, token_id=next_tok, finish_reason=finish_reason, stats=res)
                    self._maybe_spill_kv()
                    return

                yield StreamChunk(delta_text=piece, token_id=next_tok)

            if hit_end:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                res = GenerationResult(
                    text=decoded_so_far,
                    prompt_tokens=len(prompt_ids),
                    cached_prefix_tokens=prefix_hit,
                    completion_tokens=len(generated_ids),
                    finish_reason=finish_reason,
                    prefill_time_s=prefill_time_s,
                    decode_time_s=time.perf_counter() - t_decode_start,
                    speculative_accepted=spec_accepted,
                    speculative_drafted=spec_drafted,
                    mtp_accepted=mtp_accepted,
                    mtp_drafted=mtp_drafted,
                )
                yield StreamChunk(delta_text="", token_id=all_tokens[-1], finish_reason=finish_reason, stats=res)
                self._maybe_spill_kv()
                return

    def generate(
        self,
        prompt_or_ids: Union[str, Sequence[int]],
        sampling: Optional[SamplingConfig] = None,
    ) -> GenerationResult:
        last_stats: Optional[GenerationResult] = None
        pieces: List[str] = []
        for chunk in self.stream_tokens(prompt_or_ids, sampling):
            if chunk.delta_text:
                pieces.append(chunk.delta_text)
            if chunk.stats is not None:
                last_stats = chunk.stats
        assert last_stats is not None
        return last_stats

    def chat(
        self,
        messages: Sequence[Dict[str, Any]],
        sampling: Optional[SamplingConfig] = None,
        enable_thinking: Optional[bool] = None,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        tool_choice: Any = None,
    ) -> GenerationResult:
        from .agentic_protocol import ReasoningAndToolStreamParser

        formatted = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            tools=tools,
            tool_choice=tool_choice,
            default_enable_thinking=self.default_enable_thinking,
        )
        raw_res = self.generate(formatted, sampling)
        parser = ReasoningAndToolStreamParser(
            prompt_str=formatted,
            tools=tools,
            parse_tool_calls=(tool_choice != "none"),
            emit_reasoning=True,
        )
        parser.feed(raw_res.text)
        parser.finalize()
        raw_res.text = parser.full_content
        raw_res.reasoning_content = parser.full_reasoning
        if parser.has_tool_calls:
            raw_res.tool_calls = [tc.to_openai_dict() for tc in parser.parsed_tool_calls]
            raw_res.finish_reason = "tool_calls"
        return raw_res

    async def astream_tokens(
        self,
        prompt_or_ids: Union[str, Sequence[int]],
        sampling: Optional[SamplingConfig] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Thread-offloaded async token generator so FastAPI's event loop stays 100% responsive."""
        async with self._async_lock:
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[Union[StreamChunk, BaseException, None]] = asyncio.Queue()

            def _produce() -> None:
                try:
                    for ch in self.stream_tokens(prompt_or_ids, sampling):
                        loop.call_soon_threadsafe(queue.put_nowait, ch)
                except BaseException as exc:
                    loop.call_soon_threadsafe(queue.put_nowait, exc)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            fut = loop.run_in_executor(self._gpu_executor, _produce)
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    if isinstance(item, BaseException):
                        raise item
                    yield item
            finally:
                await fut
