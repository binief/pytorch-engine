"""
Unified Multi-GPU GGUF Transformer (`0% llama.cpp`) supporting:
- Dense, Sparse Mixture-of-Experts (MoE), AND Hybrid Gated-DeltaNet + Gated Attention GGUF checkpoints
- Architectures: Llama / Llama-3, Mistral, Mixtral, Qwen2, Qwen2-MoE, Qwen3, Qwen3-MoE,
  Qwen3.5 / Qwen3.6 / Qwen3.8 (`qwen35`, `qwen35moe`, `qwen3next`), Gemma 2, Gemma 3
- Dual-Mode Multi-GPU Execution on Kaggle T4 x2 (`cuda:0` + `cuda:1`):
  1. 2-GPU Tensor Parallelism (`TP=2`, activates both T4 memory buses simultaneously at 640 GB/s!)
  2. Byte-balanced Pipeline Parallelism (`PP=2`) + Per-Stage `torch.cuda.CUDAGraph` `M=1` Decode Replay!
- Zero-VRAM Horizontal Weight Fusion (`FusedLinearPack` for QKV, Gate+Up, and DeltaNet projections)
- `logits_to_keep` slicing before `lm_head` (prevents 15GB+ vocab-logit OOMs during 64K-128K prefill!)
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gguf_parser import GGUFFile, GGUFModelConfig
from .kv_cache import (
    EngineKVCacheManager,
    LayerKVCache,
    LayerRecurrentState,
    exec_gated_deltanet_decode_step,
    exec_gated_deltanet_multirow,
    exec_gated_deltanet_scan,
)
from .quant_kernels import (
    FusedLinearPack,
    GGUFQuantLinear,
    GGUFQuantMoEExperts,
    cuda_device_scope,
    exec_fused_act_mul_rowscale,
    exec_fused_add_rmsnorm,
    exec_fused_rmsnorm,
    exec_fused_rmsnorm_silu_gate,
    load_float_tensor,
)


class GGUFRMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.dim = int(weight.shape[-1])
        self.register_buffer("weight", weight.to(torch.float32).contiguous(), persistent=False)

    def forward(self, x: torch.Tensor, out_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        return exec_fused_rmsnorm(x, self.weight, self.eps, out_dtype=out_dtype)

    def forward_with_residual(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        out_dtype: torch.dtype = torch.float16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return exec_fused_add_rmsnorm(x, delta, self.weight, self.eps, out_dtype=out_dtype)


class RotaryEmbeddingCache:
    """
    Precomputed RoPE cos/sin table per GPU device supporting both NeoX (split-half) and GGUF-Llama (interleaved).

    Also implements context-extension scaling so GGUF checkpoints can run far beyond their trained
    window (`max_context_len` larger than the checkpoint's native context):
      - `linear`    — positions divided by `factor` (inv_freq / factor)
      - `ntk-aware` — base re-scaled as `theta * factor ** (dim / (dim - 2))`
      - `yarn`      — YaRN ramp (correction dims from `beta_fast`/`beta_slow` around the original
                      context length) plus the standard `mscale = 1 + 0.1 * ext_factor * log(factor)`
                      attention-factor, applied by multiplying the softmax scale (`self.attn_factor`).
    """

    def __init__(
        self,
        rope_dim: int,
        rope_theta: float,
        neox_style: bool,
        rope_scaling_type: str = "none",
        rope_scaling_factor: float = 1.0,
        rope_original_context_len: int = 0,
        rope_yarn_beta_fast: float = 32.0,
        rope_yarn_beta_slow: float = 1.0,
        rope_yarn_ext_factor: float = 1.0,
        rope_yarn_attn_factor: float = 0.0,
        rope_yarn_log_mul: float = 0.1,
        max_context_len: int = 0,
    ) -> None:
        self.rope_dim = rope_dim
        self.rope_theta = rope_theta
        self.neox_style = neox_style
        self.rope_scaling_type = self._normalize_scaling_type(rope_scaling_type)
        self.rope_scaling_factor = float(rope_scaling_factor) if rope_scaling_factor else 1.0
        if not math.isfinite(self.rope_scaling_factor) or self.rope_scaling_factor <= 0.0:
            self.rope_scaling_factor = 1.0
        self.rope_original_context_len = int(rope_original_context_len or 0)
        self.rope_yarn_beta_fast = float(rope_yarn_beta_fast) if rope_yarn_beta_fast else 32.0
        self.rope_yarn_beta_slow = float(rope_yarn_beta_slow) if rope_yarn_beta_slow else 1.0
        self.rope_yarn_ext_factor = float(rope_yarn_ext_factor) if rope_yarn_ext_factor else 1.0
        self.rope_yarn_attn_factor = float(rope_yarn_attn_factor) if rope_yarn_attn_factor else 0.0
        self.rope_yarn_log_mul = float(rope_yarn_log_mul) if rope_yarn_log_mul and rope_yarn_log_mul > 0.0 else 0.1
        self.max_context_len = int(max_context_len or 0)
        self.attn_factor = self._compute_attn_factor()
        self._tables_per_dev: Dict[torch.device, Tuple[torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def _normalize_scaling_type(raw: Optional[str]) -> str:
        stype = (raw or "none").strip().lower().replace("_", "-")
        if stype in ("", "none", "false", "0"):
            return "none"
        if stype in ("linear", "pos-interpolation", "position-interpolation"):
            return "linear"
        if stype in ("ntk", "ntk-aware", "ntkaware", "dynamic-ntk"):
            return "ntk-aware"
        if stype == "yarn":
            return "yarn"
        # Unknown scaling types must not silently lie about positions; fail loudly instead.
        raise ValueError(f"Unsupported rope scaling type: {raw!r} (expected one of: none, linear, ntk-aware, yarn)")

    def _find_correction_dim(self, num_rotations: float, base: float, max_pos: int) -> float:
        return (self.rope_dim * math.log(max_pos / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    def _yarn_original_max_pos(self) -> int:
        if self.rope_original_context_len > 0:
            return self.rope_original_context_len
        if self.max_context_len > 0:
            return self.max_context_len
        return 4096

    def _inv_freq(self, device: torch.device) -> torch.Tensor:
        idx = torch.arange(0, self.rope_dim, 2, device=device, dtype=torch.float32)
        base = float(self.rope_theta)
        factor = self.rope_scaling_factor
        stype = self.rope_scaling_type
        if stype == "linear" and factor != 1.0:
            return 1.0 / (base ** (idx / self.rope_dim)) / factor
        if stype == "ntk-aware" and factor != 1.0:
            base_ntk = base * (factor ** (self.rope_dim / max(1, self.rope_dim - 2)))
            return 1.0 / (base_ntk ** (idx / self.rope_dim))
        inv_freq_orig = 1.0 / (base ** (idx / self.rope_dim))
        if stype == "yarn" and factor != 1.0:
            original_max_pos = self._yarn_original_max_pos()
            low = math.floor(self._find_correction_dim(self.rope_yarn_beta_fast, base, original_max_pos))
            high = math.ceil(self._find_correction_dim(self.rope_yarn_beta_slow, base, original_max_pos))
            if high <= low:
                high = low + 1
            ramp_idx = torch.arange(0, self.rope_dim // 2, device=device, dtype=torch.float32)
            ramp = torch.clamp((ramp_idx - float(low)) / float(high - low), 0.0, 1.0)
            inv_freq_interp = inv_freq_orig / factor
            return inv_freq_interp * ramp + inv_freq_orig * (1.0 - ramp)
        return inv_freq_orig

    def _compute_attn_factor(self) -> float:
        if self.rope_scaling_type == "yarn" and self.rope_scaling_factor > 1.0:
            if self.rope_yarn_attn_factor > 0.0:
                return float(self.rope_yarn_attn_factor)
            ext = self.rope_yarn_ext_factor if self.rope_yarn_ext_factor > 0.0 else 1.0
            return 1.0 + self.rope_yarn_log_mul * ext * math.log(self.rope_scaling_factor)
        return 1.0

    def ensure_capacity(self, device: torch.device, needed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cached = self._tables_per_dev.get(device)
        if cached is None or cached[0].shape[0] < needed:
            new_cap = max(8192, ((needed + 4095) // 4096) * 4096)
            with cuda_device_scope(device):
                inv_freq = self._inv_freq(device)
                positions = torch.arange(0, new_cap, device=device, dtype=torch.float32)
                freqs = torch.outer(positions, inv_freq)
                cached = (torch.cos(freqs).contiguous(), torch.sin(freqs).contiguous())
            self._tables_per_dev[device] = cached
        return cached

    def _get_cos_sin(
        self,
        device: torch.device,
        start_pos: int,
        seq_len: int,
        pos_tensor: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if pos_tensor is not None and seq_len == 1:
            cached = self._tables_per_dev[device]
            idx64 = pos_tensor.to(torch.long)
            cos = cached[0].index_select(0, idx64).view(1, 1, 1, -1)
            sin = cached[1].index_select(0, idx64).view(1, 1, 1, -1)
            return cos, sin

        needed = start_pos + seq_len
        cached = self.ensure_capacity(device, needed)
        cos = cached[0][start_pos:needed].view(1, seq_len, 1, -1)
        sin = cached[1][start_pos:needed].view(1, seq_len, 1, -1)
        return cos, sin

    def apply(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        start_pos: int,
        pos_tensor: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q: `[B, S, n_heads, head_dim]`
        k: `[B, S, n_kv_heads, head_dim]`
        """
        seq_len = q.shape[1]
        cos, sin = self._get_cos_sin(q.device, start_pos, seq_len, pos_tensor=pos_tensor)
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)

    def _rotate(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_rot = x[..., : self.rope_dim].float()
        x_pass = x[..., self.rope_dim :]

        if self.neox_style:
            half = self.rope_dim // 2
            x1 = x_rot[..., :half]
            x2 = x_rot[..., half:]
            rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(orig_dtype)
        else:
            x_even = x_rot[..., 0::2]
            x_odd = x_rot[..., 1::2]
            out_even = x_even * cos - x_odd * sin
            out_odd = x_even * sin + x_odd * cos
            rotated = torch.stack([out_even, out_odd], dim=-1).flatten(-2).to(orig_dtype)

        if x_pass.numel() > 0:
            return torch.cat([rotated, x_pass], dim=-1)
        return rotated


class GGUFAttention(nn.Module):
    def __init__(
        self,
        config: GGUFModelConfig,
        rope: RotaryEmbeddingCache,
        n_heads: int,
        n_kv_heads: int,
        qkv_pack: FusedLinearPack,
        o_proj: GGUFQuantLinear,
        has_q_gate: bool,
        is_pre_fused_qkv: bool,
        q_norm: Optional[GGUFRMSNorm],
        k_norm: Optional[GGUFRMSNorm],
    ) -> None:
        super().__init__()
        self.config = config
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = config.head_dim
        self.rope = rope
        self.qkv_pack = qkv_pack
        self.o_proj = o_proj
        self.has_q_gate = has_q_gate
        self.is_pre_fused_qkv = is_pre_fused_qkv
        self.q_norm = q_norm
        self.k_norm = k_norm

        if config.query_pre_attn_scalar > 0:
            self.sm_scale = 1.0 / math.sqrt(config.query_pre_attn_scalar)
        else:
            self.sm_scale = 1.0 / math.sqrt(self.head_dim)

    @classmethod
    def from_gguf(
        cls,
        gguf_file: GGUFFile,
        config: GGUFModelConfig,
        layer_idx: int,
        rope: RotaryEmbeddingCache,
        device: torch.device,
    ) -> "GGUFAttention":
        prefix = f"blk.{layer_idx}"
        if f"{prefix}.attn_q.weight" in gguf_file.tensors:
            q_proj = GGUFQuantLinear.from_gguf(
                gguf_file, f"{prefix}.attn_q.weight", f"{prefix}.attn_q.bias", device=device
            )
            k_proj = GGUFQuantLinear.from_gguf(
                gguf_file, f"{prefix}.attn_k.weight", f"{prefix}.attn_k.bias", device=device
            )
            v_proj = GGUFQuantLinear.from_gguf(
                gguf_file, f"{prefix}.attn_v.weight", f"{prefix}.attn_v.bias", device=device
            )
            has_q_gate = q_proj.out_features == 2 * config.n_heads * config.head_dim
            qkv_pack = FusedLinearPack([q_proj, k_proj, v_proj])
            is_pre_fused = False
            del q_proj, k_proj, v_proj
        else:
            qkv_single = GGUFQuantLinear.from_gguf(
                gguf_file, f"{prefix}.attn_qkv.weight", f"{prefix}.attn_qkv.bias", device=device
            )
            has_q_gate = False
            qkv_pack = FusedLinearPack([qkv_single])
            is_pre_fused = True
            del qkv_single

        o_proj = GGUFQuantLinear.from_gguf(
            gguf_file, f"{prefix}.attn_output.weight", f"{prefix}.attn_output.bias", device=device
        )

        q_norm_name = f"{prefix}.attn_q_norm.weight"
        k_norm_name = f"{prefix}.attn_k_norm.weight"
        q_norm = (
            GGUFRMSNorm(load_float_tensor(gguf_file, q_norm_name, device=device), config.rms_norm_eps)
            if q_norm_name in gguf_file.tensors
            else None
        )
        k_norm = (
            GGUFRMSNorm(load_float_tensor(gguf_file, k_norm_name, device=device), config.rms_norm_eps)
            if k_norm_name in gguf_file.tensors
            else None
        )
        return cls(
            config=config,
            rope=rope,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            qkv_pack=qkv_pack,
            o_proj=o_proj,
            has_q_gate=has_q_gate,
            is_pre_fused_qkv=is_pre_fused,
            q_norm=q_norm,
            k_norm=k_norm,
        )

    @classmethod
    def from_gguf_tp_shards(
        cls,
        gguf_file: GGUFFile,
        config: GGUFModelConfig,
        layer_idx: int,
        rope: RotaryEmbeddingCache,
        tp_devices: Sequence[torch.device],
    ) -> List["GGUFAttention"]:
        tp = len(tp_devices)
        # Repack full tensors on host; only final TP shards reside on GPUs.
        load_dev = torch.device("cpu")
        prefix = f"blk.{layer_idx}"
        h_per = config.n_heads // tp
        kv_per = config.n_kv_heads // tp
        hd = config.head_dim

        if f"{prefix}.attn_q.weight" in gguf_file.tensors:
            q_full = GGUFQuantLinear.from_gguf(
                gguf_file, f"{prefix}.attn_q.weight", f"{prefix}.attn_q.bias", device=load_dev
            )
            k_full = GGUFQuantLinear.from_gguf(
                gguf_file, f"{prefix}.attn_k.weight", f"{prefix}.attn_k.bias", device=load_dev
            )
            v_full = GGUFQuantLinear.from_gguf(
                gguf_file, f"{prefix}.attn_v.weight", f"{prefix}.attn_v.bias", device=load_dev
            )
            has_q_gate = q_full.out_features == 2 * config.n_heads * hd
            q_rows_per = h_per * (2 * hd if has_q_gate else hd)
            kv_rows_per = kv_per * hd
            q_shards = [q_full.slice_output_rows(slice(r * q_rows_per, (r + 1) * q_rows_per), tp_devices[r]) for r in range(tp)]
            k_shards = [k_full.slice_output_rows(slice(r * kv_rows_per, (r + 1) * kv_rows_per), tp_devices[r]) for r in range(tp)]
            v_shards = [v_full.slice_output_rows(slice(r * kv_rows_per, (r + 1) * kv_rows_per), tp_devices[r]) for r in range(tp)]
            del q_full, k_full, v_full
        else:
            qkv_full = GGUFQuantLinear.from_gguf(
                gguf_file, f"{prefix}.attn_qkv.weight", f"{prefix}.attn_qkv.bias", device=load_dev
            )
            has_q_gate = False
            q_total = config.n_heads * hd
            kv_total = config.n_kv_heads * hd
            q_rows_per = h_per * hd
            kv_rows_per = kv_per * hd
            q_shards = [qkv_full.slice_output_rows(slice(r * q_rows_per, (r + 1) * q_rows_per), tp_devices[r]) for r in range(tp)]
            k_shards = [
                qkv_full.slice_output_rows(slice(q_total + r * kv_rows_per, q_total + (r + 1) * kv_rows_per), tp_devices[r])
                for r in range(tp)
            ]
            v_shards = [
                qkv_full.slice_output_rows(
                    slice(q_total + kv_total + r * kv_rows_per, q_total + kv_total + (r + 1) * kv_rows_per),
                    tp_devices[r],
                )
                for r in range(tp)
            ]
            del qkv_full

        o_full = GGUFQuantLinear.from_gguf(
            gguf_file, f"{prefix}.attn_output.weight", f"{prefix}.attn_output.bias", device=load_dev
        )
        o_blk_cols = h_per * hd
        o_shards = [
            o_full.slice_input_blocks([r], o_blk_cols, tp_devices[r], keep_bias=(r == 0))
            for r in range(tp)
        ]
        del o_full

        q_norm_name = f"{prefix}.attn_q_norm.weight"
        k_norm_name = f"{prefix}.attn_k_norm.weight"
        q_nw = load_float_tensor(gguf_file, q_norm_name, device=torch.device("cpu")) if q_norm_name in gguf_file.tensors else None
        k_nw = load_float_tensor(gguf_file, k_norm_name, device=torch.device("cpu")) if k_norm_name in gguf_file.tensors else None

        shards: List[GGUFAttention] = []
        for r, dev in enumerate(tp_devices):
            with cuda_device_scope(dev):
                pack = FusedLinearPack([q_shards[r], k_shards[r], v_shards[r]])
                qn = GGUFRMSNorm(q_nw.to(dev), config.rms_norm_eps) if q_nw is not None else None
                kn = GGUFRMSNorm(k_nw.to(dev), config.rms_norm_eps) if k_nw is not None else None
                shards.append(
                    cls(
                        config=config,
                        rope=rope,
                        n_heads=h_per,
                        n_kv_heads=kv_per,
                        qkv_pack=pack,
                        o_proj=o_shards[r],
                        has_q_gate=has_q_gate,
                        is_pre_fused_qkv=False,
                        q_norm=qn,
                        k_norm=kn,
                    )
                )
        return shards

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: LayerKVCache,
        start_pos: int,
        pos_tensor: Optional[torch.Tensor] = None,
        kv_pos: Optional[int] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        attn_gate: Optional[torch.Tensor] = None

        if self.is_pre_fused_qkv:
            q_dim = self.n_heads * self.head_dim
            kv_dim = self.n_kv_heads * self.head_dim
            qkv_raw = self.qkv_pack.forward_single_concat(x)
            q_raw, k_raw, v_raw = torch.split(qkv_raw, [q_dim, kv_dim, kv_dim], dim=-1)
            q = q_raw.view(bsz, seq_len, self.n_heads, self.head_dim)
            k = k_raw.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
            v = v_raw.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        else:
            q_raw, k_raw, v_raw = self.qkv_pack(x)
            if self.has_q_gate:
                q_and_gate = q_raw.view(bsz, seq_len, self.n_heads, self.head_dim * 2)
                q, attn_gate = q_and_gate.chunk(2, dim=-1)
            else:
                q = q_raw.view(bsz, seq_len, self.n_heads, self.head_dim)
            k = k_raw.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
            v = v_raw.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        q, k = self.rope.apply(q, k, start_pos, pos_tensor=pos_tensor)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        eff_kv_pos = start_pos if kv_pos is None else kv_pos
        kv_cache.append(k, v, eff_kv_pos, pos_tensor=pos_tensor)
        # YaRN folds its `mscale` attention factor into the softmax scale (equivalent to scaling `q`).
        attn_out = kv_cache.attend(
            q,
            start_pos=eff_kv_pos,
            sm_scale=self.sm_scale * getattr(self.rope, "attn_factor", 1.0),
            softcap=self.config.attn_logit_softcapping,
            pos_tensor=pos_tensor,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.head_dim)
        if attn_gate is not None:
            gate_sig = torch.sigmoid(attn_gate.float()).to(attn_out.dtype)
            attn_out = attn_out * gate_sig.reshape(bsz, seq_len, self.n_heads * self.head_dim)
        return self.o_proj(attn_out)


class GGUFGatedDeltaNet(nn.Module):
    """
    Gated-DeltaNet Linear-Attention / SSM block for Qwen3.5 / Qwen3.6 / Qwen3.8 (`qwen35`, `qwen35moe`, `qwen3next`).
    Fuses `[wqkv | wqkv_gate | ssm_beta | ssm_alpha]` + 2-launch Triton S=1 decode step + fused Gated-RMSNorm!
    """

    def __init__(
        self,
        eps: float,
        proj_pack: FusedLinearPack,
        ssm_out: GGUFQuantLinear,
        conv_weight: torch.Tensor,
        ssm_dt: torch.Tensor,
        ssm_a: torch.Tensor,
        ssm_norm_weight: torch.Tensor,
        d_inner: int,
        num_v_heads: int,
        head_v_dim: int,
        conv_dim: int,
        key_dim: int,
        head_k_dim: int,
        num_k_heads: int,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.proj_pack = proj_pack
        self.ssm_out = ssm_out
        self.d_inner = d_inner
        self.num_v_heads = num_v_heads
        self.head_v_dim = head_v_dim
        self.conv_dim = conv_dim
        self.key_dim = key_dim
        self.head_k_dim = head_k_dim
        self.num_k_heads = num_k_heads
        self.conv_kernel = int(conv_weight.shape[-1])
        self.q_scale = 1.0 / math.sqrt(head_k_dim)

        self.register_buffer("conv_weight", conv_weight.reshape(conv_dim, 1, self.conv_kernel).contiguous(), persistent=False)
        self.register_buffer("ssm_dt", ssm_dt.view(-1).contiguous(), persistent=False)
        self.register_buffer("ssm_a", ssm_a.view(-1).contiguous(), persistent=False)
        self.register_buffer("ssm_norm_weight", ssm_norm_weight.to(torch.float32).view(-1).contiguous(), persistent=False)

    @classmethod
    def from_gguf(
        cls,
        gguf_file: GGUFFile,
        config: GGUFModelConfig,
        layer_idx: int,
        device: torch.device,
    ) -> "GGUFGatedDeltaNet":
        prefix = f"blk.{layer_idx}"
        wqkv = GGUFQuantLinear.from_gguf(gguf_file, f"{prefix}.attn_qkv.weight", device=device)
        wqkv_gate = GGUFQuantLinear.from_gguf(gguf_file, f"{prefix}.attn_gate.weight", device=device)
        ssm_beta = GGUFQuantLinear.from_gguf(gguf_file, f"{prefix}.ssm_beta.weight", device=device)
        ssm_alpha = GGUFQuantLinear.from_gguf(gguf_file, f"{prefix}.ssm_alpha.weight", device=device)
        ssm_out = GGUFQuantLinear.from_gguf(gguf_file, f"{prefix}.ssm_out.weight", device=device)

        d_inner = wqkv_gate.out_features
        num_v_heads = ssm_beta.out_features
        head_v_dim = d_inner // num_v_heads
        conv_dim = wqkv.out_features
        key_dim = (conv_dim - d_inner) // 2
        head_k_dim = config.ssm_state_size if config.ssm_state_size > 0 else head_v_dim
        num_k_heads = key_dim // head_k_dim

        proj_pack = FusedLinearPack([wqkv, wqkv_gate, ssm_beta, ssm_alpha])
        del wqkv, wqkv_gate, ssm_beta, ssm_alpha

        conv_w = load_float_tensor(gguf_file, f"{prefix}.ssm_conv1d.weight", device=device, dtype=torch.float32)
        ssm_dt = load_float_tensor(gguf_file, f"{prefix}.ssm_dt.bias", device=device, dtype=torch.float32)
        ssm_a = load_float_tensor(gguf_file, f"{prefix}.ssm_a", device=device, dtype=torch.float32)
        norm_w = load_float_tensor(gguf_file, f"{prefix}.ssm_norm.weight", device=device, dtype=torch.float32)

        return cls(
            eps=config.rms_norm_eps,
            proj_pack=proj_pack,
            ssm_out=ssm_out,
            conv_weight=conv_w,
            ssm_dt=ssm_dt,
            ssm_a=ssm_a,
            ssm_norm_weight=norm_w,
            d_inner=d_inner,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            conv_dim=conv_dim,
            key_dim=key_dim,
            head_k_dim=head_k_dim,
            num_k_heads=num_k_heads,
        )

    @classmethod
    def from_gguf_tp_shards(
        cls,
        gguf_file: GGUFFile,
        config: GGUFModelConfig,
        layer_idx: int,
        tp_devices: Sequence[torch.device],
    ) -> List["GGUFGatedDeltaNet"]:
        """
        Exact Tiled-Head Tensor Parallelism for Gated-DeltaNet (`V` head `vh` pairs with `K` head `vh % num_k_heads`).
        Rank `r` owns `k_heads = [r*(Hk/tp) .. (r+1)*(Hk/tp) - 1]` and all `rep` V-head tiles `tile*Hk + k_heads`,
        so zero cross-GPU communication is needed until the final `ssm_out` all-reduce!
        """
        tp = len(tp_devices)
        # Repack full tensors on host; only final TP shards reside on GPUs.
        load_dev = torch.device("cpu")
        prefix = f"blk.{layer_idx}"

        wqkv_full = GGUFQuantLinear.from_gguf(gguf_file, f"{prefix}.attn_qkv.weight", device=load_dev)
        wgate_full = GGUFQuantLinear.from_gguf(gguf_file, f"{prefix}.attn_gate.weight", device=load_dev)
        wbeta_full = GGUFQuantLinear.from_gguf(gguf_file, f"{prefix}.ssm_beta.weight", device=load_dev)
        walpha_full = GGUFQuantLinear.from_gguf(gguf_file, f"{prefix}.ssm_alpha.weight", device=load_dev)
        wout_full = GGUFQuantLinear.from_gguf(gguf_file, f"{prefix}.ssm_out.weight", device=load_dev)

        d_inner = wgate_full.out_features
        num_v_heads = wbeta_full.out_features
        head_v_dim = d_inner // num_v_heads
        conv_dim = wqkv_full.out_features
        key_dim = (conv_dim - d_inner) // 2
        head_k_dim = config.ssm_state_size if config.ssm_state_size > 0 else head_v_dim
        num_k_heads = key_dim // head_k_dim
        rep = num_v_heads // num_k_heads

        hk_per = num_k_heads // tp
        hv_per = num_v_heads // tp
        key_dim_per = hk_per * head_k_dim
        d_inner_per = hv_per * head_v_dim
        conv_dim_per = 2 * key_dim_per + d_inner_per

        conv_w_cpu = load_float_tensor(gguf_file, f"{prefix}.ssm_conv1d.weight", device=torch.device("cpu"), dtype=torch.float32)
        ssm_dt_cpu = load_float_tensor(gguf_file, f"{prefix}.ssm_dt.bias", device=torch.device("cpu"), dtype=torch.float32).view(-1)
        ssm_a_cpu = load_float_tensor(gguf_file, f"{prefix}.ssm_a", device=torch.device("cpu"), dtype=torch.float32).view(-1)
        norm_w_cpu = load_float_tensor(gguf_file, f"{prefix}.ssm_norm.weight", device=torch.device("cpu"), dtype=torch.float32)

        shards: List[GGUFGatedDeltaNet] = []
        for r, dev in enumerate(tp_devices):
            k_heads_r = list(range(r * hk_per, (r + 1) * hk_per))
            # Tiled V-head indices preserving `[tile_0_k_heads, tile_1_k_heads, ...]` order!
            v_heads_r = [tile * num_k_heads + kh for tile in range(rep) for kh in k_heads_r]

            q_rows = [kh * head_k_dim + d for kh in k_heads_r for d in range(head_k_dim)]
            k_rows = [key_dim + kh * head_k_dim + d for kh in k_heads_r for d in range(head_k_dim)]
            v_rows_in_qkv = [2 * key_dim + vh * head_v_dim + d for vh in v_heads_r for d in range(head_v_dim)]
            qkv_row_idx = q_rows + k_rows + v_rows_in_qkv
            gate_row_idx = [vh * head_v_dim + d for vh in v_heads_r for d in range(head_v_dim)]

            wqkv_r = wqkv_full.slice_output_rows(qkv_row_idx, dev)
            wgate_r = wgate_full.slice_output_rows(gate_row_idx, dev)
            wbeta_r = wbeta_full.slice_output_rows(v_heads_r, dev)
            walpha_r = walpha_full.slice_output_rows(v_heads_r, dev)
            wout_r = wout_full.slice_input_blocks(v_heads_r, head_v_dim, dev, keep_bias=(r == 0))

            qkv_idx_cpu = torch.tensor(qkv_row_idx, dtype=torch.long)
            vh_idx_cpu = torch.tensor(v_heads_r, dtype=torch.long)

            with cuda_device_scope(dev):
                pack_r = FusedLinearPack([wqkv_r, wgate_r, wbeta_r, walpha_r])
                shards.append(
                    cls(
                        eps=config.rms_norm_eps,
                        proj_pack=pack_r,
                        ssm_out=wout_r,
                        conv_weight=conv_w_cpu.index_select(0, qkv_idx_cpu).to(dev),
                        ssm_dt=ssm_dt_cpu.index_select(0, vh_idx_cpu).to(dev),
                        ssm_a=ssm_a_cpu.index_select(0, vh_idx_cpu).to(dev),
                        ssm_norm_weight=norm_w_cpu.to(dev),
                        d_inner=d_inner_per,
                        num_v_heads=hv_per,
                        head_v_dim=head_v_dim,
                        conv_dim=conv_dim_per,
                        key_dim=key_dim_per,
                        head_k_dim=head_k_dim,
                        num_k_heads=hk_per,
                    )
                )

        del wqkv_full, wgate_full, wbeta_full, walpha_full, wout_full
        return shards

    def forward(self, x: torch.Tensor, rec_state: LayerRecurrentState) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        if bsz != 1:
            raise ValueError(f"GGUFGatedDeltaNet expects batch_size=1, got {bsz}")

        conv_state, ssm_state = rec_state.ensure_allocated(
            self.conv_dim, self.conv_kernel, self.num_v_heads, self.head_v_dim, self.head_k_dim
        )

        qkv_mix, z, beta_raw, alpha_raw = self.proj_pack(x)

        # Ultra-fast S=1 CUDA Triton Decode Path (2 launches instead of 22!)
        if seq_len == 1:
            y_fast = exec_gated_deltanet_decode_step(
                qkv_mix=qkv_mix,
                beta_raw=beta_raw,
                alpha_raw=alpha_raw,
                conv_state=conv_state,
                conv_weight=self.conv_weight,
                ssm_dt=self.ssm_dt,
                ssm_a=self.ssm_a,
                ssm_state=ssm_state,
                key_dim=self.key_dim,
                num_k_heads=self.num_k_heads,
                num_v_heads=self.num_v_heads,
                head_k_dim=self.head_k_dim,
                head_v_dim=self.head_v_dim,
                q_scale=self.q_scale,
                eps=self.eps,
            )
            if y_fast is not None:
                z_heads = z.view(1, self.num_v_heads, self.head_v_dim)
                y_gated = exec_fused_rmsnorm_silu_gate(y_fast, z_heads, self.ssm_norm_weight, self.eps)
                return self.ssm_out(y_gated.reshape(1, 1, self.d_inner))
        elif 1 < seq_len <= 16:
            # Ultra-fast Multi-Row (`2 <= S <= 16`) Speculative Verify Path (2 launches + zero-recompute rollback!)
            y_multi = exec_gated_deltanet_multirow(
                qkv_mix=qkv_mix,
                beta_raw=beta_raw,
                alpha_raw=alpha_raw,
                conv_state=conv_state,
                conv_weight=self.conv_weight,
                ssm_dt=self.ssm_dt,
                ssm_a=self.ssm_a,
                ssm_state=ssm_state,
                key_dim=self.key_dim,
                num_k_heads=self.num_k_heads,
                num_v_heads=self.num_v_heads,
                head_k_dim=self.head_k_dim,
                head_v_dim=self.head_v_dim,
                q_scale=self.q_scale,
                eps=self.eps,
                rec_state=rec_state,
            )
            if y_multi is not None:
                z_heads = z.view(seq_len, self.num_v_heads, self.head_v_dim)
                y_gated = exec_fused_rmsnorm_silu_gate(y_multi, z_heads, self.ssm_norm_weight, self.eps).to(x.dtype)
                return self.ssm_out(y_gated.reshape(1, seq_len, self.d_inner))

        beta = torch.sigmoid(beta_raw.float()).squeeze(0)
        alpha = alpha_raw.float().squeeze(0)
        g = F.softplus(alpha + self.ssm_dt) * self.ssm_a
        decay = torch.exp(g)

        qkv_t = qkv_mix.float().transpose(1, 2)
        conv_in = torch.cat([conv_state, qkv_t], dim=-1)
        if self.conv_kernel > 1:
            conv_state.copy_(conv_in[:, :, -(self.conv_kernel - 1) :])
        conv_out = F.silu(F.conv1d(conv_in, self.conv_weight, groups=self.conv_dim))
        conv_out = conv_out.squeeze(0).transpose(0, 1).contiguous()

        q_raw, k_raw, v_raw = torch.split(conv_out, [self.key_dim, self.key_dim, self.d_inner], dim=-1)
        q = q_raw.view(seq_len, self.num_k_heads, self.head_k_dim)
        k = k_raw.view(seq_len, self.num_k_heads, self.head_k_dim)
        v = v_raw.view(seq_len, self.num_v_heads, self.head_v_dim)

        q = q * torch.rsqrt(q.pow(2).sum(dim=-1, keepdim=True) + self.eps)
        k = k * torch.rsqrt(k.pow(2).sum(dim=-1, keepdim=True) + self.eps)
        q_scaled = q * self.q_scale

        if rec_state.verify_active and 1 < seq_len <= 16:
            rec_state.record_verify_inputs(
                pre_ssm_state=ssm_state.clone(),
                conv_in=conv_in,
                q_scaled=q_scaled,
                k=k,
                v=v,
                decay=decay,
                beta=beta,
                hist_len=max(1, self.conv_kernel - 1),
            )

        # `exec_gated_deltanet_scan` tiles `num_k_heads -> num_v_heads` natively without `.repeat()`!
        y = exec_gated_deltanet_scan(q_scaled, k, v, decay, beta, ssm_state)

        z_heads = z.view(seq_len, self.num_v_heads, self.head_v_dim)
        y_gated = exec_fused_rmsnorm_silu_gate(y, z_heads, self.ssm_norm_weight, self.eps).to(x.dtype)
        return self.ssm_out(y_gated.reshape(1, seq_len, self.d_inner))


class GGUFDenseMLP(nn.Module):
    def __init__(
        self,
        gate_up_pack: FusedLinearPack,
        down_proj: GGUFQuantLinear,
        use_gelu: bool,
    ) -> None:
        super().__init__()
        self.gate_up_pack = gate_up_pack
        self.down_proj = down_proj
        self.use_gelu = use_gelu

    @classmethod
    def from_gguf(
        cls,
        gguf_file: GGUFFile,
        config: GGUFModelConfig,
        layer_idx: int,
        device: torch.device,
        prefix_override: Optional[str] = None,
    ) -> "GGUFDenseMLP":
        prefix = prefix_override or f"blk.{layer_idx}"
        use_gelu = config.arch.startswith("gemma")
        gate_proj = GGUFQuantLinear.from_gguf(gguf_file, f"{prefix}.ffn_gate.weight", device=device)
        up_proj = GGUFQuantLinear.from_gguf(gguf_file, f"{prefix}.ffn_up.weight", device=device)
        down_proj = GGUFQuantLinear.from_gguf(gguf_file, f"{prefix}.ffn_down.weight", device=device)
        pack = FusedLinearPack([gate_proj, up_proj])
        del gate_proj, up_proj
        return cls(pack, down_proj, use_gelu=use_gelu)

    @classmethod
    def from_gguf_tp_shards(
        cls,
        gguf_file: GGUFFile,
        config: GGUFModelConfig,
        layer_idx: int,
        tp_devices: Sequence[torch.device],
    ) -> List["GGUFDenseMLP"]:
        tp = len(tp_devices)
        # Repack full tensors on host; only final TP shards reside on GPUs.
        load_dev = torch.device("cpu")
        prefix = f"blk.{layer_idx}"
        use_gelu = config.arch.startswith("gemma")

        gate_full = GGUFQuantLinear.from_gguf(gguf_file, f"{prefix}.ffn_gate.weight", device=load_dev)
        up_full = GGUFQuantLinear.from_gguf(gguf_file, f"{prefix}.ffn_up.weight", device=load_dev)
        down_full = GGUFQuantLinear.from_gguf(gguf_file, f"{prefix}.ffn_down.weight", device=load_dev)

        inter_per = gate_full.out_features // tp
        shards: List[GGUFDenseMLP] = []
        for r, dev in enumerate(tp_devices):
            row_sl = slice(r * inter_per, (r + 1) * inter_per)
            g_r = gate_full.slice_output_rows(row_sl, dev)
            u_r = up_full.slice_output_rows(row_sl, dev)
            d_r = down_full.slice_input_blocks([r], inter_per, dev, keep_bias=(r == 0))
            with cuda_device_scope(dev):
                shards.append(cls(FusedLinearPack([g_r, u_r]), d_r, use_gelu=use_gelu))

        del gate_full, up_full, down_full
        return shards

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_up_pack.is_single_contiguous:
            gu = self.gate_up_pack.forward_single_concat(x)
            down_in, row_scale = exec_fused_act_mul_rowscale(gu, None, use_gelu=self.use_gelu)
        else:
            gate, up = self.gate_up_pack(x)
            down_in, row_scale = exec_fused_act_mul_rowscale(gate, up, use_gelu=self.use_gelu)
        return self.down_proj(down_in.to(x.dtype)).float() * row_scale


class GGUFSparseMoEMLP(nn.Module):
    """
    Sparse Mixture-of-Experts FFN block supporting:
    - Qwen3-MoE (`Qwen3-30B-A3B`), Qwen3.5-MoE (`Qwen3.5-35B-A3B`), Qwen2-MoE, and Mixtral!
    - Only reads active `top_k` expert tensors per token from GPU VRAM.
    """

    def __init__(
        self,
        gguf_file: GGUFFile,
        config: GGUFModelConfig,
        layer_idx: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.config = config
        self.n_experts = config.n_experts
        self.top_k = max(1, config.n_experts_used)
        self.norm_topk_prob = config.norm_topk_prob
        self.use_gelu = config.arch.startswith("gemma")

        prefix = f"blk.{layer_idx}"
        router_w = load_float_tensor(gguf_file, f"{prefix}.ffn_gate_inp.weight", device=device, dtype=torch.float32)
        self.register_buffer("router_weight", router_w, persistent=False)

        self.gate_exps = GGUFQuantMoEExperts.from_gguf(gguf_file, f"{prefix}.ffn_gate_exps.weight", device=device)
        self.up_exps = GGUFQuantMoEExperts.from_gguf(gguf_file, f"{prefix}.ffn_up_exps.weight", device=device)
        self.down_exps = GGUFQuantMoEExperts.from_gguf(gguf_file, f"{prefix}.ffn_down_exps.weight", device=device)

        shexp_gate_name = f"{prefix}.ffn_gate_shexp.weight"
        if shexp_gate_name in gguf_file.tensors:
            sh_g = GGUFQuantLinear.from_gguf(gguf_file, shexp_gate_name, device=device)
            sh_u = GGUFQuantLinear.from_gguf(gguf_file, f"{prefix}.ffn_up_shexp.weight", device=device)
            self.shared_gate_up: Optional[FusedLinearPack] = FusedLinearPack([sh_g, sh_u])
            del sh_g, sh_u
            self.shared_down: Optional[GGUFQuantLinear] = GGUFQuantLinear.from_gguf(
                gguf_file, f"{prefix}.ffn_down_shexp.weight", device=device
            )
            shexp_inp_name = f"{prefix}.ffn_gate_inp_shexp.weight"
            if shexp_inp_name in gguf_file.tensors:
                sh_inp = load_float_tensor(gguf_file, shexp_inp_name, device=device, dtype=torch.float32).view(1, -1)
                self.register_buffer("shared_inp_gate", sh_inp, persistent=False)
            else:
                self.shared_inp_gate = None
        else:
            self.shared_gate_up = None
            self.shared_down = None
            self.shared_inp_gate = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])  # [M, H]

        router_logits = F.linear(x_2d.float(), self.router_weight)  # [M, E]
        probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_indices = torch.topk(probs, k=self.top_k, dim=-1)  # [M, K]

        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-20)

        out_2d = torch.zeros_like(x_2d, dtype=torch.float32)

        for exp_id in topk_indices.unique().tolist():
            exp_mask = topk_indices == exp_id
            tok_idx, slot_idx = torch.where(exp_mask)
            x_e = x_2d.index_select(0, tok_idx)

            g_e = self.gate_exps.forward_expert(exp_id, x_e)
            u_e = self.up_exps.forward_expert(exp_id, x_e)
            down_in_e, row_scale_e = exec_fused_act_mul_rowscale(g_e, u_e, use_gelu=self.use_gelu)
            d_e = self.down_exps.forward_expert(exp_id, down_in_e.to(x_e.dtype)).float() * row_scale_e

            w_e = topk_weights[tok_idx, slot_idx].unsqueeze(-1)
            out_2d.index_add_(0, tok_idx, d_e * w_e)

        if self.shared_gate_up is not None and self.shared_down is not None:
            if self.shared_gate_up.is_single_contiguous:
                sh_gu = self.shared_gate_up.forward_single_concat(x_2d)
                sh_in, sh_row_scale = exec_fused_act_mul_rowscale(sh_gu, None, use_gelu=self.use_gelu)
            else:
                sh_g, sh_u = self.shared_gate_up(x_2d)
                sh_in, sh_row_scale = exec_fused_act_mul_rowscale(sh_g, sh_u, use_gelu=self.use_gelu)
            sh_out = self.shared_down(sh_in.to(x_2d.dtype)).float() * sh_row_scale
            if self.shared_inp_gate is not None:
                sh_scale = torch.sigmoid(F.linear(x_2d.float(), self.shared_inp_gate))
                sh_out = sh_out * sh_scale
            out_2d = out_2d + sh_out

        return out_2d.reshape(orig_shape)


def _resolve_block_norm_keys(gguf_file: GGUFFile, layer_idx: int) -> Tuple[Optional[str], str, Optional[str]]:
    prefix = f"blk.{layer_idx}"
    ffn_norm_key = f"{prefix}.ffn_norm.weight"
    post_attn_key = (
        f"{prefix}.post_attention_norm.weight"
        if f"{prefix}.post_attention_norm.weight" in gguf_file.tensors
        else f"{prefix}.attn_post_norm.weight"
    )
    if ffn_norm_key in gguf_file.tensors:
        attn_post = post_attn_key if post_attn_key in gguf_file.tensors else None
        eff_ffn_norm = ffn_norm_key
    else:
        attn_post = None
        eff_ffn_norm = post_attn_key

    post_ffn_cand = (
        f"{prefix}.post_ffw_norm.weight"
        if f"{prefix}.post_ffw_norm.weight" in gguf_file.tensors
        else f"{prefix}.ffn_post_norm.weight"
    )
    ffn_post = post_ffn_cand if post_ffn_cand in gguf_file.tensors else None
    return attn_post, eff_ffn_norm, ffn_post


class GGUFTransformerBlock(nn.Module):
    def __init__(
        self,
        gguf_file: GGUFFile,
        config: GGUFModelConfig,
        layer_idx: int,
        rope: RotaryEmbeddingCache,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.device = device
        prefix = f"blk.{layer_idx}"

        self.attn_norm = GGUFRMSNorm(
            load_float_tensor(gguf_file, f"{prefix}.attn_norm.weight", device=device),
            config.rms_norm_eps,
        )

        if f"{prefix}.ssm_out.weight" in gguf_file.tensors:
            self.is_recurrent = True
            self.attn: nn.Module = GGUFGatedDeltaNet.from_gguf(gguf_file, config, layer_idx, device=device)
        else:
            self.is_recurrent = False
            self.attn = GGUFAttention.from_gguf(gguf_file, config, layer_idx, rope, device=device)

        attn_post_key, eff_ffn_key, ffn_post_key = _resolve_block_norm_keys(gguf_file, layer_idx)
        self.attn_post_norm = (
            GGUFRMSNorm(load_float_tensor(gguf_file, attn_post_key, device=device), config.rms_norm_eps)
            if attn_post_key is not None
            else None
        )
        self.ffn_norm = GGUFRMSNorm(
            load_float_tensor(gguf_file, eff_ffn_key, device=device),
            config.rms_norm_eps,
        )

        if f"{prefix}.ffn_gate_exps.weight" in gguf_file.tensors:
            self.mlp: nn.Module = GGUFSparseMoEMLP(gguf_file, config, layer_idx, device=device)
        else:
            self.mlp = GGUFDenseMLP.from_gguf(gguf_file, config, layer_idx, device=device)

        self.ffn_post_norm = (
            GGUFRMSNorm(load_float_tensor(gguf_file, ffn_post_key, device=device), config.rms_norm_eps)
            if ffn_post_key is not None
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        start_pos: int,
        rec_state: Optional[LayerRecurrentState] = None,
        compute_dtype: torch.dtype = torch.float16,
        pos_tensor: Optional[torch.Tensor] = None,
        kv_pos: Optional[int] = None,
    ) -> torch.Tensor:
        if x.device != self.device:
            x = x.to(self.device, non_blocking=True)

        with cuda_device_scope(self.device):
            normed_attn = self.attn_norm(x, out_dtype=compute_dtype)
            if self.is_recurrent:
                assert rec_state is not None
                attn_delta = self.attn(normed_attn, rec_state).float()
            else:
                assert kv_cache is not None
                attn_delta = self.attn(
                    normed_attn, kv_cache, start_pos, pos_tensor=pos_tensor, kv_pos=kv_pos
                ).float()

            if self.attn_post_norm is not None:
                attn_delta = self.attn_post_norm(attn_delta, out_dtype=torch.float32)
            h, normed_ffn = self.ffn_norm.forward_with_residual(x, attn_delta, out_dtype=compute_dtype)

            ffn_delta = self.mlp(normed_ffn).float()
            if self.ffn_post_norm is not None:
                ffn_delta = self.ffn_post_norm(ffn_delta, out_dtype=torch.float32)
            return h + ffn_delta


class TPTransformerBlock(nn.Module):
    """
    2-GPU (or N-GPU) Tensor-Parallel Transformer Block:
    Enqueues Attention / Gated-DeltaNet and SwiGLU MLP shards across all TP GPUs asynchronously
    so BOTH Tesla T4 GDDR6 memory buses (`2 x 320 = 640 GB/s`) stream weights simultaneously!
    """

    def __init__(
        self,
        gguf_file: GGUFFile,
        config: GGUFModelConfig,
        layer_idx: int,
        rope: RotaryEmbeddingCache,
        tp_devices: Sequence[torch.device],
    ) -> None:
        super().__init__()
        self.tp_devices = list(tp_devices)
        self.primary_device = self.tp_devices[0]
        prefix = f"blk.{layer_idx}"

        attn_nw_cpu = load_float_tensor(gguf_file, f"{prefix}.attn_norm.weight", device=torch.device("cpu"))
        self.attn_norm = GGUFRMSNorm(attn_nw_cpu.to(self.primary_device), config.rms_norm_eps)

        if f"{prefix}.ssm_out.weight" in gguf_file.tensors:
            self.is_recurrent = True
            self.attn_shards = nn.ModuleList(
                GGUFGatedDeltaNet.from_gguf_tp_shards(gguf_file, config, layer_idx, self.tp_devices)
            )
        else:
            self.is_recurrent = False
            self.attn_shards = nn.ModuleList(
                GGUFAttention.from_gguf_tp_shards(gguf_file, config, layer_idx, rope, self.tp_devices)
            )

        attn_post_key, eff_ffn_key, ffn_post_key = _resolve_block_norm_keys(gguf_file, layer_idx)
        self.attn_post_norm = (
            GGUFRMSNorm(
                load_float_tensor(gguf_file, attn_post_key, device=self.primary_device),
                config.rms_norm_eps,
            )
            if attn_post_key is not None
            else None
        )
        self.ffn_norm = GGUFRMSNorm(
            load_float_tensor(gguf_file, eff_ffn_key, device=self.primary_device),
            config.rms_norm_eps,
        )
        self.mlp_shards = nn.ModuleList(
            GGUFDenseMLP.from_gguf_tp_shards(gguf_file, config, layer_idx, self.tp_devices)
        )
        self.ffn_post_norm = (
            GGUFRMSNorm(
                load_float_tensor(gguf_file, ffn_post_key, device=self.primary_device),
                config.rms_norm_eps,
            )
            if ffn_post_key is not None
            else None
        )

    def forward_tp(
        self,
        x: torch.Tensor,
        tp_kv_caches: Optional[List[LayerKVCache]],
        start_pos: int,
        tp_rec_states: Optional[List[LayerRecurrentState]] = None,
        compute_dtype: torch.dtype = torch.float16,
        kv_pos: Optional[int] = None,
    ) -> torch.Tensor:
        dev0 = self.primary_device
        if x.device != dev0:
            x = x.to(dev0, non_blocking=True)

        with cuda_device_scope(dev0):
            normed_0 = self.attn_norm(x, out_dtype=compute_dtype)

        # Broadcast tiny FP16 activation (`~10 KB` at S=1) to peer GPUs, then enqueue ALL shards asynchronously!
        normed_inputs = [normed_0] + [normed_0.to(dev, non_blocking=True) for dev in self.tp_devices[1:]]

        attn_outs: List[torch.Tensor] = []
        for r, dev in enumerate(self.tp_devices):
            with cuda_device_scope(dev):
                if self.is_recurrent:
                    assert tp_rec_states is not None
                    attn_outs.append(self.attn_shards[r](normed_inputs[r], tp_rec_states[r]))
                else:
                    assert tp_kv_caches is not None
                    attn_outs.append(self.attn_shards[r](normed_inputs[r], tp_kv_caches[r], start_pos, kv_pos=kv_pos))

        with cuda_device_scope(dev0):
            attn_delta = attn_outs[0].float()
            for part in attn_outs[1:]:
                attn_delta = attn_delta + part.to(dev0, non_blocking=True).float()
            if self.attn_post_norm is not None:
                attn_delta = self.attn_post_norm(attn_delta, out_dtype=torch.float32)
            h, normed_ffn_0 = self.ffn_norm.forward_with_residual(x, attn_delta, out_dtype=compute_dtype)

        ffn_inputs = [normed_ffn_0] + [normed_ffn_0.to(dev, non_blocking=True) for dev in self.tp_devices[1:]]
        mlp_outs: List[torch.Tensor] = []
        for r, dev in enumerate(self.tp_devices):
            with cuda_device_scope(dev):
                mlp_outs.append(self.mlp_shards[r](ffn_inputs[r]))

        with cuda_device_scope(dev0):
            ffn_delta = mlp_outs[0].float()
            for part in mlp_outs[1:]:
                ffn_delta = ffn_delta + part.to(dev0, non_blocking=True).float()
            if self.ffn_post_norm is not None:
                ffn_delta = self.ffn_post_norm(ffn_delta, out_dtype=torch.float32)
            return h + ffn_delta


def _has_complete_mtp_block(gguf_file: GGUFFile, config: GGUFModelConfig) -> bool:
    """Return `True` iff GGUF contains a complete `blk.{n_layers}.nextn.*` MTP draft block."""
    if config.nextn_predict_layers < 1:
        return False
    p = f"blk.{config.n_layers}"
    t = gguf_file.tensors
    has_nextn = (
        f"{p}.nextn.eh_proj.weight" in t
        and f"{p}.nextn.enorm.weight" in t
        and f"{p}.nextn.hnorm.weight" in t
    )
    has_attn = f"{p}.attn_norm.weight" in t and (
        f"{p}.attn_qkv.weight" in t or f"{p}.attn_q.weight" in t
    )
    has_ffn = f"{p}.ffn_gate.weight" in t or f"{p}.ffn_gate_exps.weight" in t
    return bool(has_nextn and has_attn and has_ffn)


def _can_tp_block(gguf_file: GGUFFile, config: GGUFModelConfig, lid: int, tp_world: int) -> bool:
    p = f"blk.{lid}"
    if f"{p}.ffn_gate_exps.weight" in gguf_file.tensors:
        return False
    inter = int(gguf_file.tensors[f"{p}.ffn_gate.weight"].shape[0])
    if inter % tp_world != 0 or (inter // tp_world) % 32 != 0:
        return False
    if f"{p}.ssm_out.weight" in gguf_file.tensors:
        d_inner = int(gguf_file.tensors[f"{p}.attn_gate.weight"].shape[0])
        num_v_heads = int(gguf_file.tensors[f"{p}.ssm_beta.weight"].shape[0])
        conv_dim = int(gguf_file.tensors[f"{p}.attn_qkv.weight"].shape[0])
        key_dim = (conv_dim - d_inner) // 2
        head_v_dim = d_inner // num_v_heads
        head_k_dim = config.ssm_state_size if config.ssm_state_size > 0 else head_v_dim
        num_k_heads = key_dim // head_k_dim
        if num_k_heads % tp_world != 0 or num_v_heads % num_k_heads != 0 or head_v_dim % 32 != 0:
            return False
    return True


def _can_use_tensor_parallel(
    gguf_file: GGUFFile,
    config: GGUFModelConfig,
    tp_world: int,
) -> bool:
    """Verify exact group-aligned divisibility across all transformer blocks before enabling TP."""
    if tp_world < 2:
        return False
    if config.n_heads % tp_world != 0 or config.n_kv_heads % tp_world != 0:
        return False
    if ((config.n_heads // tp_world) * config.head_dim) % 32 != 0:
        return False

    for lid in range(config.n_layers):
        if not _can_tp_block(gguf_file, config, lid, tp_world):
            return False
    return True


class GGUFMTPDraftHead(nn.Module):
    """
    Native embedded GGUF Multi-Token Prediction (`blk.{n_layers}.nextn.*` / `llama.cpp` `graph_mtp`) head:
      `[enorm(embed(tok)), hnorm(h_prev)] -> eh_proj -> Full-Attention Decoder Block -> shared_head_norm -> lm_head`
    Supports both single-GPU/`PP` (`GGUFTransformerBlock`, dense or MoE) and `TP=2` (`TPTransformerBlock`)!
    """

    def __init__(
        self,
        gguf_file: GGUFFile,
        config: GGUFModelConfig,
        layer_idx: int,
        rope: RotaryEmbeddingCache,
        device: torch.device,
        tp_devices: Optional[Sequence[torch.device]] = None,
        compute_dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.compute_dtype = compute_dtype
        p = f"blk.{layer_idx}"

        tp_list = list(tp_devices) if tp_devices and len(tp_devices) > 1 and _can_tp_block(gguf_file, config, layer_idx, len(tp_devices)) else []
        self.is_tensor_parallel = bool(tp_list)
        self.tp_devices = tp_list
        self.device = tp_list[0] if tp_list else device

        self.enorm = GGUFRMSNorm(
            load_float_tensor(gguf_file, f"{p}.nextn.enorm.weight", device=self.device),
            config.rms_norm_eps,
        )
        self.hnorm = GGUFRMSNorm(
            load_float_tensor(gguf_file, f"{p}.nextn.hnorm.weight", device=self.device),
            config.rms_norm_eps,
        )

        tp_w = len(self.tp_devices)
        eh_rows = int(gguf_file.tensors[f"{p}.nextn.eh_proj.weight"].shape[0])
        stage_eh = self.is_tensor_parallel and eh_rows % tp_w == 0 and (eh_rows // tp_w) % 32 == 0
        eh_full = GGUFQuantLinear.from_gguf(
            gguf_file, f"{p}.nextn.eh_proj.weight",
            device=torch.device("cpu") if stage_eh else self.device,
        )
        if self.is_tensor_parallel and eh_full.out_features % tp_w == 0 and (eh_full.out_features // tp_w) % 32 == 0:
            chunk = eh_full.out_features // tp_w
            self.eh_proj_shards = nn.ModuleList(
                [eh_full.slice_output_rows(slice(r * chunk, (r + 1) * chunk), self.tp_devices[r]) for r in range(tp_w)]
            )
            self.eh_proj: Optional[GGUFQuantLinear] = None
            del eh_full
        else:
            self.eh_proj_shards = nn.ModuleList()
            self.eh_proj = eh_full

        self.own_embed: Optional[GGUFQuantLinear] = (
            GGUFQuantLinear.from_gguf(gguf_file, f"{p}.nextn.embed_tokens.weight", device=self.device)
            if f"{p}.nextn.embed_tokens.weight" in gguf_file.tensors
            else None
        )

        if self.is_tensor_parallel:
            self.block: nn.Module = TPTransformerBlock(
                gguf_file=gguf_file,
                config=config,
                layer_idx=layer_idx,
                rope=rope,
                tp_devices=self.tp_devices,
            )
        else:
            self.block = GGUFTransformerBlock(
                gguf_file=gguf_file,
                config=config,
                layer_idx=layer_idx,
                rope=rope,
                device=self.device,
            )

        self.shared_head_norm: Optional[GGUFRMSNorm] = (
            GGUFRMSNorm(
                load_float_tensor(gguf_file, f"{p}.nextn.shared_head_norm.weight", device=self.device),
                config.rms_norm_eps,
            )
            if f"{p}.nextn.shared_head_norm.weight" in gguf_file.tensors
            else None
        )
        self.own_lm_head: Optional[GGUFQuantLinear] = (
            GGUFQuantLinear.from_gguf(gguf_file, f"{p}.nextn.shared_head_head.weight", device=self.device)
            if f"{p}.nextn.shared_head_head.weight" in gguf_file.tensors
            else None
        )

    def _project_eh(self, cat_eh: torch.Tensor) -> torch.Tensor:
        if len(self.eh_proj_shards) > 0:
            eh_inputs = [cat_eh] + [cat_eh.to(dev, non_blocking=True) for dev in self.tp_devices[1:]]
            parts: List[torch.Tensor] = []
            for r, dev in enumerate(self.tp_devices):
                with cuda_device_scope(dev):
                    parts.append(self.eh_proj_shards[r](eh_inputs[r]))
            with cuda_device_scope(self.device):
                return torch.cat([pt.to(self.device, non_blocking=True) for pt in parts], dim=-1).float()
        assert self.eh_proj is not None
        with cuda_device_scope(self.device):
            return self.eh_proj(cat_eh).float()

    def _forward_block(
        self,
        token_ids: torch.Tensor,
        h_prev: torch.Tensor,
        kv_manager: EngineKVCacheManager,
        start_pos: int,
        trunk_token_embd: GGUFQuantLinear,
    ) -> torch.Tensor:
        if self.own_embed is not None:
            with cuda_device_scope(self.device):
                emb = self.own_embed.embed_lookup(token_ids.to(self.device), dtype=torch.float32)
        else:
            emb_dev = trunk_token_embd.scales.device
            with cuda_device_scope(emb_dev):
                emb = trunk_token_embd.embed_lookup(token_ids.to(emb_dev), dtype=torch.float32)
            if emb.device != self.device:
                emb = emb.to(self.device, non_blocking=True)

        with cuda_device_scope(self.device):
            if self.config.embed_scale != 1.0:
                emb = emb * self.config.embed_scale
            e_norm = self.enorm(emb, out_dtype=self.compute_dtype)
            h_norm = self.hnorm(h_prev.to(self.device, non_blocking=True), out_dtype=self.compute_dtype)
            cat_eh = torch.cat([e_norm, h_norm], dim=-1)

        x = self._project_eh(cat_eh)
        eff_kv_pos = start_pos - kv_manager.mtp_cache_start

        if self.is_tensor_parallel:
            return self.block.forward_tp(  # type: ignore[operator]
                x,
                kv_manager.mtp_tp_kv_caches,
                start_pos,
                None,
                self.compute_dtype,
                kv_pos=eff_kv_pos,
            )
        assert kv_manager.mtp_kv_cache is not None
        return self.block(
            x,
            kv_manager.mtp_kv_cache,
            start_pos,
            None,
            self.compute_dtype,
            kv_pos=eff_kv_pos,
        )

    @torch.inference_mode()
    def prefill_catch_up(
        self,
        prompt_ids: torch.Tensor,
        trunk_h_norm: torch.Tensor,
        kv_manager: EngineKVCacheManager,
        start_pos: int,
        trunk_token_embd: GGUFQuantLinear,
        prev_h: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Populate MTP full-attention KV cache over `prompt_ids` `[1, S]` at `[start_pos .. start_pos + S)`
        using `[prev_h (or 0), trunk_h_norm[:-1]]`, skipping `lm_head` so prefill overhead is `< 1.5%`!
        When `kv_manager.mtp_kv_window` (`W`, e.g. `4096`) is set and `S > W`, automatically rebases
        `mtp_cache_start` and forwards ONLY the trailing `W` tokens (`O(W)` time & VRAM even on 256K prompts!).
        """
        seq_len = int(prompt_ids.shape[1])
        if seq_len <= 0:
            return
        win = kv_manager.mtp_kv_window
        with cuda_device_scope(self.device):
            if win is not None and seq_len > win:
                skip = seq_len - win
                new_abs_start = start_pos + skip
                kv_manager.reset_mtp_cache(new_abs_start)
                prompt_ids = prompt_ids[:, skip:]
                h_in = trunk_h_norm[:, skip - 1 : seq_len - 1, :].to(self.device, non_blocking=True)
                start_pos = new_abs_start
                seq_len = win
            else:
                th = trunk_h_norm.to(self.device, non_blocking=True)
                if prev_h is not None:
                    first_h = prev_h.to(device=self.device, dtype=th.dtype, non_blocking=True).view(1, 1, -1)
                else:
                    first_h = torch.zeros((1, 1, self.config.hidden_size), device=self.device, dtype=th.dtype)
                h_in = torch.cat([first_h, th[:, :-1, :]], dim=1) if seq_len > 1 else first_h

        prefill_chunk = min(2048, win) if (win is not None and win > 0) else 2048
        for off in range(0, seq_len, prefill_chunk):
            end = min(seq_len, off + prefill_chunk)
            _ = self._forward_block(
                prompt_ids[:, off:end],
                h_in[:, off:end, :],
                kv_manager,
                start_pos + off,
                trunk_token_embd,
            )
        kv_manager.mtp_exact_kv_len = start_pos + seq_len

    @torch.inference_mode()
    def forward_step(
        self,
        token_ids: torch.Tensor,
        h_prev: torch.Tensor,
        kv_manager: EngineKVCacheManager,
        start_pos: int,
        trunk_model: "GGUFTransformerModel",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fused catch-up + draft step: runs MTP block over `S >= 1` rows (`[catchup_rows..., seed_row]`),
        applies `shared_head_norm` + `lm_head` ONLY on the final row, and returns `(logits_f32[1,1,V], h_nextn[1,1,H])`!
        """
        block_out = self._forward_block(
            token_ids,
            h_prev,
            kv_manager,
            start_pos,
            trunk_model.token_embd,
        )
        last_raw = block_out[:, -1:, :]
        if self.shared_head_norm is not None:
            with cuda_device_scope(self.device):
                h_nextn = self.shared_head_norm(last_raw.to(self.device), out_dtype=self.compute_dtype)
        else:
            norm_dev = trunk_model.first_device if trunk_model.is_tensor_parallel else trunk_model.last_device
            with cuda_device_scope(norm_dev):
                h_nextn = trunk_model.output_norm(last_raw.to(norm_dev), out_dtype=self.compute_dtype)

        if self.own_lm_head is not None:
            with cuda_device_scope(self.device):
                logits_f32 = self.own_lm_head(h_nextn.to(self.device)).float()
                if self.config.final_logit_softcapping > 0.0:
                    cap = self.config.final_logit_softcapping
                    logits_f32 = torch.tanh(logits_f32 / cap) * cap
            return logits_f32, h_nextn
        return trunk_model.project_normed_hidden(h_nextn), h_nextn


def resolve_device_list(devices: Optional[Sequence[str]] = None) -> List[torch.device]:
    if devices is not None and len(devices) > 0:
        return [torch.device(d) for d in devices]
    if torch.cuda.is_available():
        return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    return [torch.device("cpu")]


def compute_layer_device_map(
    n_layers: int,
    devices: Optional[Sequence[str]] = None,
    gguf_file: Optional[GGUFFile] = None,
    reserve_mtp_on_last: bool = False,
) -> List[torch.device]:
    """
    Distribute transformer blocks across all available GPUs (e.g., `cuda:0` and `cuda:1` on Kaggle T4 x2)
    accounting for exact per-block GGUF tensor sizes + `token_embd.weight` (GPU 0) + `output.weight` / MTP (last GPU),
    so free VRAM headroom is balanced evenly across GPUs!
    """
    dev_list = resolve_device_list(devices)
    if len(dev_list) == 1 or n_layers <= 1:
        return [dev_list[0]] * n_layers

    n_devs = len(dev_list)
    free_bytes: List[float] = []
    for d in dev_list:
        if d.type == "cuda":
            free_b, _ = torch.cuda.mem_get_info(d)
            free_bytes.append(max(float(free_b), 1.0))
        else:
            free_bytes.append(16.0 * (1024**3))

    if gguf_file is not None:
        soa_factor = 1.12
        embd_b = float(gguf_file.tensors["token_embd.weight"].n_bytes) * soa_factor if "token_embd.weight" in gguf_file.tensors else 0.0
        head_b = float(gguf_file.tensors["output.weight"].n_bytes) * soa_factor if "output.weight" in gguf_file.tensors else 0.0
        mtp_b = 0.0

        layer_bytes = [0.0] * n_layers
        for t_name, t_info in gguf_file.tensors.items():
            if t_name.startswith("blk."):
                parts = t_name.split(".", 2)
                if len(parts) >= 2 and parts[1].isdigit():
                    lid = int(parts[1])
                    if 0 <= lid < n_layers:
                        layer_bytes[lid] += float(t_info.n_bytes) * soa_factor
                    elif reserve_mtp_on_last and lid == n_layers:
                        mtp_b += float(t_info.n_bytes) * soa_factor

        avail = list(free_bytes)
        avail[0] = max(1.0, avail[0] - embd_b)
        avail[-1] = max(1.0, avail[-1] - head_b - mtp_b)

        total_avail = sum(avail)
        total_layer_b = max(sum(layer_bytes), 1.0)
        targets = [total_layer_b * (a / total_avail) for a in avail]

        mapping: List[torch.device] = []
        dev_idx = 0
        acc_b = 0.0
        for lid in range(n_layers):
            remaining_layers = n_layers - lid
            remaining_devs = n_devs - 1 - dev_idx
            if (
                dev_idx < n_devs - 1
                and len(mapping) > 0
                and (acc_b + 0.5 * layer_bytes[lid] > targets[dev_idx])
                and remaining_layers >= remaining_devs
            ):
                dev_idx += 1
                acc_b = 0.0
            if remaining_layers == remaining_devs and dev_idx < n_devs - 1:
                dev_idx += 1
                acc_b = 0.0
            mapping.append(dev_list[dev_idx])
            acc_b += layer_bytes[lid]
        return mapping

    free_bytes[0] *= 0.94
    total_w = sum(free_bytes)
    counts = [max(1, round(n_layers * (w / total_w))) for w in free_bytes]
    while sum(counts) > n_layers:
        counts[counts.index(max(counts))] -= 1
    while sum(counts) < n_layers:
        counts[counts.index(min(counts))] += 1

    mapping = []
    for dev, cnt in zip(dev_list, counts):
        mapping.extend([dev] * cnt)
    return mapping


class GGUFTransformerModel(nn.Module):
    """
    End-to-end GGUF Causal LM supporting both:
    - Multi-GPU Tensor Parallelism (`TP=2`, streaming weights from all GPUs in parallel!), and
    - Pipeline Parallelism (`PP`) + Per-Stage `torch.cuda.CUDAGraph` single-token decode replay!
    """

    def __init__(
        self,
        gguf_file: GGUFFile,
        config: GGUFModelConfig,
        devices: Optional[Sequence[str]] = None,
        compute_dtype: torch.dtype = torch.float16,
        parallel_mode: Literal["auto", "tp", "pp"] = "auto",
        enable_cuda_graph: bool = True,
        load_mtp: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.compute_dtype = compute_dtype
        dev_list = resolve_device_list(devices)

        want_tp = False
        if parallel_mode == "tp" and len(dev_list) >= 2:
            want_tp = _can_use_tensor_parallel(gguf_file, config, len(dev_list))
        elif parallel_mode == "auto" and len(dev_list) >= 2 and all(d.type == "cuda" for d in dev_list) and not config.is_moe:
            want_tp = _can_use_tensor_parallel(gguf_file, config, len(dev_list))

        self.is_tensor_parallel = want_tp
        self.tp_devices: List[torch.device] = list(dev_list) if want_tp else []
        self.enable_cuda_graph = bool(enable_cuda_graph and not self.is_tensor_parallel and not config.is_moe)
        has_mtp_in_gguf = bool(load_mtp and _has_complete_mtp_block(gguf_file, config))

        self.rope = RotaryEmbeddingCache(
            rope_dim=config.rope_dim,
            rope_theta=config.rope_theta,
            neox_style=config.rope_neox,
            rope_scaling_type=config.rope_scaling_type,
            rope_scaling_factor=config.rope_scaling_factor,
            rope_original_context_len=config.rope_original_context_len,
            rope_yarn_beta_fast=config.rope_yarn_beta_fast,
            rope_yarn_beta_slow=config.rope_yarn_beta_slow,
            rope_yarn_ext_factor=config.rope_yarn_ext_factor,
            rope_yarn_attn_factor=config.rope_yarn_attn_factor,
            rope_yarn_log_mul=config.rope_yarn_log_mul,
            max_context_len=config.max_context_len,
        )

        load_start = time.monotonic()
        def report_load(message: str) -> None:
            print(f"[PyGGUF] Loading {message} ({time.monotonic() - load_start:.1f}s elapsed)", flush=True)

        if self.is_tensor_parallel:
            self.first_device = self.tp_devices[0]
            self.last_device = self.tp_devices[0]
            self.layer_devices = [self.first_device] * config.n_layers

            report_load("token embeddings")
            self.token_embd = GGUFQuantLinear.from_gguf(gguf_file, "token_embd.weight", device=self.first_device)
            def load_tp_block(i: int) -> TPTransformerBlock:
                report_load(f"layer {i + 1}/{config.n_layers} (TP)")
                return TPTransformerBlock(
                        gguf_file=gguf_file,
                        config=config,
                        layer_idx=i,
                        rope=self.rope,
                        tp_devices=self.tp_devices,
                    )
            self.layers = nn.ModuleList([load_tp_block(i) for i in range(config.n_layers)])
            self.recurrent_layer_flags: List[bool] = [bool(blk.is_recurrent) for blk in self.layers]
            self.has_recurrent_layers: bool = any(self.recurrent_layer_flags)

            self.output_norm = GGUFRMSNorm(
                load_float_tensor(gguf_file, "output_norm.weight", device=self.first_device),
                config.rms_norm_eps,
            )

            # Shard vocab projection across all TP GPUs so 248K-vocab `lm_head` also runs at 2x bandwidth!
            tp_w = len(self.tp_devices)
            report_load("output head")
            if "output.weight" in gguf_file.tensors:
                head_full = GGUFQuantLinear.from_gguf(gguf_file, "output.weight", device=torch.device("cpu"))
                self.tied_embeddings = False
            else:
                head_full = self.token_embd
                self.tied_embeddings = True

            vocab = head_full.out_features
            v_chunk = (vocab + tp_w - 1) // tp_w
            self.lm_head_shards = nn.ModuleList(
                [
                    head_full.slice_output_rows(slice(r * v_chunk, min(vocab, (r + 1) * v_chunk)), self.tp_devices[r])
                    for r in range(tp_w)
                ]
            )
            self.lm_head: Optional[GGUFQuantLinear] = self.lm_head_shards[0]
            if not self.tied_embeddings:
                del head_full
        else:
            self.layer_devices = compute_layer_device_map(
                config.n_layers,
                devices,
                gguf_file=gguf_file,
                reserve_mtp_on_last=has_mtp_in_gguf,
            )
            self.first_device = self.layer_devices[0]
            self.last_device = self.layer_devices[-1]

            report_load("token embeddings")
            self.token_embd = GGUFQuantLinear.from_gguf(gguf_file, "token_embd.weight", device=self.first_device)
            def load_pp_block(i: int) -> GGUFTransformerBlock:
                report_load(f"layer {i + 1}/{config.n_layers} ({self.layer_devices[i]})")
                return GGUFTransformerBlock(
                        gguf_file=gguf_file,
                        config=config,
                        layer_idx=i,
                        rope=self.rope,
                        device=self.layer_devices[i],
                    )
            self.layers = nn.ModuleList([load_pp_block(i) for i in range(config.n_layers)])
            self.recurrent_layer_flags = [bool(blk.is_recurrent) for blk in self.layers]
            self.has_recurrent_layers = any(self.recurrent_layer_flags)

            self.output_norm = GGUFRMSNorm(
                load_float_tensor(gguf_file, "output_norm.weight", device=self.last_device),
                config.rms_norm_eps,
            )

            self.lm_head_shards = nn.ModuleList()
            report_load("output head")
            if "output.weight" in gguf_file.tensors:
                self.lm_head = GGUFQuantLinear.from_gguf(gguf_file, "output.weight", device=self.last_device)
                self.tied_embeddings = False
            else:
                self.lm_head = None
                self.tied_embeddings = True

        if has_mtp_in_gguf:
            report_load("MTP draft head")
        self.mtp_head: Optional[GGUFMTPDraftHead] = (
            GGUFMTPDraftHead(
                gguf_file=gguf_file,
                config=config,
                layer_idx=config.n_layers,
                rope=self.rope,
                device=self.last_device,
                tp_devices=self.tp_devices if self.is_tensor_parallel else None,
                compute_dtype=compute_dtype,
            )
            if has_mtp_in_gguf
            else None
        )

        # Per-GPU-stage CUDA Graph state for M=1 decode (on PP / 1-GPU CUDA runs)
        self._decode_graphs_ready = False
        self._decode_graph_caps: Tuple[int, ...] = ()
        self._stage_ranges: List[Tuple[torch.device, int, int]] = []
        self._stage_graphs: List[Any] = []
        self._stage_in_bufs: List[torch.Tensor] = []
        self._stage_out_bufs: List[torch.Tensor] = []
        self._stage_pos_bufs: Dict[torch.device, torch.Tensor] = {}
        self._graph_token_buf: Optional[torch.Tensor] = None
        self._graph_logits_buf: Optional[torch.Tensor] = None
        self._graph_normed_buf: Optional[torch.Tensor] = None
        self._head_graph: Any = None
        self._head_in_buf: Optional[torch.Tensor] = None

    def _compute_pp_stages(self) -> List[Tuple[torch.device, int, int]]:
        stages: List[Tuple[torch.device, int, int]] = []
        if not self.layer_devices:
            return stages
        cur_dev = self.layer_devices[0]
        s_start = 0
        for idx in range(1, len(self.layer_devices)):
            if self.layer_devices[idx] != cur_dev:
                stages.append((cur_dev, s_start, idx))
                cur_dev = self.layer_devices[idx]
                s_start = idx
        stages.append((cur_dev, s_start, len(self.layer_devices)))
        return stages

    def _capture_decode_graphs(self, kv_manager: EngineKVCacheManager) -> None:
        """
        Capture per-GPU `torch.cuda.CUDAGraph`s for `M=1` decode.
        Each stage graph runs strictly on a single CUDA device so non-P2P Dual-T4 hosts never trigger
        forbidden cross-device stream syncs during graph capture!
        """
        stages = self._compute_pp_stages()
        self._stage_ranges = stages
        self._stage_graphs = []
        self._stage_in_bufs = []
        self._stage_out_bufs = []
        self._stage_pos_bufs = {}

        # Snapshot live recurrent & KV state so warmup/capture leaves model state 100% untouched!
        rec_snap = kv_manager.snapshot_recurrent_gpu() if kv_manager.has_recurrent else {}
        saved_lens = [lkv.seq_len for lkv in kv_manager._iter_all_kv_caches()]

        # Ensure RoPE & recurrent buffers exist on every stage GPU before capture
        unique_devs = {dev for dev, _, _ in stages} | {self.first_device, self.last_device}
        for dev in unique_devs:
            self.rope.ensure_capacity(dev, self.config.max_context_len)
            with cuda_device_scope(dev):
                self._stage_pos_bufs[dev] = torch.zeros((1,), device=dev, dtype=torch.int32)

        for idx, blk in enumerate(self.layers):
            if blk.is_recurrent:
                st = kv_manager.recurrent_states[idx]
                assert st is not None
                dn: GGUFGatedDeltaNet = blk.attn  # type: ignore[assignment]
                st.ensure_allocated(dn.conv_dim, dn.conv_kernel, dn.num_v_heads, dn.head_v_dim, dn.head_k_dim)
            else:
                lkv = kv_manager.layers[idx]
                assert lkv is not None
                if lkv._part_out_scratch is None:
                    with cuda_device_scope(lkv.device):
                        n_h = blk.attn.n_heads  # type: ignore[attr-defined]
                        lkv._part_out_scratch = torch.empty((n_h, 32, lkv.head_dim), device=lkv.device, dtype=torch.float32)
                        lkv._part_lse_scratch = torch.empty((n_h, 32), device=lkv.device, dtype=torch.float32)

        with cuda_device_scope(self.first_device):
            self._graph_token_buf = torch.zeros((1, 1), device=self.first_device, dtype=torch.long)

        for stage_idx, (dev, l0, l1) in enumerate(stages):
            pos_buf = self._stage_pos_bufs[dev]
            with cuda_device_scope(dev):
                in_buf = torch.zeros((1, 1, self.config.hidden_size), device=dev, dtype=torch.float32)

                def _run_stage(
                    _s_idx: int = stage_idx,
                    _l0: int = l0,
                    _l1: int = l1,
                    _in_b: torch.Tensor = in_buf,
                    _pos_b: torch.Tensor = pos_buf,
                ) -> torch.Tensor:
                    if _s_idx == 0:
                        assert self._graph_token_buf is not None
                        h_cur = self.token_embd.embed_lookup(self._graph_token_buf, dtype=torch.float32)
                        if self.config.embed_scale != 1.0:
                            h_cur = h_cur * self.config.embed_scale
                    else:
                        h_cur = _in_b
                    for b_i in range(_l0, _l1):
                        h_cur = self.layers[b_i](
                            h_cur,
                            kv_manager.layers[b_i],
                            0,
                            kv_manager.recurrent_states[b_i],
                            self.compute_dtype,
                            pos_tensor=_pos_b,
                        )
                    return h_cur

                # Warmup on side stream before capture
                s_warm = torch.cuda.Stream(device=dev)
                s_warm.wait_stream(torch.cuda.current_stream(dev))
                with torch.cuda.stream(s_warm):
                    _ = _run_stage()
                torch.cuda.current_stream(dev).wait_stream(s_warm)

                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=s_warm):
                    out_buf = _run_stage()

                self._stage_in_bufs.append(in_buf)
                self._stage_out_bufs.append(out_buf)
                self._stage_graphs.append(g)

        head_dev = self.last_device if self.lm_head is not None else self.first_device
        with cuda_device_scope(head_dev):
            head_in = torch.zeros((1, 1, self.config.hidden_size), device=head_dev, dtype=torch.float32)

            def _run_head(_hin: torch.Tensor = head_in) -> Tuple[torch.Tensor, torch.Tensor]:
                if self.last_device == head_dev:
                    normed = self.output_norm(_hin, out_dtype=self.compute_dtype)
                else:
                    normed = _hin.to(self.compute_dtype)
                proj = self.lm_head(normed) if self.lm_head is not None else self.token_embd(normed)
                lf = proj.float()
                if self.config.final_logit_softcapping > 0.0:
                    cap = self.config.final_logit_softcapping
                    lf = torch.tanh(lf / cap) * cap
                return lf, normed

            sh_warm = torch.cuda.Stream(device=head_dev)
            sh_warm.wait_stream(torch.cuda.current_stream(head_dev))
            with torch.cuda.stream(sh_warm):
                _ = _run_head()
            torch.cuda.current_stream(head_dev).wait_stream(sh_warm)

            gh = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gh, stream=sh_warm):
                logits_buf, normed_buf = _run_head()

            self._head_in_buf = head_in
            self._head_graph = gh
            self._graph_logits_buf = logits_buf
            self._graph_normed_buf = normed_buf

        # Restore live KV lengths & recurrent states untouched
        for lkv, slen in zip(kv_manager._iter_all_kv_caches(), saved_lens):
            lkv.rewind_to(slen)
        if kv_manager.has_recurrent:
            kv_manager.restore_recurrent_gpu(rec_snap, kv_manager.recurrent_seq_len)

        self._decode_graph_caps = tuple(lkv.capacity for lkv in kv_manager._iter_all_kv_caches())
        self._decode_graphs_ready = True

    @torch.inference_mode()
    def forward_decode_step(
        self,
        token_id: int,
        kv_manager: EngineKVCacheManager,
        start_pos: int,
        return_normed_hidden: bool = False,
    ) -> Any:
        """Fast single-token (`M=1`) decode using `torch.cuda.CUDAGraph` replay when supported."""
        if (
            not self.enable_cuda_graph
            or self.first_device.type != "cuda"
            or (self.last_device != self.first_device and self.lm_head is None)
        ):
            step_inp = torch.tensor([[token_id]], device=self.first_device, dtype=torch.long)
            return self.forward(
                step_inp,
                kv_manager,
                start_pos=start_pos,
                logits_to_keep=1,
                return_normed_hidden=return_normed_hidden,
            )

        # Ensure KV cache pages won't reallocate mid-step; re-capture only when page capacity grows (every 2048 toks)
        kv_manager.ensure_all_kv_capacity(start_pos + 1)
        cur_caps = tuple(lkv.capacity for lkv in kv_manager._iter_all_kv_caches())
        if not self._decode_graphs_ready or cur_caps != self._decode_graph_caps:
            try:
                self._capture_decode_graphs(kv_manager)
            except Exception:
                self.enable_cuda_graph = False
                step_inp = torch.tensor([[token_id]], device=self.first_device, dtype=torch.long)
                return self.forward(
                    step_inp,
                    kv_manager,
                    start_pos=start_pos,
                    logits_to_keep=1,
                    return_normed_hidden=return_normed_hidden,
                )

        assert self._graph_token_buf is not None and self._head_in_buf is not None and self._graph_logits_buf is not None
        with cuda_device_scope(self.first_device):
            self._graph_token_buf.fill_(token_id)
        for dev, pbuf in self._stage_pos_bufs.items():
            with cuda_device_scope(dev):
                pbuf.fill_(start_pos)

        for s_idx, (dev, _, _) in enumerate(self._stage_ranges):
            if s_idx > 0:
                self._stage_in_bufs[s_idx].copy_(self._stage_out_bufs[s_idx - 1])
            with cuda_device_scope(dev):
                self._stage_graphs[s_idx].replay()

        for lkv in kv_manager._iter_all_kv_caches():
            lkv.rewind_to(start_pos + 1)

        head_dev = self._head_in_buf.device
        self._head_in_buf.copy_(self._stage_out_bufs[-1])
        with cuda_device_scope(head_dev):
            self._head_graph.replay()
        if return_normed_hidden:
            assert self._graph_normed_buf is not None
            return self._graph_logits_buf, self._graph_normed_buf
        return self._graph_logits_buf

    def _norm_trunk_hidden(self, raw_h: torch.Tensor) -> torch.Tensor:
        target_dev = self.first_device if self.is_tensor_parallel else self.last_device
        with cuda_device_scope(target_dev):
            return self.output_norm(raw_h.to(target_dev), out_dtype=self.compute_dtype)

    def project_normed_hidden(self, h_norm: torch.Tensor) -> torch.Tensor:
        """Project post-`output_norm` hidden states `[1, S, H]` through `lm_head` (TP-sharded or single-GPU)."""
        if self.is_tensor_parallel:
            h0 = h_norm.to(self.first_device, non_blocking=True)
            h_inputs = [h0] + [h0.to(dev, non_blocking=True) for dev in self.tp_devices[1:]]
            logit_parts: List[torch.Tensor] = []
            for r, dev in enumerate(self.tp_devices):
                with cuda_device_scope(dev):
                    logit_parts.append(self.lm_head_shards[r](h_inputs[r]))
            with cuda_device_scope(self.first_device):
                logits = torch.cat([p.to(self.first_device, non_blocking=True) for p in logit_parts], dim=-1)
        else:
            if self.lm_head is not None:
                with cuda_device_scope(self.last_device):
                    logits = self.lm_head(h_norm.to(self.last_device))
            else:
                with cuda_device_scope(self.first_device):
                    logits = self.token_embd(h_norm.to(self.first_device))

        with cuda_device_scope(logits.device):
            logits_f32 = logits.float()
            if self.config.final_logit_softcapping > 0.0:
                cap = self.config.final_logit_softcapping
                logits_f32 = torch.tanh(logits_f32 / cap) * cap
            return logits_f32

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        kv_manager: EngineKVCacheManager,
        start_pos: int = 0,
        logits_to_keep: int = 1,
        return_normed_hidden: bool = False,
    ) -> Any:
        """
        Forward `input_ids` `[1, S]` starting at `start_pos`.
        Slices hidden states to `logits_to_keep` BEFORE projecting to vocab size!
        When `return_normed_hidden=True`, also returns ALL rows' post-`output_norm` hidden states (`t_h_nextn`)
        without computing extra `lm_head` vocab rows!
        """
        with cuda_device_scope(self.first_device):
            ids = input_ids.to(self.first_device)
            h = self.token_embd.embed_lookup(ids, dtype=torch.float32)
            if self.config.embed_scale != 1.0:
                h = h * self.config.embed_scale

        seq_len = h.shape[1]
        prefill_chunk = 2048
        if seq_len > prefill_chunk and logits_to_keep == 1:
            normed_chunks: List[torch.Tensor] = []
            last_h: Optional[torch.Tensor] = None
            for offset in range(0, seq_len, prefill_chunk):
                end = min(seq_len, offset + prefill_chunk)
                sub_h = h[:, offset:end, :]
                for idx, block in enumerate(self.layers):
                    if self.is_tensor_parallel:
                        sub_h = block.forward_tp(  # type: ignore[operator]
                            sub_h,
                            kv_manager.tp_layers[idx],
                            start_pos + offset,
                            kv_manager.tp_recurrent_states[idx],
                            self.compute_dtype,
                        )
                    else:
                        sub_h = block(
                            sub_h,
                            kv_manager.layers[idx],
                            start_pos + offset,
                            kv_manager.recurrent_states[idx],
                            self.compute_dtype,
                        )
                if return_normed_hidden:
                    normed_chunks.append(self._norm_trunk_hidden(sub_h))
                last_h = sub_h[:, -1:, :]
            assert last_h is not None
            if return_normed_hidden:
                h_norm_all = torch.cat(normed_chunks, dim=1)
                logits_f32 = self.project_normed_hidden(h_norm_all[:, -1:, :])
                return logits_f32, h_norm_all
            return self.project_normed_hidden(self._norm_trunk_hidden(last_h))

        for idx, block in enumerate(self.layers):
            if self.is_tensor_parallel:
                h = block.forward_tp(  # type: ignore[operator]
                    h,
                    kv_manager.tp_layers[idx],
                    start_pos,
                    kv_manager.tp_recurrent_states[idx],
                    self.compute_dtype,
                )
            else:
                h = block(
                    h,
                    kv_manager.layers[idx],
                    start_pos,
                    kv_manager.recurrent_states[idx],
                    self.compute_dtype,
                )

        if return_normed_hidden:
            h_norm_all = self._norm_trunk_hidden(h)
            h_for_logits = (
                h_norm_all[:, -logits_to_keep:, :]
                if (logits_to_keep > 0 and h_norm_all.shape[1] > logits_to_keep)
                else h_norm_all
            )
            return self.project_normed_hidden(h_for_logits), h_norm_all

        if logits_to_keep > 0 and h.shape[1] > logits_to_keep:
            h = h[:, -logits_to_keep:, :]
        return self.project_normed_hidden(self._norm_trunk_hidden(h))
