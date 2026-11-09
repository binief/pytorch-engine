"""
Pure-Python zero-copy `mmap` GGUF v2/v3 parser, architecture config extractor,
and embedded GGUF tokenizer / chat-template renderer (0% llama.cpp).
"""

from __future__ import annotations

import json
import math
import mmap
import struct
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np


class GGMLQuantType(IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29
    BF16 = 30
    TQ1_0 = 34
    TQ2_0 = 35
    MXFP4 = 39
    NVFP4 = 40


# (elements_per_block, bytes_per_block)
GGML_BLOCK_INFO: Dict[int, Tuple[int, int]] = {
    GGMLQuantType.F32: (1, 4),
    GGMLQuantType.F16: (1, 2),
    GGMLQuantType.BF16: (1, 2),
    GGMLQuantType.Q4_0: (32, 18),
    GGMLQuantType.Q4_1: (32, 20),
    GGMLQuantType.Q5_0: (32, 22),
    GGMLQuantType.Q5_1: (32, 24),
    GGMLQuantType.Q8_0: (32, 34),
    GGMLQuantType.Q8_1: (32, 40),
    GGMLQuantType.Q2_K: (256, 84),
    GGMLQuantType.Q3_K: (256, 110),
    GGMLQuantType.Q4_K: (256, 144),
    GGMLQuantType.Q5_K: (256, 176),
    GGMLQuantType.Q6_K: (256, 210),
    GGMLQuantType.Q8_K: (256, 292),
    GGMLQuantType.IQ2_XXS: (256, 66),
    GGMLQuantType.IQ2_XS: (256, 74),
    GGMLQuantType.IQ3_XXS: (256, 98),
    GGMLQuantType.IQ1_S: (256, 50),
    GGMLQuantType.IQ4_NL: (32, 18),
    GGMLQuantType.IQ3_S: (256, 110),
    GGMLQuantType.IQ2_S: (256, 82),
    GGMLQuantType.IQ4_XS: (256, 136),
    GGMLQuantType.I8: (1, 1),
    GGMLQuantType.I16: (1, 2),
    GGMLQuantType.I32: (1, 4),
    GGMLQuantType.I64: (1, 8),
    GGMLQuantType.F64: (1, 8),
    GGMLQuantType.IQ1_M: (256, 56),
    GGMLQuantType.TQ1_0: (256, 54),
    GGMLQuantType.TQ2_0: (256, 66),
    GGMLQuantType.MXFP4: (32, 17),
    GGMLQuantType.NVFP4: (64, 36),
}


class _GGUFValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


_SCALAR_STRUCTS: Dict[int, Tuple[str, int, Any]] = {
    _GGUFValueType.UINT8: ("<B", 1, np.uint8),
    _GGUFValueType.INT8: ("<b", 1, np.int8),
    _GGUFValueType.UINT16: ("<H", 2, np.uint16),
    _GGUFValueType.INT16: ("<h", 2, np.int16),
    _GGUFValueType.UINT32: ("<I", 4, np.uint32),
    _GGUFValueType.INT32: ("<i", 4, np.int32),
    _GGUFValueType.FLOAT32: ("<f", 4, np.float32),
    _GGUFValueType.BOOL: ("<?", 1, np.bool_),
    _GGUFValueType.UINT64: ("<Q", 8, np.uint64),
    _GGUFValueType.INT64: ("<q", 8, np.int64),
    _GGUFValueType.FLOAT64: ("<d", 8, np.float64),
}


@dataclass
class GGUFTensorInfo:
    name: str
    ggml_dims: Tuple[int, ...]  # GGML order [ne0, ne1, ...] where ne0 is innermost contiguous
    shape: Tuple[int, ...]      # PyTorch row-major shape (reversed ggml_dims)
    qtype: int
    offset: int                 # Byte offset relative to tensor_data_offset
    n_elements: int
    n_bytes: int

    @property
    def qtype_name(self) -> str:
        try:
            return GGMLQuantType(self.qtype).name
        except ValueError:
            return f"UNKNOWN_{self.qtype}"


@dataclass
class GGUFModelConfig:
    arch: str
    name: str
    vocab_size: int
    n_layers: int
    hidden_size: int
    intermediate_size: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_dim: int = 0
    rope_neox: bool = True
    max_context_len: int = 32768
    # MoE parameters (0 for dense models)
    n_experts: int = 0
    n_experts_used: int = 0
    moe_intermediate_size: int = 0
    shared_expert_intermediate_size: int = 0
    norm_topk_prob: bool = True
    # Hybrid Gated-DeltaNet / SSM & Multi-Token Prediction (Qwen3.5 / Qwen3.6 / Qwen3.8 / Qwen3Next)
    nextn_predict_layers: int = 0
    full_attention_interval: int = 0
    ssm_conv_kernel: int = 0
    ssm_inner_size: int = 0
    ssm_state_size: int = 0
    ssm_dt_rank: int = 0
    ssm_group_count: int = 0
    # Gemma-specific scaling / softcapping
    embed_scale: float = 1.0
    attn_logit_softcapping: float = 0.0
    final_logit_softcapping: float = 0.0
    query_pre_attn_scalar: float = 0.0
    # Special tokens & chat template
    bos_token_id: Optional[int] = None
    eos_token_ids: List[int] = field(default_factory=list)
    chat_template: Optional[str] = None

    @property
    def is_moe(self) -> bool:
        return self.n_experts > 1

    @property
    def is_hybrid_ssm(self) -> bool:
        return self.ssm_inner_size > 0 and self.ssm_state_size > 0


class GGUFFile:
    """
    Zero-copy memory-mapped GGUF v2/v3 file reader in pure Python.
    Streams tensor slices directly from OS page cache without duplicating model memory.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"GGUF file not found: {self.path}")

        self._fh: BinaryIO = open(self.path, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        self._pos = 0

        magic = self._read_bytes(4)
        if magic != b"GGUF":
            self.close()
            raise ValueError(f"Invalid GGUF magic {magic!r} in {self.path}")

        self.version = self._read_scalar("<I", 4)
        if self.version not in (2, 3):
            self.close()
            raise ValueError(f"Unsupported GGUF version {self.version} (expected 2 or 3)")

        self.tensor_count = self._read_scalar("<Q", 8)
        self.kv_count = self._read_scalar("<Q", 8)

        self.metadata: Dict[str, Any] = {}
        for _ in range(self.kv_count):
            key = self._read_string()
            val_type = self._read_scalar("<I", 4)
            self.metadata[key] = self._read_value(val_type)

        self.alignment = int(self.metadata.get("general.alignment", 32))
        self.tensors: Dict[str, GGUFTensorInfo] = {}

        for _ in range(self.tensor_count):
            name = self._read_string()
            n_dims = self._read_scalar("<I", 4)
            ggml_dims = tuple(self._read_scalar("<Q", 8) for _ in range(n_dims))
            qtype = self._read_scalar("<I", 4)
            rel_offset = self._read_scalar("<Q", 8)

            shape = tuple(reversed(ggml_dims)) if n_dims > 0 else (1,)
            n_elements = 1
            for d in ggml_dims:
                n_elements *= d

            if qtype not in GGML_BLOCK_INFO:
                raise ValueError(f"Unsupported GGML tensor quantization type id={qtype} on {name}")
            block_elems, block_bytes = GGML_BLOCK_INFO[qtype]
            if n_elements % block_elems != 0:
                raise ValueError(
                    f"Tensor {name} has {n_elements} elements, not divisible by block size {block_elems}"
                )
            n_bytes = (n_elements // block_elems) * block_bytes

            self.tensors[name] = GGUFTensorInfo(
                name=name,
                ggml_dims=ggml_dims,
                shape=shape,
                qtype=qtype,
                offset=rel_offset,
                n_elements=n_elements,
                n_bytes=n_bytes,
            )

        # Align end of header to `alignment` bytes
        rem = self._pos % self.alignment
        self.tensor_data_offset = self._pos + ((self.alignment - rem) if rem else 0)

    def _read_bytes(self, n: int) -> bytes:
        b = self._mm[self._pos : self._pos + n]
        self._pos += n
        return b

    def _read_scalar(self, fmt: str, size: int) -> Any:
        val = struct.unpack_from(fmt, self._mm, self._pos)[0]
        self._pos += size
        return val

    def _read_string(self) -> str:
        length = struct.unpack_from("<Q", self._mm, self._pos)[0]
        self._pos += 8
        raw = self._mm[self._pos : self._pos + length]
        self._pos += length
        return raw.decode("utf-8", errors="replace")

    def _read_value(self, vtype: int) -> Any:
        if vtype == _GGUFValueType.STRING:
            return self._read_string()
        if vtype == _GGUFValueType.ARRAY:
            elem_type = self._read_scalar("<I", 4)
            length = self._read_scalar("<Q", 8)
            if elem_type == _GGUFValueType.STRING:
                # Fast tight loop using local unpack_from for 150k+ vocab arrays
                mm = self._mm
                pos = self._pos
                unpack_q = struct.unpack_from
                out_strs: List[str] = []
                append = out_strs.append
                for _ in range(length):
                    slen = unpack_q("<Q", mm, pos)[0]
                    pos += 8
                    append(mm[pos : pos + slen].decode("utf-8", errors="replace"))
                    pos += slen
                self._pos = pos
                return out_strs
            if elem_type in _SCALAR_STRUCTS:
                _, elem_size, np_dtype = _SCALAR_STRUCTS[elem_type]
                nbytes = length * elem_size
                arr = np.frombuffer(self._mm, dtype=np_dtype, count=length, offset=self._pos).copy()
                self._pos += nbytes
                return arr.tolist()
            return [self._read_value(elem_type) for _ in range(length)]

        if vtype in _SCALAR_STRUCTS:
            fmt, size, _ = _SCALAR_STRUCTS[vtype]
            return self._read_scalar(fmt, size)
        raise ValueError(f"Unknown GGUF metadata value type: {vtype}")

    def get_tensor_bytes(self, name: str) -> np.ndarray:
        """Return a zero-copy uint8 NumPy view into the memory-mapped tensor bytes."""
        info = self.tensors[name]
        abs_offset = self.tensor_data_offset + info.offset
        return np.frombuffer(self._mm, dtype=np.uint8, count=info.n_bytes, offset=abs_offset)

    def build_model_config(self) -> GGUFModelConfig:
        meta = self.metadata
        arch = str(meta.get("general.architecture", "llama")).lower()
        name = str(meta.get("general.name", self.path.stem))

        def _get(suffix: str, default: Any = None) -> Any:
            if f"{arch}.{suffix}" in meta:
                return meta[f"{arch}.{suffix}"]
            return default

        def _max_nonzero_int(val: Any, fallback: int) -> int:
            if isinstance(val, list):
                nz = [int(x) for x in val if int(x) > 0]
                return max(nz) if nz else fallback
            return int(val) if val is not None else fallback

        hidden_size = int(_get("embedding_length", 4096))
        total_blocks = int(_get("block_count", 32))
        nextn_layers = int(_get("nextn_predict_layers", 0) or 0)

        # Exclude trailing Multi-Token Prediction (`nextn` / MTP) draft blocks from trunk layers
        # (e.g., Qwen3.8-27B has block_count=65 with blk.64.nextn.* and nextn_predict_layers=1 -> 64 trunk layers)
        if nextn_layers > 0 and total_blocks > nextn_layers:
            n_layers = total_blocks - nextn_layers
        else:
            n_layers = total_blocks
            while n_layers > 1 and (
                f"blk.{n_layers - 1}.nextn.eh_proj.weight" in self.tensors
                or (
                    f"blk.{n_layers - 1}.attn_norm.weight" not in self.tensors
                    and f"blk.{n_layers - 1}.attn_q.weight" not in self.tensors
                    and f"blk.{n_layers - 1}.attn_qkv.weight" not in self.tensors
                )
            ):
                n_layers -= 1
                nextn_layers += 1

        n_heads = _max_nonzero_int(_get("attention.head_count", 32), 32)
        n_kv_heads = _max_nonzero_int(_get("attention.head_count_kv", n_heads), n_heads)

        head_dim = _max_nonzero_int(_get("attention.key_length", hidden_size // n_heads), hidden_size // n_heads)
        ffn_raw = _get("feed_forward_length", hidden_size * 4)
        intermediate_size = _max_nonzero_int(ffn_raw, hidden_size * 4)

        # Infer vocab_size from token_embd tensor shape [vocab_size, hidden_size] or tokenizer tokens
        if "token_embd.weight" in self.tensors:
            vocab_size = int(self.tensors["token_embd.weight"].shape[0])
        else:
            tokens = meta.get("tokenizer.ggml.tokens", [])
            vocab_size = len(tokens)

        rms_eps = float(_get("attention.layer_norm_rms_epsilon", 1e-6))
        rope_theta = float(_get("rope.freq_base", 10000.0))
        rope_dim = int(_get("rope.dimension_count", head_dim))
        max_ctx = int(_get("context_length", 32768))

        # GGUF `llama` uses interleaved/adjacent-pair RoPE because convert_hf_to_gguf permutes Q/K.
        # Qwen2/Qwen3/Qwen3.5/Qwen3.8/Gemma/Phi3 keep HF's NeoX split-half RoPE.
        rope_neox = arch not in ("llama",)

        # MoE fields
        n_experts = int(_get("expert_count", 0) or 0)
        n_experts_used = int(_get("expert_used_count", 0) or 0)
        moe_ffn = int(_get("expert_feed_forward_length", intermediate_size) or intermediate_size)
        shared_moe_ffn = int(_get("expert_shared_feed_forward_length", 0) or 0)
        norm_topk = bool(
            _get("expert_weights_norm", arch in ("qwen3moe", "qwen35moe", "qwen3next", "mixtral", "llama"))
        )

        # Hybrid Gated-DeltaNet / SSM fields (`qwen35`, `qwen35moe`, `qwen3next`)
        full_attn_interval = int(_get("full_attention_interval", 0) or 0)
        ssm_conv_kernel = int(_get("ssm.conv_kernel", 0) or 0)
        ssm_inner_size = int(_get("ssm.inner_size", 0) or 0)
        ssm_state_size = int(_get("ssm.state_size", 0) or 0)
        ssm_dt_rank = int(_get("ssm.time_step_rank", 0) or 0)
        ssm_group_count = int(_get("ssm.group_count", 0) or 0)

        # Gemma family specifics
        embed_scale = math.sqrt(hidden_size) if arch.startswith("gemma") else 1.0
        attn_softcap = float(_get("attn_logit_softcapping", 0.0) or 0.0)
        final_softcap = float(_get("final_logit_softcapping", 0.0) or 0.0)
        query_scalar = float(_get("attention.query_pre_attn_scalar", 0.0) or 0.0)

        bos_id = meta.get("tokenizer.ggml.bos_token_id")
        eos_ids: List[int] = []
        if "tokenizer.ggml.eos_token_id" in meta:
            eos_ids.append(int(meta["tokenizer.ggml.eos_token_id"]))
        extra_eos = meta.get("tokenizer.ggml.eos_token_ids")
        if isinstance(extra_eos, list):
            for eid in extra_eos:
                if int(eid) not in eos_ids:
                    eos_ids.append(int(eid))

        chat_template = meta.get("tokenizer.chat_template")

        return GGUFModelConfig(
            arch=arch,
            name=name,
            vocab_size=vocab_size,
            n_layers=n_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_eps,
            rope_theta=rope_theta,
            rope_dim=rope_dim,
            rope_neox=rope_neox,
            max_context_len=max_ctx,
            n_experts=n_experts,
            n_experts_used=n_experts_used,
            moe_intermediate_size=moe_ffn,
            shared_expert_intermediate_size=shared_moe_ffn,
            norm_topk_prob=norm_topk,
            nextn_predict_layers=nextn_layers,
            full_attention_interval=full_attn_interval,
            ssm_conv_kernel=ssm_conv_kernel,
            ssm_inner_size=ssm_inner_size,
            ssm_state_size=ssm_state_size,
            ssm_dt_rank=ssm_dt_rank,
            ssm_group_count=ssm_group_count,
            embed_scale=embed_scale,
            attn_logit_softcapping=attn_softcap,
            final_logit_softcapping=final_softcap,
            query_pre_attn_scalar=query_scalar,
            bos_token_id=int(bos_id) if bos_id is not None else None,
            eos_token_ids=eos_ids,
            chat_template=str(chat_template) if chat_template else None,
        )

    def close(self) -> None:
        if getattr(self, "_mm", None) is not None:
            self._mm.close()
            self._mm = None  # type: ignore[assignment]
        if getattr(self, "_fh", None) is not None:
            self._fh.close()
            self._fh = None  # type: ignore[assignment]

    def __enter__(self) -> "GGUFFile":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class GGUFTokenizer:
    """
    Tokenizes text and renders OpenAI chat messages either via:
    1. An optional HuggingFace tokenizer (`hf_tokenizer="Qwen/Qwen3-8B"` or local dir), OR
    2. Directly from the embedded `tokenizer.ggml.*` tables inside the `.gguf` file!
    """

    def __init__(
        self,
        gguf_file: GGUFFile,
        config: GGUFModelConfig,
        hf_tokenizer: Optional[str] = None,
        chat_template: Optional[str] = None,
        chat_template_mode: Literal["auto", "gguf", "qwen35_xml", "qwen3_json"] = "auto",
    ) -> None:
        self.config = config
        self.chat_template_mode = chat_template_mode
        self.custom_chat_template: Optional[str] = None
        if chat_template:
            cand_path = Path(chat_template)
            if cand_path.is_file():
                self.custom_chat_template = cand_path.read_text(encoding="utf-8")
            else:
                self.custom_chat_template = chat_template
        self._hf_tok: Any = None
        self._rust_tok: Any = None
        self._vocab_list: List[str] = gguf_file.metadata.get("tokenizer.ggml.tokens", [])
        self._token_to_id: Dict[str, int] = {t: i for i, t in enumerate(self._vocab_list)}
        self.add_bos_token: bool = bool(
            gguf_file.metadata.get("tokenizer.ggml.add_bos_token", config.arch in ("llama", "gemma", "gemma2", "gemma3"))
        )

        # Register Common Chat Stop Tokens (<|im_end|>, <|eot_id|>, <end_of_turn>, <|endoftext|>)
        for special_stop in ("<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|endoftext|>", "</s>"):
            sid = self._token_to_id.get(special_stop)
            if sid is not None and sid not in self.config.eos_token_ids:
                self.config.eos_token_ids.append(sid)

        if hf_tokenizer:
            try:
                from transformers import AutoTokenizer

                self._hf_tok = AutoTokenizer.from_pretrained(hf_tokenizer, trust_remote_code=True)
                if self._hf_tok.eos_token_id is not None and self._hf_tok.eos_token_id not in self.config.eos_token_ids:
                    self.config.eos_token_ids.append(int(self._hf_tok.eos_token_id))
                return
            except Exception as exc:
                print(f"[PyGGUF] Note: Could not load hf_tokenizer='{hf_tokenizer}' ({exc}); using embedded GGUF tokenizer.")

        # Build fast Rust `tokenizers.Tokenizer` directly from GGUF metadata when available
        self._rust_tok = self._build_embedded_rust_tokenizer(gguf_file.metadata)

    def _build_embedded_rust_tokenizer(self, meta: Dict[str, Any]) -> Any:
        tokens: List[str] = meta.get("tokenizer.ggml.tokens", [])
        if not tokens:
            return None
        try:
            from tokenizers import AddedToken, Tokenizer
        except ImportError:
            return None

        tok_model_type = str(meta.get("tokenizer.ggml.model", "gpt2")).lower()
        merges_raw: List[str] = meta.get("tokenizer.ggml.merges", [])
        token_types: List[int] = meta.get("tokenizer.ggml.token_type", [])

        vocab_dict = {tok: idx for idx, tok in enumerate(tokens)}

        def _keep_visible_in_decode(tok: str) -> bool:
            # Keep reasoning & tool-call delimiters visible so `ReasoningAndToolStreamParser`
            # can cleanly separate `<think>...</think>` into `reasoning_content` and parse `<tool_call>`!
            if tok in (
                "<think>",
                "</think>",
                "<thinking>",
                "</thinking>",
                "<redacted_thinking>",
                "</redacted_thinking>",
                "<tool_call>",
                "</tool_call>",
                "<tool_response>",
                "</tool_response>",
                "</function>",
                "</parameter>",
                "<|python_tag|>",
            ):
                return True
            if tok.startswith("<function=") or tok.startswith("<parameter="):
                return True
            return False

        try:
            if tok_model_type in ("gpt2", "bpe") and merges_raw:
                merges = []
                for m in merges_raw:
                    parts = m.split(" ", 1)
                    if len(parts) == 2:
                        merges.append((parts[0], parts[1]))
                # Construct JSON representation for tokenizers.Tokenizer.from_str (works across versions)
                tok_json = {
                    "version": "1.0",
                    "truncation": None,
                    "padding": None,
                    "added_tokens": [
                        {
                            "id": idx,
                            "content": tok,
                            "single_word": False,
                            "lstrip": False,
                            "rstrip": False,
                            "normalized": False,
                            "special": not _keep_visible_in_decode(tok),
                        }
                        for idx, tok in enumerate(tokens)
                        if (idx < len(token_types) and token_types[idx] in (3, 4))
                        or (tok.startswith("<|") and tok.endswith("|>"))
                    ],
                    "normalizer": None,
                    "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
                    "post_processor": None,
                    "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
                    "model": {
                        "type": "BPE",
                        "dropout": None,
                        "unk_token": None,
                        "continuing_subword_prefix": "",
                        "end_of_word_suffix": "",
                        "fuse_unk": False,
                        "byte_fallback": False,
                        "vocab": vocab_dict,
                        "merges": merges,
                    },
                }
                return Tokenizer.from_str(json.dumps(tok_json))

            if tok_model_type == "llama":
                scores = meta.get("tokenizer.ggml.scores", [0.0] * len(tokens))
                unk_id = int(meta.get("tokenizer.ggml.unknown_token_id", 0))
                tok_json = {
                    "version": "1.0",
                    "truncation": None,
                    "padding": None,
                    "added_tokens": [
                        {
                            "id": idx,
                            "content": tok,
                            "single_word": False,
                            "lstrip": False,
                            "rstrip": False,
                            "normalized": False,
                            "special": not _keep_visible_in_decode(tok),
                        }
                        for idx, tok in enumerate(tokens)
                        if idx < len(token_types) and token_types[idx] in (3, 4)
                    ],
                    "normalizer": {"type": "Prepend", "prepend": "▁"},
                    "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "never"},
                    "post_processor": None,
                    "decoder": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always"},
                    "model": {
                        "type": "Unigram",
                        "unk_id": unk_id,
                        "vocab": [[tok, float(scores[idx]) if idx < len(scores) else 0.0] for idx, tok in enumerate(tokens)],
                        "byte_fallback": True,
                    },
                }
                return Tokenizer.from_str(json.dumps(tok_json))
        except Exception:
            return None
        return None

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        if self._hf_tok is not None:
            return list(self._hf_tok.encode(text, add_special_tokens=add_special_tokens))

        if self._rust_tok is not None:
            ids = list(self._rust_tok.encode(text, add_special_tokens=False).ids)
            if add_special_tokens and self.add_bos_token and self.config.bos_token_id is not None:
                if not ids or ids[0] != self.config.bos_token_id:
                    ids = [self.config.bos_token_id] + ids
            return ids

        # Pure-Python greedy longest-match fallback (useful for synthetic/tiny unit test vocabs)
        ids = self._greedy_encode(text)
        if add_special_tokens and self.add_bos_token and self.config.bos_token_id is not None:
            if not ids or ids[0] != self.config.bos_token_id:
                ids = [self.config.bos_token_id] + ids
        return ids

    def _greedy_encode(self, text: str) -> List[int]:
        if not self._vocab_list:
            return [b % max(self.config.vocab_size, 1) for b in text.encode("utf-8")]
        out: List[int] = []
        i = 0
        n = len(text)
        max_tok_len = min(32, n)
        while i < n:
            matched = False
            for end in range(min(n, i + max_tok_len), i, -1):
                sub = text[i:end]
                if sub in self._token_to_id:
                    out.append(self._token_to_id[sub])
                    i = end
                    matched = True
                    break
            if not matched:
                # Byte fallback <0xXX> or raw id 0
                byte_val = text[i].encode("utf-8")[0]
                hex_tok = f"<0x{byte_val:02X}>"
                out.append(self._token_to_id.get(hex_tok, byte_val % len(self._vocab_list)))
                i += 1
        return out

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        if not token_ids:
            return ""
        if self._hf_tok is not None:
            return str(self._hf_tok.decode(list(token_ids), skip_special_tokens=skip_special_tokens))
        if self._rust_tok is not None:
            return str(self._rust_tok.decode(list(token_ids), skip_special_tokens=skip_special_tokens))

        pieces: List[str] = []
        eos_set = set(self.config.eos_token_ids)
        for tid in token_ids:
            if skip_special_tokens and (tid in eos_set or tid == self.config.bos_token_id):
                continue
            if 0 <= tid < len(self._vocab_list):
                tok = self._vocab_list[tid]
                pieces.append(tok.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\n"))
        return "".join(pieces)

    def _select_jinja_templates(self) -> List[str]:
        from .agentic_protocol import QWEN3_AGENTIC_JINJA_TEMPLATE, QWEN35_AGENTIC_JINJA_TEMPLATE

        candidates: List[str] = []
        if self.custom_chat_template:
            candidates.append(self.custom_chat_template)

        arch = self.config.arch
        if self.chat_template_mode == "qwen35_xml":
            candidates.append(QWEN35_AGENTIC_JINJA_TEMPLATE)
        elif self.chat_template_mode == "qwen3_json":
            candidates.append(QWEN3_AGENTIC_JINJA_TEMPLATE)
        elif self.chat_template_mode == "gguf":
            if self.config.chat_template:
                candidates.append(self.config.chat_template)
        else:
            # `"auto"` mode: prefer hardened agentic templates for Qwen3.5/3.6/3.8 & Qwen3
            # (fixes `developer` role crashes, mid-conversation `system` crashes, and `|items` string crashes!)
            if arch in ("qwen35", "qwen35moe", "qwen3next"):
                candidates.append(QWEN35_AGENTIC_JINJA_TEMPLATE)
            elif arch in ("qwen3", "qwen3moe"):
                candidates.append(QWEN3_AGENTIC_JINJA_TEMPLATE)

        if self.config.chat_template and self.config.chat_template not in candidates:
            candidates.append(self.config.chat_template)

        if arch.startswith("qwen") and QWEN3_AGENTIC_JINJA_TEMPLATE not in candidates:
            candidates.append(QWEN3_AGENTIC_JINJA_TEMPLATE)

        return candidates

    def apply_chat_template(
        self,
        messages: Sequence[Dict[str, Any]],
        add_generation_prompt: bool = True,
        enable_thinking: Optional[bool] = None,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        tool_choice: Any = None,
        default_enable_thinking: bool = True,
    ) -> str:
        from .agentic_protocol import (
            apply_tool_choice_directive,
            normalize_messages_for_template,
            resolve_enable_thinking,
        )

        eff_thinking = resolve_enable_thinking(
            messages=messages,
            explicit_enable_thinking=enable_thinking,
            default_enable_thinking=default_enable_thinking,
        )
        norm_msgs = normalize_messages_for_template(messages)
        norm_msgs, eff_tools = apply_tool_choice_directive(norm_msgs, tools, tool_choice)

        if (
            self.custom_chat_template is None
            and self.chat_template_mode == "gguf"
            and self._hf_tok is not None
            and getattr(self._hf_tok, "chat_template", None)
        ):
            try:
                return str(
                    self._hf_tok.apply_chat_template(
                        norm_msgs,
                        tools=eff_tools,
                        tokenize=False,
                        add_generation_prompt=add_generation_prompt,
                        enable_thinking=eff_thinking,
                    )
                )
            except Exception:
                pass

        # Render via Jinja2 sandbox across prioritized candidate templates
        candidates = self._select_jinja_templates()
        if candidates:
            try:
                from jinja2 import BaseLoader
                from jinja2.sandbox import ImmutableSandboxedEnvironment

                def _raise_exception(msg: str) -> str:
                    raise ValueError(msg)

                env = ImmutableSandboxedEnvironment(
                    loader=BaseLoader(),
                    trim_blocks=True,
                    lstrip_blocks=True,
                    extensions=["jinja2.ext.loopcontrols"],
                )
                env.globals["raise_exception"] = _raise_exception
                bos_str = (
                    self._vocab_list[self.config.bos_token_id]
                    if (self.config.bos_token_id is not None and self.config.bos_token_id < len(self._vocab_list))
                    else ""
                )
                eos_str = (
                    self._vocab_list[self.config.eos_token_ids[0]]
                    if (self.config.eos_token_ids and self.config.eos_token_ids[0] < len(self._vocab_list))
                    else ""
                )

                for tmpl_src in candidates:
                    try:
                        tmpl = env.from_string(tmpl_src)
                        return tmpl.render(
                            messages=norm_msgs,
                            add_generation_prompt=add_generation_prompt,
                            bos_token=bos_str,
                            eos_token=eos_str,
                            tools=eff_tools,
                            enable_thinking=eff_thinking,
                            add_vision_id=False,
                        )
                    except Exception:
                        continue
            except ImportError:
                pass

        # Architecture-aware fallback formatting (ChatML for Qwen, Gemma turn tags, Llama-3 headers)
        arch = self.config.arch
        if arch.startswith("gemma"):
            parts: List[str] = []
            for m in messages:
                role = "model" if m["role"] == "assistant" else "user"
                parts.append(f"<start_of_turn>{role}\n{m['content']}<end_of_turn>\n")
            if add_generation_prompt:
                parts.append("<start_of_turn>model\n")
            return "".join(parts)

        if arch == "llama" and "<|start_header_id|>" in self._token_to_id:
            parts = []
            for m in messages:
                parts.append(f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content']}<|eot_id|>")
            if add_generation_prompt:
                parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
            return "".join(parts)

        # Default ChatML (Qwen2 / Qwen3 / Mistral / OpenHermes)
        parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)
