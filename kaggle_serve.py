"""
kaggle_serve.py
===============
One-call Kaggle Notebook & CLI launcher for `pygguf_engine` (0% llama.cpp).

Features:
- Fast 1-2 Gbps GGUF download directly into `/kaggle/tmp` (~1.1 TB scratch disk)
- Loads Dense, Sparse MoE, & Hybrid Gated-DeltaNet GGUFs (`Qwen3.8-27B-UD-Q4_K_M`, `Qwen3.5`, `Qwen3-30B-A3B`, `Llama-3`, `Gemma-3`)
  onto Single T4 or Dual T4 (`cuda:0` + `cuda:1`, 30-32 GB VRAM)
- Starts FastAPI OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`, `/v1/models`)
- Optional free Cloudflare HTTPS Tunnel (`https://xxxx.trycloudflare.com/v1`) + API Key

Usage inside a Kaggle Notebook cell:
------------------------------------
    from kaggle_serve import download_gguf_kaggle, launch_kaggle_server

    gguf_path = download_gguf_kaggle(
        repo_id="bartowski/Qwen_Qwen3-8B-GGUF",
        filename="Qwen_Qwen3-8B-Q4_K_M.gguf",
    )

    handle = launch_kaggle_server(
        gguf_path=gguf_path,
        hf_tokenizer="Qwen/Qwen3-8B",   # Optional (uses embedded GGUF tokenizer if omitted!)
        max_context_len=65536,          # 64K context with INT8 KV cache
        kv_dtype="int8",                # "int4" unlocks 128K-256K+ context
        # Either the GGUF's own {arch}.rope.scaling.* metadata is used verbatim, or override here:
        # rope_scaling_type="yarn", rope_scaling_factor=4.0, rope_scaling_original_context_length=16384,
        enable_tunnel=True,             # Exposes public https://....trycloudflare.com/v1
        api_key="kaggle-secret-key",
        background=True,                # Keeps notebook cell non-blocking!
    )
"""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence, Union

import torch
import uvicorn

from pygguf_engine import CloudflareTunnel, PyGGUFEngine, create_openai_app


def download_gguf_kaggle(
    repo_id: str,
    filename: str,
    dest_dir: Union[str, Path] = "/kaggle/tmp/gguf_models",
    hf_token: Optional[str] = None,
) -> Path:
    """
    Download a `.gguf` file into `/kaggle/tmp` (avoiding `/kaggle/working`'s 20GB cap)
    using `hf_transfer` to saturate Kaggle's 1-2 Gbps Google Cloud network link.
    """
    dest = Path(dest_dir) if Path("/kaggle").exists() else Path("./models")
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / filename
    if target.exists() and target.stat().st_size > 0:
        print(f"[Kaggle] Using cached GGUF file: {target}")
        return target

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        subprocess.check_call(["pip", "install", "-q", "huggingface_hub[hf_transfer]"])

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import hf_hub_download

    print(f"[Kaggle] Downloading {repo_id}/{filename} -> {dest} ...")
    t0 = time.time()
    out_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(dest),
        token=hf_token or os.environ.get("HF_TOKEN"),
    )
    dt = max(time.time() - t0, 1e-3)
    gib = Path(out_path).stat().st_size / (1024**3)
    print(f"[Kaggle] Downloaded {gib:.2f} GiB in {dt:.1f}s ({gib * 8 / dt:.2f} Gbps)")
    return Path(out_path)


@dataclass
class ServerHandle:
    engine: Optional[PyGGUFEngine]
    local_base_url: str
    public_base_url: Optional[str]
    api_key: Optional[str]
    uvicorn_server: Optional[uvicorn.Server]
    tunnel: Optional[CloudflareTunnel] = None
    thread: Optional[threading.Thread] = None

    def stop(self, timeout: float = 30.0) -> None:
        """Stop a background server and release owned model references.

        Active requests must finish before the server exits. On timeout, keep the
        model alive and raise rather than start a second model on top of it.
        Other user-held references to the engine can still keep VRAM allocated.
        """
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        server = self.uvicorn_server
        if server is not None:
            server.should_exit = True
        if self.tunnel is not None:
            self.tunnel.stop()
            self.tunnel = None
        thread = self.thread
        if thread is not None and thread.is_alive():
            if thread is threading.current_thread():
                raise RuntimeError("Cannot wait for the server from its own thread")
            thread.join(timeout=timeout)
            if thread.is_alive():
                raise RuntimeError(
                    f"Server is still shutting down after {timeout:g}s; finish or cancel active requests "
                    "and call stop() again before launching another model."
                )
        self.thread = None
        self.uvicorn_server = None
        self.engine = None
        # Uvicorn's config retains the app and its closure over the engine.
        # Drop local references as well before collecting cyclic objects.
        del server, thread
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def launch_kaggle_server(
    gguf_path: Union[str, Path],
    *,
    hf_tokenizer: Optional[str] = None,
    chat_template: Optional[str] = None,
    chat_template_mode: Literal["auto", "gguf", "qwen35_xml", "qwen3_json"] = "auto",
    enable_thinking: bool = True,
    include_reasoning: bool = True,
    model_alias: Optional[str] = None,
    max_context_len: int = 32768,
    kv_dtype: Literal["int4", "int8", "fp16"] = "int8",
    devices: Optional[Sequence[str]] = None,
    parallel_mode: Literal["auto", "tp", "pp"] = "auto",
    cpu_threads: Optional[int] = None,
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
    rope_scaling_type: Optional[Literal["none", "linear", "ntk-aware", "yarn"]] = None,
    rope_scaling_factor: Optional[float] = None,
    rope_scaling_original_context_length: Optional[int] = None,
    rope_yarn_beta_fast: Optional[float] = None,
    rope_yarn_beta_slow: Optional[float] = None,
    rope_yarn_ext_factor: Optional[float] = None,
    rope_yarn_attn_factor: Optional[float] = None,
    rope_yarn_log_mul: Optional[float] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    api_key: Optional[str] = None,
    enable_tunnel: bool = True,
    background: bool = True,
) -> ServerHandle:
    """Load the GGUF model into `PyGGUFEngine`, launch the OpenAI server, and optionally open Cloudflare Tunnel."""
    if cpu_threads is not None:
        if isinstance(cpu_threads, bool) or not isinstance(cpu_threads, int) or cpu_threads < 1:
            raise ValueError("cpu_threads must be a positive integer or None")
        # PyTorch intra-op setting is process-wide; set it before model construction.
        torch.set_num_threads(cpu_threads)
        print(f"[PyGGUF] PyTorch CPU intra-op threads: {torch.get_num_threads()}")

    # Reclaim any orphaned GPU VRAM from earlier notebook cell exceptions/re-runs before loading weights
    for attr in ("last_traceback", "last_value", "last_type"):
        if hasattr(sys, attr):
            setattr(sys, attr, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    engine = PyGGUFEngine(
        gguf_path=gguf_path,
        hf_tokenizer=hf_tokenizer,
        chat_template=chat_template,
        chat_template_mode=chat_template_mode,
        default_enable_thinking=enable_thinking,
        max_context_len=max_context_len,
        kv_dtype=kv_dtype,
        devices=devices,
        parallel_mode=parallel_mode,
        enable_cuda_graph=enable_cuda_graph,
        flash_attn=flash_attn,
        kv_cpu_spill=kv_cpu_spill,
        speculative_mtp=speculative_mtp,
        mtp_draft_tokens=mtp_draft_tokens,
        mtp_adaptive=mtp_adaptive,
        mtp_max_draft_tokens=mtp_max_draft_tokens,
        mtp_confidence_threshold=mtp_confidence_threshold,
        mtp_kv_window=mtp_kv_window,
        speculative_ngram=speculative_ngram,
        model_alias=model_alias,
        rope_scaling_type=rope_scaling_type,
        rope_scaling_factor=rope_scaling_factor,
        rope_scaling_original_context_length=rope_scaling_original_context_length,
        rope_yarn_beta_fast=rope_yarn_beta_fast,
        rope_yarn_beta_slow=rope_yarn_beta_slow,
        rope_yarn_ext_factor=rope_yarn_ext_factor,
        rope_yarn_attn_factor=rope_yarn_attn_factor,
        rope_yarn_log_mul=rope_yarn_log_mul,
    )
    app = create_openai_app(engine, api_key=api_key, include_reasoning=include_reasoning)
    uv_cfg = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(uv_cfg)

    local_v1 = f"http://127.0.0.1:{port}/v1"
    tunnel: Optional[CloudflareTunnel] = None
    public_v1: Optional[str] = None

    if not background:
        if enable_tunnel:
            tunnel = CloudflareTunnel(port=port)
            pub_root = tunnel.start()
            public_v1 = f"{pub_root}/v1"
            _print_banner(engine.model_id, local_v1, public_v1, api_key)
        else:
            _print_banner(engine.model_id, local_v1, None, api_key)
        server.run()
        return ServerHandle(engine, local_v1, public_v1, api_key, server, tunnel, None)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait until local FastAPI server responds on /health
    deadline = time.time() + 30.0
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1.0) as resp:
                if resp.status == 200:
                    break
        except Exception:
            time.sleep(0.2)

    if enable_tunnel:
        tunnel = CloudflareTunnel(port=port)
        pub_root = tunnel.start()
        public_v1 = f"{pub_root}/v1"

    _print_banner(engine.model_id, local_v1, public_v1, api_key)
    return ServerHandle(engine, local_v1, public_v1, api_key, server, tunnel, thread)


def _print_banner(
    model_id: str,
    local_v1: str,
    public_v1: Optional[str],
    api_key: Optional[str],
) -> None:
    shown_key = api_key or "not-required"
    target_url = public_v1 or local_v1
    print("\n" + "=" * 72)
    print(" PyGGUF OpenAI Server is LIVE (100% Python/PyTorch/Triton, 0% llama.cpp)")
    print("=" * 72)
    print(f"  Model ID        : {model_id}")
    print(f"  Local Base URL  : {local_v1}")
    if public_v1:
        print(f"  Public HTTPS URL: {public_v1}")
    print(f"  API Key         : {shown_key}")
    print("-" * 72)
    print("  Python OpenAI SDK snippet:")
    print("    from openai import OpenAI")
    print(f'    client = OpenAI(base_url="{target_url}", api_key="{shown_key}")')
    print(
        f'    resp = client.chat.completions.create(model="{model_id}", '
        'messages=[{"role": "user", "content": "Hello!"}], stream=True)'
    )
    print("=" * 72 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve GGUF models with PyGGUF OpenAI API (no llama.cpp)")
    parser.add_argument("--gguf", type=str, default=None, help="Local path to .gguf file")
    parser.add_argument("--repo", type=str, default=None, help="HuggingFace repo_id to download GGUF from")
    parser.add_argument("--file", type=str, default=None, help="GGUF filename inside --repo")
    parser.add_argument("--tokenizer", type=str, default=None, help="Optional HuggingFace base tokenizer repo")
    parser.add_argument("--alias", type=str, default=None, help="Served OpenAI model ID")
    parser.add_argument("--ctx", type=int, default=32768, help="Max context window (e.g. 32768, 131072, 262144)")
    parser.add_argument("--kv-dtype", choices=["int4", "int8", "fp16"], default="int8", help="KV cache precision (int4 unlocks 256K+ ctx!)")
    parser.add_argument("--port", type=int, default=8000, help="HTTP server port")
    parser.add_argument("--api-key", type=str, default=os.environ.get("OPENAI_API_KEY"), help="Optional Bearer API key")
    parser.add_argument("--tunnel", action="store_true", help="Expose public https://*.trycloudflare.com/v1 URL")
    parser.add_argument("--no-speculative", action="store_true", help="Disable BOTH N-gram and draft-mtp speculative decoding")
    parser.add_argument("--no-mtp", action="store_true", help="Disable embedded GGUF draft-mtp head (0 GB MTP VRAM loaded)")
    parser.add_argument("--mtp-draft-tokens", type=int, default=None, help="Pin exact MTP draft tokens per step (1..8; omit for adaptive k=1..3)")
    parser.add_argument("--no-adaptive-mtp", action="store_true", help="Disable confidence-gated adaptive MTP depth (k=1..3)")
    parser.add_argument("--mtp-max-draft-tokens", type=int, default=3, help="Max adaptive MTP draft depth when confident (default: 3)")
    parser.add_argument("--mtp-conf-threshold", type=float, default=0.55, help="Softmax confidence threshold for adaptive MTP chaining")
    parser.add_argument("--mtp-kv-window", type=int, default=4096, help="Sliding-window ring KV size for MTP head (0 = unbounded)")
    parser.add_argument("--no-ngram", action="store_true", help="Disable N-gram self-speculative decoding")
    parser.add_argument("--chat-template", type=str, default=None, help="Optional path to custom .jinja template file")
    parser.add_argument(
        "--chat-template-mode",
        choices=["auto", "gguf", "qwen35_xml", "qwen3_json"],
        default="auto",
        help="Agentic Jinja template mode (default: auto)",
    )
    parser.add_argument("--no-thinking", action="store_true", help="Disable <think> reasoning by default")
    parser.add_argument("--hide-reasoning", action="store_true", help="Omit reasoning_content from API responses")
    parser.add_argument(
        "--parallel-mode",
        choices=["auto", "tp", "pp"],
        default="auto",
        help="Multi-GPU execution mode: auto (TP=2 for Dense/Hybrid, PP for MoE), tp, or pp",
    )
    parser.add_argument("--cpu-threads", type=int, default=None, help="PyTorch CPU intra-op threads (process-wide; default: unchanged)")
    parser.add_argument("--no-cuda-graph", action="store_true", help="Disable CUDAGraph capture for M=1 decode")
    parser.add_argument("--no-flash-attn", action="store_true", help="Disable the fused Triton GQA flash-attention (Path 3a); falls back to chunked online-softmax attention")
    parser.add_argument("--kv-cpu-spill", action="store_true", help="Park trunk attention KV + recurrent state in CPU RAM between turns; restored automatically on prefix-cache hits (opt-in, saves VRAM for 256K sessions)")
    parser.add_argument("--rope-scaling-type", choices=["none", "linear", "ntk-aware", "yarn"], default=None,
                        help="Override RoPE scaling from GGUF metadata (extend context beyond trained length)")
    parser.add_argument("--rope-scaling-factor", type=float, default=None, help="RoPE scaling factor, e.g. 2.0/4.0/8.0 for YaRN")
    parser.add_argument("--rope-scaling-orig-ctx", type=int, default=None, help="Original (pre-extension) context length for YaRN correction dims, e.g. 32768")
    parser.add_argument("--rope-yarn-beta-fast", type=float, default=None, help="YaRN beta_fast (default: 32.0)")
    parser.add_argument("--rope-yarn-beta-slow", type=float, default=None, help="YaRN beta_slow (default: 1.0)")
    parser.add_argument("--rope-yarn-ext-factor", type=float, default=None, help="YaRN ext_factor for the mscale attention factor (default: 1.0)")
    parser.add_argument("--rope-yarn-attn-factor", type=float, default=None, help="Override YaRN attention (mscale) factor explicitly (default: auto = 1 + log_mul * ext_factor * log(factor))")
    parser.add_argument("--rope-yarn-log-mul", type=float, default=None, help="YaRN mscale log-multiplier (default: 0.1)")
    args = parser.parse_args()

    if args.gguf:
        gguf_path = Path(args.gguf)
    elif args.repo and args.file:
        gguf_path = download_gguf_kaggle(args.repo, args.file)
    else:
        parser.error("Provide either `--gguf /path/to/model.gguf` OR `--repo <hf_repo> --file <name.gguf>`.")

    launch_kaggle_server(
        gguf_path=gguf_path,
        hf_tokenizer=args.tokenizer,
        chat_template=args.chat_template,
        chat_template_mode=args.chat_template_mode,
        enable_thinking=not args.no_thinking,
        include_reasoning=not args.hide_reasoning,
        model_alias=args.alias,
        max_context_len=args.ctx,
        kv_dtype=args.kv_dtype,
        parallel_mode=args.parallel_mode,
        cpu_threads=args.cpu_threads,
        enable_cuda_graph=not args.no_cuda_graph,
        flash_attn=not args.no_flash_attn,
        kv_cpu_spill=args.kv_cpu_spill,
        speculative_mtp=(not args.no_mtp) and (not args.no_speculative),
        mtp_draft_tokens=args.mtp_draft_tokens,
        mtp_adaptive=not args.no_adaptive_mtp,
        mtp_max_draft_tokens=args.mtp_max_draft_tokens,
        mtp_confidence_threshold=args.mtp_conf_threshold,
        mtp_kv_window=args.mtp_kv_window,
        speculative_ngram=(not args.no_ngram) and (not args.no_speculative),
        rope_scaling_type=args.rope_scaling_type,
        rope_scaling_factor=args.rope_scaling_factor,
        rope_scaling_original_context_length=args.rope_scaling_orig_ctx,
        rope_yarn_beta_fast=args.rope_yarn_beta_fast,
        rope_yarn_beta_slow=args.rope_yarn_beta_slow,
        rope_yarn_ext_factor=args.rope_yarn_ext_factor,
        rope_yarn_attn_factor=args.rope_yarn_attn_factor,
        rope_yarn_log_mul=args.rope_yarn_log_mul,
        port=args.port,
        api_key=args.api_key,
        enable_tunnel=args.tunnel,
        background=False,
    )


if __name__ == "__main__":
    main()
