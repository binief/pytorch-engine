"""
pygguf_engine
=============
100% Python + PyTorch + OpenAI Triton (`@triton.jit`) GGUF Inference Engine & OpenAI Server.
Zero `llama.cpp`, zero C++ compilation — supports both Dense and Sparse MoE GGUF models,
INT8 Quantized KV Cache for 32K-128K+ context, Dual-GPU sharding on Kaggle T4 x2,
Prefix KV Cache reuse, N-gram Self-Speculative Decoding, and Cloudflare HTTPS Tunnels.
"""

from .agentic_protocol import (
    QWEN3_AGENTIC_JINJA_TEMPLATE,
    QWEN35_AGENTIC_JINJA_TEMPLATE,
    ParsedToolCall,
    ReasoningAndToolStreamParser,
)
from .engine import GenerationResult, PyGGUFEngine, SamplingConfig, StreamChunk
from .gguf_parser import GGUFFile, GGUFModelConfig, GGUFTokenizer
from .openai_server import CloudflareTunnel, create_openai_app

__all__ = [
    "CloudflareTunnel",
    "GGUFFile",
    "GGUFModelConfig",
    "GGUFTokenizer",
    "GenerationResult",
    "ParsedToolCall",
    "PyGGUFEngine",
    "QWEN35_AGENTIC_JINJA_TEMPLATE",
    "QWEN3_AGENTIC_JINJA_TEMPLATE",
    "ReasoningAndToolStreamParser",
    "SamplingConfig",
    "StreamChunk",
    "create_openai_app",
]
