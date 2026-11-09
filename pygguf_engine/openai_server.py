"""
OpenAI-Compatible FastAPI Server for `PyGGUFEngine` + Built-in Cloudflare HTTPS Tunnel
for instant Kaggle Notebook external access (`https://xxxx.trycloudflare.com/v1`).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.request
import uuid
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Sequence, Union

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .agentic_protocol import ReasoningAndToolStreamParser, resolve_enable_thinking
from .engine import PyGGUFEngine, SamplingConfig


# ==============================================================================
# 1. OpenAI-Compatible Request / Response Schemas (Full Tools + Reasoning)
# ==============================================================================

class ChatMessageInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Union[str, List[Any], None] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    reasoning_content: Optional[str] = None

    def normalized_text(self) -> str:
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        parts: List[str] = []
        for item in self.content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)

    def to_template_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "role": self.role,
            "content": self.normalized_text(),
        }
        if self.name is not None:
            d["name"] = self.name
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.reasoning_content is not None:
            d["reasoning_content"] = self.reasoning_content
        return d


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="allow")

    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: Optional[str] = None
    messages: List[ChatMessageInput]
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    min_p: float = 0.05
    repetition_penalty: float = 1.0
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    seed: Optional[int] = None
    cache_prompt: bool = True
    enable_thinking: Optional[bool] = None
    reasoning_effort: Optional[str] = None
    thinking: Optional[Dict[str, Any]] = None
    include_reasoning: Optional[bool] = None
    chat_template_kwargs: Optional[Dict[str, Any]] = None
    # Per-request speculative decoding controls (`extra_body={"speculative_mtp": False}` etc.)
    speculative: Optional[bool] = None
    speculative_mtp: Optional[bool] = None
    speculative_ngram: Optional[bool] = None
    mtp_draft_tokens: Optional[int] = None
    mtp_adaptive: Optional[bool] = None
    mtp_confidence_threshold: Optional[float] = None


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: Optional[str] = None
    prompt: Union[str, List[int]]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    min_p: float = 0.05
    repetition_penalty: float = 1.0
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    seed: Optional[int] = None
    cache_prompt: bool = True
    speculative: Optional[bool] = None
    speculative_mtp: Optional[bool] = None
    speculative_ngram: Optional[bool] = None
    mtp_draft_tokens: Optional[int] = None
    mtp_adaptive: Optional[bool] = None
    mtp_confidence_threshold: Optional[float] = None


# ==============================================================================
# 2. FastAPI Application Factory
# ==============================================================================

def _normalize_stops(stop: Optional[Union[str, List[str]]]) -> List[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop] if stop else []
    return [s for s in stop if s]


def create_openai_app(
    engine: PyGGUFEngine,
    api_key: Optional[str] = None,
    include_reasoning: bool = True,
) -> FastAPI:
    """Create a FastAPI app exposing `/v1/models`, `/v1/chat/completions`, `/v1/completions`, and `/health`."""
    app = FastAPI(
        title="PyGGUF OpenAI Server",
        version="1.0.0",
        description="100% Python + PyTorch + OpenAI Triton GGUF Serving Engine (0% llama.cpp)",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    created_ts = int(time.time())

    async def verify_auth(authorization: Optional[str] = Header(default=None)) -> None:
        if not api_key:
            return
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing Bearer API key in Authorization header.",
            )
        token = authorization[len("Bearer ") :].strip()
        if token != api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key.",
            )

    @app.get("/health")
    @app.get("/v1/health")
    async def health() -> Dict[str, Any]:
        return {"status": "ok", **engine.vram_report()}

    @app.get("/v1/models", dependencies=[Depends(verify_auth)])
    async def list_models() -> Dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": engine.model_id,
                    "object": "model",
                    "created": created_ts,
                    "owned_by": "pygguf-engine",
                    "meta": {
                        "arch": engine.config.arch,
                        "max_context_len": engine.config.max_context_len,
                        "kv_dtype": engine.kv_dtype,
                        "is_moe": engine.config.is_moe,
                        "mtp_loaded": engine.model.mtp_head is not None,
                        "speculative_mtp_default": bool(engine.speculative_mtp and engine.model.mtp_head is not None),
                        "mtp_draft_tokens_default": engine.mtp_draft_tokens,
                        "speculative_ngram_default": engine.speculative_ngram,
                    },
                }
            ],
        }

    @app.post("/v1/chat/completions", dependencies=[Depends(verify_auth)])
    async def chat_completions(req: ChatCompletionRequest) -> Any:
        norm_messages = [m.to_template_dict() for m in req.messages]
        eff_thinking = resolve_enable_thinking(
            messages=norm_messages,
            explicit_enable_thinking=req.enable_thinking,
            chat_template_kwargs=req.chat_template_kwargs,
            reasoning_effort=req.reasoning_effort,
            thinking_param=req.thinking,
            default_enable_thinking=engine.default_enable_thinking,
        )
        emit_reasoning = req.include_reasoning if req.include_reasoning is not None else include_reasoning
        parse_tools = req.tool_choice != "none"

        prompt_str = engine.tokenizer.apply_chat_template(
            norm_messages,
            add_generation_prompt=True,
            enable_thinking=eff_thinking,
            tools=req.tools,
            tool_choice=req.tool_choice,
            default_enable_thinking=engine.default_enable_thinking,
        )

        max_toks = req.max_completion_tokens or req.max_tokens or 1024
        use_mtp = req.speculative_mtp if req.speculative_mtp is not None else req.speculative
        use_ngram = req.speculative_ngram if req.speculative_ngram is not None else req.speculative
        sampling = SamplingConfig(
            max_tokens=max_toks,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            min_p=req.min_p,
            repetition_penalty=req.repetition_penalty,
            stop_strings=_normalize_stops(req.stop),
            cache_prompt=req.cache_prompt,
            seed=req.seed,
            use_mtp=use_mtp,
            use_ngram=use_ngram,
            mtp_draft_tokens=req.mtp_draft_tokens,
            mtp_adaptive=req.mtp_adaptive,
            mtp_confidence_threshold=req.mtp_confidence_threshold,
        )

        req_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        model_name = req.model or engine.model_id
        now = int(time.time())

        if not req.stream:
            parser = ReasoningAndToolStreamParser(
                prompt_str=prompt_str,
                tools=req.tools,
                parse_tool_calls=parse_tools,
                emit_reasoning=emit_reasoning,
            )
            stats = None
            async for ch in engine.astream_tokens(prompt_str, sampling):
                if ch.delta_text:
                    parser.feed(ch.delta_text)
                if ch.stats is not None:
                    stats = ch.stats
            parser.finalize()
            assert stats is not None

            clean_content = parser.full_content
            msg_payload: Dict[str, Any] = {
                "role": "assistant",
                "content": (None if (parser.has_tool_calls and not clean_content.strip()) else clean_content),
            }
            if emit_reasoning and parser.full_reasoning:
                msg_payload["reasoning_content"] = parser.full_reasoning
            if parser.has_tool_calls:
                msg_payload["tool_calls"] = [tc.to_openai_dict() for tc in parser.parsed_tool_calls]
                finish_reason: str = "tool_calls"
            else:
                finish_reason = stats.finish_reason

            return {
                "id": req_id,
                "object": "chat.completion",
                "created": now,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": msg_payload,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": stats.prompt_tokens,
                    "completion_tokens": stats.completion_tokens,
                    "total_tokens": stats.prompt_tokens + stats.completion_tokens,
                    "prompt_tokens_details": {"cached_tokens": stats.cached_prefix_tokens},
                },
                "timings": {
                    "prompt_per_second": round(stats.prompt_tok_per_s, 2),
                    "predicted_per_second": round(stats.decode_tok_per_s, 2),
                    "prefill_ms": round(stats.prefill_time_s * 1000, 2),
                    "decode_ms": round(stats.decode_time_s * 1000, 2),
                    "speculative_accepted": stats.speculative_accepted,
                    "speculative_drafted": stats.speculative_drafted,
                    "mtp_accepted": stats.mtp_accepted,
                    "mtp_drafted": stats.mtp_drafted,
                },
            }

        async def _sse_generator() -> AsyncIterator[str]:
            parser = ReasoningAndToolStreamParser(
                prompt_str=prompt_str,
                tools=req.tools,
                parse_tool_calls=parse_tools,
                emit_reasoning=emit_reasoning,
            )
            emitted_tool_idx = 0

            # Initial role chunk expected by OpenAI SDK / OpenCode / Cline / Continue
            role_chunk = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": now,
                "model": model_name,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(role_chunk)}\n\n"

            def _format_emission(em: Any) -> List[str]:
                nonlocal emitted_tool_idx
                sse_Payloads: List[str] = []
                if em.reasoning_delta:
                    r_chunk = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": now,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"reasoning_content": em.reasoning_delta},
                                "finish_reason": None,
                            }
                        ],
                    }
                    sse_Payloads.append(f"data: {json.dumps(r_chunk, ensure_ascii=False)}\n\n")
                if em.content_delta:
                    c_chunk = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": now,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": em.content_delta},
                                "finish_reason": None,
                            }
                        ],
                    }
                    sse_Payloads.append(f"data: {json.dumps(c_chunk, ensure_ascii=False)}\n\n")
                if em.tool_calls:
                    tc_deltas = []
                    for tc in em.tool_calls:
                        tc_deltas.append(tc.to_openai_stream_delta(emitted_tool_idx))
                        emitted_tool_idx += 1
                    t_chunk = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": now,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"tool_calls": tc_deltas},
                                "finish_reason": None,
                            }
                        ],
                    }
                    sse_Payloads.append(f"data: {json.dumps(t_chunk, ensure_ascii=False)}\n\n")
                return sse_Payloads

            final_stats = None
            raw_finish_reason: str = "stop"
            try:
                async for ch in engine.astream_tokens(prompt_str, sampling):
                    if ch.delta_text:
                        for sse_str in _format_emission(parser.feed(ch.delta_text)):
                            yield sse_str
                    if ch.finish_reason is not None:
                        final_stats = ch.stats
                        raw_finish_reason = ch.finish_reason

                # Flush any remaining buffered content / auto-healed tool call at stream end
                for sse_str in _format_emission(parser.finalize()):
                    yield sse_str

                eff_finish = "tool_calls" if parser.has_tool_calls else raw_finish_reason
                end_chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": now,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": eff_finish}],
                }
                yield f"data: {json.dumps(end_chunk)}\n\n"
            except Exception as exc:
                import traceback
                traceback.print_exc()
                err_chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": now,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": f"\n[PyGGUF Server Error: {type(exc).__name__}: {exc}]"},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(err_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                return

            if req.stream_options and req.stream_options.include_usage and final_stats is not None:
                usage_chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": now,
                    "model": model_name,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": final_stats.prompt_tokens,
                        "completion_tokens": final_stats.completion_tokens,
                        "total_tokens": final_stats.prompt_tokens + final_stats.completion_tokens,
                    },
                }
                yield f"data: {json.dumps(usage_chunk)}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(_sse_generator(), media_type="text/event-stream")

    @app.post("/v1/completions", dependencies=[Depends(verify_auth)])
    async def completions(req: CompletionRequest) -> Any:
        use_mtp = req.speculative_mtp if req.speculative_mtp is not None else req.speculative
        use_ngram = req.speculative_ngram if req.speculative_ngram is not None else req.speculative
        sampling = SamplingConfig(
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            min_p=req.min_p,
            repetition_penalty=req.repetition_penalty,
            stop_strings=_normalize_stops(req.stop),
            cache_prompt=req.cache_prompt,
            seed=req.seed,
            use_mtp=use_mtp,
            use_ngram=use_ngram,
            mtp_draft_tokens=req.mtp_draft_tokens,
            mtp_adaptive=req.mtp_adaptive,
            mtp_confidence_threshold=req.mtp_confidence_threshold,
        )
        req_id = f"cmpl-{uuid.uuid4().hex[:24]}"
        model_name = req.model or engine.model_id
        now = int(time.time())

        if not req.stream:
            stats = None
            async for ch in engine.astream_tokens(req.prompt, sampling):
                if ch.stats is not None:
                    stats = ch.stats
            assert stats is not None
            return {
                "id": req_id,
                "object": "text_completion",
                "created": now,
                "model": model_name,
                "choices": [{"index": 0, "text": stats.text, "finish_reason": stats.finish_reason}],
                "usage": {
                    "prompt_tokens": stats.prompt_tokens,
                    "completion_tokens": stats.completion_tokens,
                    "total_tokens": stats.prompt_tokens + stats.completion_tokens,
                },
            }

        async def _sse_cmpl() -> AsyncIterator[str]:
            async for ch in engine.astream_tokens(req.prompt, sampling):
                if ch.delta_text or ch.finish_reason is not None:
                    payload = {
                        "id": req_id,
                        "object": "text_completion",
                        "created": now,
                        "model": model_name,
                        "choices": [{"index": 0, "text": ch.delta_text, "finish_reason": ch.finish_reason}],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_sse_cmpl(), media_type="text/event-stream")

    return app


# ==============================================================================
# 3. Cloudflare Quick Tunnel Helper (Free Public HTTPS URL from Kaggle!)
# ==============================================================================

class CloudflareTunnel:
    """Spawns a free `trycloudflare.com` HTTPS tunnel pointing to the local OpenAI server."""

    def __init__(self, port: int = 8000, bin_path: str = "/tmp/cloudflared") -> None:
        self.port = port
        self.bin_path = bin_path
        self.public_url: Optional[str] = None
        self._proc: Optional[subprocess.Popen[str]] = None

    def start(self, timeout: float = 30.0) -> str:
        bin_file = shutil.which("cloudflared") or self.bin_path
        if not os.path.exists(bin_file):
            print("[Tunnel] Downloading cloudflared static Linux x64 binary...")
            urllib.request.urlretrieve(
                "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
                bin_file,
            )
            os.chmod(bin_file, 0o755)

        self._proc = subprocess.Popen(
            [bin_file, "tunnel", "--url", f"http://127.0.0.1:{self.port}", "--no-autoupdate"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        url_pattern = re.compile(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com")
        found_event = threading.Event()

        def _reader() -> None:
            assert self._proc is not None and self._proc.stdout is not None
            for line in self._proc.stdout:
                m = url_pattern.search(line)
                if m and self.public_url is None:
                    self.public_url = m.group(0)
                    found_event.set()

        threading.Thread(target=_reader, daemon=True).start()
        if not found_event.wait(timeout=timeout) or not self.public_url:
            self.stop()
            raise TimeoutError("Timed out waiting for Cloudflare Quick Tunnel URL.")
        return self.public_url

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
