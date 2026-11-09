"""
Agentic Chat Template, Reasoning (`<think>...</think>`), and Tool-Call Protocol Engine
for `PyGGUFEngine` OpenAI-Compatible Serving (`0% llama.cpp`).

Solves the two classic agent-harness pain points on Qwen3 / Qwen3.5 / Qwen3.6 / Qwen3.8 / Llama / DeepSeek GGUFs:
1. `<think>...</think>` leakage:
   - Incremental state-machine parser strips `<think>...</think>` (including pre-opened `<think>` Prompts
     on `Qwen3.5` / `Qwen3.8`, empty `<think></think>` blocks, and orphan `</think>` tags) out of `content`
     and routes non-empty thoughts cleanly to `reasoning_content` (`delta.reasoning_content` in SSE).
2. Multi-Turn Tool Calling across Coding & Agent Harnesses (OpenCode, Claude Code, Cline, Roo Code,
   Continue, Cursor, Aider, OpenAI Agents SDK, LangGraph, Qwen-Agent):
   - Built-in hardened Jinja2 templates supporting `developer` role, mid-conversation `system` messages,
     `tool` responses, and string-or-dict `tool_calls.function.arguments` without Jinja crashes.
   - Dual-protocol `<tool_call>` output parser supporting BOTH:
     * Qwen3.5 / Qwen3.8 / Qwen3-Coder XML (`<function=name><parameter=k>v</parameter></function>`)
       with JSON-Schema-aware type coercion (`int`, `float`, `bool`, `dict`, `list`, `str`) and safe
       handling of `<`, `>`, `&` inside code parameters!
     * Qwen3 / Qwen2.5 / Hermes / Llama JSON (`{"name": ..., "arguments": ...}`).
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union


# ==============================================================================
# 1. Hardened Agentic Jinja2 Templates (Qwen3.5/3.8 XML & Qwen3 JSON)
# ==============================================================================

# Battle-tested Qwen3.5 / Qwen3.6 / Qwen3.8 (`qwen35`, `qwen35moe`, `qwen3next`) Jinja template:
# - Supports `developer` role and mid-conversation `system` messages
# - Supports system-only or tool-only turns without raising exceptions
# - Normalizes `tool_call.arguments` whether mapping or JSON string
# - Never injects empty `<think></think>` into past history (preserves 100% KV cache prefix hits!)
# - Enforces `</think>` closure BEFORE `<tool_call>`
QWEN35_AGENTIC_JINJA_TEMPLATE = r"""{%- macro render_content(content) %}
    {%- if content is string %}
        {{- content }}
    {%- elif content is iterable and content is not mapping %}
        {%- for item in content %}
            {%- if item is string %}
                {{- item }}
            {%- elif item is mapping and 'text' in item %}
                {{- item.text }}
            {%- endif %}
        {%- endfor %}
    {%- elif content is none or content is undefined %}
        {{- '' }}
    {%- else %}
        {{- content | string }}
    {%- endif %}
{%- endmacro %}
{%- set ns_flags = namespace(enable_thinking=true) %}
{%- if enable_thinking is defined %}
    {%- set ns_flags.enable_thinking = enable_thinking %}
{%- endif %}
{%- set system_content = '' %}
{%- set has_first_system = false %}
{%- if messages and (messages[0].role == 'system' or messages[0].role == 'developer') %}
    {%- set has_first_system = true %}
    {%- set system_content = render_content(messages[0].content) | trim %}
    {%- if '<|think_off|>' in system_content or '/no_think' in system_content %}
        {%- if enable_thinking is not defined %}
            {%- set ns_flags.enable_thinking = false %}
        {%- endif %}
        {%- set system_content = system_content | replace('<|think_off|>', '') | trim %}
    {%- endif %}
    {%- if '<|think_on|>' in system_content %}
        {%- set ns_flags.enable_thinking = true %}
        {%- set system_content = system_content | replace('<|think_on|>', '') | trim %}
    {%- endif %}
{%- endif %}
{%- if tools and tools is iterable and tools is not mapping %}
    {{- '<|im_start|>system\n' }}
    {{- '# Tools\n\nYou have access to the following functions:\n\n<tools>' }}
    {%- for tool in tools %}
        {{- '\n' }}
        {%- set fn = tool.function if (tool is mapping and tool.function is defined) else tool %}
        {{- fn | tojson }}
    {%- endfor %}
    {{- '\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n' }}
    {%- if ns_flags.enable_thinking %}
        {{- '<think>\nBrief reasoning and plan\n</think>\n' }}
    {%- endif %}
    {{- '<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block nested within <tool_call></tool_call> XML tags.\n- Required parameters MUST be specified.\n- Place internal reasoning inside <think></think> and close </think> BEFORE outputting <tool_call>.\n- The <tool_call> and <function=...> tags MUST start at the beginning of a new line with NO indentation.\n- To call multiple functions, emit a separate closed <tool_call></tool_call> block for each function.\n- If no tool call is needed, answer the user normally right after </think>.\n</IMPORTANT>' }}
    {%- if has_first_system and system_content %}
        {{- '\n\n' + system_content }}
    {%- endif %}
    {{- '<|im_end|>\n' }}
{%- elif has_first_system and system_content %}
    {{- '<|im_start|>system\n' + system_content + '<|im_end|>\n' }}
{%- endif %}
{%- for message in messages %}
    {%- set role = message.role %}
    {%- set is_sys = (role == 'system' or role == 'developer') %}
    {%- set content = render_content(message.content) | trim %}
    {%- if '<|think_off|>' in content %}
        {%- set ns_flags.enable_thinking = false %}
        {%- set content = content | replace('<|think_off|>', '') | trim %}
    {%- elif '<|think_on|>' in content %}
        {%- set ns_flags.enable_thinking = true %}
        {%- set content = content | replace('<|think_on|>', '') | trim %}
    {%- endif %}
    {%- if is_sys %}
        {%- if not loop.first and content %}
            {{- '<|im_start|>system\n' + content + '<|im_end|>\n' }}
        {%- endif %}
    {%- elif role == 'user' %}
        {{- '<|im_start|>user\n' + content + '<|im_end|>\n' }}
    {%- elif role == 'assistant' %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is defined and message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content | trim %}
        {%- elif '</think>' in content %}
            {%- set reasoning_content = content.split('</think>')[0].split('<think>')[-1] | trim %}
            {%- set content = content.split('</think>')[-1] | trim %}
        {%- endif %}
        {%- if message.tool_calls and '<tool_call>' in content %}
            {%- set content = content.split('<tool_call>')[0] | trim %}
        {%- endif %}
        {{- '<|im_start|>assistant' }}
        {%- if reasoning_content and ns_flags.enable_thinking %}
            {{- '\n<think>\n' + reasoning_content + '\n</think>' }}
        {%- endif %}
        {%- if content %}
            {{- '\n' + content }}
        {%- elif not message.tool_calls %}
            {{- '\n' }}
        {%- endif %}
        {%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}
            {%- for tc in message.tool_calls %}
                {%- set fn = tc.function if (tc is mapping and tc.function is defined) else tc %}
                {{- '\n<tool_call>\n<function=' ~ fn.name ~ '>\n' }}
                {%- if fn.arguments is defined and fn.arguments is mapping %}
                    {%- for k in fn.arguments %}
                        {%- set v = fn.arguments[k] %}
                        {{- '<parameter=' ~ k ~ '>\n' }}
                        {%- if v is mapping or (v is iterable and v is not string) %}
                            {{- v | tojson }}
                        {%- else %}
                            {{- v | string }}
                        {%- endif %}
                        {{- '\n</parameter>\n' }}
                    {%- endfor %}
                {%- elif fn.arguments is defined and fn.arguments is string and (fn.arguments | trim) %}
                    {{- (fn.arguments | trim) ~ '\n' }}
                {%- endif %}
                {{- '</function>\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif role == 'tool' %}
        {%- if loop.first or messages[loop.index0 - 1].role != 'tool' %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' + content + '\n</tool_response>' }}
        {%- if loop.last or messages[loop.index0 + 1].role != 'tool' %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- else %}
        {{- '<|im_start|>' + role + '\n' + content + '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if ns_flags.enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}
    {%- endif %}
{%- endif %}"""


# Hardened Qwen3 / Qwen3-MoE / Qwen2.5 JSON tool-calling Jinja template
QWEN3_AGENTIC_JINJA_TEMPLATE = r"""{%- macro render_content(content) %}
    {%- if content is string %}
        {{- content }}
    {%- elif content is iterable and content is not mapping %}
        {%- for item in content %}
            {%- if item is string %}
                {{- item }}
            {%- elif item is mapping and 'text' in item %}
                {{- item.text }}
            {%- endif %}
        {%- endfor %}
    {%- elif content is none or content is undefined %}
        {{- '' }}
    {%- else %}
        {{- content | string }}
    {%- endif %}
{%- endmacro %}
{%- set has_first_system = messages and (messages[0].role == 'system' or messages[0].role == 'developer') %}
{%- set first_sys = render_content(messages[0].content) | trim if has_first_system else '' %}
{%- if tools and tools is iterable and tools is not mapping %}
    {{- '<|im_start|>system\n' }}
    {%- if first_sys %}
        {{- first_sys + '\n\n' }}
    {%- endif %}
    {{- '# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>' }}
    {%- for tool in tools %}
        {{- '\n' + (tool | tojson) }}
    {%- endfor %}
    {{- '\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n' }}
{%- elif has_first_system and first_sys %}
    {{- '<|im_start|>system\n' + first_sys + '<|im_end|>\n' }}
{%- endif %}
{%- for message in messages %}
    {%- set role = message.role %}
    {%- set is_sys = (role == 'system' or role == 'developer') %}
    {%- set content = render_content(message.content) | trim %}
    {%- if is_sys %}
        {%- if not loop.first and content %}
            {{- '<|im_start|>system\n' + content + '<|im_end|>\n' }}
        {%- endif %}
    {%- elif role == 'user' %}
        {{- '<|im_start|>user\n' + content + '<|im_end|>\n' }}
    {%- elif role == 'assistant' %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is defined and message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content | trim %}
        {%- elif '</think>' in content %}
            {%- set reasoning_content = content.split('</think>')[0].split('<think>')[-1] | trim %}
            {%- set content = content.split('</think>')[-1] | trim %}
        {%- endif %}
        {{- '<|im_start|>assistant\n' }}
        {%- if reasoning_content %}
            {{- '<think>\n' + reasoning_content + '\n</think>\n\n' }}
        {%- endif %}
        {{- content }}
        {%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}
            {%- for tc in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- set fn = tc.function if (tc is mapping and tc.function is defined) else tc %}
                {%- set args_json = fn.arguments if fn.arguments is string else (fn.arguments | tojson) %}
                {{- '<tool_call>\n{"name": "' ~ fn.name ~ '", "arguments": ' ~ args_json ~ '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif role == 'tool' %}
        {%- if loop.first or messages[loop.index0 - 1].role != 'tool' %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' + content + '\n</tool_response>' }}
        {%- if loop.last or messages[loop.index0 + 1].role != 'tool' %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- else %}
        {{- '<|im_start|>' + role + '\n' + content + '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}"""


def normalize_messages_for_template(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize OpenAI-format messages before Jinja rendering so NO template crashes on:
    - `role="developer"` (used by OpenCode, Codex CLI, OpenAI o-series)
    - `content=None` (sent on assistant tool_calls messages)
    - `tool_calls[*].function.arguments` passed as a JSON string when Jinja uses `|items`!
    """
    normalized: List[Dict[str, Any]] = []
    for raw_msg in messages:
        msg = dict(raw_msg)
        role = str(msg.get("role", "user"))
        if role == "developer":
            msg["role"] = "system"

        # Normalize content (handle None and OpenAI content-part arrays)
        content = msg.get("content")
        if content is None:
            msg["content"] = ""
        elif isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(str(item.get("text", "")))
            msg["content"] = "\n".join(parts)

        # Normalize assistant tool_calls so `function.arguments` is a parsed dict when valid JSON
        raw_tcs = msg.get("tool_calls")
        if isinstance(raw_tcs, list) and raw_tcs:
            clean_tcs: List[Dict[str, Any]] = []
            for tc in raw_tcs:
                if not isinstance(tc, dict):
                    continue
                tc_copy = dict(tc)
                fn = tc_copy.get("function")
                if isinstance(fn, dict):
                    fn_copy = dict(fn)
                    args = fn_copy.get("arguments", {})
                    if isinstance(args, str):
                        args_str = args.strip()
                        if args_str:
                            try:
                                parsed = json.loads(args_str)
                                if isinstance(parsed, dict):
                                    fn_copy["arguments"] = parsed
                            except Exception:
                                pass
                        else:
                            fn_copy["arguments"] = {}
                    elif args is None:
                        fn_copy["arguments"] = {}
                    tc_copy["function"] = fn_copy
                clean_tcs.append(tc_copy)
            msg["tool_calls"] = clean_tcs

        normalized.append(msg)
    return normalized


def apply_tool_choice_directive(
    messages: List[Dict[str, Any]],
    tools: Optional[Sequence[Dict[str, Any]]],
    tool_choice: Any,
) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    """Handle OpenAI `tool_choice` (`"none"`, `"auto"`, `"required"`, or specific function dict)."""
    if not tools or tool_choice == "none":
        return messages, None

    tool_list = list(tools)
    directive: Optional[str] = None
    if tool_choice == "required":
        directive = "IMPORTANT: You MUST call at least one tool from <tools> in this response using <tool_call>."
    elif isinstance(tool_choice, dict):
        fn_name = (
            tool_choice.get("function", {}).get("name")
            if isinstance(tool_choice.get("function"), dict)
            else tool_choice.get("name")
        )
        if fn_name:
            tool_list = [
                t
                for t in tool_list
                if (t.get("function", {}).get("name") if isinstance(t.get("function"), dict) else t.get("name"))
                == fn_name
            ] or tool_list
            directive = f'IMPORTANT: You MUST call the function "{fn_name}" in this response using <tool_call>.'

    if directive:
        msgs = [dict(m) for m in messages]
        if msgs and msgs[0].get("role") == "system":
            msgs[0]["content"] = f"{msgs[0].get('content', '')}\n\n{directive}".strip()
        else:
            msgs.insert(0, {"role": "system", "content": directive})
        return msgs, tool_list

    return messages, tool_list


def resolve_enable_thinking(
    messages: Sequence[Dict[str, Any]],
    explicit_enable_thinking: Optional[bool] = None,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
    reasoning_effort: Optional[str] = None,
    thinking_param: Optional[Dict[str, Any]] = None,
    default_enable_thinking: bool = True,
) -> bool:
    """
    Unified reasoning-mode resolver supporting OpenAI, vLLM, Anthropic/OpenCode, and Qwen conventions:
    1. Explicit `enable_thinking: bool`
    2. `chat_template_kwargs: {"enable_thinking": bool}`
    3. `thinking: {"type": "disabled" | "enabled"}` (Claude Code / OpenCode / LiteLLM)
    4. `reasoning_effort: "none"` -> False; `"low" | "medium" | "high"` -> True (OpenAI o-series)
    5. Inline `/no_think`, `/think`, `<|think_off|>`, `<|think_on|>` tags in system/user messages
    6. Server-wide `default_enable_thinking`
    """
    if explicit_enable_thinking is not None:
        return bool(explicit_enable_thinking)
    if chat_template_kwargs and "enable_thinking" in chat_template_kwargs:
        return bool(chat_template_kwargs["enable_thinking"])
    if isinstance(thinking_param, dict) and "type" in thinking_param:
        t_type = str(thinking_param["type"]).lower()
        if t_type in ("disabled", "off", "none", "false"):
            return False
        if t_type in ("enabled", "on", "true", "adaptive"):
            return True
    if reasoning_effort is not None:
        eff = str(reasoning_effort).strip().lower()
        if eff in ("none", "off", "disabled", "false", "0"):
            return False
        if eff in ("low", "medium", "high", "minimal", "max"):
            return True

    # Check inline directives in messages (latest directive wins)
    state: Optional[bool] = None
    for m in messages:
        if m.get("role") in ("system", "developer", "user"):
            txt = str(m.get("content", ""))
            if "/no_think" in txt or "<|think_off|>" in txt:
                state = False
            if "/think" in txt or "<|think_on|>" in txt:
                # Ensure `/no_think` didn't falsely match `/think`
                cleaned = txt.replace("/no_think", "")
                if "/think" in cleaned or "<|think_on|>" in cleaned:
                    state = True
    if state is not None:
        return state

    return default_enable_thinking


# ==============================================================================
# 2. Schema-Aware Dual XML (`qwen3_xml`) + JSON (`qwen3_json`) Tool Call Parser
# ==============================================================================

@dataclass
class ParsedToolCall:
    id: str
    name: str
    arguments_json: str

    def to_openai_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments_json,
            },
        }

    def to_openai_stream_delta(self, index: int) -> Dict[str, Any]:
        return {
            "index": index,
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments_json,
            },
        }


def _build_tool_param_schema_map(
    tools: Optional[Sequence[Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """Map `func_name -> {param_name -> json_schema_dict}` for accurate XML value type coercion."""
    schema_map: Dict[str, Dict[str, Any]] = {}
    if not tools:
        return schema_map
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        params = fn.get("parameters")
        if isinstance(name, str) and isinstance(params, dict):
            props = params.get("properties")
            if isinstance(props, dict):
                schema_map[name] = {str(k): (v if isinstance(v, dict) else {}) for k, v in props.items()}
            else:
                schema_map[name] = {}
    return schema_map


def _extract_schema_types(prop_schema: Dict[str, Any]) -> List[str]:
    """Extract allowed JSON schema types (handling `type: "integer"`, `type: ["integer", "null"]`, `anyOf`, `oneOf`)."""
    types: List[str] = []
    t = prop_schema.get("type")
    if isinstance(t, str):
        types.append(t)
    elif isinstance(t, list):
        types.extend(str(x) for x in t)
    for key in ("anyOf", "oneOf"):
        sub = prop_schema.get(key)
        if isinstance(sub, list):
            for item in sub:
                if isinstance(item, dict):
                    types.extend(_extract_schema_types(item))
    return types


def _coerce_xml_param_value(raw_val: str, prop_schema: Optional[Dict[str, Any]]) -> Any:
    """
    Coerce a raw XML parameter string to the expected JSON Schema type WITHOUT corrupting
    multi-line source code or strings containing `<`, `>`, `&`, or JSON snippets!
    """
    allowed = _extract_schema_types(prop_schema) if prop_schema else []
    stripped = raw_val.strip()

    # If schema explicitly says ONLY string, preserve exact string (do NOT turn "123" or "{}" into int/dict!)
    if allowed and all(t in ("string", "null") for t in allowed):
        if "null" in allowed and stripped.lower() == "null":
            return None
        return raw_val

    if stripped.lower() == "null" and (not allowed or "null" in allowed):
        return None

    if not allowed or "boolean" in allowed:
        if stripped.lower() == "true" and ("boolean" in allowed or not allowed):
            if "string" not in allowed:
                return True
        if stripped.lower() == "false" and ("boolean" in allowed or not allowed):
            if "string" not in allowed:
                return False

    if "integer" in allowed:
        try:
            return int(stripped)
        except ValueError:
            pass

    if "number" in allowed:
        try:
            return float(stripped)
        except ValueError:
            pass

    if "object" in allowed or "array" in allowed:
        try:
            return json.loads(stripped)
        except Exception:
            pass

    # Schema-less fallback: parse obvious JSON objects/arrays, booleans, or numbers if unambiguous
    if not allowed:
        if (stripped.startswith("{") and stripped.endswith("}")) or (
            stripped.startswith("[") and stripped.endswith("]")
        ):
            try:
                return json.loads(stripped)
            except Exception:
                pass
        if stripped.lower() == "true":
            return True
        if stripped.lower() == "false":
            return False
        if re.fullmatch(r"-?(?:0|[1-9]\d*)", stripped):
            try:
                return int(stripped)
            except ValueError:
                pass
        if re.fullmatch(r"-?(?:0|[1-9]\d*)\.\d+", stripped):
            try:
                return float(stripped)
            except ValueError:
                pass

    return raw_val


_FUNC_HEADER_RE = re.compile(r"<function\s*=\s*([^>\s]+)\s*>", re.DOTALL)
_PARAM_OPEN_RE = re.compile(r"<parameter\s*=\s*([^>\s]+)\s*>", re.DOTALL)


def parse_single_tool_call_block(
    block_body: str,
    schema_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[ParsedToolCall]:
    """
    Parse the interior of a `<tool_call> ... </tool_call>` block.
    Supports:
      1. Qwen3.5 / Qwen3.8 / Qwen3-Coder XML: `<function=name><parameter=k>v</parameter></function>`
         (Uses exact `<parameter=...>` ... `</parameter>` boundary slicing so `<` and `>` inside code NEVER break!)
      2. Qwen3 / Qwen2.5 / Hermes JSON: `{"name": "fn", "arguments": {...}}` (or JSON list of calls)
    """
    text = block_body.strip()
    if not text:
        return []

    results: List[ParsedToolCall] = []

    # --- Format 1: XML `<function=NAME>...</function>` ---
    fn_matches = list(_FUNC_HEADER_RE.finditer(text))
    if fn_matches:
        for idx, fm in enumerate(fn_matches):
            fn_name = fm.group(1).strip().strip("\"'")
            body_start = fm.end()
            next_fn_start = fn_matches[idx + 1].start() if idx + 1 < len(fn_matches) else len(text)
            fn_chunk = text[body_start:next_fn_start]
            end_fn_pos = fn_chunk.rfind("</function>")
            if end_fn_pos != -1:
                fn_chunk = fn_chunk[:end_fn_pos]

            fn_props = (schema_map or {}).get(fn_name, {})
            args_dict: Dict[str, Any] = {}

            # Scan `<parameter=KEY>` ... `</parameter>` sequentially (safe for `<` / `>` inside values!)
            p_opens = list(_PARAM_OPEN_RE.finditer(fn_chunk))
            if p_opens:
                for p_idx, pm in enumerate(p_opens):
                    p_name = pm.group(1).strip().strip("\"'")
                    val_start = pm.end()
                    val_limit = p_opens[p_idx + 1].start() if p_idx + 1 < len(p_opens) else len(fn_chunk)
                    raw_segment = fn_chunk[val_start:val_limit]
                    close_pos = raw_segment.rfind("</parameter>")
                    if close_pos != -1:
                        raw_val = raw_segment[:close_pos]
                    else:
                        raw_val = raw_segment
                    # Strip the single formatting newline right after `<parameter=k>\n` and right before `\n</parameter>`
                    if raw_val.startswith("\r\n"):
                        raw_val = raw_val[2:]
                    elif raw_val.startswith("\n"):
                        raw_val = raw_val[1:]
                    if raw_val.endswith("\r\n"):
                        raw_val = raw_val[:-2]
                    elif raw_val.endswith("\n"):
                        raw_val = raw_val[:-1]

                    args_dict[p_name] = _coerce_xml_param_value(raw_val, fn_props.get(p_name))
            else:
                # Fallback if model put raw JSON inside `<function=name>{...}</function>`
                inner = fn_chunk.strip()
                if inner.startswith("{"):
                    try:
                        parsed_inner = json.loads(inner)
                        if isinstance(parsed_inner, dict):
                            args_dict = parsed_inner
                    except Exception:
                        pass

            results.append(
                ParsedToolCall(
                    id=f"call_{uuid.uuid4().hex[:24]}",
                    name=fn_name,
                    arguments_json=json.dumps(args_dict, ensure_ascii=False),
                )
            )
        return results

    # --- Format 2: JSON `{"name": ..., "arguments": ...}` (or `[...]`) ---
    # Strip optional markdown fences if a smaller model wrapped JSON in ```json ... ```
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    candidates: List[Any] = []
    try:
        loaded = json.loads(text)
        candidates = loaded if isinstance(loaded, list) else [loaded]
    except Exception:
        # Try line-by-line or brace-balanced extraction
        decoder = json.JSONDecoder()
        pos = 0
        while pos < len(text):
            brace = text.find("{", pos)
            if brace == -1:
                break
            try:
                obj, end_idx = decoder.raw_decode(text, brace)
                candidates.append(obj)
                pos = end_idx
            except Exception:
                pos = brace + 1

    for item in candidates:
        if not isinstance(item, dict):
            continue
        fn_obj = item.get("function") if isinstance(item.get("function"), dict) else item
        name = fn_obj.get("name") or fn_obj.get("tool")
        if not isinstance(name, str) or not name.strip():
            continue
        args = fn_obj.get("arguments", fn_obj.get("parameters", {}))
        if isinstance(args, str):
            try:
                json.loads(args)
                args_json = args
            except Exception:
                args_json = json.dumps({"raw": args}, ensure_ascii=False)
        else:
            args_json = json.dumps(args if args is not None else {}, ensure_ascii=False)
        results.append(
            ParsedToolCall(
                id=f"call_{uuid.uuid4().hex[:24]}",
                name=name.strip(),
                arguments_json=args_json,
            )
        )
    return results


# ==============================================================================
# 3. Incremental Streaming & Non-Streaming Reasoning + Tool State Machine
# ==============================================================================

_THINK_OPEN_TAGS = ("<think>", "<thinking>", "<redacted_thinking>")
_THINK_CLOSE_TAGS = ("</think>", "</thinking>", "</redacted_thinking>")
_TOOL_OPEN_TAGS = ("<tool_call>", "<|python_tag|>")
_TOOL_CLOSE_TAG = "</tool_call>"


def _longest_partial_tag_suffix(text: str, tags: Sequence[str]) -> int:
    """
    Return the length of the longest suffix of `text` that is a proper non-empty prefix
    of any tag in `tags`. Returns 0 immediately when no tag prefix matches at the end!
    """
    if not text:
        return 0
    max_len = max(len(t) for t in tags)
    tail = text[-max_len:]
    for k in range(min(len(tail), max_len - 1), 0, -1):
        suffix = tail[-k:]
        for tag in tags:
            if tag.startswith(suffix):
                return k
    return 0


def _find_earliest_tag(text: str, tags: Sequence[str]) -> Tuple[int, str]:
    best_pos = -1
    best_tag = ""
    for tag in tags:
        p = text.find(tag)
        if p != -1 and (best_pos == -1 or p < best_pos):
            best_pos = p
            best_tag = tag
    return best_pos, best_tag


@dataclass
class StreamParseEmission:
    reasoning_delta: str = ""
    content_delta: str = ""
    tool_calls: List[ParsedToolCall] = field(default_factory=list)


class ReasoningAndToolStreamParser:
    """
    Real-time incremental parser for both SSE streaming and non-streaming `/v1/chat/completions`:
    - Separates `<think>...</think>` into `reasoning_delta` (NEVER emitting `<think></think>` into `content_delta`!).
    - Suppresses pure-whitespace empty think blocks (`<think>\\n\\n</think>`) completely!
    - Handles prompts that pre-open `<think>\\n` (`Qwen3.5` / `Qwen3.8`) as well as models that emit `<think>` themselves (`Qwen3`).
    - Intercepts `<tool_call>...</tool_call>` blocks (including auto-closing unclosed `<think>` before `<tool_call>`)
      and converts them into structured OpenAI `ParsedToolCall` objects!
    """

    def __init__(
        self,
        prompt_str: str,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        parse_tool_calls: bool = True,
        emit_reasoning: bool = True,
    ) -> None:
        self.schema_map = _build_tool_param_schema_map(tools)
        self.parse_tool_calls = parse_tool_calls
        self.emit_reasoning = emit_reasoning

        rstripped = prompt_str.rstrip(" \t\r\n")
        self.prompt_opened_think = any(rstripped.endswith(t) for t in _THINK_OPEN_TAGS)

        self._state: Literal["START", "IN_THINK", "AFTER_THINK", "IN_CONTENT", "IN_TOOL"] = (
            "IN_THINK" if self.prompt_opened_think else "START"
        )
        self._buf: str = ""
        self._think_started_non_ws: bool = False
        self._active_tool_open_tag: str = "<tool_call>"

        # Full accumulated outputs for non-streaming summary & finish_reason inspection
        self.reasoning_parts: List[str] = []
        self.content_parts: List[str] = []
        self.parsed_tool_calls: List[ParsedToolCall] = []

    def feed(self, delta_text: str) -> StreamParseEmission:
        if not delta_text:
            return StreamParseEmission()
        self._buf += delta_text
        return self._drain(final=False)

    def finalize(self) -> StreamParseEmission:
        return self._drain(final=True)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.parsed_tool_calls) > 0

    @property
    def full_reasoning(self) -> str:
        return "".join(self.reasoning_parts).strip()

    @property
    def full_content(self) -> str:
        return "".join(self.content_parts)

    def _emit_reasoning_piece(self, text: str, out: StreamParseEmission) -> None:
        if not text:
            return
        if not self._think_started_non_ws:
            stripped_lead = text.lstrip(" \t\r\n")
            if not stripped_lead:
                return
            text = stripped_lead
            self._think_started_non_ws = True
        self.reasoning_parts.append(text)
        if self.emit_reasoning:
            out.reasoning_delta += text

    def _emit_content_piece(self, text: str, out: StreamParseEmission) -> None:
        if not text:
            return
        self.content_parts.append(text)
        out.content_delta += text

    def _drain(self, final: bool) -> StreamParseEmission:
        out = StreamParseEmission()

        while self._buf:
            if self._state == "START":
                stripped = self._buf.lstrip(" \t\r\n")
                if not stripped:
                    if final:
                        self._buf = ""
                    break

                # 1. Model opened `<think>` at start of reply
                matched_open = next((t for t in _THINK_OPEN_TAGS if stripped.startswith(t)), None)
                if matched_open is not None:
                    self._buf = stripped[len(matched_open) :]
                    self._think_started_non_ws = False
                    self._state = "IN_THINK"
                    continue

                # 2. Orphan `</think>` at start when thinking was bypassed in prompt
                matched_orphan_close = next((t for t in _THINK_CLOSE_TAGS if stripped.startswith(t)), None)
                if matched_orphan_close is not None:
                    self._buf = stripped[len(matched_orphan_close) :]
                    self._state = "AFTER_THINK"
                    continue

                # 3. Direct `<tool_call>` at start of reply
                if self.parse_tool_calls:
                    matched_tool = next((t for t in _TOOL_OPEN_TAGS if stripped.startswith(t)), None)
                    if matched_tool is not None:
                        self._active_tool_open_tag = matched_tool
                        self._buf = stripped[len(matched_tool) :]
                        self._state = "IN_TOOL"
                        continue

                # 4. Wait if `stripped` is still a partial prefix of `<think>` / `</think>` / `<tool_call>`
                start_watch = list(_THINK_OPEN_TAGS) + list(_THINK_CLOSE_TAGS)
                if self.parse_tool_calls:
                    start_watch.extend(_TOOL_OPEN_TAGS)
                if not final and any(tag.startswith(stripped) for tag in start_watch):
                    break

                # Otherwise normal assistant content begins immediately!
                self._state = "IN_CONTENT"
                continue

            if self._state == "IN_THINK":
                close_pos, close_tag = _find_earliest_tag(self._buf, _THINK_CLOSE_TAGS)
                # Also guard against unclosed `<think>` going straight into `<tool_call>`!
                tool_pos = self._buf.find("<tool_call>") if self.parse_tool_calls else -1

                if close_pos != -1 and (tool_pos == -1 or close_pos < tool_pos):
                    before = self._buf[:close_pos]
                    if self._think_started_non_ws:
                        before = before.rstrip(" \t\r\n")
                    self._emit_reasoning_piece(before, out)
                    self._buf = self._buf[close_pos + len(close_tag) :]
                    self._state = "AFTER_THINK"
                    continue

                if tool_pos != -1:
                    before = self._buf[:tool_pos].rstrip(" \t\r\n")
                    self._emit_reasoning_piece(before, out)
                    self._active_tool_open_tag = "<tool_call>"
                    self._buf = self._buf[tool_pos + len("<tool_call>") :]
                    self._state = "IN_TOOL"
                    continue

                if final:
                    # Stream ended before `</think>` (e.g. hit `max_tokens` during reasoning)
                    self._emit_reasoning_piece(self._buf, out)
                    self._buf = ""
                    break

                # If we haven't seen non-whitespace yet, hold pure whitespace in buffer
                # so empty `<think>\n\n</think>` emits 0 reasoning chunks!
                if not self._think_started_non_ws and not self._buf.strip(" \t\r\n"):
                    break

                watch_tags = list(_THINK_CLOSE_TAGS) + (["<tool_call>"] if self.parse_tool_calls else [])
                hold = _longest_partial_tag_suffix(self._buf, watch_tags)
                safe_len = len(self._buf) - hold
                if safe_len > 0:
                    self._emit_reasoning_piece(self._buf[:safe_len], out)
                    self._buf = self._buf[safe_len:]
                break

            if self._state == "AFTER_THINK":
                # Strip formatting newlines right after `</think>` before content or `<tool_call>`
                stripped = self._buf.lstrip(" \t\r\n")
                if not stripped:
                    if not final:
                        self._buf = ""
                    else:
                        self._buf = ""
                    break
                self._buf = stripped
                # Check if another `<think>` or orphan `</think>` or immediate `<tool_call>` follows
                if any(self._buf.startswith(t) for t in _THINK_OPEN_TAGS) or any(
                    self._buf.startswith(t) for t in _THINK_CLOSE_TAGS
                ):
                    self._state = "START"
                    continue
                if self.parse_tool_calls:
                    matched_tool = next((t for t in _TOOL_OPEN_TAGS if self._buf.startswith(t)), None)
                    if matched_tool is not None:
                        self._active_tool_open_tag = matched_tool
                        self._buf = self._buf[len(matched_tool) :]
                        self._state = "IN_TOOL"
                        continue
                    if not final and any(t.startswith(self._buf) for t in _TOOL_OPEN_TAGS):
                        break
                self._state = "IN_CONTENT"
                continue

            if self._state == "IN_CONTENT":
                watch_content: List[str] = list(_THINK_CLOSE_TAGS)
                if self.parse_tool_calls:
                    watch_content.extend(_TOOL_OPEN_TAGS)

                pos, tag = _find_earliest_tag(self._buf, watch_content)
                if pos != -1:
                    if tag in _THINK_CLOSE_TAGS:
                        # Strip stray orphan `</think>` in mid-stream
                        self._emit_content_piece(self._buf[:pos], out)
                        self._buf = self._buf[pos + len(tag) :].lstrip("\r\n")
                        continue
                    # Tool call opened! Trim trailing whitespace right before `<tool_call>`
                    before_tool = self._buf[:pos].rstrip(" \t\r\n")
                    if before_tool:
                        self._emit_content_piece(before_tool, out)
                    self._active_tool_open_tag = tag
                    self._buf = self._buf[pos + len(tag) :]
                    self._state = "IN_TOOL"
                    continue

                if final:
                    self._emit_content_piece(self._buf, out)
                    self._buf = ""
                    break

                hold = _longest_partial_tag_suffix(self._buf, watch_content)
                safe_len = len(self._buf) - hold
                if safe_len > 0:
                    self._emit_content_piece(self._buf[:safe_len], out)
                    self._buf = self._buf[safe_len:]
                break

            if self._state == "IN_TOOL":
                if self._active_tool_open_tag == "<|python_tag|>":
                    # Llama-3 `<|python_tag|>` runs until EOS (`final=True`)
                    if final:
                        calls = parse_single_tool_call_block(self._buf, self.schema_map)
                        self.parsed_tool_calls.extend(calls)
                        out.tool_calls.extend(calls)
                        self._buf = ""
                    break

                close_idx = self._buf.find(_TOOL_CLOSE_TAG)
                if close_idx != -1:
                    raw_block = self._buf[:close_idx]
                    self._buf = self._buf[close_idx + len(_TOOL_CLOSE_TAG) :]
                    calls = parse_single_tool_call_block(raw_block, self.schema_map)
                    self.parsed_tool_calls.extend(calls)
                    out.tool_calls.extend(calls)
                    # After `</tool_call>`, transition to `AFTER_THINK` so consecutive `<tool_call>` blocks
                    # or trailing text are handled seamlessly!
                    self._state = "AFTER_THINK"
                    continue

                if final:
                    # Auto-heal if generation stopped right after `</function>` or `}` before `</tool_call>`
                    calls = parse_single_tool_call_block(self._buf, self.schema_map)
                    if calls:
                        self.parsed_tool_calls.extend(calls)
                        out.tool_calls.extend(calls)
                    self._buf = ""
                break

        return out
