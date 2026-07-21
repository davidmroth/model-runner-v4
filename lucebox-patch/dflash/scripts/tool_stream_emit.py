"""OpenAI-style incremental tool_calls deltas while Qwen XML accumulates.

The model still emits ``<tool_call><function=NAME><parameter=…>`` XML.
This helper watches the growing buffer and yields OpenAI chat-completion
``delta.tool_calls`` fragments:

1. When the function name is committed → one chunk with ``name`` + empty args
2. When each parameter closes → append-only ``arguments`` JSON suffix
3. When a call ends → advance ``index`` for the next tool

Final batch ``parse_tool_calls`` remains the source of truth at EOS; callers
should skip a second full emit when ``streamed_any`` is set.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

# Closed parameters only — do not treat EOS as a terminator (unsafe mid-stream).
_STREAM_PARAM_RE = re.compile(
    r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>",
    re.DOTALL,
)
_FUNCTION_OPEN_RE = re.compile(r"<function=([^>\s]+)>")
_FUNCTION_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


@dataclass
class ToolStreamState:
    """Mutable emit state for one HTTP stream's tool_buffer."""

    index: int = 0
    call_id: str | None = None
    name: str | None = None
    name_emitted: bool = False
    args_emitted: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    emitted_param_keys: set[str] = field(default_factory=set)
    streamed_any: bool = False
    # How much of tool_buffer we have already scanned for completed calls.
    scan_from: int = 0


def _new_call_id() -> str:
    return "call_" + uuid.uuid4().hex[:24]


def _tool_call_delta(
    *,
    index: int,
    call_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> dict[str, Any]:
    fn: dict[str, Any] = {}
    if name is not None:
        fn["name"] = name
    if arguments is not None:
        fn["arguments"] = arguments
    entry: dict[str, Any] = {"index": index}
    if call_id is not None:
        entry["id"] = call_id
        entry["type"] = "function"
    if fn:
        entry["function"] = fn
    return {"tool_calls": [entry]}


def _convert_param_value(param_value: str, param_name: str, param_config: dict) -> Any:
    """Same coercion rules as server_tools._convert_param_value (closed values)."""
    import ast

    if param_value.lower() == "null":
        return None
    if param_name not in param_config:
        return param_value
    cfg = param_config[param_name]
    if isinstance(cfg, dict) and "type" in cfg:
        ptype = str(cfg["type"]).strip().lower()
    elif isinstance(cfg, dict) and "anyOf" in cfg:
        ptype = "object"
    else:
        ptype = "string"
    if ptype in ("string", "str", "text", "varchar", "char", "enum"):
        return param_value
    if any(ptype.startswith(p) for p in ("int", "uint", "long", "short", "unsigned")):
        try:
            return int(param_value)
        except (ValueError, TypeError):
            return param_value
    if ptype.startswith("num") or ptype.startswith("float"):
        try:
            f = float(param_value)
            return f if f - int(f) != 0 else int(f)
        except (ValueError, TypeError):
            return param_value
    if ptype in ("boolean", "bool", "binary"):
        return param_value.lower() == "true"
    if (
        ptype in ("object", "array", "arr")
        or ptype.startswith("dict")
        or ptype.startswith("list")
    ):
        try:
            return json.loads(param_value)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    try:
        return ast.literal_eval(param_value)
    except (ValueError, SyntaxError, TypeError):
        return param_value


def _find_tool_properties(tools, function_name: str) -> dict:
    for t in tools or []:
        fn = t.function if hasattr(t, "function") else t.get("function", {})
        if hasattr(fn, "model_dump"):
            fn = fn.model_dump()
        if fn.get("name") == function_name:
            params = fn.get("parameters", {})
            if isinstance(params, dict):
                return params.get("properties", {})
    return {}


def _reset_call(state: ToolStreamState) -> None:
    state.call_id = None
    state.name = None
    state.name_emitted = False
    state.args_emitted = ""
    state.params = {}
    state.emitted_param_keys = set()


def _open_args_json(params: dict[str, Any]) -> str:
    """JSON object text without the final ``}`` so later keys can append."""
    if not params:
        return ""
    dumped = json.dumps(params, ensure_ascii=False)
    if dumped.endswith("}"):
        return dumped[:-1]
    return dumped


def _emit_args_suffix(state: ToolStreamState, *, closing: bool = False) -> dict[str, Any] | None:
    """Emit append-only arguments fragments (OpenAI concatenation semantics)."""
    if closing:
        if not state.name_emitted:
            return None
        if not state.args_emitted:
            # Name only, no params — emit empty object in one shot.
            state.args_emitted = "{}"
            state.streamed_any = True
            return _tool_call_delta(index=state.index, arguments="{}")
        if state.args_emitted.endswith("}"):
            return None
        state.args_emitted += "}"
        state.streamed_any = True
        return _tool_call_delta(index=state.index, arguments="}")

    open_json = _open_args_json(state.params)
    if not open_json:
        return None
    if not open_json.startswith(state.args_emitted):
        return None
    suffix = open_json[len(state.args_emitted) :]
    if not suffix:
        return None
    state.args_emitted = open_json
    state.streamed_any = True
    return _tool_call_delta(index=state.index, arguments=suffix)


def feed_tool_stream(
    tool_buffer: str,
    state: ToolStreamState,
    tools=None,
) -> list[dict[str, Any]]:
    """Scan *tool_buffer* and return zero or more OpenAI delta objects."""
    out: list[dict[str, Any]] = []

    def _commit_name(region: str) -> None:
        nonlocal out
        if state.name_emitted:
            return
        m = _FUNCTION_OPEN_RE.search(region)
        if not m:
            return
        name = m.group(1).strip()
        if not _FUNCTION_NAME_RE.match(name):
            return
        state.name = name
        state.call_id = _new_call_id()
        state.name_emitted = True
        state.streamed_any = True
        out.append(
            _tool_call_delta(
                index=state.index,
                call_id=state.call_id,
                name=name,
                arguments="",
            )
        )

    def _commit_closed_params(region: str) -> None:
        nonlocal out
        if not state.name_emitted or not state.name:
            return
        param_config = _find_tool_properties(tools, state.name)
        for pm in _STREAM_PARAM_RE.finditer(region):
            key = pm.group(1).strip()
            if not key or not _FUNCTION_NAME_RE.match(key):
                continue
            if key in state.emitted_param_keys:
                continue
            raw_val = pm.group(2)
            state.params[key] = _convert_param_value(raw_val, key, param_config)
            state.emitted_param_keys.add(key)
            delta = _emit_args_suffix(state)
            if delta:
                out.append(delta)

    while state.scan_from <= len(tool_buffer):
        region = tool_buffer[state.scan_from :]
        if not region:
            break

        before = len(out)
        _commit_name(region)
        _commit_closed_params(region)

        end_fn = region.find("</function>")
        end_tc = region.find("</tool_call>")
        ends = [i for i in (end_fn, end_tc) if i != -1]
        if ends and state.name_emitted:
            # Flush params once more for anything closed just before the tag.
            _commit_closed_params(region)
            close_delta = _emit_args_suffix(state, closing=True)
            if close_delta:
                out.append(close_delta)
            end_at = min(ends)
            if end_fn != -1 and end_fn == end_at:
                close_len = len("</function>")
            else:
                close_len = len("</tool_call>")
            abs_end = state.scan_from + end_at + close_len
            tc_close = tool_buffer.find("</tool_call>", state.scan_from)
            if tc_close != -1 and tc_close <= abs_end + 32:
                abs_end = max(abs_end, tc_close + len("</tool_call>"))
            state.scan_from = abs_end
            state.index += 1
            _reset_call(state)
            continue

        # No call closed — stop until more buffer arrives.
        if len(out) == before:
            break
        # Name/args progressed but call still open — wait for more tokens.
        break

    return out


def should_skip_final_tool_emit(state: ToolStreamState) -> bool:
    """True when incremental emit already sent tool_calls; finish-only at EOS."""
    return bool(state.streamed_any)
