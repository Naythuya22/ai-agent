"""Cloud backends for Agent Mode plus the WiFi → local fallback chain.

Chat Mode never imports this file — the whole point of the split is that a
cloud round-trip (or a failed one) can only ever change Agent Mode's replies.

Chain, in order::

    no usable network      → local Gemma 4B
    network + cloud on     → providers in fallback order
                              → healthy keys round-robin inside a provider
                              → 429 parks that key, next key is tried
                              → all keys of a provider gone → next provider
    every provider failed  → local Gemma 4B

HTTP goes through :func:`_http_json`, which normalises every failure into a
:class:`CloudError` carrying `kind` (``rate_limit`` / ``auth`` / ``server`` /
``network`` / ``bad_response``) plus a parsed `retry_after`. Tests replace it
outright, so no test ever opens a socket.
"""
from __future__ import annotations

import copy
import inspect
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
import urllib.error
import urllib.request
from typing import Any, Callable

import config
import core.net as net
from core import context_trim as CT
from core import model_registry as MR
from core.key_manager import KeyManager, mask, redact
from core.log import get_logger

log = get_logger("agent_backend")

HTTP_TIMEOUT = 30

MAX_PARALLEL_TOOLS = 4

# A context overflow is answered by trimming harder and trying once — see
# `AgentBackend.step`. Not a full second walk of the fallback chain: the other
# providers share this transcript and will refuse it for the same reason.
CONTEXT_RETRY_FACTOR = 0.75

# Provider error bodies can echo an API key back; anything that reaches
# a log line or an exception message goes through this first.
_KEY_RE = re.compile(r"sk-[A-Za-z0-9]{10,}")


def _sanitize(body: str, max_len: int = 500) -> str:
    """Mask any API key, then cap — safe to embed in logs and errors."""
    return _KEY_RE.sub("sk-****", str(body or ""))[:max_len]


# USD per 1K input tokens — shown in Settings before a cloud model is used.
COST_PER_1K: dict = {
    "openai:gpt-4o": 2.50,
    "openai:gpt-4o-mini": 0.15,
    "gemini:gemini-2.0-flash": 0.10,
    "gemini:gemini-1.5-flash": 0.075,
    "anthropic:claude-3-5-haiku-latest": 0.80,
    "anthropic:claude-3-5-sonnet-latest": 3.00,
    "openrouter:openai/gpt-4o-mini": 0.15,
    "openrouter:anthropic/claude-3.5-haiku": 0.80,
    "openrouter:google/gemini-2.0-flash": 0.10,
    "openrouter:google/gemini-2.5-flash:free": 0.00,
    "gemini:gemini-2.5-flash": 0.30,
    "gemini:gemini-2.5-pro": 1.25,
    "deepseek:deepseek-chat": 0.27,
    "deepseek:deepseek-reasoner": 0.55,
}
COST_FALLBACK = {"openai": 0.50, "gemini": 0.10, "openrouter": 0.30,
                 "anthropic": 1.00, "deepseek": 0.30}

# Preset model choices offered in Settings; free text is always allowed too.
MODELS: dict = {
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "o4-mini"],
    "gemini": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash",
               "gemini-2.5-flash-lite"],
    "openrouter": ["google/gemini-2.5-flash", "openai/gpt-4o-mini",
                   "anthropic/claude-3.5-haiku", "deepseek/deepseek-chat",
                   "meta-llama/llama-3.1-8b-instruct"],
    "anthropic": ["claude-3-5-haiku-latest", "claude-3-5-sonnet-latest",
                  "claude-3-opus-latest"],
    "deepseek": ["deepseek-chat", "deepseek-reasoner"],
}


# Tools that must run serially even if the model asks in one batch.
# Read-only tools are safe to parallelize; anything that mutates
# the filesystem or runs arbitrary code is not.
SERIAL_ONLY_TOOLS = frozenset({
    "write_file", "run_python",
})


def cost_per_1k(provider: str, model: str) -> float | None:
    """`$ / 1K input tokens`, or None when we simply do not know."""
    key = "%s:%s" % (provider, (model or "").strip().lower())
    if key in COST_PER_1K:
        return COST_PER_1K[key]
    return COST_FALLBACK.get(provider)


# Model ids this build does not know about are legal — Settings takes free text
# for a custom endpoint or a provider that renamed a model — so this only ever
# warns. Logged once per id per process: `step()` asks on every turn and
# `status()` on every Settings poll.
_WARNED_MODELS: set = set()


def _warn_unknown_model(provider: str, model: str) -> None:
    """Say once that a configured model is not one of the known presets.

    A wrong id is the single most common reason a cloud turn answers
    `model_not_found`, and the log line that says so is cheaper to find than
    the 404 itself.
    """
    name = str(model or "").strip()
    known = MODELS.get(provider) or []
    if not name or not known or name in known:
        return
    tag = "%s/%s" % (provider, name)
    if tag in _WARNED_MODELS:
        return
    _WARNED_MODELS.add(tag)
    log.warning("configured model %s is not one of the known %s ids (%s) — "
                "if the provider answers model_not_found, this is the reason",
                tag, provider, ", ".join(known))


class CloudError(Exception):
    """Any provider failure, normalised."""

    def __init__(self, kind: str, message: str = "", status: int = 0,
                 retry_after: float | None = None):
        super().__init__(message or kind)
        self.kind = kind          # rate_limit | auth | server | network | bad_response
        self.status = status
        self.retry_after = retry_after
        self.message = message


class BackendExhausted(Exception):
    """Every provider ran out of usable keys — the caller drops to local."""


class Step:
    """One model turn, in the shape the agent loop already understands."""

    def __init__(self, text: str = "", tool_calls: list | None = None,
                 usage: dict | None = None, model: str = "",
                 provider: str = "", key_id: str = "",
                 reasoning: str = ""):
        self.text = text or ""
        self.tool_calls = tool_calls or []   # [{"function":{"name","arguments"}}]
        self.usage = usage or {}
        self.model = model
        self.provider = provider
        self.key_id = key_id
        # DeepSeek's thinking models return `reasoning_content`, and
        # the API refuses the next turn (`400 ... must be passed back
        # to the API`) unless it is replayed with the assistant turn.
        self.reasoning = reasoning or ""


# ───────────────────────────── HTTP ─────────────────────────────

def _http_json(url: str, payload: Any, headers: dict,
               timeout: float = HTTP_TIMEOUT) -> dict:
    data = payload if isinstance(payload, (bytes, bytearray)) else \
        json.dumps(payload, ensure_ascii=False).encode("utf-8")
    hdrs = {"Content-Type": "application/json", "User-Agent": "ai-agent/1"}
    hdrs.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        raw = ""
        try:
            raw = e.read().decode("utf-8", "replace")
        except Exception:
            pass
        raise _status_error(e.code, raw, e.headers)
    except urllib.error.URLError as e:
        raise CloudError("network", str(getattr(e, "reason", e) or e))
    except OSError as e:
        raise CloudError("network", str(e))
    try:
        out = json.loads(body)
    except ValueError:
        raise CloudError("bad_response", _sanitize(body, 300))
    if not isinstance(out, dict):
        raise CloudError("bad_response", _sanitize(str(out), 300))
    return out


def _status_error(status: int, raw: str, headers) -> CloudError:
    msg = _sanitize(raw, 400)
    if status == 429 or status == 403 and "rate" in msg.lower():
        retry = None
        if headers is not None:
            try:
                retry = float(headers.get("Retry-After"))
            except (TypeError, ValueError):
                retry = None
        return CloudError("rate_limit", msg, status, retry)
    if status in (401, 403):
        return CloudError("auth", msg, status)
    if status >= 500:
        return CloudError("server", msg, status)
    return CloudError("bad_response", msg, status)


def _sse_events(lines):
    """Yield ``(event, data)`` for each complete event in an SSE line stream.

    The subset of the SSE format the providers actually use: a blank line
    dispatches, ``data:`` lines build the payload (several are joined with
    ``\\n`` per spec), ``event:`` names it (default ``message``), a line
    starting with ``:`` is a comment, and OpenAI's ``data: [DONE]``
    terminator dispatches as ``("done", "[DONE]")`` and ends the stream.
    A trailing event with no blank line after it is dispatched too — some
    servers close the connection without one.
    """
    event, parts = "message", []
    for raw in lines:
        line = raw.rstrip("\r\n")
        if not line:
            if parts:
                data = "\n".join(parts)
                done = data.strip() == "[DONE]"
                yield ("done", "[DONE]") if done else (event, data)
                if done:
                    return
            event, parts = "message", []
            continue
        if line.startswith(":"):
            continue                    # comment / keep-alive
        if line.startswith("event:"):
            event = line[6:].lstrip() or "message"
        elif line.startswith("data:"):
            value = line[5:]
            if value.startswith(" "):   # one optional space after the colon
                value = value[1:]
            parts.append(value)
        # id:, retry: and anything else are ignored on purpose
    if parts:                           # server closed without a blank line
        data = "\n".join(parts)
        yield ("done", "[DONE]") if data.strip() == "[DONE]" else (event, data)


def _http_stream(url: str, payload: Any, headers: dict,
                 timeout: float = HTTP_TIMEOUT):
    """POST, then yield the response's SSE lines as they arrive.

    Line-iterator twin of ``_http_json``: the same error mapping (an HTTP
    status becomes the same ``CloudError`` the retry logic already knows),
    but the body is consumed incrementally instead of read whole. The
    response is closed in a ``finally`` so a client that stops reading
    mid-stream does not leak the socket.
    """
    data = payload if isinstance(payload, (bytes, bytearray)) else \
        json.dumps(payload, ensure_ascii=False).encode("utf-8")
    hdrs = {"Content-Type": "application/json", "User-Agent": "ai-agent/1",
            "Accept": "text/event-stream"}
    hdrs.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raw = ""
        try:
            raw = e.read().decode("utf-8", "replace")
        except Exception:
            pass
        raise _status_error(e.code, raw, e.headers)
    except urllib.error.URLError as e:
        raise CloudError("network", str(getattr(e, "reason", e) or e))
    except OSError as e:
        raise CloudError("network", str(e))
    try:
        for raw_line in resp:
            yield raw_line.decode("utf-8", "replace").rstrip("\r\n")
    except OSError as e:
        raise CloudError("network", str(e))
    finally:
        try:
            resp.close()
        except Exception:
            pass


def _no_stream_transport(*args, **kwargs):
    """Stand-in when a test injected ``http`` but no stream transport.

    Raises instead of dialling out: reaching this means the fixture wired
    the fake transport to the wrong slot, and a real socket from a test
    would be worse than a loud failure.
    """
    raise CloudError("network",
                     "no stream transport injected for tests")


def _stream_transport(http, http_stream):
    """Which POST-to-SSE implementation ``request_stream`` may use.

    A production call passes neither and gets the real ``_http_stream``.
    Tests inject a fake; when they inject only ``http`` (the non-stream
    transport) they have not provided a stream one, so use
    ``_no_stream_transport`` rather than risk a real connection.
    """
    if http_stream is not None:
        return http_stream
    if http is not None:
        return _no_stream_transport
    return _http_stream


def _guarded_stream(events, *, provider, model):
    """Turn a failure after the first delta into an in-band error frame.

    This is the contract's ``("error", ...)`` yield: before any text has
    gone out, a CloudError still propagates — that is the pre-stream
    failure the retry logic already understands — but once deltas have
    been emitted the caller cannot retry silently, so the failure is
    reported in-band instead. The message is sanitised like every other
    one that can reach a log or the UI.
    """
    emitted = False
    try:
        for kind, payload in events:
            if kind == "delta":
                emitted = True
            yield kind, payload
    except CloudError as e:
        if not emitted:
            raise
        yield "error", {
            "error": "%s%s: %s" % (e.kind,
                                   ("/%s" % e.status) if e.status else "",
                                   _sanitize(e.message or "", 200)),
            "kind": e.kind,
            "status": e.status,
            "provider": provider,
            "model": model,
        }


# ──────────────────────────── tools ────────────────────────────

# Python annotation (or default-value type) → JSON Schema type. Annotations
# arrive as objects or, under `from __future__ import annotations`, as the
# strings `"str"` / `"int"` — both spellings are accepted.
_TYPE_MAP = {
    "str": "string", "string": "string", "text": "string", "Any": "string",
    "int": "integer", "integer": "integer",
    "float": "number", "double": "number", "number": "number",
    "bool": "boolean", "boolean": "boolean",
    "list": "array", "tuple": "array", "List": "array", "Sequence": "array",
    "dict": "object", "Dict": "object", "map": "object", "Path": "string",
}


def _type_of(value: Any) -> str:
    """JSON Schema type inferred from a default value."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"


def _doc_args(doc: str) -> dict:
    """Arg descriptions out of a Google-style ``Args:`` block.

    A description per parameter is what makes a cloud model pass the right
    arguments; anything unparseable simply yields fewer descriptions.
    """
    out: dict = {}
    if not doc:
        return out
    lines = doc.expandtabs(4).splitlines()
    i = 0
    while i < len(lines) and not re.match(r"^\s*Args:\s*$", lines[i]):
        i += 1
    if i >= len(lines):
        return out
    base = len(lines[i]) - len(lines[i].lstrip())
    i += 1
    current = None
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        ind = len(line) - len(line.lstrip())
        if ind <= base:
            break                       # next section / end of docstring
        m = re.match(r"^\s*([A-Za-z_]\w*)\s*(?:\([^)]*\))?\s*:\s*(.*)$", line)
        if m and (len(line) - len(line.lstrip())) <= base + 8:
            current = m.group(1)
            out[current] = m.group(2).strip()
        elif current and ind > base:
            out[current] = (out[current] + " " + line.strip()).strip()
        i += 1
    return out


def function_to_schema(fn) -> dict | None:
    """Build an OpenAI-style function schema from a plain python callable.

    This is the fallback that keeps the cloud path working whatever
    ``litert_lm.tool_from_function`` hands back: the wrapper object may be
    any shape, but the function it wraps is ours — signature and docstring
    are enough to describe it.
    """
    if not callable(fn):
        return None
    name = getattr(fn, "__name__", "") or ""
    if not name:
        return None
    doc = inspect.getdoc(fn) or ""
    first = (doc.split("\n")[0] or "").strip() if doc else ""
    args_doc = _doc_args(doc)
    props: dict = {}
    required: list = []
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        sig = None
    if sig is not None:
        for pname, p in sig.parameters.items():
            if p.kind in (inspect.Parameter.VAR_POSITIONAL,
                          inspect.Parameter.VAR_KEYWORD):
                continue
            if p.annotation is not inspect.Parameter.empty:
                ann = getattr(p.annotation, "__name__", None) or str(p.annotation)
                jtype = _TYPE_MAP.get(str(ann).strip().strip("'\"")) or \
                    _TYPE_MAP.get(str(ann), "string")
            elif p.default is not inspect.Parameter.empty:
                jtype = _type_of(p.default)
            else:
                jtype = "string"
            prop: dict = {"type": jtype}
            desc = args_doc.get(pname)
            if desc:
                prop["description"] = desc
            if p.default is not inspect.Parameter.empty:
                try:
                    json.dumps(p.default)
                    prop["default"] = p.default
                except (TypeError, ValueError):
                    pass
            else:
                required.append(pname)
            props[pname] = prop
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": first,
            "parameters": {"type": "object", "properties": props,
                           "required": required},
        },
    }


def tool_schema(tool: Any) -> dict | None:
    """Normalise anything the tool loop may hold into an OpenAI-style dict.

    Accepts, in order: an OpenAI dict, a ``litert_lm`` Tool (whatever it
    exposes — ``get_tool_description()``, a wrapped callable, or an attribute
    bag), the test stub's ``{"name", "fn"}``, or a bare function. Returns
    ``None`` for anything it cannot describe — a dropped tool beats
    ``TypeError: Object of type _FunctionTool is not JSON serializable``
    half way through a request.
    """
    if tool is None:
        return None
    # A bare function: the schema comes straight from its signature. Objects
    # with __call__ are NOT taken here — they are handled below, where their
    # wrapped function is looked for by name.
    if inspect.isroutine(tool):
        return function_to_schema(tool)
    if isinstance(tool, dict):
        if tool.get("type") == "function" or isinstance(tool.get("function"), dict):
            return tool if tool.get("type") == "function" else \
                {"type": "function", "function": tool["function"]}
        # test stub: {"name": ..., "fn": <callable>}
        fn = tool.get("fn")
        if callable(fn):
            return function_to_schema(fn)
        return None
    getter = getattr(tool, "get_tool_description", None)
    if callable(getter):
        try:
            desc = getter()
        except Exception as e:
            log.warning("get_tool_description failed for %r: %s",
                        type(tool).__name__, e)
            desc = None
        if isinstance(desc, dict):
            if isinstance(desc.get("function"), dict):
                return {"type": "function", "function": desc["function"]}
            if desc.get("name"):
                return {"type": "function", "function": desc}
        # unusable description → fall through to the wrapped function
    for attr in ("fn", "func", "function", "callable", "callback", "target"):
        inner = getattr(tool, attr, None)
        if callable(inner) and not isinstance(inner, type):
            s = function_to_schema(inner)
            if s is not None:
                return s
    n = getattr(tool, "name", "") or ""
    if isinstance(n, str) and n:
        params = getattr(tool, "parameters", None) or \
            getattr(tool, "parameters_schema", None) or \
            getattr(tool, "input_schema", None)
        if not isinstance(params, dict):
            params = {"type": "object", "properties": {}}
        return {
            "type": "function",
            "function": {
                "name": n,
                "description": str(getattr(tool, "description", "") or ""),
                "parameters": params,
            },
        }
    return None


def _tool_schemas(tools) -> list:
    """OpenAI-style dicts only — and JSON-serialisable, guaranteed.

    The wire test runs here so no adapter can ever hand a python object to
    ``json.dumps``: an undescribable tool is dropped (and logged) rather than
    failing the whole turn.
    """
    out = []
    for t in tools or []:
        s = tool_schema(t)
        if not (isinstance(s, dict) and isinstance(s.get("function"), dict)):
            if t is not None:
                log.warning("no schema for tool %r — not sent to the provider",
                            type(t).__name__)
            continue
        try:
            json.dumps(s, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            log.warning("tool schema for %s is not JSON serializable (%s) — "
                        "dropped", (s.get("function") or {}).get("name"), e)
            continue
        out.append(s)
    return out


# ─────────────────────── message normalising ───────────────────────

def _reasoning_of(m: dict):
    """The reasoning a turn was produced with — None when never recorded.

    Presence, not truthiness: DeepSeek's thinking models refuse the next
    turn (`400 ... must be passed back to the API`) unless the assistant
    turn carries `reasoning_content` — even when the reasoning itself is
    the empty string.
    """
    if "reasoning_content" in m:
        val = m.get("reasoning_content")
    elif "reasoning" in m:
        val = m.get("reasoning")
    else:
        return None
    return "" if val is None else val


def normalize(messages: list) -> list:
    """Stored transcript → canonical turns for the adapters.

    Our history keeps `tool_call` and `tool_result` as separate entries and
    never replays the call itself (see `_ensure_conv`). Cloud APIs require the
    call and its result to sit next to each other under an assistant message,
    so calls are re-attached here. Call ids are synthesised from the position,
    which keeps them stable across steps within a turn.
    """
    out: list = []
    pending = None      # {"id","name","args","reasoning"} waiting for its result
    n = 0
    for m in messages or []:
        role = (m or {}).get("role")
        if role in ("user", "assistant"):
            # The transcript stores `text`; cloud APIs want `content`. Messages
            # that already carry `content` (a test, a replayed reply) pass
            # through untouched instead of being emptied.
            content = m.get("text")
            if content is None:
                content = m.get("content", "")
            if isinstance(content, list):
                out.append({"role": role, "content": content})
            else:
                entry = {"role": role, "content": content or ""}
                # DeepSeek thinking mode: an assistant turn that was
                # produced with reasoning must carry it back — empty
                # string included — or the next turn dies with a 400.
                if role == "assistant":
                    reasoning = _reasoning_of(m)
                    if reasoning is not None:
                        entry["reasoning_content"] = reasoning
                out.append(entry)
        elif role == "tool_call":
            n += 1
            pending = {
                "id": "call_%d" % n,
                "name": m.get("tool", ""),
                "args": m.get("args") or {},
                "reasoning": _reasoning_of(m),
            }
        elif role == "tool_result":
            n += 1
            if pending is None:
                pending = {"id": "call_%d" % n, "name": m.get("tool", ""),
                           "args": {}, "reasoning": None}
            assistant_msg = {
                "role": "assistant",
                "tool_calls": [{"id": pending["id"], "type": "function",
                                "function": {"name": pending["name"],
                                             # The wire format wants the
                                             # arguments JSON-encoded as a
                                             # string; a dict here is what
                                             # drew `422 invalid type: map,
                                             # expected a string` from
                                             # DeepSeek/OpenAI.
                                             "arguments": json.dumps(
                                                 pending["args"],
                                                 ensure_ascii=False)}}],
            }
            if pending["reasoning"] is not None:
                assistant_msg["reasoning_content"] = pending["reasoning"]
            out.append(assistant_msg)
            out.append({"role": "tool", "tool_call_id": pending["id"],
                        "name": pending["name"],
                        "content": m.get("text", "") or ""})
            pending = None
    # A call still waiting for its result when the transcript ends is dropped on
    # purpose: it never reached `out` (only a `tool_result` emits the pair), and
    # replaying it would draw a 400 demanding the matching tool messages.
    return out


# ─────────────────────── reasoning policy ───────────────────────

# Providers whose thinking models refuse a turn unless the previous
# assistant turn's `reasoning_content` is replayed verbatim — DeepSeek
# answers `400 ... must be passed back to the API` when it is missing.
THINKING_PROVIDERS = {"deepseek"}

# Deep-copying the transcript per request is cheap until it is not; a
# payload past this size is flagged in the log so a runaway history is
# visible instead of silently cloning megabytes on every single turn.
_POLICY_LOG_LIMIT = 200 * 1024


def _is_thinking_model(provider: str, model: str) -> bool:
    """Does this provider+model think — and so replay `reasoning_content`?"""
    if provider not in THINKING_PROVIDERS:
        return False
    name = (model or "").strip().lower()
    if not name:
        return True
    # deepseek-chat is the non-thinking model; every other name
    # (deepseek-reasoner, deepseek-flash, free text) thinks.
    return "chat" not in name


def _apply_reasoning_policy(messages: list, provider: str,
                            thinking: bool) -> list:
    """Per-provider wire policy for `reasoning_content`, on a private copy.

    Thinking models: every assistant turn must carry the key, the empty
    string included — the API rejects a missing one. Every other provider:
    the key must be gone — some reject unknown fields.

    The caller's list is never mutated, so the same normalised transcript
    can be re-applied per provider (and re-flipped by the 400 safety net
    in `AgentBackend.step`) without side effects.
    """
    try:
        size = len(json.dumps(messages, ensure_ascii=False))
    except (TypeError, ValueError):
        size = 0
    if size > _POLICY_LOG_LIMIT:
        log.warning("reasoning policy: applying to a %d KB transcript",
                    size // 1024)
    out = copy.deepcopy(messages or [])
    if thinking:
        for m in out:
            if isinstance(m, dict) and m.get("role") == "assistant":
                m.setdefault("reasoning_content", "")
                if m["reasoning_content"] is None:
                    m["reasoning_content"] = ""
    else:
        for m in out:
            if isinstance(m, dict):
                m.pop("reasoning_content", None)
    return out


def _args(value) -> dict:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {"_raw": value}
        except ValueError:
            return {"_raw": value}
    return {"_raw": str(value)}


def _tool_calls_from_openai(message: dict) -> list:
    out = []
    for tc in (message.get("tool_calls") or []):
        fn = tc.get("function") or {}
        out.append({"function": {"name": fn.get("name", ""),
                                 "arguments": _args(fn.get("arguments"))}})
    return out


# ─────────────────────────── adapters ───────────────────────────

class Adapter:
    provider = ""

    def request(self, *, key: str, model: str, system: str, messages: list,
                tools: list, max_tokens: int) -> tuple[str, dict]:
        """POST to the provider. Raises CloudError. Returns (url, body)."""
        raise NotImplementedError

    def parse(self, data: dict, *, model: str, key_id: str) -> Step:
        raise NotImplementedError

    def request_stream(self, *, key: str, model: str, system: str,
                       messages: list, tools: list, max_tokens: int,
                       http=None, http_stream=None):
        """Streaming twin of request(): a generator over the provider's SSE.

        Yields ``("delta", text)`` fragments and then exactly one
        ``("done", info)``; failures raise ``CloudError`` (the caller turns
        a post-first-byte one into an ``("error", ...)`` frame). ``http`` /
        ``http_stream`` select the transport — see ``_stream_transport``.
        """
        raise NotImplementedError


class OpenAIAdapter(Adapter):
    provider = "openai"

    def __init__(self, base_url: str = ""):
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")

    def request(self, *, key, model, system, messages, tools, max_tokens):
        msgs = list(messages or [])
        # OpenAI-compatible APIs take the system prompt as a message;
        # dropping it made every cloud reply lose the ညီ persona (and
        # the "reply in Burmese" rule) — English answers only.
        if system:
            msgs.insert(0, {"role": "system", "content": system})
        body: dict = {"model": model, "messages": msgs,
                      "max_tokens": max_tokens}
        if tools:
            body["tools"] = tools
        url = self.base_url + "/chat/completions"
        return url, body, {"Authorization": "Bearer %s" % key}

    def parse(self, data, *, model, key_id):
        try:
            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
        except (AttributeError, IndexError, TypeError):
            raise CloudError("bad_response", str(data)[:300])
        if not isinstance(msg, dict) or not msg:
            raise CloudError("bad_response", str(data)[:300])
        return Step(
            text=msg.get("content") or "",
            tool_calls=_tool_calls_from_openai(msg),
            usage=data.get("usage") or {},
            model=data.get("model") or model,
            provider=self.provider,
            key_id=key_id,
            reasoning=msg.get("reasoning_content") or "",
        )

    def request_stream(self, *, key, model, system, messages, tools,
                       max_tokens, http=None, http_stream=None):
        """``"stream": true`` on the endpoint request() already builds.

        Yields ``("delta", text)`` fragments, one ``("done", info)``, and
        — for a failure after the first delta only — an ``("error", ...)``
        frame (see ``_guarded_stream``); earlier failures raise CloudError.

        OpenRouter and DeepSeek inherit this unchanged — they speak the
        OpenAI dialect and only their base_url differs.
        No ``stream_options`` is sent: gateways that speak this dialect
        but predate the field reject unknown keys, and a missing usage
        block only costs the meter an estimate it already falls back to.
        """
        url, body, headers = self.request(
            key=key, model=model, system=system, messages=messages,
            tools=tools, max_tokens=max_tokens)
        body["stream"] = True
        transport = _stream_transport(http, http_stream)
        events = _openai_stream(transport(url, body, headers),
                                model=model, provider=self.provider)
        for event in _guarded_stream(events, provider=self.provider,
                                     model=model):
            yield event


class GeminiAdapter(Adapter):
    provider = "gemini"

    def __init__(self, base_url: str = ""):
        self.base_url = (base_url or
                         "https://generativelanguage.googleapis.com/v1beta").rstrip("/")

    def request(self, *, key, model, system, messages, tools, max_tokens):
        contents = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            parts = []
            if m["role"] == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    fn = tc.get("function") or {}
                    parts.append({"functionCall": {
                        "name": fn.get("name", ""),
                        "args": _args(fn.get("arguments")),
                    }})
                if m.get("content"):
                    parts.append({"text": m["content"]})
            elif m["role"] == "tool":
                parts.append({"functionResponse": {
                    "name": m.get("name", ""),
                    "response": {"result": m.get("content", "")},
                }})
            else:
                parts.append({"text": m.get("content", "")})
            contents.append({"role": role, "parts": parts})
        body: dict = {
            "contents": contents,
            "generationConfig": {"maxOutputTokens": max_tokens},
        }
        if system:
            body["system_instruction"] = {"parts": [{"text": system}]}
        decls = [(_fix_sch(t)["function"]) for t in tools or []]
        if decls:
            body["tools"] = [{"function_declarations": decls}]
        url = "%s/models/%s:generateContent?key=%s" % (
            self.base_url, (model or "").removeprefix("models/"), key)
        return url, body, {}

    def parse(self, data, *, model, key_id):
        try:
            cand = (data.get("candidates") or [{}])[0]
            parts = (cand.get("content") or {}).get("parts") or []
        except (AttributeError, IndexError, TypeError):
            raise CloudError("bad_response", str(data)[:300])
        if not parts:
            raise CloudError("bad_response", str(data)[:300])
        text, calls = "", []
        for p in parts:
            if not isinstance(p, dict):
                continue
            if p.get("text"):
                text += p["text"]
            if p.get("functionCall"):
                fc = p["functionCall"]
                if isinstance(fc, dict):
                    raw = fc["args"] if "args" in fc else fc.get("arguments")
                    calls.append({"function": {
                        "name": fc.get("name", ""),
                        "arguments": _args(raw)}})
        u = data.get("usageMetadata") or {}
        usage = {}
        if u.get("promptTokenCount") is not None:
            usage["prompt_tokens"] = u.get("promptTokenCount")
        if u.get("candidatesTokenCount") is not None:
            usage["completion_tokens"] = u.get("candidatesTokenCount")
        return Step(text=text, tool_calls=calls, usage=usage,
                    model=model, provider=self.provider, key_id=key_id)

    def request_stream(self, *, key, model, system, messages, tools,
                       max_tokens, http=None, http_stream=None):
        """The same payload request() builds, on the SSE twin of the URL.

        Yields ``("delta", text)`` fragments, one ``("done", info)``, and
        — for a failure after the first delta only — an ``("error", ...)``
        frame (see ``_guarded_stream``); earlier failures raise CloudError.

        ``:streamGenerateContent?alt=sse`` is ``:generateContent``
        streamed — swapping that fragment is the whole difference.
        """
        url, body, headers = self.request(
            key=key, model=model, system=system, messages=messages,
            tools=tools, max_tokens=max_tokens)
        url = url.replace(":generateContent?",
                          ":streamGenerateContent?alt=sse&", 1)
        transport = _stream_transport(http, http_stream)
        events = _gemini_stream(transport(url, body, headers),
                                model=model, provider=self.provider)
        for event in _guarded_stream(events, provider=self.provider,
                                     model=model):
            yield event


class OpenRouterAdapter(OpenAIAdapter):
    """OpenRouter speaks the OpenAI dialect behind a different base_url."""
    provider = "openrouter"

    def __init__(self, base_url: str = ""):
        self.base_url = (base_url or "https://openrouter.ai/api/v1").rstrip("/")


class DeepSeekAdapter(OpenAIAdapter):
    """DeepSeek speaks the OpenAI dialect; only the base_url differs."""
    provider = "deepseek"

    def __init__(self, base_url: str = ""):
        self.base_url = (base_url or "https://api.deepseek.com/v1").rstrip("/")


class AnthropicAdapter(Adapter):
    provider = "anthropic"
    VERSION = "2023-06-01"

    def __init__(self, base_url: str = ""):
        self.base_url = (base_url or "https://api.anthropic.com/v1").rstrip("/")

    def request(self, *, key, model, system, messages, tools, max_tokens):
        out = []
        for m in messages:
            if m["role"] == "assistant":
                blocks = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    blocks.append({"type": "tool_use",
                                   "id": tc.get("id") or "call_x",
                                   "name": fn.get("name", ""),
                                   "input": _args(fn.get("arguments"))})
                if blocks:
                    out.append({"role": "assistant", "content": blocks})
            elif m["role"] == "tool":
                out.append({"role": "user", "content": [{
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id") or "call_x",
                    "content": m.get("content", ""),
                }]})
            else:
                out.append({"role": "user",
                            "content": m.get("content", "")})
        body: dict = {"model": model, "max_tokens": max_tokens,
                      "messages": out}
        if system:
            body["system"] = system
        if tools:
            body["tools"] = [{"name": t["function"]["name"],
                              "description": t["function"].get("description", ""),
                              "input_schema": t["function"].get("parameters")
                              or {"type": "object", "properties": {}}}
                             for t in tools]
        url = self.base_url + "/messages"
        return url, body, {"x-api-key": key, "anthropic-version": self.VERSION}

    def parse(self, data, *, model, key_id):
        blocks = data.get("content")
        if not isinstance(blocks, list) or not blocks:
            raise CloudError("bad_response", str(data)[:300])
        text, calls = "", []
        for b in blocks:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text":
                text += b.get("text") or ""
            elif b.get("type") == "tool_use":
                calls.append({"function": {"name": b.get("name", ""),
                                           "arguments": _args(b.get("input"))}})
        u = data.get("usage") or {}
        usage = {}
        if u.get("input_tokens") is not None:
            usage["prompt_tokens"] = u.get("input_tokens")
        if u.get("output_tokens") is not None:
            usage["completion_tokens"] = u.get("output_tokens")
        return Step(text=text, tool_calls=calls, usage=usage,
                    model=model, provider=self.provider, key_id=key_id)

    def request_stream(self, *, key, model, system, messages, tools,
                       max_tokens, http=None, http_stream=None):
        """``"stream": true`` on /messages; named events carry the shape.

        Yields ``("delta", text)`` fragments, one ``("done", info)``, and
        — for a failure after the first delta only — an ``("error", ...)``
        frame (see ``_guarded_stream``); earlier failures, including the
        stream's own ``error`` event, raise CloudError.
        """
        url, body, headers = self.request(
            key=key, model=model, system=system, messages=messages,
            tools=tools, max_tokens=max_tokens)
        body["stream"] = True
        transport = _stream_transport(http, http_stream)
        events = _anthropic_stream(transport(url, body, headers),
                                   model=model, provider=self.provider)
        for event in _guarded_stream(events, provider=self.provider,
                                     model=model):
            yield event


# ── streaming parsers (SSE → deltas + one "done" per step) ──────────────
# One per wire dialect, shared by the adapters above: request() builds the
# URL and headers, these consume the response. Each ends with exactly one
# ("done", info) whose keys mirror what adapter.parse() would have
# returned; key_id stays "" — the backend owns key ids and fills it in.

def _openai_stream(lines, *, model, provider):
    """Consume an OpenAI-dialect SSE body: text deltas, then one ``done``.

    ``reasoning_content`` accumulates instead of streaming — it is replay
    material for the next request, not something the user reads — and
    ``tool_calls`` merge by ``index`` because one call's arguments may be
    split across chunks. Usage is taken from whichever chunk carries it
    (the final one, when the server bothers).
    """
    reasoning, usage, calls = "", {}, {}
    for event, data in _sse_events(lines):
        if event == "done":             # data: [DONE]
            break
        try:
            obj = json.loads(data)
        except ValueError:
            continue                    # keep-alive or non-JSON noise
        if not isinstance(obj, dict):
            continue
        if isinstance(obj.get("usage"), dict) and obj["usage"]:
            usage = obj["usage"]
        choices = obj.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        piece = delta.get("content")
        if isinstance(piece, str) and piece:
            yield "delta", piece
        r = delta.get("reasoning_content")
        if isinstance(r, str):
            reasoning += r
        for tc in delta.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            idx = tc.get("index")
            slot = calls.setdefault(
                idx if isinstance(idx, int) else 0,
                {"function": {"name": "", "arguments": ""}})
            fn = tc.get("function")
            if isinstance(fn, dict):
                if isinstance(fn.get("name"), str):
                    slot["function"]["name"] += fn["name"]
                if isinstance(fn.get("arguments"), str):
                    slot["function"]["arguments"] += fn["arguments"]
    tool_calls = [v for _, v in sorted(calls.items())]
    for tc in tool_calls:              # the wire carries a JSON string
        tc["function"]["arguments"] = _args(tc["function"]["arguments"])
    yield "done", {"usage": usage, "tool_calls": tool_calls,
                   "reasoning": reasoning, "provider": provider,
                   "model": model, "key_id": ""}


def _gemini_stream(lines, *, model, provider):
    """Consume Gemini's ``alt=sse`` body: parts yield text as they arrive.

    The last ``usageMetadata`` seen wins — Gemini repeats the counts on
    later chunks with the running total.
    """
    usage, calls = {}, []
    for event, data in _sse_events(lines):
        if event == "done":
            break
        try:
            obj = json.loads(data)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        u = obj.get("usageMetadata")
        if isinstance(u, dict):
            if u.get("promptTokenCount") is not None:
                usage["prompt_tokens"] = u.get("promptTokenCount")
            if u.get("candidatesTokenCount") is not None:
                usage["completion_tokens"] = u.get("candidatesTokenCount")
        candidates = obj.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            continue
        cand = candidates[0]
        if not isinstance(cand, dict):
            continue
        content = cand.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(parts, list):
            continue
        for p in parts:
            if not isinstance(p, dict):
                continue
            text = p.get("text")
            if isinstance(text, str) and text:
                yield "delta", text
            fc = p.get("functionCall")
            if isinstance(fc, dict):
                # No part index exists on this wire and each functionCall
                # arrives whole, so arrival order is the order the
                # non-streaming parse() collects them in.
                raw = fc["args"] if "args" in fc else fc.get("arguments")
                calls.append({"function": {
                    "name": fc.get("name", ""),
                    "arguments": _args(raw)}})
    yield "done", {"usage": usage, "tool_calls": calls, "reasoning": "",
                   "provider": provider, "model": model, "key_id": ""}


def _anthropic_stream(lines, *, model, provider):
    """Consume Anthropic's message stream: named events, one JSON payload each.

    Text streams through ``content_block_delta``; a tool call's arguments
    buffer per content-block ``index`` as ``input_json_delta`` fragments
    and are parsed once the stream ends. An ``error`` event raises
    CloudError: before the first delta that is an ordinary pre-stream
    failure for the caller's retry logic, after it the caller turns it
    into an ``("error", ...)`` frame.
    """
    usage, tool_calls = {}, []
    names, args_buf, order = {}, {}, []
    for event, data in _sse_events(lines):
        if event == "done":
            break
        try:
            obj = json.loads(data)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        etype = obj.get("type") or event
        if etype == "error":
            err = obj.get("error")
            err = err if isinstance(err, dict) else {}
            why = str(err.get("type") or "error")
            msg = str(err.get("message") or "")
            if "rate_limit" in why:
                kind, status = "rate_limit", 429
            elif "authentication" in why:
                kind, status = "auth", 401
            elif "permission" in why:
                kind, status = "auth", 403
            elif "overloaded" in why:
                kind, status = "server", 529   # their outage, not the key
            else:
                kind, status = "bad_response", 400
            raise CloudError(kind, "%s: %s" % (why, _sanitize(msg, 300)),
                             status)
        if etype == "message_start":
            m = obj.get("message")
            u = m.get("usage") if isinstance(m, dict) else None
            if isinstance(u, dict) and u.get("input_tokens") is not None:
                usage["prompt_tokens"] = u.get("input_tokens")
        elif etype == "content_block_start":
            block = obj.get("content_block")
            if isinstance(block, dict) and block.get("type") == "tool_use":
                idx = obj.get("index")
                if idx not in names:
                    order.append(idx)  # blocks answer in index order
                names[idx] = block.get("name", "")
                args_buf.setdefault(idx, "")
        elif etype == "content_block_delta":
            delta = obj.get("delta")
            if not isinstance(delta, dict):
                continue
            if delta.get("type") == "text_delta":
                text = delta.get("text")
                if isinstance(text, str) and text:
                    yield "delta", text
            elif delta.get("type") == "input_json_delta":
                frag = delta.get("partial_json")
                if isinstance(frag, str):
                    idx = obj.get("index")
                    args_buf[idx] = args_buf.get(idx, "") + frag
        elif etype == "message_delta":
            u = obj.get("usage")
            if isinstance(u, dict):
                if u.get("output_tokens") is not None:
                    usage["completion_tokens"] = u.get("output_tokens")
                if u.get("input_tokens") is not None:
                    usage["prompt_tokens"] = u.get("input_tokens")
        # content_block_stop / message_stop / ping: nothing to record
    for idx in order:
        tool_calls.append({"function": {
            "name": names.get(idx, ""),
            "arguments": _args(args_buf.get(idx, ""))}})
    yield "done", {"usage": usage, "tool_calls": tool_calls,
                   "reasoning": "", "provider": provider,
                   "model": model, "key_id": ""}


def _fix_sch(sch: dict) -> dict:
    """Gemini's functionDeclarations must not carry `type: "function"`."""
    if isinstance(sch, dict) and isinstance(sch.get("function"), dict):
        return {"function": dict(sch["function"])}
    return sch


ADAPTERS: dict = {
    "openai": OpenAIAdapter,
    "gemini": GeminiAdapter,
    "openrouter": OpenRouterAdapter,
    "anthropic": AnthropicAdapter,
    "deepseek": DeepSeekAdapter,
}


# ───────────────────────── the backend ─────────────────────────

class AgentBackend:
    """Chooses provider + key for one Agent Mode step, with fallback."""

    def __init__(self, keys: KeyManager | None = None,
                 net_state: Callable[[], str] | None = None,
                 http: Callable[..., dict] | None = None):
        self.keys = keys or KeyManager()
        self._net_state = net_state or (lambda: net.state())
        self._http = http or _http_json
        # Live context accounting for the meter: what the last step resolved,
        # how much of it the transcript used, and whether it had to be trimmed.
        self._ctx: dict = {
            "model": "", "provider": "", "base_url": "", "max": 0, "used": 0,
            "pct": 0.0, "source": "", "endpoint": "", "matched": "",
            "trimmed": False, "dropped": 0, "summarized": 0,
            "over_budget": False, "exact": False, "last_model": "", "at": 0.0,
        }

    # ---------- context window (model-aware) ----------
    def context_state(self) -> dict:
        """Copy of the live context record — safe to serialise as-is."""
        return dict(self._ctx)

    @property
    def current_max_ctx(self) -> int:
        return int(self._ctx.get("max") or 0)

    @property
    def current_used_ctx(self) -> int:
        return int(self._ctx.get("used") or 0)

    @property
    def current_model(self) -> str:
        return str(self._ctx.get("model") or "")

    def _record_context(self, provider: str, model: str, base: str,
                        info: dict, fit: dict) -> None:
        """Publish what this provider's window is and how full it is."""
        used = int(fit.get("after") or 0)
        limit = int(info.get("limit") or 0)
        self._ctx.update({
            "model": model or "",
            "provider": provider or "",
            "base_url": base or "",
            "max": limit,
            "used": used,
            "pct": round(used / limit * 100, 1) if limit else 0.0,
            "source": info.get("source", ""),
            "endpoint": info.get("endpoint", ""),
            "matched": info.get("matched", ""),
            "trimmed": bool(fit.get("trimmed")),
            "dropped": int(fit.get("dropped") or 0),
            "summarized": int(fit.get("summarized") or 0),
            "over_budget": bool(fit.get("over_budget")),
            "at": time.time(),
        })
        if fit.get("trimmed"):
            log.info("context trimmed for %s/%s: %d of %d tokens kept (%d "
                     "messages dropped, %d summarized)", provider, model, used,
                     limit, fit.get("dropped") or 0, fit.get("summarized") or 0)

    def _fit(self, messages: list, system: str, tools: list, limit: int,
             factor: float = 1.0) -> dict:
        """Trim the transcript to the model's window, with headroom to spare."""
        return CT.trim_messages(messages, limit=max(MR.MIN_WINDOW,
                                                    int(limit * factor)),
                                reserve=CT.RESERVE, system=system, tools=tools)

    # ---------- config freshness ----------
    def refresh(self) -> None:
        """Re-read agent_config.json on the next query.

        One AgentBackend lives for the whole process (AgentMode caches it),
        while Settings writes the file from a different request. Without this
        a mode switch — Cloud Only in particular — would not be seen until the
        server restarts, and the turn would silently stay on the local model.
        The rotation cursor is kept: only the config cache is dropped.
        """
        try:
            self.keys.invalidate()
        except Exception as e:      # never break a turn over a reload
            log.warning("config refresh failed: %s", e)

    # ---------- capability ----------
    def online(self) -> bool:
        try:
            return self._net_state() == net.ONLINE
        except Exception as e:
            log.warning("network probe failed: %s", e)
            return False

    def fallback_order(self) -> list:
        self.refresh()
        cfg = self.keys.load()
        order = [p for p in (cfg.get("fallback") or []) if p in ADAPTERS]
        for p in ADAPTERS:
            if p not in order:
                order.append(p)
        return order

    def usable(self) -> list:
        """Providers that are enabled AND still hold a usable key."""
        self.refresh()
        cfg = self.keys.load()
        out = []
        for p in self.fallback_order():
            prov = (cfg.get("providers") or {}).get(p) or {}
            if prov.get("enabled") is False:
                continue
            if self.keys.healthy(p):
                out.append(p)
        return out

    @property
    def mode(self) -> str:
        """auto | local | cloud — see core.key_manager.MODES."""
        self.refresh()
        return self.keys.mode()

    @property
    def fallback_to_local(self) -> bool:
        """True unless the user pinned Cloud Only."""
        return self.mode != "cloud"

    def why_not_cloud(self) -> str:
        """Machine-readable reason `should_use_cloud()` said no ("" if yes)."""
        self.refresh()
        cfg = self.keys.load()
        if self.mode == "local":
            return "local_only"
        if not cfg.get("prefer_cloud", True):
            return "local_only"
        if not self.online():
            return "offline"
        if not self.usable():
            return "no_usable_key"
        if not cfg.get("cost_ack", False):
            return "cost_not_acknowledged"
        return ""

    def planned_context(self) -> dict:
        """Window of the provider that would answer the *next* turn.

        The meter has to be right before the first reply too, so this answers
        from the same provider order `step()` walks rather than from the last
        turn's record.
        """
        try:
            providers = self.usable()
        except Exception as e:
            log.debug("usable() failed while planning context: %s", e)
            providers = []
        if not providers:
            try:
                providers = self.fallback_order()
            except Exception as e:
                log.debug("fallback_order() failed while planning context: %s",
                          e)
                providers = []
        cfg = self.keys.load()
        for provider in providers:
            pc = (cfg.get("providers") or {}).get(provider) or {}
            model = pc.get("model") or ""
            if not model:
                continue
            info = MR.describe_limit(model, pc.get("base_url") or "")
            info["provider"] = provider
            return info
        return {"model": "", "base_url": "", "endpoint": "unknown",
                "limit": MR.UNKNOWN_CLOUD_DEFAULT, "source": "none",
                "matched": "", "provider": ""}

    def should_use_cloud(self) -> bool:
        self.refresh()
        if self.mode == "local":
            return False
        cfg = self.keys.load()
        if not cfg.get("prefer_cloud", True):
            return False
        if not cfg.get("cost_ack", False):
            return False       # the price must have been shown first
        return self.online() and bool(self.usable())

    # ---------- one step ----------
    def step(self, *, system: str, messages: list, tools: list,
             max_tokens: int | None = None) -> Step:
        # None means "whatever this install is configured to allow". A value
        # that is given is honoured only as far as the ceiling allows, so no
        # caller can quietly ask the provider for its maximum.
        max_tokens = config.clamp_max_tokens(max_tokens,
                                             config.agent_max_tokens())
        self.refresh()
        if not self.online():
            raise BackendExhausted("no network")
        # The transcript stores {"text": ...}; every cloud API wants
        # {"content": ...}. Skipping this is what made Cloud Only answer
        # "providers exhausted" on a 422 `missing field content`.
        messages = normalize(messages)
        # LiteRT hands back live Tool / _FunctionTool objects; the wire only
        # knows OpenAI-style dicts. Sending the object is what raised
        # `TypeError: Object of type _FunctionTool is not JSON serializable`
        # on every Cloud Only turn. Convert once, for every adapter.
        tools = _tool_schemas(tools)
        last: CloudError | None = None
        tried = []
        for provider in self.usable():
            adapter = ADAPTERS[provider]()
            prov_cfg = (self.keys.load().get("providers") or {}).get(provider) or {}
            model = prov_cfg.get("model") or ""
            base = prov_cfg.get("base_url") or ""
            _warn_unknown_model(provider, model)
            if base and adapter.__class__.__name__ in _BASE_URL_ADAPTERS:
                adapter = adapter.__class__(base)

            def _attempt(entry, wire_msgs):
                url, body, headers = adapter.request(
                    key=entry.get("key", ""), model=model, system=system,
                    messages=wire_msgs, tools=tools, max_tokens=max_tokens)
                data = self._http(url, body, headers)
                step = adapter.parse(data, model=model, key_id=entry.get("id"))
                self.keys.mark_ok(provider, entry.get("id"))
                self._log_ok(step, provider, model, entry)
                return step

            # This provider's window decides the budget — not
            # Config.context_size, which is the *local* engine's KV cache and
            # is 8192 for a model that takes a million tokens.
            info = MR.describe_limit(model, base)
            limit = int(info["limit"])
            fit = self._fit(messages, system, tools, limit)
            self._record_context(provider, model, base, info, fit)
            trimmed_msgs = fit["messages"]

            # Wire policy is per provider: a thinking DeepSeek model must
            # see `reasoning_content` replayed on every assistant turn,
            # every other provider must not see the key at all.
            wire = _apply_reasoning_policy(
                trimmed_msgs, provider, _is_thinking_model(provider, model))
            seen_ids = set()
            while True:
                entry = self.keys.next_key(provider)
                if entry is None:
                    break
                if entry.get("id") in seen_ids:
                    # Every usable key of this provider has had its turn.
                    # Server/bad-response errors do not park a key, so without
                    # this the loop would spin on the same key forever.
                    break
                seen_ids.add(entry.get("id"))
                # One context retry per KEY, not one per provider: a stubborn
                # key must not spend the retry that the next key still needs,
                # or keys 2..n would get no resilience at all. Still bounded —
                # seen_ids caps each key at one attempt plus one retry.
                context_retried = False
                tried.append("%s/%s" % (provider, mask(entry.get("key", ""))))
                try:
                    return _attempt(entry, wire)
                except CloudError as e:
                    if (e.status == 400
                            and "reasoning_content" in (e.message or "")):
                        # The provider rejected the replay itself — flip the
                        # policy once and try exactly one more time before
                        # the failure is handled like any other 400.
                        log.warning("%s 400 named reasoning_content — "
                                    "flipping the replay policy once",
                                    provider)
                        wire = _apply_reasoning_policy(
                            trimmed_msgs, provider,
                            provider in THINKING_PROVIDERS)
                        try:
                            return _attempt(entry, wire)
                        except CloudError as retry_err:
                            e = retry_err
                    if not context_retried and MR.is_context_error(e.message or ""):
                        # Too much history for this window. Remember whatever
                        # limit the provider named, cut harder, and try the
                        # same key once more — a second refusal then falls
                        # through to the normal error handling like any 400.
                        # Deliberately separate from the reasoning retry above:
                        # the two faults are independent and neither may borrow
                        # the other's single retry.
                        context_retried = True
                        learned = MR.record_limit(model, base, text=e.message)
                        if learned and learned != limit:
                            info = MR.describe_limit(model, base)
                            info["limit"] = learned
                            limit = learned
                        # Even when the provider named no number, retry against
                        # three quarters of the window we believed in — the
                        # estimate that got us here was optimistic.
                        fit = self._fit(messages, system, tools, limit,
                                        CONTEXT_RETRY_FACTOR)
                        self._record_context(provider, model, base, info, fit)
                        wire = _apply_reasoning_policy(
                            fit["messages"], provider,
                            _is_thinking_model(provider, model))
                        log.warning("%s says the context does not fit (%d "
                                    "tokens, window %s) — retrying once at %d",
                                    provider, fit["before"], limit,
                                    int(limit * CONTEXT_RETRY_FACTOR))
                        try:
                            return _attempt(entry, wire)
                        except CloudError as retry_err:
                            e = retry_err
                    last = e
                    self._handle_error(provider, entry, e, model)
                    if e.kind in ("network",):
                        # No point walking the remaining providers — none of
                        # them can be reached either.
                        raise BackendExhausted("network")
        detail = ""
        if last is not None:
            detail = "%s%s: %s" % (
                last.kind, ("/%s" % last.status) if last.status else "",
                _sanitize(last.message or "", 200))
        raise BackendExhausted("providers exhausted: %s%s" % (
            ", ".join(tried) or "no keys",
            (" — " + detail) if detail else ""))

    # ---------- one step, streamed ----------
    def step_stream(self, *, system: str, messages: list, tools: list,
                    max_tokens: int | None = None, http=None,
                    http_stream=None):
        """Yield ("delta", str) then one ("done", info) from the winning
        provider, or ("error", {...}) after the first delta.
        Mirrors step()'s ladder. Does NOT call the local engine.

        info = {usage, tool_calls, reasoning, provider, model, key_id}

        A failure before the first delta walks the same ladder step() walks
        — reasoning flip, context retry at CONTEXT_RETRY_FACTOR,
        _handle_error, next key, next provider — and when the whole chain
        is spent this returns without yielding instead of raising
        BackendExhausted: an empty stream is the signal the caller uses to
        decide about the local fallback.
        """
        # mirrors step() L1425 — keep in sync on any change
        max_tokens = config.clamp_max_tokens(max_tokens,
                                             config.agent_max_tokens())
        self.refresh()                              # mirrors step() L1427
        if not self.online():
            # mirrors step() L1428 — there step() raises
            # BackendExhausted("no network"); a stream yields nothing and
            # the caller decides about the local fallback.
            return
        # mirrors step() L1433 — {"text": ...} → {"content": ...}
        messages = normalize(messages)
        # mirrors step() L1438 — LiteRT Tool objects → wire dicts
        tools = _tool_schemas(tools)
        last: CloudError | None = None
        tried = []

        for provider in self.usable():              # mirrors step() L1441
            adapter = ADAPTERS[provider]()
            prov_cfg = (self.keys.load().get("providers") or {}).get(provider) or {}
            model = prov_cfg.get("model") or ""
            base = prov_cfg.get("base_url") or ""
            _warn_unknown_model(provider, model)
            # mirrors step() L1447 — keep in sync on any change
            if base and adapter.__class__.__name__ in _BASE_URL_ADAPTERS:
                adapter = adapter.__class__(base)

            def _pump(entry, wire_msgs):
                """Forward one attempt's frames verbatim.

                A CloudError escapes only while nothing has been forwarded
                yet — that is the pre-stream failure step() L1491's ladder
                already understands. Once a delta is out, a failure is
                reported in-band as an ("error", ...) frame and the stream
                stops: no key and no provider is retried for output the
                caller has already seen.
                """
                emitted = False
                try:
                    for kind, payload in adapter.request_stream(
                            key=entry.get("key", ""), model=model,
                            system=system, messages=wire_msgs, tools=tools,
                            max_tokens=max_tokens, http=http,
                            http_stream=http_stream):
                        if kind == "delta":
                            emitted = True
                        elif kind == "done":
                            payload = dict(payload or {})
                            # the backend owns key ids; the parsers leave ""
                            # — see the streaming parsers' header comment.
                            payload["key_id"] = entry.get("id")
                            # mirrors step() L1456 — a key is good only once
                            # the whole answer really arrived
                            self.keys.mark_ok(provider, entry.get("id"))
                            # mirrors step() L1457 — same meter update
                            self._log_ok(Step(
                                tool_calls=payload.get("tool_calls") or [],
                                usage=payload.get("usage") or {},
                                model=payload.get("model") or model,
                                provider=provider,
                                key_id=entry.get("id")),
                                provider, model, entry)
                            yield kind, payload
                            return          # exactly one done, then stop
                        elif kind == "error":
                            yield kind, payload
                            return          # one in-band error frame ends it
                        yield kind, payload
                    # A stream that simply stopped carried no answer at all.
                    raise CloudError("bad_response",
                                     "stream ended without a done event")
                except CloudError as e:
                    if not emitted:
                        raise           # pre-first-byte → step()'s ladder
                    # mirrors step() L1491's failure path, post-first-byte:
                    # in-band instead of a retry, shaped like the frame
                    # _guarded_stream emits (the same five keys).
                    yield "error", {
                        "error": "%s%s: %s" % (
                            e.kind, ("/%s" % e.status) if e.status else "",
                            _sanitize(e.message or "", 200)),
                        "kind": e.kind,
                        "status": e.status,
                        "provider": provider,
                        "model": model,
                    }

            # mirrors step() L1463 — this provider's window decides the
            # budget, not Config.context_size
            info = MR.describe_limit(model, base)
            limit = int(info["limit"])
            # mirrors step() L1465 — keep in sync on any change
            fit = self._fit(messages, system, tools, limit)
            # mirrors step() L1466 — keep in sync on any change
            self._record_context(provider, model, base, info, fit)
            trimmed_msgs = fit["messages"]

            # mirrors step() L1472 — wire policy is per provider
            wire = _apply_reasoning_policy(
                trimmed_msgs, provider, _is_thinking_model(provider, model))
            seen_ids = set()                        # mirrors step() L1474
            while True:                             # mirrors step() L1475
                entry = self.keys.next_key(provider)
                if entry is None:
                    break
                if entry.get("id") in seen_ids:
                    # mirrors step() L1479 — every usable key had its turn;
                    # server/bad-response errors park no key, so without
                    # this the loop would spin on the same key forever.
                    break
                seen_ids.add(entry.get("id"))       # mirrors step() L1484
                # mirrors step() L1489 — one context retry per KEY
                context_retried = False
                tried.append("%s/%s" % (provider, mask(entry.get("key", ""))))
                try:
                    # mirrors step() L1491 — keep in sync on any change
                    yield from _pump(entry, wire)
                    # One ("done", ...) frame went out, or one in-band
                    # ("error", ...) frame: either way this stream is over —
                    # no further keys, no further providers.
                    return
                except CloudError as e:
                    # mirrors step() L1494 — keep in sync on any change
                    if (e.status == 400
                            and "reasoning_content" in (e.message or "")):
                        log.warning("%s 400 named reasoning_content — "
                                    "flipping the replay policy once", provider)
                        wire = _apply_reasoning_policy(
                            trimmed_msgs, provider,
                            provider in THINKING_PROVIDERS)
                        try:
                            yield from _pump(entry, wire)
                            return
                        except CloudError as retry_err:
                            e = retry_err
                    # mirrors step() L1509 — keep in sync on any change
                    if not context_retried and MR.is_context_error(e.message or ""):
                        context_retried = True
                        learned = MR.record_limit(model, base, text=e.message)
                        if learned and learned != limit:
                            info = MR.describe_limit(model, base)
                            info["limit"] = learned
                            limit = learned
                        # mirrors step() L1526 — cut harder, same key once
                        fit = self._fit(messages, system, tools, limit,
                                        CONTEXT_RETRY_FACTOR)
                        self._record_context(provider, model, base, info, fit)
                        wire = _apply_reasoning_policy(
                            fit["messages"], provider,
                            _is_thinking_model(provider, model))
                        log.warning("%s says the context does not fit (%d "
                                    "tokens, window %s) — retrying once at %d",
                                    provider, fit["before"], limit,
                                    int(limit * CONTEXT_RETRY_FACTOR))
                        try:
                            yield from _pump(entry, wire)
                            return
                        except CloudError as retry_err:
                            e = retry_err
                    # mirrors step() L1540 — keep in sync on any change
                    last = e
                    self._handle_error(provider, entry, e, model)
                    if e.kind in ("network",):
                        # mirrors step() L1542, minus the raise: no other
                        # provider is reachable either, so end silently and
                        # let the caller decide about the local fallback.
                        return
        # mirrors step() L1546 — every provider ran out of attempts
        detail = ""
        if last is not None:
            detail = "%s%s: %s" % (
                last.kind, ("/%s" % last.status) if last.status else "",
                _sanitize(last.message or "", 200))
        log.warning("cloud stream providers exhausted: %s%s",
                    ", ".join(tried) or "no keys",
                    (" — " + detail) if detail else "")
        # mirrors step() L1551's BackendExhausted message — logged, never
        # raised: an empty stream is what tells the caller to go local.
        return

    def _handle_error(self, provider: str, entry: dict, err: CloudError,
                      model: str) -> None:
        safe = redact(_sanitize(err.message or err.kind, 500),
                      self.keys.all_keys())
        if err.kind == "rate_limit":
            wait = self.keys.mark_rate_limited(provider, entry.get("id"),
                                               err.retry_after)
            log.warning("%s key %s hit 429 — parked for %.0fs%s",
                        provider, mask(entry.get("key", "")), wait,
                        ("; %s" % safe[:120]) if safe else "")
        elif err.kind == "auth":
            self.keys.mark_auth_failed(provider, entry.get("id"), safe)
            log.warning("%s key %s rejected (%s) — marked dead",
                        provider, mask(entry.get("key", "")), err.status)
        else:
            log.warning("%s step failed (%s, status %s): %s",
                        provider, err.kind, err.status, safe[:200])

    def _log_ok(self, step: Step, provider: str, model: str, entry: dict) -> None:
        u = step.usage or {}
        log.info("cloud step ok: provider=%s model=%s key=%s prompt=%s completion=%s tools=%s",
                 provider, step.model or model, mask(entry.get("key", "")),
                 u.get("prompt_tokens"), u.get("completion_tokens"),
                 len(step.tool_calls))
        # The provider's own prompt count beats our estimate, and the meter is
        # only as good as the number behind it.
        prompt = u.get("prompt_tokens")
        if self._ctx.get("provider") == provider and prompt:
            try:
                used = int(prompt)
            except (TypeError, ValueError):
                return
            limit = int(self._ctx.get("max") or 0)
            self._ctx["used"] = used
            self._ctx["pct"] = round(used / limit * 100, 1) if limit else 0.0
            self._ctx["exact"] = True

    # ---------- for Settings ----------
    def status(self) -> dict:
        self.refresh()
        cfg = self.keys.load()
        for p in ADAPTERS:
            _warn_unknown_model(
                p, ((cfg.get("providers") or {}).get(p) or {}).get("model", ""))
        return {
            "online": self.online(),
            "state": self._net_state(),
            "backend_mode": self.mode,
            "fallback_to_local": self.fallback_to_local,
            "why_not_cloud": self.why_not_cloud(),
            "models": MODELS,
            "prefer_cloud": bool(cfg.get("prefer_cloud", True)),
            "cost_ack": bool(cfg.get("cost_ack", False)),
            "fallback": self.fallback_order(),
            "usable": self.usable(),
            "will_use_cloud": self.should_use_cloud(),
            "local_model": _local_model_name(),
            # The model's real window plus the last turn's usage — what the
            # status bar paints, so it is answered from here and not re-derived
            # from the local engine's KV cache.
            "context": self.context_state(),
            "providers": {
                p: {
                    "model": (cfg.get("providers") or {}).get(p, {}).get("model"),
                    "enabled": (cfg.get("providers") or {}).get(p, {}).get("enabled", True),
                    "cost_per_1k": cost_per_1k(
                        p, (cfg.get("providers") or {}).get(p, {}).get("model", "")),
                    # Read-only view of the window each provider would get —
                    # Settings shows it next to the model field.
                    "context_limit": MR.describe_limit(
                        (cfg.get("providers") or {}).get(p, {}).get("model", ""),
                        (cfg.get("providers") or {}).get(p, {}).get("base_url", "")
                    )["limit"],
                    "keys": self.keys.summary().get(p, {}),
                }
                for p in ADAPTERS
            },
        }

    def test_key(self, provider: str, key: str, model: str = "") -> dict:
        """One cheap round-trip to prove a key works. Never logs the key."""
        if provider not in ADAPTERS:
            return {"ok": False, "error": "unknown provider"}
        self.refresh()
        cfg = self.keys.load()
        model = model or (cfg.get("providers") or {}).get(provider, {}).get("model") or ""
        adapter = ADAPTERS[provider]()
        import time as _t
        t0 = _t.time()
        try:
            url, body, headers = adapter.request(
                key=key, model=model, system="Reply with the single word: ok",
                messages=[{"role": "user", "content": "ping"}], tools=[],
                max_tokens=config.PROBE_MAX_TOKENS)
            data = self._http(url, body, headers, 20)
            adapter.parse(data, model=model, key_id="")
            return {"ok": True, "latency_ms": int((_t.time() - t0) * 1000),
                    "model": model}
        except CloudError as e:
            return {"ok": False, "kind": e.kind, "status": e.status,
                    "error": redact(e.message or e.kind, [key])[:300]}
        except Exception as e:
            return {"ok": False, "kind": "error",
                    "error": "%s: %s" % (type(e).__name__, e)}

    # ---------- parallel tools ----------
    def tools_are_parallel_safe(self, names):
        """True when a batch of tool names can run concurrently."""
        return (len(names) > 1
                and not any(n in SERIAL_ONLY_TOOLS for n in names))

    def run_tools_parallel(self, calls, *,
                            executor=None, timeout=30.0):
        """Run a list of tool calls concurrently; return results in order.

        Args:
            calls: list of dicts, each {"name": str, "args": dict,
                   "fn": callable}. The caller resolves `fn` — this
                   method does not know how tools are dispatched.

        Returns:
            list of {"name": str, "args": dict, "result": str,
                     "error": str|None, "elapsed_ms": int}
            — one per input call, SAME ORDER as input.

        Rules:
          - Bounded: at most MAX_PARALLEL_TOOLS workers.
          - Each call has its own timeout; a timeout becomes
            error="timeout" with result="".
          - An exception inside one call NEVER kills the batch — it
            becomes error="<ExcType>: <msg>" and other calls continue.
          - Never raises. Never touches self._messages. Never writes
            to disk. Caller owns side effects.
          - If len(calls) <= 1, run inline (no pool) — a single call
            gains nothing from a thread hop.
        """
        if not calls:
            return []
        if len(calls) <= 1:
            c = calls[0]
            name = c.get("name") or ""
            args = c.get("args") or {}
            fn = c.get("fn")
            t0 = time.time()
            try:
                res = fn(**args) if callable(fn) else ""
                if res is None:
                    res = ""
                elif not isinstance(res, str):
                    try:
                        res = str(res)
                    except Exception:
                        res = ""
                return [{"name": name, "args": args, "result": res,
                         "error": None, "elapsed_ms": int((time.time()-t0)*1000)}]
            except Exception as e:
                return [{"name": name, "args": args, "result": "",
                         "error": "%s: %s" % (type(e).__name__, e),
                         "elapsed_ms": int((time.time()-t0)*1000)}]

        own_pool = executor is None
        pool = executor or ThreadPoolExecutor(
            max_workers=min(MAX_PARALLEL_TOOLS, len(calls)))

        def _run(c):
            name = c.get("name") or ""
            args = c.get("args") or {}
            fn = c.get("fn")
            t0 = time.time()
            try:
                res = fn(**args) if callable(fn) else ""
                if res is None:
                    res = ""
                elif not isinstance(res, str):
                    try:
                        res = str(res)
                    except Exception:
                        res = ""
                return {"name": name, "args": args, "result": res,
                        "error": None, "elapsed_ms": int((time.time()-t0)*1000)}
            except FuturesTimeoutError:
                return {"name": name, "args": args, "result": "",
                        "error": "timeout", "elapsed_ms": int((time.time()-t0)*1000)}
            except Exception as e:
                return {"name": name, "args": args, "result": "",
                        "error": "%s: %s" % (type(e).__name__, e),
                        "elapsed_ms": int((time.time()-t0)*1000)}

        try:
            futures = [pool.submit(_run, c) for c in calls]
            results = [f.result(timeout=timeout) for f in futures]
            return results
        except Exception as e:
            t1 = time.time()
            results = []
            for c in calls:
                results.append({"name": c.get("name") or "", "args": c.get("args") or {},
                                "result": "", "error": "%s: %s" % (type(e).__name__, e),
                                "elapsed_ms": int((t1)*0)})  # best effort
            return results
        finally:
            if own_pool:
                pool.shutdown(wait=False)


_BASE_URL_ADAPTERS = ("OpenAIAdapter", "OpenRouterAdapter",
                      "DeepSeekAdapter", "GeminiAdapter", "AnthropicAdapter")


def _local_model_name() -> str:
    """File name of the local weights, e.g. ``gemma-4-E4B-it.litertlm``.

    There is no module-level MODEL_PATH — the path lives in config.json — so
    fall back to a plain read of that file rather than Config.load(), which
    would repair/rewrite it as a side effect.
    """
    try:
        import config
        p = str(getattr(config, "MODEL_PATH", "") or "")
        if not p:
            cfg_file = getattr(config, "CONFIG_FILE", None)
            if cfg_file and Path(cfg_file).exists():
                data = json.loads(Path(cfg_file).read_text(encoding="utf-8"))
                p = str(data.get("model_path") or "")
        return p.rsplit("/", 1)[-1] if p else ""
    except Exception:
        return ""


def friendly_local_name() -> str:
    """``gemma-4-E4B-it.litertlm`` → ``Gemma 4`` — what the status bar shows."""
    raw = _local_model_name()
    if not raw:
        return "Local model"
    base = raw
    for ext in (".litertlm", ".gguf", ".bin", ".safetensors"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    m = re.match(r"(?i)^gemma-(\d+)", base)
    if m:
        return "Gemma " + m.group(1)
    return base.replace("_", " ").replace("-", " ").strip() or "Local model"
