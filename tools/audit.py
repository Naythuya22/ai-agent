"""Self-audit tools for agent mode."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

_MAX_CHARS = 4000
_LOG_TAIL_BYTES = 4096
# Batch 4.2e: refreshed after 4.2 shipped parallel_tool_calls.
# self_audit / capability_list are the tool that reports this list,
# so they are NOT gaps.
# 2026-10-10: multi_modal_input was half-wrong — local vision is wired
# (core/llm.py + routes/chat.py image upload). Split into cloud_vision_input
# (image turns stay local; per-provider payloads not wired) and audio_input
# (no route, no STT). cost_tracking -> cost_tracking_cumulative: the rate
# table (cost_per_1k) and per-turn usage exist; a running session total does
# not.
_KNOWN_GAPS = (
    "vector_memory",
    "cost_tracking_cumulative",  # rate exists, cumulative missing
    "eval_harness",              # code tests exist, AI eval does not
    "self_modification",
    "cloud_vision_input",        # local yes, cloud no
    "audio_input",               # local + cloud no
)
_BASELINE_TOOLS = (
    "list_files", "read_file", "write_file",
    "run_command", "run_python",
    "set_alarm", "set_timer", "create_calendar_event", "open_url",
    "web_search", "web_fetch", "smart_search",
    "memory_read", "memory_write", "memory_search", "memory_forget",
)

try:
    _HOME = Path.home()
except Exception:
    _HOME = Path(".")
_MODES_DIR = Path(__file__).parent.parent / "modes"
_MEMORY_FILE = _HOME / ".ai-agent" / "memory.json"
_LOG_FILE = _HOME / ".ai-agent" / "logs" / "agent.log"

_WARN_RE = re.compile(r"\bWARN(?:ING)?\b|(?:^|\s)W(?:[/:\s]|$)", re.I)
# (?<!") … (?!") — do NOT count a token that is itself inside double quotes.
# Provider 402 bodies put a JSON key `"error":` on the same WARNING line, and
# reflection notes quote messages like `"Error: duplicate fact"`. Without the
# lookaround those counted as real errors (tail of 2026-10-10: E=6, all false;
# true count 0). Real ERROR lines from the logger are bare tokens, never quoted.
_ERR_RE = re.compile(r'(?<!")\b(?:ERROR|CRITICAL|FATAL|EXCEPTION|TRACEBACK)\b(?!")'
                     r"|(?:^|\s)E(?:[/:\s]|$)", re.I)


def _tool_names() -> tuple[str, ...]:
    """Function names from ALL_TOOLS (not the __all__ string constants)."""
    try:
        import importlib
        mod = importlib.import_module("tools")
        tools_list = getattr(mod, "ALL_TOOLS", None)
        if isinstance(tools_list, (list, tuple)) and tools_list:
            names = []
            for t in tools_list:
                n = getattr(t, "__name__", None)
                if isinstance(n, str) and n and not n.endswith("_TOOLS"):
                    names.append(n)
            if names:
                return tuple(dict.fromkeys(names))
        return _BASELINE_TOOLS
    except Exception:
        return _BASELINE_TOOLS


def _modes() -> tuple[str, ...]:
    """Mode module names present in modes/ (chat, agent, ...)."""
    try:
        return tuple(sorted(n[:-3] for n in os.listdir(_MODES_DIR)
                            if n.endswith(".py") and not n.startswith("_")))
    except Exception:
        return ()


def _providers() -> tuple[str, ...]:
    """Provider names from DEFAULT_CONFIG["providers"] subkey."""
    try:
        from core import key_manager as KM
        cfg = getattr(KM, "DEFAULT_CONFIG", None)
        if isinstance(cfg, dict):
            provs = cfg.get("providers") or {}
            if isinstance(provs, dict) and provs:
                return tuple(str(k) for k in provs)
    except Exception:
        pass
    return ()


def _context_window() -> str:
    """Local KV-cache window (8K by default), or '?' when unknown."""
    try:
        from core import model_registry as MR
        # "gemma:4b" is the shipped local default; asking for it gives
        # the local window, not the 128K unknown-cloud fallback.
        for model in ("gemma:4b", "gemma-4-E4B-it.litertlm", ""):
            try:
                r = MR.resolve_limit(model, "")
            except Exception:
                continue
            if isinstance(r, int) and r > 0:
                tag = "(local)" if model else "(unknown)"
                return f"{r} tokens {tag}"
            if isinstance(r, dict):
                for k in ("limit", "total", "max", "context_window"):
                    if isinstance(r.get(k), int) and r[k] > 0:
                        tag = "(local)" if model else "(unknown)"
                        return f"{r[k]} tokens {tag}"
    except Exception:
        pass
    return "?"


def _memory_stats() -> dict:
    """Counts from ~/.ai-agent/memory.json; '?' per field if unreadable."""
    out = {"facts": "?", "topics": "?", "identity": "?"}
    try:
        data = json.loads(_MEMORY_FILE.read_text(encoding="utf-8"))
        for src, dst in (("facts", "facts"), ("recent_topics", "topics"),
                         ("identity", "identity")):
            v = data.get(src) if isinstance(data, dict) else None
            if isinstance(v, (list, tuple, dict)):
                out[dst] = len(v)
    except Exception:
        pass
    return out


def _log_stats() -> str:
    """Bounded tail of agent.log: line count plus warn/error counts."""
    try:
        with open(_LOG_FILE, "rb") as fh:
            size = fh.seek(0, os.SEEK_END)
            fh.seek(max(0, size - _LOG_TAIL_BYTES))
            raw = fh.read().decode("utf-8", errors="replace")
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        if size > _LOG_TAIL_BYTES and lines:
            lines = lines[1:]  # drop the partial first tail line
        w = sum(1 for ln in lines if _WARN_RE.search(ln))
        e = sum(1 for ln in lines if _ERR_RE.search(ln))
        return f"tail {_LOG_TAIL_BYTES} B, {len(lines)} lines, W={w}, E={e}"
    except Exception:
        return "?"


def _collect() -> dict:
    """One read-only pass over every audit source. Never raises."""
    return {"tools": _tool_names(), "modes": _modes(),
            "providers": _providers(), "context": _context_window(),
            "memory": _memory_stats(), "log": _log_stats(),
            "gaps": _KNOWN_GAPS}


def _cap(text: str) -> str:
    """Hard cap every report at ~4000 chars."""
    return text if len(text) <= _MAX_CHARS else text[:_MAX_CHARS] + "\n... (truncated)"


def self_audit() -> str:
    """Report what this agent can and cannot do, right now.

    Use this before answering anything about your own capabilities, and
    before promising a feature you may not have. Read-only: never runs
    commands, never writes files, never talks to a network.

    Reports: tools (count + names), modes, providers, context window,
    memory stats, recent log warnings/errors, and the known gaps
    (vector_memory, cost_tracking_cumulative, eval_harness,
    self_modification, cloud_vision_input, audio_input). Missing sources
    become '?'.
    """
    s = _collect()
    mem, tools, ctx = s["memory"], s["tools"], s["context"]
    lines = [
        "self-audit (read-only)",
        f"tools: {len(tools)} — {', '.join(tools)}" if tools else "tools: ?",
        f"modes: {', '.join(s['modes'])}" if s["modes"] else "modes: ?",
        (f"providers: {len(s['providers'])} — {', '.join(s['providers'])}"
         if s["providers"] else "providers: ?"),
        "context: ?" if ctx == "?" else f"context: {ctx} (model_registry)",
        f"memory: facts={mem['facts']}, topics={mem['topics']}, "
        f"identity={mem['identity']}",
        f"log: {s['log']}",
        f"gaps: {', '.join(s['gaps'])}",
    ]
    return _cap("\n".join(lines) + "\n")


def capability_list() -> str:
    """One-line-per-capability summary.

    Returns one capability per line in the form
        "<name>: yes|no — <one-line reason>"
    Same sources as self_audit(); only the shape differs. Useful for a
    quick "what am I right now?" check. Read-only; never raises.
    """
    s = _collect()
    mem = s["memory"]
    lines = [f"{n}: yes — registered tool" for n in s["tools"]]
    lines += ([f"{n}_mode: yes — modes/{n}.py present" for n in s["modes"]]
              or ["modes: no — modes/ directory unreadable"])
    lines += ([f"{p}: yes — provider in key_manager.DEFAULT_CONFIG"
               for p in s["providers"]]
              or ["providers: no — key_manager.DEFAULT_CONFIG unreadable"])
    # Batch 4.2e: code mode lives in routes/code.py + core/opencode.py,
    # not in modes/. Report it separately so "modes: chat, agent" is not
    # read as "no code mode".
    root = Path(__file__).parent.parent
    code_route = (root / "routes" / "code.py").is_file()
    code_core = (root / "core" / "opencode.py").is_file()
    if code_route and code_core:
        lines.append("code_mode: yes (routes/code.py + core/opencode.py)")
    else:
        missing = []
        if not code_route:
            missing.append("routes/code.py")
        if not code_core:
            missing.append("core/opencode.py")
        lines.append("code_mode: no (missing " + ", ".join(missing) + ")")
    lines.append("context_window: no — model_registry unavailable"
                 if s["context"] == "?" else
                 f"context_window: yes — {s['context']} (model_registry)")
    known = "?" not in (str(mem["facts"]), str(mem["topics"]),
                        str(mem["identity"]))
    lines.append(f"memory: yes — facts={mem['facts']}, "
                 f"topics={mem['topics']}, identity={mem['identity']}"
                 if known else "memory: no — ~/.ai-agent/memory.json unreadable")
    lines.append(f"log: yes — {s['log']}" if s["log"] != "?" else
                 "log: no — ~/.ai-agent/logs/agent.log unreadable")
    # Batch 4.2e: do NOT give the model a free-form reason to
    # embellish — the audit only knows that a name is on the gap list.
    lines += [f"{g}: not implemented" for g in s["gaps"]]
    return _cap("\n".join(lines) + "\n")
