"""Child process for the built-in Code backend (core/builtin.py).

Runs one Code-mode turn on the app's own agent engine and prints opencode
`--format json` events, one per line:

    {"type": "step_start", "sessionID": ..., "part": {}}
    {"type": "text",       "sessionID": ..., "part": {"text": ...}}
    {"type": "tool_use",   "sessionID": ..., "part": {"tool": ...,
        "state": {"status": ..., "input": {...}, "title": ...}}}
    {"type": "step_finish","sessionID": ..., "part": {"tokens": {...}}}
    {"type": "error",      "sessionID": ..., "error": {"name": ...,
        "data": {"message": ...}}}

core.code_stream parses exactly this dialect for the generic backends, so
the routes and the UI see the same shapes they see from opencode.

The conversation transcript is kept per session id under the shared store
(core.builtin.history_dir()) in modes/agent.py's entry shapes —
{"role": "user"|"assistant", "text": ...}, {"role": "tool_call", ...},
{"role": "tool_result", ...} — because core.agent_backend.step() (via
normalize()) already understands them.

Tool set: the file tools (jailed to the active project, which is the Code
UI's selected folder), plus run_command and run_python — the same shell
power the CLI backends have inside the project folder. Nothing else: no
memory, web, Android or video tools in Code mode.

v1 scope: the cloud route only (AgentBackend.step). When no cloud route
exists (no provider key, Local Only, offline) the run ends with one
friendly error event instead of dying silently.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from pathlib import Path

MAX_STEPS = 30
MAX_TOOL_RESULT_CHARS = 16000
HISTORY_KEEP = 80

SYSTEM_PROMPT = (
    "You are the built-in coding engine of an AI assistant app running "
    "on Android (Termux). You work inside one project folder — the "
    "current directory — and your file tools are restricted to it.\n"
    "\n"
    "How to work:\n"
    "- Use list_files and read_file to understand the project before "
    "changing it.\n"
    "- Use write_file to create or rewrite files with their complete "
    "contents.\n"
    "- Use run_python or run_command to run code and tests, and check "
    "the output before claiming success.\n"
    "- Make the smallest change that fully solves the task. Do not "
    "rewrite unrelated code.\n"
    "- If a step fails, read the error, fix the cause, and try again "
    "instead of repeating the same call.\n"
    "- When the task is done, stop calling tools and reply with a short "
    "summary of what you changed and how you verified it.\n"
    "\n"
    "Reply in the language the user wrote in."
)

# Engine tool name -> (opencode tool name for the event stream).
# write_file must surface as opencode's "write" so core.code_stream
# reports a file create/edit frame for it.
_EVENT_TOOL = {
    "write_file": "write",
    "read_file": "read",
    "list_files": "list",
    "run_command": "bash",
    "run_python": "bash",
}

_SID_RE = re.compile(r"^[A-Za-z0-9_-]{8,80}$")


def _emit(event: dict) -> None:
    sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _code_tools():
    """(wrapped tools for the engine, {name: function} for execution)."""
    from tools.file import list_files, read_file, write_file
    from tools.shell import run_command
    from tools.python_runner import run_python

    fns = [list_files, read_file, write_file, run_command, run_python]
    try:
        from litert_lm import tool_from_function
        wrapped = [tool_from_function(fn) for fn in fns]
    except Exception:
        wrapped = list(fns)
    return wrapped, {fn.__name__: fn for fn in fns}


def _history_path(session_id: str) -> Path:
    from core.builtin import history_dir
    return history_dir() / f"{session_id}.json"


def _load_history(session_id: str) -> list:
    try:
        data = json.loads(_history_path(session_id).read_text(
            encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def _save_history(session_id: str, messages: list) -> None:
    keep = list(messages[-HISTORY_KEEP:])
    # Never resume mid tool-pair: a transcript that opens with a tool
    # result the model never asked for confuses every provider.
    while keep and keep[0].get("role") in ("tool_call", "tool_result"):
        keep.pop(0)
    path = _history_path(session_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(keep, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(path)
    except OSError:
        pass  # Losing history restarts the conversation; it must not fail the run.


def _parse_args(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            data = json.loads(raw)
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}
    return {}


def _execute(fn_map: dict, name: str, args: dict) -> str:
    fn = fn_map.get(name)
    if fn is None:
        return f"Error: unknown tool: {name}"
    try:
        result = fn(**args)
    except TypeError as e:
        return f"Error: bad arguments for {name}: {e}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
    text = result if isinstance(result, str) else str(result)
    if len(text) > MAX_TOOL_RESULT_CHARS:
        text = (text[:MAX_TOOL_RESULT_CHARS]
                + f"\n... [truncated {len(text) - MAX_TOOL_RESULT_CHARS} chars]")
    return text


def _tool_event(session_id: str, name: str, args: dict, result: str,
                existed: bool) -> dict:
    event_tool = _EVENT_TOOL.get(name, name)
    spec = dict(args)
    title = ""
    if name == "write_file":
        spec["filePath"] = str(args.get("path") or "")
        title = f"write {args.get('path') or ''}".strip()
    elif name in ("read_file", "list_files"):
        title = f"{name} {args.get('path') or args.get('directory') or ''}".strip()
    elif name in ("run_command", "run_python"):
        cmd = str(args.get("command") or args.get("code") or "")
        title = cmd[:120]
    state = {
        "status": "error" if result.startswith("Error:") else "completed",
        "input": spec,
        "title": title[:200],
    }
    if name == "write_file":
        state["metadata"] = {"exists": existed}
    return {
        "type": "tool_use",
        "sessionID": session_id,
        "part": {"tool": event_tool, "callID": uuid.uuid4().hex,
                 "state": state},
    }


def _action_label(name: str, args: dict, result: str) -> str:
    """One short human label for a finished tool call — the material
    the closing line is built from when the model itself says nothing."""
    if name == "write_file":
        label = f"wrote {args.get('path') or 'a file'}"
    elif name == "read_file":
        label = f"read {args.get('path') or 'a file'}"
    elif name == "list_files":
        label = f"listed {args.get('path') or args.get('directory') or 'the folder'}"
    elif name == "run_command":
        label = f"ran `{str(args.get('command') or '')[:60]}`"
    elif name == "run_python":
        label = "ran Python code"
    else:
        label = name or "tool"
    if result.startswith("Error:"):
        label += " (failed)"
    return label


def run(prompt: str, workdir: str, session_id: str | None) -> int:
    sid = session_id if session_id and _SID_RE.match(session_id) \
        else uuid.uuid4().hex

    # Everything from here on is inside the guard: a child that dies before
    # its first event is reported by the caller as "exited 1 with no
    # output", which tells the user nothing. Whatever goes wrong —
    # an import, the engine's construction, a bad store file — must come
    # back as one error event carrying the real exception.
    try:
        return _run_guarded(prompt, workdir, sid)
    except Exception as e:
        import traceback
        _emit({"type": "error", "sessionID": sid, "error": {
            "name": type(e).__name__,
            "data": {"message":
                     f"{type(e).__name__}: {e}\n"
                     + "\n".join(traceback.format_exc()
                                 .rstrip().splitlines()[-4:])}}})
        return 1


def _run_guarded(prompt: str, workdir: str, sid: str) -> int:
    try:
        os_chdir = Path(workdir)
        if os_chdir.is_dir():
            import os
            os.chdir(os_chdir)
    except OSError:
        pass

    from core.agent_backend import AgentBackend, BackendExhausted

    wrapped, fn_map = _code_tools()
    messages = _load_history(sid)
    messages.append({"role": "user", "text": prompt})

    backend = AgentBackend()
    spoke = False
    done_labels: list[str] = []
    try:
        for _step_no in range(MAX_STEPS):
            _emit({"type": "step_start", "sessionID": sid, "part": {}})
            step = backend.step(system=SYSTEM_PROMPT, messages=messages,
                                tools=wrapped)
            if step.text:
                spoke = True
                _emit({"type": "text", "sessionID": sid,
                       "part": {"text": step.text}})
                messages.append({"role": "assistant", "text": step.text})
            if step.usage:
                _emit({"type": "step_finish", "sessionID": sid,
                       "part": {"tokens": step.usage, "cost": 0}})
            if not step.tool_calls:
                if not spoke and done_labels:
                    # The model did the work and finished without a
                    # word — the UI would show "(no output)". Close
                    # with a deterministic summary of the actions
                    # instead; no extra model call is spent on it.
                    shown = done_labels[:5]
                    if len(done_labels) > 5:
                        shown.append(
                            f"and {len(done_labels) - 5} more actions")
                    closing = "Done — " + "; ".join(shown) + "."
                    _emit({"type": "text", "sessionID": sid,
                           "part": {"text": closing}})
                    messages.append({"role": "assistant",
                                     "text": closing})
                break
            for call in step.tool_calls:
                fn_info = call.get("function", {}) \
                    if isinstance(call, dict) else {}
                name = fn_info.get("name", "")
                args = _parse_args(fn_info.get("arguments"))
                existed = False
                if name == "write_file" and args.get("path"):
                    existed = Path(str(args["path"])).exists()
                result = _execute(fn_map, name, args)
                done_labels.append(_action_label(name, args, result))
                _emit(_tool_event(sid, name, args, result, existed))
                messages.append({"role": "tool_call", "tool": name,
                                 "args": args,
                                 "reasoning": step.reasoning,
                                 "thought_signature": call.get("thought_signature") if isinstance(call, dict) else None})
                messages.append({"role": "tool_result", "tool": name,
                                 "text": result})
        else:
            _emit({"type": "text", "sessionID": sid, "part": {
                "text": f"(Stopped after {MAX_STEPS} steps — the task "
                        "may be incomplete.)"}})
            messages.append({"role": "assistant",
                             "text": f"(Stopped after {MAX_STEPS} steps.)"})
    except BackendExhausted:
        _emit({"type": "error", "sessionID": sid, "error": {
            "name": "NoCloudRoute",
            "data": {"message":
                     "No cloud model is available for Code mode. Add a "
                     "provider key in Settings (the same keys Agent mode "
                     "uses), or switch the Agent backend away from Local "
                     "Only, then try again."}}})
        _save_history(sid, messages)
        return 1
    except Exception as e:
        _emit({"type": "error", "sessionID": sid, "error": {
            "name": type(e).__name__,
            "data": {"message": str(e)}}})
        _save_history(sid, messages)
        return 1

    _save_history(sid, messages)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="builtin_runner")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--session-id", default=None)
    ns = parser.parse_args(argv)
    return run(ns.prompt, ns.workdir, ns.session_id)


if __name__ == "__main__":
    raise SystemExit(main())
