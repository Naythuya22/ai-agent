"""core/code_stream.py — real-time opencode streaming for Code mode.

opencode replaces aider here. The old implementation shelled out to aider and
guessed at its human-readable stdout ("Applied edit to x.py", "Commit abc123
aider: ..."). `opencode run --format json` instead emits a machine-readable
event stream, so files, tool calls and token usage are read off the events
rather than pattern-matched out of prose.

Used by routes/code.py's /api/code/web/run/stream endpoint.
"""
from __future__ import annotations

import json
import re
import select
import subprocess
import time

from core.log import get_logger

log = get_logger("code_stream")

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

# opencode tool name -> the edit operation we report to the UI.
_EDIT_TOOLS = {"write": "create", "edit": "edit", "patch": "edit"}

# Streaming ceilings, mirroring routes.code.RUN_TIMEOUT for the whole run: a
# child that has gone quiet (stalled provider call, frozen proot mount, an
# opencode bug) is killed instead of pinning the worker thread, code_lock and
# the child forever. The idle budget must stay safely under the hard one,
# and both above zero.
STREAM_IDLE_TIMEOUT = 90     # seconds with no output at all
STREAM_HARD_TIMEOUT = 300    # seconds for the whole run


def _strip_ansi(s: str) -> str:
    """Remove ANSI escape sequences from a string."""
    return _ANSI_RE.sub("", s)


def sse_event(obj: dict) -> str:
    """Format a dict as a Server-Sent Event frame."""
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


class OpencodeError(RuntimeError):
    """Raised when opencode cannot be run at all (missing binary, bad env)."""


def parse_event(raw: str) -> dict | None:
    """One JSON line of `opencode run --format json` output, or None.

    Returns None for anything unparseable rather than raising: a stray
    warning line must not abort a run that is otherwise fine.
    """
    line = _strip_ansi(raw).strip()
    if not line or not line.startswith("{"):
        return None
    try:
        event = json.loads(line)
    except ValueError:
        return None
    return event if isinstance(event, dict) else None


def summarize(events: list[dict]) -> dict:
    """Fold a finished event stream into the shape the API returns.

    Kept separate from the live generator so a one-shot run can reuse it, and
    so the parsing is unit-testable without spawning a subprocess.
    """
    reply_parts: list[str] = []
    files: list[dict] = []
    commits: list[dict] = []
    tools: list[dict] = []
    session_id = None
    tokens = {}
    cost = 0.0
    error = None

    for event in events:
        kind = event.get("type")
        part = event.get("part") or {}
        session_id = event.get("sessionID") or session_id

        if kind == "text":
            text = part.get("text")
            if text:
                reply_parts.append(text)

        elif kind == "error":
            err = event.get("error")
            if isinstance(err, dict):
                data = err.get("data") if isinstance(err.get("data"), dict) else {}
                msg = data.get("message") or err.get("name") or json.dumps(
                    err, ensure_ascii=False)
                if data.get("ref"):
                    msg = f"{msg} (ref {data['ref']})"
                error = msg
            elif err:
                error = str(err)

        elif kind == "tool_use":
            state = part.get("state") or {}
            tool = part.get("tool") or "?"
            spec = state.get("input") or {}
            title = state.get("title") or ""
            entry = {
                "tool": tool,
                "status": state.get("status") or "unknown",
                "title": title[:200],
            }
            path = spec.get("filePath") or spec.get("path")
            if tool in _EDIT_TOOLS and path:
                op = _EDIT_TOOLS[tool]
                # A 'write' to a path that already existed is an edit in
                # everything but name; the UI only shows the label.
                if tool == "write" and (state.get("metadata") or {}).get("exists"):
                    op = "edit"
                entry["op"] = op
                entry["path"] = str(path)
                if not any(f["path"] == entry["path"] for f in files):
                    files.append(entry)
            tools.append(entry)

        elif kind == "step_finish":
            if part.get("tokens"):
                tokens = part["tokens"]
            if part.get("cost"):
                cost = part["cost"]

    return {
        "reply": "".join(reply_parts).strip(),
        "files": files,
        "commits": commits,
        "tools": tools,
        "tokens": tokens,
        "cost": cost,
        "session_id": session_id,
        "error": error,
    }


def run_opencode(cmd, workdir, env, timeout=None):
    """Run opencode to completion, returning (returncode, stdout_lines).

    Used by the non-streaming endpoint. Streaming goes through
    stream_opencode() below so the caller sees output as it arrives.
    """
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(workdir),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as e:
        raise OpencodeError(str(e)) from e
    except subprocess.TimeoutExpired:
        raise OpencodeError(f"opencode timed out after {timeout}s") from None
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, output.splitlines(keepends=True)


def stream_opencode(cmd, workdir, env, lock, on_session=None, on_event=None):
    """Generator — runs opencode, yielding SSE strings as events arrive.

    opencode writes a JSON event per line, so line-buffered reading is all the
    streaming we need; no scraping, no partial-line guessing.

    Args:
        cmd:         argv from core.opencode.build_command()
        workdir:     cwd for the child process
        env:         environment from core.opencode.env()
        lock:        threading.Lock serialising concurrent runs
        on_session:  called with the session id as soon as it is known, so the
                     caller can persist it even if the run is later aborted
        on_event:    called with every parsed event, for live UI updates

    Yields:
        SSE-formatted strings (already ending in \\n\\n)
    """
    proc = None
    timed_out = None
    events: list[dict] = []
    t0 = time.time()
    step_n = 0
    seen_calls: set[str] = set()

    # Callers normally pass oc.env(), which already sets this; defaulting it
    # here too so a bare {} (tests, ad-hoc callers) still reads line by line.
    env = dict(env or {})
    env.setdefault("PYTHONUNBUFFERED", "1")

    with lock:
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(workdir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True,
            )
            last_output = time.time()
            while True:
                # The ceilings are checked before every poll: readline() below
                # blocks forever once opencode stops writing, which would pin
                # this thread, code_lock and the child along with it.
                now = time.time()
                if now - last_output > STREAM_IDLE_TIMEOUT:
                    timed_out = "idle_timeout"
                    break
                if now - t0 > STREAM_HARD_TIMEOUT:
                    timed_out = "hard_timeout"
                    break
                try:
                    ready = bool(select.select([proc.stdout], [], [], 1.0)[0])
                except (OSError, ValueError):
                    # A stdout that cannot be polled (a test fake, a stream
                    # with no fileno): fall back to the blocking read this
                    # loop replaced, so those readers behave as before.
                    ready = True
                if not ready:
                    continue
                line = proc.stdout.readline()
                if line == "" and proc.poll() is not None:
                    break  # EOF: the child closed its output and has exited
                last_output = time.time()
                event = parse_event(line)
                if event is None:
                    # Not JSON: almost certainly a warning. Surface it rather
                    # than dropping it silently, but do not treat it as a
                    # result.
                    noise = _strip_ansi(line).rstrip("\r\n")
                    if noise.strip():
                        yield sse_event({"type": "log", "text": noise})
                    continue

                events.append(event)
                etype = event.get("type")
                if etype == "step_start":
                    # Announced the moment opencode starts a step, which is
                    # seconds before any text lands — the one live signal
                    # `run --format json` gives during generation.
                    step_n += 1
                    yield sse_event({"type": "step", "n": step_n})
                elif etype == "text":
                    text = (event.get("part") or {}).get("text")
                    if text:
                        yield sse_event({"type": "chunk", "text": text})
                elif etype == "tool_use":
                    part = event.get("part") or {}
                    state = part.get("state") or {}
                    spec = state.get("input") or {}
                    tool = part.get("tool") or "?"
                    call_id = part.get("callID") or state.get("id") or ""
                    # opencode can report the same call more than once
                    # (running, then completed): announce it a single time so
                    # the transcript does not show the tool twice.
                    fresh = not call_id or call_id not in seen_calls
                    if call_id and fresh:
                        seen_calls.add(call_id)
                    path = spec.get("filePath") or spec.get("path")
                    if fresh:
                        if tool in _EDIT_TOOLS and path:
                            op = _EDIT_TOOLS[tool]
                            if tool == "write" and (state.get("metadata") or {}).get("exists"):
                                op = "edit"
                            yield sse_event({"type": "file", "op": op, "path": str(path)})
                        else:
                            yield sse_event({
                                "type": "tool",
                                "tool": tool,
                                "status": state.get("status") or "unknown",
                                "title": (state.get("title") or "")[:200],
                            })

                if event.get("sessionID") and on_session:
                    on_session(event["sessionID"])
                if on_event:
                    on_event(event)

            if timed_out is None:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except GeneratorExit:
            # The browser hung up (user navigated away, or pressed stop).
            # opencode is mid-edit here, so it must not be left running.
            _terminate(proc)
            raise
        except OpencodeError:
            raise
        except Exception as e:
            yield sse_event({"type": "error", "error": f"{type(e).__name__}: {e}"})
            return
        finally:
            _terminate(proc)

    elapsed = round(time.time() - t0, 1)
    summary = summarize(events)

    if timed_out == "idle_timeout":
        # _terminate (finally above) has already run; say why the run stopped
        # before the terminal end frame so the UI is not left guessing.
        yield sse_event({
            "type": "error",
            "error": f"opencode idle for {STREAM_IDLE_TIMEOUT}s with no output",
            "reason": "idle_timeout",
        })
    elif timed_out == "hard_timeout":
        yield sse_event({
            "type": "error",
            "error": f"opencode exceeded the {STREAM_HARD_TIMEOUT}s run limit",
            "reason": "hard_timeout",
        })
    elif not summary["reply"] and not summary["files"] and proc and proc.returncode:
        yield sse_event({
            "type": "error",
            "error": summary["error"]
                     or f"opencode exited {proc.returncode} with no output",
        })

    # File events were already yielded live above, so they are not repeated
    # here — only carried in the end payload for clients that want the set.
    yield sse_event({
        "type": "end",
        "reply": summary["reply"] or "(no output)",
        "files": summary["files"],
        "tools": summary["tools"],
        "tokens": summary["tokens"],
        "cost": summary["cost"],
        "session_id": summary["session_id"],
        "elapsed": elapsed,
        "rc": proc.returncode if proc else -1,
    })


def _terminate(proc) -> None:
    """Stop the child if it is still going. Never raises."""
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception as e:
        log.debug("could not terminate the child process: %s: %s",
                  type(e).__name__, e)