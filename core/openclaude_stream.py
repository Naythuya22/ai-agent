"""Claude `stream-json` → the SSE vocabulary Code mode already speaks.

openclaude writes one JSON object per line, but the objects are Claude Code's
own stream-json dialect (`system/init`, `stream_event`, `assistant`, `user`,
`result`), not opencode's `run --format json`. The UI only understands the
frames core/code_stream.py already emits, so everything is folded here into
six normalized event types:

    step    a new model turn started          {"type":"step","n":1}
    chunk   text the user should see now      {"type":"chunk","text":"..."}
    file    an edit landed on a path          {"type":"file","op","path"}
    tool    some other tool call              {"type":"tool","tool","status"}
    end     the run finished (tokens, cost)   {"type":"end", ...}
    log     a line that was not an event      {"type":"log","text":"..."}

`stream_openclaude()` is the generator routes/code.py streams through: same
shape, same timeouts and same child reaping as stream_opencode(), so the two
backends are interchangeable behind code_stream.stream_code().
"""
from __future__ import annotations

import json
import select
import subprocess
import time

from core.code_stream import (
    OpencodeError,
    STREAM_HARD_TIMEOUT,
    STREAM_IDLE_TIMEOUT,
    _strip_ansi,
    _terminate,
    sse_event,
)
from core.log import get_logger

log = get_logger("openclaude_stream")

# Claude tool name -> the edit operation we report to the UI.
# opencode calls them write/edit/patch; Claude calls them Write/Edit/NotebookEdit.
_EDIT_TOOLS = {"Write": "create", "Edit": "edit", "NotebookEdit": "edit"}

# Tool names whose input carries the path being changed.
_PATH_KEYS = ("file_path", "path", "notebook_path")


def parse_event(raw: str) -> dict | None:
    """One JSON line of `openclaude -p --output-format stream-json`, or None.

    Returns None for anything unparseable rather than raising: a stray
    warning line must not abort a run that is otherwise fine.
    """
    line = _strip_ansi(raw or "").strip()
    if not line or not line.startswith("{"):
        return None
    try:
        event = json.loads(line)
    except ValueError:
        return None
    return event if isinstance(event, dict) else None


def _tokens_from(usage: dict) -> dict:
    """Claude usage -> the shape opencode's end frame carries."""
    try:
        inp = int(usage.get("input_tokens") or 0)
        out = int(usage.get("output_tokens") or 0)
        cache_r = int(usage.get("cache_read_input_tokens") or 0)
        cache_w = int(usage.get("cache_creation_input_tokens") or 0)
    except (TypeError, ValueError):
        return {}
    return {
        "total": inp + out + cache_r + cache_w,
        "input": inp,
        "output": out,
        "reasoning": 0,
        "cache": {"write": cache_w, "read": cache_r},
    }


class Normalizer:
    """Stateful fold of raw stream-json events into the six frame types.

    State is required across events: the step counter, the tool ids already
    announced (Claude reports a tool_use twice — once partial, once whole)
    and whether this message's text already reached the UI as deltas.
    """

    def __init__(self) -> None:
        self.step_n = 0
        self.seen_tools: set[str] = set()
        self.saw_text_delta = False

    # -- helpers -----------------------------------------------------------

    def _path_of(self, spec: dict) -> str | None:
        for key in _PATH_KEYS:
            if spec.get(key):
                return str(spec[key])
        return None

    def _tool_events(self, block: dict) -> list[dict]:
        """One tool_use content block → zero or one file/tool event."""
        call_id = str(block.get("id") or "")
        if call_id:
            if call_id in self.seen_tools:
                return []
            self.seen_tools.add(call_id)
        name = str(block.get("name") or "?")
        spec = block.get("input") if isinstance(block.get("input"), dict) else {}
        path = self._path_of(spec)
        if name in _EDIT_TOOLS and path:
            return [{
                "type": "file",
                "op": _EDIT_TOOLS[name],
                "path": path,
                "tool": name,
            }]
        if name == "Bash":
            title = str(spec.get("command") or "")[:200]
        elif path:
            title = path[:200]
        else:
            title = ""
        return [{
            "type": "tool",
            "tool": name,
            # Announced when the aggregated assistant message lands, i.e. as
            # the CLI is about to run it; the summary reports the same status
            # so a finished run does not read as a tool still in flight.
            "status": "completed",
            "title": title,
        }]

    def _end_event(self, raw: dict) -> dict:
        result = raw.get("result")
        reply = result if isinstance(result, str) else ""
        usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
        error = None
        if raw.get("is_error"):
            error = reply or str(raw.get("subtype") or "openclaude error")
        elif raw.get("error"):
            error = str(raw.get("error"))
        return {
            "type": "end",
            "reply": reply,
            "tokens": _tokens_from(usage),
            "cost": float(raw.get("total_cost_usd") or 0.0),
            "session_id": raw.get("session_id"),
            "error": error,
        }

    # -- the fold ----------------------------------------------------------

    def feed(self, raw: dict) -> list[dict]:
        """One parsed stream-json object → zero or more normalized events."""
        if not isinstance(raw, dict):
            return []
        out: list[dict] = []
        sid = raw.get("session_id")

        def add(event: dict) -> None:
            if sid and not event.get("session_id"):
                event["session_id"] = sid
            out.append(event)

        kind = raw.get("type")

        if kind == "stream_event":
            ev = raw.get("event") if isinstance(raw.get("event"), dict) else {}
            etype = ev.get("type")
            if etype == "message_start":
                self.step_n += 1
                self.saw_text_delta = False
                add({"type": "step", "n": self.step_n})
            elif etype == "content_block_delta":
                delta = ev.get("delta") if isinstance(ev.get("delta"), dict) else {}
                text = delta.get("text")
                if delta.get("type") == "text_delta" and text:
                    self.saw_text_delta = True
                    add({"type": "chunk", "text": str(text)})
            return out

        if kind == "assistant":
            # The whole message, tool_use inputs included. Used for text only
            # when no deltas carried it already, so nothing is doubled.
            msg = raw.get("message") if isinstance(raw.get("message"), dict) else {}
            for block in msg.get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text = block.get("text")
                    if text and not self.saw_text_delta:
                        add({"type": "chunk", "text": str(text)})
                elif block.get("type") == "tool_use":
                    for event in self._tool_events(block):
                        add(event)
            # Next message starts clean even if it arrives without a
            # message_start of its own.
            self.saw_text_delta = False
            if raw.get("error"):
                add({"type": "end", "reply": "", "tokens": {}, "cost": 0.0,
                     "session_id": raw.get("session_id"),
                     "error": str(raw.get("error"))})
            return out

        if kind == "result":
            add(self._end_event(raw))
            return out

        if kind == "error":
            err = raw.get("error") or raw.get("message") or raw
            if not isinstance(err, str):
                err = json.dumps(err, ensure_ascii=False)[:400]
            add({"type": "end", "reply": "", "tokens": {}, "cost": 0.0,
                 "session_id": raw.get("session_id"), "error": err})
            return out

        # system/init, user (tool results), hook events, unknown future
        # types: nothing to show, but the session id still counts.
        return out


def normalize(raw: dict, state: Normalizer | None = None) -> list[dict]:
    """Single-event convenience wrapper. Pass `state` to keep step numbering
    and tool de-duplication across a whole stream — parse_lines() does."""
    return (state or Normalizer()).feed(raw)


def parse_lines(lines) -> list[dict]:
    """Raw output lines → the normalized events of a finished run.

    Warnings and non-JSON chatter become `log` events, exactly as
    core/code_stream.parse_event() callers expect: a run must explain itself
    instead of silently dropping the line.

    Accepts either a list of lines or the whole captured output as one
    string — subprocess callers hand back both shapes.
    """
    if isinstance(lines, str):
        lines = lines.splitlines(keepends=True)
    state = Normalizer()
    events: list[dict] = []
    for line in lines or []:
        raw = parse_event(line if isinstance(line, str) else str(line))
        if raw is None:
            noise = _strip_ansi(line if isinstance(line, str) else str(line))
            noise = noise.rstrip("\r\n")
            if noise.strip():
                events.append({"type": "log", "text": noise})
            continue
        events.extend(state.feed(raw))
    return events


def summarize(events: list[dict]) -> dict:
    """Fold normalized events into the shape the API returns.

    Same keys as core/code_stream.summarize() so a caller can swap the two:
    reply, files, commits, tools, tokens, cost, session_id, error.
    """
    reply_parts: list[str] = []
    files: list[dict] = []
    tools: list[dict] = []
    session_id = None
    tokens: dict = {}
    cost = 0.0
    error = None
    end_reply = ""

    for event in events:
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        session_id = event.get("session_id") or session_id

        if kind == "chunk":
            text = event.get("text")
            if text:
                reply_parts.append(text)

        elif kind == "file":
            path = str(event.get("path") or "")
            if not path:
                continue
            entry = {
                "tool": event.get("tool") or "Write",
                "status": "completed",
                "title": path[:200],
                "op": event.get("op") or "edit",
                "path": path,
            }
            if not any(f["path"] == path for f in files):
                files.append(entry)
            if not any(t.get("path") == path for t in tools):
                tools.append(entry)

        elif kind == "tool":
            tools.append({
                "tool": event.get("tool") or "?",
                "status": event.get("status") or "unknown",
                "title": (event.get("title") or "")[:200],
            })

        elif kind == "end":
            end_reply = event.get("reply") or end_reply
            if event.get("tokens"):
                tokens = event["tokens"]
            if event.get("cost"):
                cost = event["cost"]
            error = event.get("error")

    return {
        "reply": ("".join(reply_parts).strip() or (end_reply or "").strip()),
        "files": files,
        "commits": [],
        "tools": tools,
        "tokens": tokens,
        "cost": cost,
        "session_id": session_id,
        "error": error,
    }


def stream_openclaude(cmd, workdir, env, lock, on_session=None, on_event=None):
    """Generator — runs openclaude, yielding SSE strings as events arrive.

    Same contract as core/code_stream.stream_opencode(): line-buffered
    reading, idle and hard ceilings, the child reaped on every exit path, and
    a terminal `end` frame whether the run finished or was cut short.

    Args:
        cmd:         argv from core.openclaude.build_command()
        workdir:     cwd for the child process
        env:         environment from core.openclaude.env()
        lock:        threading.Lock serialising concurrent runs
        on_session:  called with the session id as soon as it is known
        on_event:    called with every parsed raw event, for live UI updates

    Yields:
        SSE-formatted strings (already ending in \\n\\n)
    """
    proc = None
    timed_out = None
    events: list[dict] = []
    t0 = time.time()
    state = Normalizer()

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
                # Checked before every poll: readline() blocks forever once
                # openclaude goes quiet, which would pin this thread, the
                # lock and the child along with it.
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
                    # with no fileno): fall back to the blocking read.
                    ready = True
                if not ready:
                    continue
                line = proc.stdout.readline()
                if line == "" and proc.poll() is not None:
                    break  # EOF: the child closed its output and has exited
                last_output = time.time()

                raw = parse_event(line)
                if raw is None:
                    # Not JSON: a spinner fragment, a warning, a stack trace.
                    # Surface it rather than dropping it silently.
                    noise = _strip_ansi(line).rstrip("\r\n")
                    if noise.strip():
                        event = {"type": "log", "text": noise}
                        events.append(event)
                        yield sse_event(event)
                    continue

                if raw.get("session_id") and on_session:
                    on_session(raw["session_id"])
                for event in state.feed(raw):
                    events.append(event)
                    if event.get("type") == "end":
                        # The terminal frame is built from summarize() below,
                        # so a partial `result` never closes the stream early.
                        continue
                    yield sse_event(event)
                if on_event:
                    on_event(raw)

            if timed_out is None:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except GeneratorExit:
            # The browser hung up (user navigated away, or pressed stop).
            # openclaude is mid-edit here, so it must not be left running.
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
        yield sse_event({
            "type": "error",
            "error": f"openclaude idle for {STREAM_IDLE_TIMEOUT}s with no output",
            "reason": "idle_timeout",
        })
    elif timed_out == "hard_timeout":
        yield sse_event({
            "type": "error",
            "error": f"openclaude exceeded the {STREAM_HARD_TIMEOUT}s run limit",
            "reason": "hard_timeout",
        })
    elif summary["error"]:
        # is_error from `result` (auth failure, bad flag): say why before the
        # end frame, so the UI does not file a failed run as a normal reply.
        yield sse_event({"type": "error", "error": summary["error"]})
    elif not summary["reply"] and not summary["files"] and proc and proc.returncode:
        yield sse_event({
            "type": "error",
            "error": f"openclaude exited {proc.returncode} with no output",
        })

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
