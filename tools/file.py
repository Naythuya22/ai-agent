"""File tools for agent mode.

The three tools here are path-jailed (tools/_fs_guard.py): every path
must resolve inside the jail root — the active Code-mode project when one
is selected, otherwise this repo's own root (config.PROJECT_ROOT).
Selecting a project in Code mode is how the user widens where the agent
may work; there is no separate UI for the jail.

The jail is on by default. AI_AGENT_FILE_JAIL=off
(config.FILE_JAIL_ENABLED) disables it entirely — a power-user /
recovery escape hatch, not a mode anyone should run in by default.
"""
from __future__ import annotations

import os
from pathlib import Path

import config
from tools._fs_guard import create_backup, is_safe_path

# ── read budget context (set by modes.agent before each read_file) ──
# Signature of read_file() must NOT change: it is exposed as a tool
# schema, and the model would try to fill model/base_url itself.
_read_ctx = {"model": "", "base_url": "", "used_tokens": 0}


# ── path jail (P0; always on since Release 16) ──────────────────
# Every path the file tools touch goes through _guard() first: a traversal
# segment is refused outright, and the path must resolve inside the jail
# root. The root is the active Code-mode project when one is selected —
# selecting a project is the user's standing choice of where the agent may
# work — and this repo's own root otherwise, so Agent Mode with no project
# selected is still confined. The check runs before stat/open/write, so an
# escape attempt never reaches the call it is aiming at.

def _active_project_root() -> str:
    """Active project folder, or '' when no project is selected.

    Read fresh on every call: the Code-mode dropdown can change the
    selection between two tool calls, and the store is a small JSON file.
    """
    try:
        from core import projects as projects_store
        proj = projects_store.get_active()
    except Exception:
        return ""
    if not isinstance(proj, dict):
        return ""
    return str(proj.get("path") or "")


def _jail_root() -> str:
    """Where the file tools may work: the active project, else the repo root.

    config.PROJECT_ROOT is read per call, not cached at import — the same
    reason _fs_guard._backup_dir re-reads config.CONFIG_DIR: a test (or a
    re-pointed harness) can move the root, and a stale import-time path
    would jail the tools into the wrong tree.
    """
    return _active_project_root() or str(config.PROJECT_ROOT)


def _jail_enabled() -> bool:
    """The AI_AGENT_FILE_JAIL opt-out, read per call like the root."""
    return bool(getattr(config, "FILE_JAIL_ENABLED", True))


def _guard(raw_path: str) -> str:
    """Error string for a path argument; '' means the path is allowed."""
    p = Path(raw_path).expanduser()
    if ".." in p.parts:
        return f"Error: unsafe path (path traversal): {raw_path}"
    if not _jail_enabled():
        return ""
    root = _jail_root()
    if not is_safe_path(str(p), root):
        return f"Error: unsafe path (outside allowed root {root}): {raw_path}"
    return ""


def set_read_context(*, model: str = "", base_url: str = "",
                     used_tokens: int = 0) -> None:
    """Called by AgentMode._execute_tool before dispatching read_file."""
    _read_ctx["model"] = model or ""
    _read_ctx["base_url"] = base_url or ""
    try:
        _read_ctx["used_tokens"] = int(used_tokens or 0)
    except (TypeError, ValueError):
        _read_ctx["used_tokens"] = 0


def _resolve_total(model: str, base_url: str) -> int:
    """Wrap MR.resolve_limit — tolerate int or dict, never raise."""
    try:
        from core import model_registry as MR
    except Exception:
        return 8_000
    try:
        r = MR.resolve_limit(model, base_url)
    except Exception:
        return 8_000
    if isinstance(r, int) and r > 0:
        return r
    if isinstance(r, dict):
        for k in ("limit", "total", "max", "context_window"):
            v = r.get(k)
            if isinstance(v, int) and v > 0:
                return v
    return 8_000


def _max_chars_for_read() -> int:
    """Model- and usage-aware char budget for one read_file call."""
    if not getattr(config, "AGENT_ADAPTIVE_READ", True):
        return getattr(config, "AGENT_READ_FLOOR", 4_000)
    total = _resolve_total(_read_ctx["model"], _read_ctx["base_url"])
    reserve = getattr(config, "AGENT_READ_RESERVE", 1_000)
    floor = getattr(config, "AGENT_READ_FLOOR", 4_000)
    ceiling = getattr(config, "AGENT_READ_CEILING", 100_000)
    available = total - _read_ctx["used_tokens"] - reserve
    if available <= 0:
        return floor
    token_budget = int(available * 0.25)
    chars = token_budget * 2          # Burmese-conservative
    return max(floor, min(chars, ceiling))


def list_files(directory: str) -> str:
    """List files in a directory.

    Args:
        directory: The directory path to list.
    """
    try:
        blocked = _guard(directory)
        if blocked:
            return blocked
        p = Path(directory).expanduser()
        if not p.exists():
            return f"Error: directory not found: {directory}"
        if not p.is_dir():
            return f"Error: not a directory: {directory}"
        entries = sorted(os.listdir(p))
        if not entries:
            return f"(empty directory: {directory})"
        # Cap at 100 entries
        shown = entries[:100]
        result = "\n".join(shown)
        if len(entries) > 100:
            result += f"\n... ({len(entries) - 100} more)"
        return result
    except PermissionError:
        return f"Error: permission denied: {directory}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def read_file(path: str, start_line: int = 1, end_line: int = 200) -> str:
    """Read contents of a text file (line-based, paginated).

    Args:
        path: Path to the file to read.
        start_line: First line to read (1-indexed). Default 1.
        end_line: Last line to read (inclusive). Default 200.
                  For longer files, call again with start_line set
                  to the "next" value shown in the footer.
    """
    MAX_CHARS = _max_chars_for_read()
    MAX_LINES = 500
    try:
        blocked = _guard(path)
        if blocked:
            return blocked
        p = Path(path).expanduser()
        if not p.exists():
            return f"Error: file not found: {path}"
        if not p.is_file():
            return f"Error: not a file: {path}"

        try:
            start_line = int(start_line)
            end_line = int(end_line)
        except (ValueError, TypeError):
            return "Error: start_line/end_line must be integers"
        if start_line < 1:
            start_line = 1
        if end_line < start_line:
            return f"Error: end_line ({end_line}) < start_line ({start_line})"
        if end_line - start_line > MAX_LINES:
            end_line = start_line + MAX_LINES

        text = p.read_text(encoding="utf-8", errors="replace")
        all_lines = text.splitlines()
        total = len(all_lines)

        if total == 0:
            return f"(empty file: {path})"
        if start_line > total:
            return f"Error: start_line {start_line} > total lines {total} in {path}"

        actual_end = min(end_line, total)
        sliced = all_lines[start_line - 1:actual_end]
        body = chr(10).join(sliced)

        char_capped = False
        if len(body) > MAX_CHARS:
            body = body[:MAX_CHARS]
            char_capped = True

        parts = [f"[{path}: lines {start_line}-{actual_end} of {total}]", body]
        if char_capped:
            next_line = start_line + body.count(chr(10))
            parts.append(
                f"\n--- OUTPUT TRUNCATED ---\n"
                f"Read {len(body)} of {sum(len(l)+1 for l in sliced)} chars "
                f"(budget {MAX_CHARS}).\n"
                f"To continue: read_file(path=\"{path}\", "
                f"start_line={next_line})"
            )
        elif actual_end < total:
            parts.append(
                f"... ({total - actual_end} more lines, "
                f"use start_line={actual_end + 1})"
            )
        return chr(10).join(parts)
    except PermissionError:
        return f"Error: permission denied: {path}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

def write_file(path: str, content: str) -> str:
    """Write text content to a file.

    Overwriting an existing file takes a backup first (tools/_fs_guard.py);
    the backup path comes back in the reply so the model — and the user —
    can point at it when an edit needs undoing. A backup that cannot be
    written stops the overwrite instead of passing silently.

    Args:
        path: Path to the file to write.
        content: Text content to write.
    """
    try:
        blocked = _guard(path)
        if blocked:
            return blocked
        p = Path(path).expanduser()
        note = ""
        if p.is_file():
            note = "\nBackup saved at: " + create_backup(str(p))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to {path}" + note
    except PermissionError:
        return f"Error: permission denied: {path}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


ALL_TOOLS = [list_files, read_file, write_file]
