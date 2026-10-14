"""Code mode's openclaude integration — the second Code-mode backend.

openclaude (a Claude Code compatible CLI, v0.32.0) speaks a different dialect
from opencode: it runs directly from Termux (no proot-distro), it is driven
with `--output-format stream-json` instead of `opencode run --format json`,
and it authenticates with ANTHROPIC_API_KEY rather than a managed
opencode.json bridge.

The public surface deliberately mirrors core/opencode.py — is_installed(),
version(), env(), build_command(), the session map, doctor() — so
routes/code.py can hold one module behind `_backend()` and call the same
names on either. Only the bodies differ.

Session state lives in its own file (code_sessions_openclaude.json) so the two
backends never resume each other's conversation ids.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
from pathlib import Path

from core.auth import configured_token
from core.log import get_logger

log = get_logger("openclaude")

# openclaude installs as a plain Termux binary on PATH — no proot wrapper, so
# the argv handed to subprocess is what actually runs. Kept as a module
# attribute because routes mirror how they read oc.PROOT_CMD.
PROOT_CMD: list[str] = []

BINARY = "openclaude"

# --- Provider Configuration ---
PROVIDER = "anthropic"
# No local bridge: openclaude talks to Anthropic itself, so there is no base
# URL for the UI to show. Empty string keeps the doctor payload's shape.
BRIDGE_BASE = ""
# openclaude takes Claude aliases/full names, not `provider/model`.
MODEL_ID = "sonnet"
MODEL_REF = MODEL_ID


def config_file() -> Path:
    # openclaude's own config dir (its backups live in .openclaude/backups).
    return Path.home() / ".openclaude" / ".openclaude.json"


def session_file() -> Path:
    # Separate store: opencode's code_sessions.json must stay untouched and
    # must never be handed a Claude session uuid.
    return Path("/sdcard/.ai-agent") / "code_sessions_openclaude.json"


DEFAULT_TIMEOUT = 300

INSTALL_HINT = (
    "openclaude is not installed.\n"
    "  Install it in Termux:\n"
    "  npm install -g @twigpine/openclaude\n"
    "  Then check: openclaude --version"
)

_session_lock = threading.Lock()


# ── binary ────────────────────────────────────────────────────────────────

def is_installed() -> bool:
    try:
        return shutil.which(BINARY) is not None
    except OSError:
        # A PATH that cannot be searched (odd $HOME, unreadable dir) answers
        # "not installed" instead of taking doctor() down with it.
        return False


def version(timeout: int = 5) -> str | None:
    try:
        r = subprocess.run(
            [BINARY, "--version"],
            capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return (r.stdout or "").strip() or None


def missing_install_message() -> str:
    return INSTALL_HINT


def tui_command(workdir: Path | str) -> list[str]:
    """argv for the ttyd browser terminal, already chdir'd to `workdir`.

    opencode takes the folder as a positional argument; openclaude's
    positional slot is the prompt, so the working directory comes from cwd
    alone and the argv stays bare.
    """
    return [BINARY]


# ── managed config ────────────────────────────────────────────────────────

def bridge_config(model_id: str | None = None, base: str | None = None) -> dict:
    """Managed config content — openclaude needs none (signature parity only).

    It authenticates through ANTHROPIC_API_KEY (see env()), so there is no
    bridge provider block to keep in sync. Always an empty dict: a caller
    that writes this would only ever write nothing.
    """
    return {}


def write_bridge_config(
    model_id: str | None = None, base: str | None = None,
) -> Path:
    """Return the config path without touching the user's own settings.

    core/opencode.write_bridge_config() rewrites opencode.json because that
    file is ours to manage. ~/.openclaude/.openclaude.json belongs to the CLI,
    so rewriting it here would clobber the user's login and preferences.
    """
    return config_file()


# ── environment ───────────────────────────────────────────────────────────

def anthropic_api_key() -> str | None:
    """ANTHROPIC_API_KEY for the child: env first, then the stored key ring."""
    key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if key:
        return key
    try:
        from core.key_manager import KeyManager
        entry = KeyManager().next_key("anthropic")
        if entry:
            stored = (entry.get("key") or "").strip()
            if stored:
                return stored
    except Exception as e:
        # A missing/broken agent_config.json is not fatal — openclaude then
        # runs unauthenticated and says so in its own stream.
        log.debug("no anthropic key available: %s: %s", type(e).__name__, e)
    return None


def env(config_file: Path | None = None, **extra) -> dict:
    out = os.environ.copy()
    out["NO_COLOR"] = "1"
    out["TERM"] = "dumb"
    out["CI"] = "1"
    out["PYTHONUNBUFFERED"] = "1"
    if config_file is not None:
        out["CLAUDE_CONFIG_DIR"] = str(Path(config_file).parent)
    key = anthropic_api_key()
    if key:
        out["ANTHROPIC_API_KEY"] = key
    tok = configured_token()
    if tok:
        out["AI_AGENT_TOKEN"] = tok
    out.update(extra)
    # Same reason as core/opencode.env(): a musl LD_PRELOAD exported by the
    # opencode wrapper must never reach this child either.
    out.pop("LD_PRELOAD", None)
    return out


# ── command ───────────────────────────────────────────────────────────────

# Tool name -> the edit operation the UI shows. opencode calls them
# write/edit/patch; Claude calls them Write/Edit/NotebookEdit.
_EDIT_TOOLS = {"Write": "create", "Edit": "edit", "NotebookEdit": "edit"}


def _add_dirs(files: list[str] | None) -> list[str]:
    """Paths for --add-dir: a folder is allowed as-is, a file's folder is."""
    dirs: list[str] = []
    for f in files or []:
        p = Path(str(f))
        d = p if p.is_dir() else p.parent
        text = str(d)
        if text and text not in dirs:
            dirs.append(text)
    return dirs


def build_command(
    prompt: str,
    workdir: Path | str,
    session_id: str | None = None,
    model: str | None = None,
    files: list[str] | None = None,
    agent: str | None = None,
    auto: bool = True,
    logs: bool = False,
) -> list[str]:
    """Assemble the `openclaude -p --output-format stream-json` argv.

    --verbose is not optional: with --print, stream-json output is rejected
    without it ("--output-format=stream-json requires --verbose"), so the run
    would die before producing a single event.

    `auto=True` maps to --permission-mode bypassPermissions, the flag Code
    mode's unattended runs need; `auto=False` leaves the CLI's own default.

    `logs` is accepted for signature parity with core.opencode.build_command
    and does nothing: --verbose above already surfaces the CLI's own output.

    The prompt is the final positional (openclaude has no `--` separator), so
    it is never mistaken for a flag.
    """
    argv = [
        BINARY, "-p",
        "--verbose",
        "--output-format", "stream-json",
        "--include-partial-messages",
    ]
    if auto:
        argv += ["--permission-mode", "bypassPermissions"]
    model = model or MODEL_REF
    if model:
        argv += ["--model", model]
    if session_id:
        argv += ["--session-id", session_id]
    if agent:
        argv += ["--agent", agent]
    for d in _add_dirs(files):
        argv += ["--add-dir", d]
    argv.append(str(prompt))
    return argv


def session_delete_command(session_id: str) -> list[str] | None:
    """openclaude has no `session delete` subcommand — nothing to run.

    Returning None (instead of an argv for a command that does not exist)
    tells the caller to forget the id locally and stop; the route checks for
    it before spawning anything.
    """
    return None


_STALE_SESSION_RE = re.compile(
    r"(session\s+(not\s+found|expired|id\b.*\binvalid)"
    r"|no\s+conversation\s+found"
    r"|invalid.{0,24}session)",
    re.I,
)


def session_is_stale(output: str) -> bool:
    return bool(_STALE_SESSION_RE.search(output or ""))


# ── session map ───────────────────────────────────────────────────────────

def _key(workdir: Path | str) -> str:
    path = Path(workdir).expanduser()
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _read_sessions() -> dict:
    try:
        data = json.loads(session_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_sessions(data: dict) -> None:
    target = session_file()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(target)
    except OSError as e:
        log.warning("could not persist openclaude sessions to %s: %s: %s",
                    target, type(e).__name__, e)


def get_session(workdir: Path | str) -> str | None:
    with _session_lock:
        return _read_sessions().get(_key(workdir))


def remember_session(workdir: Path | str, session_id: str) -> None:
    if not session_id:
        return
    with _session_lock:
        data = _read_sessions()
        data[_key(workdir)] = session_id
        _write_sessions(data)


def forget_session(workdir: Path | str) -> str | None:
    with _session_lock:
        data = _read_sessions()
        gone = data.pop(_key(workdir), None)
        if gone is not None:
            _write_sessions(data)
    return gone


def doctor() -> dict:
    """Health of the openclaude backend. Never raises, never hangs.

    The CLI's own `openclaude doctor` subcommand is deliberately *not* run
    here: it starts an Ink TUI, which dies on "Raw mode is not supported" and
    then spins on stdin forever when there is no terminal. Its name is
    reported instead so a human can run it where a TTY exists.
    """
    installed = is_installed()
    cfg_path = config_file()
    return {
        "installed": installed,
        "version": version(),
        "binary": BINARY,
        "backend": "openclaude",
        "config_file": str(cfg_path),
        "config_exists": cfg_path.exists(),
        "bridge_base": BRIDGE_BASE,
        "model": MODEL_REF,
        "sessions": _read_sessions(),
        "api_key": bool(anthropic_api_key()),
        "doctor_cmd": f"{BINARY} doctor",
        "hint": None if installed else INSTALL_HINT,
    }
