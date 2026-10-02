"""Code mode's opencode integration."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import shlex
from pathlib import Path

from core.log import get_logger

log = get_logger("opencode")

# OpenCode is installed inside the Ubuntu proot-distro environment.
PROOT_CMD = ["proot-distro", "login", "ubuntu", "--"]

# ttyd (web/code page) launches opencode from Termux, so it needs the plain
# name on PATH. Kept as a module attribute because routes reference oc.BINARY.
BINARY = "opencode"

# --- Provider Configuration ---
PROVIDER = "opencode"
BRIDGE_BASE = "http://127.0.0.1:8080/v1"
MODEL_ID = "mimo-v2.6-flash-free"
MODEL_REF = f"{PROVIDER}/{MODEL_ID}"


def config_file() -> Path:
    # Use /sdcard/ so both Termux and Ubuntu can access it
    return Path("/sdcard/.ai-agent") / "opencode.json"


def session_file() -> Path:
    return Path("/sdcard/.ai-agent") / "code_sessions.json"

DEFAULT_TIMEOUT = 300

INSTALL_HINT = (
    "opencode is not installed in the Ubuntu environment.\n"
    "  Run these commands in Termux:\n"
    "  proot-distro login ubuntu\n"
    "  apt update && apt install nodejs npm -y\n"
    "  npm install -g --allow-scripts=opencode-ai opencode-ai\n"
    "  Then check: proot-distro login ubuntu -- opencode --version"
)

_session_lock = threading.Lock()


# ── binary ────────────────────────────────────────────────────────────────

def is_installed() -> bool:
    if shutil.which("proot-distro") is None:
        return False
    try:
        r = subprocess.run(
            PROOT_CMD + ["bash", "-lc", "which opencode"],
            capture_output=True, text=True, timeout=5,
        )
        return r.returncode == 0 and bool(r.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


def version(timeout: int = 5) -> str | None:
    try:
        r = subprocess.run(
            PROOT_CMD + ["bash", "-lc", "opencode --version"],
            capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return (r.stdout or "").strip() or None


def missing_install_message() -> str:
    return INSTALL_HINT


# ── managed config ────────────────────────────────────────────────────────

def bridge_config(model_id: str | None = None, base: str | None = None) -> dict:
    # We are using OpenCode's native provider, so we return empty dict.
    # This prevents the app from looking for a local bridge.
    return {}


def write_bridge_config(
    model_id: str | None = None, base: str | None = None,
) -> Path:
    target = config_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    cfg_data = bridge_config(model_id, base)
    if not cfg_data:
        return target
    payload = json.dumps(cfg_data, indent=2) + "\n"
    try:
        if target.exists() and target.read_text() == payload:
            return target
    except OSError as e:
        # Unreadable old config is not fatal — fall through and overwrite it.
        log.debug("could not read the existing bridge config: %s: %s",
                  type(e).__name__, e)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(target)
    return target


# ── environment ───────────────────────────────────────────────────────────

def env(config_file: Path | None = None, **extra) -> dict:
    cfg = config_file if config_file is not None else write_bridge_config()
    out = os.environ.copy()
    out["OPENCODE_CONFIG"] = str(cfg)
    out["NO_COLOR"] = "1"
    out["TERM"] = "dumb"
    out["CI"] = "1"
    out["PYTHONUNBUFFERED"] = "1"
    out.update(extra)
    return out


# ── command ───────────────────────────────────────────────────────────────

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
    """Assemble the `opencode run` argv inside Ubuntu using bash -lc.

    `logs=True` adds --print-logs: opencode's own INFO lines then reach the
    stream as soon as they are written (~1s in, before the model answers), so
    the status bar is not frozen for the whole generation.
    """
    oc_args = ["opencode", "run", "--format", "json"]
    if logs:
        oc_args += ["--print-logs", "--log-level", "INFO"]
    if auto:
        oc_args.append("--auto")
    model = model or MODEL_REF
    if model:
        oc_args += ["--model", model]
    if agent:
        oc_args += ["--agent", agent]
    if session_id:
        oc_args += ["--session", session_id]
    for f in files or []:
        oc_args += ["--file", str(f)]
    oc_args += ["--", prompt]

    oc_cmd_str = " ".join(shlex.quote(arg) for arg in oc_args)
    cfg_path = config_file()

    # bash -lc loads ~/.bashrc (important for PATH inside proot).
    # PYTHONUNBUFFERED keeps opencode's stdout flowing line by line instead of
    # sitting in a block buffer — that is what made the stream look frozen.
    # shlex.quote on both paths: a home path with a space (or a quote) used to
    # produce a broken cd/export and a run that failed before it started.
    bash_script = (
        "export PYTHONUNBUFFERED=1 && "
        f"export OPENCODE_CONFIG={shlex.quote(str(cfg_path))} && "
        f"cd {shlex.quote(str(workdir))} && "
        f"{oc_cmd_str}"
    )

    return PROOT_CMD + ["bash", "-lc", bash_script]


def session_delete_command(session_id: str) -> list[str]:
    oc_args = ["opencode", "session", "delete", session_id]
    oc_cmd_str = " ".join(shlex.quote(arg) for arg in oc_args)
    return PROOT_CMD + ["bash", "-lc", oc_cmd_str]


_STALE_SESSION_RE = re.compile(r"session not found", re.I)


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
        # Dropping this means the next run starts with a stale session map —
        # the conversation still exists on opencode's side, but the bridge
        # will not know about it, so surface it.
        log.warning("could not persist opencode sessions to %s: %s: %s",
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
    cfg_path = config_file()
    return {
        "installed": is_installed(),
        "version": version(),
        "binary": f"{' '.join(PROOT_CMD)} bash -lc 'opencode'",
        "config_file": str(cfg_path),
        "config_exists": cfg_path.exists(),
        "bridge_base": BRIDGE_BASE,
        "model": MODEL_REF,
        "sessions": _read_sessions(),
        "hint": None if is_installed() else INSTALL_HINT,
    }
