"""Built-in Code-mode backend: the app's own agent engine, no external CLI.

The other backends (core/opencode.py, core/openclaude.py) drive an external
program the user must install and configure separately — on Termux that
means a Node CLI with its own provider keys and config files, and (for
opencode) a proot Ubuntu layer underneath. This backend removes the whole
layer: Code mode runs on core.agent_backend.AgentBackend, the same engine
Agent mode uses, so the provider keys and model choices in Settings are the
only setup there is, and the file tools' project jail (the Code UI's active
project) bounds where it can write.

The routes layer treats every backend the same way — it asks for an argv
and streams the child's stdout — so the engine runs as a child process
(`python -m core.builtin_runner`) that speaks opencode's `--format json`
event dialect. core.code_stream's generic path parses that dialect for any
registered backend that is not opencode/openclaude itself, so no route or
parser changes were needed to add this backend.

Sessions: the runner keeps the conversation transcript per session id in
the shared store; this module keeps the workdir -> session id map, in the
same shape core/opencode.py keeps its own.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

from core.store_paths import shared_store_dir
from core.log import get_logger

log = get_logger("builtin")

NAME = "builtin"

# The engine has no single model of its own: a run uses whatever provider
# and model Agent mode is configured with. Kept for surface parity with
# the other backends (doctor() reports it).
MODEL_REF = ""

INSTALL_HINT = (
    "The built-in code engine needs a cloud provider key: open Settings "
    "and add one (the same keys Agent mode uses). No separate install is "
    "required."
)


def session_file() -> Path:
    return shared_store_dir() / "code_sessions_builtin.json"


def config_file() -> Path:
    return shared_store_dir() / "builtin.json"


def write_bridge_config(
    model_id: str | None = None, base: str | None = None,
) -> Path:
    """Return the config path, writing the marker file on first use.

    The CLI backends use this hook to (re)write the config their external
    program reads. The built-in engine has no external program and no
    bridge — its tools run in-process — so there is nothing to configure;
    the file is only a marker that the doctor route's existence check can
    report truthfully.
    """
    target = config_file()
    if not target.exists():
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps({"backend": NAME, "bridge": "in-process"},
                           indent=2, sort_keys=True),
                encoding="utf-8")
        except OSError as e:
            log.warning("could not write builtin marker %s: %s: %s",
                        target, type(e).__name__, e)
    return target


def history_dir() -> Path:
    return shared_store_dir() / "code_builtin_history"


def is_installed() -> bool:
    """Always true: the engine ships inside the app.

    "Installed" here means "can be offered to the user". Whether a run can
    actually answer depends on a cloud route existing (a provider key in
    Settings); when none does, the runner says so as an error event rather
    than this probe hiding the backend entirely — a buyer with no key yet
    should still see Code mode and be told what it needs.
    """
    return True


def version(timeout: int = 5) -> str | None:
    return "built-in engine"


def missing_install_message() -> str:
    return INSTALL_HINT


def env(**extra) -> dict:
    out = os.environ.copy()
    out["NO_COLOR"] = "1"
    out["TERM"] = "dumb"
    out["CI"] = "1"
    out["PYTHONUNBUFFERED"] = "1"
    # The routes run the child with cwd = the user's project folder, where
    # the app's own packages are not importable unless the app root is on
    # PYTHONPATH. Without this the runner dies at `python -m
    # core.builtin_runner` before it can emit a single event.
    root = str(Path(__file__).resolve().parent.parent)
    existing = out.get("PYTHONPATH", "")
    out["PYTHONPATH"] = root + (os.pathsep + existing if existing else "")
    out.update(extra)
    # Same reason as core/openclaude.env(): a musl LD_PRELOAD exported by a
    # Termux wrapper must never reach the child.
    out.pop("LD_PRELOAD", None)
    return out


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
    """Argv for one run: this package's runner module as a child process.

    `model`, `files`, `agent`, `auto` and `logs` are accepted for signature
    parity with the CLI backends and ignored: the model comes from the
    Agent settings, and the runner's tool set already includes the file
    tools the extra arguments would describe.
    """
    argv = [
        sys.executable, "-m", "core.builtin_runner",
        "--workdir", str(workdir),
        "--prompt", str(prompt),
    ]
    if session_id:
        argv += ["--session-id", str(session_id)]
    return argv


def session_is_stale(output: str) -> bool:
    """The runner owns its history files, so an id it is handed is either
    resumable or simply starts a fresh transcript under the same id — there
    is no external session store to disagree with it."""
    return False


# ── session map (workdir -> session id), mirroring core/opencode.py ────

_session_lock = threading.Lock()


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
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True),
                       encoding="utf-8")
        tmp.replace(target)
    except OSError as e:
        log.warning("could not persist builtin sessions to %s: %s: %s",
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
        "binary": "built-in (in-app agent engine)",
        "config_file": str(cfg_path),
        "config_exists": cfg_path.exists(),
        "bridge_base": "",
        "model": MODEL_REF,
        "sessions": _read_sessions(),
        "hint": None,
    }
