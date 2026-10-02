"""Run one project .py file for agent mode (Session 23, Q1).

The shell tool no longer allows interpreters, so this is the only way the
agent executes Python — and it only runs files that live inside this project:

  - path must resolve inside PROJECT_ROOT and end in .py
  - timeout + output cap (same limits as the shell tool)
  - network access is blocked (in-process guard: socket, ctypes/ssl imports,
    shell-out via os.system/os.popen, and known network binaries)
  - child gets a scrubbed env and runs with cwd = project root

Limitation (documented): the guard is in-process, not an OS sandbox — a
script that finds an unpatched escape could still reach the network. It is
a defence-in-depth layer for model-written scripts, not a hostile-code jail.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

TIMEOUT_SEC = 10
MAX_TIMEOUT = 60
MAX_OUTPUT = 4096
ALLOWED_SUFFIX = ".py"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_BOOTSTRAP = r'''
import os
import runpy
import sys

path, root = sys.argv[1], sys.argv[2]
sys.argv = [path]
sys.path[:0] = [os.path.dirname(path), root]


def _blocked(*a, **k):
    raise RuntimeError("network access is disabled for project scripts")


# --- socket: every entry point a script could use ---
import socket as _sock_mod
import _socket as _sock_c
for _m in (_sock_mod, _sock_c):
    for _n in ("socket", "create_connection", "getaddrinfo", "socketpair",
               "gethostbyname", "gethostbyname_ex"):
        if hasattr(_m, _n):
            setattr(_m, _n, _blocked)

# --- process spawn: no shell strings, no network binaries ---
_BAN = {"curl", "wget", "nc", "ncat", "netcat", "telnet", "ssh", "scp",
        "ftp", "tftp", "ping", "traceroute", "nmap", "lynx", "links"}


def _check_args(args, kwargs):
    if kwargs.get("shell") or isinstance(args, str):
        raise RuntimeError("shell commands are disabled for project scripts")
    cmd = args[0] if args else None
    toks = list(cmd) if isinstance(cmd, (list, tuple)) else [cmd]
    toks += list(args[1:])
    for tok in toks:
        if os.path.basename(str(tok)) in _BAN:
            raise RuntimeError("network access is disabled for project scripts")


import subprocess as _sp
for _fn in ("Popen", "run", "call", "check_call", "check_output"):
    _orig = getattr(_sp, _fn)

    def _wrap(*a, _orig=_orig, **k):
        _check_args(a, k)
        return _orig(*a, **k)

    setattr(_sp, _fn, _wrap)


def _no_shell(*a, **k):
    raise RuntimeError("shell commands are disabled for project scripts")


import os as _os
_os.system = _no_shell
_os.popen = _no_shell

# ctypes / ssl က ဆိုက်ကိုယ်တိုင် ခေါ်နိုင်လို့ import ပိတ်
for _name in ("ctypes", "cffi", "_ssl", "ssl"):
    sys.modules[_name] = None

runpy.run_path(path, run_name="__main__")
'''


def _child_env() -> dict:
    """Secret / proxy မပါတဲ့ minimal env။"""
    return {
        "PATH": os.environ.get("PATH") or "/usr/bin:/bin",
        "HOME": os.environ.get("HOME") or str(Path.home()),
        "LANG": os.environ.get("LANG") or "C.UTF-8",
        "TERM": "dumb",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


def _resolve(arg, root: Path):
    """Path ကို project root ထဲ ရှိ/`.py` ဖြစ်အောင် စစ်။ error string / resolved path"""
    p = Path(str(arg)).expanduser()
    if not p.is_absolute():
        p = root / p
    try:
        p = p.resolve()
    except OSError as e:
        return f"Error: cannot resolve path: {e}"
    if not p.is_relative_to(root):
        return f"Error: outside the project folder: {p}"
    if p.suffix != ALLOWED_SUFFIX:
        return f"Error: only {ALLOWED_SUFFIX} files can be run: {p.name}"
    if not p.is_file():
        return f"Error: file not found: {p}"
    return p


def _clamp_timeout(timeout) -> int:
    """Timeout ကို 1..MAX_TIMEOUT အတွင်း ပိတ်။ non-int ဆို ValueError။"""
    try:
        value = int(timeout)
    except (TypeError, ValueError):
        raise ValueError("timeout must be an integer") from None
    return max(1, min(value, MAX_TIMEOUT))


def run_python(path: str, timeout: int = TIMEOUT_SEC) -> str:
    """Run a Python file that lives inside this project (network blocked).

    Args:
        path: The .py file to run. Relative paths are resolved from the
              project root. Files outside the project are refused.
        timeout: Seconds before the run is killed (default 10, max 60).
    """
    root = PROJECT_ROOT
    target = _resolve(path, root)
    if isinstance(target, str):
        return target

    try:
        timeout = _clamp_timeout(timeout)
    except ValueError as e:
        return f"Error: {e}"

    argv = [sys.executable, "-c", _BOOTSTRAP, str(target), str(root)]
    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(root),
            env=_child_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError as e:
        return f"Error: {type(e).__name__}: {e}"

    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            proc.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        return f"Error: command timed out ({timeout}s)"

    text = ((out or "") + (err or "")).strip()
    if not text:
        return f"(no output, exit code {proc.returncode})"
    if len(text) > MAX_OUTPUT:
        text = text[:MAX_OUTPUT] + f"\n... (truncated, {len(text)} total chars)"
    return text
