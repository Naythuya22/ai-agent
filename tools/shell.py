"""Safe read-only shell command tool for agent mode.

Session 23 (Q1) — the allowlist is the only thing that decides what runs, and
it runs WITHOUT a shell:

  - python / python3 / node / awk / sed are off the list (each one is
    arbitrary code execution); Python goes through tools.python_runner.
    (Release 26: a bare `python3 --version` probe is the single
    exception — it runs no code.)
  - Release 26: commands run with cwd pinned to the file-tools jail
    root, and path arguments resolving outside the allowed roots
    (jail root, $PREFIX, temp dir, PATH dirs) are refused up front —
    see the shell jail gate below.
  - segments are split into argv and executed with shell=False, so
    ; && || newlines $( ` > and friends never reach an interpreter.
  - the blocklist stays as a second line of defence.
  - the child gets a scrubbed env, so `env` / `printenv` cannot leak
    server secrets.
"""
from __future__ import annotations

import os
import shlex
import signal
import subprocess
import threading
from pathlib import Path

# ─── လုံခြုံရေး — whitelist (interpreter မပါ) ───
ALLOWED_BASE = {
    # အုပ်စု A — ဖတ်
    "ls", "cat", "head", "tail", "wc", "find", "stat", "du", "df",
    "file", "pwd", "tree", "readlink", "realpath", "basename", "dirname",
    # အုပ်စု B — စိတ်ဖြာ (awk/sed ဖယ်ထားပြီး)
    "grep", "sort", "uniq", "cut", "tr", "rev", "echo", "diff", "comm",
    # အုပ်စု C — System (env က scrubbed env ပဲ မြင်ရ)
    "ps", "top", "free", "uptime", "env", "which", "whoami",
    "uname", "date", "printenv", "id", "groups", "sleep",
    # အုပ်စု D — subcommand ကန့်သတ်ထားသူမျှသာ
    "git", "pip", "pip3",
}
# ဖယ်ထားသော interpreter/executors: python, python3, node, awk, sed

GIT_ALLOWED_SUB = {
    "log", "status", "diff", "branch", "show", "remote",
    "config", "rev-parse", "ls-files", "describe", "tag",
}

# git config က write ခွင့်ပေးလိုက်ရင် (core.pager / sshCommand / alias)
# နောက်ပိုင်း git log က command ပြန် run လုပ်နိုင်လို့ read-only ပုံစံပဲ ခွင့်ပြု။
GIT_CONFIG_RO_FLAGS = {"--get", "--get-all", "--get-regexp", "--get-urlmatch",
                       "--list", "-l"}

PIP_ALLOWED_SUB = {"list", "show", "freeze"}

# ─── အန္တရာယ် pattern (allowlist ရှုံးရင် ဒုတိယခံ) ───
DANGEROUS_PATTERNS = [
    "rm -rf", "rm -fr", "rm -r", "/dev/", "/sys/", "/proc/",
    "| sh", "| bash", "| zsh", "|sh", "|bash",
    "; rm ", ";rm ", "&& rm ", "|| rm ",
    "sudo ", "su -", "doas ",
    "curl ", "wget ", "nc -", "telnet ", "ssh ",
    "chmod ", "chown ", "chgrp ",
    "mkfs", "dd if=", "dd of=",
    "shutdown", "reboot", "poweroff",
    "kill ", "pkill ", "killall ",
    ":(){",
    "bash", "sh -", "zsh",
]

SUBST_PATTERNS = ["`", "$(", "${"]

TIMEOUT_SEC = 10
MAX_OUTPUT = 4096


def _split_pipeline(cmd: str) -> list:
    """| ပဲ pipeline အဖြစ် ခွဲ။ ; && || newline က validation မှာ ပိတ်ထားပြီး။"""
    parts = cmd.split("|")
    return [p.strip() for p in parts]


def _first_word(seg: str) -> str:
    """Segment ရဲ့ ပထမဆုံး စကားလုံး (VAR=x ကျော်)။"""
    try:
        toks = shlex.split(seg)
    except ValueError:
        return ""
    i = 0
    while i < len(toks) and "=" in toks[i] and not toks[i].startswith("-"):
        i += 1
    return toks[i] if i < len(toks) else ""


def _subcommand(toks: list, skip: set) -> str:
    """Flags နဲ့ flag value တွေ ကျော်ပြီး subcommand ထုတ်။"""
    i = 1
    while i < len(toks):
        t = toks[i]
        if t in skip and i + 1 < len(toks):
            i += 2
            continue
        if t.startswith("-"):
            i += 1
            continue
        return t
    return ""


def _validate(cmd: str):
    """မှားရင် error string ပြန်၊ မှန်ရင် None။"""
    if not cmd or not cmd.strip():
        return "Error: empty command"

    # shell=False ဆိုပေမဲ့ shell syntax ပါရင် အဓိပ္ပာယ်မဲ့/အန္တရာယ်ရှိလို့ ပိတ်။
    if "\n" in cmd or "\r" in cmd:
        return "Error: newline ခွင့်မပြု"
    for sep in ("&&", "||", ";"):
        if sep in cmd:
            return f"Error: separator ခွင့်မပြု: {sep}"

    low = cmd.lower()
    for pat in DANGEROUS_PATTERNS:
        if pat in low:
            return f"Error: blocked pattern: {pat}"

    for pat in SUBST_PATTERNS:
        if pat in cmd:
            return "Error: command substitution ခွင့်မပြု"

    if ">" in cmd or "<" in cmd:
        return "Error: redirect ခွင့်မပြု"

    raw_segs = _split_pipeline(cmd)
    segments = [s for s in raw_segs if s]
    if not segments:
        return "Error: no command"
    if len(segments) != len(raw_segs):
        return "Error: empty pipeline segment"

    for seg in segments:
        first = _first_word(seg)
        if not first:
            return f"Error: parse မရ: {seg[:40]}"
        if first in ("python", "python3"):
            # Release 26 narrow carve-out: a bare version probe
            # (`python3 --version` / `-V`, nothing else) is not code
            # execution — it prints one line and exits — and health
            # checks ask for it. Every other interpreter use stays
            # refused; real Python still goes through run_python.
            try:
                _toks = shlex.split(seg)
            except ValueError:
                return f"Error: bad syntax: {seg[:40]}"
            _i = 0
            while _i < len(_toks) and "=" in _toks[_i] \
                    and not _toks[_i].startswith("-"):
                _i += 1
            if _toks[_i + 1:] not in (["--version"], ["-V"]):
                return f"Error: command ခွင့်မပြု: {first}"
        elif first not in ALLOWED_BASE:
            return f"Error: command ခွင့်မပြု: {first}"

        try:
            toks = shlex.split(seg)
        except ValueError:
            return f"Error: bad syntax: {seg[:40]}"

        if first == "git":
            sub = _subcommand(toks, {"-C", "--git-dir", "--work-tree",
                                     "--work-dir", "-c"})
            if sub not in GIT_ALLOWED_SUB:
                return f"Error: git subcommand ခွင့်မပြု: {sub or '(none)'}"
            if sub == "config" and not any(t in GIT_CONFIG_RO_FLAGS for t in toks):
                return "Error: git config write ခွင့်မပြု (read-only ပဲ)"

        if first in ("pip", "pip3"):
            sub = _subcommand(toks, {"--index-url", "--target", "-t",
                                     "--python", "-p"})
            if sub not in PIP_ALLOWED_SUB:
                return f"Error: pip subcommand ခွင့်မပြု: {sub or '(none)'}"

    return None


def _cwd() -> str:
    home_repo = Path.home() / "ai-agent"
    if home_repo.is_dir():
        return str(home_repo)
    # ဒီ machine မှာ repo က home ထဲ မဟုတ်ရင် package root ပဲ။
    return str(Path(__file__).resolve().parent.parent)


# ─── Release 26: shell jail gate ────────────────────────────────────
#
# Release 16 jailed the *file* tools; run_command still executed with
# the legacy cwd and happily read /etc/hostname (tester evaluation
# #4). Full shell confinement is impossible — an allowlisted binary
# can be argued into many shapes — so this is the same pragmatic gate
# as R16, applied to the shell: the child runs with cwd pinned to the
# current file-tools jail root, and any path-shaped argument that
# resolves outside the allowed roots is refused BEFORE anything
# executes. Allowed absolute roots: the jail root itself, $PREFIX
# (Termux's usr — interpreters and binaries live there), the system
# temp dir, and the directories on PATH (standard binary locations;
# bare command names were always fine). Everything else absolute —
# /etc, /storage, other apps' private dirs, ~ expanded outside the
# jail — is refused. Relative paths resolve inside the jail root via
# the cwd pin, so `../` escapes are caught by the same check. The
# gate follows the file jail's switch: AI_AGENT_FILE_JAIL=off turns
# both off, and the legacy cwd comes back with it.

def _jail():
    """(enabled, root) for the shell gate, mirroring tools.file's jail.

    Read per call through tools.file's own helpers — the root is the
    active Code-mode project when one is selected, exactly like the
    file tools — and any failure degrades to the pre-Release-26
    behaviour rather than breaking the tool.
    """
    try:
        from tools.file import _jail_enabled, _jail_root
        return bool(_jail_enabled()), str(_jail_root())
    except Exception:
        return False, ""


def _allowed_roots(jail_root: str) -> list:
    roots = [jail_root]
    prefix = os.environ.get("PREFIX")
    if prefix:
        roots.append(prefix)
    import tempfile
    roots.append(tempfile.gettempdir())
    for d in (os.environ.get("PATH") or "").split(os.pathsep):
        if d:
            roots.append(d)
    return roots


def _path_candidate(tok: str) -> str:
    """The path spelling inside one argv token, or "" when the token
    names no path (a bare word, a flag, a NAME=value with a non-path
    value, a number…)."""
    t = tok
    if "=" in t:
        _, _, val = t.partition("=")
        if val.startswith(("/", "~", "$")) or "/" in val:
            t = val
        elif t.startswith("--"):
            t = val
        else:
            return ""
    if t.startswith(("/", "~", "$")) or "/" in t:
        return t
    return ""


def _jail_gate_error(segments: list) -> str:
    """BLOCKED error for the first out-of-roots path token, else "".

    ``segments`` is the tokenised pipeline (argv lists). Runs only
    when the file jail is enabled.
    """
    enabled, root = _jail()
    if not enabled or not root:
        return ""
    from tools._fs_guard import is_safe_path
    roots = _allowed_roots(root)
    for toks in segments:
        for tok in toks:
            cand = _path_candidate(tok)
            if not cand:
                continue
            exp = os.path.expanduser(os.path.expandvars(cand))
            if exp.startswith("$"):
                return ("Error: BLOCKED: path outside the allowed "
                        f"roots: {cand} (unresolvable variable)")
            p = Path(exp)
            if not p.is_absolute() and ".." in p.parts:
                # A traversal is judged against the jail root alone —
                # the same rule tools.file._guard applies. Letting the
                # wider roots judge it would wave `../` through
                # whenever the jail happens to sit under the temp dir.
                if not is_safe_path(str(Path(root) / p), root):
                    return ("Error: BLOCKED: path outside the allowed "
                            f"roots: {cand}")
                continue
            target = str(p if p.is_absolute() else Path(root) / p)
            if not any(is_safe_path(target, r) for r in roots):
                return ("Error: BLOCKED: path outside the allowed "
                        f"roots: {cand}")
    return ""


def _run_cwd() -> str:
    """Where commands execute: the jail root while the jail is on
    (relative paths then resolve inside it), the legacy cwd when the
    jail is off or the root is unusable."""
    enabled, root = _jail()
    if enabled and root and Path(root).is_dir():
        return root
    return _cwd()


def _child_env() -> dict:
    """Secret မပါတဲ့ minimal env — `env` command က ဒါပဲ မြင်ရမယ်။"""
    return {
        "PATH": os.environ.get("PATH") or "/usr/bin:/bin",
        "HOME": os.environ.get("HOME") or str(Path.home()),
        "LANG": os.environ.get("LANG") or "C.UTF-8",
        "TERM": "dumb",
    }


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


def _execute(segs: list, timeout: float) -> tuple[str, int] | str:
    """argv segments ကို pipe ချိုးပြီး shell=False နဲ့ run။ (stdout, rc) / error string."""
    procs = []
    err_bufs = []
    timer = None
    timed_out = threading.Event()

    def _drain(stream, buf):
        try:
            for line in stream:
                buf.append(line)
        except Exception:
            pass

    def _on_timeout():
        timed_out.set()
        for p in procs:
            _kill_tree(p)

    try:
        prev = None
        for argv in segs:
            p = subprocess.Popen(
                argv,
                cwd=_run_cwd(),
                env=_child_env(),
                stdin=prev if prev is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            if prev is not None:
                prev.close()
            procs.append(p)
            buf = []
            err_bufs.append(buf)
            threading.Thread(target=_drain, args=(p.stderr, buf),
                             daemon=True).start()
            prev = p.stdout

        timer = threading.Timer(timeout, _on_timeout)
        timer.daemon = True
        timer.start()

        out = procs[-1].stdout.read()
        timer.cancel()
        timed = timed_out.is_set()
        for p in procs:
            try:
                p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                _kill_tree(p)
                try:
                    p.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
        if timed:
            return f"Error: command timed out ({timeout:g}s)"
    except FileNotFoundError as e:
        return f"Error: {e.filename or 'command'}: not found"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
    finally:
        if timer is not None:
            timer.cancel()
        for p in procs:
            for stream in (p.stdout, p.stderr):
                if stream and not stream.closed:
                    try:
                        stream.close()
                    except OSError:
                        pass

    err = "".join("".join(b) for b in err_bufs)
    return out + err, procs[-1].returncode


def _shell_gate_error() -> str:
    """BLOCKED text when this install has shell commands switched off.

    Release 28: fresh installs start with run_command disabled (the
    beginner default stamped at first server boot); the user enables
    it in Settings. Installs from before the gate have no stored
    value and keep the historical behaviour — KeyManager resolves
    that. Any failure to read the config degrades to enabled, the
    pre-gate behaviour, rather than breaking the tool.
    """
    try:
        from core.key_manager import KeyManager
        if KeyManager().shell_enabled():
            return ""
    except Exception:
        return ""
    return ("Error: BLOCKED: shell commands are disabled on this "
            "install (beginner safety default). The user can enable "
            "them in Settings → System → Shell commands.")


def run_command(cmd: str, timeout: float = TIMEOUT_SEC) -> str:
    """Run a safe read-only Termux shell command (no shell interpreter).

    ခွင့်ပြု: ls, cat, head, tail, wc, find, stat, du, df,
             grep, sort, uniq, cut, echo, diff,
             ps, free, uptime, env, which, whoami, uname, date,
             git log/status/diff/branch/show/config(read-only),
             pip list/show/freeze

    ပိတ်: python (version probe ကလွဲ), node, awk, sed, rm, mv, chmod,
          sudo, curl, wget, ; && || newline, redirect, substitution,
          jail root အပြင်က absolute path (ဥပမာ /etc/...)

    Python script run ချင်ရင် tools.python_runner.run_python သုံးပါ။

    Args:
        cmd: Run မယ့် shell command။
        timeout: စောင့်မည့် စက္ကန့် (default 10)။
    """
    # Release 28: the enable gate runs before everything — a disabled
    # install refuses uniformly, whatever the command spells.
    gate_on = _shell_gate_error()
    if gate_on:
        return gate_on

    err = _validate(cmd)
    if err:
        return err

    segs = []
    for seg in _split_pipeline(cmd):
        if not seg:
            return "Error: empty pipeline segment"
        try:
            toks = shlex.split(seg)
        except ValueError as e:
            return f"Error: bad syntax: {e}"
        if not toks:
            return "Error: no command"
        segs.append(toks)

    # Release 26: the jail gate runs on the tokenised pipeline BEFORE
    # anything executes — a refused command has no side effects.
    gate = _jail_gate_error(segs)
    if gate:
        return gate

    result = _execute(segs, timeout)
    if isinstance(result, str):
        return result
    out, returncode = result

    out = (out or "").strip()
    if not out:
        return f"(no output, exit code {returncode})"
    if len(out) > MAX_OUTPUT:
        out = out[:MAX_OUTPUT] + f"\n... (truncated, {len(out)} total chars)"
    return out
