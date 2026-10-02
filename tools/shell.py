"""Safe read-only shell command tool for agent mode."""
from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

# ─── လုံခြုံရေး — ၄ အုပ်စု whitelist ───
ALLOWED_BASE = {
    # အုပ်စု A — ဖတ်
    "ls", "cat", "head", "tail", "wc", "find", "stat", "du", "df",
    "file", "pwd", "tree", "readlink", "realpath", "basename", "dirname",
    # အုပ်စု B — စိတ်ဖြာ
    "grep", "awk", "sed", "sort", "uniq", "cut", "tr", "rev", "echo",
    "diff", "comm",
    # အုပ်စု C — System
    "ps", "top", "free", "uptime", "env", "which", "whoami",
    "uname", "date", "printenv", "id", "groups",
    # အုပ်စု D — Execute
    "python", "python3", "node", "git", "pip", "pip3",
}

GIT_ALLOWED_SUB = {
    "log", "status", "diff", "branch", "show", "remote",
    "config", "rev-parse", "ls-files", "describe", "tag",
}

PIP_ALLOWED_SUB = {"list", "show", "freeze"}

PYTHON_BLOCKED_MOD = {"pip", "pip3"}

# ─── အန္တရာယ် pattern ───
DANGEROUS_PATTERNS = [
    "rm -rf", "rm -fr", "rm -r", "/dev/", "/sys/", "/proc/",
    "| sh", "| bash", "| zsh", "|sh", "|bash",
    "; rm ", ";rm ", "&& rm ", "|| rm ",
    "sudo ", "sudo", "su -", "doas ",
    "curl ", "wget ", "nc -", "telnet ", "ssh ",
    "chmod ", "chown ", "chgrp ",
    "mkfs", "dd if=", "dd of=",
    "shutdown", "reboot", "poweroff",
    "kill ", "pkill ", "killall ",
    ":(){",
]

SUBST_PATTERNS = ["`", "$(", "${"]

TIMEOUT_SEC = 10
MAX_OUTPUT = 4096


def _split_pipeline(cmd: str) -> list:
    """Split on |, ||, &&, ; — return list of segments."""
    parts = re.split(r"\s*(?:\|\||&&|;|\|)\s*", cmd)
    return [p.strip() for p in parts if p.strip()]


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


def _validate(cmd: str):
    """မှားရင် error string ပြန်၊ မှန်ရင် None။"""
    if not cmd or not cmd.strip():
        return "Error: empty command"

    low = cmd.lower()
    for pat in DANGEROUS_PATTERNS:
        if pat in low:
            return f"Error: blocked pattern: {pat}"

    for pat in SUBST_PATTERNS:
        if pat in cmd:
            return "Error: command substitution ခွင့်မပြု"

    # Redirect စစ် — 2>&1 သာ ခွင့်ပြု
    tmp = cmd.replace("2>&1", "")
    if ">" in tmp:
        return "Error: redirect ခွင့်မပြု"

    segments = _split_pipeline(cmd)
    if not segments:
        return "Error: no command"

    for seg in segments:
        first = _first_word(seg)
        if not first:
            return f"Error: parse မရ: {seg[:40]}"
        if first not in ALLOWED_BASE:
            return f"Error: command ခွင့်မပြု: {first}"

        try:
            toks = shlex.split(seg)
        except ValueError:
            return f"Error: bad syntax: {seg[:40]}"

        # sed -i ပိတ်
        if first == "sed" and "-i" in toks:
            return "Error: sed -i ခွင့်မပြု"

        # git subcommand စစ် (flag value ကျော်)
        if first == "git":
            skip_next = {"-C", "--git-dir", "--work-tree", "--work-dir", "-c"}
            sub = ""
            i = 1
            while i < len(toks):
                t = toks[i]
                if t in skip_next and i + 1 < len(toks):
                    i += 2
                    continue
                if t.startswith("-"):
                    i += 1
                    continue
                sub = t
                break
            if sub not in GIT_ALLOWED_SUB:
                return f"Error: git subcommand ခွင့်မပြု: {sub or '(none)'}"

        # pip subcommand စစ် (flag value ကျော်)
        if first in ("pip", "pip3"):
            skip_next = {"--index-url", "--target", "-t", "--python", "-p"}
            sub = ""
            i = 1
            while i < len(toks):
                t = toks[i]
                if t in skip_next and i + 1 < len(toks):
                    i += 2
                    continue
                if t.startswith("-"):
                    i += 1
                    continue
                sub = t
                break
            if sub not in PIP_ALLOWED_SUB:
                return f"Error: pip subcommand ခွင့်မပြု: {sub or '(none)'}"

        # python -m pip စစ်
        if first in ("python", "python3"):
            if "-m" in toks:
                midx = toks.index("-m")
                if midx + 1 < len(toks):
                    mod = toks[midx + 1]
                    if mod in PYTHON_BLOCKED_MOD:
                        return f"Error: python -m {mod} ခွင့်မပြု"

    return None


def run_command(cmd: str) -> str:
    """Run a safe read-only Termux shell command.

    ခွင့်ပြု: ls, cat, head, tail, wc, find, stat, du, df,
             grep, awk, sed, sort, uniq, cut, echo, diff,
             ps, free, uptime, env, which, whoami, uname, date,
             python, python3, git log/status/diff/branch/show,
             pip list/show/freeze

    ပိတ်: rm, mv, chmod, sudo, curl, wget, apt, kill

    Args:
        cmd: Run မယ့် shell command။
    """
    err = _validate(cmd)
    if err:
        return err

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=str(Path.home() / "ai-agent"),
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out ({TIMEOUT_SEC}s)"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

    out = (result.stdout or "") + (result.stderr or "")
    out = out.strip()
    if not out:
        return f"(no output, exit code {result.returncode})"
    if len(out) > MAX_OUTPUT:
        out = out[:MAX_OUTPUT] + f"\n... (truncated, {len(out)} total chars)"
    return out
