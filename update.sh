#!/bin/sh
# Update ai-agent to the newest version from git (beta testers).
#
#   sh update.sh
#
# Fast-forward only: this never merges, never rebases and never
# touches uncommitted work. If the checkout has modified tracked
# files it stops and says so — an update must never eat a tester's
# own changes. Untracked files (models, caches) do not block it.
#
# After a successful update, a server already running FROM THIS
# checkout is restarted in place (same command, same port) so the
# new code takes effect without the tester doing it by hand. The
# restart is best-effort: discovery reads /proc and only ever touches
# a process whose working folder IS this checkout.
set -eu
cd "$(dirname "$0")"

if [ ! -d .git ]; then
    echo "error: this folder is not a git checkout, so update.sh cannot update it." >&2
    echo "       (Zip နဲ့ သွင်းထားတာဆို zip အသစ်ကို ထပ်ပြီး သွင်းပါ။)" >&2
    exit 1
fi

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "error: you have uncommitted changes — update stopped, nothing was changed." >&2
    echo "       Commit or stash them first (git status ကို ကြည့်ပါ), then run update.sh again." >&2
    exit 1
fi

old=$(git describe --tags 2>/dev/null || git rev-parse --short HEAD)

git pull --ff-only

new=$(git describe --tags 2>/dev/null || git rev-parse --short HEAD)

if [ "$old" = "$new" ]; then
    echo "Already up to date: $new"
    exit 0
fi

echo "Updated: $old -> $new"

# Restart a running server from this checkout, best effort. Python is
# guaranteed present (the app needs it). The helper never fails the
# update: worst case it prints how to restart by hand.
python3 - "$PWD" <<'PYEOF' || true
import os, signal, subprocess, sys, time
import urllib.error, urllib.request
from pathlib import Path

root = Path(sys.argv[1])

def server_pids(want):
    """Pids whose cmdline mentions server.py AND whose cwd resolves
    to this checkout. The cwd check is the safety gate — a server.py
    process living anywhere else is never touched. This script's own
    process tree (the shell + this python) is excluded as well."""
    found = []
    proc = Path("/proc")
    if not proc.is_dir():
        return found
    try:
        want = want.resolve()
    except Exception:
        pass
    mine = set()
    pid = os.getpid()
    while pid and pid > 1 and pid not in mine:
        mine.add(pid)
        try:
            stat = (proc / str(pid) / "stat").read_text()
            pid = int(stat.rsplit(")", 1)[1].split()[1])
        except Exception:
            break
    for ent in proc.iterdir():
        if not ent.name.isdigit():
            continue
        pid = int(ent.name)
        if pid in mine:
            continue
        try:
            if b"server.py" not in (ent / "cmdline").read_bytes():
                continue
            if (ent / "cwd").resolve() == want:
                found.append(pid)
        except Exception:
            continue
    return found

def proc_text(pid, name):
    try:
        return Path(f"/proc/{pid}/{name}").read_bytes()
    except Exception:
        return b""

def proc_argv(pid):
    return [p.decode("utf-8", "replace")
            for p in proc_text(pid, "cmdline").split(b"\0") if p]

def proc_env(pid):
    env = {}
    for kv in proc_text(pid, "environ").split(b"\0"):
        if b"=" in kv:
            k, v = kv.split(b"=", 1)
            env[k.decode("utf-8", "replace")] = v.decode("utf-8", "replace")
    return env

def parse_port(argv):
    for i, a in enumerate(argv):
        cand = None
        if a == "--port" and i + 1 < len(argv):
            cand = argv[i + 1]
        elif a.startswith("--port="):
            cand = a.split("=", 1)[1]
        if cand is not None:
            try:
                return int(cand)
            except ValueError:
                pass
    return 8080

def alive(pid):
    try:
        stat = (Path("/proc") / str(pid) / "stat").read_text()
        return stat.rsplit(")", 1)[1].split()[0] != "Z"
    except Exception:
        return False

pids = server_pids(root)
if not pids:
    print("Restart the server to use it:  python3 server.py   (server ကို ပိတ်ပြီး ပြန်ဖွင့်ပါ)")
    sys.exit(0)

first = pids[0]
argv = proc_argv(first)
env = proc_env(first)
home = env.get("HOME") or str(Path.home())
port = parse_port(argv)
ok = False
if argv:
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass
    deadline = time.time() + 5
    while time.time() < deadline and any(alive(p) for p in pids):
        time.sleep(0.2)
    for pid in pids:
        if alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
    time.sleep(0.3)
    try:
        logdir = Path(home) / ".ai-agent" / "logs"
        logdir.mkdir(parents=True, exist_ok=True)
        logf = open(logdir / "server-autorestart.log", "ab")
        subprocess.Popen(argv, cwd=str(root), env=env or None,
                         stdin=subprocess.DEVNULL, stdout=logf,
                         stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.time() + 15
        while time.time() < deadline and not ok:
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/", timeout=2) as r:
                    ok = bool(r.status)
            except urllib.error.HTTPError:
                ok = True  # any HTTP answer means the server is up
            except Exception:
                time.sleep(0.5)
    except Exception:
        ok = False

if ok:
    print(f"🔁 Server restarted — အသစ် ပြန်တက်ပြီ (port {port})။")
else:
    print("⚠️ Update အောင်မြင်ပေမယ့် server restart မအောင်မြင်ပါ — python3 server.py နဲ့ ကိုယ်တိုင် ပြန်စပါ။")
PYEOF
