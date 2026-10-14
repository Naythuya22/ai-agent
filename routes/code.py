"""Code mode — opencode runs, project registry, folder browsing.

opencode replaced aider here. It runs as a subprocess against the local
OpenAI-compatible bridge (`BRIDGE_BASE` below) via a managed opencode.json, so
edits land in real phone storage. Two things differ from the old aider setup:

  - opencode emits a JSON event stream (`--format json`), so files, tool calls
    and token usage are read off events instead of scraped from prose.
  - conversation history is opencode's own session, resumed per project folder
    via --session, rather than a history list pasted into each prompt.

Runs are serialised through `routes._state.code_lock`.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

from flask import (
    Blueprint,
    Response,
    has_request_context,
    jsonify,
    request,
    stream_with_context,
)

import config
from core import code_backends
from core import opencode as oc
from core import projects as projects_store
from core.code_stream import (
    OpencodeError,
    parse_event as cs_parse,
    run_code,
    run_opencode,
    sse_event,
    stream_code,
    stream_opencode,
    summarize,
    summarize_output,
)
from core.log import get_logger
from routes._helpers import json_body
from routes._state import code_lock

log = get_logger("routes.code")

bp = Blueprint("code", __name__)

_ttyd_proc = None

CODE_HISTORY_CAP = 20
BRIDGE_BASE = oc.BRIDGE_BASE
RUN_TIMEOUT = oc.DEFAULT_TIMEOUT
# The UI cannot render more than this, and a huge diff must not be shipped to
# a mobile client — /api/code/diff truncates its payload here.
DIFF_CAP = 200 * 1024

LANG_DIRECTIVE = (
    "LANGUAGE RULE (critical): Match the user's language in your REPLY text. "
    "If the user writes in Burmese (မြန်မာ), reply ONLY in Burmese. "
    "Keep code, file names, identifiers, and diffs in English. "
    "Only the natural-language explanation should match the user's language."
)

BURMESE_SUMMARY_DIRECTIVE = (
    "REPLY TAIL (required): end every reply with a section "
    "titled '## မြန်မာလို အကျဉ်း' explaining in plain Burmese: "
    "(1) what changed, file by file; (2) why; (3) warnings or "
    "next steps; (4) the test count. Keep code terms — function "
    "names, paths, git hashes, HTTP statuses — in English; do "
    "not translate them. The rest of the reply stays in English."
)


# ---------- shared helpers ----------

def ensure_git_identity():
    try:
        r = subprocess.run(
            ["git", "config", "--global", "user.email"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode != 0 or not r.stdout.strip():
            subprocess.run(
                ["git", "config", "--global", "user.email", "ai-agent@localhost"],
                capture_output=True, timeout=5,
            )
            subprocess.run(
                ["git", "config", "--global", "user.name", "ai-agent"],
                capture_output=True, timeout=5,
            )
    except Exception as e:
        # Without an identity every later `git commit` fails — the streamed
        # run would report success while storing nothing.
        log.warning("could not set git identity: %s: %s", type(e).__name__, e)


def build_code_prompt(task, code_history=None):
    """Prompt for opencode.

    No history block any more: opencode's session already holds the
    conversation, so replaying it here would duplicate context (and was the
    reason the old aider path truncated messages to 280 chars).
    """
    return f"{LANG_DIRECTIVE}\n\n{BURMESE_SUMMARY_DIRECTIVE}\n\nCurrent request:\nUser: {task}"


def commit_changes(workdir, files, message):
    """Commit the files opencode touched, so the UI's commit list still means
    something.

    aider auto-committed every run (AIDER_NO_AUTO_COMMITS=0); opencode leaves
    git alone. Same commit-after-each-change behaviour is kept here, scoped to
    only the paths the run reported, and skipped if the folder is not a repo.

    Always returns a dict: {"ok": True, "hash", "message"} on success, else
    {"ok": False, "reason"} — returning None on every failure path left the
    caller unable to say *why* no commit appeared.
    """
    paths = [f.get("path") for f in (files or []) if f.get("path")]
    if not paths:
        return {"ok": False, "reason": "no files to commit"}
    try:
        inside = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=str(workdir), capture_output=True, text=True, timeout=5,
        )
        if inside.returncode != 0 or inside.stdout.strip() != "true":
            return {"ok": False, "reason": "not a git repository"}
        added = subprocess.run(
            ["git", "add", "--"] + paths,
            cwd=str(workdir), capture_output=True, text=True, timeout=15,
        )
        if added.returncode != 0:
            return {"ok": False,
                    "reason": "git add failed: " + added.stderr[:200]}
        staged = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=str(workdir), capture_output=True, text=True, timeout=15,
        )
        if not staged.stdout.strip():
            return {"ok": False, "reason": "nothing staged"}
        first = message.strip().splitlines()[0] if message.strip() else "opencode"
        done = subprocess.run(
            ["git", "commit", "-m", first[:72]],
            cwd=str(workdir), capture_output=True, text=True, timeout=20,
        )
        if done.returncode != 0:
            return {"ok": False,
                    "reason": "git commit failed: " + done.stderr[:200]}
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(workdir), capture_output=True, text=True, timeout=5,
        )
        return {"ok": True, "hash": head.stdout.strip(),
                "message": first[:72]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "git timed out"}
    except Exception as e:
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}


_code_history = []  # [{role, text}, ...] — a UI/debug mirror, not the model's context


def _record_code_turn(task, reply):
    """Kept for the /history endpoint only; no longer feeds the prompt."""
    _code_history.append({"role": "user", "text": task})
    _code_history.append({"role": "assistant", "text": reply})
    if len(_code_history) > CODE_HISTORY_CAP:
        _code_history[:] = _code_history[-CODE_HISTORY_CAP:]


def _backend_name(project=None) -> str:
    """Which backend this request should run on.

    Precedence: ?backend=<name> on the request, then the active project's
    `backend` field, then AI_AGENT_CODE_BACKEND (config.CODE_BACKEND_DEFAULT).
    An unknown name anywhere in that chain degrades to the default instead of
    failing the request — a stale project record must not wedge code mode.
    """
    name = ""
    if has_request_context():
        name = (request.args.get("backend") or "").strip()
    if not name:
        proj = project if project is not None else projects_store.get_active()
        name = str((proj or {}).get("backend") or "").strip()
    if not name:
        name = str(getattr(config, "CODE_BACKEND_DEFAULT", "") or "").strip()
    return code_backends.resolve(name)


def _backend(project=None):
    """The backend module for this request: core.opencode or core.openclaude.

    Callers use the same names on it either way (is_installed(), env(),
    build_command(), doctor(), the session map) — that shared surface is what
    makes the two backends interchangeable here.
    """
    return code_backends.get(_backend_name(project)) or oc


def _require_backend(project=None):
    """None when the active backend is usable, else a (message, status) failure."""
    b = _backend(project)
    if b.is_installed():
        return None
    return b.missing_install_message(), 500


# The old name, kept: callers and tests that ask for "the" backend (opencode)
# still get the resolver above, which answers with opencode unless the project
# or the environment says otherwise.
_require_opencode = _require_backend


def _session_for(workdir):
    """Session id to resume for this folder, if a previous run left one.

    Each backend reads its own store, so an opencode session id is never
    handed to openclaude (and vice versa).
    """
    return _backend().get_session(workdir)


def _remember_session(workdir, session_id):
    if session_id:
        _backend().remember_session(workdir, session_id)


def _run(workdir, prompt, env=None, timeout=RUN_TIMEOUT):
    """`opencode run` to completion, retrying once on a stale --session id.

    opencode validates the id before it does anything else: an id it does not
    recognise exits 1 with a bare `Error: Session not found` on stderr and not
    one JSON event. remember_session() refuses to store an empty id, so the bad
    entry would survive and every later message would fail the same way — a
    wedged code mode with nothing on screen to explain it. So the run is
    retried as a fresh session, which also repairs the map.

    Returns (returncode, output_lines).
    """
    b = _backend()
    name = _backend_name()
    env = b.env() if env is None else env
    session_id = _session_for(workdir)
    rc, lines = 0, []
    for attempt in (1, 2):
        cmd = b.build_command(prompt, workdir, session_id=session_id)
        rc, lines = run_code(cmd, workdir, env, timeout=timeout, backend=name)
        if (rc != 0 and session_id and attempt == 1
                and b.session_is_stale("\n".join(lines))):
            b.forget_session(workdir)
            session_id = None
            continue
        return rc, lines
    return rc, lines


def _parse_line(line):
    """The dict inside an SSE frame, or None if it is not one."""
    if not line.startswith("data: "):
        return None
    try:
        event = json.loads(line[6:])
    except ValueError:
        return None
    return event if isinstance(event, dict) else None


def _stream(workdir, prompt, env):
    """Yield SSE frames for one streamed run, recovering from a stale session.

    The same recovery _run() does, but it has to happen inside the stream. A
    stale id comes back as a single log line with nothing in front of it — no
    chunk, no tool call, not even an end frame — which is the one place it can
    be told apart from a run that is merely quiet, and those frames are
    swallowed instead of shown as a dead end.
    """
    b = _backend()
    name = _backend_name()
    session_id = _session_for(workdir)
    for attempt in (1, 2):
        cmd = b.build_command(prompt, workdir, session_id=session_id,
                              logs=True)
        frames = stream_code(
            cmd,
            workdir=workdir,
            env=env,
            lock=code_lock,
            backend=name,
            on_session=lambda sid: _remember_session(workdir, sid),
        )
        produced = False
        try:
            for line in frames:
                event = _parse_line(line)
                if event is None:
                    yield line
                    continue
                kind = event.get("type")
                if kind in ("chunk", "file", "tool", "step", "end"):
                    produced = True
                elif (kind == "log" and not produced and session_id
                      and attempt == 1
                      and b.session_is_stale(event.get("text", ""))):
                    b.forget_session(workdir)
                    session_id = None
                    break
                yield line
            else:
                return
        finally:
            # close(), not just dropping the reference: it is what reaps a
            # child left running by the attempt we abandoned.
            frames.close()


def _resolve_workdir():
    """Active project path if set, else the default shared-storage folder."""
    active = projects_store.get_active()
    if active and active.get("path"):
        workdir = Path(active["path"])
        if not workdir.exists() or not workdir.is_dir():
            return None, f"project path missing: {workdir}"
    else:
        workdir = Path.home() / "storage" / "shared" / "opencode-work"
    return workdir, None


def _ensure_safe_directory(workdir):
    """Android fuse mounts trip git's ownership check — allow this dir."""
    try:
        subprocess.run(
            ["git", "config", "--global", "--add", "safe.directory", str(workdir)],
            capture_output=True, timeout=5,
        )
    except Exception as e:
        # git then refuses the Android fuse mount and commits silently fail.
        log.warning("could not allow safe.directory for %s: %s: %s",
                    workdir, type(e).__name__, e)


def _git(workdir, args, timeout=15):
    """Run `git <args>` in workdir. Returns (returncode, stdout, stderr).

    Nothing escapes: a missing git binary answers (127, "", "git not found"),
    a hung command (124, "", "timeout") — callers only ever read the tuple,
    so the endpoints can answer with a reason instead of a 500 traceback.
    """
    try:
        r = subprocess.run(["git"] + list(args), cwd=str(workdir),
                           capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except FileNotFoundError:
        return 127, "", "git not found"
    except Exception as e:
        return 1, "", f"{type(e).__name__}: {e}"


def _project_command_reply(task):
    """Handle the /project sub-commands. Returns a reply payload dict."""
    parts = task.split()
    sub = parts[1] if len(parts) > 1 else "help"
    arg = " ".join(parts[2:]).strip()

    if sub == "list":
        projects = projects_store.list_all()
        active = projects_store.get_active()
        if not projects:
            msg = "📁 No projects yet.\n\nAdd: /project add <path>\nExample: /project add ~/storage/shared/Download"
        else:
            lines = ["📁 Projects:"]
            for proj in projects:
                mark = " ✅" if active and active["id"] == proj["id"] else ""
                lines.append(f"  • {proj['name']} [{proj['id']}]{mark}")
                lines.append(f"      {proj['path']}")
            lines.append("")
            lines.append("Switch: /project use <id-or-name>")
            msg = "\n".join(lines)
        return {"reply": msg, "files": [], "commits": [],
                "raw": msg, "elapsed": 0.0, "rc": 0}

    if sub == "add":
        if not arg:
            return {"reply": "Usage: /project add <path>", "files": [], "commits": [], "raw": "", "elapsed": 0.0, "rc": 1}
        try:
            proj = projects_store.add("", arg)
        except ValueError as e:
            return {"reply": f"❌ {e}", "files": [], "commits": [], "raw": "", "elapsed": 0.0, "rc": 1}
        except Exception as e:
            return {"reply": f"❌ {type(e).__name__}: {e}", "files": [], "commits": [], "raw": "", "elapsed": 0.0, "rc": 1}
        msg = f"✅ Added project: **{proj['name']}** (id: {proj['id']})\n📁 {proj['path']}\n\nNow: /project use {proj['id']}"
        return {"reply": msg, "files": [], "commits": [], "raw": msg, "elapsed": 0.0, "rc": 0}

    if sub == "use":
        if not arg:
            return {"reply": "Usage: /project use <id-or-name>", "files": [], "commits": [], "raw": "", "elapsed": 0.0, "rc": 1}
        target = None
        for proj in projects_store.list_all():
            if arg == proj["id"] or arg.lower() == proj["name"].lower():
                target = proj
                break
        if not target:
            return {"reply": f"❌ Not found: {arg}\nRun: /project list", "files": [], "commits": [], "raw": "", "elapsed": 0.0, "rc": 1}
        projects_store.set_active(target["id"])
        msg = f"✅ Active project: **{target['name']}**\n📁 {target['path']}\n\nAll code tasks now run in this folder."
        return {"reply": msg, "files": [], "commits": [], "raw": msg, "elapsed": 0.0, "rc": 0}

    if sub == "remove":
        if not arg:
            return {"reply": "Usage: /project remove <id>", "files": [], "commits": [], "raw": "", "elapsed": 0.0, "rc": 1}
        ok = projects_store.remove(arg)
        msg = f"✅ Removed: {arg}" if ok else f"❌ Not found: {arg}"
        return {"reply": msg, "files": [], "commits": [], "raw": "", "elapsed": 0.0, "rc": 0}

    help_msg = (
        "📁 **Project commands**\n\n"
        "• /project list — show all\n"
        "• /project add <path> — register a folder\n"
        "• /project use <id|name> — switch active\n"
        "• /project remove <id> — delete\n\n"
        "Example:\n"
        "  /project add ~/storage/shared/Download\n"
        "  /project use Download"
    )
    return {"reply": help_msg, "files": [], "commits": [], "raw": help_msg, "elapsed": 0.0, "rc": 0}


def _model_reply(msg, rc=0):
    """The reply payload shape every chat sub-command answers with."""
    return {"reply": msg, "files": [], "commits": [],
            "raw": msg, "elapsed": 0.0, "rc": rc}


def _model_command_reply(task):
    """Handle the /model sub-commands. Returns a reply payload dict.

    /model            — every backend with its install state, and which one
                        this project is on
    /model <backend>  — point the active project at that backend
    /model status     — the backend in force, its version and the project
    """
    parts = task.split()
    sub = parts[1] if len(parts) > 1 else ""

    if not sub or sub == "list":
        current = _backend_name()
        lines = ["🧠 **Code backends**", ""]
        for name in code_backends.names():
            ok = code_backends.installed(name)
            lines.append(
                f"  • {name} {'✅ installed' if ok else '❌ not installed'}"
                + ("  ← active" if name == current else "")
            )
        lines.append("")
        lines.append("Switch: /model <backend>")
        lines.append("Detail: /model status")
        return _model_reply("\n".join(lines))

    if sub == "status":
        b = _backend()
        active = projects_store.get_active()
        return _model_reply(
            f"🧠 Backend: **{_backend_name()}**\n"
            f"Version: {b.version() or 'unknown'}\n"
            f"Installed: {'yes' if b.is_installed() else 'no'}\n"
            f"Project: {active.get('name') if active else '(none)'}"
        )

    # Anything else names the backend to switch to.
    name = sub
    if name not in code_backends.names():
        return _model_reply(
            f"❌ Unknown backend: {name}\n"
            f"Available: {', '.join(code_backends.names())}", rc=1)

    active = projects_store.get_active()
    if not active:
        return _model_reply(
            "❌ No active project to switch.\n"
            "Add one first: /project add <path>", rc=1)
    if not projects_store.set_backend(active["id"], name):
        return _model_reply(f"❌ Could not set the backend for "
                            f"{active.get('name') or active['id']}", rc=1)

    b = code_backends.get(name)
    msg = f"✅ Code backend: **{name}**\n📁 {active.get('path', '')}"
    if not b.is_installed():
        msg += "\n\n⚠️ Not installed:\n" + b.missing_install_message()
    return _model_reply(msg)


# ---------- /api/code (single-shot, fixed workdir) ----------

@bp.route("/api/code", methods=["POST"])
def api_code():
    """One-shot opencode run against an explicit workdir."""
    body = json_body()
    task = (body.get("task") or "").strip()
    workdir = (body.get("workdir") or str(Path.home() / "opencode-work")).strip()
    if not task:
        return jsonify({"error": "task required"}), 400

    missing = _require_opencode()
    if missing:
        return jsonify({"error": missing[0]}), missing[1]

    wd = Path(workdir).expanduser()
    if not wd.exists() or not wd.is_dir():
        return jsonify({"error": f"workdir not found: {wd}"}), 400

    ensure_git_identity()
    _ensure_safe_directory(wd)

    prompt = build_code_prompt(task)

    # This path previously bypassed the lock the streamed path takes
    # (stream_opencode takes code_lock for its whole run), and opencode writes
    # real files in the shared folder — so a one-shot run and a streamed run
    # could edit the same files at once and corrupt them.
    with code_lock:
        t0 = time.time()
        try:
            rc, lines = _run(wd, prompt)
        except OpencodeError as e:
            return jsonify({"error": str(e)}), 500

        summary = summarize_output(lines, backend=_backend_name())
        _remember_session(wd, summary["session_id"])
        # Only a clean run gets committed, same as the streaming path: a failed
        # run can leave a half-applied edit behind.
        made, commit_error = None, None
        if rc == 0:
            res = commit_changes(wd, summary["files"], task)
            if res["ok"]:
                made = {"hash": res["hash"], "message": res["message"]}
            elif summary["files"]:
                commit_error = res["reason"]

    return jsonify({
        "ok": rc == 0,
        "workdir": str(wd),
        "returncode": rc,
        "reply": summary["reply"],
        "files": summary["files"],
        "commits": [made] if made else [],
        "commit_error": commit_error,
        "tokens": summary["tokens"],
        "cost": summary["cost"],
        "session_id": summary["session_id"],
        "output": "\n".join(l.rstrip() for l in lines)[:4000],
        "elapsed": round(time.time() - t0, 2),
    })


@bp.route("/api/code/diff", methods=["GET"])
def api_code_diff():
    """Unified diff of what changed, for the UI's diff viewer.

    scope picks which git comparison to show: the working tree against the
    index (default), the index against HEAD, or everything against the last
    commit. commit=<hash> instead shows one commit's own changes (git show) —
    what the View diff button asks for, since the tree is clean right after a
    commit. Failures answer 200 with {ok: False, reason} — a folder that is
    not a repo is an ordinary state to ask about, not a server error.
    """
    scope = (request.args.get("scope") or "workdir").strip()
    if scope not in ("workdir", "staged", "head"):
        return jsonify({"error": "scope must be workdir|staged|head"}), 400

    # commit must be a bare hex hash: the regex is the only thing standing
    # between the query string and git, so no ref, path or option can pass.
    commit = (request.args.get("commit") or "").strip()
    if commit and not re.fullmatch(r"[0-9a-f]{7,40}", commit):
        return jsonify(
            {"error": "commit must be a 7-40 char lowercase hex"}), 400

    workdir, err = _resolve_workdir()
    if err:
        return jsonify({"error": err}), 500

    _ensure_safe_directory(workdir)

    if commit:
        args = ["show", "--format=", "--patch", commit]
        # --numstat, not --stat: the parser below reads tab-separated
        # insertions/deletions/path, which only numstat emits.
        args_stat = ["show", "--format=", "--numstat", commit]
    elif scope == "workdir":
        args = ["diff"]
        args_stat = args + ["--numstat"]
    elif scope == "staged":
        args = ["diff", "--cached"]
        args_stat = args + ["--numstat"]
    else:
        args = ["diff", "HEAD"]
        args_stat = args + ["--numstat"]

    with code_lock:
        rc_in, _, _ = _git(workdir, ["rev-parse", "--is-inside-work-tree"])
        if rc_in != 0:
            return jsonify({"ok": False, "reason": "not a git repository"})
        rc_d, out, err_d = _git(workdir, args)
        if rc_d != 0:
            if commit:
                return jsonify({"ok": False,
                                "reason": ("commit not found: "
                                           + commit)[:200]})
            reason = (err_d or "").strip()[:200] or "git diff failed"
            return jsonify({"ok": False, "reason": reason})
        rc_n, numstat, _ = _git(workdir, args_stat)

    files = insertions = deletions = 0
    if rc_n == 0:
        for line in numstat.splitlines():
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            try:
                ins = int(parts[0])
                dels = int(parts[1])
            except ValueError:
                continue
            files += 1
            insertions += ins
            deletions += dels

    return jsonify({
        "ok": True,
        "scope": ("commit:" + commit) if commit else scope,
        "stat": {"files": files, "insertions": insertions,
                 "deletions": deletions},
        "diff": out[:DIFF_CAP],
        "workdir": str(workdir),
    })


@bp.route("/api/code/rollback", methods=["POST"])
def api_code_rollback():
    """Undo the last N commits with `git reset --hard HEAD~N`.

    confirm must be the literal boolean true and count an int in 1..10, both
    checked before git runs: the route only ever builds HEAD~<count>, never a
    ref or path from the client, so a bad call cannot reset elsewhere.
    Failures answer 200 with {ok: False, reason} — a folder that is not a repo
    is an ordinary state to ask about, not a server error.
    """
    body = json_body()
    if body.get("confirm") is not True:
        return jsonify({"error": "confirm must be true"}), 400
    count = body.get("count", 1)
    if not isinstance(count, int) or isinstance(count, bool) or not (1 <= count <= 10):
        return jsonify({"error": "count must be an int in 1..10"}), 400

    workdir, err = _resolve_workdir()
    if err:
        return jsonify({"error": err}), 500

    _ensure_safe_directory(workdir)

    with code_lock:
        rc_in, _, _ = _git(workdir, ["rev-parse", "--is-inside-work-tree"])
        if rc_in != 0:
            return jsonify({"ok": False, "reason": "not a git repository"})

        rc_b, before, _ = _git(workdir, ["rev-parse", "--short", "HEAD"])
        if rc_b != 0:
            return jsonify({"ok": False, "reason": "no commits to roll back"})

        rc_r, _, err_r = _git(workdir,
                              ["reset", "--hard", "HEAD~%d" % count])
        if rc_r != 0:
            reason = (err_r or "").strip()[:200] or "git reset failed"
            if rc_r == 124:
                reason = "timeout"
            return jsonify({"ok": False, "reason": reason})

        rc_a, after, _ = _git(workdir, ["rev-parse", "--short", "HEAD"])

    return jsonify({
        "ok": True,
        "before": before.strip(),
        "after": after.strip(),
        "count": count,
        "workdir": str(workdir),
    })


# ---------- browser-terminal (ttyd) ----------

@bp.route("/api/code/web/start", methods=["POST"])
def api_code_web_start():
    """Launch ttyd running the opencode TUI (browser terminal).

    opencode is invoked directly rather than through a ~/bin/opencode-web
    wrapper script: this way it inherits the managed bridge config and the
    active project's model, with no separate file to keep in sync.
    """
    global _ttyd_proc

    if _ttyd_proc is not None and _ttyd_proc.poll() is None:
        return jsonify({
            "ok": True,
            "url": "http://127.0.0.1:8081/terminal",
            "pid": _ttyd_proc.pid,
            "status": "already_running",
        })

    missing = _require_opencode()
    if missing:
        return jsonify({"error": missing[0]}), missing[1]

    workdir, err = _resolve_workdir()
    if err:
        return jsonify({"error": err}), 500

    # opencode takes the folder as a positional argument; openclaude's
    # positional slot is the prompt, so each backend decides its own TUI
    # argv and the working directory comes from cwd.
    b = _backend()
    inner = ([str(a) for a in b.tui_command(workdir)]
             if hasattr(b, "tui_command")
             else [str(b.BINARY), "--model", b.MODEL_REF, str(workdir)])
    cmd = ["ttyd", "-W", "-i", "127.0.0.1", "-p", "8081",
           "--base-path", "/terminal"] + inner
    try:
        _ttyd_proc = subprocess.Popen(
            cmd,
            cwd=str(workdir),
            env=b.env(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return jsonify({
            "error": "ttyd not installed",
            "hint": "pkg install ttyd",
        }), 500
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    return jsonify({
        "ok": True,
        "url": "http://127.0.0.1:8081/terminal",
        "pid": _ttyd_proc.pid,
        "workdir": str(workdir),
        "command": cmd[6:],
        "status": "started",
    })


@bp.route("/api/code/web/status", methods=["GET"])
def api_code_web_status():
    """Check if ttyd is running."""
    if _ttyd_proc is None:
        return jsonify({"running": False})
    if _ttyd_proc.poll() is None:
        return jsonify({"running": True, "pid": _ttyd_proc.pid})
    return jsonify({"running": False, "exited": _ttyd_proc.returncode})


# ---------- chat-style one-shot opencode ----------

@bp.route("/api/code/web/run", methods=["POST"])
def api_code_web_run():
    """Chat-style opencode — one message per call, real files in phone storage.

    Conversation continuity comes from opencode's own session: the id from the
    previous run in this folder is resumed, so follow-ups work without the
    caller passing any history.
    """
    task = (json_body().get("message") or "").strip()
    if not task:
        return jsonify({"error": "message required"}), 400

    # /project and /model are handled by us, so they run before the backend
    # is even looked at — switching backends must work when none is installed.
    if task.startswith("/project"):
        return jsonify(_project_command_reply(task))
    if task == "/model" or task.startswith("/model "):
        return jsonify(_model_command_reply(task))

    missing = _require_opencode()
    if missing:
        return jsonify({"error": missing[0]}), missing[1]

    workdir, err = _resolve_workdir()
    if err:
        return jsonify({"error": err}), 500
    try:
        workdir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return jsonify({"error": f"mkdir failed: {e}"}), 500

    _ensure_safe_directory(workdir)
    ensure_git_identity()

    prompt = build_code_prompt(task)

    # This path previously bypassed the lock the streamed path takes
    # (stream_opencode takes code_lock for its whole run), and opencode writes
    # real files in the shared folder — so a one-shot run and a streamed run
    # could edit the same files at once and corrupt them.
    with code_lock:
        t0 = time.time()
        try:
            rc, lines = _run(workdir, prompt)
        except OpencodeError as e:
            return jsonify({"error": str(e)}), 500

        summary = summarize_output(lines, backend=_backend_name())
        _remember_session(workdir, summary["session_id"])

        commits, commit_error = [], None
        if rc == 0:
            made = commit_changes(workdir, summary["files"], task)
            if made["ok"]:
                commits.append({"hash": made["hash"],
                                "message": made["message"]})
            elif summary["files"]:
                commit_error = made["reason"]

        shown = summary["reply"] or summary.get("error") or "(no output)"
        _record_code_turn(task, shown[:500])

    return jsonify({
        "reply": shown,
        "files": summary["files"],
        "commits": commits,
        "commit_error": commit_error,
        "tools": summary["tools"],
        "tokens": summary["tokens"],
        "cost": summary["cost"],
        "session_id": summary["session_id"],
        "workdir": str(workdir),
        "elapsed": round(time.time() - t0, 1),
        "rc": rc,
        "history_len": len(_code_history),
    })


@bp.route("/api/code/web/clear", methods=["POST"])
def api_code_web_clear():
    """Forget the code-mode conversation.

    opencode owns the real history, so clearing means deleting its session for
    this folder, not just emptying a local list.
    """
    global _code_history
    n = len(_code_history)
    _code_history = []

    deleted = None
    b = _backend()
    workdir, err = _resolve_workdir()
    if not err and workdir:
        session_id = b.forget_session(workdir)
        # openclaude has no `session delete` subcommand: its command comes
        # back as None and only the local id is forgotten, which is all this
        # endpoint can do for it anyway.
        cmd = b.session_delete_command(session_id) if session_id else None
        if session_id and cmd and b.is_installed():
            try:
                subprocess.run(
                    cmd,
                    cwd=str(workdir),
                    capture_output=True,
                    timeout=20,
                )
                deleted = session_id
            except Exception as e:
                log.warning("could not delete %s session %s: %s: %s",
                            b.BINARY, session_id, type(e).__name__, e)
    return jsonify({"ok": True, "cleared": n, "session_deleted": deleted})


@bp.route("/api/code/web/history", methods=["GET"])
def api_code_web_history():
    """Current code-mode mirror plus the opencode session that backs it."""
    workdir, err = _resolve_workdir()
    return jsonify({
        "count": len(_code_history),
        "messages": _code_history,
        "session_id": _session_for(workdir) if not err else None,
        "workdir": str(workdir) if not err else None,
    })


@bp.route("/api/code/doctor", methods=["GET"])
def api_code_doctor():
    """Is code mode actually runnable? Everything the UI needs to say so.

    `?backend=openclaude` reports that backend instead. With no query the
    active project and AI_AGENT_CODE_BACKEND decide, which is "opencode" on a
    stock setup — so a plain call answers exactly as it always has.
    """
    b = _backend()
    info = b.doctor()
    info["backend"] = _backend_name()
    if not info["installed"]:
        info["ok"] = False
        info["error"] = b.missing_install_message()
    else:
        try:
            b.write_bridge_config()
            info["config_exists"] = b.config_file().exists()
            info["ok"] = True
        except Exception as e:
            info["ok"] = False
            info["error"] = f"could not write {b.config_file()}: {e}"
    return jsonify(info)


@bp.route("/api/code/backends", methods=["GET"])
def api_code_backends():
    """Registered backend names, in display order.

    A bare JSON list: the UI asks once and renders it. Per-backend health is
    /api/code/doctor?backend=<name>, which already reports install state,
    version and sessions for whichever backend is asked about.
    """
    return jsonify(code_backends.names())


@bp.route("/api/code/backend", methods=["POST"])
def api_code_backend():
    """Point the active project at a backend: {"name": "openclaude"}.

    The choice hangs off the project (projects.set_backend) so it survives a
    restart and is scoped to that folder — with no active project there is
    nothing to hang it on, and answering 400 says so instead of dropping the
    request on the floor.
    """
    body = json_body()
    name = str(body.get("name") or "").strip()
    if not name:
        return jsonify({"error": "name required",
                        "available": code_backends.names()}), 400
    if name not in code_backends.names():
        return jsonify({"error": f"unknown backend: {name}",
                        "available": code_backends.names()}), 400

    active = projects_store.get_active()
    if not active:
        return jsonify({"error": "no active project",
                        "hint": "add one first: /project add <path>"}), 400
    if not projects_store.set_backend(active["id"], name):
        return jsonify({"error": "could not set backend"}), 500
    return jsonify({
        "ok": True,
        "backend": name,
        "project": active["id"],
        "installed": code_backends.installed(name),
        "available": code_backends.availability(),
    })


# ---------- project registry ----------

@bp.route("/api/code/projects", methods=["GET"])
def api_projects_list():
    return jsonify({
        "projects": projects_store.list_all(),
        "active": projects_store.get_active(),
    })


@bp.route("/api/code/projects", methods=["POST"])
def api_projects_add():
    body = json_body()
    name = (body.get("name") or "").strip()
    path = (body.get("path") or "").strip()
    if not path:
        return jsonify({"error": "path required"}), 400
    try:
        proj = projects_store.add(name, path)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
    return jsonify({"ok": True, "project": proj})


@bp.route("/api/code/projects/<pid>", methods=["DELETE"])
def api_projects_remove(pid):
    if not projects_store.remove(pid):
        return jsonify({"error": "not found"}), 404
    return jsonify({"ok": True})


@bp.route("/api/code/projects/<pid>/activate", methods=["POST"])
def api_projects_activate(pid):
    if not projects_store.set_active(pid):
        return jsonify({"error": "not found"}), 404
    return jsonify({"ok": True, "active": projects_store.get_active()})


# ---------- folder picker ----------

@bp.route("/api/code/browse", methods=["GET"])
def api_code_browse():
    """List folders under a path (for folder picker)."""
    raw = request.args.get("path", "").strip()
    if not raw:
        raw = str(Path.home() / "storage" / "shared")
    base = Path(raw).expanduser()
    try:
        base = base.resolve()
    except Exception:
        return jsonify({"error": "invalid path"}), 400
    if not base.exists() or not base.is_dir():
        return jsonify({"error": f"not a directory: {base}"}), 400

    folders = []
    files_count = 0
    try:
        for entry in sorted(base.iterdir(), key=lambda e: e.name.lower()):
            if entry.name.startswith("."):
                continue
            try:
                if entry.is_dir():
                    folders.append({"name": entry.name, "path": str(entry)})
                else:
                    files_count += 1
            except Exception:
                continue
    except PermissionError:
        return jsonify({"error": "permission denied"}), 403
    except OSError as e:
        # /storage/emulated can vanish mid-request while the
        # SD card mounts — answer instead of a 500 traceback.
        return jsonify({"error": f"cannot list {base}: {e}"}), 500

    parent = str(base.parent) if base.parent != base else None
    return jsonify({
        "path": str(base),
        "parent": parent,
        "folders": folders[:200],
        "files_count": files_count,
    })


# ---------- streaming opencode ----------

@bp.route("/api/code/web/run/stream", methods=["POST"])
def api_code_web_run_stream():
    """Streaming version of /api/code/web/run — opencode events over SSE."""
    task = (json_body().get("message") or "").strip()
    if not task:
        return jsonify({"error": "message required"}), 400

    # Project/model intercept — the browser reads this endpoint as SSE
    # only, so wrap the sync JSON reply as a single end frame.
    if (task.startswith("/project") or task == "/model"
            or task.startswith("/model ")):
        r = api_code_web_run()
        if isinstance(r, tuple):
            r = r[0]
        try:
            body = r.get_json() or {}
        except AttributeError:
            body = {}
        return Response(
            sse_event({
                "type": "end",
                "reply": body.get("reply") or "",
                "files": body.get("files") or [],
                "tools": [],
                "tokens": {},
                "cost": None,
                "session_id": None,
                "elapsed": body.get("elapsed", 0.0),
                "rc": body.get("rc", 0),
            }),
            mimetype="text/event-stream",
        )

    missing = _require_opencode()
    if missing:
        return jsonify({"error": missing[0]}), missing[1]

    workdir, err = _resolve_workdir()
    if err:
        return jsonify({"error": err}), 500
    try:
        workdir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return jsonify({"error": f"mkdir failed: {e}"}), 500

    _ensure_safe_directory(workdir)
    ensure_git_identity()

    env = _backend().env(PYTHONUNBUFFERED="1")
    prompt = build_code_prompt(task)

    def generate():
        reply = None
        try:
            for line in _stream(workdir, prompt, env):
                event = _parse_line(line)
                if isinstance(event, dict) and event.get("type") == "end":
                    reply = event.get("reply")
                    commit = None
                    if event.get("rc") == 0 and event.get("files"):
                        res = commit_changes(workdir, event["files"], task)
                        if res["ok"]:
                            commit = {"hash": res["hash"],
                                      "message": res["message"]}
                        else:
                            # Files were reported but nothing was committed:
                            # its own frame, ahead of `end`, so the UI can say
                            # why the commit list stayed empty.
                            yield sse_event({"type": "commit_error",
                                             "reason": res["reason"]})
                    if commit:
                        # Its own frame, ahead of `end`: the UI renders commits
                        # from a top-level `commit` event, so nesting it in the
                        # end payload left the streamed commit invisible. The
                        # dict itself is left clean so `end.commits` matches
                        # what the one-shot route returns.
                        yield sse_event({"type": "commit", **commit})
                        event["commits"] = [commit]
                        line = sse_event(event)
                yield line
        finally:
            # Reached on normal completion and on client disconnect alike, so
            # the abort path inside the stream and this both matter. The real
            # reply when there is one, so a streamed turn is not filed as a
            # bare "(streamed)" placeholder in /api/code/web/history.
            _record_code_turn(task, (reply or "(streamed)")[:500])

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            # Every proxy between opencode and the browser buffers a body it
            # thinks is not finished; without these the frames arrive in one
            # lump at the end instead of as they happen.
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )
