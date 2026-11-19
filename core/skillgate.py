"""Skill gate — the proposal pipeline for agent-learned skills (Phase 1).

The agent improves by drafting new skills, but a draft must never become
a live skill on its own. Layout under ``~/.ai-agent``:

    skills/            live skills (owned by core/skills.py)
    skills-inbox/      agent proposals waiting for the owner (this module)
    skills-archive/    previous live versions, kept on every overwrite
    skills-audit.jsonl append-only record of the whole pipeline

Flow: the agent calls the ``propose_skill`` tool → the draft lands in
the inbox with provenance metadata (where it came from, what evidence
supports it, which app version it was learned under) → the gate checks
it (format, size, placeholder, secret shapes, injection markers) → the
OWNER applies or rejects it by hand:

    python3 -m core.skillgate list
    python3 -m core.skillgate check <name>
    python3 -m core.skillgate approve <name> [--force]
    python3 -m core.skillgate reject <name> [reason]
    python3 -m core.skillgate status

Probation A: there is no auto-apply path in this phase, by design.
Injection markers are warnings, not hard failures — ordinary templates
can legitimately contain phrases like "act as" — so an owner may accept
them explicitly with --force; the override is written to the audit log.
Secret shapes and format errors are hard failures and cannot be forced.

Honest scope: this gate is a review workflow with an audit trail, not
a sandbox against the agent itself. Its value is that every learned
skill passes human eyes, carries provenance, and can be rolled back
from the archive.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import config
from core import skills as _skills
from core.log import get_logger
from core.safety import redact_secrets, scan_injection

log = get_logger("skillgate")

_BASE = Path.home() / ".ai-agent"
INBOX_DIR = _BASE / "skills-inbox"
ARCHIVE_DIR = _BASE / "skills-archive"
AUDIT_PATH = _BASE / "skills-audit.jsonl"

_SOURCES = ("experience", "web", "user", "other")
_INDEX_CAP = 900  # chars for the agent system-prompt skills block


class SkillGateError(ValueError):
    """A proposal/apply problem the caller should surface, not hide."""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ensure_dirs() -> None:
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


def _inbox_file(name: str) -> Path:
    return INBOX_DIR / f"{name}.md"


def _meta_file(name: str) -> Path:
    return INBOX_DIR / f"{name}.meta.json"


def _audit(event: str, name: str, **fields) -> None:
    """Append one audit line. Best-effort: a full disk must not block
    the owner's apply, but the failure is logged loudly."""
    rec = {"ts": _now(), "event": event, "name": name}
    rec.update(fields)
    try:
        _ensure_dirs()
        with AUDIT_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("skills audit write failed (%s: %s)",
                    type(e).__name__, e)


def _read_meta(name: str) -> dict:
    try:
        return json.loads(_meta_file(name).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_inbox(name: str) -> tuple:
    """(meta, body, raw_text) for an inbox proposal. Raises SkillGateError."""
    f = _inbox_file(name)
    if not f.is_file():
        raise SkillGateError(f"no proposal named {name!r} in the inbox")
    raw = f.read_text(encoding="utf-8", errors="replace")
    meta, body = _skills._parse_frontmatter(raw)
    return meta, body, raw


# ── propose ──────────────────────────────────────────────────────

def propose(name: str, description: str, body: str, source: str = "experience",
            evidence: str = "", args: str = "text") -> dict:
    """Write a proposal into the inbox and gate-check it.

    The proposal is recorded even when the gate fails — a failed draft
    is evidence too (it counts against the probation streak). Returns
    {"name": ..., "gate": <check report>}.
    """
    if not config.SKILL_PROPOSALS_ENABLED:
        raise SkillGateError(
            "skill proposals are disabled (AI_AGENT_SKILL_PROPOSALS=0)")
    content = _skills.build_markdown(name, description, body, args)
    safe = _skills._sanitize_name(name)
    src = (source or "").strip().lower()
    if src not in _SOURCES:
        src = "other"
    _ensure_dirs()
    target = _inbox_file(safe)
    tmp = INBOX_DIR / f".{safe}.md.tmp"
    try:
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(target)
    except Exception:
        try:
            tmp.unlink()
        except Exception:
            pass
        raise
    meta = {
        "name": safe,
        "source": src,
        "evidence": (evidence or "").strip()[:500],
        "app_version": config.VERSION,
        "proposed_at": _now(),
        "sha256": _sha(content),
        "proposer": "agent",
    }
    _meta_file(safe).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    report = check(safe)
    _audit("proposed", safe, source=src, gate_ok=report["ok"],
           failures=report["failures"], warnings=report["warnings"])
    return {"name": safe, "gate": report}


# ── capability availability (advisory) ───────────────────────────
# A skill whose procedure leans on a capability this install has
# turned off will fail mid-task — the 2026-11-19 lesson, where the
# first learned skill drove shell commands on an install whose
# shell was disabled. The gate WARNS (never blocks: the owner may
# enable the capability before the skill is used). To cover a new
# gated capability, add ONE row to _CAPABILITY_ROWS — (tool name as
# it appears in a skill body, label, availability probe). Probes
# fail OPEN (treated as available) when the state cannot be read:
# advice must never cry wolf on a guess.

def _shell_enabled() -> bool:
    try:
        from core.key_manager import KeyManager
        return bool(KeyManager().shell_enabled())
    except Exception:
        return True


_CAPABILITY_ROWS = (
    ("run_command", "shell commands", _shell_enabled),
)


def _capability_warnings(body: str) -> list:
    out = []
    for tool, label, probe in _CAPABILITY_ROWS:
        if not re.search(r"\b" + re.escape(tool) + r"\b", body or ""):
            continue
        try:
            available = bool(probe())
        except Exception:
            available = True
        if not available:
            out.append(
                f"capability: body uses {tool} but {label} are "
                "currently disabled on this install — the skill "
                "will fail at that step until they are enabled")
    return out


# ── gate ─────────────────────────────────────────────────────────

def check(name: str) -> dict:
    """Run the gate over an inbox proposal.

    Hard failures block approval: bad format, oversize, missing
    {{args}}, secret shapes. Injection markers are warnings (the
    owner may accept them with --force at approve time).
    """
    safe = _skills._sanitize_name(name)
    if not safe:
        raise SkillGateError(f"invalid skill name: {name!r}")
    meta, body, raw = _parse_inbox(safe)
    failures: list = []
    warnings: list = []

    description = meta.get("description", "")
    skill_name = meta.get("name") or safe
    args = meta.get("args", "text")

    # format: the stored file must rebuild cleanly through the same
    # validator that save_skill uses.
    try:
        _skills.build_markdown(skill_name, description, body, args)
    except ValueError as e:
        failures.append(f"format: {e}")
    if skill_name != safe:
        failures.append(
            f"format: frontmatter name {skill_name!r} != file name {safe!r}")

    if len(raw.encode("utf-8")) > _skills._MAX_FILE_SIZE:
        failures.append("size: proposal exceeds the 50 KB skill cap")
    if "{{args}}" not in body:
        failures.append("placeholder: body has no {{args}}")
    if redact_secrets(body) != body or redact_secrets(description) != description:
        failures.append("secrets: body or description contains a secret shape")

    markers = scan_injection(description + "\n" + body)
    if markers:
        warnings.append("injection markers: " + ", ".join(markers))

    warnings.extend(_capability_warnings(body))

    update = _skills.load_skill(safe) is not None

    # trial render: the template must produce usable text.
    rendered = ""
    if not failures:
        rendered = _skills.render(
            {"name": skill_name, "description": description,
             "args": args, "body": body}, "sample input")
        if not rendered.strip():
            failures.append("render: template renders to empty text")

    return {
        "name": safe,
        "ok": not failures,
        "failures": failures,
        "warnings": warnings,
        "update": update,
        "description": description,
        "meta": _read_meta(safe),
        "render_preview": rendered[:200],
    }


# ── owner actions ────────────────────────────────────────────────

def approve(name: str, force: bool = False) -> dict:
    """Apply an inbox proposal to the live skills dir (owner only).

    A live skill of the same name is archived first, so any apply can
    be rolled back by hand from skills-archive/.
    """
    safe = _skills._sanitize_name(name)
    if not safe:
        raise SkillGateError(f"invalid skill name: {name!r}")
    report = check(safe)
    if not report["ok"]:
        raise SkillGateError(
            "gate failed: " + "; ".join(report["failures"]))
    if report["warnings"] and not force:
        raise SkillGateError(
            "gate warnings need an explicit accept — re-run with "
            "--force after reading them: " + "; ".join(report["warnings"]))
    meta, body, _raw = _parse_inbox(safe)

    archived = False
    live_file = _skills.SKILLS_DIR / f"{safe}.md"
    if live_file.is_file():
        _ensure_dirs()
        archive_file = ARCHIVE_DIR / f"{safe}-{_stamp()}.md"
        n = 2
        while archive_file.exists():
            archive_file = ARCHIVE_DIR / f"{safe}-{_stamp()}-{n}.md"
            n += 1
        archive_file.write_bytes(live_file.read_bytes())
        archived = True

    saved = _skills.save_skill(
        safe, meta.get("description", ""), body, meta.get("args", "text"))
    new_raw = live_file.read_text(encoding="utf-8", errors="replace")
    for f in (_inbox_file(safe), _meta_file(safe)):
        try:
            f.unlink()
        except Exception:
            pass
    _audit("approved", safe, sha256=_sha(new_raw), archived=archived,
           forced=bool(force and report["warnings"]),
           warnings=report["warnings"])
    return {"name": safe, "archived": archived, **saved}


def reject(name: str, reason: str = "") -> dict:
    """Drop an inbox proposal (owner only), with the reason on record."""
    safe = _skills._sanitize_name(name)
    if not safe:
        raise SkillGateError(f"invalid skill name: {name!r}")
    if not _inbox_file(safe).is_file():
        raise SkillGateError(f"no proposal named {safe!r} in the inbox")
    for f in (_inbox_file(safe), _meta_file(safe)):
        try:
            f.unlink()
        except Exception:
            pass
    _audit("rejected", safe, reason=(reason or "").strip()[:200])
    return {"name": safe, "rejected": True}


def list_inbox() -> list:
    """All pending proposals with provenance + gate summary."""
    _ensure_dirs()
    out = []
    for f in sorted(INBOX_DIR.glob("*.md")):
        if f.name.startswith("_") or f.name.startswith("."):
            continue
        name = f.stem
        try:
            report = check(name)
            out.append({
                "name": name,
                "meta": report["meta"],
                "description": report["description"],
                "gate_ok": report["ok"],
                "failures": report["failures"],
                "warnings": report["warnings"],
                "update": report["update"],
            })
        except Exception as e:
            out.append({"name": name, "error": f"{type(e).__name__}: {e}"})
    return out


# ── usage record (the outcome loop) ──────────────────────────────

def record_load(name: str) -> None:
    safe = _skills._sanitize_name(name)
    if safe:
        _audit("loaded", safe)


def record_outcome(name: str, ok: bool, note: str = "") -> None:
    safe = _skills._sanitize_name(name)
    if safe:
        _audit("outcome", safe, ok=bool(ok), note=(note or "").strip()[:200])


def _read_audit() -> list:
    try:
        lines = AUDIT_PATH.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def status() -> dict:
    """Pipeline counters — the probation numbers live here.

    consecutive_clean = approvals since the last rejected proposal or
    gate-failed proposal. Phase 1 graduation yardstick (owner-set):
    10 consecutive clean approvals with no rejection/gate failure in
    between earns a probation review; any failure resets the count.
    """
    events = _read_audit()
    counts = {"proposed": 0, "approved": 0, "rejected": 0}
    streak = 0
    uses: dict = {}
    for ev in events:
        kind = ev.get("event")
        if kind == "proposed":
            counts["proposed"] += 1
            if not ev.get("gate_ok"):
                streak = 0
        elif kind == "approved":
            counts["approved"] += 1
            streak += 1
        elif kind == "rejected":
            counts["rejected"] += 1
            streak = 0
        elif kind in ("loaded", "outcome"):
            u = uses.setdefault(ev.get("name", "?"),
                                {"loaded": 0, "worked": 0, "failed": 0})
            if kind == "loaded":
                u["loaded"] += 1
            elif ev.get("ok"):
                u["worked"] += 1
            else:
                u["failed"] += 1
    return {
        "live_skills": len(_skills.list_skills()),
        "inbox": len(list_inbox()),
        "proposals": counts,
        "consecutive_clean": streak,
        "uses": uses,
        "proposals_enabled": bool(config.SKILL_PROPOSALS_ENABLED),
    }


# ── agent visibility ─────────────────────────────────────────────

def agent_index_block(max_chars: int = _INDEX_CAP) -> str:
    """Compact skills index for the agent's system prompt.

    Names + one-line descriptions only; full text is loaded on demand
    through the use_skill tool. Empty string when there are no skills,
    so a skill-less install pays zero context.
    """
    try:
        skills = _skills.list_skills()
    except Exception:
        return ""
    if not skills:
        return ""
    lines = [
        "## Learned skills",
        "When a task matches a skill below, call use_skill(name) and "
        "follow the returned procedure; afterwards report whether it "
        "worked with skill_outcome. After a successful new procedure "
        "that would help again, draft it with propose_skill — drafts "
        "go to an inbox and only become real skills when the owner "
        "approves them.",
    ]
    shown = 0
    for s in skills:
        line = f"- {s['name']}: {s['description'] or '(no description)'}"
        if len("\n".join(lines)) + len(line) + 1 > max_chars:
            rest = len(skills) - shown
            lines.append(f"- … and {rest} more not shown")
            break
        lines.append(line)
        shown += 1
    return "\n".join(lines)


# ── owner CLI ────────────────────────────────────────────────────

def _print_report(report: dict) -> None:
    meta = report.get("meta") or {}
    print(f"proposal: {report['name']}")
    if meta:
        print(f"  source: {meta.get('source', '?')}  "
              f"app: {meta.get('app_version', '?')}  "
              f"at: {meta.get('proposed_at', '?')}")
        if meta.get("evidence"):
            print(f"  evidence: {meta['evidence']}")
    print(f"  description: {report.get('description', '')}")
    print(f"  update of a live skill: {report.get('update', False)}")
    for f in report["failures"]:
        print(f"  FAIL: {f}")
    for w in report["warnings"]:
        print(f"  WARN: {w}")
    print(f"  gate: {'PASS' if report['ok'] else 'FAIL'}")


def main(argv: list | None = None) -> int:
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__.split("Flow:")[0])
        print("usage: python3 -m core.skillgate "
              "list|check <name>|approve <name> [--force]|"
              "reject <name> [reason]|status")
        return 2
    cmd, rest = argv[0], argv[1:]
    try:
        if cmd == "list":
            items = list_inbox()
            if not items:
                print("inbox is empty")
            for it in items:
                if "error" in it:
                    print(f"- {it['name']}: unreadable ({it['error']})")
                    continue
                gate = "PASS" if it["gate_ok"] else "FAIL"
                warn = f", {len(it['warnings'])} warning(s)" if it["warnings"] else ""
                upd = ", updates a live skill" if it["update"] else ""
                print(f"- {it['name']} [{gate}{warn}{upd}] "
                      f"{it['description']}")
            return 0
        if cmd == "check" and rest:
            _print_report(check(rest[0]))
            return 0
        if cmd == "approve" and rest:
            force = "--force" in rest
            names = [a for a in rest if not a.startswith("--")]
            if not names:
                print("usage: python3 -m core.skillgate approve <name> [--force]")
                return 2
            name = names[0]
            _print_report(check(name))
            out = approve(name, force=force)
            print(f"APPLIED: {out['name']} is now a live skill"
                  + (" (previous version archived)" if out["archived"] else ""))
            return 0
        if cmd == "reject" and rest:
            reason = rest[1] if len(rest) > 1 else ""
            out = reject(rest[0], reason)
            print(f"REJECTED: {out['name']} removed from the inbox")
            return 0
        if cmd == "status":
            st = status()
            print(f"live skills: {st['live_skills']}   inbox: {st['inbox']}")
            print(f"proposals: {st['proposals']['proposed']}   "
                  f"approved: {st['proposals']['approved']}   "
                  f"rejected: {st['proposals']['rejected']}")
            print(f"consecutive clean approvals: {st['consecutive_clean']}")
            print(f"proposals enabled: {st['proposals_enabled']}")
            for name, u in sorted(st["uses"].items()):
                print(f"  {name}: loaded {u['loaded']}, "
                      f"worked {u['worked']}, failed {u['failed']}")
            return 0
    except SkillGateError as e:
        print(f"skillgate: {e}")
        return 1
    print("usage: python3 -m core.skillgate "
          "list|check <name>|approve <name> [--force]|"
          "reject <name> [reason]|status")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
