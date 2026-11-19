"""Memory gate — evidence-bound long-term memory (Phase 2).

Phase 1 gave the agent a gated skill library. This module gives it a
gated MEMORY: durable facts that survive new chats, injected into
every system prompt as a small bounded block. The gate is mechanical,
not approval-based — the owner audits a weekly digest instead of
approving every entry:

    propose_memory(key, text, source_kind, evidence)
      source_kind "owner" / "tool": the evidence quote must be found
        VERBATIM in the evidence log → the entry goes ACTIVE.
      source_kind "inference": the agent's own conclusion → the entry
        goes to QUARANTINE (marked unverified, expires in 14 days
        unless re-evidenced). Inferences are never auto-accepted;
        that is permanent policy, not a probation stage.

The evidence log (~/.ai-agent/memory-evidence.jsonl) is append-only
and filled ONLY by the harness — owner messages (recorded in
begin_turn) and tool results (recorded in _guard_result, after
redaction). The agent's own prose is never written to it, so the
agent can never cite itself as its own evidence. That is the exact
failure this design exists to prevent: on 2026-11-19 the agent
"remembered" a gate message that existed in no version of the code,
because its only source was its own earlier reply.

Layout under ~/.ai-agent:

    memory/entries.json    the store (active + quarantine + history)
    memory-evidence.jsonl  harness-written evidence (FIFO-capped)
    memory-audit.jsonl     append-only record of every gate decision

Owner surface:

    python3 -m core.memorygate list
    python3 -m core.memorygate digest
    python3 -m core.memorygate add <key> <text...>
    python3 -m core.memorygate kill <id>
    python3 -m core.memorygate status

Honest scope: quote verification proves a quote exists in a real
owner/tool record — not that a conclusion drawn from it is sound.
That is why conclusions stay quarantined, and why the claim text and
its quote travel together into the audit and the digest.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import config
from core.log import get_logger

log = get_logger("memorygate")

_BASE = Path.home() / ".ai-agent"
MEMORY_DIR = _BASE / "memory"
ENTRIES_PATH = MEMORY_DIR / "entries.json"
EVIDENCE_PATH = _BASE / "memory-evidence.jsonl"
AUDIT_PATH = _BASE / "memory-audit.jsonl"

_BLOCK_CAP = 1200        # chars for the whole injected memory block
_QUARANTINE_CAP = 300    # chars of that budget for unverified entries
_TEXT_MAX = 500          # one entry's claim text
_QUOTE_MIN = 8           # shortest acceptable evidence quote
_EVIDENCE_KEEP = 300     # evidence records retained (FIFO)
_EVIDENCE_TEXT_CAP = 4000
QUARANTINE_DAYS = 14

_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")
_KINDS = ("owner", "tool")


class MemoryGateError(ValueError):
    """A proposal problem the caller should surface, not hide."""


def _now_dt() -> datetime:
    return datetime.now().astimezone()


def _now() -> str:
    return _now_dt().isoformat(timespec="seconds")


def _ensure_dirs() -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)


def _audit(event: str, key: str, **fields) -> None:
    """Append one audit line. Best-effort, like the skill gate's."""
    rec = {"ts": _now(), "event": event, "key": key}
    rec.update(fields)
    try:
        _ensure_dirs()
        with AUDIT_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("memory audit write failed (%s: %s)",
                    type(e).__name__, e)


def _read_audit() -> list:
    try:
        lines = AUDIT_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except ValueError:
            continue
    return out


# ── the store ────────────────────────────────────────────────────

def _load_entries() -> list:
    try:
        data = json.loads(ENTRIES_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def _save_entries(entries: list) -> None:
    _ensure_dirs()
    tmp = ENTRIES_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(entries, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    tmp.replace(ENTRIES_PATH)


def _entry_id(key: str, text: str) -> str:
    raw = f"{key}\n{text}\n{_now()}"
    return "m_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]


def _expire(entries: list) -> list:
    """Mark quarantined entries past their expiry. Returns entries."""
    now = _now_dt()
    for e in entries:
        if e.get("status") != "quarantine" or not e.get("expires"):
            continue
        try:
            exp = datetime.fromisoformat(e["expires"])
        except ValueError:
            continue
        if exp <= now:
            e["status"] = "expired"
            _audit("expired", e.get("key", "?"), id=e.get("id", ""))
    return entries


def _norm(s: str) -> str:
    return " ".join(str(s).split())


# ── evidence (harness-written only) ──────────────────────────────

def record_evidence(kind: str, text: str) -> None:
    """Append one harness-observed record. Called from begin_turn
    (owner messages) and _guard_result (tool results, already
    redacted). NEVER called with agent prose — that is the whole
    point of the design. Best-effort: evidence must never break a
    turn."""
    if kind not in _KINDS or not text or not str(text).strip():
        return
    try:
        _ensure_dirs()
        rec = {"ts": _now(), "kind": kind,
               "text": str(text)[:_EVIDENCE_TEXT_CAP]}
        with EVIDENCE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        _prune_evidence()
    except Exception as e:
        log.warning("evidence write failed (%s: %s)",
                    type(e).__name__, e)


def _prune_evidence() -> None:
    try:
        lines = EVIDENCE_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    if len(lines) <= _EVIDENCE_KEEP * 2:
        return
    try:
        EVIDENCE_PATH.write_text(
            "\n".join(lines[-_EVIDENCE_KEEP:]) + "\n", encoding="utf-8")
    except OSError:
        pass


def _evidence_texts(kind: str) -> list:
    try:
        lines = EVIDENCE_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for ln in lines:
        try:
            rec = json.loads(ln)
        except ValueError:
            continue
        if rec.get("kind") == kind and rec.get("text"):
            out.append(_norm(rec["text"]))
    return out


def _quote_found(quote: str, kind: str) -> bool:
    q = _norm(quote)
    if len(q) < _QUOTE_MIN:
        return False
    return any(q in t for t in _evidence_texts(kind))


# ── the gate ─────────────────────────────────────────────────────

def propose(key: str, text: str, source_kind: str,
            evidence: str = "") -> dict:
    """Gate one agent memory proposal. Returns the stored entry plus
    a verdict: "active" | "quarantine" | "duplicate"."""
    if not config.MEMORY_PROPOSALS_ENABLED:
        raise MemoryGateError(
            "memory proposals are disabled on this install "
            "(AI_AGENT_MEMORY_PROPOSALS=0)")
    key = (key or "").strip().lower()
    if not _KEY_RE.match(key):
        raise MemoryGateError(
            "key must be a short slug (a-z, 0-9, dash, underscore)")
    text = (text or "").strip()
    if not text:
        raise MemoryGateError("text must not be empty")
    if len(text) > _TEXT_MAX:
        raise MemoryGateError(f"text is over {_TEXT_MAX} chars")
    if source_kind not in ("owner", "tool", "inference"):
        raise MemoryGateError(
            "source_kind must be owner, tool or inference")

    entries = _expire(_load_entries())

    def _store(status: str, reason: str) -> dict:
        entry = {
            "id": _entry_id(key, text),
            "key": key,
            "text": text,
            "tier": source_kind,
            "evidence": (evidence or "")[:1000],
            "status": status,
            "reason": reason,
            "source": "agent",
            "ts": _now(),
            "expires": (_now_dt() + timedelta(days=QUARANTINE_DAYS))
                       .isoformat(timespec="seconds")
                       if status == "quarantine" else None,
        }
        entries.append(entry)
        _save_entries(entries)
        _audit("accepted" if status == "active" else "quarantined",
               key, id=entry["id"], tier=source_kind, reason=reason)
        return {"entry": entry, "verdict": status}

    active_same_key = [e for e in entries
                       if e.get("key") == key
                       and e.get("status") == "active"]
    for e in active_same_key:
        if _norm(e.get("text", "")) == _norm(text):
            _audit("duplicate", key, id=e.get("id", ""))
            return {"entry": e, "verdict": "duplicate"}
    if active_same_key:
        return _store("quarantine",
                      "conflicts with an active entry for this key; "
                      "the owner decides in the digest")
    if source_kind == "inference":
        return _store("quarantine",
                      "agent inference — unverified by design")
    if not _quote_found(evidence or "", source_kind):
        return _store("quarantine",
                      f"evidence quote not found in {source_kind} "
                      "records (or under the minimum length)")
    return _store("active", "evidence quote verified")


def add_owner(key: str, text: str) -> dict:
    """Owner-authored entry (CLI seeding / corrections). The owner
    IS the source, so no quote check; supersedes any existing entry
    for the key, with the supersession written to the audit."""
    key = (key or "").strip().lower()
    if not _KEY_RE.match(key):
        raise MemoryGateError(
            "key must be a short slug (a-z, 0-9, dash, underscore)")
    text = (text or "").strip()
    if not text or len(text) > _TEXT_MAX:
        raise MemoryGateError(f"text must be 1..{_TEXT_MAX} chars")
    entries = _expire(_load_entries())
    for e in entries:
        if e.get("key") == key and e.get("status") in (
                "active", "quarantine"):
            e["status"] = "superseded"
            _audit("superseded", key, id=e.get("id", ""), by="owner")
    entry = {
        "id": _entry_id(key, text), "key": key, "text": text,
        "tier": "owner", "evidence": "(owner direct entry)",
        "status": "active", "reason": "owner-authored",
        "source": "owner-cli", "ts": _now(), "expires": None,
    }
    entries.append(entry)
    _save_entries(entries)
    _audit("owner_set", key, id=entry["id"])
    return {"entry": entry, "verdict": "active"}


def kill(entry_id: str) -> dict:
    entries = _load_entries()
    for e in entries:
        if e.get("id") == entry_id and e.get("status") in (
                "active", "quarantine"):
            e["status"] = "killed"
            _save_entries(entries)
            _audit("killed", e.get("key", "?"), id=entry_id,
                   tier=e.get("tier", ""))
            return e
    raise MemoryGateError(f"no live entry with id {entry_id}")


def list_entries() -> list:
    entries = _expire(_load_entries())
    _save_entries(entries)
    return entries


def status() -> dict:
    entries = list_entries()
    counts = {"active": 0, "quarantine": 0, "expired": 0,
              "killed": 0, "superseded": 0}
    for e in entries:
        if e.get("status") in counts:
            counts[e["status"]] += 1
    totals = {"proposed": 0, "accepted": 0, "quarantined": 0,
              "poison_kills": 0}
    for ev in _read_audit():
        kind = ev.get("event")
        if kind in ("accepted", "quarantined"):
            totals["proposed"] += 1
        if kind == "accepted":
            totals["accepted"] += 1
        elif kind == "quarantined":
            totals["quarantined"] += 1
        elif kind == "killed" and ev.get("tier") in ("owner", "tool"):
            # an auto-accepted entry the owner had to kill — the
            # trust signal that shrinks the digest interval again
            totals["poison_kills"] += 1
    return {
        "entries": counts,
        "proposals": totals,
        "proposals_enabled": bool(config.MEMORY_PROPOSALS_ENABLED),
    }


def digest() -> dict:
    entries = list_entries()
    return {
        "active": [e for e in entries if e.get("status") == "active"],
        "quarantine": [e for e in entries
                       if e.get("status") == "quarantine"],
        "status": status(),
    }


# ── agent visibility ─────────────────────────────────────────────

def agent_memory_block(max_chars: int = _BLOCK_CAP) -> str:
    """Compact memory for the system prompt. Active entries first;
    quarantined ones ride in a clearly-marked unverified section
    with a smaller budget. Empty string when there is nothing, so a
    memory-less install pays zero context."""
    try:
        entries = list_entries()
    except Exception:
        return ""
    active = [e for e in entries if e.get("status") == "active"]
    quarantined = [e for e in entries
                   if e.get("status") == "quarantine"]
    if not active and not quarantined:
        return ""
    lines = [
        "## Long-term memory",
        "Facts that survived the memory gate. Rely on the verified "
        "ones; treat the unverified section as unproven. Propose new "
        "durable facts with propose_memory (verbatim evidence quote "
        "required — never cite your own reply).",
    ]
    for e in active:
        line = f"- [{e.get('key', '?')}] {e.get('text', '')}"
        if len("\n".join(lines)) + len(line) + 1 > max_chars:
            lines.append("- … (memory truncated — oldest/least "
                         "recent entries not shown)")
            break
        lines.append(line)
    if quarantined:
        qlines = ["Unverified (quarantine — do NOT rely on these "
                  "without checking):"]
        for e in quarantined:
            line = f"- [{e.get('key', '?')}] {e.get('text', '')}"
            if len("\n".join(qlines)) + len(line) + 1 > _QUARANTINE_CAP:
                break
            qlines.append(line)
        block = "\n".join(lines) + "\n" + "\n".join(qlines)
        if len(block) <= max_chars:
            return block
    return "\n".join(lines)[:max_chars]


# ── owner CLI ────────────────────────────────────────────────────

def _print_entry(e: dict) -> None:
    print(f"- {e.get('id', '?')} [{e.get('key', '?')}] "
          f"({e.get('tier', '?')}, {e.get('status', '?')}) "
          f"{e.get('text', '')}")
    if e.get("reason"):
        print(f"    reason: {e['reason']}")
    if e.get("evidence"):
        print(f"    evidence: {e['evidence'][:120]}")


def main(argv: list | None = None) -> int:
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    usage = ("usage: python3 -m core.memorygate "
             "list|digest|add <key> <text...>|kill <id>|status")
    if not argv:
        print(__doc__.split("Layout")[0])
        print(usage)
        return 2
    cmd, rest = argv[0], argv[1:]
    try:
        if cmd == "list":
            items = list_entries()
            live = [e for e in items if e.get("status") in (
                "active", "quarantine")]
            if not live:
                print("memory is empty")
            for e in live:
                _print_entry(e)
            return 0
        if cmd == "digest":
            d = digest()
            print(f"active: {len(d['active'])}   "
                  f"quarantine: {len(d['quarantine'])}")
            for e in d["quarantine"]:
                _print_entry(e)
            st = d["status"]
            print(f"poison kills (auto-accepted, owner-killed): "
                  f"{st['proposals']['poison_kills']}")
            return 0
        if cmd == "add" and len(rest) >= 2:
            out = add_owner(rest[0], " ".join(rest[1:]))
            print(f"SET: [{out['entry']['key']}] "
                  f"{out['entry']['text']}")
            return 0
        if cmd == "kill" and rest:
            e = kill(rest[0])
            print(f"KILLED: {e.get('id')} [{e.get('key')}]")
            return 0
        if cmd == "status":
            st = status()
            print(f"active: {st['entries']['active']}   "
                  f"quarantine: {st['entries']['quarantine']}   "
                  f"expired: {st['entries']['expired']}   "
                  f"killed: {st['entries']['killed']}")
            print(f"proposals: {st['proposals']['proposed']}   "
                  f"accepted: {st['proposals']['accepted']}   "
                  f"quarantined: {st['proposals']['quarantined']}   "
                  f"poison kills: {st['proposals']['poison_kills']}")
            print(f"proposals enabled: {st['proposals_enabled']}")
            return 0
    except MemoryGateError as e:
        print(f"memorygate: {e}")
        return 1
    print(usage)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
