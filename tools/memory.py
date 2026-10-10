"""Memory tools for agent mode."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from core.tfidf import BM25Lite
from memory import (
    Memory, MAX_FACT_CHARS,
    _has_negative_self, _is_rule_reference, _is_tool_limitation,
    _normalize_for_dup,
)


def _iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _load() -> Memory | str:
    try:
        return Memory.load()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def _compact(d: dict, max_chars: int) -> str:
    text = ""
    if d.get("identity"):
        parts = []
        for k, v in sorted(d["identity"].items()):
            parts.append(f"{k}={v}")
        text += "identity: " + ", ".join(parts) + "\n"
    if d.get("profile"):
        p = d["profile"]
        text += f"profile: ai_name={p.get('ai_name','')}, user_name={p.get('user_name','')}\n"
    projects = d.get("active_projects") or []
    text += f"projects: {len(projects)}"
    if projects:
        for pr in projects[:3]:
            text += f" ({pr.get('path','')} — {pr.get('note','')})"
        if len(projects) > 3:
            text += f" ... ({len(projects)-3} more)"
    text += "\n"
    topics = d.get("recent_topics") or []
    text += f"topics: {len(topics)}"
    if topics:
        for t in topics[:3]:
            text += f" ({t.get('text','')})"
        if len(topics) > 3:
            text += f" ... ({len(topics)-3} more)"
    text += "\n"
    facts = d.get("facts") or []
    text += f"facts: {len(facts)}"
    if facts:
        for f in facts[:3]:
            text += f" ({f.get('text','')})"
        if len(facts) > 3:
            text += f" ... ({len(facts)-3} more)"
    if len(text) > max_chars:
        text = text[:max_chars] + "... (truncated)"
    return text.strip()


def memory_read() -> str:
    """Read the current persistent memory: identity fields,
    active projects, recent topics, and every fact on file.

    Use this before answering anything personal, before assuming
    what the user is working on, and before writing anything new
    (so you do not duplicate entries).

    Returns a compact, human-readable summary.
    """
    try:
        m = _load()
        if isinstance(m, str):
            return m
        return _compact(m.data, 1500)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def memory_write(fact: str, category: str = "fact") -> str:
    """Add ONE memory entry.

    Args:
        fact: the text to remember (max 500 chars).
        category: one of: fact, identity, topic, project.
            fact      -> a durable statement about the user / work
            identity  -> a rule the agent must follow (tone,
                         language, name)
            topic     -> a recent subject of work
            project   -> a project path the user is working on

    Refuses: empty text, > 500 chars, unknown category, duplicates
    (exact or near), negative self-statements, rule references, tool
    limitations. Returns what was written — or why it was rejected,
    with the specific reason. Never raises on a user-reachable path.
    """
    try:
        text = (fact or "").strip()
        if not text:
            return "Error: empty fact"
        if len(text) > MAX_FACT_CHARS:
            return f"Error: fact too long ({len(text)} > 500)"
        cat = (category or "fact").strip().lower()
        if cat not in {"fact", "identity", "topic", "project"}:
            return f"Error: unknown category: {category}"
        m = _load()
        if isinstance(m, str):
            return m
        if cat == "fact":
            ok, reason = m.add_fact(text, source="manual")
            if not ok:
                if reason == "exact duplicate":
                    return "Error: duplicate fact"
                return f"Error: fact rejected ({reason})"
            return f"Wrote {cat}: {text}"
        elif cat == "identity":
            if _has_negative_self(text):
                return "Error: identity rejected (negative self)"
            if _is_rule_reference(text):
                return "Error: identity rejected (rule reference)"
            identity = m.data.setdefault("identity", {})
            norm = _normalize_for_dup(text)
            for k, v in identity.items():
                if _normalize_for_dup(v) == norm:
                    return f"Error: identity duplicate of {k}"
            keys = list(identity.keys())
            rule_keys = [k for k in keys if k.startswith("rule_")]
            if len(rule_keys) >= 20:
                return "Error: identity rules at capacity (20)"
            n = 0
            while f"rule_{n}" in keys:
                n += 1
            key = f"rule_{n}"
            identity[key] = text
            m.save()
            return f"Wrote {cat}: {text}"
        elif cat == "topic":
            topics = m.data.setdefault("recent_topics", [])
            topics.append({"text": text, "ts": _iso_now()})
            m.save()
            return f"Wrote {cat}: {text}"
        elif cat == "project":
            projects = m.data.setdefault("active_projects", [])
            projects.append({"path": text, "note": ""})
            m.save()
            return f"Wrote {cat}: {text}"
        return f"Error: unknown category: {category}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def memory_search(query: str) -> str:
    """Search memory: exact substring matches first, then BM25-ranked
    fuzzy matches (Burmese n-gram tokens).

    Returns matching entries with their category, or 'no matches'.
    Use before writing to check whether a fact is already known.
    """
    try:
        q = (query or "").strip().lower()
        if not q:
            return "Error: empty query"
        m = _load()
        if isinstance(m, str):
            return m
        d = m.data
        hits: list[str] = []
        seen: set[str] = set()

        def _add(label: str, text: str) -> None:
            line = f"{label}: {text}"
            if line not in seen:
                seen.add(line)
                hits.append(line)

        # 1) exact substring (kept — guaranteed correct, case-insensitive)
        p = d.get("profile") or {}
        for v in p.values():
            if v and q in str(v).lower():
                _add("profile", str(v))
        for k, v in (d.get("identity") or {}).items():
            if q in k.lower() or q in str(v).lower():
                _add("identity", f"{k}={v}")
        for f in d.get("facts") or []:
            if q in str(f.get("text", "")).lower():
                _add("fact", str(f.get("text", "")))
        for t in d.get("recent_topics") or []:
            if q in str(t.get("text", "")).lower():
                _add("topic", str(t.get("text", "")))
        for pr in d.get("active_projects") or []:
            if q in str(pr.get("path", "")).lower() or q in str(pr.get("note", "")).lower():
                _add("project", f"{pr.get('path','')} — {pr.get('note','')}")

        # 2) BM25 fuzzy pass — Burmese n-gram / Latin word overlap
        #    (appends ranked hits the substring pass missed)
        items: list[tuple[str, str, str]] = []  # (id, label, display)
        table: dict[str, tuple[str, str]] = {}
        for v in p.values():
            if v:
                items.append((f"profile:{v}", "profile", str(v)))
        for k, v in (d.get("identity") or {}).items():
            items.append((f"identity:{k}", "identity", f"{k}={v}"))
        for f in d.get("facts") or []:
            items.append((f"fact:{f.get('text','')}", "fact", str(f.get("text", ""))))
        for t in d.get("recent_topics") or []:
            items.append((f"topic:{t.get('text','')}", "topic", str(t.get("text", ""))))
        for pr in d.get("active_projects") or []:
            items.append(
                (f"project:{pr.get('path','')}", "project",
                 f"{pr.get('path','')} — {pr.get('note','')}"),
            )
        for item_id, lab, disp in items:
            table[item_id] = (lab, disp)
        if items:
            idx = BM25Lite()
            idx.build([(i, txt) for i, _lab, txt in items])
            for item_id, _score in idx.search(q, k=10):
                lab, disp = table[item_id]
                _add(lab, disp)

        if not hits:
            return "no matches"
        out = []
        for h in hits[:20]:
            out.append(h)
        if len(hits) > 20:
            out.append(f"... ({len(hits)-20} more)")
        return "\n".join(out)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def memory_forget(match: str) -> str:
    """Remove every memory entry whose text contains `match`
    (case-insensitive).

    Removes from facts / topics / projects only. Identity fields
    and the two names (ai_name, user_name) are never removed
    through this tool — that keeps the agent's own rules stable.

    Returns what was removed and how many entries were touched.
    """
    try:
        mstr = (match or "").strip().lower()
        if not mstr:
            return "Error: empty match"
        m = _load()
        if isinstance(m, str):
            return m
        removed = []
        d = m.data
        facts = d.get("facts") or []
        new_facts = []
        for f in facts:
            t = str(f.get("text", "")).lower()
            if mstr in t:
                removed.append(f"fact: {f.get('text','')}")
            else:
                new_facts.append(f)
        d["facts"] = new_facts
        topics = d.get("recent_topics") or []
        new_topics = []
        for t in topics:
            txt = str(t.get("text", "")).lower()
            if mstr in txt:
                removed.append(f"topic: {t.get('text','')}")
            else:
                new_topics.append(t)
        d["recent_topics"] = new_topics
        projects = d.get("active_projects") or []
        new_projects = []
        for pr in projects:
            txt = (str(pr.get("path","")) + " " + str(pr.get("note",""))).lower()
            if mstr in txt:
                removed.append(f"project: {pr.get('path','')} — {pr.get('note','')}")
            else:
                new_projects.append(pr)
        d["active_projects"] = new_projects
        if not removed:
            return "no matches"
        m.save()
        out = [f"Removed {len(removed)} entr(y/ies):"]
        for r in removed:
            out.append(f"- {r}")
        return "\n".join(out)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
