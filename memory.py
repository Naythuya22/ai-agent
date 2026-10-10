"""Persistent memory layer for ai-agent (Design 3 — Phase 3C-1)."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from core.log import get_logger

log = get_logger("memory")

MEMORY_DIR = Path.home() / ".ai-agent"
MEMORY_FILE = MEMORY_DIR / "memory.json"

MAX_FACT_CHARS = 500
MAX_CONTEXT_CHARS = 1500

# ── Write-time validation ──────────────────────────────────────────────
#
# The agent must not record its own limits, tool quirks, or references to
# existing rules as durable facts. They are transient (they change when the
# code changes), they pollute the system prompt on the next turn, and they
# were the source of the 2026-10-10 cleanup (9 entries removed by hand).
#
# Three filters, kept narrow on purpose — a fact that merely mentions one
# of these phrases is not the same as a fact that is one of these phrases.

_NEGATIVE_SELF_MARKERS = (
    # Burmese — must contain the self-reference (ညီ / ငါ့) or an explicit
    # capability phrase. A bare "မရှိပါ" is too broad: "ငါ့မှာ ကားမရှိပါ"
    # is a legitimate user fact about the user, not about the agent.
    "ညီ မရှိပါ", "ညီ မလုပ်နိုင်ပါ", "ညီ မပြုနိုင်ကြောင်း",
    "ညီ့မှာ", "ညီမှာ ဘာအားနည်းချက်", "ညီ့မှာ ဘာအားနည်းချက်",
    "စွမ်းရည်မရှိ", "စွမ်းရည် မရှိ",
    "ညီ မတတ်ပါ", "ညီ မသိပါ",
    # English — first person or explicit agent reference
    "i cannot", "i am only able", "i'm only able",
    "i do not have", "i don't have",
    "i am not able", "i'm not able",
    "agent cannot", "assistant cannot",
    "as an ai, i", "as an ai i",
)

# A fact that names another rule is a reference, not a new fact.
_RULE_REFERENCE_RE = re.compile(r"\brule_\d+\b")

# A fact about a tool's own behavior belongs in the code, not in memory.
_TOOL_LIMITATION_RE = re.compile(
    r"\btool\s+\w+\s+(fails|errors|does not|doesn't|cannot)\b",
    re.IGNORECASE,
)


def _has_negative_self(text: str) -> bool:
    """True when the text describes the agent's own limits."""
    low = " ".join(str(text or "").split()).lower()
    return any(m in low for m in _NEGATIVE_SELF_MARKERS)


def _is_rule_reference(text: str) -> bool:
    return bool(_RULE_REFERENCE_RE.search(str(text or "")))


def _is_tool_limitation(text: str) -> bool:
    return bool(_TOOL_LIMITATION_RE.search(str(text or "")))


def _normalize_for_dup(text: str) -> str:
    """Lowercase, collapse whitespace, strip a leading ``key=`` when present.

    ``tone=short, answer first`` and ``short, answer first`` normalize to
    the same string — which is exactly the duplication that slipped
    through the exact-match check on 2026-10-06.
    """
    s = " ".join(str(text or "").split()).lower()
    if "=" in s:
        head, _, tail = s.partition("=")
        if head and len(head) <= 20 and " " not in head:
            s = tail.strip()
    return s



def _quarantine(path: Path) -> None:
    """Copy a broken memory file aside so a later save cannot destroy it."""
    try:
        if not path.exists():
            return
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        side = path.with_name(f"{path.name}.corrupt-{ts}")
        side.write_bytes(path.read_bytes())
        log.error("corrupt memory store preserved at %s", side)
    except OSError as e:
        log.error("could not preserve corrupt memory store %s: %s: %s",
                  path, type(e).__name__, e)


class Memory:
    """Shared, persistent memory across modes."""

    VERSION = 1

    def __init__(self, path: Path | None = None):
        self.path = path or MEMORY_FILE
        self.data = self._empty()

    @staticmethod
    def _empty() -> dict:
        return {
            "version": Memory.VERSION,
            "profile": {},
            "identity": {},
            "active_projects": [],
            "recent_topics": [],
            "facts": [],
            "last_updated": "",
        }

    # ---------- IO ----------

    @classmethod
    def load(cls, path: Path | None = None) -> "Memory":
        m = cls(path)
        if not m.path.exists():
            return m
        try:
            raw = json.loads(m.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            # Facts/profile vanish silently here if we just return empty —
            # and the next save() would overwrite the file for good.
            log.error("memory store unreadable (%s: %s) — starting empty",
                      type(e).__name__, e)
            _quarantine(m.path)
            return m
        merged = m._empty()
        for k in merged:
            if k in raw:
                merged[k] = raw[k]
        m.data = merged
        return m

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.data["last_updated"] = datetime.now().isoformat(timespec="seconds")
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(self.data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(self.path)
        except OSError as e:
            # Callers of add_fact/set_profile ignore the return value, so this
            # log line is the only trace that the write was lost.
            log.error("memory save failed for %s: %s: %s",
                      self.path, type(e).__name__, e)
            raise

    # ---------- Mutations ----------

    def add_fact(self, text: str, source: str = "manual") -> tuple[bool, str]:
        """Append one fact. Returns ``(ok, reason)``.

        ``reason`` is "" on success; otherwise one of: ``empty``,
        ``too long``, ``negative self``, ``rule reference``,
        ``tool limitation``, ``exact duplicate``, ``near-duplicate``.
        Callers that only need success/failure use the bool half.
        """
        text = text.strip()
        if not text:
            return False, "empty"
        if len(text) > MAX_FACT_CHARS:
            return False, "too long"
        if _has_negative_self(text):
            log.debug("fact rejected (negative self): %s", text[:60])
            return False, "negative self"
        if _is_rule_reference(text):
            log.debug("fact rejected (rule reference): %s", text[:60])
            return False, "rule reference"
        if _is_tool_limitation(text):
            log.debug("fact rejected (tool limitation): %s", text[:60])
            return False, "tool limitation"
        norm = _normalize_for_dup(text)
        for f in self.data["facts"]:
            if f.get("text") == text:
                return False, "exact duplicate"
            if _normalize_for_dup(f.get("text", "")) == norm:
                log.debug("fact rejected (normalized dup of existing): %s",
                          text[:60])
                return False, "near-duplicate"
        self.data["facts"].append({
            "text": text,
            "source": source,
            "ts": datetime.now().isoformat(timespec="seconds"),
        })
        self.save()
        return True, ""

    def set_identity(self, key: str, value: str) -> None:
        key = key.strip()
        value = value.strip()
        if not key or not value:
            return
        if _has_negative_self(value):
            log.debug("identity rejected (negative self): %s=%s",
                      key, value[:60])
            return
        existing = self.data.setdefault("identity", {})
        norm = _normalize_for_dup(value)
        for k, v in existing.items():
            if k == key:
                continue
            if _normalize_for_dup(v) == norm:
                log.debug("identity rejected (cross-key dup of %s): %s=%s",
                          k, key, value[:60])
                return
        existing[key] = value
        self.save()

    def forget(self, keyword: str) -> int:
        keyword = keyword.strip().lower()
        if not keyword:
            return 0
        n = 0

        kept = []
        for f in self.data["facts"]:
            if keyword in f.get("text", "").lower():
                n += 1
            else:
                kept.append(f)
        self.data["facts"] = kept

        for k in list(self.data["identity"].keys()):
            v = str(self.data["identity"][k]).lower()
            if keyword in k.lower() or keyword in v:
                del self.data["identity"][k]
                n += 1

        kept_topics = []
        for t in self.data["recent_topics"]:
            text = t.get("text", "") if isinstance(t, dict) else str(t)
            if keyword in text.lower():
                n += 1
            else:
                kept_topics.append(t)
        self.data["recent_topics"] = kept_topics

        kept_projects = []
        for p in self.data["active_projects"]:
            text = p.get("path", "") if isinstance(p, dict) else str(p)
            if keyword in text.lower():
                n += 1
            else:
                kept_projects.append(p)
        self.data["active_projects"] = kept_projects

        if n > 0:
            self.save()
        return n

    # ---------- Profile ----------

    def set_profile(self, key: str, value: str) -> bool:
        key = (key or "").strip()
        value = (value or "").strip()
        if not key or not value:
            return False
        self.data.setdefault("profile", {})[key] = value
        self.save()
        return True

    def unset_profile(self, key: str) -> bool:
        key = (key or "").strip()
        if not key:
            return False
        prof = self.data.setdefault("profile", {})
        if key in prof:
            del prof[key]
            self.save()
            return True
        return False

    def get_profile(self) -> dict:
        return dict(self.data.get("profile", {}))

    def get_profile_prompt(self) -> str:
        """Return profile formatted as natural system-prompt text."""
        prof = self.data.get("profile", {})
        if not prof:
            return ""

        lines = []
        user_name = prof.get("user_name")
        ai_name = prof.get("ai_name")
        rel = prof.get("relationship")
        tone = prof.get("tone")
        lang = prof.get("language")

        if ai_name:
            lines.append(f'Your name is "{ai_name}".')
        if user_name:
            lines.append(f'The user is "{user_name}".')
        if rel:
            lines.append(f'Relationship: {rel}.')
        if tone:
            lines.append(f'Tone: {tone}.')
        if lang:
            lines.append(f'Preferred language: {lang}.')

        extra = {k: v for k, v in prof.items()
                 if k not in ("user_name", "ai_name", "relationship", "tone", "language")}
        for k, v in extra.items():
            lines.append(f"{k}: {v}.")

        return "\n".join(lines)

    def get_system_hint(self, max_chars: int = MAX_CONTEXT_CHARS) -> str:
        """Combined profile + context for system prompt."""
        parts = []
        prof = self.get_profile_prompt()
        if prof:
            parts.append(prof)
        ctx = self.get_context(max_chars=max_chars)
        if ctx:
            parts.append("Context:\n" + ctx)
        return "\n\n".join(parts)

    # ---------- Read ----------

    def get_context(self, max_chars: int = MAX_CONTEXT_CHARS) -> str:
        """Compact context for system prompt injection."""
        parts = []

        ident = self.data.get("identity", {})
        if ident:
            parts.append("Identity: " + "; ".join(f"{k}={v}" for k, v in ident.items()))

        projects = self.data.get("active_projects", [])
        if projects:
            items = []
            for p in projects[:3]:
                items.append(p.get("path", str(p)) if isinstance(p, dict) else str(p))
            parts.append("Projects: " + ", ".join(items))

        topics = self.data.get("recent_topics", [])
        if topics:
            items = []
            for t in topics[:5]:
                items.append(t.get("text", "") if isinstance(t, dict) else str(t))
            parts.append("Recent: " + "; ".join(items))

        facts = self.data.get("facts", [])
        if facts:
            items = [f.get("text", "") for f in facts[-10:]]
            parts.append("Facts: " + "; ".join(items))

        text = "\n".join(parts)
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        return text

    def show(self) -> str:
        lines = []

        prof = self.data.get("profile", {})
        if prof:
            lines.append("Profile:")
            for k, v in prof.items():
                lines.append(f"  {k}: {v}")

        ident = self.data.get("identity", {})
        if ident:
            lines.append("Identity:")
            for k, v in ident.items():
                lines.append(f"  {k}: {v}")

        projects = self.data.get("active_projects", [])
        if projects:
            lines.append("Projects:")
            for p in projects:
                lines.append(f"  - {p}")

        topics = self.data.get("recent_topics", [])
        if topics:
            lines.append("Recent topics:")
            for t in topics:
                lines.append(f"  - {t}")

        facts = self.data.get("facts", [])
        if facts:
            lines.append(f"Facts ({len(facts)}):")
            for f in facts:
                lines.append(f"  - {f.get('text', '')}")

        if not lines:
            return "(memory empty)"

        lines.append(f"\nUpdated: {self.data.get('last_updated', 'never')}")
        return "\n".join(lines)

    def stats(self) -> dict:
        return {
            "profile": len(self.data.get("profile", {})),
            "identity": len(self.data.get("identity", {})),
            "projects": len(self.data.get("active_projects", [])),
            "topics": len(self.data.get("recent_topics", [])),
            "facts": len(self.data.get("facts", [])),
        }
