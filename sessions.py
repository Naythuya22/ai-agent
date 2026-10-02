"""Session persistence v2 — multi-session per mode (Phase 3C-4)."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

from core.log import get_logger

log = get_logger("sessions")

SESS_DIR = Path.home() / ".ai-agent"
SESS_FILE = SESS_DIR / "sessions.json"
ARCHIVE_DIR = SESS_DIR / "archives"

MAX_MESSAGES = 40
MAX_TITLE = 40


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _new_id(mode: str) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    short = uuid.uuid4().hex[:4]
    return f"{mode}-{ts}-{short}"


class SessionStore:
    VERSION = 2

    def __init__(self, path: Path | None = None):
        self.path = path or SESS_FILE
        self.archive_dir = self.path.parent / "archives"

    def _empty(self) -> dict:
        return {"version": self.VERSION, "sessions": {},
                "active": {"chat": None, "agent": None}}

    def _load_raw(self) -> dict:
        if not self.path.exists():
            return self._empty()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            # Not fatal — but this is exactly the case where history vanishes
            # and nobody notices. Move the broken file aside before any later
            # save overwrites it, so the data is still recoverable by hand.
            log.error("session store unreadable (%s: %s) — starting empty",
                      type(e).__name__, e)
            self._quarantine_corrupt()
            return self._empty()
        if raw.get("version") != self.VERSION:
            raw = self._migrate_v1(raw)
            # Persist migration so IDs stay stable across reads
            try:
                self._save_all(raw)
            except Exception as e:
                log.warning("could not persist v1 session migration: %s: %s",
                            type(e).__name__, e)
        base = self._empty()
        if isinstance(raw.get("sessions"), dict):
            base["sessions"] = raw["sessions"]
        if isinstance(raw.get("active"), dict):
            for k in ("chat", "agent"):
                if k in raw["active"]:
                    base["active"][k] = raw["active"][k]
        return base

    def _migrate_v1(self, raw: dict) -> dict:
        try:
            if self.path.exists():
                bak = self.path.with_suffix(".json.bak.v1migration")
                bak.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            log.warning("could not back up pre-migration sessions file: %s: %s",
                        type(e).__name__, e)
        out = self._empty()
        for mode in ("chat", "agent"):
            msgs = raw.get(mode, [])
            if not msgs:
                continue
            sid = _new_id(mode)
            title = self._auto_title(msgs, mode)
            out["sessions"][sid] = {
                "mode": mode, "title": title,
                "created": _now(), "updated": _now(),
                "messages": msgs[-MAX_MESSAGES:],
            }
            out["active"][mode] = sid
        return out

    def _quarantine_corrupt(self) -> None:
        """Copy a broken store aside so a later save cannot destroy it."""
        try:
            if not self.path.exists():
                return
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            side = self.path.with_name(f"{self.path.name}.corrupt-{ts}")
            side.write_bytes(self.path.read_bytes())
            log.error("corrupt session store preserved at %s", side)
        except OSError as e:
            log.error("could not preserve corrupt session store %s: %s: %s",
                      self.path, type(e).__name__, e)

    def _save_all(self, data: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.path)
        except OSError as e:
            # Surface it (callers decide whether to fail the turn) but make
            # sure there is a record — a dropped save is otherwise invisible.
            log.error("session save failed for %s: %s: %s",
                      self.path, type(e).__name__, e)
            raise

    @staticmethod
    def _auto_title(messages: list, mode: str) -> str:
        for m in messages:
            if m.get("role") == "user":
                t = (m.get("text", "") or "").strip().replace("\n", " ")
                if t:
                    return t[:MAX_TITLE] + ("..." if len(t) > MAX_TITLE else "")
        return f"New {mode} chat"

    def list_sessions(self, mode: str | None = None) -> list:
        data = self._load_raw()
        out = []
        for sid, s in data["sessions"].items():
            if mode and s.get("mode") != mode:
                continue
            out.append({
                "id": sid, "mode": s.get("mode", "?"),
                "title": s.get("title", "Untitled"),
                "created": s.get("created", ""),
                "updated": s.get("updated", ""),
                "count": len(s.get("messages", [])),
            })
        out.sort(key=lambda x: x["updated"], reverse=True)
        return out

    def create_session(self, mode: str, title: str = "") -> str:
        data = self._load_raw()
        sid = _new_id(mode)
        data["sessions"][sid] = {
            "mode": mode, "title": title or f"New {mode} chat",
            "created": _now(), "updated": _now(), "messages": [],
        }
        data["active"][mode] = sid
        self._save_all(data)
        return sid

    def get_session(self, sid: str) -> dict | None:
        data = self._load_raw()
        s = data["sessions"].get(sid)
        return dict(s) if s else None

    def get_active_id(self, mode: str) -> str | None:
        data = self._load_raw()
        sid = data["active"].get(mode)
        if sid and sid in data["sessions"]:
            return sid
        return None

    def set_active(self, mode: str, sid: str) -> bool:
        data = self._load_raw()
        if sid not in data["sessions"]:
            return False
        if data["sessions"][sid].get("mode") != mode:
            return False
        data["active"][mode] = sid
        self._save_all(data)
        return True

    def append_message(self, sid: str, role: str, text: str) -> None:
        data = self._load_raw()
        s = data["sessions"].get(sid)
        if not s:
            return
        s.setdefault("messages", []).append({"role": role, "text": text})
        s["messages"] = s["messages"][-MAX_MESSAGES:]
        s["updated"] = _now()
        if s.get("title", "").startswith("New ") and role == "user":
            s["title"] = self._auto_title(s["messages"], s.get("mode", "chat"))
        self._save_all(data)

    def delete_session(self, sid: str) -> bool:
        data = self._load_raw()
        s = data["sessions"].pop(sid, None)
        if not s:
            return False
        mode = s.get("mode", "chat")
        if data["active"].get(mode) == sid:
            remaining = [k for k, v in data["sessions"].items() if v.get("mode") == mode]
            data["active"][mode] = remaining[0] if remaining else None
        self._save_all(data)
        return True

    def rename_session(self, sid: str, title: str) -> bool:
        data = self._load_raw()
        s = data["sessions"].get(sid)
        if not s:
            return False
        s["title"] = (title or "").strip()[:MAX_TITLE] or "Untitled"
        s["updated"] = _now()
        self._save_all(data)
        return True

    def get_mode(self, mode: str) -> list:
        sid = self.get_active_id(mode)
        if not sid:
            return []
        s = self.get_session(sid)
        return list(s.get("messages", [])) if s else []

    def save_mode(self, mode: str, messages: list) -> None:
        data = self._load_raw()
        sid = data["active"].get(mode)
        if not sid or sid not in data["sessions"]:
            sid = self.create_session(mode)
            data = self._load_raw()
        s = data["sessions"][sid]
        s["messages"] = messages[-MAX_MESSAGES:]
        s["updated"] = _now()
        if s.get("title", "").startswith("New "):
            s["title"] = self._auto_title(s["messages"], mode)
        self._save_all(data)

    def clear_mode(self, mode: str) -> None:
        self.create_session(mode)

    def archive_mode(self, mode: str) -> str | None:
        sid = self.get_active_id(mode)
        if not sid:
            self.create_session(mode)
            return None
        s = self.get_session(sid) or {}
        msgs = s.get("messages", [])
        if not msgs:
            self.create_session(mode)
            return None
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = self.archive_dir / f"{sid}-{ts}.json"
        try:
            out.write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError as e:
            log.error("could not archive session %s to %s: %s: %s",
                      sid, out, type(e).__name__, e)
            raise
        self.create_session(mode)
        return str(out)
