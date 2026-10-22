"""Mid-turn checkpoints for Agent Mode (Release 19).

Sessions persist only at turn END (``AgentMode._save_turn``). If the
server dies mid-turn — Termux killed by Android, a crash, the battery
— the in-flight turn vanishes with it: the user's request and every
tool result so far are lost, and worse, side-effecting tools (file
writes) already ran with no record anywhere, so simply re-asking can
redo them. This store is the missing half: while a turn runs, the
agent loop snapshots its working transcript here after every tool
result; when a session is next loaded and a checkpoint is waiting,
the transcript is restored and the interruption is marked in it, so
the work resumes instead of restarting.

Shape (version 1) — ONE file under CONFIG_DIR, keyed by session id::

    {"version": 1,
     "checkpoints": {"agent-20261022-…": {
         "session_id": "agent-20261022-…",
         "user_text": "the in-flight request",
         "messages": [ …the working transcript, tool pairs included… ],
         "steps": 3,
         "saved_at": "2026-10-22T20:40:00"}},
     "updated": "2026-10-22T20:40:00"}

A single file (rather than one file per session) keeps the write to
one atomic tmp+replace — the usage_store pattern — and needs no
filename hygiene for session ids. At most one checkpoint per session
exists: a new snapshot overwrites the session's previous one, a
completed turn deletes it, and a restore consumes it. An entry whose
session is deleted without ever being loaded again simply stays
inert — it is keyed by an id nothing will ever look up again — until
some later write of the file carries it along; entries are small
(the transcript is the same bounded list the session store keeps).

Every module-level operation re-reads the file (the terminal app and
the web server are separate processes sharing it) and NEVER raises:
a checkpoint failure must not be able to affect the turn it only
describes.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

import config
from core import data_version
from core.log import get_logger

log = get_logger("checkpoint_store")

VERSION = 1
FILE_NAME = "checkpoints.json"

_lock = threading.Lock()


def default_path() -> Path:
    """Where the checkpoints live, resolved per call like the usage
    ledger's: tests re-point config.CONFIG_DIR after import, and a
    stale import-time path would write into the previous run's dir."""
    return Path(config.CONFIG_DIR) / FILE_NAME


def _stamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _quarantine(path: Path) -> None:
    """Copy a broken checkpoint file aside so a later save cannot
    destroy it (same rule as the session/usage stores)."""
    try:
        if not path.exists():
            return
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        side = path.with_name(f"{path.name}.corrupt-{ts}")
        side.write_bytes(path.read_bytes())
        log.error("corrupt checkpoint store preserved at %s", side)
    except OSError as e:
        log.error("could not preserve corrupt checkpoint store %s: %s: %s",
                  path, type(e).__name__, e)


def _valid_entry(entry) -> bool:
    """A checkpoint is resumable only in the shape save() writes:
    a dict whose ``messages`` is a non-empty list of message dicts
    with string roles, and whose stored session id — when present —
    matches the key it was filed under (checked by the caller)."""
    if not isinstance(entry, dict):
        return False
    msgs = entry.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return False
    return all(isinstance(m, dict) and isinstance(m.get("role"), str)
               for m in msgs)


class CheckpointStore:
    """Load/save for checkpoints.json. Mirrors UsageStore's IO shape."""

    VERSION = VERSION

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path is not None else default_path()
        self.data = self._empty()

    @staticmethod
    def _empty() -> dict:
        return {"version": VERSION, "checkpoints": {}, "updated": ""}

    # ---------- IO ----------

    @classmethod
    def load(cls, path: Path | None = None) -> "CheckpointStore":
        store = cls(path)
        if not store.path.exists():
            return store
        try:
            raw = json.loads(store.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            # A checkpoint file that cannot be read means "no resume
            # available" — but say so, and keep the bytes: silently
            # starting empty would let the next save destroy the only
            # copy of an interrupted turn.
            log.error("checkpoint store unreadable (%s: %s) — "
                      "starting empty", type(e).__name__, e)
            _quarantine(store.path)
            return store
        if not isinstance(raw, dict):
            log.error("checkpoint store is not a JSON object — "
                      "starting empty")
            _quarantine(store.path)
            return store
        data_version.warn_if_newer("checkpoint store", raw, VERSION, log)
        merged = cls._empty()
        if isinstance(raw.get("checkpoints"), dict):
            merged["checkpoints"] = raw["checkpoints"]
        if isinstance(raw.get("updated"), str):
            merged["updated"] = raw["updated"]
        store.data = merged
        return store

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.data["version"] = VERSION
            self.data["updated"] = _stamp()
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(self.data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(self.path)
        except OSError as e:
            log.error("checkpoint save failed for %s: %s: %s",
                      self.path, type(e).__name__, e)
            raise

    # ---------- mutation / access ----------

    def put(self, session_id: str, user_text: str, messages: list,
            steps: int) -> None:
        """File (or re-file) the snapshot for one session."""
        session_id = (session_id or "").strip()
        if not session_id:
            return
        self.data.setdefault("checkpoints", {})[session_id] = {
            "session_id": session_id,
            "user_text": user_text or "",
            "messages": list(messages),
            "steps": int(steps or 0),
            "saved_at": _stamp(),
        }

    def get(self, session_id: str) -> dict | None:
        """The checkpoint for ``session_id``, or None when there is
        none — or when what is filed under that id is not a resumable
        checkpoint (malformed entries are invisible here and dropped
        by take_checkpoint)."""
        session_id = (session_id or "").strip()
        if not session_id:
            return None
        entry = (self.data.get("checkpoints") or {}).get(session_id)
        if not _valid_entry(entry):
            return None
        if entry.get("session_id") not in (None, session_id):
            return None
        return entry

    def discard(self, session_id: str) -> bool:
        """Drop one session's checkpoint; True when one was filed."""
        session_id = (session_id or "").strip()
        if not session_id:
            return False
        return self.data.setdefault("checkpoints", {}).pop(
            session_id, None) is not None


# ── module-level operations (what modes/agent.py calls) ──
# All three NEVER raise: they are called from inside a running turn
# (save/clear) and from session load (take), and none of those may be
# disturbed by a checkpoint problem.

def save_checkpoint(session_id: str, user_text: str, messages: list,
                    steps: int) -> bool:
    """Snapshot one session's in-flight turn. True when it landed."""
    try:
        if not (session_id or "").strip():
            return False
        with _lock:
            store = CheckpointStore.load()
            store.put(session_id, user_text, messages, steps)
            store.save()
        return True
    except Exception as e:
        log.warning("checkpoint save failed: %s: %s", type(e).__name__, e)
        return False


def take_checkpoint(session_id: str) -> dict | None:
    """Load AND consume the checkpoint for ``session_id``.

    Consuming on read is what makes a resume happen exactly once: a
    second load finds nothing and proceeds normally. A malformed
    entry under the id is dropped as well — it can never become
    resumable, and leaving it would re-check the same broken bytes
    on every start.
    """
    try:
        if not (session_id or "").strip():
            return None
        with _lock:
            store = CheckpointStore.load()
            entry = store.get(session_id)
            if store.discard(session_id):
                store.save()
            return entry
    except Exception as e:
        log.warning("checkpoint take failed: %s: %s", type(e).__name__, e)
        return None


def clear_checkpoint(session_id: str) -> bool:
    """Drop one session's checkpoint (its turn completed). True when
    one was actually filed."""
    try:
        if not (session_id or "").strip():
            return False
        with _lock:
            store = CheckpointStore.load()
            if store.discard(session_id):
                store.save()
                return True
        return False
    except Exception as e:
        log.warning("checkpoint clear failed: %s: %s", type(e).__name__, e)
        return False
