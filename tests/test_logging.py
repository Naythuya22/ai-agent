"""Logging + silent-failure tests.

Before core.log existed the tree had no `logging` at all: 173 `except Exception`
blocks swallowed everything, so a session or memory save that failed left no
trace. These tests pin the new guarantees:

  * records reach `<CONFIG_DIR>/logs/agent.log`,
  * a broken store is moved aside instead of being overwritten,
  * a failed save is logged before it propagates,
  * logging itself can never take the app down (read-only HOME, full disk).
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path

from _harness import bootstrap, current_home

home = bootstrap()

import core.log as clog  # noqa: E402
import config  # noqa: E402
from core.log import get_logger, log_path  # noqa: E402


def _flush() -> None:
    for handler in get_logger().handlers:
        try:
            handler.flush()
        except Exception:
            pass


def _log_text() -> str:
    _flush()
    try:
        return log_path().read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


class LogCase(unittest.TestCase):
    """Re-points the logger at this test's HOME before and after each test."""

    def setUp(self):
        self._cfg_dir = config.CONFIG_DIR
        self._cfg_file = config.CONFIG_FILE
        config.CONFIG_DIR = current_home() / ".ai-agent"
        config.CONFIG_FILE = config.CONFIG_DIR / "config.json"
        clog.reset_for_tests()
        self.addCleanup(self._restore)

    def _restore(self):
        config.CONFIG_DIR = self._cfg_dir
        config.CONFIG_FILE = self._cfg_file
        clog.reset_for_tests()


class TestLogFile(LogCase):
    def test_records_land_in_the_config_dir_log(self):
        marker = "memory-save-marker-xyz"
        get_logger("tests").warning("%s", marker)
        self.assertIn(marker, _log_text())

    def test_log_lives_next_to_the_other_state(self):
        self.assertEqual(log_path(), config.CONFIG_DIR / "logs" / "agent.log")
        self.assertTrue(log_path().is_file())

    def test_tail_returns_recent_lines(self):
        for i in range(5):
            get_logger("tests").info("tail-line-%d", i)
        lines = clog.tail(3)
        self.assertEqual(len(lines), 3)
        self.assertIn("tail-line-4", lines[-1])
        self.assertIn("tail-line-2", lines[0])

    def test_child_loggers_share_the_one_file(self):
        get_logger("sessions").error("from-sessions")
        get_logger("mode.chat").error("from-chat")
        text = _log_text()
        self.assertIn("from-sessions", text)
        self.assertIn("from-chat", text)

    def test_an_unusable_log_dir_never_raises(self):
        """Read-only or nonsensical HOME must not break the app on import."""
        blocker = current_home() / "not-a-dir"
        blocker.write_text("x", encoding="utf-8")
        config.CONFIG_DIR = blocker / "logs"  # parent is a file -> OSError
        clog.reset_for_tests()
        try:
            get_logger("tests").error("still must not raise")
        finally:
            config.CONFIG_DIR = self._cfg_dir
            config.CONFIG_FILE = config.CONFIG_DIR / "config.json"
            clog.reset_for_tests()


class TestCorruptStoreIsKept(LogCase):
    """A store we cannot read must survive, not be silently overwritten."""

    def _store_path(self) -> Path:
        d = config.CONFIG_DIR
        d.mkdir(parents=True, exist_ok=True)
        return d / "sessions.json"

    def test_corrupt_sessions_file_is_quarantined(self):
        from sessions import SessionStore

        path = self._store_path()
        path.write_text("{ this is not json", encoding="utf-8")

        store = SessionStore(path=path)
        data = store._load_raw()

        self.assertEqual(data["sessions"], {}, "must fall back to empty")
        sidecars = list(path.parent.glob("sessions.json.corrupt-*"))
        self.assertEqual(len(sidecars), 1, "broken file must be preserved")
        self.assertEqual(sidecars[0].read_text(encoding="utf-8"),
                         "{ this is not json")
        self.assertIn("session store unreadable", _log_text())

    def test_corrupt_memory_file_is_quarantined(self):
        from memory import Memory

        d = config.CONFIG_DIR
        d.mkdir(parents=True, exist_ok=True)
        path = d / "memory.json"
        path.write_text("]broken[", encoding="utf-8")

        m = Memory.load(path)

        self.assertEqual(m.data["facts"], [])
        sidecars = list(d.glob("memory.json.corrupt-*"))
        self.assertEqual(len(sidecars), 1)
        self.assertEqual(sidecars[0].read_text(encoding="utf-8"), "]broken[")
        self.assertIn("memory store unreadable", _log_text())


class TestSaveFailuresAreLogged(LogCase):
    """The whole point: a lost save has to leave a record."""

    def test_memory_save_failure_is_logged_before_raising(self):
        from memory import Memory

        # Parent is a regular file, so mkdir() raises OSError on save().
        blocker = current_home() / "blocker"
        blocker.write_text("x", encoding="utf-8")
        m = Memory(path=blocker / "memory.json")

        with self.assertRaises(OSError):
            m.save()

        self.assertIn("memory save failed", _log_text())

    def test_session_save_failure_is_logged_before_raising(self):
        from sessions import SessionStore

        blocker = current_home() / "blocker2"
        blocker.write_text("x", encoding="utf-8")
        store = SessionStore(path=blocker / "sessions.json")
        store._load_raw()  # fine — file does not exist yet

        with self.assertRaises(OSError):
            store._save_all({"version": 2, "sessions": {}, "active": {}})

        self.assertIn("session save failed", _log_text())

    def test_a_failing_repl_save_is_visible_in_the_log(self):
        """The path app.py's /logs command reads must contain the failure."""
        from modes import chat as chat_mod  # noqa: F401  (ensures import works)
        from memory import Memory

        blocker = current_home() / "blocker3"
        blocker.write_text("x", encoding="utf-8")
        m = Memory(path=blocker / "memory.json")
        try:
            m.add_fact("should not persist")
        except OSError:
            pass
        self.assertIn("memory save failed", _log_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
