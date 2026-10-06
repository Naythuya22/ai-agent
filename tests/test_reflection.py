"""Reflection loop unit tests."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from _harness import bootstrap

bootstrap()  # AGENT_REFLECT_ENABLED is False after this

# Redirect HOME BEFORE importing tools.memory (MEMORY_DIR is computed at import time)
_TMP = tempfile.mkdtemp(prefix="ai-agent-test-reflect-")
os.environ["HOME"] = _TMP

import memory as _mem_mod  # noqa: E402

_mem_mod.MEMORY_DIR = Path(_TMP) / ".ai-agent"
_mem_mod.MEMORY_FILE = _mem_mod.MEMORY_DIR / "memory.json"
_mem_mod.MEMORY_DIR.mkdir(parents=True, exist_ok=True)

from tools import memory as _tool_mem  # noqa: E402,F401

import config as cfg  # noqa: E402
from modes import agent as agent_mod  # noqa: E402


class _ReflectBase(unittest.TestCase):
    """Turn reflection ON for this suite; restore on the way out."""

    def setUp(self):
        self._saved = cfg.AGENT_REFLECT_ENABLED
        cfg.AGENT_REFLECT_ENABLED = True

    def tearDown(self):
        cfg.AGENT_REFLECT_ENABLED = self._saved

    def _store_bytes(self):
        p = _mem_mod.MEMORY_FILE
        return p.read_bytes() if p.exists() else None

    def _call_reflect(self, m):
        return m._reflect(
            user_message="hello",
            tool_names=["read_file"],
            error_count=0,
            final_text="done",
        )


def _bare_mode(*, reflect_reply):
    """Build an AgentMode without running __init__; stub the call.

    _reflect_call is the only LLM entry point — replace it with a
    function that returns a canned reply, so no network is opened
    and no backend is required.
    """
    m = agent_mod.AgentMode.__new__(agent_mod.AgentMode)
    m._reflect_call = lambda prompt: reflect_reply
    return m


class TestReflection(_ReflectBase):
    def test_reflect_disabled_returns_none(self):
        cfg.AGENT_REFLECT_ENABLED = False
        before = self._store_bytes()
        m = _bare_mode(reflect_reply="NOTE: should not save")
        self.assertIsNone(self._call_reflect(m))
        self.assertEqual(self._store_bytes(), before)

    def test_reflect_note_saves_and_returns_lesson(self):
        lesson = "read_file needs an absolute path"
        m = _bare_mode(reflect_reply="NOTE: " + lesson)
        self.assertEqual(self._call_reflect(m), lesson)
        after = self._store_bytes()
        self.assertIsNotNone(after)
        self.assertIn(lesson.encode(), after)

    def test_reflect_ok_saves_nothing(self):
        before = self._store_bytes()
        m = _bare_mode(reflect_reply="OK")
        self.assertIsNone(self._call_reflect(m))
        self.assertEqual(self._store_bytes(), before)

    def test_reflect_unexpected_reply_saves_nothing(self):
        before = self._store_bytes()
        m = _bare_mode(reflect_reply="sure thing!")
        self.assertIsNone(self._call_reflect(m))
        self.assertEqual(self._store_bytes(), before)

    def test_reflect_empty_reply_returns_none(self):
        m = _bare_mode(reflect_reply="")
        self.assertIsNone(self._call_reflect(m))

    def test_reflect_empty_note_returns_none(self):
        m = _bare_mode(reflect_reply="NOTE:")
        self.assertIsNone(self._call_reflect(m))

    def test_reflect_truncates_lesson_to_200_chars(self):
        m = _bare_mode(reflect_reply="NOTE: " + "x" * 400)
        result = self._call_reflect(m)
        self.assertEqual(result, "x" * 200)
        after = self._store_bytes()
        self.assertIsNotNone(after)
        self.assertIn(b"x" * 200, after)
        self.assertNotIn(b"x" * 201, after)

    def test_reflect_call_failure_returns_none(self):
        m = _bare_mode(reflect_reply="")

        def _boom(prompt):
            raise RuntimeError("llm unavailable")

        m._reflect_call = _boom
        self.assertIsNone(self._call_reflect(m))

    def test_finish_turn_swallows_reflect_errors(self):
        m = _bare_mode(reflect_reply="")
        m._turn_tools = []
        m._turn_errors = 0

        def _boom(**kwargs):
            raise RuntimeError("reflect blew up")

        m._reflect = _boom
        self.assertIsNone(m._finish_turn("hello", "done"))

    def test_finish_turn_passes_turn_state(self):
        m = _bare_mode(reflect_reply="")
        m._turn_tools = ["read_file", "web_search"]
        m._turn_errors = 2
        seen = {}

        def _rec(**kwargs):
            seen.update(kwargs)
            return None

        m._reflect = _rec
        self.assertIsNone(m._finish_turn("hello", "done"))
        self.assertEqual(seen.get("tool_names"), ["read_file", "web_search"])
        self.assertEqual(seen.get("error_count"), 2)
