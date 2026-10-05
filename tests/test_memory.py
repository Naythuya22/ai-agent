"""Memory tool unit tests."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

from _harness import bootstrap

bootstrap()

# Redirect HOME BEFORE importing tools.memory (MEMORY_DIR is computed at import time)
_TMP = tempfile.mkdtemp(prefix="ai-agent-test-memory-")
os.environ["HOME"] = _TMP

import memory as _mem_mod  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_mem_mod.MEMORY_DIR = _Path(_TMP) / ".ai-agent"
_mem_mod.MEMORY_FILE = _mem_mod.MEMORY_DIR / "memory.json"

from tools import memory as M  # noqa: E402


class _MemBase(unittest.TestCase):
    def setUp(self):
        p = _mem_mod.MEMORY_FILE
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            p.unlink()
        self.mem = _mem_mod.Memory()
        self.mem.save()

    def tearDown(self):
        p = _mem_mod.MEMORY_FILE
        if p.exists():
            p.unlink()


class MemoryToolTests(_MemBase):
    def test_read_empty_returns_string(self):
        r = M.memory_read()
        self.assertIsInstance(r, str)
        # fresh store has no identity entries yet, so _compact skips the
        # section header entirely. Only facts/projects/topics always print.
        self.assertIn("facts: 0", r)

    def test_write_fact_and_read_back(self):
        w = M.memory_write("user prefers Burmese", "fact")
        self.assertIsInstance(w, str)
        self.assertTrue(w.startswith("Wrote fact:"))
        r = M.memory_read()
        self.assertIn("user prefers Burmese", r)

    def test_write_topic_and_read_back(self):
        w = M.memory_write("debugging streaming", "topic")
        self.assertIsInstance(w, str)
        self.assertTrue(w.startswith("Wrote topic:"))
        r = M.memory_read()
        self.assertIn("debugging streaming", r)

    def test_write_project_and_read_back(self):
        w = M.memory_write("/tmp/proj", "project")
        self.assertIsInstance(w, str)
        self.assertTrue(w.startswith("Wrote project:"))
        r = M.memory_read()
        self.assertIn("/tmp/proj", r)

    def test_write_identity_increments_rule_key(self):
        w1 = M.memory_write("be concise", "identity")
        w2 = M.memory_write("use Burmese if asked", "identity")
        self.assertIsInstance(w1, str)
        self.assertIsInstance(w2, str)
        self.assertTrue(w2.startswith("Wrote identity:"))
        m = _mem_mod.Memory.load()
        self.assertIn("rule_0", m.data.get("identity", {}))
        self.assertIn("rule_1", m.data.get("identity", {}))

    def test_write_rejects_empty(self):
        w1 = M.memory_write("", "fact")
        w2 = M.memory_write("   ", "fact")
        self.assertIsInstance(w1, str)
        self.assertIsInstance(w2, str)
        self.assertTrue(w1.startswith("Error:"))
        self.assertTrue(w2.startswith("Error:"))

    def test_write_rejects_too_long(self):
        w = M.memory_write("x" * 501, "fact")
        self.assertIsInstance(w, str)
        self.assertTrue(w.startswith("Error:"))
        self.assertIn("too long", w)
        self.assertIn("500", w)

    def test_write_rejects_unknown_category(self):
        w = M.memory_write("hi", "banana")
        self.assertIsInstance(w, str)
        self.assertTrue(w.startswith("Error: unknown category"))

    def test_write_rejects_duplicate_fact(self):
        M.memory_write("alpha", "fact")
        w2 = M.memory_write("alpha", "fact")
        self.assertIsInstance(w2, str)
        self.assertTrue(w2.startswith("Error: duplicate"))

    def test_search_and_forget_roundtrip(self):
        M.memory_write("alpha", "topic")
        M.memory_write("beta", "topic")
        M.memory_write("gamma", "topic")
        s = M.memory_search("alpha")
        self.assertIsInstance(s, str)
        self.assertIn("alpha", s)
        self.assertNotIn("beta", s)
        self.assertNotIn("gamma", s)
        f = M.memory_forget("beta")
        self.assertIsInstance(f, str)
        self.assertIn("Removed 1", f)
        r = M.memory_read()
        self.assertNotIn("beta", r)
        self.assertIn("alpha", r)
        self.assertIn("gamma", r)
        fn = M.memory_forget("nope")
        self.assertIsInstance(fn, str)
        self.assertEqual("no matches", fn)
