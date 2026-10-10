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

    def test_write_rejects_negative_self_with_specific_reason(self):
        w = M.memory_write("ညီ မလုပ်နိုင်ပါ", "fact")
        self.assertEqual(w, "Error: fact rejected (negative self)")

    def test_write_rejects_near_duplicate_with_specific_reason(self):
        M.memory_write("User likes Python", "fact")
        w = M.memory_write("user likes python", "fact")
        self.assertEqual(w, "Error: fact rejected (near-duplicate)")

    def test_write_rejects_rule_reference_with_specific_reason(self):
        w = M.memory_write("follow rule_3 exactly", "fact")
        self.assertEqual(w, "Error: fact rejected (rule reference)")

    def test_write_rejects_tool_limitation_with_specific_reason(self):
        w = M.memory_write("tool web_search fails on some sites", "fact")
        self.assertEqual(w, "Error: fact rejected (tool limitation)")

    def test_write_exact_duplicate_still_says_duplicate(self):
        M.memory_write("beta", "fact")
        w = M.memory_write("beta", "fact")
        self.assertEqual(w, "Error: duplicate fact")

    def test_add_fact_returns_ok_and_reason(self):
        ok, reason = self.mem.add_fact("direct call works")
        self.assertTrue(ok)
        self.assertEqual(reason, "")
        ok2, reason2 = self.mem.add_fact("direct call works")
        self.assertFalse(ok2)
        self.assertEqual(reason2, "exact duplicate")
        ok3, reason3 = self.mem.add_fact("ညီ မရှိပါ")
        self.assertFalse(ok3)
        self.assertEqual(reason3, "negative self")

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


class ProfileValidationTests(_MemBase):
    """set_profile() must only accept known keys, bounded values, and
    never a negative self-statement — profile fields land in the system
    prompt."""

    def test_known_keys_are_accepted(self):
        for k in ("user_name", "ai_name", "relationship", "tone", "language"):
            ok, reason = self.mem.set_profile(k, "value-%s" % k)
            self.assertTrue(ok, msg="%s: %s" % (k, reason))
            self.assertEqual(reason, "")
        prof = self.mem.get_profile()
        self.assertEqual(len(prof), 5)

    def test_unknown_key_is_rejected(self):
        ok, reason = self.mem.set_profile("evil_key", "anything")
        self.assertFalse(ok)
        self.assertEqual(reason, "unknown key")
        self.assertNotIn("evil_key", self.mem.get_profile())

    def test_empty_key_or_value_is_rejected(self):
        self.assertEqual(self.mem.set_profile("", "x"), (False, "empty"))
        self.assertEqual(self.mem.set_profile("tone", ""), (False, "empty"))
        self.assertEqual(self.mem.set_profile("   ", "x"), (False, "empty"))
        self.assertEqual(self.mem.set_profile("tone", "   "), (False, "empty"))

    def test_too_long_value_is_rejected(self):
        ok, reason = self.mem.set_profile("tone", "x" * 101)
        self.assertFalse(ok)
        self.assertEqual(reason, "too long")
        ok2, reason2 = self.mem.set_profile("tone", "x" * 100)
        self.assertTrue(ok2, msg=reason2)

    def test_negative_self_value_is_rejected(self):
        ok, reason = self.mem.set_profile("tone", "ညီ မလုပ်နိုင်ပါ")
        self.assertFalse(ok)
        self.assertEqual(reason, "negative self")
        self.assertNotIn("tone", self.mem.get_profile())

    def test_set_profile_returns_ok_and_reason(self):
        ok, reason = self.mem.set_profile("user_name", "ကိုကြီး")
        self.assertTrue(ok)
        self.assertEqual(reason, "")
        # re-set same key overwrites, does not duplicate
        ok2, reason2 = self.mem.set_profile("user_name", "ကိုကြီး အသစ်")
        self.assertTrue(ok2, msg=reason2)
        self.assertEqual(self.mem.get_profile()["user_name"], "ကိုကြီး အသစ်")
