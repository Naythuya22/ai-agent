"""Release 26 — memory is no longer write-only (FIX 3, eval #3).

The eval: a fact saved by the memory tool reached memory.json, but
/api/memory served a stale long-lived instance and a NEW session did
not recall the fact. The fix is at the shared layer — Memory tracks
the file's (mtime_ns, size) stamp and refresh()es before every public
read and mutation — not per-route patches.

Pins:
  (a) a second instance's save is visible to the first instance's
      show()/get_context() without any reload call by the reader;
  (b) a stale instance's own mutation cannot clobber newer facts
      (refresh-before-save);
  (c) the recall path end to end: the real tool function
      (tools.memory.memory_write) → the routes-side instance's
      _with_memory injection for a new turn contains the fact;
  (d) auto-extraction discipline: a translation exercise no longer
      becomes a standing language fact, while genuine preferences
      still extract (and bare one-off mentions do not).

Run:  sh tests/run.sh -p "test_memory_refresh.py"
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from _harness import bootstrap

bootstrap()

import memory as mem_mod  # noqa: E402
from memory import Memory  # noqa: E402
from memory_extract import extract  # noqa: E402

FACT = "အကို ကော်ဖီ ကြိုက်တယ် (coffee at 7am)"


class TwoInstanceCase(unittest.TestCase):
    """The eval's topology: tools load per call, routes hold one."""

    def setUp(self):
        self.path = Path(tempfile.mkdtemp(prefix="mem-refresh-")) / "memory.json"
        self.routes_side = Memory.load(self.path)   # long-lived instance
        self.routes_side.save()

    def tool_save(self, text):
        tool_side = Memory.load(self.path)          # tools/memory._load()
        ok, reason = tool_side.add_fact(text, source="tool")
        assert ok, reason

    def test_tool_save_is_visible_to_the_long_lived_instance(self):
        self.assertNotIn(FACT, self.routes_side.show())
        self.tool_save(FACT)
        self.assertIn(FACT, self.routes_side.show())       # /api/memory
        self.assertIn(FACT, self.routes_side.get_context())  # injection

    def test_stale_instance_save_cannot_clobber_newer_facts(self):
        self.tool_save(FACT)
        ok, _ = self.routes_side.add_fact("second fact from routes side")
        self.assertTrue(ok)
        on_disk = Memory.load(self.path)
        texts = [f["text"] for f in on_disk.data["facts"]]
        self.assertIn(FACT, texts)
        self.assertIn("second fact from routes side", texts)

    def test_profile_set_by_tool_side_is_seen(self):
        tool_side = Memory.load(self.path)
        ok, _ = tool_side.set_profile("user_name", "KoHtet")
        self.assertTrue(ok)
        self.assertEqual(self.routes_side.get_profile().get("user_name"),
                         "KoHtet")
        self.assertIn("KoHtet", self.routes_side.get_system_hint())

    def test_unchanged_file_means_no_refresh(self):
        self.assertFalse(self.routes_side.refresh())
        self.tool_save(FACT)
        self.assertTrue(self.routes_side.refresh())
        self.assertFalse(self.routes_side.refresh())


class RecallInjectionCase(unittest.TestCase):
    """Save through the real tool path → a new turn's injected memory
    (AgentMode._with_memory, which the routes-side singleton feeds)
    contains the fact."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mem-recall-"))
        self._saved_file = mem_mod.MEMORY_FILE
        mem_mod.MEMORY_FILE = self.tmp / "memory.json"
        self.addCleanup(setattr, mem_mod, "MEMORY_FILE", self._saved_file)
        mem_mod.MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    def test_tool_write_reaches_new_turn_injection(self):
        from modes.agent import AgentMode
        from tools import memory as mem_tool

        routes_side = Memory.load()          # what _state.ensure_modes holds
        routes_side.save()

        class _LLM:
            _engine = None

        mode = AgentMode(_LLM(), routes_side)
        before = mode._with_memory("SYS")
        self.assertNotIn(FACT, before)

        out = mem_tool.memory_write(FACT)    # the tool path, fresh load
        self.assertNotIn("Error", out)

        after = mode._with_memory("SYS")     # a NEW session's first turn
        self.assertIn(FACT, after)
        self.assertIn(FACT, routes_side.show())


class ExtractionDisciplineCase(unittest.TestCase):
    """The junk fact from the eval — and the genuine cases around it."""

    def test_translation_exercise_saves_no_language(self):
        ex = extract("ဒီစာကြောင်းကို အင်္ဂလိပ်လို ဘာသာပြန်ပေးပါ: "
                     "'ကျွန်တော် ကော်ဖီ ကြိုက်တယ်'")
        self.assertNotIn("language", ex["profile"])

    def test_english_translation_request_saves_no_language(self):
        ex = extract("Translate this into Burmese: I like coffee")
        self.assertNotIn("language", ex["profile"])

    def test_bare_mention_is_not_a_preference(self):
        ex = extract("မြန်မာလို ရှင်းပြပါ")
        self.assertNotIn("language", ex["profile"])

    def test_genuine_burmese_preference_still_extracts(self):
        ex = extract("မြန်မာလိုပဲ ပြောပါ")
        self.assertEqual(ex["profile"].get("language"), "Burmese")

    def test_genuine_english_preference_still_extracts(self):
        ex = extract("Please speak in English")
        self.assertEqual(ex["profile"].get("language"), "English")


if __name__ == "__main__":
    unittest.main()
