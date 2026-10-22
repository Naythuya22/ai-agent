"""Tests for the model-aware read_file char cap (Batch 1.5)."""
import os
import sys
import unittest
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import config
from tools import file as file_mod


class _CtxCase(unittest.TestCase):
    def tearDown(self):
        file_mod.set_read_context()   # reset


class CapFormulaTests(_CtxCase):
    def test_local_model_floors_at_4000(self):
        file_mod.set_read_context(model="gemma:4b", base_url="",
                                  used_tokens=0)
        self.assertEqual(file_mod._max_chars_for_read(),
                         config.AGENT_READ_FLOOR)

    def test_deepseek_empty_context_allows_more(self):
        file_mod.set_read_context(model="deepseek-flash", base_url="",
                                  used_tokens=0)
        n = file_mod._max_chars_for_read()
        self.assertGreater(n, 20_000)
        self.assertLessEqual(n, config.AGENT_READ_CEILING)

    def test_gpt4o_empty_context_allows_more(self):
        file_mod.set_read_context(model="gpt-4o", base_url="",
                                  used_tokens=0)
        self.assertGreater(file_mod._max_chars_for_read(), 40_000)

    def test_gemini_capped_at_ceiling(self):
        file_mod.set_read_context(model="gemini-1.5-pro", base_url="",
                                  used_tokens=0)
        self.assertEqual(file_mod._max_chars_for_read(),
                         config.AGENT_READ_CEILING)

    def test_usage_reduces_budget(self):
        file_mod.set_read_context(model="deepseek-flash", base_url="",
                                  used_tokens=0)
        full = file_mod._max_chars_for_read()
        file_mod.set_read_context(model="deepseek-flash", base_url="",
                                  used_tokens=30_000)
        tight = file_mod._max_chars_for_read()
        self.assertLess(tight, full)

    def test_floor_when_full(self):
        file_mod.set_read_context(model="deepseek-flash", base_url="",
                                  used_tokens=64_000)
        self.assertEqual(file_mod._max_chars_for_read(),
                         config.AGENT_READ_FLOOR)

    def test_flag_off_returns_floor(self):
        saved = config.AGENT_ADAPTIVE_READ
        try:
            config.AGENT_ADAPTIVE_READ = False
            file_mod.set_read_context(model="gpt-4o", base_url="",
                                      used_tokens=0)
            self.assertEqual(file_mod._max_chars_for_read(),
                             config.AGENT_READ_FLOOR)
        finally:
            config.AGENT_ADAPTIVE_READ = saved

    def test_unknown_model_defaults_safely(self):
        file_mod.set_read_context(model="some-unknown-xl", base_url="",
                                  used_tokens=0)
        n = file_mod._max_chars_for_read()
        self.assertGreaterEqual(n, config.AGENT_READ_FLOOR)
        self.assertLessEqual(n, config.AGENT_READ_CEILING)


class ReadFileTruncationTests(_CtxCase):
    def setUp(self):
        self.tmp = Path.home() / ".ai-agent-test-big.txt"
        self.tmp.write_text("\n".join(f"line {i}" for i in range(1, 2001)),
                            encoding="utf-8")
        # Release 16: read_file is jailed to config.PROJECT_ROOT when no
        # project is active, and this case's fixture lives under HOME —
        # point the root resolver at HOME for the case (tools/file.py
        # reads config.PROJECT_ROOT per call, so the patch is the seam).
        self._saved_root = config.PROJECT_ROOT
        config.PROJECT_ROOT = Path.home()

    def tearDown(self):
        config.PROJECT_ROOT = self._saved_root
        self.tmp.unlink(missing_ok=True)
        file_mod.set_read_context()

    def test_truncation_notice_has_next_line(self):
        file_mod.set_read_context(model="gemma:4b", base_url="",
                                  used_tokens=0)
        out = file_mod.read_file(str(self.tmp), start_line=1, end_line=500)
        self.assertIn("OUTPUT TRUNCATED", out)
        self.assertIn("start_line=", out)

    def test_no_notice_when_under_cap(self):
        file_mod.set_read_context(model="gemini-1.5-pro", base_url="",
                                  used_tokens=0)
        out = file_mod.read_file(str(self.tmp), start_line=1, end_line=10)
        self.assertNotIn("OUTPUT TRUNCATED", out)


if __name__ == "__main__":
    unittest.main()
