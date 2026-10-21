"""shared_store_dir: /sdcard on Android, ~/.ai-agent everywhere else.

The Code-mode session stores hardcoded /sdcard/.ai-agent, which only
works where that path can be created: on the phone (a symlink into
shared storage) and — accidentally — on any machine whose tests run
as root. On a CI runner the mkdir is refused, the session writes were
silently dropped, and ~29 session tests failed while passing on every
developer machine. These tests pin the preference order with injected
directories so they mean the same thing on every machine.

Run:  sh tests/run.sh -p "test_store_paths.py"
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from _harness import bootstrap

bootstrap()

from core.store_paths import shared_store_dir


class TestSharedStoreDir(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ai-agent-store-"))
        self.home = self.tmp / "home-store"

    def test_existing_sdcard_dir_wins(self):
        sd = self.tmp / "sd" / ".ai-agent"
        sd.mkdir(parents=True)
        self.assertEqual(shared_store_dir(sd, self.home), sd)
        self.assertFalse(self.home.exists(), "fallback must stay untouched")

    def test_creatable_sdcard_dir_wins(self):
        sd = self.tmp / "sd" / ".ai-agent"
        self.assertEqual(shared_store_dir(sd, self.home), sd)
        self.assertTrue(sd.is_dir())

    def test_unusable_sdcard_falls_back_to_home(self):
        # A regular file where the parent dir would have to go makes
        # mkdir raise NotADirectoryError (an OSError) on every platform.
        blocker = self.tmp / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        sd = blocker / ".ai-agent"
        self.assertEqual(shared_store_dir(sd, self.home), self.home)
        self.assertTrue(self.home.is_dir())

    def test_real_call_sites_use_the_store(self):
        from core import opencode, openclaude
        self.assertEqual(opencode.session_file().name, "code_sessions.json")
        self.assertEqual(opencode.config_file().name, "opencode.json")
        self.assertEqual(openclaude.session_file().name,
                         "code_sessions_openclaude.json")
        self.assertEqual(opencode.session_file().parent,
                         openclaude.session_file().parent)


if __name__ == "__main__":
    unittest.main()
