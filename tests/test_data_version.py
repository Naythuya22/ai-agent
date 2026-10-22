"""core/data_version.py + the two stores v1.10.0 brought under the rule.

memory / sessions / code projects / model limits already carried a
``"version"`` key. The server config did not, and the keys file had
the key in its defaults but never checked it on load. These tests
pin the shared helper (unversioned == 1, newer == warn and carry
on) and the keys-file behaviour: a newer file loads with a warning,
its keys survive, and no API payload ever exposes the version.

Run:  sh tests/run.sh -p "test_data_version.py"
"""
from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path

from _harness import bootstrap

bootstrap()

from core import data_version as DV  # noqa: E402
from core.key_manager import DATA_VERSION, DEFAULT_CONFIG, KeyManager  # noqa: E402


class TestFileVersion(unittest.TestCase):
    def test_unversioned_means_one(self):
        self.assertEqual(DV.file_version({}), 1)
        self.assertEqual(DV.file_version({"keys": {}}), 1)

    def test_explicit_version_is_read(self):
        self.assertEqual(DV.file_version({"version": 2}), 2)
        self.assertEqual(DV.file_version({"version": 99}), 99)

    def test_garbage_versions_fall_back_to_one(self):
        self.assertEqual(DV.file_version({"version": "3"}), 1)
        self.assertEqual(DV.file_version({"version": 0}), 1)
        self.assertEqual(DV.file_version({"version": True}), 1)
        self.assertEqual(DV.file_version([1, 2, 3]), 1)
        self.assertEqual(DV.file_version(None), 1)


class TestWarnIfNewer(unittest.TestCase):
    def setUp(self):
        self.log = logging.getLogger("test-data-version")

    def test_current_and_older_are_quiet(self):
        with self.assertNoLogs("test-data-version", level="WARNING"):
            self.assertEqual(DV.warn_if_newer("x.json", {}, 1, self.log), 1)
            self.assertEqual(
                DV.warn_if_newer("x.json", {"version": 1}, 1, self.log), 1)

    def test_newer_warns_and_reports_its_version(self):
        with self.assertLogs("test-data-version", level="WARNING") as cm:
            v = DV.warn_if_newer("x.json", {"version": 7}, 1, self.log)
        self.assertEqual(v, 7)
        self.assertIn("version 7", cm.output[0])


class TestKeyManagerVersioning(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp()) / "agent_config.json"
        self.km = KeyManager(self.path)

    def test_version_constant_matches_the_default_config(self):
        # DEFAULT_CONFIG has always carried "version": 1 — the stamp
        # constant must never drift away from it.
        self.assertEqual(DATA_VERSION, DEFAULT_CONFIG["version"])

    def test_save_stamps_the_version_into_the_file(self):
        self.km.add_key("openai", "sk-version-test-key")
        on_disk = json.loads(self.path.read_text())
        self.assertEqual(on_disk["version"], DATA_VERSION)
        self.assertNotIn("version", self.km.public())

    def test_unversioned_file_loads_as_version_one(self):
        self.path.write_text(json.dumps({"keys": {"openai": []}}))
        km = KeyManager(self.path)
        km.load()
        self.assertEqual(km.data_version, 1)

    def test_newer_file_loads_warns_and_restamps_on_save(self):
        self.km.add_key("openai", "sk-version-test-key")
        on_disk = json.loads(self.path.read_text())
        on_disk["version"] = 99
        self.path.write_text(json.dumps(on_disk))
        km = KeyManager(self.path)
        with self.assertLogs("aiagent.key_manager", level="WARNING"):
            km.load()
        self.assertEqual(km.data_version, 99)
        self.assertEqual(km.entries("openai")[0]["key"],
                         "sk-version-test-key")
        km.save()
        self.assertEqual(json.loads(self.path.read_text())["version"],
                         DATA_VERSION)
        self.assertEqual(km.entries("openai")[0]["key"],
                         "sk-version-test-key")


if __name__ == "__main__":
    unittest.main()
