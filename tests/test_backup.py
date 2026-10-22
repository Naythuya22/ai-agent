"""backup.py — export / list / import of the user's irreplaceable data.

Everything runs against throwaway roots: ``do_export``/``do_import`` take
the three data roots as an argument precisely so no test ever touches the
real ``~/.ai-agent`` or the live keys file.

Run:  sh tests/run.sh -p "test_backup.py"
"""
from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

from _harness import bootstrap

bootstrap()

import backup as BK  # noqa: E402


def stage(root: Path) -> dict[str, Path]:
    """A miniature data tree covering all three roots."""
    home = root / "home" / ".ai-agent"
    (home / "skills").mkdir(parents=True)
    (home / "logs").mkdir()
    (home / "memory.json").write_text('{"facts": ["a"]}')
    (home / "sessions.json").write_text('{"sessions": []}')
    (home / "auth_token").write_text("tok123")
    (home / "skills" / "demo.md").write_text("skill")
    (home / "logs" / "agent.log").write_text("noise")
    (home / "junk.pyc").write_bytes(b"\x00")
    data = root / "project" / "data"
    data.mkdir(parents=True)
    (data / "agent_config.json").write_text('{"openai": {"keys": ["k"]}}')
    sdcard = root / "sdcard" / ".ai-agent"
    sdcard.mkdir(parents=True)
    (sdcard / "code_sessions.json").write_text("{}")
    return {"home": home, "data": data, "sdcard": sdcard}


class BackupCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ai-agent-backup-"))
        self.roots = stage(self.tmp)
        self.out = self.tmp / "out.zip"

    def export(self, **kw) -> Path:
        return BK.do_export(self.out, roots=self.roots, **kw)


class TestExport(BackupCase):
    def test_covers_all_roots_and_skips_noise(self):
        path = self.export()
        manifest = BK.read_manifest(path)
        entries = {f["entry"] for f in manifest["files"]}
        self.assertIn("home/memory.json", entries)
        self.assertIn("home/sessions.json", entries)
        self.assertIn("home/skills/demo.md", entries)
        self.assertIn("project-data/agent_config.json", entries)
        self.assertIn("sdcard/code_sessions.json", entries)
        self.assertFalse(any("logs/" in e for e in entries))
        self.assertFalse(any(e.endswith(".pyc") for e in entries))
        self.assertEqual(manifest["format"], 1)
        self.assertTrue(manifest["includes_secrets"])

    def test_secret_backup_is_0600(self):
        path = self.export()
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_no_secrets_excludes_keys_and_token(self):
        path = self.export(include_secrets=False)
        manifest = BK.read_manifest(path)
        entries = {f["entry"] for f in manifest["files"]}
        self.assertNotIn("project-data/agent_config.json", entries)
        self.assertNotIn("home/auth_token", entries)
        self.assertIn("home/memory.json", entries)
        self.assertFalse(manifest["includes_secrets"])

    def test_checksums_match_contents(self):
        path = self.export()
        manifest = BK.read_manifest(path)
        with zipfile.ZipFile(path) as z:
            for item in manifest["files"]:
                self.assertEqual(BK._sha256(z.read(item["entry"])),
                                 item["sha256"])


class TestImport(BackupCase):
    def test_round_trip_restores_changed_and_deleted_files(self):
        path = self.export()
        memory = self.roots["home"] / "memory.json"
        sessions = self.roots["home"] / "sessions.json"
        memory.write_text('{"facts": ["changed"]}')
        sessions.unlink()
        result = BK.do_import(path, roots=self.roots,
                              snapshot_dir=self.tmp)
        self.assertEqual(memory.read_text(), '{"facts": ["a"]}')
        self.assertEqual(sessions.read_text(), '{"sessions": []}')
        self.assertGreaterEqual(result["restored"], 5)
        self.assertTrue(Path(result["snapshot"]).exists())

    def test_tampered_backup_is_refused_and_disk_untouched(self):
        path = self.export()
        evil = self.tmp / "evil.zip"
        with zipfile.ZipFile(path) as zin, \
                zipfile.ZipFile(evil, "w") as zout:
            for name in zin.namelist():
                data = zin.read(name)
                if name == "home/memory.json":
                    data = b'{"facts": ["forged"]}'
                zout.writestr(name, data)
        memory = self.roots["home"] / "memory.json"
        before = memory.read_text()
        with self.assertRaises(BK.BackupError):
            BK.do_import(evil, roots=self.roots, snapshot_dir=self.tmp)
        self.assertEqual(memory.read_text(), before)

    def test_escaping_entry_is_refused(self):
        evil = self.tmp / "escape.zip"
        manifest = {"format": 1, "app": "ai-agent",
                    "created_at": "2026-01-01T00:00:00",
                    "includes_secrets": False,
                    "files": [{"entry": "home/../../evil.txt", "size": 1,
                               "sha256": BK._sha256(b"x")}]}
        with zipfile.ZipFile(evil, "w") as z:
            z.writestr("manifest.json", json.dumps(manifest))
            z.writestr("home/../../evil.txt", b"x")
        with self.assertRaises(BK.BackupError):
            BK.do_import(evil, roots=self.roots, snapshot_dir=self.tmp)
        self.assertFalse((self.tmp / "evil.txt").exists())

    def test_non_backup_file_is_refused(self):
        junk = self.tmp / "junk.zip"
        junk.write_bytes(b"not a zip at all")
        with self.assertRaises(BK.BackupError):
            BK.read_manifest(junk)


class TestAutoBackup(BackupCase):
    """maybe_auto_backup: freshness gate, creation, retention pruning."""

    NOW = datetime(2026, 10, 22, 12, 0, 0)

    def setUp(self):
        super().setUp()
        self.home_dir = self.tmp / "fakehome"
        self.home_dir.mkdir()

    def _fake_backup(self, name: str, age_days: float) -> Path:
        p = self.home_dir / name
        p.write_bytes(b"old backup bytes")
        ts = (self.NOW - timedelta(days=age_days)).timestamp()
        os.utime(p, (ts, ts))
        return p

    def _auto(self, **kw):
        return BK.maybe_auto_backup(self.NOW, home_dir=self.home_dir,
                                    roots=self.roots, **kw)

    def _names(self) -> list[str]:
        return sorted(p.name for p in
                      self.home_dir.glob("ai-agent-backup-*.zip"))

    def test_no_existing_backup_creates_one(self):
        made = self._auto()
        self.assertIsNotNone(made)
        self.assertTrue(made.exists())
        self.assertEqual(made.parent, self.home_dir)
        manifest = BK.read_manifest(made)
        entries = {f["entry"] for f in manifest["files"]}
        self.assertIn("home/memory.json", entries)
        self.assertIn("project-data/agent_config.json", entries)

    def test_fresh_backup_is_left_alone(self):
        self._fake_backup("ai-agent-backup-20261021-120000.zip", 1)
        self.assertIsNone(self._auto())
        self.assertEqual(self._names(),
                         ["ai-agent-backup-20261021-120000.zip"])

    def test_stale_backup_triggers_a_new_one(self):
        self._fake_backup("ai-agent-backup-20261014-120000.zip", 8)
        made = self._auto()
        self.assertIsNotNone(made)
        self.assertEqual(len(self._names()), 2)

    def test_just_under_a_week_is_still_fresh(self):
        self._fake_backup("ai-agent-backup-20261015-130000.zip",
                          7 - 1 / 24)
        self.assertIsNone(self._auto())

    def test_prunes_to_newest_four_and_spares_snapshots(self):
        seeded = [self._fake_backup(
            f"ai-agent-backup-202610{10 + i:02d}-120000.zip", 12 - i)
            for i in range(5)]
        snap = self.home_dir / "ai-agent-pre-restore-20261001-000000.zip"
        snap.write_bytes(b"safety snapshot")
        made = self._auto()
        self.assertIsNotNone(made)
        names = self._names()
        self.assertEqual(len(names), 4)
        self.assertIn(made.name, names)
        self.assertNotIn(seeded[0].name, names)  # oldest pruned
        self.assertNotIn(seeded[1].name, names)  # second-oldest pruned
        self.assertTrue(snap.exists())           # snapshot untouched


if __name__ == "__main__":
    unittest.main()
