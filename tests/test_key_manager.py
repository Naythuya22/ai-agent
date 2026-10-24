"""core.key_manager — key storage, rotation state, and the two safety rules:
the file is 0600 and no raw key ever leaves the process in a serialised form.

Run:  sh tests/run.sh -p "test_key_manager.py"
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from datetime import datetime, timezone
from unittest import mock
from pathlib import Path

from _harness import bootstrap

bootstrap()

from core import key_manager as KM  # noqa: E402
from core.key_manager import KeyManager, mask, redact  # noqa: E402


class KeyManagerTest(unittest.TestCase):
    def setUp(self):
        # The temp HOME is shared for the whole module, so every test starts
        # from an empty config instead of inheriting the previous test's keys.
        self.km = KeyManager()
        if self.km.path.exists():
            self.km.path.unlink()

    def test_mask_hides_the_middle_and_keeps_recognition(self):
        m = mask("sk-live-abcdefgh1234")
        self.assertNotIn("abcdefgh", m)
        self.assertTrue(m.startswith("sk-l"))
        self.assertTrue(m.endswith("1234"))
        self.assertEqual(mask(""), "")
        self.assertEqual(mask("short"), "*****")

    def test_redact_strips_every_known_key_from_text(self):
        k = "sk-live-supersecret-9999"
        self.assertNotIn(k, redact("error: bad key %s rejected" % k, [k]))
        self.assertIn("rejected", redact("error: bad key %s rejected" % k, [k]))
        self.assertEqual(redact(None, [k]), None)

    def test_file_is_created_with_0600(self):
        self.km.add_key("openai", "sk-file-mode-00000000")
        self.assertTrue(self.km.path.exists())
        self.assertEqual(self.km.path.stat().st_mode & 0o777, 0o600)

    def test_a_loose_file_is_tightened_on_read(self):
        self.km.add_key("openai", "sk-loose-mode-11111111")
        os.chmod(str(self.km.path), 0o644)
        fresh = KeyManager()
        fresh.load()
        self.assertEqual(fresh.path.stat().st_mode & 0o777, 0o600)

    def test_add_is_idempotent(self):
        self.km.add_key("openai", "sk-dup-dup-dup-dup", "a")
        self.km.add_key("openai", "sk-dup-dup-dup-dup", "b")
        self.assertEqual(len(self.km.entries("openai")), 1)

    def test_add_validates_provider_and_length(self):
        with self.assertRaises(ValueError):
            self.km.add_key("nope", "sk-abcdefgh")
        with self.assertRaises(ValueError):
            self.km.add_key("openai", "short")

    def test_public_view_never_contains_a_raw_key(self):
        self.km.add_key("openai", "sk-raw-should-not-leak-4242")
        self.km.add_key("gemini", "AIza-raw-gemini-key-9999")
        blob = json.dumps(self.km.public())
        self.assertNotIn("sk-raw-should-not-leak-4242", blob)
        self.assertNotIn("AIza-raw-gemini-key-9999", blob)
        self.assertIn("key_masked", blob)

    # ---------- backend mode ----------
    def test_backend_mode_defaults_to_auto(self):
        km = KeyManager()
        self.assertEqual(km.mode(), "auto")
        self.assertTrue(km.load()["prefer_cloud"])

    def test_set_mode_keeps_the_boolean_view_in_step(self):
        for mode, expect in (("local", False), ("auto", True), ("cloud", True)):
            km = KeyManager()
            km.set_mode(mode)
            self.assertEqual(km.mode(), mode, msg=mode)
            self.assertEqual(bool(km.load()["prefer_cloud"]), expect, msg=mode)
            self.assertEqual(km.public()["backend_mode"], mode, msg=mode)

    def test_set_mode_rejects_unknown_values(self):
        with self.assertRaises(ValueError):
            KeyManager().set_mode("sometimes")

    def test_legacy_config_without_mode_is_derived_from_prefer_cloud(self):
        self.km.path.parent.mkdir(parents=True, exist_ok=True)
        self.km.path.write_text(json.dumps({"prefer_cloud": False}),
                                encoding="utf-8")
        km = KeyManager()
        self.assertEqual(km.mode(), "local")

    # ---------- rotation ----------
    def test_next_key_round_robins(self):
        a = self.km.add_key("openai", "sk-alpha-aaaaaaaaaaaa")
        b = self.km.add_key("openai", "sk-beta-bbbbbbbbbbbbbb")
        picks = [self.km.next_key("openai")["id"] for _ in range(4)]
        self.assertEqual(sorted({picks[0], picks[1]}), sorted([a["id"], b["id"]]))
        self.assertEqual(picks[0], picks[2])
        self.assertEqual(picks[1], picks[3])

    def test_rate_limited_key_is_parked_then_returned(self):
        e = self.km.add_key("openai", "sk-cool-cccccccccccccc")
        other = self.km.add_key("openai", "sk-cool2-dddddddddddd")
        wait = self.km.mark_rate_limited("openai", e["id"], 30.0)
        self.assertGreater(wait, 25.0)
        healthy = [k["id"] for k in self.km.healthy("openai")]
        self.assertNotIn(e["id"], healthy)
        self.assertIn(other["id"], healthy)
        # expiry brings the key back
        for k in self.km.entries("openai"):
            if k["id"] == e["id"]:
                k["cooldown_until"] = time.time() - 1
        healthy = [k["id"] for k in self.km.healthy("openai")]
        self.assertIn(e["id"], healthy)

    def test_default_cooldown_when_retry_after_is_missing(self):
        e = self.km.add_key("openai", "sk-cool3-eeeeeeeeeeeeee")
        wait = self.km.mark_rate_limited("openai", e["id"])
        self.assertGreaterEqual(wait, KM.DEFAULT_COOLDOWN - 2)

    # ---------- daily-quota parking (Release 30) ----------

    @staticmethod
    def _ts(y, m, d, hh=0, mm=0):
        return datetime(y, m, d, hh, mm, tzinfo=timezone.utc).timestamp()

    def test_quota_reset_gemini_ends_at_next_pacific_midnight_pdt(self):
        # 2026-07-15 17:00 UTC is 10:00 PDT; the next Pacific midnight
        # is 2026-07-16 00:00 PDT = 07:00 UTC.
        now = self._ts(2026, 7, 15, 17)
        self.assertEqual(KM.quota_reset_after("gemini", now),
                         self._ts(2026, 7, 16, 7))

    def test_quota_reset_gemini_ends_at_next_pacific_midnight_pst(self):
        # 2026-01-15 20:00 UTC is 12:00 PST; the next Pacific midnight
        # is 2026-01-16 00:00 PST = 08:00 UTC.
        now = self._ts(2026, 1, 15, 20)
        self.assertEqual(KM.quota_reset_after("gemini", now),
                         self._ts(2026, 1, 16, 8))

    def test_quota_reset_utc_providers_end_at_next_utc_midnight(self):
        now = self._ts(2026, 7, 15, 17)
        for provider in ("groq", "openrouter", "openai"):
            self.assertEqual(KM.quota_reset_after(provider, now),
                             self._ts(2026, 7, 16, 0), msg=provider)

    def test_quota_reset_always_lands_within_25_hours(self):
        # Includes the autumn DST day (2026-11-01, 25 hours long in
        # Pacific): just after its midnight the park is ~24.5h.
        for now in (self._ts(2026, 7, 15, 17), self._ts(2026, 1, 15, 20),
                    self._ts(2026, 11, 1, 7, 30), self._ts(2026, 3, 8, 8),
                    self._ts(2026, 10, 24, 0, 30)):
            for provider in ("gemini", "groq", "openrouter"):
                park = KM.quota_reset_after(provider, now) - now
                self.assertGreater(park, 0, msg=(provider, now))
                self.assertLessEqual(park, 25 * 3600, msg=(provider, now))

    def test_pacific_midnight_computed_fallback_matches(self):
        # The no-tzdata path must answer identically — pinned directly
        # so a Termux without tzdata is covered on any test host.
        self.assertEqual(KM._pacific_midnight_computed(self._ts(2026, 7, 15, 17)),
                         self._ts(2026, 7, 16, 7))
        self.assertEqual(KM._pacific_midnight_computed(self._ts(2026, 1, 15, 20)),
                         self._ts(2026, 1, 16, 8))
        self.assertEqual(KM._pacific_midnight_computed(self._ts(2026, 11, 1, 7, 30)),
                         self._ts(2026, 11, 2, 8))

    def test_pacific_offset_phases_and_switch_boundaries(self):
        def at(y, m, d, hh, mm=0):
            return datetime(y, m, d, hh, mm, tzinfo=timezone.utc)
        self.assertEqual(KM._pacific_offset_hours_at(at(2026, 7, 15, 17)), -7)
        self.assertEqual(KM._pacific_offset_hours_at(at(2026, 1, 15, 20)), -8)
        # Spring forward 2026: PDT starts Sun Mar 8, 10:00 UTC.
        self.assertEqual(KM._pacific_offset_hours_at(at(2026, 3, 8, 9, 59)), -8)
        self.assertEqual(KM._pacific_offset_hours_at(at(2026, 3, 8, 10, 0)), -7)
        # Fall back 2026: PST resumes Sun Nov 1, 09:00 UTC.
        self.assertEqual(KM._pacific_offset_hours_at(at(2026, 11, 1, 8, 59)), -7)
        self.assertEqual(KM._pacific_offset_hours_at(at(2026, 11, 1, 9, 0)), -8)

    def test_mark_quota_exhausted_parks_until_the_reset(self):
        fixed = self._ts(2026, 7, 15, 17)          # 10:00 PDT
        e = self.km.add_key("gemini", "gem-quota1-aaaaaaaaaaaaaa")
        other = self.km.add_key("gemini", "gem-quota2-bbbbbbbbbbbbbb")
        with mock.patch.object(KM, "_now", return_value=fixed):
            wait = self.km.mark_quota_exhausted("gemini", e["id"])
            entry = [k for k in self.km.entries("gemini")
                     if k["id"] == e["id"]][0]
            self.assertEqual(entry["status"], "rate_limited")
            self.assertEqual(entry["last_error"], "quota exhausted (daily)")
            self.assertEqual(entry["cooldown_until"], self._ts(2026, 7, 16, 7))
            self.assertEqual(wait, self._ts(2026, 7, 16, 7) - fixed)
            healthy = [k["id"] for k in self.km.healthy("gemini")]
            self.assertNotIn(e["id"], healthy)
            self.assertIn(other["id"], healthy)
            # a success clears the park like any other (mark_ok resets
            # status + cooldown by design)
            self.km.mark_ok("gemini", e["id"])
            entry = [k for k in self.km.entries("gemini")
                     if k["id"] == e["id"]][0]
            self.assertEqual(entry["status"], "ok")
            self.assertEqual(entry["cooldown_until"], 0.0)
            self.assertIn(e["id"],
                          [k["id"] for k in self.km.healthy("gemini")])

    def test_mark_quota_exhausted_utc_provider_ends_at_utc_midnight(self):
        fixed = self._ts(2026, 7, 15, 17)
        e = self.km.add_key("groq", "groq-quota1-aaaaaaaaaaaa")
        with mock.patch.object(KM, "_now", return_value=fixed):
            self.km.mark_quota_exhausted("groq", e["id"])
            entry = self.km.entries("groq")[0]
            self.assertEqual(entry["cooldown_until"], self._ts(2026, 7, 16, 0))

    def test_auth_failure_marks_key_dead_forever(self):
        e = self.km.add_key("openai", "sk-dead-ffffffffffffffff")
        self.km.mark_auth_failed("openai", e["id"], "401 Invalid key sk-dead-ffffffffffffffff")
        self.assertEqual(self.km.healthy("openai"), [])
        self.assertEqual(self.km.entries("openai")[0]["status"], "dead")
        # the stored error must already be masked
        self.assertNotIn("sk-dead-ffffffffffffffff",
                         self.km.entries("openai")[0]["last_error"] or "")

    def test_mark_ok_clears_state_and_counts_use(self):
        e = self.km.add_key("openai", "sk-ok-gggggggggggggg")
        self.km.mark_rate_limited("openai", e["id"], 10.0)
        self.km.mark_ok("openai", e["id"])
        entry = self.km.entries("openai")[0]
        self.assertEqual(entry["status"], "ok")
        self.assertEqual(entry["cooldown_until"], 0.0)
        self.assertEqual(entry["used"], 1)
        self.assertEqual(entry["last_error"], None)

    def test_remove_key(self):
        e = self.km.add_key("openai", "sk-rm-hhhhhhhhhhhhhh")
        self.assertTrue(self.km.remove_key("openai", e["id"]))
        self.assertFalse(self.km.remove_key("openai", e["id"]))
        self.assertEqual(self.km.entries("openai"), [])

    def test_summary_counts_health(self):
        self.km.add_key("openai", "sk-s1-iiiiiiiiiiiiii")
        e2 = self.km.add_key("openai", "sk-s2-jjjjjjjjjjjjjj")
        self.km.mark_auth_failed("openai", e2["id"], "nope")
        s = self.km.summary()["openai"]
        self.assertEqual(s["total"], 2)
        self.assertEqual(s["healthy"], 1)
        self.assertEqual(s["dead"], 1)
        self.assertEqual(s["keys"][0]["key_masked"][:4], "sk-s")

    # ---------- file robustness ----------
    def test_corrupt_file_is_moved_aside_not_reused(self):
        self.km.path.parent.mkdir(parents=True, exist_ok=True)
        self.km.path.write_text("{ not json", encoding="utf-8")
        km = KeyManager()
        cfg = km.load()
        for p in KM.PROVIDERS:
            self.assertEqual(cfg["keys"].get(p), [], msg=p)
        self.assertTrue(km.path.with_suffix(".corrupt").exists())
        self.assertFalse(km.path.exists())

    def test_partial_config_is_merged_over_defaults(self):
        self.km.path.parent.mkdir(parents=True, exist_ok=True)
        self.km.path.write_text(json.dumps(
            {"prefer_cloud": False,
             "providers": {"openai": {"model": "gpt-4o"}}}), encoding="utf-8")
        cfg = KeyManager().load()
        self.assertFalse(cfg["prefer_cloud"])
        self.assertEqual(cfg["providers"]["openai"]["model"], "gpt-4o")
        # untouched providers keep their defaults
        self.assertIn("gemini", cfg["providers"])
        self.assertIn("deepseek", cfg["providers"])
        self.assertEqual(cfg["fallback"], list(KM.PROVIDERS))
        self.assertEqual(cfg["backend_mode"], "auto")

    def test_defaults_are_not_shared_between_instances(self):
        a = KeyManager().load()
        a["fallback"].append("mutated")
        self.assertNotIn("mutated", KeyManager().load()["fallback"])



    def test_invalidate_rereads_the_file(self):
        """Settings writes from another request — a cached copy must not win."""
        self.km.save()
        self.assertEqual(self.km.mode(), "auto")

        data = json.loads(self.km.path.read_text(encoding="utf-8"))
        data["backend_mode"] = "cloud"
        data["prefer_cloud"] = True
        self.km.path.write_text(json.dumps(data), encoding="utf-8")

        self.assertEqual(self.km.mode(), "auto", "stale cache served the file")
        self.km.invalidate()
        self.assertEqual(self.km.mode(), "cloud")

    def test_invalidate_keeps_the_rotation_cursor(self):
        self.km.add_key("openai", "sk-cursor-0000000000000000", "c")
        self.km.next_key("openai")
        cursor = dict(self.km._cursor)
        self.km.invalidate()
        self.assertEqual(self.km._cursor, cursor, "cursor lost on refresh")
        self.assertTrue(self.km.load())


class FixedPathTest(unittest.TestCase):
    """Where the key file lives: private storage, found under both HOMEs.

    The phone runs the server with HOME=/data/data/com.termux/files/home
    and the shell with HOME=/root, so a plain $HOME-derived path splits
    into two files and the keys vanish on restart. The private location
    is therefore canonicalised on the Termux files dir itself — a device
    fact, identical under both HOMEs (config.private_keys_file). Until
    v1.9.0 the file sat in <project>/data, on shared storage where any
    app with the storage permission could read it; load() moves it.
    """

    def setUp(self):
        import config as cfg_mod
        self.cfg = cfg_mod
        self._saved = (cfg_mod.DATA_DIR, cfg_mod.AGENT_CONFIG_FILE,
                       cfg_mod.DEFAULT_DATA_DIR, cfg_mod.LEGACY_HOMES,
                       cfg_mod.CONFIG_DIR, cfg_mod.TERMUX_HOME)

    def tearDown(self):
        (self.cfg.DATA_DIR, self.cfg.AGENT_CONFIG_FILE,
         self.cfg.DEFAULT_DATA_DIR, self.cfg.LEGACY_HOMES,
         self.cfg.CONFIG_DIR, self.cfg.TERMUX_HOME) = self._saved

    def _point_at(self, data: Path):
        self.cfg.DATA_DIR = data
        self.cfg.DEFAULT_DATA_DIR = data
        self.cfg.AGENT_CONFIG_FILE = data / "agent_config.json"

    def _point_private(self, root: Path):
        """Private location = <root>/home/.ai-agent (no Termux dir)."""
        self.cfg.CONFIG_DIR = root / "home" / ".ai-agent"
        self.cfg.TERMUX_HOME = root / "no-termux-here"

    def test_default_path_is_private_for_a_fresh_install(self):
        """Nothing anywhere yet: the file is born in private storage."""
        root = Path(tempfile.mkdtemp())
        self._point_at(root / "data")
        self._point_private(root)
        self.cfg.LEGACY_HOMES = ()
        km = KeyManager()
        self.assertEqual(km.path,
                         root / "home" / ".ai-agent" / "agent_config.json")
        self.assertTrue(km.path.is_absolute())

    def test_default_path_is_project_while_migration_is_pending(self):
        """Keys still on shared storage: read there until load() moves."""
        root = Path(tempfile.mkdtemp())
        self._point_at(root / "data")
        self._point_private(root)
        self.cfg.LEGACY_HOMES = ()
        (root / "data").mkdir(parents=True)
        (root / "data" / "agent_config.json").write_text("{}")
        km = KeyManager()
        self.assertEqual(km.path, root / "data" / "agent_config.json")

    def test_the_two_phone_homes_are_both_searched(self):
        dirs = [str(d) for d in self.cfg.legacy_config_dirs()]
        self.assertTrue(
            any(d.startswith("/data/data/com.termux/files/home") for d in dirs),
            dirs)
        self.assertTrue(any(d.startswith("/root") for d in dirs), dirs)

    def test_legacy_keys_are_migrated_once(self):
        root = Path(tempfile.mkdtemp())
        legacy_dir = root / "old-home" / ".ai-agent"
        legacy_dir.mkdir(parents=True)
        (legacy_dir / "agent_config.json").write_text(json.dumps({
            "keys": {"openai": [{
                "id": "k_old", "key": "sk-live-legacy-0000000000",
                "label": "", "status": "ok", "cooldown_until": 0.0, "used": 3,
            }]}},
        ), encoding="utf-8")
        self._point_at(root / "data")
        self._point_private(root)
        self.cfg.LEGACY_HOMES = (str(root / "old-home"),)

        km = KeyManager()
        # Read precedence finds the legacy copy before anything moves.
        self.assertEqual(km.path, legacy_dir / "agent_config.json")
        km.load()
        # The move chains legacy HOME → project → private; the recorded
        # source is the project file, and it is gone afterwards.
        self.assertEqual(km.migrated_from,
                         root / "data" / "agent_config.json")
        self.assertFalse((root / "data" / "agent_config.json").exists())
        self.assertEqual(km.path,
                         root / "home" / ".ai-agent" / "agent_config.json")
        self.assertEqual(km.entries("openai")[0]["key"],
                         "sk-live-legacy-0000000000")
        self.assertEqual(km.path.stat().st_mode & 0o777, 0o600)

        # Once on disk, a later instance must not copy over the live file.
        km.path.write_text(json.dumps({"keys": {"openai": []}}),
                           encoding="utf-8")
        again = KeyManager()
        again.load()
        self.assertIsNone(again.migrated_from)
        self.assertEqual(again.entries("openai"), [])

    def test_migration_stays_off_for_a_retargeted_dir(self):
        """A harness that repointed DATA_DIR must not pull live keys in."""
        root = Path(tempfile.mkdtemp())
        legacy_dir = root / "old-home" / ".ai-agent"
        legacy_dir.mkdir(parents=True)
        (legacy_dir / "agent_config.json").write_text(
            json.dumps({"keys": {"openai": [{
                "id": "k_live", "key": "sk-live-must-not-import-1111",
                "label": "", "status": "ok", "cooldown_until": 0.0}]}}),
            encoding="utf-8")
        self.cfg.DATA_DIR = root / "retargeted"
        self.cfg.DEFAULT_DATA_DIR = root / "somewhere-else"
        self.cfg.AGENT_CONFIG_FILE = self.cfg.DATA_DIR / "agent_config.json"
        self.cfg.LEGACY_HOMES = (str(root / "old-home"),)

        km = KeyManager()
        km.load()
        self.assertIsNone(km.migrated_from)
        self.assertEqual(km.entries("openai"), [])
        self.assertFalse(km.path.exists())

    def _write_keys(self, path: Path, key_id: str, raw: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"keys": {"openai": [{
            "id": key_id, "key": raw, "label": "", "status": "ok",
            "cooldown_until": 0.0, "used": 1}]}}), encoding="utf-8")

    def test_keys_migrate_from_project_to_private_on_load(self):
        root = Path(tempfile.mkdtemp())
        data = root / "data"
        self._write_keys(data / "agent_config.json", "k_p", "sk-proj-key")
        self._point_at(data)
        self._point_private(root)
        self.cfg.LEGACY_HOMES = ()
        km = KeyManager()
        self.assertEqual(km.path, data / "agent_config.json")
        km.load()
        private = root / "home" / ".ai-agent" / "agent_config.json"
        self.assertEqual(km.path, private)
        self.assertEqual(km.migrated_from, data / "agent_config.json")
        self.assertFalse((data / "agent_config.json").exists())
        self.assertEqual(km.entries("openai")[0]["key"], "sk-proj-key")
        self.assertEqual(private.stat().st_mode & 0o777, 0o600)

    def test_project_copy_wins_over_a_stale_private_copy(self):
        """Both files exist: the project copy is the live one — it is
        the only one pre-v1.9.0 code ever wrote. load() moves FROM
        it, and the displaced private copy is kept, not destroyed."""
        root = Path(tempfile.mkdtemp())
        data = root / "data"
        self._write_keys(data / "agent_config.json", "k_a", "sk-live-key")
        self._point_at(data)
        self._point_private(root)
        private = root / "home" / ".ai-agent" / "agent_config.json"
        self._write_keys(private, "k_b", "sk-stale-key")
        self.cfg.LEGACY_HOMES = ()
        km = KeyManager()
        self.assertEqual(km.path, data / "agent_config.json")
        km.load()
        self.assertEqual(km.path, private)
        self.assertEqual(km.migrated_from, data / "agent_config.json")
        self.assertEqual(km.entries("openai")[0]["key"], "sk-live-key")
        self.assertFalse((data / "agent_config.json").exists())
        superseded = list(private.parent.glob(
            "agent_config.json.superseded-*"))
        self.assertEqual(len(superseded), 1)
        self.assertIn("sk-stale-key", superseded[0].read_text())

    def test_identical_copies_just_retire_the_project_file(self):
        root = Path(tempfile.mkdtemp())
        data = root / "data"
        self._write_keys(data / "agent_config.json", "k_a", "sk-same-key")
        self._point_at(data)
        self._point_private(root)
        private = root / "home" / ".ai-agent" / "agent_config.json"
        self._write_keys(private, "k_a", "sk-same-key")
        self.cfg.LEGACY_HOMES = ()
        km = KeyManager()
        km.load()
        self.assertEqual(km.path, private)
        self.assertEqual(km.migrated_from, data / "agent_config.json")
        self.assertFalse((data / "agent_config.json").exists())
        self.assertEqual(list(private.parent.glob(
            "agent_config.json.superseded-*")), [])
        self.assertEqual(km.entries("openai")[0]["key"], "sk-same-key")

    def test_failed_private_migration_falls_back_to_project(self):
        root = Path(tempfile.mkdtemp())
        data = root / "data"
        self._write_keys(data / "agent_config.json", "k_p", "sk-proj-key")
        self._point_at(data)
        self.cfg.LEGACY_HOMES = ()
        # Block the private location: TERMUX_HOME/.ai-agent is a file,
        # so the private dir can never be created there.
        phone = root / "phone"
        phone.mkdir()
        (phone / ".ai-agent").write_text("not a dir")
        self.cfg.TERMUX_HOME = phone
        km = KeyManager()
        km.load()
        self.assertEqual(km.path, data / "agent_config.json")
        self.assertIsNone(km.migrated_from)
        self.assertEqual(km.entries("openai")[0]["key"], "sk-proj-key")
        self.assertTrue((data / "agent_config.json").exists())

    def test_termux_home_is_the_canonical_private_location(self):
        root = Path(tempfile.mkdtemp())
        data = root / "data"
        self._write_keys(data / "agent_config.json", "k_p", "sk-proj-key")
        self._point_at(data)
        self.cfg.LEGACY_HOMES = ()
        phone = root / "phone"
        phone.mkdir()
        self.cfg.TERMUX_HOME = phone
        self.cfg.CONFIG_DIR = root / "elsewhere" / ".ai-agent"
        km = KeyManager()
        km.load()
        self.assertEqual(km.path,
                         phone / ".ai-agent" / "agent_config.json")
        self.assertEqual(km.entries("openai")[0]["key"], "sk-proj-key")

    def test_save_never_raises_when_chmod_is_refused(self):
        """/storage/emulated/0 has no mode bits — keys must still be written."""
        root = Path(tempfile.mkdtemp())
        self._point_at(root / "data")
        self._point_private(root)
        self.cfg.LEGACY_HOMES = ()
        real_chmod = os.chmod

        def refuse(path, mode, **kw):
            raise OSError(1, "Operation not permitted")

        os.chmod = refuse
        try:
            km = KeyManager()
            km.add_key("openai", "sk-fuse-aaaaaaaaaaaaaaa")
            self.assertTrue(km.path.exists())
            self.assertEqual(km.entries("openai")[0]["key"],
                             "sk-fuse-aaaaaaaaaaaaaaa")
        finally:
            os.chmod = real_chmod


class ActiveKeyTest(unittest.TestCase):
    """Manual key selection: `set_active_key` drives the rotation and the UI.

    The active key is what `/api/settings/agent/keys/<p>/<id>/activate` pins
    and what the Settings radio shows. Until the user picks one, the rotation
    keeps rotating — selection changes behaviour only when it is asked for.
    """

    def setUp(self):
        self.km = KeyManager()
        if self.km.path.exists():
            self.km.path.unlink()

    def keys(self):
        return self.km.entries("openai")

    def test_set_active_key_marks_exactly_one(self):
        a = self.km.add_key("openai", "sk-act-aaaaaaaaaaaaaaaa")
        b = self.km.add_key("openai", "sk-act-bbbbbbbbbbbbbbbb")
        self.km.set_active_key("openai", b["id"])
        flags = {e["id"]: e.get("active") for e in self.keys()}
        self.assertEqual(flags[b["id"]], True)
        self.assertFalse(flags[a["id"]])

        self.km.set_active_key("openai", a["id"])
        flags = {e["id"]: e.get("active") for e in self.keys()}
        self.assertEqual(flags[a["id"]], True)
        self.assertFalse(flags[b["id"]])

    def test_active_selection_is_persisted_and_read_back(self):
        a = self.km.add_key("openai", "sk-persist-aaaaaaaaaaaa")
        b = self.km.add_key("openai", "sk-persist-bbbbbbbbbbbb")
        self.km.set_active_key("openai", b["id"])

        fresh = KeyManager()
        self.assertEqual(fresh.active_entry("openai")["id"], b["id"])
        pub = fresh.public()
        marked = [e for e in pub["keys"]["openai"] if e["active"]]
        self.assertEqual([e["id"] for e in marked], [b["id"]])
        self.assertNotIn(a["id"], [e["id"] for e in marked])
        # Still no raw key anywhere in the public view.
        self.assertNotIn("sk-persist-bbbbbbbbbbbb", json.dumps(pub))

    def test_next_key_prefers_the_active_key(self):
        a = self.km.add_key("openai", "sk-pin-aaaaaaaaaaaaaaa")
        b = self.km.add_key("openai", "sk-pin-bbbbbbbbbbbbbbbb")
        self.km.set_active_key("openai", a["id"])
        picks = {self.km.next_key("openai")["id"] for _ in range(6)}
        self.assertEqual(picks, {a["id"]},
                         "the pinned key did not win the rotation")
        self.assertNotEqual(picks, {b["id"]})

    def test_rotation_survives_when_nothing_is_pinned(self):
        """Selection is opt-in: adding keys must not silently pin the first."""
        a = self.km.add_key("openai", "sk-orb-aaaaaaaaaaaaaaa")
        b = self.km.add_key("openai", "sk-orb-bbbbbbbbbbbbbbbb")
        picks = [self.km.next_key("openai")["id"] for _ in range(4)]
        self.assertEqual(sorted({picks[0], picks[1]}),
                         sorted([a["id"], b["id"]]))

    def test_a_dead_pinned_key_hands_the_turn_back_to_the_rotation(self):
        """Pinning must never strand a provider on a key that cannot answer."""
        a = self.km.add_key("openai", "sk-dead-aaaaaaaaaaaaaaa")
        b = self.km.add_key("openai", "sk-dead-bbbbbbbbbbbbbbbb")
        self.km.set_active_key("openai", a["id"])
        self.km.mark_auth_failed("openai", a["id"], "invalid")
        self.assertEqual(self.km.next_key("openai")["id"], b["id"])

        # Cooling out of a 429 counts the same.
        self.km.mark_rate_limited("openai", b["id"], 300)
        c = self.km.add_key("openai", "sk-dead-ccccccccccccccc")
        self.assertEqual(self.km.next_key("openai")["id"], c["id"])

    def test_removing_the_active_key_promotes_a_survivor(self):
        a = self.km.add_key("openai", "sk-prm-aaaaaaaaaaaaaaa")
        b = self.km.add_key("openai", "sk-prm-bbbbbbbbbbbbbbbb")
        self.km.set_active_key("openai", a["id"])
        self.km.remove_key("openai", a["id"])
        active = self.km.active_entry("openai")
        self.assertIsNotNone(active)
        self.assertEqual(active["id"], b["id"])
        self.assertEqual(active.get("active"), True)

    def test_active_entry_falls_back_for_a_config_with_no_flags(self):
        """Configs written before selection existed still show an honest tick."""
        self.km.add_key("openai", "sk-leg-aaaaaaaaaaaaaaaa")
        b = self.km.add_key("openai", "sk-leg-bbbbbbbbbbbbbbbb")
        self.assertEqual(self.km.active_entry("openai")["id"],
                         self.keys()[0]["id"])
        self.assertEqual(self.km.public()["keys"]["openai"][0]["active"], True)
        self.assertEqual(self.km.public()["keys"]["openai"][1]["active"], False)
        self.assertNotEqual(b["id"], self.km.active_entry("openai")["id"])

    def test_unknown_key_and_provider_are_rejected(self):
        a = self.km.add_key("openai", "sk-err-aaaaaaaaaaaaaaaa")
        with self.assertRaises(KeyError):
            self.km.set_active_key("openai", "k_not_there")
        with self.assertRaises(ValueError):
            self.km.set_active_key("nope", a["id"])

    def test_summaries_report_the_active_id(self):
        a = self.km.add_key("openai", "sk-sum-aaaaaaaaaaaaaaaa")
        b = self.km.add_key("openai", "sk-sum-bbbbbbbbbbbbbbbb")
        self.km.set_active_key("openai", b["id"])
        s = self.km.summary()["openai"]
        self.assertEqual(s["active_id"], b["id"])
        self.assertEqual([e["id"] for e in s["keys"] if e["active"]],
                         [b["id"]])
        self.assertNotEqual(s["active_id"], a["id"])


if __name__ == "__main__":
    unittest.main()
