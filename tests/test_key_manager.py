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
    """The key file must not move with $HOME — the phone has two of them.

    The server runs with HOME=/data/data/com.termux/files/home and the shell
    with HOME=/root, so a HOME-derived path splits into two files and the keys
    vanish on restart.
    """

    def setUp(self):
        import config as cfg_mod
        self.cfg = cfg_mod
        self._saved = (cfg_mod.DATA_DIR, cfg_mod.AGENT_CONFIG_FILE,
                       cfg_mod.DEFAULT_DATA_DIR, cfg_mod.LEGACY_HOMES)

    def tearDown(self):
        (self.cfg.DATA_DIR, self.cfg.AGENT_CONFIG_FILE,
         self.cfg.DEFAULT_DATA_DIR, self.cfg.LEGACY_HOMES) = self._saved

    def _point_at(self, data: Path):
        self.cfg.DATA_DIR = data
        self.cfg.DEFAULT_DATA_DIR = data
        self.cfg.AGENT_CONFIG_FILE = data / "agent_config.json"

    def test_default_path_is_the_project_data_dir(self):
        """Out of the box the file is <project>/data/agent_config.json."""
        self._point_at(self.cfg.PROJECT_ROOT / "data")
        km = KeyManager()
        self.assertEqual(km.path,
                         self.cfg.PROJECT_ROOT / "data" / "agent_config.json")
        self.assertTrue(km.path.is_absolute())

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
        self.cfg.LEGACY_HOMES = (str(root / "old-home"),)

        km = KeyManager()
        self.assertFalse(km.path.exists())
        km.load()
        self.assertEqual(km.migrated_from, legacy_dir / "agent_config.json")
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

    def test_save_never_raises_when_chmod_is_refused(self):
        """/storage/emulated/0 has no mode bits — keys must still be written."""
        root = Path(tempfile.mkdtemp())
        self._point_at(root / "data")
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
