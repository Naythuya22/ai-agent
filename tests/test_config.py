"""Model discovery / config tests.

The model is often kept outside the repo (e.g. alongside it), so a fresh
install with no ~/.ai-agent/config.json must still find it.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from _harness import bootstrap, current_home

bootstrap()

import config as config_mod  # noqa: E402  (must follow bootstrap)
from config import Config, find_model_file, model_missing_message  # noqa: E402


class TestDiscovery(unittest.TestCase):
    def setUp(self):
        self._orig = (
            config_mod.MODELS_DIR,
            config_mod.LEGACY_MODEL_DIRS,
            config_mod.PROJECT_ROOT,
        )
        self._tmp = Path(tempfile.mkdtemp())
        # Isolate the search: the dev machine has a real model in a sibling
        # directory, so "no model anywhere" is not a reachable state here.
        config_mod.LEGACY_MODEL_DIRS = []
        config_mod.PROJECT_ROOT = self._tmp / "isolated-project"

    def tearDown(self):
        (config_mod.MODELS_DIR, config_mod.LEGACY_MODEL_DIRS,
         config_mod.PROJECT_ROOT) = self._orig

    def test_finds_model_in_bundled_dir(self):
        config_mod.MODELS_DIR = self._tmp / "models"
        (config_mod.MODELS_DIR).mkdir(parents=True)
        target = config_mod.MODELS_DIR / "m.litertlm"
        target.write_bytes(b"x")
        self.assertEqual(find_model_file(), target)

    def test_bundled_dir_wins_over_legacy(self):
        config_mod.MODELS_DIR = self._tmp / "models"
        config_mod.MODELS_DIR.mkdir(parents=True)
        bundled = config_mod.MODELS_DIR / "m.litertlm"
        bundled.write_bytes(b"x")
        legacy = self._tmp / "legacy"
        legacy.mkdir()
        (legacy / "old.litertlm").write_bytes(b"x")
        config_mod.LEGACY_MODEL_DIRS = [legacy]
        self.assertEqual(find_model_file(), bundled)

    def test_searches_nested_folders(self):
        """Real layout is <dir>/<Model>/<hash>/<model>.litertlm."""
        config_mod.MODELS_DIR = self._tmp / "models"
        deep = config_mod.MODELS_DIR / "Gemma_4_E4B_it" / "28299f30"
        deep.mkdir(parents=True)
        target = deep / "gemma.litertlm"
        target.write_bytes(b"x")
        self.assertEqual(find_model_file(), target)

    def test_missing_dir_is_not_fatal(self):
        config_mod.MODELS_DIR = self._tmp / "does-not-exist"
        self.assertIsNone(find_model_file())

    def test_empty_dir_returns_none(self):
        config_mod.MODELS_DIR = self._tmp / "empty"
        config_mod.MODELS_DIR.mkdir(parents=True)
        self.assertIsNone(find_model_file())

    def test_non_litertlm_files_ignored(self):
        config_mod.MODELS_DIR = self._tmp / "models"
        config_mod.MODELS_DIR.mkdir(parents=True)
        (config_mod.MODELS_DIR / "notes.txt").write_bytes(b"x")
        (config_mod.MODELS_DIR / "model.bin").write_bytes(b"x")
        self.assertIsNone(find_model_file())

    def test_newest_wins(self):
        import os
        import time

        config_mod.MODELS_DIR = self._tmp / "models"
        config_mod.MODELS_DIR.mkdir(parents=True)
        old = config_mod.MODELS_DIR / "old.litertlm"
        new = config_mod.MODELS_DIR / "new.litertlm"
        old.write_bytes(b"x")
        new.write_bytes(b"x")
        os.utime(old, (time.time() - 9999, time.time() - 9999))
        self.assertEqual(find_model_file(), new)

    def test_sibling_dirs_are_searched(self):
        """A model kept next to the project must be found."""
        proj = self._tmp / "tree" / "proj" / "ai-agent"
        proj.mkdir(parents=True)
        store = self._tmp / "tree" / "proj" / "ai-agent-model"
        store.mkdir(parents=True)
        target = store / "m.litertlm"
        target.write_bytes(b"x")

        config_mod.MODELS_DIR = proj / "models"
        config_mod.PROJECT_ROOT = proj
        dirs = config_mod.model_search_dirs()
        self.assertIn(store, dirs)
        self.assertEqual(config_mod.find_model_file(), target)

    def test_sibling_two_levels_up(self):
        """MODEL_ROOT is /<x>/<y>/ai-agent but the store is at /<x>/ai-agent-model."""
        proj = self._tmp / "a" / "b" / "ai-agent"
        proj.mkdir(parents=True)
        store = self._tmp / "a" / "ai-agent-model"
        store.mkdir(parents=True)
        target = store / "m.litertlm"
        target.write_bytes(b"x")

        config_mod.MODELS_DIR = proj / "models"
        config_mod.PROJECT_ROOT = proj
        self.assertEqual(config_mod.find_model_file(), target)

    def test_walk_is_bounded(self):
        """Must not walk the entire filesystem."""
        config_mod.PROJECT_ROOT = Path("/a/b/c/d/e/f/g")
        self.assertLessEqual(len(config_mod.model_search_dirs()), 1 + 3)

    def test_hidden_dirs_skipped(self):
        proj = self._tmp / "p"
        proj.mkdir(parents=True)
        hidden = proj / ".hidden-model-store"
        hidden.mkdir()
        (hidden / "m.litertlm").write_bytes(b"x")
        config_mod.PROJECT_ROOT = proj
        self.assertNotIn(hidden, config_mod.model_search_dirs())


class TestConfigRepair(unittest.TestCase):
    def test_stale_path_is_repaired_and_persisted(self):
        model = bootstrap()  # rewrites config with a valid model
        cfg = Config.load()
        self.assertEqual(cfg.model_path, str(model))
        self.assertEqual(cfg.validate(), [])

    def test_saved_config_survives_reload(self):
        model = bootstrap()
        first = Config.load().model_path
        second = Config.load().model_path
        self.assertEqual(first, second)
        self.assertEqual(first, str(model))

    def test_corrupt_config_falls_back(self):
        bootstrap()
        (current_home() / ".ai-agent" / "config.json").write_text("{not json")
        cfg = Config.load()
        self.assertIsInstance(cfg.model_path, str)

    def test_unknown_keys_dropped(self):
        bootstrap()
        home = current_home()
        (home / ".ai-agent" / "config.json").write_text(
            json.dumps({"model_path": str(home / "fake-models" / "stub-model.litertlm"),
                        "bogus_key": 1})
        )
        self.assertEqual(Config.load().validate(), [])

    def test_bad_context_size_reported(self):
        bootstrap()
        home = current_home()
        (home / ".ai-agent" / "config.json").write_text(
            json.dumps({"model_path": str(home / "fake-models" / "stub-model.litertlm"),
                        "context_size": 4096})
        )
        self.assertTrue(any("context_size" in e for e in Config.load().validate()))

    def test_bad_backend_reported(self):
        bootstrap()
        home = current_home()
        (home / ".ai-agent" / "config.json").write_text(
            json.dumps({"model_path": str(home / "fake-models" / "stub-model.litertlm"),
                        "backend": "TPU"})
        )
        self.assertTrue(any("backend" in e for e in Config.load().validate()))

    def test_stale_saved_path_is_overwritten_on_disk(self):
        bootstrap()
        home = current_home()
        (home / ".ai-agent" / "config.json").write_text(
            json.dumps({"model_path": "/gone/moved-away.litertlm"})
        )
        repaired = Config.load().model_path
        on_disk = json.loads((home / ".ai-agent" / "config.json").read_text())
        self.assertEqual(on_disk["model_path"], repaired)
        self.assertNotEqual(repaired, "/gone/moved-away.litertlm")


class TestErrorMessage(unittest.TestCase):
    def test_lists_searched_dirs(self):
        msg = model_missing_message("/nope.litertlm")
        self.assertIn("/nope.litertlm", msg)
        self.assertIn("copy", msg.lower())

    def test_points_at_a_model_that_exists(self):
        """When one is reachable, the message must name the file."""
        proj = Path(tempfile.mkdtemp()) / "ai-agent"
        proj.mkdir(parents=True)
        store = proj.parent / "ai-agent-model"
        store.mkdir()
        target = store / "m.litertlm"
        target.write_bytes(b"x")

        orig = (config_mod.MODELS_DIR, config_mod.LEGACY_MODEL_DIRS,
                config_mod.PROJECT_ROOT)
        config_mod.MODELS_DIR = proj / "models"
        config_mod.LEGACY_MODEL_DIRS = []
        config_mod.PROJECT_ROOT = proj
        try:
            msg = model_missing_message("")
            self.assertIn(str(target), msg, msg)
            self.assertIn("model_path", msg)
        finally:
            (config_mod.MODELS_DIR, config_mod.LEGACY_MODEL_DIRS,
             config_mod.PROJECT_ROOT) = orig


if __name__ == "__main__":
    unittest.main(verbosity=2)