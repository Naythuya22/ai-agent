"""Backend registry tests: names, lookup, defaults and availability."""
from __future__ import annotations

import unittest
from contextlib import closing
from unittest.mock import patch

from _harness import bootstrap

bootstrap()

from core import builtin as bi  # noqa: E402
from core import code_backends as cb  # noqa: E402
from core import opencode as oc  # noqa: E402
from core import openclaude as ocl  # noqa: E402


class TestNames(unittest.TestCase):
    def test_registers_all_backends_in_order(self):
        # builtin leads: it is the default and the only backend a fresh
        # install can use with zero extra setup.
        self.assertEqual(cb.names(), ["builtin", "opencode", "openclaude"])

    def test_names_is_a_copy(self):
        got = cb.names()
        got.append("bogus")
        self.assertEqual(cb.names(), ["builtin", "opencode", "openclaude"])


class TestGet(unittest.TestCase):
    def test_builtin_resolves_to_the_builtin_module(self):
        self.assertIs(cb.get("builtin"), bi)

    def test_opencode_resolves_to_the_opencode_module(self):
        self.assertIs(cb.get("opencode"), oc)

    def test_openclaude_resolves_to_the_openclaude_module(self):
        self.assertIs(cb.get("openclaude"), ocl)

    def test_unknown_names_answer_none(self):
        for bad in ("claude", "aider", "OPENCLAUDE", None, "", "  "):
            self.assertIsNone(cb.get(bad), repr(bad))

    def test_surrounding_whitespace_is_ignored(self):
        self.assertIs(cb.get("  openclaude  "), ocl)

    def test_a_non_string_does_not_raise(self):
        self.assertIsNone(cb.get(object()))


class TestDefault(unittest.TestCase):
    def test_the_config_default_is_builtin(self):
        # config leaves the choice open (""); the registry fallback is the
        # built-in engine, the only zero-setup backend.
        import config
        self.assertEqual(config.CODE_BACKEND_DEFAULT, "")
        self.assertEqual(cb.default_name(), "builtin")

    def test_an_env_named_backend_is_honoured(self):
        import config
        with patch.object(config, "CODE_BACKEND_DEFAULT", "openclaude"):
            self.assertEqual(cb.default_name(), "openclaude")

    def test_an_unregistered_default_falls_back(self):
        import config
        with patch.object(config, "CODE_BACKEND_DEFAULT", "aider"):
            self.assertEqual(cb.default_name(), "builtin")

    def test_resolve_prefers_a_registered_name(self):
        import config
        with patch.object(config, "CODE_BACKEND_DEFAULT", "opencode"):
            self.assertEqual(cb.resolve("openclaude"), "openclaude")
            self.assertEqual(cb.resolve("nope"), "opencode")
            self.assertEqual(cb.resolve(None), "opencode")
            self.assertEqual(cb.resolve(""), "opencode")


class TestAvailable(unittest.TestCase):
    # builtin.is_installed() is always True (the engine ships in the app),
    # so it appears in every availability result below.

    def test_available_lists_only_installed_backends(self):
        with patch.object(oc, "is_installed", return_value=True):
            with patch.object(ocl, "is_installed", return_value=False):
                self.assertEqual(cb.available(), ["builtin", "opencode"])

    def test_available_is_all_when_all_are_installed(self):
        with patch.object(oc, "is_installed", return_value=True):
            with patch.object(ocl, "is_installed", return_value=True):
                self.assertEqual(cb.available(),
                                 ["builtin", "opencode", "openclaude"])

    def test_availability_reports_each_backend(self):
        with patch.object(oc, "is_installed", return_value=True):
            with patch.object(ocl, "is_installed", return_value=False):
                self.assertEqual(cb.availability(),
                                 {"builtin": True, "opencode": True,
                                  "openclaude": False})

    def test_installed_is_false_for_an_unknown_backend(self):
        self.assertFalse(cb.installed("aider"))

    def test_a_probe_that_raises_counts_as_not_installed(self):
        with patch.object(oc, "is_installed", side_effect=OSError("boom")):
            self.assertFalse(cb.installed("opencode"))
            self.assertNotIn("opencode", cb.available())

    def test_installed_follows_the_backend(self):
        with patch.object(oc, "is_installed", return_value=False):
            with patch.object(ocl, "is_installed", return_value=True):
                self.assertFalse(cb.installed("opencode"))
                self.assertTrue(cb.installed("openclaude"))


class TestBackendEndpoints(unittest.TestCase):
    """The routes that choose a backend: /api/code/backends, .../backend,
    doctor?backend= and the /model chat command."""

    def setUp(self):
        from _harness import current_home, get_app

        self.client = get_app().test_client()
        from core import projects as ps
        self.ps = ps
        self._prev_active = ps.load().get("active")
        folder = current_home() / "be-proj"
        folder.mkdir(parents=True, exist_ok=True)
        self.proj = ps.add("be-proj", str(folder))
        ps.set_active(self.proj["id"])
        self.addCleanup(self._restore)

    def _restore(self):
        ps = self.ps
        ps.set_backend(self.proj["id"], None)
        data = ps.load()
        if self._prev_active and any(p.get("id") == self._prev_active
                                     for p in data["projects"]):
            ps.set_active(self._prev_active)
        ps.remove(self.proj["id"])

    def _post(self, path, **kw):
        with closing(self.client.post(path, **kw)) as r:
            return r.status_code, r.get_json()

    def _get(self, path):
        with closing(self.client.get(path)) as r:
            return r.status_code, r.get_json()

    # -- GET /api/code/backends ------------------------------------------

    def test_backends_is_a_json_list_of_all_names(self):
        status, body = self._get("/api/code/backends")
        self.assertEqual(status, 200)
        self.assertIsInstance(body, list)
        self.assertEqual(body, ["builtin", "opencode", "openclaude"])

    # -- POST /api/code/backend ------------------------------------------

    def test_setting_the_backend_stores_it_on_the_project(self):
        status, body = self._post("/api/code/backend", json={"name": "openclaude"})
        self.assertEqual(status, 200, body)
        self.assertTrue(body["ok"])
        self.assertEqual(body["backend"], "openclaude")
        self.assertEqual(self.ps.get_active()["backend"], "openclaude")

    def test_an_unknown_backend_is_rejected(self):
        status, body = self._post("/api/code/backend", json={"name": "aider"})
        self.assertEqual(status, 400)
        self.assertIn("unknown backend", body["error"])
        self.assertEqual(body["available"],
                         ["builtin", "opencode", "openclaude"])
        self.assertIsNone(self.ps.get_active()["backend"])

    def test_a_missing_name_is_rejected(self):
        status, body = self._post("/api/code/backend", json={})
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "name required")

    def test_no_active_project_is_reported_not_swallowed(self):
        active = self.ps.get_active()
        self.ps.set_active(None) if hasattr(self.ps, "set_active") else None
        self.ps._save({"version": 1, "projects": [], "active": None})
        status, body = self._post("/api/code/backend", json={"name": "openclaude"})
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "no active project")
        # put the fixture back for the cleanup pass
        self.ps._save({"version": 1, "projects": [self.proj],
                       "active": self.proj["id"]})

    # -- GET /api/code/doctor --------------------------------------------

    def test_doctor_without_a_query_stays_on_builtin(self):
        status, body = self._get("/api/code/doctor")
        self.assertEqual(status, 200)
        for key in ("installed", "version", "config_file", "model",
                    "bridge_base", "ok"):
            self.assertIn(key, body)
        self.assertEqual(body["backend"], "builtin")
        self.assertEqual(body["binary"], "built-in (in-app agent engine)")
        self.assertTrue(body["installed"])

    def test_doctor_accepts_a_backend_query(self):
        with patch.object(ocl, "is_installed", return_value=True):
            with patch.object(ocl, "version", return_value="0.32.0 (OpenClaude)"):
                status, body = self._get("/api/code/doctor?backend=openclaude")
        self.assertEqual(status, 200)
        self.assertEqual(body["backend"], "openclaude")
        self.assertEqual(body["binary"], "openclaude")
        self.assertTrue(body["ok"], body)
        self.assertTrue(body["installed"])

    def test_a_stored_backend_drives_doctor_without_a_query(self):
        self.ps.set_backend(self.proj["id"], "openclaude")
        with patch.object(ocl, "is_installed", return_value=True):
            with patch.object(ocl, "version", return_value="0.32.0"):
                status, body = self._get("/api/code/doctor")
        self.assertEqual(status, 200)
        self.assertEqual(body["backend"], "openclaude")

    # -- /model in code-mode chat ----------------------------------------

    def _chat(self, message):
        status, body = self._post("/api/code/web/run", json={"message": message})
        return status, body

    def test_model_lists_all_backends(self):
        status, body = self._chat("/model")
        self.assertEqual(status, 200, body)
        self.assertIn("builtin", body["reply"])
        self.assertIn("opencode", body["reply"])
        self.assertIn("openclaude", body["reply"])
        self.assertIn("Code backends", body["reply"])

    def test_model_status_names_the_current_backend(self):
        status, body = self._chat("/model status")
        self.assertEqual(status, 200, body)
        self.assertIn("Backend", body["reply"])
        self.assertIn("builtin", body["reply"])
        self.assertIn("Version", body["reply"])

    def test_model_switches_the_active_project(self):
        with patch.object(ocl, "is_installed", return_value=True):
            status, body = self._chat("/model openclaude")
        self.assertEqual(status, 200, body)
        self.assertEqual(body["rc"], 0, body["reply"])
        self.assertIn("openclaude", body["reply"])
        self.assertEqual(self.ps.get_active()["backend"], "openclaude")

    def test_model_rejects_an_unknown_backend(self):
        status, body = self._chat("/model aider")
        self.assertEqual(status, 200, body)
        self.assertEqual(body["rc"], 1)
        self.assertIn("Unknown backend", body["reply"])
        self.assertIsNone(self.ps.get_active()["backend"])

    def test_model_works_when_no_binary_is_installed(self):
        with patch.object(oc, "is_installed", return_value=False):
            with patch.object(ocl, "is_installed", return_value=False):
                status, body = self._chat("/model")
        self.assertEqual(status, 200, body)
        self.assertIn("not installed", body["reply"])

    def test_a_model_like_word_is_not_a_command(self):
        # "/models" is not the /model command: it falls through to a normal
        # run, which needs a usable backend. The default backend is now the
        # built-in engine, so the gate is patched shut on all three to pin
        # the fall-through contract — a backend error, never a /model
        # command reply — identically on every machine.
        with patch.object(bi, "is_installed", return_value=False):
            with patch.object(oc, "is_installed", return_value=False):
                with patch.object(ocl, "is_installed", return_value=False):
                    status, body = self._chat("/models")
        self.assertEqual(status, 500, body)
        self.assertIn("provider key", body["error"])
        self.assertNotIn("reply", body)


if __name__ == "__main__":
    unittest.main()
