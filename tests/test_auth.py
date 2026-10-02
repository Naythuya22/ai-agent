"""core/auth.py — Session 23 (Q2/Q3): opt-in shared-secret guard."""
from _harness import bootstrap

bootstrap()

import os  # noqa: E402
import stat  # noqa: E402
import tempfile  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402

from _harness import get_app  # noqa: E402
from core import auth  # noqa: E402

client = get_app().test_client()

TOKEN = "test-secret-token-42"
V1_BODY = {"messages": [{"role": "user", "content": "hi"}]}


class _EnvIsolation(unittest.TestCase):
    """Both classes start from 'no token anywhere' and restore on the way out."""

    def setUp(self):
        self._env = os.environ.pop(auth.ENV_TOKEN, None)
        self.addCleanup(self._restore_env)
        self._file = auth._token_file()
        self._saved_file = None
        if self._file.exists():
            self._saved_file = self._file.read_text(encoding="utf-8")
            self._file.unlink()
        self.addCleanup(self._restore_file)

    def _restore_env(self):
        os.environ.pop(auth.ENV_TOKEN, None)
        if self._env is not None:
            os.environ[auth.ENV_TOKEN] = self._env

    def _restore_file(self):
        os.environ.pop(auth.ENV_TOKEN, None)
        if self._saved_file is None:
            if self._file.exists():
                self._file.unlink()
        else:
            self._file.write_text(self._saved_file, encoding="utf-8")


class TestAuthOffByDefault(_EnvIsolation):
    """No env, no file → guard is a no-op. This is what keeps the 225-test
    suite honest: enforcement only happens when a token actually exists."""

    def test_no_token_configured(self):
        self.assertIsNone(auth.configured_token())

    def test_api_stays_open(self):
        self.assertEqual(client.get("/api/sessions").status_code, 200)

    def test_v1_stays_open(self):
        r = client.post("/v1/chat/completions", json=V1_BODY)
        self.assertEqual(r.status_code, 200)


class TestAuthEnforced(_EnvIsolation):
    """AI_AGENT_TOKEN set → everything except the shell/probes requires it."""

    def setUp(self):
        super().setUp()
        os.environ[auth.ENV_TOKEN] = TOKEN

    def test_api_without_token_is_401(self):
        r = client.get("/api/sessions")
        self.assertEqual(r.status_code, 401)
        self.assertEqual(r.get_json()["error"], "unauthorized")

    def test_x_agent_token_header_is_accepted(self):
        r = client.get("/api/sessions", headers={"X-Agent-Token": TOKEN})
        self.assertEqual(r.status_code, 200)

    def test_bearer_header_is_accepted(self):
        r = client.get("/api/sessions",
                       headers={"Authorization": f"Bearer {TOKEN}"})
        self.assertEqual(r.status_code, 200)

    def test_query_token_is_accepted(self):
        r = client.get(f"/api/sessions?token={TOKEN}")
        self.assertEqual(r.status_code, 200)

    def test_wrong_token_is_401(self):
        for kwargs in ({"headers": {"X-Agent-Token": "nope"}},
                       {"headers": {"Authorization": "Bearer nope"}},
                       {"query_string": {"token": "nope"}}):
            with self.subTest(kwargs=kwargs):
                self.assertEqual(client.get("/api/sessions", **kwargs).status_code,
                                 401)

    def test_v1_requires_a_token(self):
        self.assertEqual(client.post("/v1/chat/completions", json=V1_BODY).status_code,
                         401)
        r = client.post("/v1/chat/completions", json=V1_BODY,
                        headers={"Authorization": f"Bearer {TOKEN}"})
        self.assertEqual(r.status_code, 200)

    def test_memory_write_requires_a_token(self):
        self.assertEqual(client.post("/api/memory",
                                     json={"key": "k", "val": "v"}).status_code,
                         401)

    def test_pwa_shell_and_probes_stay_open(self):
        for path in ("/", "/manifest.json", "/sw.js", "/api/health",
                     "/api/thermal", "/api/sw-version", "/api/net",
                     "/api/mode", "/static/style.css",
                     "/icons/icon-192.png"):
            with self.subTest(path=path):
                self.assertEqual(client.get(path).status_code, 200, path)


class TestTokenHandling(unittest.TestCase):
    """Token file lifecycle (always runs inside the module's temp HOME)."""

    def setUp(self):
        self._env = os.environ.pop(auth.ENV_TOKEN, None)
        self.addCleanup(self._restore)
        self._file = auth._token_file()
        self._saved = None
        if self._file.exists():
            self._saved = self._file.read_text(encoding="utf-8")
            self._file.unlink()

    def _restore(self):
        os.environ.pop(auth.ENV_TOKEN, None)
        if self._saved is None:
            if self._file.exists():
                self._file.unlink()
        else:
            self._file.write_text(self._saved, encoding="utf-8")
        if self._env is not None:
            os.environ[auth.ENV_TOKEN] = self._env

    def test_generate_token_persists_with_0600(self):
        tok = auth.generate_token()
        self.assertTrue(self._file.exists())
        self.assertEqual(stat.S_IMODE(self._file.stat().st_mode), 0o600)
        self.assertGreaterEqual(len(tok), 24)
        self.assertEqual(auth.configured_token(), tok)

    def test_generated_token_is_not_reused_verbatim(self):
        self.assertNotEqual(auth.generate_token(), auth.generate_token())

    def test_env_wins_over_the_file(self):
        self._file.write_text("file-token\n", encoding="utf-8")
        os.environ[auth.ENV_TOKEN] = TOKEN
        self.assertEqual(auth.configured_token(), TOKEN)
        os.environ.pop(auth.ENV_TOKEN, None)
        self.assertEqual(auth.configured_token(), "file-token")

    def test_empty_file_is_ignored(self):
        self._file.write_text("   \n", encoding="utf-8")
        self.assertIsNone(auth.configured_token())


class TestClientLibrariesCarryTheToken(unittest.TestCase):
    """Q3: our own HTTP clients attach Bearer when auth is on (Q2: off →
    no header, so nothing changes for the default test suite)."""

    def setUp(self):
        self._env = os.environ.pop(auth.ENV_TOKEN, None)
        self.addCleanup(self._restore)

    def _restore(self):
        os.environ.pop(auth.ENV_TOKEN, None)
        if self._env is not None:
            os.environ[auth.ENV_TOKEN] = self._env

    def test_http_llm_sends_no_header_when_auth_is_off(self):
        from core import http_llm
        self.assertEqual(http_llm._auth_headers(), {})

    def test_http_llm_sends_bearer_when_auth_is_on(self):
        from core import http_llm
        os.environ[auth.ENV_TOKEN] = TOKEN
        self.assertEqual(http_llm._auth_headers(),
                         {"Authorization": f"Bearer {TOKEN}"})

    def test_opencode_env_carries_the_token(self):
        from core import opencode as oc
        self.assertNotIn("AI_AGENT_TOKEN", oc.env())
        os.environ[auth.ENV_TOKEN] = TOKEN
        self.assertEqual(oc.env().get("AI_AGENT_TOKEN"), TOKEN)

    def test_bridge_config_embeds_the_token_only_when_a_base_is_asked_for(self):
        from core import opencode as oc
        os.environ[auth.ENV_TOKEN] = TOKEN
        self.assertEqual(oc.bridge_config(), {}, "native default must stay empty")
        cfg = oc.bridge_config(base=oc.BRIDGE_BASE)
        opts = cfg["provider"]["bridge"]["options"]
        self.assertEqual(opts["baseURL"], oc.BRIDGE_BASE)
        self.assertEqual(opts["apiKey"], TOKEN)
        self.assertEqual(cfg["model"], f"bridge/{oc.MODEL_ID}")


if __name__ == "__main__":
    unittest.main()
