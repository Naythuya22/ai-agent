"""routes/terminal.py — the same-origin ttyd proxy (P1-B2B-1 / P1-B2B-2).

Every test here is self-contained: no socket, no ttyd, no 127.0.0.1:8081.
The upstream is always replaced with unittest.mock, so a machine with ttyd
running (or without it) sees the same suite.

Covered: path/query validation, the 502 when ttyd is down, streaming,
header stripping, the /terminal/ auth exemption, and the argv lock on
ttyd's --base-path.
"""
from __future__ import annotations

import os
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from _harness import bootstrap, get_app

bootstrap()

from core import auth  # noqa: E402

app = get_app()
client = app.test_client()

CODE_PY = Path(__file__).resolve().parent.parent / "routes" / "code.py"


class _FakeUpstream:
    """What urlopen() gives back when ttyd answers: read/close/headers/status."""

    def __init__(self, body: bytes = b"hi"):
        self._body = body
        self.headers: dict = {}
        self.status = 200
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        # Like a socket: the body once, then EOF, or the proxy loops forever.
        data, self._body = self._body, b""
        return data

    def close(self) -> None:
        self.closed = True


class _EnvIsolation(unittest.TestCase):
    """Start from 'no token anywhere' and put it back on the way out."""

    def setUp(self):
        self._env = os.environ.pop(auth.ENV_TOKEN, None)
        self.addCleanup(self._restore)

    def _restore(self):
        os.environ.pop(auth.ENV_TOKEN, None)
        if self._env is not None:
            os.environ[auth.ENV_TOKEN] = self._env


class TestPathValidation(_EnvIsolation):
    """The proxy must not become a path-traversal or amplification vector."""

    def test_rejects_dotdot_in_path(self):
        # URL-encoded so the .. survives routing and reaches _validate_path.
        r = client.get("/terminal/%2e%2e/etc/passwd")
        self.assertEqual(r.status_code, 400)
        self.assertIn("..", r.get_json()["error"])

    def test_rejects_backslash(self):
        r = client.get("/terminal/foo%5cbar")
        self.assertEqual(r.status_code, 400)
        self.assertIn("backslash", r.get_json()["error"])

    def test_rejects_long_query(self):
        r = client.get("/terminal/?x=" + "a" * 5000)
        self.assertIn(r.status_code, (400, 414))
        self.assertIn("error", r.get_json())


class TestProxying(_EnvIsolation):
    """The upstream is mocked: nothing here ever opens a socket."""

    def test_ttyd_down_returns_502(self):
        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.URLError("down")):
            r = client.get("/terminal/")
        self.assertEqual(r.status_code, 502)
        body = r.get_json()
        self.assertEqual(body["error"], "ttyd not running")

    def test_proxies_get_streams(self):
        fake = _FakeUpstream(b"hi")
        with patch("urllib.request.urlopen", return_value=fake) as up:
            r = client.get("/terminal/index.html")
            data = r.get_data()
        self.assertEqual(r.status_code, 200)
        self.assertEqual(data, b"hi")
        self.assertEqual(up.call_count, 1)
        self.assertTrue(fake.closed, "the upstream was never closed")

    def test_strips_agent_token(self):
        with patch("urllib.request.urlopen",
                   return_value=_FakeUpstream()) as up:
            client.get("/terminal/", headers={"X-Agent-Token": "secret",
                                              "Host": "evil.example",
                                              "Content-Type": "text/plain"})
        req = up.call_args[0][0]
        sent = {k.lower() for k in req.headers}
        self.assertNotIn("x-agent-token", sent,
                         "the token must not reach ttyd")
        self.assertNotIn("host", sent, "Host must not be forwarded")
        self.assertIn("content-type", sent)


class TestAuthExemption(_EnvIsolation):
    """An iframe cannot send X-Agent-Token: /terminal/* must be open."""

    def test_exempt_from_auth(self):
        os.environ[auth.ENV_TOKEN] = "secret"
        with patch("urllib.request.urlopen", return_value=_FakeUpstream()):
            r = client.get("/terminal/")
        self.assertEqual(r.status_code, 200)
        # The token really was in force, so the 200 is the exemption.
        self.assertEqual(client.get("/api/sessions").status_code, 401)


class TestWiring(unittest.TestCase):
    """ttyd must keep serving the proxy under the same /terminal base path."""

    def test_argv_includes_base_path(self):
        src = CODE_PY.read_text(encoding="utf-8")
        i = src.index('"ttyd"')
        window = src[i:i + 400]
        self.assertIn("--base-path", window)
        self.assertIn("/terminal", window)


if __name__ == "__main__":
    unittest.main()
