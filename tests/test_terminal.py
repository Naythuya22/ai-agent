"""routes/terminal.py — the same-origin ttyd proxy + WS relay (P1-B2B-1/3a).

Every test here is self-contained: no socket, no ttyd, no 127.0.0.1:8081.
The upstream is always replaced with unittest.mock, so a machine with ttyd
running (or without it) sees the same suite.

Covered: path/query validation, the 502 when ttyd is down, streaming,
header stripping, the /terminal/ auth exemption, the argv lock on
ttyd's --base-path, and the /terminal/ws relay contract.
"""
from __future__ import annotations

import os
import threading
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from _harness import bootstrap, get_app

bootstrap()

from core import auth  # noqa: E402
from routes import terminal  # noqa: E402
from simple_websocket import ConnectionClosed  # noqa: E402

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


class _SockThreading:
    """What `routes.terminal.threading` looks like to the WS tests.

    Threads stay real (a fake would also have to fake Thread.__init__'s own
    internals); Events are recorded here instead, so a fake socket can flip
    the relay's stop flag without reaching into terminal_ws's stack frame.
    """

    Thread = threading.Thread

    def __init__(self, pre_set: bool = False):
        self.events = []
        self._pre_set = pre_set

    def Event(self):
        e = threading.Event()
        if self._pre_set:
            e.set()
        self.events.append(e)
        return e

    def stop(self):
        """Called by a fake socket: stop the relay from the inside."""
        if self.events:
            self.events[-1].set()


class _FakeBrowserWS:
    """The browser end: hands the relay its input, records what comes back."""

    def __init__(self, incoming=(), on_send=None):
        self._incoming = list(incoming)
        self.sent = []
        self.closed = []
        self._on_send = on_send

    def receive(self, timeout=None):
        if self._incoming:
            return self._incoming.pop(0)
        return None  # idle: _recv() treats this as "nothing yet"

    def send(self, data):
        self.sent.append(data)
        if self._on_send is not None:
            self._on_send()

    def close(self, code=None, message=None):
        self.closed.append((code, message))


class _FakeTtydWS:
    """The ttyd end: echoes b"hi" only after the browser byte has landed."""

    def __init__(self, url=None, headers=None):
        self.url = url
        self.headers = headers
        self.sent = []
        self.closed = False
        self._delivered = False

    def send(self, data):
        self.sent.append(data)

    def receive(self, timeout=None):
        if not self.sent:
            return None  # nothing to echo back yet
        if not self._delivered:
            self._delivered = True
            return b"hi"
        raise ConnectionClosed()  # swallowed by the relay's _recv()

    def close(self, *args, **kwargs):
        self.closed = True


class TestWebSocketRelay(unittest.TestCase):
    """`/terminal/ws` relays bytes; no test here ever opens a real socket."""

    def _run_relay(self, ws, shim=None, path="/terminal/ws", timeout=3.0):
        """Drive terminal_ws on a worker thread so a stuck relay fails fast."""
        shim = _SockThreading() if shim is None else shim
        failure = []

        def _run():
            with app.test_request_context(path):
                try:
                    terminal.terminal_ws(ws)
                except BaseException as exc:  # report to the main thread
                    failure.append(exc)

        with patch.object(terminal, "threading", shim):
            worker = threading.Thread(target=_run, daemon=True)
            worker.start()
            worker.join(timeout)

        self.assertFalse(worker.is_alive(), "the relay never stopped")
        if failure:
            raise failure[0]
        return shim

    def test_ws_route_registered(self):
        self.assertIn("/terminal/ws", str(app.url_map))

    def test_ws_closes_on_ttyd_down(self):
        ws = _FakeBrowserWS()
        with patch("simple_websocket.Client", side_effect=RuntimeError("down")):
            self._run_relay(ws)
        self.assertEqual(ws.closed, [(1011, "ttyd unavailable")])

    def test_ws_bridges_bytes(self):
        shim = _SockThreading()
        ws = _FakeBrowserWS(incoming=[b"hi"], on_send=shim.stop)
        upstreams = []

        def fake_client(url, headers=None):
            up = _FakeTtydWS(url, headers)
            upstreams.append(up)
            return up

        with patch("simple_websocket.Client", fake_client):
            self._run_relay(ws, shim=shim)

        self.assertEqual(len(upstreams), 1)
        up = upstreams[0]
        self.assertEqual(up.sent, [b"hi"], "browser byte never reached ttyd")
        self.assertEqual(ws.sent, [b"hi"], "ttyd byte never reached the browser")
        self.assertTrue(up.closed, "the upstream socket was never closed")

    def test_ws_query_is_forwarded(self):
        # stop is already set: the relay runs end to end and exits at once.
        shim = _SockThreading(pre_set=True)
        urls = []

        def fake_client(url, headers=None):
            urls.append(url)
            return _FakeTtydWS(url, headers)

        with patch("simple_websocket.Client", fake_client):
            self._run_relay(_FakeBrowserWS(), shim=shim, path="/terminal/ws?arg=x")

        self.assertEqual(len(urls), 1, "the upstream client was never built")
        self.assertTrue(urls[0].endswith("?arg=x"),
                        f"query string dropped: {urls[0]}")
        self.assertEqual(urls[0], terminal.TTYD_WS + "?arg=x")


if __name__ == "__main__":
    unittest.main()
