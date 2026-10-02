"""Connectivity / offline-degradation tests.

Covers core.net (tri-state probe with TTL) and the fail-fast guards in the
search tools. No network access needed: the probe is stubbed, and the HTTP
adapter is exercised against a real local socket.
"""
from __future__ import annotations

import threading
import time
import unittest

from _harness import bootstrap

bootstrap()

from core import net  # noqa: E402


class _NetCase(unittest.TestCase):
    def setUp(self):
        net.reset()
        self._probe = net._probe
        net._probe = lambda: (True, 1.0, None)

    def tearDown(self):
        net._probe = self._probe
        net.reset()


class TestStates(_NetCase):
    def test_online(self):
        self.assertEqual(net.state(), "online")
        self.assertTrue(net.is_online())
        self.assertIsNone(net.describe()["last_error"])

    def test_offline(self):
        net._probe = lambda: (False, None, "Network is unreachable")
        self.assertFalse(net.is_online())
        self.assertEqual(net.state(), "offline")

    def test_unknown_when_probe_raises(self):
        """Fail open: a broken probe must not be reported as 'offline'."""
        def boom():
            raise RuntimeError("probe exploded")
        net._probe = boom
        self.assertTrue(net.is_online(), "a failing probe must not block the app")

    def test_describe_shape(self):
        d = net.describe()
        for key in ("state", "online", "latency_ms", "checked_at", "age_s", "last_error"):
            self.assertIn(key, d)
        self.assertIn(d["state"], (net.ONLINE, net.OFFLINE, net.UNKNOWN))


class TestCache(_NetCase):
    def test_result_is_cached_for_the_ttl(self):
        calls = []

        def counting():
            calls.append(1)
            return True, 2.0, None

        net._probe = counting
        net.state()
        net.state()
        net.state()
        self.assertEqual(len(calls), 1, "probe ran more than once inside the TTL")

    def test_refresh_forces_a_new_probe(self):
        calls = []

        def counting():
            calls.append(1)
            return True, 2.0, None

        net._probe = counting
        net.state()
        net.state(force=True)
        self.assertEqual(len(calls), 2)

    def test_reset_clears_the_cache(self):
        calls = []

        def counting():
            calls.append(1)
            return True, 2.0, None

        net._probe = counting
        net.state()
        net.reset()
        net.state()
        self.assertEqual(len(calls), 2)

    def test_concurrent_checks_probe_once(self):
        """Two threads must not both pay for a probe."""
        calls = []
        gate = threading.Lock()

        def slow():
            with gate:
                calls.append(1)
            time.sleep(0.05)
            return True, 1.0, None

        net._probe = slow
        threads = [threading.Thread(target=net.is_online) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        self.assertFalse(any(t.is_alive() for t in threads), "check() deadlocked")
        self.assertEqual(len(calls), 1, f"probed {len(calls)} times under concurrency")


class TestOfflineNotice(_NetCase):
    def test_english_query_gets_an_english_line(self):
        net._probe = lambda: (False, None, "down")
        msg = net.offline_notice("web_search", "news") or ""
        self.assertTrue(msg.strip(), "notice must not be empty")
        self.assertIn("OFFLINE", msg.upper(), "must state the state plainly")
        self.assertIn("Do NOT retry", msg, "must stop the model retrying")
        self.assertIn("internet connection", msg)
        self.assertIn("news", msg, "must say what it failed to look up")

    def test_burmese_query_gets_a_burmese_line(self):
        net._probe = lambda: (False, None, "down")
        msg = net.offline_notice("web_search", "နေပြည့်သောနေ") or ""
        self.assertIn("မစစ်နိုင်ပါ", msg, "Burmese users must get a Burmese line")

    def test_none_when_online(self):
        self.assertIsNone(net.offline_notice("web_search", "news"))


class TestToolFailFast(_NetCase):
    def setUp(self):
        super().setUp()
        import tools.web_search as ws
        from tools.smart_search import smart_search

        self.smart_search = smart_search
        self.ws = ws

    def test_smart_search_offline_is_instant_and_honest(self):
        net._probe = lambda: (False, None, "down")
        t0 = time.time()
        out = self.smart_search("news")
        elapsed = time.time() - t0
        self.assertLess(elapsed, 0.5, f"took {elapsed:.2f}s — not fail-fast")
        self.assertNotIn("ရှာမတွေ့ပါ", out, "must not claim 'not found' when offline")
        self.assertIn("OFFLINE", out.upper())

    def test_web_search_offline_is_instant(self):
        net._probe = lambda: (False, None, "down")
        t0 = time.time()
        out = self.ws.web_search("anything", 3)
        self.assertLess(time.time() - t0, 0.5)
        self.assertIn("OFFLINE", out.upper())

    def test_web_fetch_offline_is_instant(self):
        net._probe = lambda: (False, None, "down")
        t0 = time.time()
        out = self.ws.web_fetch("https://example.com")
        self.assertLess(time.time() - t0, 0.5)
        self.assertIn("OFFLINE", out.upper())

    def test_url_safety_checked_before_connectivity(self):
        """A blocked URL must report the block, not the network state."""
        net._probe = lambda: (True, 1.0, None)
        for url in ("http://127.0.0.1:8080/x", "http://localhost/x",
                    "http://169.254.169.254/latest/meta-data/", "file:///etc/passwd"):
            out = self.ws.web_fetch(url)
            self.assertNotIn("OFFLINE", out.upper(), f"{url} should be refused outright")

    def test_online_path_does_not_say_offline(self):
        net._probe = lambda: (True, 1.0, None)
        self.assertNotIn("OFFLINE", self.ws.web_fetch("http://10.0.0.1:1/x").upper())


class TestHttpAdapter(unittest.TestCase):
    """Exercise the stdlib urllib adapter over a real socket."""

    @classmethod
    def setUpClass(cls):
        import http.server
        import socketserver

        page = (
            b"<html><body><table>"
            b"<tr><td><a class='result-link' href='https://x.dev/p'>X Dev Page</a></td></tr>"
            b"<tr><td class='result-snippet'>A snippet about asyncio.</td></tr>"
            b"</table></body></html>"
        )

        class Handler(http.server.BaseHTTPRequestHandler):
            close_connection = True  # don't hold the socket after the response

            def do_GET(self):
                if self.path.startswith("/boom"):
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(b"no")
                    return
                if self.path.startswith("/redir"):
                    self.send_response(302)
                    self.send_header("Location", "/ok")
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)

            def log_message(self, *a):
                pass

        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True
            block_on_close = False

        cls.srv = Server(("127.0.0.1", 0), Handler)
        cls.port = cls.srv.server_address[1]
        cls.thread = threading.Thread(target=cls.srv.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        cls.thread.join(timeout=5)

    def setUp(self):
        import tools.web_search as ws

        self.ws = ws

    @staticmethod
    def _parse(ws, html):
        p = ws._DDGParser()
        p.feed(html)
        return p.results

    def test_status_and_body(self):
        r = self.ws._http_get(self.base + "/ok")
        self.assertEqual(r.status_code, 200)
        self.assertIn("X Dev Page", r.text)

    def test_headers_are_case_insensitive(self):
        r = self.ws._http_get(self.base + "/ok")
        self.assertTrue(r.headers.get("Content-Type", "").startswith("text/html"))
        self.assertTrue(r.headers.get("content-type", "").startswith("text/html"))

    def test_charset_is_honoured(self):
        r = self.ws._http_get(self.base + "/ok")
        self.assertEqual(r.encoding, "utf-8")

    def test_redirects_are_followed(self):
        self.assertEqual(self.ws._http_get(self.base + "/redir").status_code, 200)

    def test_http_error_is_a_response_not_an_exception(self):
        self.assertEqual(self.ws._http_get(self.base + "/boom").status_code, 500)

    def test_transport_failure_raises_network_error(self):
        with self.assertRaises(self.ws.NetworkError):
            self.ws._http_get("http://127.0.0.1:1/x", timeout=1)

    def test_max_bytes_caps_the_read(self):
        # reads max_bytes+1 so the caller can detect truncation
        r = self.ws._http_get(self.base + "/ok", max_bytes=10)
        self.assertLessEqual(len(r._body), 11)

    def test_parser_extracts_title_url_snippet(self):
        results = self._parse(self.ws, self.ws._http_get(self.base + "/ok").text)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "X Dev Page")
        self.assertEqual(results[0]["url"], "https://x.dev/p")
        self.assertIn("asyncio", results[0]["snippet"])

    def test_parser_ignores_non_result_markup(self):
        self.assertEqual(self._parse(self.ws, "<html><body>nope</body></html>"), [])

    def test_parser_needs_a_snippet(self):
        """A link with no snippet is not a result."""
        html = ("<table><tr><td><a class='result-link' href='https://a.dev'>A</a></td></tr>"
                "</table>")
        self.assertEqual(self._parse(self.ws, html), [])

    def test_search_output_is_wrapped_as_untrusted(self):
        saved, self.ws._DDG_ENDPOINT = self.ws._DDG_ENDPOINT, self.base + "/ok"
        try:
            out = self.ws.web_search("asyncio", 3)
        finally:
            self.ws._DDG_ENDPOINT = saved
        self.assertIn("<untrusted>", out)
        self.assertIn("X Dev Page", out)
        self.assertIn("https://x.dev/p", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)