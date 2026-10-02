"""Route-level smoke tests: every endpoint answers, and answers sanely.

Run:  python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import unittest

from contextlib import closing

from _harness import bootstrap, get_app

MODEL = bootstrap()
app = get_app()
client = app.test_client()

# (method, path, query, json-body, expected-status)
# expected None = "any status, but must not raise"
CASES = [
    ("GET", "/", None, None, 200),
    ("GET", "/api/health", None, None, 200),
    ("GET", "/api/sw-version", None, None, 200),
    ("GET", "/api/thermal", None, None, 200),
    ("GET", "/api/net", None, None, 200),
    ("GET", "/api/net", {"refresh": "1"}, None, 200),
    ("GET", "/manifest.json", None, None, 200),
    ("GET", "/sw.js", None, None, 200),
    ("GET", "/v1/models", None, None, 200),
    ("GET", "/api/mode", None, None, 200),
    ("GET", "/api/ctx", None, None, 200),
    ("GET", "/api/memory", None, None, 200),
    ("GET", "/api/profile", None, None, 200),
    ("GET", "/api/sessions", None, None, 200),
    ("GET", "/api/sessions", {"mode": "chat"}, None, 200),
    ("GET", "/api/sessions/nope", None, None, 404),
    ("GET", "/api/history", None, None, 200),
    ("GET", "/api/skills", None, None, 200),
    ("GET", "/api/skills/nope", None, None, 404),
    ("GET", "/api/code/projects", None, None, 200),
    ("GET", "/api/code/browse", None, None, 400),  # default dir absent in test HOME
    ("GET", "/api/code/browse", {"path": "/definitely/not/here"}, None, 400),
    ("GET", "/api/code/web/history", None, None, 200),
    ("GET", "/api/code/web/status", None, None, 200),
    ("GET", "/icons/icon-192.png", None, None, 200),
    ("GET", "/static/style.css", None, None, 200),  # served from web/
    ("GET", "/static/../../server.py", None, None, 404),  # no path traversal
    ("GET", "/nope-does-not-exist", None, None, 404),

    # Missing / empty payload must be a clean 400, never a 500.
    ("POST", "/api/chat", None, {}, 400),
    ("POST", "/api/chat", {"message": "   "}, None, 400),
    ("POST", "/api/agent", None, {}, 400),
    ("POST", "/api/agent/stream", None, {}, 400),
    ("POST", "/api/chat/stream", None, {}, 400),
    ("POST", "/api/remember", None, {}, 400),
    ("POST", "/api/profile", None, {}, 400),
    ("POST", "/api/code", None, {}, 400),
    ("POST", "/api/code/web/run", None, {}, 400),
    ("POST", "/api/code/web/run", {"message": "/project list"}, None, None),
    ("POST", "/api/code/web/run/stream", None, {}, 400),
    ("POST", "/api/skills/create", None, {}, 400),
    ("POST", "/api/skills/create", {"name": "x"}, None, 400),
    ("POST", "/api/skills/x/run", None, {}, 400),
    ("POST", "/api/skills/x/run", None, {"args": "a", "mode": "bogus"}, 400),
    ("POST", "/api/skills/x/run", None, {"args": "a", "mode": "chat"}, 404),
    ("POST", "/api/context/1234", None, None, 400),
    ("POST", "/api/context/abc", None, None, 400),
    ("POST", "/v1/chat/completions", None, {}, 400),
    ("POST", "/v1/chat/completions", None, {"messages": [{"role": "user", "content": "hi"}]}, 200),

    # Write endpoints
    ("POST", "/api/clear", None, None, 200),
    ("POST", "/api/summary", None, {}, 400),  # empty session: "too few messages"
    ("POST", "/api/new", None, {}, None),
    ("POST", "/api/code/web/clear", None, None, 200),
    ("POST", "/api/code/projects", None, {}, 400),
    ("POST", "/api/code/projects/zz/activate", None, None, 404),
    ("DELETE", "/api/code/projects/zz", None, None, None),
    ("DELETE", "/api/skills/nope", None, None, 404),

    # Wrong method must be 405, never silently 200.
    ("POST", "/api/health", None, None, 405),
    ("POST", "/api/net", None, None, 405),
    ("GET", "/api/agent", None, None, 405),
]


class TestRoutes(unittest.TestCase):
    def test_all_cases(self):
        failures = []
        for method, path, query, body, expect in CASES:
            kwargs = {}
            if query:
                kwargs["query_string"] = query
            if body is not None:
                kwargs["json"] = body
            label = f"{method} {path}" + (f" ?{query}" if query else "")
            try:
                with closing(client.open(path, method=method, **kwargs)) as r:
                    if expect is not None and r.status_code != expect:
                        failures.append(
                            f"{label}: expected {expect}, got {r.status_code} "
                            f"body={r.get_data(as_text=True)[:120]!r}"
                        )
            except Exception as e:  # noqa: BLE001
                failures.append(f"{label}: RAISED {type(e).__name__}: {e}")
                continue
        self.assertEqual(failures, [], "\n".join(failures))

    def test_unknown_route_is_404_json_or_html(self):
        with closing(client.get("/totally/unknown")) as r:
            self.assertEqual(r.status_code, 404)

    def test_api_net_shape(self):
        with closing(client.get("/api/net")) as r:
            self.assertEqual(r.status_code, 200)
            body = r.get_json()
        for key in ("state", "online", "checked_at", "age_s"):
            self.assertIn(key, body)
        self.assertIn(body["state"], ("online", "offline", "unknown"))

    def test_health_is_ok(self):
        with closing(client.get("/api/health")) as r:
            body = r.get_json()
        self.assertEqual(body.get("status"), "ok", body)

    def test_sw_version_matches_sw_js(self):
        """The service worker must advertise the same cache name it serves."""
        import re

        with closing(client.get("/api/sw-version")) as r:
            version = r.get_json()["version"]
        with closing(client.get("/sw.js")) as r:
            js = r.get_data(as_text=True)
        m = re.search(r"const CACHE = 'ai-agent-([^']+)'", js)
        self.assertIsNotNone(m, "sw.js has no CACHE constant")
        self.assertEqual(version, m.group(1))

    def test_index_html_is_served(self):
        with closing(client.get("/")) as r:
            body = r.get_data(as_text=True)
        self.assertIn("<html", body.lower())
        self.assertIn("/api/net", body, "UI should poll /api/net")

    def test_every_json_error_is_parseable(self):
        """A client parsing errors must never hit invalid JSON."""
        for method, path, _q, body, _e in CASES:
            kwargs = {"json": body} if body is not None else {}
            with closing(client.open(path, method=method, **kwargs)) as r:
                ctype = (r.headers.get("Content-Type") or "")
                if "json" in ctype:
                    try:
                        json.loads(r.get_data(as_text=True))
                    except Exception as e:  # noqa: BLE001
                        self.fail(f"{method} {path}: bad JSON ({e})")


if __name__ == "__main__":
    unittest.main(verbosity=2)