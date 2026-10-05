"""Tool error propagation.

A search backend that raised used to be swallowed by a bare ``except: continue``,
which handed the model a short result set and no idea anything had failed — the
shape of a hallucination. The failure is now recorded and shown.

Run:  sh tests/run.sh -p "test_tool_errors.py"
"""
from __future__ import annotations

import unittest

from _harness import bootstrap, get_app  # noqa: F401

bootstrap()

import config  # noqa: E402

# `from tools import smart_search` would hand back the FUNCTION, not the module:
# tools/__init__.py does `from tools.smart_search import smart_search`, which
# rebinds the name on the package. Import the module itself.
import importlib
ss = importlib.import_module("tools.smart_search")


class _GoodBackend:
    """A backend that answers."""

    prefer_raw = False

    def available(self):
        return True

    def search(self, q, max_results=5):
        return [{"title": "a result", "url": "https://example.com"}]


class _BadBackend:
    """A backend that dies, quoting a credential in the error body."""

    prefer_raw = False

    def __init__(self, message="backend exploded"):
        self.message = message

    def available(self):
        return True

    def search(self, q, max_results=5):
        raise RuntimeError(self.message)


class _MissingBackend:
    """Unavailable is not an error — it is simply not used."""

    prefer_raw = False

    def available(self):
        return False

    def search(self, q, max_results=5):  # pragma: no cover - never reached
        raise AssertionError("must not be called")


class CollectResultsTests(unittest.TestCase):
    def test_collect_returns_results_and_errors(self):
        out, errors = ss._collect_results(
            [_GoodBackend(), _BadBackend()], "q", "q", per_backend=5, max_total=10)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["status"], "failed")
        self.assertEqual(errors[0]["backend"], "_BadBackend")
        self.assertIn("RuntimeError", errors[0]["error"])

    def test_unavailable_backend_is_not_an_error(self):
        out, errors = ss._collect_results(
            [_MissingBackend(), _GoodBackend()], "q", "q", 5, 10)
        self.assertEqual(errors, [])
        self.assertEqual(len(out), 1)

    def test_error_text_is_capped(self):
        out, errors = ss._collect_results(
            [_BadBackend("x" * 5000)], "q", "q", 5, 10)
        self.assertLessEqual(len(errors[0]["error"]), ss._MAX_ERROR_CHARS)

    def test_error_text_is_one_line(self):
        out, errors = ss._collect_results(
            [_BadBackend("line one\nline two")], "q", "q", 5, 10)
        self.assertNotIn("\n", errors[0]["error"])


class SmartSearchTests(unittest.TestCase):
    """End-to-end through smart_search() itself, with the network stubbed out."""

    def setUp(self):
        # category="coding" skips classify() and is_burmese(); a non-empty raw
        # result set skips the DDG fallback, so nothing here touches a socket.
        self._saved = {name: getattr(ss, name) for name in
                       ("offline_notice", "translate", "route", "clean",
                        "pack", "sources_line")}
        ss.offline_notice = lambda *a, **k: None
        ss.translate = lambda q, llm=None: q
        ss.clean = lambda raw, max_keep=5: list(raw)
        ss.pack = lambda items, q: "<context>%d</context>" % len(items)
        ss.sources_line = lambda items: ""

    def tearDown(self):
        for name, fn in self._saved.items():
            setattr(ss, name, fn)

    def _run(self, backends):
        ss.route = lambda cat: backends
        return ss.smart_search("hello", category="coding", llm=None)

    def test_smart_search_returns_error_records_on_backend_failure(self):
        out = self._run([_GoodBackend(), _BadBackend()])
        self.assertIn("TOOL_ERRORS:", out)
        self.assertIn("_BadBackend", out)
        self.assertIn("failed", out)
        # the useful half still comes through
        self.assertIn("<context>1</context>", out)

    def test_tool_errors_visible_flag_can_disable(self):
        saved = config.TOOL_ERRORS_VISIBLE
        config.TOOL_ERRORS_VISIBLE = False
        try:
            out = self._run([_GoodBackend(), _BadBackend()])
            self.assertNotIn("TOOL_ERRORS:", out)
            self.assertIn("<context>1</context>", out)
        finally:
            config.TOOL_ERRORS_VISIBLE = saved

    def test_sanitize_strips_api_key_from_tool_error(self):
        secret = "sk-live-ABCDEFGHIJKLMNOP"
        out = self._run([_GoodBackend(), _BadBackend("upstream said %s" % secret)])
        self.assertIn("TOOL_ERRORS:", out)
        self.assertNotIn(secret, out, "a key in an error body must not reach the model")
        self.assertIn("[redacted]", out)

    def test_no_error_block_when_everything_worked(self):
        out = self._run([_GoodBackend()])
        self.assertNotIn("TOOL_ERRORS:", out)

    def test_return_type_is_unchanged(self):
        # smart_search is registered as a tool through tool_from_function; its
        # schema says str, so this must never become a tuple.
        self.assertIsInstance(self._run([_GoodBackend()]), str)
        self.assertIsInstance(self._run([_GoodBackend(), _BadBackend()]), str)


if __name__ == "__main__":
    unittest.main()