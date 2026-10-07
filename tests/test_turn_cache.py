"""Per-turn tool result cache.

The tool loops already refuse to run the same call twice in a turn, but they
answer the repeat with a bare "(skipped: …)" — so the model loses the data it
asked for and asks again. This returns the data instead, tagged "[cached]".

Scope is one user turn, instance state only, read-only tools only.

Run:  sh tests/run.sh -p "test_turn_cache.py"
"""
from __future__ import annotations

import unittest

from _harness import bootstrap, get_app  # noqa: F401

bootstrap()

import config  # noqa: E402
from modes import agent as agent_mod  # noqa: E402


def _make_tool(name, calls, result="CONTENT"):
    """A real function (it needs __name__) that records every execution."""
    def _fn(**kwargs):
        calls.append(name)
        return "%s(%s)" % (result, sorted(kwargs.items()))
    _fn.__name__ = name
    return _fn


def _bare_mode():
    m = agent_mod.AgentMode.__new__(agent_mod.AgentMode)
    m._messages = []
    m._turn_cache = {}
    m._turn_cache_order = []
    return m


class TurnCacheTests(unittest.TestCase):
    def setUp(self):
        self._real_tools = agent_mod.ALL_TOOLS
        self.calls = []

    def tearDown(self):
        agent_mod.ALL_TOOLS = self._real_tools

    def _mode_with(self, name, result="CONTENT"):
        tool = _make_tool(name, self.calls, result)
        agent_mod.ALL_TOOLS = [tool]
        return _bare_mode()

    def test_turn_cache_hits_on_repeated_read(self):
        m = self._mode_with("read_file")
        first = m._execute_tool("read_file", {"path": "/tmp/a.txt"})
        second = m._execute_tool("read_file", {"path": "/tmp/a.txt"})
        self.assertEqual(len(self.calls), 1, "the tool must run once")
        self.assertTrue(second.startswith("[cached] "), second)
        self.assertIn(first, second)

    def test_turn_cache_key_ignores_arg_order(self):
        m = self._mode_with("read_file")
        m._execute_tool("read_file", {"path": "/tmp/a", "start_line": 1})
        m._execute_tool("read_file", {"start_line": 1, "path": "/tmp/a"})
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(m._cache_key("read_file", {"a": 1, "b": 2}),
                         m._cache_key("read_file", {"b": 2, "a": 1}))

    def test_turn_cache_cleared_between_turns(self):
        m = self._mode_with("read_file")
        m._execute_tool("read_file", {"path": "/tmp/a.txt"})
        self.assertEqual(len(m._turn_cache), 1)
        m._clear_turn_cache()
        self.assertEqual(m._turn_cache, {})
        self.assertEqual(m._turn_cache_order, [])
        out = m._execute_tool("read_file", {"path": "/tmp/a.txt"})
        self.assertEqual(len(self.calls), 2, "a new turn must re-read")
        self.assertFalse(out.startswith("[cached] "))

    def test_turn_cache_not_used_for_write_file(self):
        m = self._mode_with("write_file", result="WROTE")
        m._execute_tool("write_file", {"path": "/tmp/a", "content": "x"})
        out = m._execute_tool("write_file", {"path": "/tmp/a", "content": "x"})
        self.assertEqual(len(self.calls), 2, "a write must never be replayed")
        self.assertFalse(out.startswith("[cached] "))
        self.assertEqual(m._turn_cache, {})

    def test_turn_cache_not_used_for_run_python(self):
        m = self._mode_with("run_python", result="RAN")
        m._execute_tool("run_python", {"code": "print(1)"})
        out = m._execute_tool("run_python", {"code": "print(1)"})
        self.assertEqual(len(self.calls), 2, "code must never be replayed")
        self.assertFalse(out.startswith("[cached] "))

    def test_turn_cache_evicts_oldest_at_cap(self):
        m = self._mode_with("read_file")
        for i in range(m._CACHE_CAP):
            m._cache_put("read_file", {"path": "/tmp/%d" % i}, "body-%d" % i)
        self.assertEqual(len(m._turn_cache), m._CACHE_CAP)
        first_key = m._cache_key("read_file", {"path": "/tmp/0"})
        self.assertIn(first_key, m._turn_cache)

        m._cache_put("read_file", {"path": "/tmp/new"}, "body-new")
        self.assertEqual(len(m._turn_cache), m._CACHE_CAP, "the cap is a cap")
        self.assertNotIn(first_key, m._turn_cache, "the oldest entry must go")
        self.assertEqual(len(m._turn_cache_order), m._CACHE_CAP)
        self.assertIn(m._cache_key("read_file", {"path": "/tmp/new"}), m._turn_cache)

    def test_turn_cache_disabled_by_flag(self):
        m = self._mode_with("read_file")
        saved = config.AGENT_TURN_CACHE
        config.AGENT_TURN_CACHE = False
        try:
            out = m._execute_tool("read_file", {"path": "/tmp/a.txt"})
            self.assertEqual(len(self.calls), 1)
            self.assertFalse(out.startswith("[cached] "))
            self.assertEqual(m._turn_cache, {})
        finally:
            config.AGENT_TURN_CACHE = saved

    def test_failed_call_is_not_cached(self):
        def boom(**kwargs):
            self.calls.append("boom")
            return "Error: file not found: /nope"
        boom.__name__ = "read_file"
        agent_mod.ALL_TOOLS = [boom]
        m = _bare_mode()
        m._execute_tool("read_file", {"path": "/nope"})
        self.assertEqual(m._turn_cache, {},
                         "a transient failure must not stick for the turn")

    def test_unknown_tool_is_never_cached(self):
        m = _bare_mode()
        self.assertTrue(m._execute_tool("nope", {}).startswith("Error:"))
        self.assertEqual(m._turn_cache, {})

    def test_cached_text_falls_back_to_the_callers_text(self):
        m = self._mode_with("read_file")
        self.assertEqual(m._cached_text("read_file", {"path": "/x"}, "fallback"),
                         "fallback")


class _CloudishMode(agent_mod.AgentMode):
    """Just enough AgentMode to reach the turn-begin bookkeeping."""

    def _mode(self):
        return "cloud"

    def _is_http(self):
        return False

    def _try_cloud(self, user_input, system=None):
        return "answered"


class TurnBoundaryTests(unittest.TestCase):
    def test_handle_clears_the_cache(self):
        m = _CloudishMode.__new__(_CloudishMode)
        m._messages = []
        m._turn_cache = {"k": "v"}
        m._turn_cache_order = ["k"]
        self.assertEqual(m.handle("hi"), "answered")
        self.assertEqual(m._turn_cache, {},
                         "every user turn starts with an empty cache")
        self.assertEqual(m._turn_cache_order, [])

    def test_reset_clears_the_cache(self):
        m = _CloudishMode.__new__(_CloudishMode)
        m._messages = []
        m._conv = None
        m._store = None
        m.turns = 3
        m._turn_cache = {"k": "v"}
        m._turn_cache_order = ["k"]
        m.reset()
        self.assertEqual(m._turn_cache, {})
        self.assertEqual(m._turn_cache_order, [])

    def test_cache_is_instance_state_not_a_module_global(self):
        a, b = _bare_mode(), _bare_mode()
        a._cache_put("read_file", {"path": "/x"}, "A")
        self.assertEqual(b._turn_cache, {},
                         "one turn's cache must never be visible to another")

class _V1RouteCacheTests(unittest.TestCase):
    """Fix 4.2i-B (H2): the v1 SSE route must clear the per-turn cache.

    The route needs a full Flask request context to run, so this test
    reads the source of routes.chat and asserts the call site is present
    in both SSE routes (v1 = parity fix, v2 = already correct).
    """

    def test_both_sse_routes_clear_the_turn_cache(self):
        import inspect
        from routes import chat as chat_routes
        src = inspect.getsource(chat_routes)
        self.assertIn("Batch 4.2i-B (H2)", src,
                      "v1 route must carry the 4.2i-B (H2) fix marker")
        self.assertGreaterEqual(
            src.count("_clear_turn_cache()"), 2,
            "both SSE routes must clear the per-turn tool cache")


class _GuardedCacheTests(unittest.TestCase):
    """Fix 4.2i-B (H3): the serial path caches the guarded text.

    A raw result and its guarded form are only distinguishable when the
    guard actually changes the text. Rather than depend on the specific
    redaction regex, this test asserts the invariant directly: the value
    stored in the cache equals the value the caller received.
    """

    def setUp(self):
        self._real_tools = agent_mod.ALL_TOOLS

    def tearDown(self):
        agent_mod.ALL_TOOLS = self._real_tools

    def test_cache_value_equals_the_returned_value(self):
        calls = []
        tool = _make_tool("read_file", calls, result="PLAIN BODY")
        agent_mod.ALL_TOOLS = [tool]
        m = _bare_mode()
        first = m._execute_tool("read_file", {"path": "/tmp/x"})
        key = m._cache_key("read_file", {"path": "/tmp/x"})
        self.assertIn(key, m._turn_cache,
                      "read_file result must be cached")
        self.assertEqual(
            m._turn_cache[key], first,
            "cache must hold exactly the guarded text the caller received")

    def test_cache_hit_still_returns_the_guarded_text(self):
        calls = []
        tool = _make_tool("read_file", calls, result="PLAIN BODY")
        agent_mod.ALL_TOOLS = [tool]
        m = _bare_mode()
        first = m._execute_tool("read_file", {"path": "/tmp/x"})
        second = m._execute_tool("read_file", {"path": "/tmp/x"})
        self.assertEqual(len(calls), 1, "the tool must run once")
        self.assertTrue(second.startswith("[cached] "), second)
        # The cached suffix must equal the guarded first result, not the
        # raw tool output.
        self.assertEqual(second, "[cached] " + first,
                         "cache hit must return the guarded text, not raw")



if __name__ == "__main__":
    unittest.main()