"""Session switch/delete must not race a live stream.

reload_session/reset close the cached Conversation and rebuild it from
disk. If a stream is mid-generation the native engine sees a freed handle
and can drop the process (audit H5). These tests pin the guard helper and
its use at every swap site.

Run:  sh tests/run.sh -p "test_session_guard.py"
"""
from __future__ import annotations

import inspect
import unittest

from _harness import bootstrap  # noqa: F401

bootstrap()

from routes import _state  # noqa: E402
from routes import sessions_api  # noqa: E402


class SessionSwapGuardTests(unittest.TestCase):
    def test_guard_acquires_both_locks(self):
        seen = {}
        with sessions_api._session_swap_guard():
            seen["agent"] = _state.agent_turn_lock._is_owned()
            seen["infer"] = _state.inference_lock._is_owned()
        self.assertTrue(seen["agent"], "agent_turn_lock must be held")
        self.assertTrue(seen["infer"], "inference_lock must be held")

    def test_guard_releases_on_exit(self):
        with sessions_api._session_swap_guard():
            pass
        self.assertFalse(_state.agent_turn_lock._is_owned(),
                         "agent_turn_lock leaked past the guard")
        self.assertFalse(_state.inference_lock._is_owned(),
                         "inference_lock leaked past the guard")


class GuardSitesTests(unittest.TestCase):
    """Every session-swap call site must take _session_swap_guard (H5)."""

    SITES = (
        "api_sessions_create",
        "api_sessions_delete",
        "api_sessions_bulk_delete",
        "api_sessions_activate",
        "api_new",
    )

    def test_all_swap_sites_use_the_guard(self):
        for name in self.SITES:
            fn = getattr(sessions_api, name)
            src = inspect.getsource(fn)
            self.assertIn("_session_swap_guard()", src,
                          "%s must take _session_swap_guard" % name)
