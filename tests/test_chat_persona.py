"""Fix 4.2i-A: chat system prompt must survive session lifecycle.

Regression guard. Before the fix, reload_session, compress_session and
reset each built a system provider that returned only the memory hint
(or None), silently dropping CHAT_SYSTEM (Burmese persona + language
rules). After the fix, every lifecycle path goes through
_system_provider(), which returns the same lazy closure __init__ built.
"""
from __future__ import annotations

import unittest

from _harness import bootstrap, get_app  # noqa: F401

bootstrap()

from modes import chat as chat_mod  # noqa: E402


class _Mem:
    def __init__(self, hint=""):
        self._hint = hint

    def get_system_hint(self):
        return self._hint


_SENTINEL = object()


class SystemProviderTests(unittest.TestCase):
    """The helper itself — CHAT_SYSTEM + hint, lazy, None-safe."""

    def _mode(self, hint="", memory=_SENTINEL):
        m = chat_mod.ChatMode.__new__(chat_mod.ChatMode)
        m.llm = None
        m.memory = _Mem(hint) if memory is _SENTINEL else memory
        return m

    def test_helper_includes_chat_system_when_no_hint(self):
        m = self._mode()
        p = m._system_provider()
        self.assertIsNotNone(p)
        self.assertEqual(p(), chat_mod.CHAT_SYSTEM)

    def test_helper_includes_chat_system_with_hint(self):
        m = self._mode(hint="USER PREFERS BURMESE")
        p = m._system_provider()
        self.assertIsNotNone(p)
        text = p()
        self.assertIn(chat_mod.CHAT_SYSTEM, text)
        self.assertIn("USER PREFERS BURMESE", text)
        self.assertTrue(text.startswith(chat_mod.CHAT_SYSTEM))

    def test_helper_returns_none_when_memory_missing(self):
        m = self._mode(memory=None)
        self.assertIsNone(m._system_provider())

    def test_helper_is_lazy_re_reads_hint(self):
        class _Mut:
            def __init__(self): self.hint = ""
            def get_system_hint(self): return self.hint
        mem = _Mut()
        m = self._mode(memory=mem)
        p = m._system_provider()
        self.assertEqual(p(), chat_mod.CHAT_SYSTEM)
        mem.hint = "NEW HINT"
        self.assertIn("NEW HINT", p())


class LifecycleUsesSharedProviderTests(unittest.TestCase):
    """Each lifecycle method must pass the shared provider to create_chat."""

    def _stub_mode(self, hint=""):
        m = chat_mod.ChatMode.__new__(chat_mod.ChatMode)
        captured = {}

        class _Session:
            def clear(self): pass
            def close(self): pass

        class _LLM:
            def create_chat(self, system_provider=None, initial_messages=None):
                captured["provider"] = system_provider
                captured["messages"] = initial_messages
                return _Session()

        class _Store:
            def set_active(self, mode, sid): return True
            def get_session(self, sid): return {"messages": []}
            def clear_mode(self, mode): pass
            def get_active_id(self, mode): return None
            def save_mode(self, mode, msgs): pass

        m.llm = _LLM()
        m.memory = _Mem(hint)
        m.turns = 0
        m.session_id = None
        m._store = _Store()
        m.session = _Session()
        return m, captured

    def test_reload_session_uses_shared_provider(self):
        m, cap = self._stub_mode(hint="H1")
        self.assertTrue(m.reload_session("sid-1"))
        p = cap.get("provider")
        self.assertIsNotNone(p, "reload_session must pass a provider")
        text = p()
        self.assertIn(chat_mod.CHAT_SYSTEM, text)
        self.assertIn("H1", text)

    def test_reset_uses_shared_provider(self):
        m, cap = self._stub_mode(hint="H2")
        m.reset()
        p = cap.get("provider")
        self.assertIsNotNone(p, "reset must pass a provider")
        text = p()
        self.assertIn(chat_mod.CHAT_SYSTEM, text)
        self.assertIn("H2", text)


if __name__ == "__main__":
    unittest.main()
