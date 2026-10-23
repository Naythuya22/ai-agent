"""Release 25 — the fresh-install dead end.

A brand-new install has no provider keys and (usually) no local
model file. Before this release the first message died on a raw
developer error — `LLMError: Model not found: (unset) / searched: … /
fix: copy a .litertlm file…` — which never mentioned that adding a
key in Settings is the intended path. Now, when the local engine
cannot start AND the cloud blocker is "no usable key", the turn
closes with friendly Burmese guidance (Settings → add a key; Gemini
is free) as a normal saved reply — on both the non-streaming path
(AgentMode.handle) and, via the same helper, the v1 stream route.

The guidance must NOT fire when keys are not the blocker: offline
and keyed-but-working installs keep their existing behaviour exactly.

Run:  sh tests/run.sh -p "test_no_key_guidance.py"
"""
from __future__ import annotations

import contextlib
import tempfile
import unittest
from pathlib import Path

from _harness import bootstrap

bootstrap()

import config  # noqa: E402
from core.agent_backend import AgentBackend  # noqa: E402
from core.key_manager import KeyManager  # noqa: E402
from core.llm import LLMError  # noqa: E402
from modes import agent as agent_mod  # noqa: E402
from modes.agent import AgentMode  # noqa: E402
from sessions import SessionStore  # noqa: E402
from tools import ALL_TOOLS  # noqa: E402

DUMMY_KEY = "<redacted>"
CLOUD_TEXT = "CLOUD_REPLY_OK"


@contextlib.contextmanager
def verification_off():
    saved = config.AGENT_VERIFY_ENABLED
    config.AGENT_VERIFY_ENABLED = False
    try:
        yield
    finally:
        config.AGENT_VERIFY_ENABLED = saved


class FunctionTool:
    def __init__(self, fn):
        self.fn = fn


class NoModelLLM:
    """Local engine stub for a fresh phone: no .litertlm anywhere,
    so load() raises the exact error a new user used to see raw."""

    def __init__(self):
        self.load_calls = 0
        self._engine = None

    def load(self):
        self.load_calls += 1
        raise LLMError("Model not found: (unset)")

    def ask(self, prompt):
        return ""


class AnswerTransport:
    """Cloud stub: answers plainly. Records every call."""

    def __init__(self):
        self.calls = []

    def __call__(self, url, body, headers, timeout=30):
        self.calls.append(url)
        return {"choices": [{"message": {"content": CLOUD_TEXT}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2}}


class NoKeyGuidanceBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="no-key-"))
        self._saved_root = config.PROJECT_ROOT
        config.PROJECT_ROOT = self.tmp
        self.addCleanup(setattr, config, "PROJECT_ROOT", self._saved_root)
        self.store = SessionStore(self.tmp / "sessions.json")
        saved_factory = agent_mod.SessionStore
        agent_mod.SessionStore = lambda: self.store
        self.addCleanup(setattr, agent_mod, "SessionStore", saved_factory)
        self.km = KeyManager()
        if self.km.path.exists():
            self.km.path.unlink()
        self.km.save()
        self.llm = NoModelLLM()

    def new_mode(self, net="online", transport=None):
        mode = AgentMode(self.llm, None)
        mode._tools = [FunctionTool(fn) for fn in ALL_TOOLS]
        be = AgentBackend(keys=self.km, net_state=lambda: net,
                          http=transport or AnswerTransport())
        mode._backend = be
        return mode

    def saved(self, sid):
        return self.store.get_session(sid)["messages"]


class FreshInstallCase(NoKeyGuidanceBase):
    def test_first_message_gets_guidance_not_raw_error(self):
        mode = self.new_mode()
        out = mode.handle("hi")
        # The guidance names the actual fix — a key, in Settings,
        # with the free option — and none of the raw error survives.
        self.assertIn("API key", out)
        self.assertIn("Settings", out)
        self.assertIn("Gemini", out)
        self.assertNotIn("Model not found", out)
        self.assertNotIn("searched:", out)
        # The local engine really was tried first (auto mode's order).
        self.assertEqual(self.llm.load_calls, 1)
        # The turn is a normal saved pair, not an exception-shaped hole.
        saved = self.saved(mode.session_id)
        self.assertEqual([m["role"] for m in saved], ["user", "assistant"])
        self.assertEqual(saved[-1]["text"], out)


class NotTheBlockerCase(NoKeyGuidanceBase):
    def test_offline_without_keys_keeps_original_error(self):
        # No keys, but the blocker is the network: the guidance would
        # be wrong advice, so the original LLMError must propagate.
        mode = self.new_mode(net="offline")
        with self.assertRaises(LLMError):
            mode.handle("hi")

    def test_guidance_helper_is_none_when_a_key_is_usable(self):
        self.km.add_key("openai", DUMMY_KEY, "main")
        cfg = self.km.load()
        cfg["cost_ack"] = True
        self.km.save()
        mode = self.new_mode()
        self.assertIsNone(mode._key_guidance("hi"))
        self.assertEqual(mode._messages, [])

    def test_keyed_install_still_answers_from_cloud(self):
        # OpenAI-shaped transport, so the keyed provider is openai —
        # the same pairing the Release 24 rig uses.
        self.km.add_key("openai", DUMMY_KEY, "main")
        cfg = self.km.load()
        cfg["cost_ack"] = True
        self.km.save()
        transport = AnswerTransport()
        mode = self.new_mode(transport=transport)
        with verification_off():
            out = mode.handle("hi")
        self.assertEqual(out, CLOUD_TEXT)
        self.assertEqual(self.llm.load_calls, 0)


if __name__ == "__main__":
    unittest.main()
