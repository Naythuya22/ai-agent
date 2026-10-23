"""Release 26 — the key-add dead end (FIX 2, tester evaluation #2).

The eval's exact state: a usable key IS present (added in Settings),
but the fresh config's cost_ack is still false — adding a key posts
to the keys route alone and never completes the one-time cost
acknowledgement (that is the Settings Save, after the UI shows the
price line). Before this release the turn fell through to the local
engine and died on the raw `LLMError: Model not found (.litertlm)`.

Now _key_guidance covers the cost_not_acknowledged blocker too: the
user gets Burmese guidance naming the exact taps (Settings → review
the cost line → Save) as a normal saved reply. The acknowledgement
gate itself is unchanged — no auto-ack on key add.

Run:  sh tests/run.sh -p "test_cost_ack_guidance.py"
"""
from __future__ import annotations

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


class FunctionTool:
    def __init__(self, fn):
        self.fn = fn


class NoModelLLM:
    """Fresh-phone local stub: load() raises the raw error the eval
    saw bubble out as HTTP 500."""

    def __init__(self):
        self.load_calls = 0
        self._engine = None

    def load(self):
        self.load_calls += 1
        raise LLMError("Model not found: (unset)")

    def ask(self, prompt):
        return ""


class CostAckBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="cost-ack-"))
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
        # The eval state, exactly: key present, cost_ack still false
        # (the fresh-config default).
        self.km.add_key("openai", DUMMY_KEY, "main")
        cfg = self.km.load()
        assert cfg["cost_ack"] is False
        self.km.save()
        self.llm = NoModelLLM()

    def new_mode(self):
        mode = AgentMode(self.llm, None)
        mode._tools = [FunctionTool(fn) for fn in ALL_TOOLS]
        mode._backend = AgentBackend(
            keys=self.km, net_state=lambda: "online",
            http=lambda url, body, headers, timeout=30: {
                "choices": [{"message": {"content": "CLOUD_OK"}}],
                "usage": {}})
        return mode

    def saved(self, sid):
        return self.store.get_session(sid)["messages"]


class CostAckGuidanceCase(CostAckBase):
    def test_blocker_is_cost_not_acknowledged(self):
        # Guard the reproduction itself: with a key present the
        # blocker must be the acknowledgement, not the key.
        mode = self.new_mode()
        self.assertEqual(mode._backend.why_not_cloud(),
                         "cost_not_acknowledged")

    def test_turn_gets_guidance_not_the_raw_error(self):
        mode = self.new_mode()
        out = mode.handle("မင်္ဂလာပါ")
        self.assertIn("Settings", out)
        self.assertIn("Save", out)
        self.assertIn("ကုန်ကျစရိတ်", out)
        self.assertNotIn("Model not found", out)
        self.assertNotIn("litertlm", out)
        # Local engine was tried first (auto mode's order), then the
        # guidance closed the turn as a normal saved pair.
        self.assertEqual(self.llm.load_calls, 1)
        saved = self.saved(mode.session_id)
        self.assertEqual([m["role"] for m in saved],
                         ["user", "assistant"])
        self.assertEqual(saved[-1]["text"], out)

    def test_helper_returns_the_cost_ack_text(self):
        mode = self.new_mode()
        out = mode._key_guidance("hi")
        self.assertEqual(out, AgentMode.COST_ACK_GUIDANCE)

    def test_acknowledgement_still_required_after_guidance(self):
        # Guidance is advice, not an auto-ack: the blocker persists
        # until the user actually Saves in Settings.
        mode = self.new_mode()
        mode.handle("မင်္ဂလာပါ")
        self.assertEqual(mode._backend.why_not_cloud(),
                         "cost_not_acknowledged")
        cfg = self.km.load()
        self.assertFalse(cfg["cost_ack"])


if __name__ == "__main__":
    unittest.main()
