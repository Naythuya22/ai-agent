"""Cloud-only boot: the server must work on a device with NO local model.

Before this, two eager loads made a model file a hard requirement for
*everything*: ``server.main()`` exited when no model resolved, and
``ensure_modes()`` loaded the engine just to build the modes — so a
phone with only cloud keys could not even open the Settings page where
keys are entered. Now the engine loads lazily at the first local turn,
which raises ``LLMError`` and surfaces as 503 with the actionable
missing-model message.

These tests stage a home whose config names no model. On a machine
where auto-discovery still finds one (the developer phone, where the
real Gemma sits in Download/gemmamodel) the premise is false and the
class skips — the with-model paths are covered by every other suite.

Run:  sh tests/run.sh -p "test_cloud_only.py"
"""
from __future__ import annotations

import json
import unittest

from _harness import bootstrap, current_home, get_app

bootstrap()


def _no_model_home() -> bool:
    """Stage a config with no model; True when discovery finds nothing."""
    bootstrap()
    home = current_home()
    cfg_file = home / ".ai-agent" / "config.json"
    cfg_file.write_text(json.dumps({"model_path": ""}), encoding="utf-8")
    import config
    return not config.Config.load().resolve_model_path()


class CloudOnlyCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.has_no_model = _no_model_home()

    def setUp(self):
        if not self.has_no_model:
            self.skipTest("a model is discoverable on this machine")
        _no_model_home()
        import routes._state as st
        self._st = st
        self._saved = (st._llm, st._memory, st._chat_mode, st._agent_mode)
        st._llm = st._memory = st._chat_mode = st._agent_mode = None

    def tearDown(self):
        st = self._st
        st._llm, st._memory, st._chat_mode, st._agent_mode = self._saved
        bootstrap()  # leave a sane dummy-model config for later suites


class TestModesWithoutModel(CloudOnlyCase):
    def test_modes_build_without_a_model(self):
        from routes._state import ensure_modes, get_agent_mode, get_chat_mode
        ensure_modes()  # must not raise
        self.assertIsNotNone(get_agent_mode())
        self.assertIsNotNone(get_chat_mode())

    def test_local_turn_reports_the_model_unavailable(self):
        from core.llm import LLMError
        from routes._state import get_chat_mode
        with self.assertRaises(LLMError) as ctx:
            get_chat_mode().session.send("hello")
        self.assertIn("Model not found", str(ctx.exception))

    def test_get_llm_default_still_loads_eagerly(self):
        """server.main()'s preload relies on the default raising here."""
        from core.llm import LLMError
        from routes._state import get_llm
        with self.assertRaises(LLMError):
            get_llm()


class TestHttpWithoutModel(CloudOnlyCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client = get_app().test_client()

    def test_ui_shell_still_serves(self):
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_agent_status_answers(self):
        r = self.client.get("/api/agent/status")
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True)[:200])

    def test_chat_completion_is_503_with_actionable_message(self):
        r = self.client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}]})
        self.assertEqual(r.status_code, 503, r.get_data(as_text=True)[:200])
        self.assertIn("Model not found", r.get_json()["error"]["message"])


if __name__ == "__main__":
    unittest.main()
