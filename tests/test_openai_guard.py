"""The /v1 SSE endpoint must serialise model work with Chat mode.

Without inference_guard the opencode bridge and /api/chat can drive the
same LiteRT engine at once — the native side sees interleaved turns and
drops the process. This file pins the guard so a future refactor cannot
remove it.

Run:  sh tests/run.sh -p "test_openai_guard.py"
"""
from __future__ import annotations

import unittest

from _harness import bootstrap  # noqa: F401

bootstrap()

from flask import Flask  # noqa: E402
from routes import _state  # noqa: E402
from routes import openai_compat  # noqa: E402


class _FakeConv:
    def __init__(self, seen):
        self._seen = seen

    def send_message(self, prompt):
        self._seen["lock_held"] = _state.inference_lock._is_owned()
        return {"content": [{"text": "hello"}]}

    def __exit__(self, *a):
        return None


class _FakeEngine:
    def __init__(self, seen):
        self._seen = seen

    def create_conversation(self, **kw):
        return _FakeConv(self._seen)


class _FakeLLM:
    def __init__(self, seen):
        self._engine = _FakeEngine(seen)


def _app():
    a = Flask(__name__)
    a.register_blueprint(openai_compat.bp)
    return a


class V1InferenceGuardTests(unittest.TestCase):
    def setUp(self):
        self.seen = {}
        self._orig = openai_compat.get_llm
        openai_compat.get_llm = lambda: _FakeLLM(self.seen)

    def tearDown(self):
        openai_compat.get_llm = self._orig

    def _post(self):
        app = _app()
        with app.test_request_context(
                "/v1/chat/completions", method="POST",
                json={"messages": [{"role": "user", "content": "hi"}]}):
            return openai_compat.chat_completions()

    def test_send_message_runs_under_inference_lock(self):
        self._post()
        self.assertTrue(self.seen.get("lock_held"),
                        "inference_lock must be held during send_message")

    def test_lock_released_after_request(self):
        self._post()
        self.assertFalse(_state.inference_lock._is_owned(),
                         "inference_lock must not leak past the request")
