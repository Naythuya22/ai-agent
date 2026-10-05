"""Output-token budget and tool-loop cap.

These numbers were hardcoded where they were used — 1024 output tokens in two
places, 8 tool iterations in two more — so nothing could raise them without
editing source. They now live in `config`, and these tests pin both halves:
the value a caller gets, and the ceiling that keeps a request bounded.

Run:  sh tests/run.sh -p "test_limits.py"
"""
from __future__ import annotations

import os
import unittest

from _harness import bootstrap, get_app

bootstrap()

import config  # noqa: E402
from core.agent_backend import AgentBackend  # noqa: E402
from core.key_manager import KeyManager  # noqa: E402
from routes import openai_compat  # noqa: E402


def _restore_env(name: str, old) -> None:
    if old is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = old


class ConfiguredLimitTests(unittest.TestCase):
    """config — defaults, the env override, and the clamping."""

    def test_defaults(self):
        # CHAT and the probe are deliberately not environment-driven, so these
        # are exact. The agent pair is env-driven, so assert the wiring rather
        # than the literal — a shell that exports AI_AGENT_MAX_TOKENS must not
        # fail the suite.
        self.assertEqual(config.CHAT_MAX_TOKENS_DEFAULT, 2048)
        self.assertEqual(config.PROBE_MAX_TOKENS, 8)
        self.assertEqual(config._env_int("AI_AGENT_NOWHERE_AT_ALL", 8192), 8192)
        self.assertEqual(config.agent_max_tokens("agent"),
                         config.AGENT_MAX_TOKENS_DEFAULT)
        self.assertEqual(config.agent_max_iter("agent"),
                         config.AGENT_MAX_ITER_DEFAULT)

    def test_agent_max_iter_default_is_configurable(self):
        """AI_AGENT_MAX_ITER moves the cap; junk and absurd values do not."""
        old = os.environ.get("AI_AGENT_MAX_ITER")
        try:
            os.environ["AI_AGENT_MAX_ITER"] = "12"
            self.assertEqual(config._env_int("AI_AGENT_MAX_ITER", 30), 12)
            os.environ["AI_AGENT_MAX_ITER"] = "not-a-number"
            self.assertEqual(config._env_int("AI_AGENT_MAX_ITER", 30), 30,
                             "a typo must not take the agent down")
            os.environ["AI_AGENT_MAX_ITER"] = "0"
            self.assertEqual(config._env_int("AI_AGENT_MAX_ITER", 30), 30)
            os.environ["AI_AGENT_MAX_ITER"] = "99999"
            # _env_int clamps to the `maximum` it is given and to nothing
            # else — production passes maximum=100 (see config.py), so the
            # test has to pass it too. Without it there is no ceiling here and
            # 99999 is the honest answer.
            self.assertEqual(
                config._env_int("AI_AGENT_MAX_ITER", 30, maximum=100), 30,
                "slow tools must not become an unbounded loop")
            self.assertEqual(config._env_int("AI_AGENT_MAX_ITER", 30), 99999,
                             "no ceiling asked for, none applied")
        finally:
            _restore_env("AI_AGENT_MAX_ITER", old)

        # The loops read the module attribute, so a retune is visible without
        # reimporting config.
        saved = config.AGENT_MAX_ITER_DEFAULT
        try:
            config.AGENT_MAX_ITER_DEFAULT = 7
            self.assertEqual(config.agent_max_iter("agent"), 7)
        finally:
            config.AGENT_MAX_ITER_DEFAULT = saved

    def test_agent_max_tokens_uses_config_default(self):
        saved = config.AGENT_MAX_TOKENS_DEFAULT
        try:
            config.AGENT_MAX_TOKENS_DEFAULT = 4096
            self.assertEqual(config.agent_max_tokens("agent"), 4096)
            config.AGENT_MAX_TOKENS_DEFAULT = 10 ** 9
            self.assertEqual(config.agent_max_tokens("agent"),
                             config.MAX_TOKENS_CEILING,
                             "even a silly value cannot pass the ceiling")
        finally:
            config.AGENT_MAX_TOKENS_DEFAULT = saved

    def test_code_mode_uses_higher_iter_cap(self):
        self.assertEqual(config.agent_max_iter("code"),
                         config.AGENT_MAX_ITER_CODE)
        self.assertGreater(config.agent_max_iter("code"),
                           config.agent_max_iter("agent"))
        self.assertGreater(config.agent_max_tokens("code"),
                           config.agent_max_tokens("agent"))
        # Anything that is not "code" gets the agent budget.
        self.assertEqual(config.agent_max_iter("chat"),
                         config.agent_max_iter("agent"))
        self.assertEqual(config.agent_max_tokens(""), config.AGENT_MAX_TOKENS_DEFAULT)

    def test_clamp_max_tokens(self):
        default = config.CHAT_MAX_TOKENS_DEFAULT
        self.assertEqual(config.clamp_max_tokens(None, default), default)
        self.assertEqual(config.clamp_max_tokens(64, default), 64)
        self.assertEqual(config.clamp_max_tokens(0, default), default)
        self.assertEqual(config.clamp_max_tokens(-5, default), default)
        self.assertEqual(config.clamp_max_tokens("abc", default), default)
        self.assertEqual(config.clamp_max_tokens(10 ** 9, default),
                         config.MAX_TOKENS_CEILING)


class BackendLimitTests(unittest.TestCase):
    """AgentBackend — the step default and the cost probe."""

    def setUp(self):
        self.km = KeyManager()
        if self.km.path.exists():
            self.km.path.unlink()

    def test_probe_still_uses_8_tokens(self):
        """The key probe must stay the cheapest call in the app."""
        seen = {}

        def http(url, body, headers, timeout=30):
            seen.update(body or {})
            return {"choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

        be = AgentBackend(keys=self.km, net_state=lambda: "online", http=http)
        out = be.test_key("deepseek", "sk-probe-aaaaaaaaaaaa4444")
        self.assertTrue(out.get("ok"), out)
        self.assertEqual(seen.get("max_tokens"), 8)
        self.assertEqual(config.PROBE_MAX_TOKENS, 8)

    def test_step_default_comes_from_config(self):
        seen = {}

        def http(url, body, headers, timeout=30):
            seen.update(body or {})
            return {"choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

        cfg = self.km.load()
        cfg["prefer_cloud"] = True
        cfg["cost_ack"] = True
        # Only OpenAI, so the request that lands in `seen` is unambiguous.
        for name in ("gemini", "openrouter", "anthropic", "deepseek"):
            self._disable(name)
        self.km.save()
        self.km.add_key("openai", "sk-alpha-aaaaaaaaaaaa1111", "alpha")

        saved = config.AGENT_MAX_TOKENS_DEFAULT
        try:
            config.AGENT_MAX_TOKENS_DEFAULT = 1234
            be = AgentBackend(keys=self.km, net_state=lambda: "online", http=http)
            be.step(system="", messages=[{"role": "user", "text": "hi"}],
                    tools=[], max_tokens=None)
            self.assertEqual(seen.get("max_tokens"), 1234,
                             "max_tokens=None means the configured default")

            # An explicit value still wins — up to the ceiling.
            be.step(system="", messages=[{"role": "user", "text": "hi"}],
                    tools=[], max_tokens=64)
            self.assertEqual(seen.get("max_tokens"), 64)
            be.step(system="", messages=[{"role": "user", "text": "hi"}],
                    tools=[], max_tokens=10 ** 9)
            self.assertEqual(seen.get("max_tokens"), config.MAX_TOKENS_CEILING)
        finally:
            config.AGENT_MAX_TOKENS_DEFAULT = saved

    def _disable(self, provider):
        cfg = self.km.load()
        cfg["providers"] = dict(cfg["providers"])
        cfg["providers"][provider] = dict(cfg["providers"][provider])
        cfg["providers"][provider]["enabled"] = False


class _FakeConv:
    def __init__(self):
        self.closed = False

    def send_message(self, prompt):
        return {"content": [{"text": "ok"}]}

    def __exit__(self, *exc):
        self.closed = True
        return False


class _FakeEngine:
    def __init__(self, sink):
        self._sink = sink

    def create_conversation(self, **kw):
        self._sink.append(kw)
        return _FakeConv()


class _FakeLLM:
    def __init__(self, sink):
        self._engine = _FakeEngine(sink)


class OpenAICompatLimitTests(unittest.TestCase):
    """`/v1/chat/completions` — default, client override, hard ceiling."""

    BODY = {"model": "gemma-4-e4b",
            "messages": [{"role": "user", "content": "hi"}]}

    def setUp(self):
        self.seen = []
        self._orig = openai_compat.get_llm
        openai_compat.get_llm = lambda: _FakeLLM(self.seen)
        self.client = get_app().test_client()

    def tearDown(self):
        openai_compat.get_llm = self._orig

    def _post(self, **overrides):
        body = dict(self.BODY)
        body.update(overrides)
        resp = self.client.post("/v1/chat/completions", json=body)
        self.assertEqual(resp.status_code, 200, resp.get_data(as_text=True))
        return self.seen[-1]

    def test_chat_default_is_2048(self):
        call = self._post()
        self.assertEqual(call["max_output_tokens"],
                         config.CHAT_MAX_TOKENS_DEFAULT)
        self.assertEqual(call["max_output_tokens"], 2048)

    def test_openai_compat_honours_client_max_tokens(self):
        self.assertEqual(self._post(max_tokens=64)["max_output_tokens"], 64)
        self.assertEqual(self._post(max_tokens=512)["max_output_tokens"], 512)
        # Absurd or malformed values fall back or clamp — never trusted raw.
        self.assertEqual(self._post(max_tokens=10 ** 9)["max_output_tokens"],
                         config.MAX_TOKENS_CEILING)
        self.assertEqual(self._post(max_tokens="nope")["max_output_tokens"],
                         config.CHAT_MAX_TOKENS_DEFAULT)
        self.assertEqual(self._post(max_tokens=0)["max_output_tokens"],
                         config.CHAT_MAX_TOKENS_DEFAULT)


if __name__ == "__main__":
    unittest.main()