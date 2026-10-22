"""core.agent_backend — the provider/key selection rules for Agent Mode.

Every test drives the backend through an injected fake transport, so the
suite never opens a socket and never sees a real key.

Run:  sh tests/run.sh -p "test_agent_backend.py"
"""
from __future__ import annotations

import json
import unittest

from _harness import bootstrap

bootstrap()

from core import agent_backend as AB  # noqa: E402
from core import model_registry as MR  # noqa: E402
from core.agent_backend import (  # noqa: E402
    AgentBackend,
    BackendExhausted,
    CloudError,
    Step,
    _apply_reasoning_policy,
    _is_thinking_model,
    cost_per_1k,
    failure_reason,
    friendly_local_name,
    normalize,
    tool_schema,
)
from core.key_manager import KeyManager  # noqa: E402

ALPHA = "sk-alpha-aaaaaaaaaaaa1111"
BETA = "sk-beta-bbbbbbbbbbbbbbbb2222"
GEM = "AIza-gemini-cccccccccccccc3333"


def http_ok(payload=None):
    """A transport that always answers 200 with a text-only completion."""
    def _f(url, body, headers, timeout=30):
        return payload or {"choices": [{"message": {"content": "ok"}}],
                           "usage": {"prompt_tokens": 7, "completion_tokens": 3}}
    return _f


def bad_request(message):
    """A 4xx in the shape `_status_error` hands the retry logic."""
    return AB.CloudError("bad_response", message, 400)


class BackendCase(unittest.TestCase):
    """Shared fixture: a config with two OpenAI keys and a Gemini key."""

    def setUp(self):
        self.km = KeyManager()
        if self.km.path.exists():
            self.km.path.unlink()
        self.km.add_key("openai", ALPHA, "alpha")
        self.km.add_key("openai", BETA, "beta")
        self.km.add_key("gemini", GEM, "gem")
        cfg = self.km.load()
        cfg["prefer_cloud"] = True
        cfg["cost_ack"] = True
        # Backend.refresh() re-reads the file every turn (that is how a
        # Settings change reaches a running server), so fixtures must persist
        # what they configure instead of leaving it in memory only.
        self.km.save()

    def set_cfg(self, **kw):
        cfg = self.km.load()
        cfg.update(kw)
        self.km.save()

    def backend(self, http=None, online=True):
        return AgentBackend(keys=self.km,
                            net_state=(lambda: "online" if online else "offline"),
                            http=http or http_ok())

    # convenience -------------------------------------------------
    def disable(self, provider):
        self.set_cfg(providers=dict(self.km.load()["providers"]))
        self.km.load()["providers"][provider]["enabled"] = False
        self.km.save()

    def exhaust(self, provider):
        """Mark every key of a provider dead (as repeated 401s would)."""
        for e in self.km.entries(provider):
            self.km.mark_auth_failed(provider, e["id"], "401")


class DecisionTests(BackendCase):
    """should_use_cloud — the four gates, in the order they are checked."""

    def test_cost_table(self):
        self.assertEqual(cost_per_1k("openai", "gpt-4o-mini"), 0.15)
        self.assertEqual(cost_per_1k("gemini", "gemini-2.0-flash"), 0.10)
        # unknown model on a known provider still gets a sane number
        self.assertEqual(cost_per_1k("openai", "gpt-9-something-new"), 0.50)
        self.assertIsNone(cost_per_1k("nonsense", "whatever"))

    def test_cloud_is_used_when_everything_is_ready(self):
        self.assertTrue(self.backend().should_use_cloud())

    def test_offline_goes_local(self):
        self.assertFalse(self.backend(online=False).should_use_cloud())

    def test_prefer_cloud_off_goes_local_even_online(self):
        self.set_cfg(prefer_cloud=False)
        self.assertFalse(self.backend().should_use_cloud())

    def test_price_must_be_acknowledged_first(self):
        """No cloud turn before the UI has shown the cost line."""
        self.set_cfg(cost_ack=False)
        be = self.backend()
        self.assertFalse(be.should_use_cloud())
        self.set_cfg(cost_ack=True)
        self.assertTrue(be.should_use_cloud())

    def test_no_usable_key_goes_local(self):
        self.exhaust("openai")
        self.exhaust("gemini")
        self.exhaust("anthropic")
        self.assertEqual(self.backend().usable(), [])
        self.assertFalse(self.backend().should_use_cloud())

    def test_disabled_provider_is_skipped(self):
        self.disable("openai")
        self.disable("gemini")
        self.exhaust("anthropic")
        self.assertEqual(self.backend().usable(), [])

    def test_cooled_down_key_is_not_usable(self):
        for e in self.km.entries("openai"):
            self.km.mark_rate_limited("openai", e["id"], 60.0)
        self.assertNotIn("openai", self.backend().usable())
        self.assertIn("gemini", self.backend().usable())

    def test_local_only_mode_never_uses_cloud(self):
        self.km.set_mode("local")
        self.km.save()
        be = self.backend()
        self.assertEqual(be.mode, "local")
        self.assertTrue(be.fallback_to_local, "Local Only is served by the local model")
        self.assertFalse(be.should_use_cloud())
        self.assertEqual(be.why_not_cloud(), "local_only")

    def test_cloud_only_mode_falls_back_to_nothing(self):
        self.km.set_mode("cloud")
        self.km.save()
        be = self.backend()
        self.assertEqual(be.mode, "cloud")
        self.assertFalse(be.fallback_to_local)
        self.assertTrue(be.should_use_cloud())
        self.assertEqual(be.why_not_cloud(), "")

    def test_why_not_cloud_names_the_real_gate(self):
        self.km.set_mode("auto")
        self.km.save()
        self.assertEqual(self.backend(online=False).why_not_cloud(), "offline")
        self.exhaust("openai"); self.exhaust("gemini"); self.exhaust("anthropic")
        self.assertEqual(self.backend().why_not_cloud(), "no_usable_key")
        self.set_cfg(cost_ack=False)
        # add a key back so the only gate left is the price acknowledgement
        self.km.add_key("openai", "sk-gate-000000000000", "gate")
        self.assertEqual(self.backend().why_not_cloud(), "cost_not_acknowledged")

    def test_status_reports_all_of_it(self):
        st = self.backend().status()
        self.assertTrue(st["online"])
        self.assertEqual(st["state"], "online")
        self.assertTrue(st["will_use_cloud"])
        self.assertIn("openai", st["fallback"])
        self.assertIn("openai", st["usable"])
        self.assertEqual(st["providers"]["openai"]["cost_per_1k"], 0.15)
        self.assertEqual(st["providers"]["openai"]["keys"]["total"], 2)
        # raw keys never appear anywhere in the status blob
        self.assertNotIn(ALPHA, json.dumps(st))


class StepTests(BackendCase):
    def test_success_returns_step_and_marks_key_used(self):
        be = self.backend()
        step = be.step(system="s", messages=[{"role": "user", "text": "hi"}],
                       tools=[], max_tokens=64)
        self.assertIsInstance(step, Step)
        self.assertEqual(step.text, "ok")
        self.assertEqual(step.provider, "openai")
        self.assertEqual(step.usage["prompt_tokens"], 7)
        used = [e["used"] for e in self.km.entries("openai")]
        self.assertEqual(max(used), 1, "the key that answered is counted once")
        self.assertEqual(sum(used), 1, "the other key was never spent")

    def test_429_parks_the_key_and_rotation_moves_on(self):
        seen = []

        def http(url, body, headers, timeout=30):
            seen.append(headers.get("Authorization", ""))
            if len(seen) == 1:
                raise CloudError("rate_limit", "slow down", 429, 12.0)
            return {"choices": [{"message": {"content": "second key worked"}}]}

        be = self.backend(http=http)
        step = be.step(system="s", messages=[], tools=[], max_tokens=8)
        self.assertEqual(step.text, "second key worked")
        self.assertEqual(len(seen), 2, "429 must trigger exactly one retry")
        self.assertNotEqual(seen[0], seen[1], "the retry must use the other key")
        parked = [e for e in self.km.entries("openai")
                  if e["status"] == "rate_limited"]
        self.assertEqual(len(parked), 1)
        self.assertGreater(parked[0]["cooldown_until"], 0)

    def test_every_key_rate_limited_raises_exhausted(self):
        def http(url, body, headers, timeout=30):
            raise CloudError("rate_limit", "", 429, 30.0)

        be = self.backend(http=http)
        with self.assertRaises(BackendExhausted):
            be.step(system="s", messages=[], tools=[], max_tokens=8)

    def test_auth_failure_marks_key_dead_then_next_provider_is_used(self):
        state = {"n": 0}

        def http(url, body, headers, timeout=30):
            state["n"] += 1
            if state["n"] <= 2:          # both OpenAI keys rejected
                raise CloudError("auth", "invalid api key", 401)
            return {"candidates": [{"content": {"parts": [{"text": "gemini"}]}}],
                    "usageMetadata": {"promptTokenCount": 4}}

        be = self.backend(http=http)
        step = be.step(system="s", messages=[], tools=[], max_tokens=8)
        self.assertEqual(step.provider, "gemini")
        self.assertEqual(step.text, "gemini")
        self.assertEqual(self.km.summary()["openai"]["dead"], 2)
        self.assertNotIn("openai", be.usable())

    def test_dead_keys_are_never_tried_again(self):
        self.exhaust("openai")
        urls = []

        def http(url, body, headers, timeout=30):
            urls.append(url)
            return {"candidates": [{"content": {"parts": [{"text": "gemini"}]}}]}

        be = self.backend(http=http)
        step = be.step(system="s", messages=[], tools=[], max_tokens=8)
        self.assertEqual(step.provider, "gemini")
        self.assertNotIn("openai", be.usable())
        for u in urls:
            self.assertNotIn("api.openai.com", u,
                             "a dead key's provider was dialed again")

    def test_network_error_stops_instead_of_walking_providers(self):
        calls = []

        def http(url, body, headers, timeout=30):
            calls.append(url)
            raise CloudError("network", "ENETUNREACH")

        be = self.backend(http=http)
        with self.assertRaises(BackendExhausted):
            be.step(system="s", messages=[], tools=[], max_tokens=8)
        # one attempt only: with no network the rest would fail the same way
        self.assertEqual(len(calls), 1)

    def test_all_providers_exhausted_raises(self):
        def http(url, body, headers, timeout=30):
            raise CloudError("server", "boom", 503)

        self.exhaust("anthropic")
        be = self.backend(http=http)
        with self.assertRaises(BackendExhausted) as ctx:
            be.step(system="s", messages=[], tools=[], max_tokens=8)
        self.assertIn("openai", str(ctx.exception))

    def test_error_messages_are_redacted_before_logging_state(self):
        def http(url, body, headers, timeout=30):
            raise CloudError("auth", "Invalid key: %s" % ALPHA, 401)

        be = self.backend(http=http)
        with self.assertRaises(BackendExhausted):
            be.step(system="s", messages=[], tools=[], max_tokens=8)
        stored = [e.get("last_error") or "" for e in self.km.entries("openai")]
        joined = " ".join(stored)
        self.assertNotIn(ALPHA, joined)
        self.assertNotIn(BETA, joined)


class Release9LockedTests(BackendCase):
    """Release 9 — locked mode: auto_switch off + a pinned provider.

    The pin alone may answer; its failure stops the turn with the
    classified reason instead of falling through the chain the user
    turned off.
    """

    def lock(self, pin):
        self.set_cfg(auto_switch=False, pinned_provider=pin)

    def test_auto_mode_is_the_default(self):
        be = self.backend()
        self.assertIsNone(be.locked_pin())
        self.assertEqual(be.route_providers(), be.usable())

    def test_locked_pin_routes_only_to_the_pin(self):
        self.lock("gemini")

        def http(url, body, headers, timeout=30):
            return {"candidates": [{"content": {"parts": [{"text": "gem"}]}}],
                    "usageMetadata": {"promptTokenCount": 3}}

        be = self.backend(http=http)
        self.assertEqual(be.locked_pin(), "gemini")
        self.assertEqual(be.route_providers(), ["gemini"])
        step = be.step(system="s", messages=[], tools=[], max_tokens=8)
        self.assertEqual(step.provider, "gemini")
        self.assertEqual(step.text, "gem")

    def test_locked_failure_does_not_fall_through(self):
        self.lock("openai")
        urls = []

        def http(url, body, headers, timeout=30):
            urls.append(url)
            raise CloudError("auth", "invalid api key", 401)

        be = self.backend(http=http)
        with self.assertRaises(BackendExhausted) as cm:
            be.step(system="s", messages=[], tools=[], max_tokens=8)
        exc = cm.exception
        self.assertEqual(exc.locked_provider, "openai")
        self.assertEqual(exc.locked_reason, "key_invalid")
        self.assertTrue(urls, "the pinned provider was attempted")
        self.assertTrue(all("openai.com" in u for u in urls),
                        "no other provider may be attempted while locked")

    def test_locked_rate_limit_reason(self):
        self.lock("openai")

        def http(url, body, headers, timeout=30):
            raise CloudError("rate_limit", "too many requests", 429, 30.0)

        be = self.backend(http=http)
        with self.assertRaises(BackendExhausted) as cm:
            be.step(system="s", messages=[], tools=[], max_tokens=8)
        self.assertEqual(cm.exception.locked_reason, "rate_limited")

    def test_locked_no_balance_reason(self):
        self.lock("openai")

        def http(url, body, headers, timeout=30):
            raise CloudError("server", "insufficient credits", 402)

        be = self.backend(http=http)
        with self.assertRaises(BackendExhausted) as cm:
            be.step(system="s", messages=[], tools=[], max_tokens=8)
        self.assertEqual(cm.exception.locked_reason, "no_balance")

    def test_locked_cooling_pin_reports_cooling(self):
        for e in self.km.entries("openai"):
            self.km.mark_rate_limited("openai", e["id"], 300.0)
        self.lock("openai")
        be = self.backend()
        with self.assertRaises(BackendExhausted) as cm:
            be.step(system="s", messages=[], tools=[], max_tokens=8)
        self.assertEqual(cm.exception.locked_reason, "cooling")

    def test_locked_pin_without_keys_reports_no_keys(self):
        self.lock("deepseek")
        be = self.backend()
        with self.assertRaises(BackendExhausted) as cm:
            be.step(system="s", messages=[], tools=[], max_tokens=8)
        self.assertEqual(cm.exception.locked_reason, "no_keys")

    def test_turning_auto_back_on_restores_the_chain(self):
        self.lock("gemini")
        self.set_cfg(auto_switch=True)
        be = self.backend()
        self.assertIsNone(be.locked_pin())
        step = be.step(system="s", messages=[], tools=[], max_tokens=8)
        self.assertEqual(step.provider, "openai")


class FailureReasonTests(unittest.TestCase):
    """failure_reason — the locked mode's user-facing vocabulary."""

    def test_classification(self):
        self.assertEqual(failure_reason(CloudError("auth", "x", 401)),
                         "key_invalid")
        self.assertEqual(failure_reason(CloudError("server", "nope", 403)),
                         "key_invalid")
        self.assertEqual(
            failure_reason(CloudError("server", "insufficient credits", 402)),
            "no_balance")
        self.assertEqual(failure_reason(CloudError("rate_limit", "x", 429)),
                         "rate_limited")
        self.assertEqual(
            failure_reason(CloudError("server", "model not found", 404)),
            "model_unavailable")
        self.assertEqual(failure_reason(CloudError("network", "x", 0)),
                         "network")
        self.assertEqual(failure_reason(CloudError("server", "boom", 500)),
                         "server_error")
        self.assertEqual(failure_reason(None), "server_error")


class AdapterTests(BackendCase):
    """Wire format: each provider must be asked in its own dialect."""

    def test_openai_request_shape(self):
        url, body, headers = AB.OpenAIAdapter().request(
            key=ALPHA, model="gpt-4o-mini", system="sys",
            messages=[{"role": "user", "content": "hi"}], tools=[], max_tokens=16)
        self.assertEqual(url, "https://api.openai.com/v1/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer " + ALPHA)
        self.assertEqual(body["model"], "gpt-4o-mini")
        # The system prompt rides along as its own message —
        # dropping it made cloud replies lose the ညီ persona.
        self.assertEqual(body["messages"][0],
                         {"role": "system", "content": "sys"})
        self.assertEqual(body["messages"][1]["content"], "hi")
        self.assertNotIn("tools", body)

    def test_openai_parse_converts_json_string_arguments(self):
        data = {"choices": [{"message": {"content": "", "tool_calls": [
            {"function": {"name": "run_command",
                          "arguments": '{"cmd": "ls"}'}}]}}],
            "usage": {"prompt_tokens": 1}}
        step = AB.OpenAIAdapter().parse(data, model="gpt-4o-mini", key_id="k1")
        self.assertEqual(step.tool_calls[0]["function"]["name"], "run_command")
        self.assertEqual(step.tool_calls[0]["function"]["arguments"], {"cmd": "ls"})

    def test_openai_parse_handles_malformed_tool_arguments(self):
        data = {"choices": [{"message": {"tool_calls": [
            {"function": {"name": "t", "arguments": "not json"}}]}}]}
        step = AB.OpenAIAdapter().parse(data, model="m", key_id="k")
        self.assertEqual(step.tool_calls[0]["function"]["arguments"],
                         {"_raw": "not json"})

    def test_openai_parse_reads_reasoning_content(self):
        data = {"choices": [{"message": {
            "content": "", "reasoning_content": "hmm",
            "tool_calls": [{"function": {"name": "t",
                                          "arguments": "{}"}}]}}],
            "usage": {"prompt_tokens": 1}}
        step = AB.OpenAIAdapter().parse(data, model="m", key_id="k")
        self.assertEqual(step.reasoning, "hmm")

    def test_openai_parse_rejects_empty_response(self):
        with self.assertRaises(CloudError):
            AB.OpenAIAdapter().parse({"choices": []}, model="m", key_id="k")

    def test_gemini_request_shape(self):
        url, body, headers = AB.GeminiAdapter().request(
            key=GEM, model="gemini-2.0-flash", system="sys",
            messages=[{"role": "user", "content": "hi"},
                      {"role": "assistant", "content": "",
                       "tool_calls": [{"function": {"name": "web_search",
                                                    "arguments": {"q": "x"}}}]},
                      {"role": "tool", "name": "web_search",
                       "tool_call_id": "call_1", "content": "result"}],
            tools=[{"type": "function", "function": {"name": "web_search",
                                                     "description": "d",
                                                     "parameters": {"type": "object"}}}],
            max_tokens=16)
        self.assertIn("/models/gemini-2.0-flash:generateContent", url)
        self.assertIn("key=" + GEM, url)
        self.assertEqual(headers, {})
        self.assertEqual(body["system_instruction"]["parts"][0]["text"], "sys")
        self.assertEqual(body["contents"][0]["role"], "user")
        self.assertEqual(body["contents"][1]["role"], "model")
        fc = body["contents"][1]["parts"][0]["functionCall"]
        self.assertEqual(fc["name"], "web_search")
        self.assertEqual(fc["args"], {"q": "x"})
        self.assertNotIn("arguments", fc,
                         "Gemini's function_call field is `args`")
        self.assertIn("functionResponse", body["contents"][2]["parts"][0])
        # function_declarations must not carry the OpenAI "type" wrapper
        decls = body["tools"][0]["function_declarations"]
        self.assertEqual(decls[0]["name"], "web_search")

    def test_gemini_parse_reads_text_toolcalls_and_usage(self):
        data = {"candidates": [{"content": {"parts": [
            {"text": "hello"},
            {"functionCall": {"name": "run_python",
                              "args": {"code": "1"}}}]}}],
            "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 5}}
        step = AB.GeminiAdapter().parse(data, model="gemini-2.0-flash", key_id="k")
        self.assertEqual(step.text, "hello")
        self.assertEqual(step.tool_calls[0]["function"]["name"], "run_python")
        self.assertEqual(step.tool_calls[0]["function"]["arguments"], {"code": "1"})
        self.assertEqual(step.usage["prompt_tokens"], 11)
        self.assertEqual(step.usage["completion_tokens"], 5)

    def test_gemini_parse_rejects_empty_candidates(self):
        with self.assertRaises(CloudError):
            AB.GeminiAdapter().parse({"candidates": []}, model="m", key_id="k")

    def test_gemini_request_uses_args_not_arguments(self):
        _, body, _ = AB.GeminiAdapter().request(
            key=GEM, model="gemini-2.5-flash", system="s",
            messages=[{"role": "user", "content": "hi"},
                      {"role": "assistant", "content": "",
                       "tool_calls": [{"function": {"name": "web_search",
                                                    "arguments": {"q": "x"}}}]},
                      {"role": "tool", "name": "web_search",
                       "tool_call_id": "call_1", "content": "r"}],
            tools=[{"type": "function", "function": {"name": "web_search",
                                                     "description": "d",
                                                     "parameters": {"type": "object"}}}],
            max_tokens=16)
        fc = body["contents"][1]["parts"][0]["functionCall"]
        self.assertIn("args", fc)
        self.assertEqual(fc["args"], {"q": "x"})
        self.assertNotIn("arguments", fc)
        wire = json.dumps(body, ensure_ascii=False)
        self.assertNotIn("arguments", wire,
                         "Gemini rejects `arguments` at function_call with 400")
        json.loads(wire)

    def test_gemini_parser_reads_args(self):
        data = {"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "run_python",
                              "args": {"code": "1"}}}]}}]}
        step = AB.GeminiAdapter().parse(data, model="gemini-2.5-flash",
                                        key_id="k")
        self.assertEqual(step.tool_calls[0]["function"]["name"], "run_python")
        self.assertEqual(step.tool_calls[0]["function"]["arguments"],
                         {"code": "1"})

    def test_gemini_parser_args_edge_shapes(self):
        def calls(part):
            data = {"candidates": [{"content": {"parts": [part]}}]}
            return AB.GeminiAdapter().parse(
                data, model="m", key_id="k").tool_calls

        self.assertEqual(
            calls({"functionCall": {"name": "t", "args": {}}})
            [0]["function"]["arguments"], {})
        self.assertEqual(
            calls({"functionCall": {"name": "t", "args": None}})
            [0]["function"]["arguments"], {})
        self.assertEqual(
            calls({"functionCall": {"name": "t"}})
            [0]["function"]["arguments"], {})
        mixed = calls({"functionCall": {"name": "t", "args": {"a": 1},
                                        "arguments": {"a": 2}}})
        self.assertEqual(mixed[0]["function"]["arguments"], {"a": 1},
                         "the spec field `args` must win over `arguments`")
        legacy = calls({"functionCall": {"name": "t",
                                         "arguments": {"a": 3}}})
        self.assertEqual(legacy[0]["function"]["arguments"], {"a": 3})

    def test_anthropic_request_shape(self):
        url, body, headers = AB.AnthropicAdapter().request(
            key=ALPHA, model="claude-3-5-haiku-latest", system="sys",
            messages=[{"role": "user", "content": "hi"},
                      {"role": "assistant", "content": "",
                       "tool_calls": [{"id": "call_1",
                                       "function": {"name": "read_file",
                                                    "arguments": {"path": "a"}}}]},
                      {"role": "tool", "name": "read_file",
                       "tool_call_id": "call_1", "content": "data"}],
            tools=[{"type": "function", "function": {
                "name": "read_file", "description": "d",
                "parameters": {"type": "object", "properties": {}}}}],
            max_tokens=16)
        self.assertEqual(url, "https://api.anthropic.com/v1/messages")
        self.assertEqual(headers["x-api-key"], ALPHA)
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        self.assertEqual(body["system"], "sys")
        self.assertEqual(body["messages"][0]["role"], "user")
        blocks = body["messages"][1]["content"]
        self.assertEqual(blocks[0]["type"], "tool_use")
        self.assertEqual(blocks[0]["input"], {"path": "a"})
        self.assertEqual(body["messages"][2]["content"][0]["type"], "tool_result")
        self.assertEqual(body["tools"][0]["name"], "read_file")

    def test_anthropic_parse_reads_text_and_tool_use(self):
        data = {"content": [{"type": "text", "text": "two calls: "},
                            {"type": "tool_use", "name": "a", "id": "c1",
                             "input": {"x": 1}}],
                "usage": {"input_tokens": 9, "output_tokens": 2}}
        step = AB.AnthropicAdapter().parse(data, model="m", key_id="k")
        self.assertEqual(step.text, "two calls: ")
        self.assertEqual(step.tool_calls[0]["function"]["arguments"], {"x": 1})
        self.assertEqual(step.usage["prompt_tokens"], 9)
        self.assertEqual(step.usage["completion_tokens"], 2)

    def test_anthropic_parse_rejects_empty_body(self):
        with self.assertRaises(CloudError):
            AB.AnthropicAdapter().parse({"content": []}, model="m", key_id="k")


class ModelPresetTests(unittest.TestCase):
    """Preset lists and the registry — the dead models from the live log."""

    def test_models_gemini_has_no_1_5_flash(self):
        self.assertNotIn("gemini-1.5-flash", AB.MODELS["gemini"])

    def test_models_gemini_has_current_flash(self):
        for m in ("gemini-2.5-flash", "gemini-2.0-flash",
                  "gemini-2.5-flash-lite"):
            self.assertIn(m, AB.MODELS["gemini"], m)
        self.assertIn("gemini-2.5-pro", AB.MODELS["gemini"])

    def test_openrouter_no_retired_free_slug(self):
        models = AB.MODELS["openrouter"]
        self.assertNotIn("google/gemini-2.5-flash:free", models)
        self.assertIn("google/gemini-2.5-flash", models)
        for m in ("deepseek/deepseek-chat", "meta-llama/llama-3.1-8b-instruct"):
            self.assertIn(m, models, m)
        self.assertTrue(models)

    def test_registry_has_new_gemini_entries(self):
        for m in ("gemini-2.5-flash", "gemini-2.0-flash",
                  "gemini-2.5-flash-lite"):
            info = MR.describe_limit(m, "", limits={})
            self.assertEqual(info["limit"], 1_048_576, m)
            self.assertEqual(info["source"], "exact", m)
        self.assertNotIn("gemini-1.5-flash", MR.MODEL_CONTEXT_LIMITS)
        openrouter = MR.describe_limit("google/gemini-2.5-flash", "", limits={})
        self.assertEqual(openrouter["limit"], 1_048_576)
        self.assertEqual(openrouter["source"], "exact")

    def test_new_presets_classify_into_a_prompt_tier(self):
        from core import prompts as P
        for m in ("gemini-2.0-flash", "gemini-2.5-flash-lite",
                  "google/gemini-2.5-flash"):
            self.assertEqual(P.classify_tier("cloud", m), "cloud_fast", m)
        self.assertEqual(P.classify_tier("cloud", "gemini-2.5-pro"),
                         "cloud_strong")


class NormaliseTests(unittest.TestCase):
    """Stored transcript → the shape each cloud API expects."""

    def test_calls_are_reattached_to_their_results(self):
        msgs = [{"role": "user", "text": "find it"},
                {"role": "tool_call", "tool": "web_search", "args": {"q": "x"}},
                {"role": "tool_result", "tool": "web_search", "text": "res"},
                {"role": "assistant", "text": "found"}]
        out = normalize(msgs)
        self.assertEqual([m["role"] for m in out],
                         ["user", "assistant", "tool", "assistant"])
        self.assertEqual(out[1]["tool_calls"][0]["id"], out[2]["tool_call_id"])
        self.assertEqual(out[1]["tool_calls"][0]["function"]["name"], "web_search")
        self.assertEqual(out[3]["content"], "found")

    def test_call_ids_are_stable_across_calls(self):
        msgs = [{"role": "user", "text": "a"},
                {"role": "tool_call", "tool": "t", "args": {}},
                {"role": "tool_result", "tool": "t", "text": "r"}]
        self.assertEqual(normalize(msgs)[1]["tool_calls"][0]["id"],
                         normalize(msgs)[1]["tool_calls"][0]["id"])

    def test_tool_arguments_are_json_strings(self):
        """DeepSeek/OpenAI deserialise `arguments` as a string —
        a dict here drew `422 invalid type: map, expected a string`."""
        out = normalize([{"role": "user", "text": "go"},
                         {"role": "tool_call", "tool": "t",
                          "args": {"a": 1, "b": "မြန်မာ"}},
                         {"role": "tool_result", "tool": "t", "text": "r"}])
        raw = out[1]["tool_calls"][0]["function"]["arguments"]
        self.assertIsInstance(raw, str)
        self.assertEqual(json.loads(raw), {"a": 1, "b": "မြန်မာ"})

    def test_an_unanswered_call_is_dropped(self):
        """A trailing `tool_call` with no result must not be replayed: an
        assistant message with `tool_calls` and no matching tool message is a
        400 on OpenAI/Anthropic/DeepSeek."""
        out = normalize([{"role": "user", "text": "go"},
                         {"role": "tool_call", "tool": "t", "args": {}}])
        self.assertEqual([m["role"] for m in out], ["user"])
        self.assertNotIn("tool_calls", out[0])

    def test_reasoning_content_round_trips(self):
        """DeepSeek thinking mode (`400 reasoning_content ... must be
        passed back`) — the reasoning must ride with the assistant turn."""
        out = normalize([
            {"role": "user", "text": "go"},
            {"role": "tool_call", "tool": "t", "args": {"a": 1},
             "reasoning": "thinking..."},
            {"role": "tool_result", "tool": "t", "text": "done"},
        ])
        self.assertEqual(out[1]["reasoning_content"], "thinking...")
        out2 = normalize([{"role": "assistant", "text": "ok",
                           "reasoning": "pondered"}])
        self.assertEqual(out2[0]["reasoning_content"], "pondered")
        # no reasoning → the key must not appear at all
        out3 = normalize([{"role": "assistant", "text": "ok"}])
        self.assertNotIn("reasoning_content", out3[0])
        # an EMPTY reasoning is still replayed — presence, not truthiness
        out4 = normalize([{"role": "assistant", "text": "ok",
                           "reasoning": ""}])
        self.assertEqual(out4[0]["reasoning_content"], "")

    def test_reasoning_content_preserved_even_when_empty(self):
        """DeepSeek thinking mode rejects a MISSING `reasoning_content`
        (`400 ... must be passed back to the API`) — so an empty
        reasoning is replayed as "", and a stored None becomes ""."""
        out = normalize([{"role": "assistant", "text": "ok",
                          "reasoning": ""}])
        self.assertIn("reasoning_content", out[0])
        self.assertEqual(out[0]["reasoning_content"], "")
        out2 = normalize([{"role": "assistant", "text": "ok",
                           "reasoning_content": ""}])
        self.assertEqual(out2[0]["reasoning_content"], "")
        out3 = normalize([{"role": "assistant", "text": "ok",
                           "reasoning_content": None}])
        self.assertEqual(out3[0]["reasoning_content"], "")
        # never recorded → the key stays absent
        out4 = normalize([{"role": "assistant", "text": "ok"}])
        self.assertNotIn("reasoning_content", out4[0])

    def test_normalize_args_json_serializable(self):
        """Nested/Unicode args still serialise to a wire string."""
        out = normalize([{"role": "user", "text": "go"},
                         {"role": "tool_call", "tool": "t",
                          "args": {"a": {"b": [1, "မြန်မာ"]}}},
                         {"role": "tool_result", "tool": "t", "text": "r"}])
        raw = out[1]["tool_calls"][0]["function"]["arguments"]
        self.assertIsInstance(raw, str)
        self.assertEqual(json.loads(raw), {"a": {"b": [1, "မြန်မာ"]}})

    def test_result_without_call_is_synthesised(self):
        out = normalize([{"role": "tool_result", "tool": "t", "text": "r"}])
        self.assertEqual(out[0]["role"], "assistant")
        self.assertEqual(out[1]["role"], "tool")
        self.assertEqual(out[1]["content"], "r")

    def test_trimmed_trailing_call_is_dropped(self):
        out = normalize([{"role": "user", "text": "a"},
                         {"role": "tool_call", "tool": "t", "args": {}}])
        self.assertEqual([m["role"] for m in out], ["user"])

    def test_empty_and_none_history(self):
        self.assertEqual(normalize(None), [])
        self.assertEqual(normalize([]), [])


class ReasoningPolicyTests(unittest.TestCase):
    """The per-provider `reasoning_content` wire policy."""

    def test_deepseek_thinking_injects_empty_reasoning(self):
        msgs = [{"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ok"}]
        out = _apply_reasoning_policy(msgs, "deepseek", True)
        self.assertEqual(out[1]["reasoning_content"], "")
        # the caller's transcript is never mutated
        self.assertNotIn("reasoning_content", msgs[1])

    def test_deepseek_thinking_keeps_stored_reasoning(self):
        msgs = [{"role": "assistant", "content": "ok",
                 "reasoning_content": "pondered"}]
        out = _apply_reasoning_policy(msgs, "deepseek", True)
        self.assertEqual(out[0]["reasoning_content"], "pondered")

    def test_openai_strips_reasoning(self):
        msgs = [{"role": "assistant", "content": "ok",
                 "reasoning_content": "hmm"}]
        out = _apply_reasoning_policy(msgs, "openai", False)
        self.assertNotIn("reasoning_content", out[0])
        self.assertEqual(msgs[0]["reasoning_content"], "hmm",
                         "the input transcript is left untouched")

    def test_thinking_model_detection(self):
        self.assertTrue(_is_thinking_model("deepseek", "deepseek-flash"))
        self.assertTrue(_is_thinking_model("deepseek", "deepseek-reasoner"))
        self.assertTrue(_is_thinking_model("deepseek", ""))
        self.assertFalse(_is_thinking_model("deepseek", "deepseek-chat"))
        self.assertFalse(_is_thinking_model("openai", "gpt-4o"))
        self.assertFalse(_is_thinking_model("gemini", "gemini-2.5-flash"))

    def test_sanitize_masks_keys_and_caps(self):
        body = "Invalid API key: sk-abcdefghij1234567890 and " + "x" * 600
        out = AB._sanitize(body, 100)
        self.assertNotIn("sk-abcdefghij", out)
        self.assertIn("sk-****", out)
        self.assertLessEqual(len(out), 100)


class ToolSchemaTests(unittest.TestCase):
    def test_openai_style_dict_passes_through(self):
        sch = tool_schema({"type": "function", "function": {"name": "f"}})
        self.assertEqual(sch["function"]["name"], "f")

    def test_bare_function_dict_is_wrapped(self):
        sch = tool_schema({"function": {"name": "g", "parameters": {}}})
        self.assertEqual(sch["type"], "function")

    def test_test_stub_is_converted(self):
        def do_thing(x):
            """Does a thing."""
            return x
        sch = tool_schema({"name": "do_thing", "fn": do_thing})
        self.assertEqual(sch["function"]["name"], "do_thing")
        self.assertEqual(sch["function"]["description"], "Does a thing.")

    def test_real_tool_object_is_used_as_is(self):
        class FakeTool:
            def get_tool_description(self):
                return {"type": "function", "function": {"name": "real"}}
        self.assertEqual(tool_schema(FakeTool())["function"]["name"], "real")

    def test_unusable_values_are_skipped(self):
        self.assertIsNone(tool_schema(None))
        self.assertIsNone(tool_schema("nonsense"))
        self.assertIsNone(tool_schema({"nothing": "here"}))


class KeyProbeTests(BackendCase):
    """POST /api/settings/agent/keys/test — prove a key with one cheap call."""

    def test_good_key_reports_latency(self):
        be = self.backend()
        out = be.test_key("openai", ALPHA)
        self.assertTrue(out["ok"])
        self.assertIn("latency_ms", out)
        self.assertNotIn(ALPHA, json.dumps(out))

    def test_bad_key_reports_kind_without_raising(self):
        def http(url, body, headers, timeout=30):
            raise CloudError("auth", "Invalid key: " + ALPHA, 401)
        be = self.backend(http=http)
        out = be.test_key("openai", ALPHA)
        self.assertFalse(out["ok"])
        self.assertEqual(out["kind"], "auth")
        self.assertNotIn(ALPHA, json.dumps(out))

    def test_unknown_provider_is_refused(self):
        out = self.backend().test_key("nope", "sk-anything-12345678")
        self.assertFalse(out["ok"])


if __name__ == "__main__":
    unittest.main()


class WireFormatTests(BackendCase):
    """What actually goes out on the wire — the 422 regression."""

    def test_transcript_text_becomes_content(self):
        seen = {}

        def spy(url, body, headers, timeout=30):
            seen["messages"] = body.get("messages")
            return {"choices": [{"message": {"content": "ok"}}]}

        be = self.backend(http=spy)
        be.step(system="s", messages=[{"role": "user", "text": "hi"}],
                tools=[], max_tokens=16)
        msgs = seen["messages"]
        self.assertEqual(msgs, [{"role": "system", "content": "s"},
                                 {"role": "user", "content": "hi"}])

    def test_messages_that_already_have_content_are_not_emptied(self):
        seen = {}

        def spy(url, body, headers, timeout=30):
            seen["messages"] = body.get("messages")
            return {"choices": [{"message": {"content": "ok"}}]}

        be = self.backend(http=spy)
        be.step(system="s", messages=[{"role": "user", "content": "keep me"}],
                tools=[], max_tokens=16)
        self.assertEqual(seen["messages"][0],
                         {"role": "system", "content": "s"})
        self.assertEqual(seen["messages"][1]["content"], "keep me")

    def test_tool_transcript_is_replayed_in_cloud_shape(self):
        out = normalize([{"role": "user", "text": "go"},
                         {"role": "tool_call", "tool": "t", "args": {"a": 1}},
                         {"role": "tool_result", "tool": "t", "text": "done"}])
        self.assertEqual([m["role"] for m in out], ["user", "assistant", "tool"])
        self.assertEqual(out[2]["content"], "done")

    def test_tool_transcript_goes_out_with_string_arguments(self):
        """The 422 regression: dict arguments on the wire."""
        seen = {}

        def spy(url, body, headers, timeout=30):
            seen["messages"] = body.get("messages")
            return {"choices": [{"message": {"content": "ok"}}]}

        be = self.backend(http=spy)
        be.step(system="s", messages=[
            {"role": "user", "text": "go"},
            {"role": "tool_call", "tool": "t", "args": {"a": 1}},
            {"role": "tool_result", "tool": "t", "text": "done"}],
            tools=[], max_tokens=16)
        msgs = seen["messages"]
        self.assertEqual([m["role"] for m in msgs],
                         ["system", "user", "assistant", "tool"])
        fn = msgs[2]["tool_calls"][0]["function"]
        self.assertEqual(fn["name"], "t")
        self.assertIsInstance(fn["arguments"], str,
                              "providers reject dict arguments")
        self.assertEqual(json.loads(fn["arguments"]), {"a": 1})
        self.assertEqual(msgs[3]["tool_call_id"],
                         msgs[2]["tool_calls"][0]["id"])

    def test_exhausted_carries_the_real_provider_error(self):
        def bad(url, body, headers, timeout=30):
            raise CloudError("bad_response", "missing field `content`",
                             status=422)

        be = self.backend(http=bad)
        with self.assertRaises(BackendExhausted) as ctx:
            be.step(system="s", messages=[{"role": "user", "text": "x"}],
                    tools=[], max_tokens=16)
        self.assertIn("422", str(ctx.exception))
        self.assertIn("missing field", str(ctx.exception))
        msg = str(ctx.exception)
        self.assertNotIn(ALPHA, msg, "a raw key reached the error message")
        self.assertIn("sk-a…", msg, "the tried key should appear masked")

    def test_custom_base_url_reaches_the_deepseek_adapter(self):
        for other in ("openai", "gemini", "openrouter", "anthropic"):
            self.disable(other)          # only DeepSeek is reachable
        self.km.load()["providers"]["deepseek"]["base_url"] = "https://eu.example/v1"
        self.km.save()
        seen = {}

        def spy(url, body, headers, timeout=30):
            seen["url"] = url
            return {"choices": [{"message": {"content": "ok"}}]}

        be = self.backend(http=spy)
        be.keys.add_key("deepseek", "sk-deepseek-000000000000", "d")
        be.step(system="s", messages=[{"role": "user", "text": "x"}],
                tools=[], max_tokens=16)
        self.assertTrue(seen["url"].startswith("https://eu.example/v1/"),
                        seen.get("url"))

    def test_reasoning_400_retries_exactly_once(self):
        """A 400 naming `reasoning_content` flips the wire policy and
        retries exactly one time before the failure is handled."""
        for other in ("openai", "gemini", "openrouter", "anthropic"):
            self.disable(other)          # only DeepSeek is reachable
        self.km.add_key("deepseek", "sk-deepseek-000000000000", "d")
        calls = []

        def http(url, body, headers, timeout=30):
            calls.append(body)
            if len(calls) == 1:
                raise CloudError(
                    "bad_response",
                    "The reasoning_content in the thinking mode must be "
                    "passed back to the API", 400)
            return {"choices": [{"message": {"content": "ok"}}]}

        be = self.backend(http=http)
        step = be.step(system="s", messages=[
            {"role": "user", "text": "go"},
            {"role": "tool_call", "tool": "t", "args": {"a": 1},
             "reasoning": "hm"},
            {"role": "tool_result", "tool": "t", "text": "done"}],
            tools=[], max_tokens=16)
        self.assertEqual(step.text, "ok")
        self.assertEqual(step.provider, "deepseek")
        self.assertEqual(len(calls), 2, "exactly one flipped retry")
        # deepseek-chat (the fixture default) is not a thinking model:
        # the first attempt had the reasoning stripped, the 400 flipped
        # the policy to replay it. (messages[0] is the system prompt,
        # [1] the user turn, [2] the assistant turn with the tool call.)
        first = calls[0]["messages"]
        second = calls[1]["messages"]
        self.assertNotIn("reasoning_content", first[2])
        self.assertEqual(second[2]["reasoning_content"], "hm")


class ConfigFreshnessTests(BackendCase):
    """Settings writes the file; a long-lived backend must see it."""

    def test_refresh_picks_up_a_mode_change_on_disk(self):
        self.km.set_mode("auto")
        self.km.save()
        be = self.backend()
        self.assertEqual(be.mode, "auto")

        self.km.set_mode("cloud")
        self.km.save()          # a different request (Settings) did this
        self.assertEqual(be.mode, "cloud",
                         "a cached backend_mode survived a Settings save")

    def test_refresh_picks_up_a_new_key(self):
        be = self.backend()
        before = be.usable()
        KeyManager().add_key("deepseek", "sk-deepseek-111111111111", "d")
        self.assertNotEqual(be.usable(), before,
                            "a key added from Settings was invisible")

    def test_refresh_keeps_the_rotation_cursor(self):
        be = self.backend()
        be.usable()
        self.km.save()
        be.refresh()
        self.assertEqual(be.keys._cursor, self.km._cursor,
                         "refresh threw away the round-robin position")


class ContextWindowTests(BackendCase):
    """The window comes from the model, not from the local KV cache.

    `Config.context_size` is 8192 for the on-device engine. Dividing a 128K
    cloud request by it reports 6% forever, and a 64K model at 62% looks empty
    right up until the provider refuses it.
    """

    def setUp(self):
        super().setUp()
        # A learned limit must never escape into the real ~/.ai-agent, and one
        # test's refusal must not decide the next test's window.
        self._limits = MR.limits_path()
        MR.reset_limits(self._limits)

    def tearDown(self):
        MR.reset_limits(self._limits)

    def use_model(self, model, provider="openai"):
        cfg = self.km.load()
        cfg.setdefault("providers", {}).setdefault(provider, {})
        cfg["providers"][provider]["model"] = model
        self.km.save()

    def test_limit_follows_the_model_not_the_local_cache(self):
        self.use_model("gpt-4o")
        be = self.backend()
        be.step(system="s", messages=[{"role": "user", "text": "hi"}],
                tools=[], max_tokens=16)
        state = be.context_state()
        self.assertEqual(state["model"], "gpt-4o")
        self.assertEqual(state["max"], 128_000)
        self.assertNotEqual(state["max"], 8192)
        self.assertEqual(be.current_max_ctx, 128_000)

    def test_context_state_is_a_copy(self):
        be = self.backend()
        state = be.context_state()
        state["max"] = 1
        self.assertNotEqual(be.context_state()["max"], 1)

    def test_usage_is_estimated_before_the_provider_answers(self):
        self.use_model("gpt-4o")
        be = self.backend()
        be.step(system="s", messages=[{"role": "user", "text": "hi"}],
                tools=[], max_tokens=16)
        self.assertGreater(be.current_used_ctx, 0)
        self.assertTrue(be.context_state()["exact"],
                        "the transport reported prompt_tokens=7 — that is exact")

    def test_big_history_is_trimmed_into_the_window(self):
        """A transcript far past the window must be cut, not sent as a 400."""
        self.use_model("gpt-4o")           # 128K, so force it with real bulk
        seen = {}

        def http(url, body, headers, timeout=30):
            seen["messages"] = body.get("messages", [])
            seen["n"] = len(seen["messages"])
            return {"choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3}}

        messages = [{"role": "system", "text": "system " * 100}]
        for i in range(400):
            messages.append({"role": "user", "text": "question %d %s" % (i, "q" * 900)})
            messages.append({"role": "assistant", "text": "answer %d %s" % (i, "a" * 900)})
        be = self.backend(http=http)
        be.step(system="system " * 100, messages=messages, tools=[], max_tokens=16)
        state = be.context_state()
        self.assertTrue(state["trimmed"])
        self.assertGreater(state["dropped"], 0)
        self.assertLess(state["used"], state["max"])
        self.assertLess(seen["n"], len(messages),
                        "the wire request still carried the whole transcript")
        self.assertEqual(seen["messages"][0]["role"], "system")

    def test_tiny_window_trims_a_small_transcript(self):
        """A 4K model on purpose — the 8K KV cache would have hidden this."""
        self.use_model("phi-3-mini")
        msgs = [{"role": "user", "text": "x" * 8000},
                {"role": "assistant", "text": "y" * 8000},
                {"role": "user", "text": "now"}]
        be = self.backend()
        be.step(system="", messages=msgs, tools=[], max_tokens=16)
        state = be.context_state()
        self.assertEqual(state["max"], 4_000)
        self.assertTrue(state["trimmed"])

    def test_no_trim_when_it_already_fits(self):
        self.use_model("gpt-4o")
        be = self.backend()
        be.step(system="s", messages=[{"role": "user", "text": "hi"}],
                tools=[], max_tokens=16)
        self.assertFalse(be.context_state()["trimmed"])
        self.assertEqual(be.context_state()["dropped"], 0)

    def test_context_refusal_is_retried_once_at_three_quarters(self):
        """A 400 about context length earns exactly one smaller retry."""
        self.use_model("gpt-4o")
        calls = []

        def http(url, body, headers, timeout=30):
            calls.append(len(body.get("messages", [])))
            if len(calls) == 1:
                raise bad_request("This model's maximum context length is "
                                  "128000 tokens. However, your messages "
                                  "resulted in 140000 tokens.")
            return {"choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3}}

        msgs = [{"role": "user", "text": "q" * 2000},
                {"role": "assistant", "text": "a" * 2000},
                {"role": "user", "text": "now"}]
        be = self.backend(http=http)
        step = be.step(system="", messages=msgs, tools=[], max_tokens=16)
        self.assertEqual(step.text, "ok")
        self.assertEqual(len(calls), 2, "exactly one retry")

    def test_the_retry_is_only_offered_once(self):
        self.use_model("gpt-4o")
        self.disable("gemini")               # two openai keys, nothing else
        calls = []

        def http(url, body, headers, timeout=30):
            calls.append(1)
            raise bad_request("prompt is too long: 140000 tokens > 128000 maximum")

        be = self.backend(http=http)
        with self.assertRaises(BackendExhausted):
            be.step(system="", messages=[{"role": "user", "text": "q" * 200}],
                    tools=[], max_tokens=16)
        # Each key gets its single retry, then the turn fails: a context refusal
        # must not turn into an endless walk of the fallback chain.
        self.assertEqual(len(calls), 4, "2 keys × (1 try + 1 retry)")

    def test_a_reported_limit_is_learned_for_next_time(self):
        self.use_model("brand-new-model-x")
        # Learned limits are filed per host when a base URL is configured, so
        # the lookup has to use the same base URL the turn used.
        base = (self.km.load()["providers"]["openai"]["base_url"] or "")
        self.assertIsNone(MR.learned_limit("brand-new-model-x", base))
        calls = []

        def http(url, body, headers, timeout=30):
            calls.append(1)
            if len(calls) == 1:
                raise bad_request("This model's maximum context length is "
                                  "40000 tokens.")
            return {"choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3}}

        be = self.backend(http=http)
        be.step(system="", messages=[{"role": "user", "text": "hi"}],
                tools=[], max_tokens=16)
        self.assertEqual(be.context_state()["max"], 40_000)
        self.assertEqual(be.context_state()["source"], "runtime")
        self.assertEqual(MR.learned_limit("brand-new-model-x", base), 40_000)

    def test_a_plain_400_is_not_retried_as_context(self):
        self.use_model("gpt-4o")
        self.disable("gemini")
        calls = []

        def http(url, body, headers, timeout=30):
            calls.append(1)
            raise bad_request("unsupported parameter: max_tokens")

        be = self.backend(http=http)
        with self.assertRaises(BackendExhausted):
            be.step(system="", messages=[{"role": "user", "text": "hi"}],
                    tools=[], max_tokens=16)
        self.assertEqual(len(calls), 2, "two keys, no context retry")

    def test_planned_context_answers_before_the_first_turn(self):
        """The meter must be right on a fresh process, not only after a turn."""
        self.use_model("gpt-4o")
        info = self.backend().planned_context()
        self.assertEqual(info["limit"], 128_000)
        self.assertEqual(info["model"], "gpt-4o")
        self.assertEqual(info["provider"], "openai")

    def test_planned_context_without_any_key_still_answers(self):
        self.exhaust("openai")
        self.exhaust("gemini")
        self.exhaust("anthropic")
        info = self.backend().planned_context()
        self.assertGreater(info["limit"], 0)

    def test_status_carries_the_window(self):
        self.use_model("gpt-4o")
        st = self.backend().status()
        self.assertEqual(st["providers"]["openai"]["context_limit"], 128_000)
        self.assertIn("context", st)
        self.assertEqual(st["context"]["max"], 0,
                         "no turn has run yet, so nothing is claimed")


class LocalNameTests(unittest.TestCase):
    def test_friendly_name_strips_the_extension(self):
        import config
        old = getattr(config, "MODEL_PATH", None)
        old_file = getattr(config, "CONFIG_FILE", None)
        try:
            config.MODEL_PATH = "/x/y/gemma-4-E4B-it.litertlm"
            self.assertEqual(friendly_local_name(), "Gemma 4")
            config.MODEL_PATH = "/x/y/some_other_model.gguf"
            self.assertEqual(friendly_local_name(), "some other model")
        finally:
            if old is None:
                del config.MODEL_PATH
            else:
                config.MODEL_PATH = old
            if old_file is not None:
                config.CONFIG_FILE = old_file

    def test_friendly_name_falls_back_to_the_config_file(self):
        """No module-level MODEL_PATH exists — config.json is the source."""
        import json as _json
        import config
        old = getattr(config, "MODEL_PATH", None)
        # This test rewrites the shared config.json: keep its original bytes
        # and put them back. Leaving {"model_path": ""} behind poisons every
        # suite that runs after this one in the same process — their engine
        # loads then find no configured model and (off the phone, where
        # auto-discovery finds the real download) fail with "model unset".
        old_file_text = (config.CONFIG_FILE.read_text(encoding="utf-8")
                         if config.CONFIG_FILE.exists() else None)
        try:
            if hasattr(config, "MODEL_PATH"):
                del config.MODEL_PATH
            config.CONFIG_FILE.write_text(
                _json.dumps({"model_path": "/models/gemma-4-E4B-it.litertlm"}),
                encoding="utf-8")
            self.assertEqual(friendly_local_name(), "Gemma 4")
            config.CONFIG_FILE.write_text(_json.dumps({"model_path": ""}),
                                          encoding="utf-8")
            self.assertEqual(friendly_local_name(), "Local model")
        finally:
            if old is None:
                try:
                    del config.MODEL_PATH
                except AttributeError:
                    pass
            else:
                config.MODEL_PATH = old
            if old_file_text is None:
                config.CONFIG_FILE.unlink(missing_ok=True)
            else:
                config.CONFIG_FILE.write_text(old_file_text, encoding="utf-8")
