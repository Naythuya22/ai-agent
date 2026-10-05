"""Cloud token streaming (Batch 2) — the SSE contract, layer by layer.

Three layers, none of them allowed to open a socket:

  * ``Adapter.request_stream`` — one provider wire dialect in, ``("delta",
    str)`` fragments and exactly one ``("done", info)`` out;
  * ``AgentBackend.step_stream`` — the retry ladder ``step()`` walks, ending
    in an empty stream instead of ``BackendExhausted``;
  * ``POST /api/agent/stream/v2`` — the named SSE frames (``delta`` /
    ``tool`` / ``error`` / ``done`` / ``fallback``) the frontend reads.

Every transport here is an injected callable: the adapters get
``http_stream=`` through kwargs, the backend gets it through ``step_stream``,
and the route never reaches a provider at all because ``_handle_cloud_stream``
is canned.

Run:  sh tests/run.sh -p "test_streaming.py"
"""
from __future__ import annotations

import json
import unittest

from _harness import bootstrap, get_app

bootstrap()

import config  # noqa: E402
from core import agent_backend as AB  # noqa: E402
from core.agent_backend import (  # noqa: E402
    AgentBackend,
    AnthropicAdapter,
    CloudError,
    GeminiAdapter,
    OpenAIAdapter,
    Step,
)
from core.key_manager import KeyManager  # noqa: E402
from modes.agent import AgentMode  # noqa: E402
from routes._state import ensure_modes, get_agent_mode  # noqa: E402

app = get_app()
client = app.test_client()

ALPHA = "sk-stream-aaaaaaaaaaaaaaaa1111"
BETA = "sk-stream-bbbbbbbbbbbbbbbbbbbb2222"
GEM = "AIza-stream-cccccccccccccc3333"
MODEL = "test-model"


# ───────────────────────── local helpers (Step 5) ─────────────────────────


class _StreamStub:
    """``http_stream`` stand-in: replays a fixed list of SSE lines."""

    def __init__(self, lines):
        self.lines = list(lines)

    def __call__(self, url, payload, headers, timeout=None):
        for line in self.lines:
            yield line


class _ScriptedStream:
    """``http_stream`` stand-in with one scripted body per attempt.

    A body is a list of steps: a string is yielded as an SSE line, an
    exception is raised at that point — before any line (a pre-first-byte
    failure) or after some of them (a post-first-byte one). The last body
    repeats for any further attempt, which is how a transport that fails
    every provider is spelled.
    """

    def __init__(self, script):
        self.script = [list(body) for body in script]
        self.calls = []

    def __call__(self, url, payload, headers, timeout=None):
        self.calls.append((url, payload, headers))
        body = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        for step in body:
            if isinstance(step, BaseException):
                raise step
            yield step


def _http_ok(url, payload, headers, timeout=30):
    """Non-stream transport: a plain completion, never a socket."""
    return {"choices": [{"message": {"content": "NON_STREAM_OK"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


def _no_http(url, payload, headers, timeout=30):
    raise AssertionError("the non-stream transport was opened: %s" % url)


def _fresh_keys(keys: dict) -> KeyManager:
    """A clean key file shaped ``{provider: [key, ...]}``, cost acknowledged."""
    km = KeyManager()
    if km.path.exists():
        km.path.unlink()
    for provider, bucket in keys.items():
        for key in bucket:
            km.add_key(provider, key, provider)
    cfg = km.load()
    cfg["prefer_cloud"] = True
    cfg["cost_ack"] = True
    km.save()
    return km


def _set_model(km: KeyManager, provider: str, model: str) -> None:
    cfg = km.load()
    cfg["providers"] = dict(cfg["providers"])
    cfg["providers"][provider] = dict(cfg["providers"][provider])
    cfg["providers"][provider]["model"] = model
    km.save()


def _backend(km: KeyManager, net: str = "online", http=None) -> AgentBackend:
    return AgentBackend(keys=km, net_state=lambda: net, http=http or _no_http)


def _stream(be: AgentBackend, stub, messages=None) -> list:
    return list(be.step_stream(
        system="",
        messages=messages if messages is not None
        else [{"role": "user", "text": "hi"}],
        tools=[], max_tokens=32, http_stream=stub))


def _lines(*pairs) -> list:
    """SSE lines for ``(event_name, payload)`` pairs, each blank-dispatched.

    A payload that is a string is written through unchanged (that is how
    ``[DONE]`` arrives); anything else is JSON-encoded.
    """
    out = []
    for event, payload in pairs:
        if event:
            out.append("event: %s" % event)
        out.append("data: %s" % (payload if isinstance(payload, str)
                                 else json.dumps(payload)))
        out.append("")
    return out


def _delta(text: str) -> dict:
    return {"choices": [{"delta": {"content": text}}]}


def _openai_body(*chunks, usage=None, done=True) -> list:
    """An OpenAI-dialect body: deltas, an optional usage chunk, ``[DONE]``."""
    pairs = [("", chunk) for chunk in chunks]
    if usage is not None:
        pairs.append(("", {"choices": [], "usage": usage}))
    if done:
        pairs.append(("", "[DONE]"))
    return _lines(*pairs)


def _events(resp) -> list:
    """``[(event_name, data), ...]`` out of a named-SSE response body."""
    out, name = [], None
    for line in resp.get_data(as_text=True).split("\n"):
        if line.startswith("event: "):
            name = line[7:]
        elif line.startswith("data: "):
            out.append((name, json.loads(line[6:])))
            name = None
    return out


# ──────────────────── Step 1 — adapter SSE parsers ────────────────────


class AdapterStreamTests(unittest.TestCase):
    """``Adapter.request_stream`` with an injected ``http_stream``."""

    def _run(self, adapter, lines) -> list:
        return list(adapter.request_stream(
            key=ALPHA, model=MODEL, system="",
            messages=[{"role": "user", "content": "hi"}], tools=[],
            max_tokens=64, http_stream=_StreamStub(lines)))

    def _done(self, frames) -> dict:
        kinds = [k for k, _ in frames]
        self.assertEqual(kinds.count("done"), 1, kinds)
        self.assertEqual(kinds[-1], "done", kinds)
        return frames[-1][1]

    # ---------- OpenAI dialect (also OpenRouter + DeepSeek) ----------

    def test_openai_stream_parses_text_delta(self):
        frames = self._run(OpenAIAdapter(),
                           _openai_body(_delta("မ"), _delta("င်္")))
        self.assertEqual([k for k, _ in frames], ["delta", "delta", "done"])
        self.assertEqual([p for k, p in frames if k == "delta"],
                         ["မ", "င်္"])
        info = self._done(frames)
        self.assertEqual(info["provider"], "openai")
        self.assertEqual(info["model"], MODEL)
        self.assertEqual(info["tool_calls"], [])
        self.assertEqual(info["reasoning"], "")
        self.assertEqual(info["usage"], {})

    def test_openai_stream_merges_tool_calls_by_index(self):
        chunks = [
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"name": "list_files", "arguments": ""}}]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": '{"dire'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": 'ctory":"."}'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 1, "function": {
                    "name": "read_file",
                    "arguments": '{"path":"a.txt"}'}}]}}]},
        ]
        frames = self._run(OpenAIAdapter(), _openai_body(*chunks))
        self.assertEqual([k for k, _ in frames], ["done"])
        self.assertEqual(self._done(frames)["tool_calls"], [
            {"function": {"name": "list_files",
                          "arguments": {"directory": "."}}},
            {"function": {"name": "read_file",
                          "arguments": {"path": "a.txt"}}},
        ])

    def test_openai_stream_accumulates_reasoning_content(self):
        frames = self._run(OpenAIAdapter(), _openai_body(
            {"choices": [{"delta": {"reasoning_content": "think "}}]},
            {"choices": [{"delta": {"reasoning_content": "hard"}}]},
            _delta("answer"),
        ))
        self.assertEqual([p for k, p in frames if k == "delta"], ["answer"])
        info = self._done(frames)
        self.assertEqual(info["reasoning"], "think hard")
        self.assertEqual(info["usage"], {})

    def test_openai_stream_stops_on_done_marker(self):
        frames = self._run(OpenAIAdapter(), _openai_body(
            _delta("before"), done=False) + _lines(("", "[DONE]")) +
            _lines(("", _delta("after"))))
        self.assertEqual([p for k, p in frames if k == "delta"], ["before"],
                         "a chunk after [DONE] was parsed")
        self.assertEqual([k for k, _ in frames], ["delta", "done"])

    # ---------- Gemini dialect ----------

    def test_gemini_stream_parses_text_parts(self):
        lines = _lines(
            ("", {"candidates": [{"content": {"parts": [
                {"text": "မ"}, {"text": "င်္"}]}}]}),
            ("", {"candidates": [{"content": {"parts": [{"text": "!"}]}}]}),
        )
        frames = self._run(GeminiAdapter(), lines)
        self.assertEqual([p for k, p in frames if k == "delta"],
                         ["မ", "င်္", "!"])
        info = self._done(frames)
        self.assertEqual(info["provider"], "gemini")
        self.assertEqual(info["model"], MODEL)
        self.assertEqual(info["usage"], {})

    def test_gemini_stream_reads_usage_metadata(self):
        lines = _lines(
            ("", {"candidates": [{"content": {"parts": [{"text": "a"}]}}],
                  "usageMetadata": {"promptTokenCount": 5,
                                    "candidatesTokenCount": 7}}),
            ("", {"candidates": [{"content": {"parts": [{"text": "b"}]}}],
                  "usageMetadata": {"promptTokenCount": 5,
                                    "candidatesTokenCount": 9}}),
        )
        info = self._done(self._run(GeminiAdapter(), lines))
        self.assertEqual(info["usage"],
                         {"prompt_tokens": 5, "completion_tokens": 9},
                         "the last usageMetadata did not win")

    # ---------- Anthropic dialect ----------

    def test_anthropic_stream_parses_text_delta(self):
        lines = _lines(
            ("message_start", {"type": "message_start",
                               "message": {"usage": {"input_tokens": 9}}}),
            ("content_block_delta", {"type": "content_block_delta",
                                     "index": 0,
                                     "delta": {"type": "text_delta",
                                               "text": "Hi"}}),
            ("content_block_delta", {"type": "content_block_delta",
                                     "index": 0,
                                     "delta": {"type": "text_delta",
                                               "text": " there"}}),
            ("message_delta", {"type": "message_delta",
                               "usage": {"output_tokens": 7}}),
        )
        frames = self._run(AnthropicAdapter(), lines)
        self.assertEqual([p for k, p in frames if k == "delta"],
                         ["Hi", " there"])
        info = self._done(frames)
        self.assertEqual(info["usage"],
                         {"prompt_tokens": 9, "completion_tokens": 7})
        self.assertEqual(info["provider"], "anthropic")

    def test_anthropic_stream_parses_input_json_delta(self):
        lines = _lines(
            ("content_block_start", {"type": "content_block_start",
                                     "index": 0,
                                     "content_block": {
                                         "type": "tool_use",
                                         "name": "list_files"}}),
            ("content_block_delta", {"type": "content_block_delta",
                                     "index": 0,
                                     "delta": {"type": "input_json_delta",
                                               "partial_json": '{"di'}}),
            ("content_block_delta", {"type": "content_block_delta",
                                     "index": 0,
                                     "delta": {"type": "input_json_delta",
                                               "partial_json": 'rectory":"."}'}}),
        )
        frames = self._run(AnthropicAdapter(), lines)
        self.assertEqual([k for k, _ in frames], ["done"])
        self.assertEqual(self._done(frames)["tool_calls"],
                         [{"function": {"name": "list_files",
                                        "arguments": {"directory": "."}}}])

    def test_anthropic_stream_event_error_raises_cloud_error(self):
        lines = _lines(("error", {"type": "error",
                                  "error": {"type": "rate_limit_error",
                                            "message": "slow down"}}))
        with self.assertRaises(CloudError) as cm:
            self._run(AnthropicAdapter(), lines)
        self.assertEqual(cm.exception.kind, "rate_limit")
        self.assertEqual(cm.exception.status, 429)


# ──────────────── Step 2 — AgentBackend.step_stream ────────────────


class StepStreamTests(unittest.TestCase):
    """The retry ladder in front of the adapters, frame by frame."""

    def setUp(self):
        self.km = _fresh_keys({"openai": [ALPHA]})
        self.be = _backend(self.km)

    def test_step_stream_yields_deltas_then_done(self):
        stub = _ScriptedStream([_openai_body(
            _delta("Hel"), _delta("lo"),
            usage={"prompt_tokens": 2, "completion_tokens": 5})])
        frames = _stream(self.be, stub)

        self.assertEqual([k for k, _ in frames], ["delta", "delta", "done"])
        self.assertEqual([p for k, p in frames if k == "delta"], ["Hel", "lo"])
        info = frames[-1][1]
        expect_model = self.km.load()["providers"]["openai"]["model"]
        self.assertEqual(info["provider"], "openai")
        self.assertEqual(info["model"], expect_model)
        self.assertEqual(info["key_id"], self.km.entries("openai")[0]["id"])
        self.assertEqual(info["usage"],
                         {"prompt_tokens": 2, "completion_tokens": 5})
        self.assertEqual(info["tool_calls"], [])
        self.assertEqual(info["reasoning"], "")
        self.assertEqual(len(stub.calls), 1)

    def test_step_stream_pre_first_byte_error_tries_next_key(self):
        km = _fresh_keys({"openai": [ALPHA, BETA]})
        be = _backend(km)
        ids = [e["id"] for e in km.entries("openai")]
        stub = _ScriptedStream([
            [CloudError("server", "upstream boom", 502)],
            _openai_body(_delta("recovered")),
        ])
        frames = _stream(be, stub)

        self.assertEqual(len(stub.calls), 2, "the second key never got a turn")
        self.assertEqual([k for k, _ in frames], ["delta", "done"])
        self.assertEqual([p for k, p in frames if k == "delta"], ["recovered"])
        self.assertEqual(frames[-1][1]["key_id"], ids[1])
        self.assertNotEqual(ids[0], ids[1])

    def test_step_stream_pre_first_byte_error_tries_next_provider(self):
        km = _fresh_keys({"openai": [ALPHA], "gemini": [GEM]})
        be = _backend(km)
        stub = _ScriptedStream([
            [CloudError("server", "openai is down", 502)],
            _lines(("", {"candidates": [{"content": {"parts": [
                {"text": "GEM"}]}}],
                "usageMetadata": {"promptTokenCount": 3,
                                  "candidatesTokenCount": 4}})),
        ])
        frames = _stream(be, stub)

        self.assertEqual(len(stub.calls), 2, "the ladder stopped at OpenAI")
        self.assertEqual([k for k, _ in frames], ["delta", "done"])
        info = frames[-1][1]
        self.assertEqual(info["provider"], "gemini")
        self.assertEqual([p for k, p in frames if k == "delta"], ["GEM"])

    def test_step_stream_post_first_byte_error_is_in_band(self):
        body = list(_lines(("", _delta("partial")))) + [
            CloudError("server", "dropped mid-answer", 502)]
        stub = _ScriptedStream([body])
        frames = _stream(self.be, stub)

        self.assertEqual([k for k, _ in frames], ["delta", "error"],
                         "a post-first-byte failure must not retry")
        error = frames[-1][1]
        self.assertEqual(sorted(error),
                         ["error", "kind", "model", "provider", "status"])
        self.assertEqual(error["kind"], "server")
        self.assertEqual(error["status"], 502)
        self.assertEqual(error["provider"], "openai")
        self.assertIn("dropped mid-answer", error["error"])
        self.assertEqual(len(stub.calls), 1)

    def test_step_stream_all_providers_fail_yields_nothing(self):
        km = _fresh_keys({"openai": [ALPHA], "gemini": [GEM]})
        be = _backend(km)
        stub = _ScriptedStream(
            [[CloudError("server", "everything is down", 502)]])

        self.assertEqual(_stream(be, stub), [])
        self.assertEqual(len(stub.calls), 2,
                         "the ladder stopped before the second provider")

    def test_step_stream_offline_yields_nothing(self):
        be = _backend(self.km, net="offline")
        stub = _ScriptedStream([_openai_body(_delta("nope"))])

        self.assertEqual(_stream(be, stub), [])
        self.assertEqual(stub.calls, [], "the provider was asked offline")

    def test_step_stream_applies_reasoning_policy_before_open(self):
        km = _fresh_keys({"deepseek": [ALPHA]})
        _set_model(km, "deepseek", "deepseek-reasoner")
        be = _backend(km)
        stub = _ScriptedStream([_openai_body(_delta("ok"))])
        _stream(be, stub, messages=[
            {"role": "user", "text": "hi"},
            {"role": "assistant", "text": "prev", "reasoning": "THOUGHT"},
        ])

        wire = stub.calls[0][1]["messages"]
        assistant = [m for m in wire if m.get("role") == "assistant"][-1]
        self.assertEqual(assistant.get("reasoning_content"), "THOUGHT",
                         "the wire opened without the replayed reasoning")


class StreamLocalEngineTests(unittest.TestCase):
    """The streaming path owns no local call at all."""

    class _LocalSpy:
        """The on-device model: every touch lands in ``log``."""

        def __init__(self):
            self.log = []

        @property
        def loaded(self):
            return any(k == "load" for k in self.log)

        def load(self):
            self.log.append(("load", None))
            self._engine = self

        def create_conversation(self, **kw):
            self.log.append(("create_conversation", kw))
            return self

        def send_message(self, msg):
            self.log.append(("message", msg))
            return {"content": [{"text": "LOCAL_ANSWER"}]}

    def test_step_stream_never_calls_local_engine(self):
        km = _fresh_keys({"openai": [ALPHA]})
        llm = self._LocalSpy()
        mode = AgentMode(llm, None)
        mode._store = None
        mode._tools = []
        mode._messages = []
        mode._backend = _backend(km)
        stub = _StreamStub(_openai_body(_delta("CLOUD_ONLY")))

        real = AB._http_stream
        AB._http_stream = stub
        try:
            frames = list(mode._try_cloud_stream())
        finally:
            AB._http_stream = real

        self.assertEqual(frames, [("delta", "CLOUD_ONLY"),
                                  ("end", "CLOUD_ONLY")])
        self.assertEqual(llm.log, [], "the local engine was touched")
        self.assertIsNone(mode._conv)


# ──────────────── Step 3 — POST /api/agent/stream/v2 ────────────────


class V2RouteTests(unittest.TestCase):
    """The v2 SSE route: the 404 gates first, then the named frames."""

    def setUp(self):
        ensure_modes()
        self.mode = get_agent_mode()
        self.mode._messages = []
        self.mode.turns = 0
        self.mode._conv = None
        self.km = _fresh_keys({"openai": [ALPHA]})
        self.net = "online"
        self.mode._backend = AgentBackend(
            keys=self.km, net_state=lambda: self.net, http=_no_http)
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        self.mode._backend = None
        if "_handle_cloud_stream" in self.mode.__dict__:
            del self.mode._handle_cloud_stream

    def post(self, body):
        return client.post("/api/agent/stream/v2", json=body)

    def canned(self, frames):
        """Stand in for the cloud turn: no provider, no local engine."""
        self.mode._handle_cloud_stream = lambda: iter(list(frames))

    # ---------- the gates ----------

    def test_v2_404_when_flag_off(self):
        self.addCleanup(setattr, config, "AGENT_CLOUD_STREAM",
                        config.AGENT_CLOUD_STREAM)
        config.AGENT_CLOUD_STREAM = False
        resp = self.post({"message": "hi"})

        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.get_json()["error"], "cloud streaming disabled")

    def test_v2_404_when_image_present(self):
        resp = self.post({"message": "hi", "image_id": "vision.png"})

        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.get_json()["error"],
                         "image turns are handled by v1")

    def test_v2_404_when_mode_local(self):
        self.km.set_mode("local")
        self.km.save()
        resp = self.post({"message": "hi"})

        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.get_json()["error"], "local mode")

    def test_v2_404_when_should_use_cloud_false(self):
        cfg = self.km.load()
        cfg["cost_ack"] = False
        self.km.save()
        resp = self.post({"message": "hi"})

        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.get_json()["error"], "cloud unavailable")

    # ---------- the frames ----------

    def test_v2_streams_event_delta_frames(self):
        self.canned([("delta", "A"), ("delta", "B"), ("end", "AB")])
        resp = self.post({"message": "hi"})

        self.assertEqual(resp.status_code, 200)
        evs = _events(resp)
        self.assertEqual([e for e, _ in evs], ["delta", "delta", "done"])
        self.assertEqual(evs[0][1], {"t": "A"})
        self.assertEqual(evs[1][1], {"t": "B"})
        self.assertEqual(evs[2][1], {"t": "AB"})

    def test_v2_streams_one_event_done_frame(self):
        self.canned([("delta", "x"), ("end", "x")])
        evs = _events(self.post({"message": "hi"}))

        self.assertEqual(len([e for e, _ in evs if e == "done"]), 1)
        self.assertEqual(evs[-1][0], "done")
        self.assertEqual(evs[-1][1], {"t": "x"})

    def test_v2_streams_tool_frame_shape(self):
        self.canned([("tool", {"name": "list_files", "args": {"a": 1},
                               "result": "ok"}),
                     ("end", "done")])
        evs = _events(self.post({"message": "hi"}))

        self.assertEqual(evs[0], ("tool", {"name": "list_files",
                                           "args": {"a": 1},
                                           "result": "ok"}))

    def test_v2_streams_one_fallback_frame_when_empty(self):
        self.canned([])
        evs = _events(self.post({"message": "hi"}))

        self.assertEqual(evs, [("fallback", {"reason": "cloud_unavailable"})])

    def test_v2_headers_no_cache(self):
        self.canned([("end", "ok")])
        resp = self.post({"message": "hi"})

        self.assertEqual(resp.mimetype, "text/event-stream")
        self.assertEqual(resp.headers.get("Cache-Control"),
                         "no-cache, no-transform")
        self.assertEqual(resp.headers.get("X-Accel-Buffering"), "no")


# ──────────────── Step 4 — integration / regression ────────────────


class StreamingRegressionTests(unittest.TestCase):
    """Batch 2 must not move anything that already worked."""

    def test_streaming_flag_off_keeps_non_streaming_step_intact(self):
        km = _fresh_keys({"openai": [ALPHA]})
        be = _backend(km, http=_http_ok)
        saved = config.AGENT_CLOUD_STREAM
        try:
            config.AGENT_CLOUD_STREAM = False
            step = be.step(system="", messages=[{"role": "user", "text": "hi"}],
                           tools=[], max_tokens=32)
        finally:
            config.AGENT_CLOUD_STREAM = saved

        self.assertIsInstance(step, Step)
        self.assertEqual(step.text, "NON_STREAM_OK")
        self.assertEqual(step.provider, "openai")

    def test_reasoning_policy_differs_deepseek_vs_openai_on_stream_open(self):
        messages = [{"role": "user", "text": "hi"},
                    {"role": "assistant", "text": "prev",
                     "reasoning": "THOUGHT"}]

        def assistant_on_the_wire(provider, model):
            km = _fresh_keys({provider: [ALPHA]})
            _set_model(km, provider, model)
            stub = _ScriptedStream([_openai_body(_delta("ok"))])
            _stream(_backend(km), stub, messages=messages)
            wire = stub.calls[0][1]["messages"]
            return [m for m in wire if m.get("role") == "assistant"][-1]

        deepseek = assistant_on_the_wire("deepseek", "deepseek-reasoner")
        openai = assistant_on_the_wire("openai", "gpt-4o-mini")

        self.assertEqual(deepseek.get("reasoning_content"), "THOUGHT",
                         "a thinking model must replay its reasoning")
        self.assertNotIn("reasoning_content", openai,
                         "a non-thinking provider was sent an unknown field")


if __name__ == "__main__":
    unittest.main()
