"""The SSE agent stream must run the same hybrid backend as ``handle()``.

The Web UI drives Agent Mode through ``POST /api/agent/stream``. Before this
the endpoint called the local conversation directly and never consulted the
cloud backend, so a phone with a key and Auto/Cloud Only answered from the
4B model while ``POST /api/agent`` answered from the provider.

These tests drive the real endpoint with a fake provider transport (no
sockets, no weights) and pin which backend produced the ``end`` event.

Run:  sh tests/run.sh -p "test_agent_stream_cloud.py"
"""
from __future__ import annotations

import contextlib
import json
import tempfile
import threading
import unittest
from pathlib import Path

from _harness import bootstrap, current_home, get_app

MODEL = bootstrap()
app = get_app()
client = app.test_client()

import config  # noqa: E402

ALPHA = "sk-cloud-only-aaaaaaaaaaaaaaaa1111"
CLOUD_TEXT = "CLOUD_STREAM_ANSWER"
LOCAL_TEXT = "LOCAL_STREAM_ANSWER"
TOOL_TEXT = "TOOL_STREAM_DONE"


@contextlib.contextmanager
def verification_off():
    """Run a test with the verification pass switched off.

    A tool turn costs three provider requests now — the call, the answer over
    the result, and the reviewer's check on that answer. Tests that count
    requests to prove the loop works turn the reviewer off so they stay a
    statement about the loop; tests/test_verification.py is where the third
    call is asserted.
    """
    saved = config.AGENT_VERIFY_ENABLED
    config.AGENT_VERIFY_ENABLED = False
    try:
        yield
    finally:
        config.AGENT_VERIFY_ENABLED = saved


def lock_held(lock) -> bool:
    """True while some thread holds ``lock`` — on every Python version.

    ``threading.RLock`` only grew a ``locked()`` method in 3.14; the suite
    also runs on older interpreters, where the same question is answered by
    probing from a helper thread (an RLock held by anyone refuses a
    non-blocking acquire from a different thread).
    """
    locked = getattr(lock, "locked", None)
    if locked is not None:
        return bool(locked())
    outcome = {}

    def probe():
        got = lock.acquire(blocking=False)
        outcome["held"] = not got
        if got:
            lock.release()

    t = threading.Thread(target=probe)
    t.start()
    t.join()
    return outcome["held"]


def frames(resp) -> list:
    out = []
    for line in resp.get_data(as_text=True).split("\n"):
        if line.startswith("data: "):
            out.append(json.loads(line[6:]))
    return out


class RecordingTransport:
    """Stands in for a provider: records the request, answers with CLOUD_TEXT."""

    def __init__(self, reply=CLOUD_TEXT):
        self.reply = reply
        self.calls = []              # (url, body, headers) per request

    def __call__(self, url, body, headers, timeout=30):
        self.calls.append((url, body, headers))
        return {"choices": [{"message": {"content": self.reply}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2}}


def local_stream(conv, message):
    """Fake local engine stream — the real one needs a phone."""
    yield "text", LOCAL_TEXT
    yield "end", {"text": LOCAL_TEXT, "tool_calls": None}


class _StreamCase(unittest.TestCase):
    """Fresh key file + fake-backed AgentBackend for the shared agent mode."""

    def setUp(self):
        from core.agent_backend import AgentBackend
        from core.key_manager import KeyManager
        from modes import agent as agent_mod
        from routes._state import ensure_modes, get_agent_mode

        ensure_modes()
        self.agent_mode = get_agent_mode()
        # Blank slate: this module must not inherit a conversation, a
        # transcript or a usage record from an earlier test.
        self.agent_mode._messages = []
        self.agent_mode.turns = 0
        self.agent_mode._conv = None
        self.km = KeyManager()
        if self.km.path.exists():
            self.km.path.unlink()
        self.net = "online"
        self.transport = RecordingTransport()
        self.agent_mode._backend = AgentBackend(
            keys=self.km, net_state=lambda: self.net, http=self.transport)
        agent_mod.record_usage("")

    def tearDown(self):
        # Drop the fake backend: its KeyManager points at this module's temp
        # config, and the next module re-points HOME. `_backend_obj()` builds
        # a fresh one from the then-current config on the next turn.
        self.agent_mode._backend = None

    def configure(self, mode="auto", key=True, cost_ack=True, net="online"):
        self.km.set_mode(mode)
        if key:
            self.km.add_key("openai", ALPHA, "alpha")
        cfg = self.km.load()
        cfg["cost_ack"] = cost_ack
        self.km.save()
        self.net = net

    def stream(self, body):
        return frames(client.post("/api/agent/stream", json=body))

    @staticmethod
    def local_turn(fn):
        """Run `fn()` with the local model stream stubbed out."""
        import routes.chat as rc

        real = rc.stream_conversation
        rc.stream_conversation = local_stream
        try:
            return fn()
        finally:
            rc.stream_conversation = real


class TestStreamUsesCloud(_StreamCase):
    """Auto mode with a key and a network: the provider answers the stream."""

    def test_cloud_answers_the_stream(self):
        self.configure(mode="auto")
        fr = self.stream({"message": "hello"})
        end = fr[-1]

        self.assertEqual(end["type"], "end")
        self.assertEqual(end["text"], CLOUD_TEXT)
        self.assertEqual(len(self.transport.calls), 1,
                         "the provider was never asked")
        self.assertIsNone(self.agent_mode._conv,
                          "the local conversation was created for a cloud turn")

        chunks = "".join(f.get("text", "") for f in fr if f["type"] == "chunk")
        self.assertEqual(chunks, CLOUD_TEXT)

        self.assertEqual(end["last_backend"], "cloud")
        self.assertEqual(end["last_model"], end["last_model"])   # present
        self.assertIn("(Cloud, last used)", end["model_label"])
        self.assertEqual(end["status"], "ready")
        self.assertFalse(end["truncated"])

        # The chat transcript recorded the turn, and the status bar agrees.
        self.assertEqual(self.agent_mode._messages[-1]["role"], "assistant")
        j = client.get("/api/agent/status").get_json()
        self.assertEqual(j["last_backend"], "cloud")
        self.assertIn("(Cloud, last used)", j["label"])

    def test_cloud_turn_releases_the_engine_lock(self):
        """The provider step must not hold inference_lock — Chat mode stays live."""
        import routes._state as st

        self.configure(mode="auto")
        self.stream({"message": "hello"})
        self.assertFalse(lock_held(st.inference_lock),
                         "the cloud step still holds the engine lock")


class TestStreamCloudValidation(_StreamCase):
    """Provider text goes through the same filter/validate path as local text."""

    def test_broken_cloud_text_is_cut_before_it_reaches_the_ui(self):
        broken = "ထိင်" + "\u1039" * 4 + "န်"
        self.transport.reply = broken
        self.configure(mode="auto")

        fr = self.stream({"message": "hello"})
        end = fr[-1]

        chunks = "".join(f.get("text", "") for f in fr if f["type"] == "chunk")
        self.assertNotIn("\u1039\u1039", chunks, "broken sequence reached the UI")
        self.assertEqual(end["text"], "ထိင်")
        self.assertTrue(end["truncated"], "a cut must be reported")
        # and the session must replay the clean text, not the raw provider text
        self.assertEqual(self.agent_mode._messages[-1]["text"], "ထိင်")


class TestStreamFallsBackToCloud(_StreamCase):
    """Auto mode with no network: the local model answers, and says so."""

    def test_offline_auto_falls_back_to_the_local_model(self):
        self.configure(mode="auto", net="offline")
        fr = self.local_turn(lambda: self.stream({"message": "hello"}))
        end = fr[-1]

        self.assertEqual(self.transport.calls, [], "the provider was asked offline")
        self.assertEqual(end["text"], LOCAL_TEXT)
        self.assertEqual(end["last_backend"], "local")
        self.assertIn("(Local, fallback)", end["model_label"])
        self.assertIsNotNone(self.agent_mode._conv)


class TestStreamLocalOnly(_StreamCase):
    """Local Only: the transport must not be touched at all."""

    def test_local_only_never_reaches_the_transport(self):
        self.configure(mode="local", net="online")
        fr = self.local_turn(lambda: self.stream({"message": "hello"}))
        end = fr[-1]

        self.assertEqual(self.transport.calls, [])
        self.assertEqual(end["last_backend"], "local")
        self.assertIn("(Local)", end["model_label"])
        self.assertNotIn("fallback", end["model_label"])


class TestStreamCloudOnly(_StreamCase):
    """Cloud Only with no usable key: the refusal, not the local model."""

    def test_cloud_only_refuses_without_a_key(self):
        self.configure(mode="cloud", key=False)
        fr = self.stream({"message": "hello"})
        end = fr[-1]

        self.assertEqual(self.transport.calls, [])
        self.assertIsNone(self.agent_mode._conv,
                          "Cloud Only fell through to the local model")
        self.assertIn("Cloud Only mode", end["text"])
        self.assertEqual(end["last_backend"], "",
                         "a refusal must not claim a backend ran")
        self.assertEqual(end["model_label"], "",
                         "a refusal must not print a model name")


class TestStreamVisionStaysLocal(_StreamCase):
    """Image turns keep using the local path (provider payloads not wired)."""

    def test_image_turn_never_reaches_the_cloud(self):
        import routes.chat as rc

        self.configure(mode="auto")     # online + key: cloud would be chosen
        img = current_home() / "vision.png"
        img.write_bytes(b"\x89PNG fake")

        real = (rc._resolve_image, rc.supports_vision,
                rc._model_path_for_vision, rc.build_vision_message,
                rc.stream_conversation)
        rc._resolve_image = lambda _i: img
        rc.supports_vision = lambda _p: (True, "")
        rc._model_path_for_vision = lambda: "stub-model.litertlm"
        rc.build_vision_message = lambda _p, _f, _m: {
            "role": "user", "content": [{"text": "image"}],
        }
        rc.stream_conversation = local_stream
        try:
            fr = self.stream({"message": "look", "image_id": "vision.png"})
        finally:
            (rc._resolve_image, rc.supports_vision,
             rc._model_path_for_vision, rc.build_vision_message,
             rc.stream_conversation) = real

        end = fr[-1]
        self.assertEqual(self.transport.calls, [],
                         "the cloud was called for an image turn")
        self.assertEqual(end["last_backend"], "local")
        self.assertEqual(end["text"], LOCAL_TEXT)


class FunctionTool:
    """LiteRT's wrapper shape on the phone: an object, not a dict.

    The test stub returns `{"name", "fn"}`, which is exactly why no test
    could see `TypeError: Object of type _FunctionTool is not JSON
    serializable` when the stream sent tools to a provider.
    """

    def __init__(self, fn):
        self.fn = fn


class ToolCallTransport:
    """A provider that asks for one tool call, then answers."""

    def __init__(self, name, args, reply=TOOL_TEXT):
        self.name = name
        self.args = args
        self.reply = reply
        self.calls = []

    def __call__(self, url, body, headers, timeout=30):
        self.calls.append((url, body, headers))
        if len(self.calls) == 1:
            return {"choices": [{"message": {
                "content": "",
                "tool_calls": [{"id": "call_1", "type": "function",
                                "function": {"name": self.name,
                                             "arguments": json.dumps(self.args)}}],
            }}], "usage": {"prompt_tokens": 4, "completion_tokens": 2}}
        return {"choices": [{"message": {"content": self.reply}}],
                "usage": {"prompt_tokens": 6, "completion_tokens": 3}}


class TestStreamCloudTools(_StreamCase):
    """The Web UI's tool path: schema → provider → executed → replayed as SSE.

    Same bug as the `handle()` tests, one layer up — this is the endpoint the
    page actually calls.
    """

    def test_tool_call_runs_and_the_answer_comes_from_the_cloud(self):
        from tools import ALL_TOOLS

        self.configure(mode="auto")
        old_tools = self.agent_mode._tools
        self.addCleanup(setattr, self.agent_mode, "_tools", old_tools)
        self.agent_mode._tools = [FunctionTool(fn) for fn in ALL_TOOLS]

        tmp = Path(tempfile.mkdtemp(prefix="stream-tools-"))
        (tmp / "seen-by-the-tool.txt").write_text("hello", "utf-8")
        # Release 16: the file tools are jailed to config.PROJECT_ROOT
        # when no project is active — point the root resolver at the tmp
        # tree so the tool really lists it (read per call by tools/file.py).
        saved_root = config.PROJECT_ROOT
        config.PROJECT_ROOT = tmp
        self.addCleanup(setattr, config, "PROJECT_ROOT", saved_root)
        tr = ToolCallTransport("list_files", {"directory": str(tmp)})
        self.agent_mode._backend._http = tr

        with verification_off():
            fr = self.stream({"message": "list my files"})
        end = fr[-1]
        self.assertEqual(end["type"], "end")
        self.assertEqual(end["text"], TOOL_TEXT)
        self.assertEqual(end["last_backend"], "cloud")

        # The tool trace the UI paints came through the stream.
        kinds = [f["type"] for f in fr]
        self.assertIn("tool_call", kinds)
        self.assertIn("tool_result", kinds)
        results = [f for f in fr if f["type"] == "tool_result"]
        self.assertIn("seen-by-the-tool.txt", results[0]["result"])
        self.assertEqual(end["tool_trace"][0]["type"], "call")
        self.assertEqual(end["tool_trace"][1]["type"], "result")

        # The wire really carried JSON tools, and the result went back.
        # Two calls: the tool call, then the answer over its result. The
        # reviewer is off here — see verification_off above.
        self.assertEqual(len(tr.calls), 2)
        json.dumps(tr.calls[0][1], ensure_ascii=False)
        self.assertIn("list_files", json.dumps(tr.calls[0][1]))
        self.assertIn("seen-by-the-tool.txt",
                      json.dumps(tr.calls[1][1]["messages"]))

        # Cloud only: the local conversation was never built.
        self.assertIsNone(self.agent_mode._conv)
        j = client.get("/api/agent/status").get_json()
        self.assertEqual(j["last_backend"], "cloud")


@contextlib.contextmanager
def small_budget(n):
    """Pin the tool-loop iteration cap for one test (Release 15).

    The loops read config.agent_max_iter() per turn, which reads this
    module attribute — the same override test_limits.py pins.
    """
    saved = config.AGENT_MAX_ITER_DEFAULT
    config.AGENT_MAX_ITER_DEFAULT = n
    try:
        yield
    finally:
        config.AGENT_MAX_ITER_DEFAULT = saved


class AlwaysToolTransport:
    """A provider that never stops asking for the same tool (Release 15).

    Every reply carries one list_files call; the directory rotates so the
    dedup guard lets every round really execute. The message content
    carries PARTIAL so the round text is observable too.
    """

    PARTIAL = "PARTIAL_WORDS"

    def __init__(self, dirs):
        self.dirs = list(dirs)
        self.calls = []

    def __call__(self, url, body, headers, timeout=30):
        self.calls.append((url, body, headers))
        d = self.dirs[min(len(self.calls) - 1, len(self.dirs) - 1)]
        return {"choices": [{"message": {
            "content": self.PARTIAL,
            "tool_calls": [{"id": "call_%d" % len(self.calls),
                            "type": "function",
                            "function": {"name": "list_files",
                                         "arguments": json.dumps(
                                             {"directory": d})}}],
        }}], "usage": {"prompt_tokens": 4, "completion_tokens": 2}}


class TestStreamBudgetNotice(_StreamCase):
    """Release 15: a streamed turn the step budget cuts short must end
    with the Burmese budget notice — not the last round's bare text —
    and the generated notice must not go through the verify pass.

    This endpoint's cloud attempt is the non-streaming _handle_cloud
    (v1 replays the finished turn as SSE frames and sanitises the text
    on the way out — the same filter_scripts allowlist that strips the
    🔒/☁️ leads of the older generated replies also strips the ⏱️ here,
    so the assertions target the Burmese body). The streaming generator
    itself is covered directly in tests/test_streaming.py.
    """

    def test_budget_exhaustion_ends_the_stream_with_the_notice(self):
        from tools import ALL_TOOLS

        self.configure(mode="auto")
        old_tools = self.agent_mode._tools
        self.addCleanup(setattr, self.agent_mode, "_tools", old_tools)
        self.agent_mode._tools = [FunctionTool(fn) for fn in ALL_TOOLS]

        tmp = Path(tempfile.mkdtemp(prefix="stream-budget-"))
        dirs = []
        for i in range(8):
            d = tmp / ("d%d" % i)
            d.mkdir()
            (d / "seen.txt").write_text("x", "utf-8")
            dirs.append(str(d))
        tr = AlwaysToolTransport(dirs)
        self.agent_mode._backend._http = tr

        with small_budget(3):
            fr = self.stream({"message": "list my files"})
        end = fr[-1]
        self.assertEqual(end["type"], "end")
        self.assertEqual(end["last_backend"], "cloud")
        text = end["text"]
        self.assertTrue(text, "an exhausted stream ended with empty text")
        self.assertIn("အဆင့် (step) ကန့်သတ်", text)
        self.assertIn("list_files", text)
        self.assertIn("ဆက်လုပ်ပါ", text)
        self.assertIn(AlwaysToolTransport.PARTIAL, text)
        # First pump plus one refresh per round = 4 provider calls. The
        # reviewer would be a fifth — the notice is generated text, not a
        # model answer, so verification stays off for it even though it
        # is enabled globally.
        self.assertEqual(len(tr.calls), 4,
                         "verify ran on a budget notice (or the loop "
                         "consumed a different number of steps)")
        # The notice is the committed assistant turn, as in handle().
        self.assertEqual(self.agent_mode._messages[-1]["role"], "assistant")
        self.assertEqual(self.agent_mode._messages[-1]["text"], text)


if __name__ == "__main__":
    unittest.main()
