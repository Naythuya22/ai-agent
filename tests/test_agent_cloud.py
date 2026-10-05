"""Cloud Only must reach the provider — and the local model must not answer.

These are the Agent Mode integration tests for the hybrid backend: they run
the real `AgentMode.handle()` with the local model replaced by a stub that
records whether it was touched, and with a fake transport standing in for the
provider. Nothing here opens a socket or loads weights.

The bug this file exists for: a Cloud Only turn used to fall through to the
local Gemma model, because the stored transcript never reached the cloud API
in a shape it accepts (422 `missing field content`) and because a mode change
was not re-read after the backend had been built.

Run:  sh tests/run.sh -p "test_agent_cloud.py"
"""
from __future__ import annotations

import contextlib
import json
import tempfile
import unittest
from pathlib import Path

from _harness import bootstrap

bootstrap()

import config  # noqa: E402
from core import http_llm  # noqa: E402
from core.agent_backend import AgentBackend, CloudError  # noqa: E402
from core.key_manager import KeyManager  # noqa: E402
from core.http_llm import HTTPLLM  # noqa: E402
from modes.agent import AgentMode  # noqa: E402
from tools import ALL_TOOLS  # noqa: E402

ALPHA = "sk-cloud-only-aaaaaaaaaaaaaaaa1111"
CLOUD_TEXT = "CLOUD_REPLY_OK"
LOCAL_TEXT = "LOCAL_MODEL_ANSWER"
HTTP_TEXT = "HTTP_TRANSPORT_ANSWER"


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


class RecordingTransport:
    """Stands in for a provider: answers, or fails the way a provider does."""

    def __init__(self, reply=CLOUD_TEXT, fail=None):
        self.reply = reply
        self.fail = fail          # a CloudError to raise, or None
        self.calls = []           # (url, body, headers) per request

    def __call__(self, url, body, headers, timeout=30):
        self.calls.append((url, body, headers))
        if self.fail is not None:
            raise self.fail
        return {"choices": [{"message": {"content": self.reply}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2}}

    @property
    def messages(self):
        return (self.calls[-1][1] or {}).get("messages") if self.calls else None


class FakeConversation:
    def __init__(self, log):
        self.log = log

    def send_message(self, msg):
        self.log.append(("message", msg))
        return {"content": [{"text": LOCAL_TEXT}]}


class FakeEngine:
    def __init__(self, log):
        self.log = log

    def create_conversation(self, **kw):
        self.log.append(("create_conversation", kw))
        return FakeConversation(self.log)


class FakeLocalLLM:
    """The local model — records every touch so a cloud turn can be proven
    not to have used it."""

    def __init__(self):
        self.log = []
        self._engine = None

    @property
    def loaded(self):
        return any(k == "load" for k, _ in self.log)

    def load(self):
        self.log.append(("load", None))
        self._engine = FakeEngine(self.log)


class FakeHTTPLLM(HTTPLLM):
    """An HTTPLLM that never talks to anything by itself."""

    def __init__(self, server_url="http://fake-local-server"):
        self.config = None
        self.server_url = server_url


class AgentCloudBase(unittest.TestCase):
    """Fresh keys, a fake local model and a fake-backed backend.

    Split from `AgentCloudCase` so the tool-loop tests can reuse the rig
    without re-running every mode test a second time.
    """
    def setUp(self):
        self.km = KeyManager()
        if self.km.path.exists():
            self.km.path.unlink()
        self.km.add_key("openai", ALPHA, "alpha")
        cfg = self.km.load()
        cfg["prefer_cloud"] = True
        cfg["cost_ack"] = True
        self.km.save()

        self.llm = FakeLocalLLM()
        self.mode = AgentMode(self.llm, None)
        self.mode._store = None        # never touch a real session file
        self.mode._tools = []
        self.mode._messages = []

    def transport(self, reply=CLOUD_TEXT, fail=None):
        return RecordingTransport(reply=reply, fail=fail)

    def prep(self, mode="cloud", reply=CLOUD_TEXT, fail=None, http=None):
        """Pin the backend mode and hand AgentMode a fake-backed backend."""
        self.km.set_mode(mode)
        self.km.save()
        self.transport_ = http or self.transport(reply=reply, fail=fail)
        be = AgentBackend(keys=self.km, net_state=lambda: "online",
                          http=self.transport_)
        self.mode._backend = be
        return be


class AgentCloudCase(AgentCloudBase):
    # ---------- Cloud Only ----------

    def test_cloud_only_answers_from_the_provider(self):
        self.prep(mode="cloud")
        out = self.mode.handle("hello")
        self.assertEqual(out, CLOUD_TEXT)
        self.assertIsNone(self.mode._conv, "the local conversation was created")
        self.assertFalse(self.llm.loaded, "the local model was loaded")

    def test_cloud_only_message_is_sent_as_content(self):
        """The 422 regression: `text` must become `content` on the wire."""
        self.prep(mode="cloud")
        self.mode.handle("hello")
        msgs = self.transport_.messages
        self.assertTrue(msgs, "no request reached the provider")
        for m in msgs:
            self.assertIn("content", m, "provider saw %r" % (m,))
            self.assertNotIn("text", m, "raw transcript leaked to the API")

    def test_cloud_only_says_so_when_the_provider_fails(self):
        self.prep(mode="cloud",
                  fail=CloudError("bad_response", "missing field content",
                                  status=422))
        out = self.mode.handle("hello")
        self.assertTrue(out.startswith("☁️ Cloud Only"), out[:80])
        self.assertIn("422", out, "the real provider error is hidden")
        self.assertIsNone(self.mode._conv)
        self.assertFalse(self.llm.loaded,
                         "Cloud Only answered from the local model")

    def test_cloud_only_refuses_local_when_the_backend_cannot_start(self):
        self.km.set_mode("cloud")
        self.km.save()
        self.mode._backend_obj = lambda: None     # backend import/construction died
        out = self.mode.handle("hello")
        self.assertTrue(out.startswith("☁️ Cloud Only"), out[:80])
        self.assertIsNone(self.mode._conv)
        self.assertFalse(self.llm.loaded)

    def test_cloud_only_never_calls_the_local_transport_in_auto(self):
        """Auto + provider failure still falls back (the rule next to Cloud Only)."""
        self.prep(mode="auto",
                  fail=CloudError("server", "boom", status=500))
        out = self.mode.handle("hello")
        self.assertEqual(out, LOCAL_TEXT, "auto mode lost its local fallback")
        self.assertTrue(self.llm.loaded)
        self.assertIsNotNone(self.mode._conv)

    def test_auto_mode_uses_the_cloud_when_everything_is_ready(self):
        self.prep(mode="auto")
        out = self.mode.handle("hello")
        self.assertEqual(out, CLOUD_TEXT)
        self.assertFalse(self.llm.loaded)

    def test_local_only_never_reaches_the_provider(self):
        self.prep(mode="local")
        out = self.mode.handle("hello")
        self.assertEqual(out, LOCAL_TEXT)
        self.assertEqual(self.transport_.calls, [],
                         "Local Only called a provider")

    # ---------- a mode change must be seen without a restart ----------

    def test_settings_change_is_seen_on_the_next_turn(self):
        """The bug: the backend is built once, so a later Settings write used
        to be invisible and the turn stayed on the local model."""
        be = self.prep(mode="auto",
                       fail=CloudError("server", "boom", status=500))
        self.assertEqual(self.mode.handle("one"), LOCAL_TEXT)

        self.km.set_mode("cloud")            # Settings → Cloud Only
        self.km.save()
        be._http = self.transport()           # and the provider is reachable again
        self.llm.log = []                     # turn 1 legitimately used local
        self.mode._conv = None

        out = self.mode.handle("two")
        self.assertEqual(out, CLOUD_TEXT,
                         "mode change ignored — still answering locally")
        self.assertFalse(self.llm.loaded, "turn 2 still went to the local model")
        self.assertIsNone(self.mode._conv, "turn 2 rebuilt the local conversation")

    def test_backend_mode_is_reread_even_when_cached(self):
        self.prep(mode="cloud")
        self.assertEqual(self.mode._mode(), "cloud")
        self.km.set_mode("local")
        self.km.save()
        self.assertEqual(self.mode._mode(), "local",
                         "a cached backend kept the old backend_mode")

    # ---------- HTTP transport ----------

    def test_auto_mode_keeps_using_the_http_transport(self):
        self.prep(mode="auto")
        posted = []

        def fake_post(url, payload, timeout):
            posted.append(url)
            return {"reply": HTTP_TEXT}

        old = http_llm._post
        http_llm._post = fake_post
        try:
            self.mode.llm = FakeHTTPLLM()
            out = self.mode.handle("hello")
        finally:
            http_llm._post = old
        self.assertEqual(out, HTTP_TEXT)
        self.assertEqual(posted, ["http://fake-local-server/api/agent"])
        self.assertEqual(self.transport_.calls, [],
                         "auto mode went to the provider despite HTTP mode")

    def test_cloud_only_wins_over_the_http_transport(self):
        self.prep(mode="cloud")
        posted = []

        def fake_post(url, payload, timeout):
            posted.append(url)
            return {"reply": HTTP_TEXT}

        old = http_llm._post
        http_llm._post = fake_post
        try:
            self.mode.llm = FakeHTTPLLM()
            out = self.mode.handle("hello")
        finally:
            http_llm._post = old
        self.assertEqual(out, CLOUD_TEXT)
        self.assertEqual(posted, [], "Cloud Only used the local HTTP server")
        self.assertFalse(self.llm.loaded)


class FunctionTool:
    """LiteRT's wrapper shape on the phone: an object, not a dict.

    The test stub's `tool_from_function` hands back `{"name", "fn"}`, which
    is why no test could see `TypeError: Object of type _FunctionTool is not
    JSON serializable`. Every test in this class uses the phone's shape.
    """

    def __init__(self, fn):
        self.fn = fn


class ToolCallTransport:
    """A provider that asks for one tool call, then answers."""

    def __init__(self, name, args, reply="TOOL_TURN_DONE"):
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

    @property
    def messages(self):
        return (self.calls[-1][1] or {}).get("messages") if self.calls else None


class CloudToolLoopCase(AgentCloudBase):
    """The cloud half of the tool loop: schema → provider → call → result.

    This is the path that died with `TypeError: Object of type _FunctionTool
    is not JSON serializable` before the request was ever built — the whole
    reason `tool_schema()` exists.
    """

    def setUp(self):
        super().setUp()
        self.mode._tools = [FunctionTool(fn) for fn in ALL_TOOLS]
        self.tmp = Path(tempfile.mkdtemp(prefix="agent-tools-"))
        (self.tmp / "seen-by-the-tool.txt").write_text("hello", "utf-8")

    def transport_tool_call(self, name="list_files", reply="TOOL_TURN_DONE"):
        return ToolCallTransport(name, {"directory": str(self.tmp)}, reply)

    def prep_tools(self, mode="cloud", reply="TOOL_TURN_DONE"):
        tr = self.transport_tool_call(reply=reply)
        be = self.prep(mode=mode, http=tr)
        return tr, be

    def test_cloud_turn_runs_the_tool_and_then_answers(self):
        tr, _ = self.prep_tools()
        with verification_off():
            out = self.mode.handle("list my files")
        self.assertEqual(out, "TOOL_TURN_DONE")

        # Two provider steps: the call, then the answer over the result.
        # (The reviewer is off here — see verification_off above.)
        self.assertEqual(len(tr.calls), 2, "the tool loop did not finish")
        first = tr.calls[0][1]
        json.dumps(first, ensure_ascii=False)      # the original TypeError
        names = [s["function"]["name"] for s in first["tools"]]
        self.assertIn("list_files", names)
        self.assertIn("list_files", json.dumps(first, ensure_ascii=False))

        # The tool really executed — not just described to the provider.
        results = [m for m in self.mode._messages
                   if m.get("role") == "tool_result"]
        self.assertTrue(results, "no tool result was recorded")
        self.assertIn("seen-by-the-tool.txt", results[-1]["text"])

        # And the result went back to the provider as a tool message.
        second = tr.calls[1][1]["messages"]
        self.assertIn("tool", [m.get("role") for m in second])
        self.assertIn("seen-by-the-tool.txt", json.dumps(second, ensure_ascii=False))

        # Cloud Only: the local model was never touched.
        self.assertIsNone(self.mode._conv)
        self.assertFalse(self.llm.loaded)
        self.assertEqual(self.mode._messages[-1]["role"], "assistant")
        self.assertEqual(self.mode._messages[-1]["text"], "TOOL_TURN_DONE")

    def test_tool_schemas_survive_auto_mode_too(self):
        tr, _ = self.prep_tools(mode="auto", reply="AUTO_TOOL_DONE")
        out = self.mode.handle("list my files")
        self.assertEqual(out, "AUTO_TOOL_DONE")
        json.dumps(tr.calls[0][1], ensure_ascii=False)
        self.assertEqual(
            sorted(s["function"]["name"] for s in tr.calls[0][1]["tools"]),
            sorted(fn.__name__ for fn in ALL_TOOLS),
            "a tool was lost between the mode and the wire")

    def test_unknown_tool_is_reported_not_raised(self):
        """A provider asking for a tool nobody registered must not kill the turn."""
        tr = ToolCallTransport("no_such_tool", {})
        self.prep(mode="cloud", http=tr)
        out = self.mode.handle("do something")
        self.assertEqual(out, "TOOL_TURN_DONE")
        results = [m for m in self.mode._messages
                   if m.get("role") == "tool_result"]
        self.assertTrue(results)
        self.assertIn("unknown tool", results[-1]["text"])


if __name__ == "__main__":
    unittest.main()
