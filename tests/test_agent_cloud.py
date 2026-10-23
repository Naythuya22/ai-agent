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
        # Release 24: the cause is classified Burmese now — the raw
        # provider status/body must NOT reach the reply (before R24
        # this test pinned the raw "422: missing field content" text).
        self.assertIn("Provider ဘက်က ပြဿနာ ရှိနေတယ်", out,
                      "the classified cause is missing")
        self.assertNotIn("422", out, "raw provider status leaked")
        self.assertNotIn("missing field content", out,
                         "raw provider body leaked")
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
        # Release 16: the file tools are jailed to config.PROJECT_ROOT
        # when no project is active — point the root resolver at this
        # case's tmp tree so the tool really lists it (tools/file.py
        # reads the root per call).
        self._saved_root = config.PROJECT_ROOT
        config.PROJECT_ROOT = self.tmp
        self.addCleanup(setattr, config, "PROJECT_ROOT", self._saved_root)

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
    carries PARTIAL so the step's partial text is observable too.
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


class CloudBudgetNoticeCase(AgentCloudBase):
    """Release 15: a cloud turn the step budget cuts short must say so —
    what the limit was, what work did get done, and how to continue —
    instead of returning the last step's bare (usually empty) text."""

    def setUp(self):
        super().setUp()
        self.mode._tools = [FunctionTool(fn) for fn in ALL_TOOLS]
        self.tmp = Path(tempfile.mkdtemp(prefix="agent-budget-"))
        self.dirs = []
        for i in range(8):
            d = self.tmp / ("d%d" % i)
            d.mkdir()
            (d / "seen.txt").write_text("x", "utf-8")
            self.dirs.append(str(d))

    def test_exhausted_cloud_turn_answers_with_the_budget_notice(self):
        tr = AlwaysToolTransport(self.dirs)
        self.prep(mode="cloud", http=tr)
        with small_budget(3):
            out = self.mode.handle("list my files")
        self.assertTrue(out, "an exhausted turn answered with empty text")
        self.assertIn("⏱️", out)
        self.assertIn("3", out)
        self.assertIn("list_files", out)
        self.assertIn("ဆက်လုပ်ပါ", out)
        # The model's partial text rides along instead of being dropped.
        self.assertIn(AlwaysToolTransport.PARTIAL, out)
        # Four provider steps: the first plus one refresh per round. The
        # reviewer would be a fifth call — the notice is generated text,
        # not a model answer, so verification stays off for it even
        # though it is enabled globally.
        self.assertEqual(len(tr.calls), 4,
                         "verify ran on a budget notice (or the loop "
                         "consumed a different number of steps)")
        # Transcript shape: the notice is the committed assistant turn,
        # and the executed rounds are still in the transcript.
        self.assertEqual(self.mode._messages[-1]["role"], "assistant")
        self.assertEqual(self.mode._messages[-1]["text"], out)
        results = [m for m in self.mode._messages
                   if m.get("role") == "tool_result"]
        self.assertEqual(len(results), 3, "executed rounds were lost")

    def test_cloud_turn_that_finishes_inside_the_budget_is_unchanged(self):
        tr = ToolCallTransport("list_files", {"directory": str(self.tmp)},
                               reply="FINISHED_WITHIN_BUDGET")
        self.prep(mode="cloud", http=tr)
        with small_budget(3), verification_off():
            out = self.mode.handle("list my files")
        self.assertEqual(out, "FINISHED_WITHIN_BUDGET")
        self.assertNotIn("⏱️", out)


def _tool_response(directory):
    return {"content": [{"text": ""}],
            "tool_calls": [{"function": {
                "name": "list_files",
                "arguments": {"directory": directory}}}]}


class ScriptedConversation:
    """Local-model stub replaying a fixed list of responses.

    The last response repeats if the loop asks again — an under-scripted
    test then fails on its assertions, not on an IndexError in the loop.
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.sent = 0

    def send_message(self, msg):
        r = self.responses[min(self.sent, len(self.responses) - 1)]
        self.sent += 1
        return r


class ScriptedEngine:
    def __init__(self, conv):
        self._conv = conv

    def create_conversation(self, **kw):
        return self._conv


class ScriptedLLM:
    """FakeLocalLLM's surface — load() plus _engine — with a script."""

    def __init__(self, conv):
        self._conv = conv
        self._engine = None
        self.loaded = False

    def load(self):
        self.loaded = True
        self._engine = ScriptedEngine(self._conv)


class LocalBudgetNoticeCase(AgentCloudBase):
    """Release 15: the local loop must not go silent at the budget edge."""

    def setUp(self):
        super().setUp()
        self.mode._tools = [FunctionTool(fn) for fn in ALL_TOOLS]
        self.tmp = Path(tempfile.mkdtemp(prefix="agent-budget-local-"))
        self.dirs = []
        for i in range(8):
            d = self.tmp / ("d%d" % i)
            d.mkdir()
            self.dirs.append(str(d))
        self.prep(mode="local")

    def drive(self, responses):
        self.mode.llm = ScriptedLLM(ScriptedConversation(responses))
        self.mode._conv = None
        return self.mode.handle("list my files")

    def test_exhausted_local_turn_answers_with_the_budget_notice(self):
        with small_budget(3):
            out = self.drive([_tool_response(d) for d in self.dirs])
        self.assertTrue(out, "an exhausted local turn answered empty")
        self.assertIn("⏱️", out)
        self.assertIn("list_files", out)
        self.assertIn("ဆက်လုပ်ပါ", out)
        self.assertEqual(self.mode._messages[-1]["role"], "assistant")
        self.assertEqual(self.mode._messages[-1]["text"], out)

    def test_local_answer_on_the_last_allowed_round_is_recovered(self):
        # Three tool rounds (the whole budget), then the answer: the loop
        # never gets a fourth look, but the text must not be lost — the
        # cloud loop's step.text fallback has always recovered it there.
        responses = [_tool_response(d) for d in self.dirs[:3]]
        responses.append({"content": [{"text": "BOUNDARY_DONE"}]})
        with small_budget(3):
            out = self.drive(responses)
        self.assertEqual(out, "BOUNDARY_DONE")
        self.assertNotIn("⏱️", out)

    def test_local_turn_that_finishes_inside_the_budget_is_unchanged(self):
        responses = [_tool_response(self.dirs[0]),
                     {"content": [{"text": "LOCAL_DONE"}]}]
        with small_budget(3):
            out = self.drive(responses)
        self.assertEqual(out, "LOCAL_DONE")
        self.assertNotIn("⏱️", out)


if __name__ == "__main__":
    unittest.main()
