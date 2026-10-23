"""Release 24 — a failed turn must keep its work and speak plainly.

Two defects from the live v1.21.0 evaluation, reproduced here with
fake backends in the exact shape they took live:

1. A file-write turn executed write_file, then the follow-up model
   step died (429 park → 402) and the turn collapsed into the
   exhaustion message: the file was on disk, but the reply said the
   turn could not be answered, the tool_trace was empty, the SAVED
   transcript held only user + failure text, and the checkpoint was
   gone. Now a tool-using turn that fails mid-work is preserved:
   transcript keeps the tool entries, the reply names the completed
   work, and the checkpoint clears only because the transcript (and
   the reply) carry the record.

2. The exhaustion reply embedded the provider's raw error body —
   JSON fragments, masked key material, text cut mid-word — and never
   mentioned a mere 60-second park. Now every failure reply is
   composed from the classified records (failure_reason vocabulary +
   key state): per-provider Burmese cause + one action hint, a park
   reads as temporary with its wait, and no raw internals appear.

Run:  sh tests/run.sh -p "test_failure_replies.py"
"""
from __future__ import annotations

import contextlib
import json
import re
import tempfile
import unittest
from pathlib import Path

from _harness import bootstrap

bootstrap()

import config  # noqa: E402
from core.agent_backend import AgentBackend, CloudError  # noqa: E402
from core.checkpoint_store import CheckpointStore  # noqa: E402
from core.key_manager import KeyManager  # noqa: E402
from modes import agent as agent_mod  # noqa: E402
from modes.agent import AgentMode  # noqa: E402
from sessions import SessionStore  # noqa: E402
from tools import ALL_TOOLS  # noqa: E402

KEY_A = "sk-openai-fake-aaaaaaaa1111"
KEY_B = "sk-openrouter-fake-bbbb2222"
CLOUD_TEXT = "CLOUD_REPLY_OK"
NOTE_TEXT = "စမ်းသပ်ချက် နံပါတ် ၃၈၅"
RAW_402 = ('{"error":{"message":"This request requires more credits, '
           'or fewer max_tokens. You requested up to 8192 tokens, but '
           'can only afford 241."}}')
RAW_429 = '{"error":{"message":"quota exceeded for this key, slow down"}}'
RAW_401 = ('{"error":{"message":"Incorrect API key provided: '
           'sk-openai-fake-aaaaaaaa1111"}}')


@contextlib.contextmanager
def verification_off():
    saved = config.AGENT_VERIFY_ENABLED
    config.AGENT_VERIFY_ENABLED = False
    try:
        yield
    finally:
        config.AGENT_VERIFY_ENABLED = saved


class FunctionTool:
    """LiteRT's wrapper shape on the phone (see test_agent_cloud.py)."""

    def __init__(self, fn):
        self.fn = fn


class FakeEngine:
    def __init__(self, conv):
        self._conv = conv

    def create_conversation(self, **kw):
        return self._conv


class FakeLocalLLM:
    """Local model stub — records whether the engine was ever loaded."""

    def __init__(self, engine=None):
        self.log = []
        self._engine = None
        self._given_engine = engine

    @property
    def loaded(self):
        return any(k == "load" for k, _ in self.log)

    def load(self):
        self.log.append(("load", None))
        self._engine = self._given_engine or FakeEngine(None)

    def ask(self, prompt):
        return ""


def tool_call_response(name, args, call_id="call_1"):
    return {"choices": [{"message": {
        "content": "",
        "tool_calls": [{"id": call_id, "type": "function",
                        "function": {"name": name,
                                     "arguments": json.dumps(args)}}],
    }}], "usage": {"prompt_tokens": 4, "completion_tokens": 2}}


class TwoProviderTransport:
    """The live repro: provider A answers with a tool call, then its
    key is parked by a 429; provider B is out of credit (402)."""

    def __init__(self, note_path):
        self.note_path = note_path
        self.calls = []

    def __call__(self, url, body, headers, timeout=30):
        self.calls.append(url)
        if "openrouter" in url:
            raise CloudError("bad_response", RAW_402, status=402)
        n = sum(1 for u in self.calls if "openrouter" not in u)
        if n == 1:
            return tool_call_response(
                "write_file",
                {"path": self.note_path, "content": NOTE_TEXT})
        raise CloudError("rate_limit", RAW_429, status=429, retry_after=60)


class AlwaysFailTransport:
    """Every request fails with the same CloudError."""

    def __init__(self, err):
        self.err = err
        self.calls = []

    def __call__(self, url, body, headers, timeout=30):
        self.calls.append(url)
        raise self.err


class ToolThenAnswerTransport:
    """One tool round, then a plain answer (a turn that completes)."""

    def __init__(self, directory):
        self.directory = directory
        self.calls = []

    def __call__(self, url, body, headers, timeout=30):
        self.calls.append((url, body, headers))
        if len(self.calls) == 1:
            return tool_call_response(
                "list_files", {"directory": self.directory})
        return {"choices": [{"message": {"content": CLOUD_TEXT}}],
                "usage": {"prompt_tokens": 6, "completion_tokens": 3}}


class DyingConversation:
    """Local engine stub: one tool round, then send_message raises."""

    def __init__(self, directory):
        self.directory = directory
        self.calls = 0

    def send_message(self, msg):
        self.calls += 1
        if self.calls == 1:
            return {"content": [{"text": ""}],
                    "tool_calls": [{"function": {
                        "name": "list_files",
                        "arguments": {"directory": self.directory}}}]}
        raise RuntimeError("simulated engine death mid-turn")


class DyingStreamBackend:
    """step_stream stub: one tool round, then the stream dies."""

    def __init__(self, directory):
        self.directory = directory
        self.rounds = 0

    def refresh(self):
        pass

    def should_use_cloud(self):
        return True

    def why_not_cloud(self):
        return ""

    def step_stream(self, system=None, messages=None, tools=None,
                    max_tokens=None):
        self.rounds += 1
        if self.rounds == 1:
            yield "done", {
                "text": "",
                "tool_calls": [{"function": {
                    "name": "list_files",
                    "arguments": {"directory": self.directory}}}],
                "usage": {}, "model": "stub-model", "provider": "openai",
                "reasoning": "",
            }
            return
        raise RuntimeError("simulated stream death mid-turn")


class FailureReplyBase(unittest.TestCase):
    """Real session store, tmp jail root, fresh keys — the
    CheckpointBase rig, plus a two-provider key set."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="failure-reply-"))
        self.dir_a = self.tmp / "a"
        self.dir_a.mkdir()
        (self.dir_a / "alpha.txt").write_text("x", "utf-8")
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
        self.km.add_key("openai", KEY_A, "alpha")
        cfg = self.km.load()
        cfg["prefer_cloud"] = True
        cfg["cost_ack"] = True
        self.km.save()
        self.llm = FakeLocalLLM()

    def new_mode(self, llm=None):
        mode = AgentMode(llm or self.llm, None)
        mode._tools = [FunctionTool(fn) for fn in ALL_TOOLS]
        return mode

    def prep(self, mode, transport, backend_mode="cloud"):
        self.km.set_mode(backend_mode)
        self.km.save()
        be = AgentBackend(keys=self.km, net_state=lambda: "online",
                          http=transport)
        mode._backend = be
        return be

    def filed(self, sid):
        return CheckpointStore.load().get(sid)

    @staticmethod
    def roles(msgs):
        return [m.get("role") for m in msgs]

    def saved(self, sid):
        return self.store.get_session(sid)["messages"]


class MidTurnExhaustionCase(FailureReplyBase):
    def test_tool_work_survives_mid_turn_exhaustion(self):
        """(a) The live repro, Cloud Only: the write lands, the next
        step dies — transcript, reply and disk must all show the work."""
        self.km.add_key("openrouter", KEY_B, "beta")
        cfg = self.km.load()
        cfg["fallback"] = ["openai", "openrouter"]
        self.km.save()
        mode = self.new_mode()
        sid = mode.session_id
        note_path = str(self.tmp / "note.txt")
        self.prep(mode, TwoProviderTransport(note_path), "cloud")
        with verification_off():
            out = mode.handle("write the note for me")

        # The side effect really happened, with the exact content.
        note = self.tmp / "note.txt"
        self.assertTrue(note.exists(), "write_file never landed")
        self.assertEqual(note.read_text("utf-8"), NOTE_TEXT)

        # The reply names the completed work and says it is recorded.
        self.assertIn("ဒီနေရာအထိ ပြီးပြီးသားပါ", out)
        self.assertIn("write_file (", out)
        self.assertIn("note.txt) — အောင်မြင်", out)
        self.assertIn("မှတ်တမ်းမှာ သိမ်းထားပါတယ်", out)
        # Classified causes for both providers, with the park's wait.
        self.assertIn("OpenAI: Limit ပြည့်သွားပြီ (429)", out)
        self.assertIn("(~60 စက္ကန့်)", out)
        self.assertIn("OpenRouter: အကောင့်မှာ ဘေ/credit မရှိတော့ဘူး", out)
        # No raw internals: no provider body, no JSON, no key material.
        self.assertNotIn("requires more credits", out)
        self.assertNotIn("quota exceeded", out)
        self.assertNotIn('{"error"', out)
        self.assertNotIn("8192", out)
        self.assertNotIn("sk-openai", out)
        self.assertNotIn("sk-openrouter", out)
        self.assertNotIn("1111", out)
        self.assertNotIn("2222", out)

        # The SAVED transcript carries the tool entries — the record
        # the routes build tool_trace from — closed by the reply.
        saved = self.saved(sid)
        self.assertEqual(self.roles(saved),
                         ["user", "tool_call", "tool_result", "assistant"])
        self.assertEqual(saved[1]["tool"], "write_file")
        self.assertEqual(saved[-1]["text"], out)
        # Checkpoint invariant: cleared, because the transcript now
        # carries the work and the reply says it is recorded.
        self.assertIsNone(self.filed(sid))

    def test_auto_mode_stops_with_summary_instead_of_rerunning_locally(self):
        """(a-auto) Same failure in auto mode: no silent local re-run
        of a tool-using turn (its side effects would repeat)."""
        self.km.add_key("openrouter", KEY_B, "beta")
        cfg = self.km.load()
        cfg["fallback"] = ["openai", "openrouter"]
        self.km.save()
        mode = self.new_mode()
        sid = mode.session_id
        note_path = str(self.tmp / "note.txt")
        self.prep(mode, TwoProviderTransport(note_path), "auto")
        with verification_off():
            out = mode.handle("write the note for me")
        self.assertIn("write_file (", out)
        self.assertIn("note.txt) — အောင်မြင်", out)
        self.assertTrue((self.tmp / "note.txt").exists())
        self.assertFalse(self.llm.loaded,
                         "auto mode re-ran the turn on the local model")
        self.assertIsNone(mode._conv)
        self.assertEqual(self.roles(self.saved(sid)),
                         ["user", "tool_call", "tool_result", "assistant"])

    def test_locked_failure_after_tools_preserves_them(self):
        """(a-locked) Shared path, reported in the R24 notes: a locked
        turn that dies after tools ran keeps the R9 text and gains the
        completed-work summary; the transcript keeps the tool entries
        and gains no duplicate user message."""
        cfg = self.km.load()
        cfg["auto_switch"] = False
        cfg["pinned_provider"] = "openai"
        self.km.save()

        note_path = str(self.tmp / "note.txt")

        class PinTransport:
            def __init__(self):
                self.calls = 0

            def __call__(self, url, body, headers, timeout=30):
                self.calls += 1
                if self.calls == 1:
                    return tool_call_response(
                        "write_file",
                        {"path": note_path, "content": NOTE_TEXT})
                raise CloudError("bad_response", RAW_402, status=402)

        mode = self.new_mode()
        sid = mode.session_id
        self.prep(mode, PinTransport(), "auto")
        with verification_off():
            out = mode.handle("write the note for me")
        self.assertIn("🔒 Lock ချထားတဲ့", out)
        self.assertIn("write_file (", out)
        self.assertIn("note.txt) — အောင်မြင်", out)
        self.assertEqual(self.roles(self.saved(sid)),
                         ["user", "tool_call", "tool_result", "assistant"])
        self.assertIsNone(self.filed(sid))


class ShortenTargetCase(unittest.TestCase):
    """(h) The summary's 48-char slot must keep a long path's filename.

    Regression test for the R24 phone-only failure: the slot used to
    head-truncate every target, and Termux paths run past 60 chars
    (/data/data/com.termux/files/usr/tmp/...), so the cut landed
    mid-directory and the filename never reached the reply. On the
    VM's short tmp paths the same tests passed — which is why this
    is pinned directly, with a synthetic Termux-shaped path, instead
    of relying on the environment's TMPDIR length."""

    TERMUX_PATH = ("/data/data/com.termux/files/usr/tmp/"
                   "failure-reply-fibwqluk/note.txt")

    def test_long_path_keeps_its_tail(self):
        self.assertGreater(len(self.TERMUX_PATH), 48)
        out = AgentMode._shorten_target(self.TERMUX_PATH)
        self.assertEqual(len(out), 48)
        self.assertEqual(out, "..." + self.TERMUX_PATH[-45:])
        self.assertTrue(out.startswith("..."))
        self.assertTrue(out.endswith("note.txt"))

    def test_long_path_label_names_the_file(self):
        # End to end through _work_summary: the rendered label must
        # close on the filename, exactly what the phone reply lost.
        mode = AgentMode.__new__(AgentMode)
        entries = [
            {"role": "tool_call", "tool": "write_file",
             "args": {"path": self.TERMUX_PATH, "content": NOTE_TEXT}},
            {"role": "tool_result", "tool": "write_file",
             "text": "Wrote note.txt"},
        ]
        summary = mode._work_summary(entries)
        self.assertIn("write_file (...", summary)
        self.assertIn("note.txt) — အောင်မြင်", summary)
        self.assertNotIn("failure-repl)", summary)

    def test_short_path_unchanged(self):
        p = "/tmp/failure-reply-x/note.txt"
        self.assertLessEqual(len(p), 48)
        self.assertEqual(AgentMode._shorten_target(p), p)

    def test_exactly_48_chars_unchanged(self):
        p = "/" + "a" * 38 + "/note.txt"
        self.assertEqual(len(p), 48)
        self.assertEqual(AgentMode._shorten_target(p), p)

    def test_long_non_path_still_head_truncated(self):
        q = ("what is the weather going to be like in yangon "
             "tomorrow morning and the day after")
        self.assertGreater(len(q), 48)
        self.assertNotIn("/", q)
        out = AgentMode._shorten_target(q)
        self.assertEqual(out, q[:48])
        self.assertFalse(out.startswith("..."))


class ClassifiedReplyCase(FailureReplyBase):
    def test_402_reply_is_classified_without_raw_body(self):
        """(b-402) No tools: the Cloud Only reply names the classified
        cause + hint and nothing of the provider's raw body."""
        mode = self.new_mode()
        self.prep(mode, AlwaysFailTransport(
            CloudError("bad_response", RAW_402, status=402)), "cloud")
        out = mode.handle("hello")
        self.assertTrue(out.startswith("☁️ Cloud Only"), out[:80])
        self.assertIn("OpenAI: အကောင့်မှာ ဘေ/credit မရှိတော့ဘူး", out)
        self.assertIn("credit ဖြည့်ပါ", out)
        self.assertNotIn("requires more credits", out)
        self.assertNotIn("8192", out)
        self.assertNotIn('{"error"', out)
        self.assertNotIn("241", out)

    def test_auth_reply_is_classified_without_raw_body(self):
        """(b-auth) A 401 classifies as key_invalid; the provider's
        message (which quotes the key itself) never reaches the reply."""
        mode = self.new_mode()
        self.prep(mode, AlwaysFailTransport(
            CloudError("auth", RAW_401, status=401)), "cloud")
        out = mode.handle("hello")
        self.assertIn("OpenAI: API key က အလုပ်မလုပ်တော့ဘူး", out)
        self.assertIn("Settings မှာ key ကို ပြန်စစ်ပါ", out)
        self.assertNotIn("Incorrect API key", out)
        self.assertNotIn("sk-openai", out)
        self.assertNotIn("1111", out)

    def test_cooling_only_reply_reads_as_temporary(self):
        """(c) The only blocker is a 45s park: the reply must say
        temporary + the wait, not declare total exhaustion."""
        mode = self.new_mode()
        self.prep(mode, AlwaysFailTransport(
            CloudError("rate_limit", RAW_429, status=429, retry_after=45)),
            "cloud")
        out = mode.handle("hello")
        self.assertIn("Provider key တွေ အကုန် cooldown ထဲမှာ ရှိတယ်", out)
        self.assertIn("ခဏ စောင့်ပြီး ပြန်စမ်းပါ", out)
        self.assertIn("(~45 စက္ကန့်)", out)
        self.assertNotIn("အားလုံး မအောင်မြင်တော့ဘူး", out)
        self.assertNotIn("quota exceeded", out)

    def test_parked_key_reply_before_any_attempt(self):
        """(c2) Every key already parked when the turn starts: same
        temporary reply, and no provider request is even attempted."""
        entry = self.km.entries("openai")[0]
        self.km.mark_rate_limited("openai", entry["id"], 60)
        transport = AlwaysFailTransport(
            CloudError("server", "must not be called", status=500))
        mode = self.new_mode()
        self.prep(mode, transport, "cloud")
        out = mode.handle("hello")
        self.assertEqual(transport.calls, [],
                         "a provider was called despite the park")
        self.assertIn("Provider key တွေ အကုန် cooldown ထဲမှာ ရှိတယ်", out)
        self.assertIn("ခဏ စောင့်ပြီး ပြန်စမ်းပါ", out)
        self.assertRegex(out, r"\(~\d+ စက္ကန့်\)")
        self.assertNotIn("သုံးလို့ရတဲ့ API Key မရှိသေးဘူး", out)


class SuccessfulTurnCase(FailureReplyBase):
    def test_successful_turn_transcript_unchanged(self):
        """(d) A turn that completes keeps the exact pre-R24 shape:
        same roles, the model's own text, nothing of the failure
        machinery in the reply or the record."""
        mode = self.new_mode()
        sid = mode.session_id
        self.prep(mode, ToolThenAnswerTransport(str(self.dir_a)), "cloud")
        with verification_off():
            out = mode.handle("list my files")
        self.assertEqual(out, CLOUD_TEXT)
        self.assertNotIn("ပြီးပြီးသားပါ", out)
        saved = self.saved(sid)
        self.assertEqual(self.roles(saved),
                         ["user", "tool_call", "tool_result", "assistant"])
        self.assertEqual(saved, mode._messages)
        self.assertEqual(saved[-1]["text"], CLOUD_TEXT)
        self.assertIsNone(self.filed(sid))


class LocalDeathCase(FailureReplyBase):
    def test_local_engine_death_after_tools_preserves_turn(self):
        """(f) The on-device loop: engine dies after one tool round —
        handle() answers with the failure reply (no raise), and the
        saved transcript carries the tool work."""
        self.km.set_mode("local")
        self.km.save()
        conv = DyingConversation(str(self.dir_a))
        llm = FakeLocalLLM(engine=FakeEngine(conv))
        mode = self.new_mode(llm)
        # _system_prompt probes the backend — give it a harmless one.
        mode._backend = AgentBackend(keys=self.km,
                                     net_state=lambda: "offline",
                                     http=ToolThenAnswerTransport(
                                         str(self.dir_a)))
        sid = mode.session_id
        out = mode.handle("LOCAL_TASK")
        self.assertIn("Local model က ကြားဖြတ် ရပ်သွားလို့", out)
        self.assertIn("list_files", out)
        self.assertIn("အောင်မြင်", out)
        self.assertIn("မှတ်တမ်းမှာ သိမ်းထားပါတယ်", out)
        saved = self.saved(sid)
        self.assertEqual(self.roles(saved),
                         ["user", "tool_call", "tool_result", "assistant"])
        self.assertEqual(saved[-1]["text"], out)
        self.assertIsNone(self.filed(sid))


class StreamDeathCase(FailureReplyBase):
    def test_stream_death_after_tools_commits_the_turn(self):
        """(g) The v2 stream wrapper: the stream dies after a tool
        round — the turn is committed with the failure reply as its
        end frame instead of being rolled back for a re-run."""
        self.km.set_mode("cloud")
        self.km.save()
        mode = self.new_mode()
        sid = mode.session_id
        mode._backend = DyingStreamBackend(str(self.dir_a))
        # The route contract: begin_turn + user append come first.
        mode.begin_turn()
        mode._messages.append({"role": "user", "text": "STREAM_TASK"})
        frames = list(mode._handle_cloud_stream())
        kinds = [k for k, _ in frames]
        self.assertIn("tool", kinds, "the tool frame never streamed")
        self.assertEqual(kinds[-1], "end")
        text = frames[-1][1]
        self.assertIn("list_files", text)
        self.assertIn("ဒီနေရာအထိ ပြီးပြီးသားပါ", text)
        saved = self.saved(sid)
        self.assertEqual(self.roles(saved),
                         ["user", "tool_call", "tool_result", "assistant"])
        self.assertEqual(saved[-1]["text"], text)
        self.assertIsNone(self.filed(sid))


if __name__ == "__main__":
    unittest.main()
