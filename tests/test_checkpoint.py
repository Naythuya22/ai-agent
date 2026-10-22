"""Release 19 — a turn interrupted mid-work must resume, not restart.

Sessions persist only at turn END (_save_turn). Before R19, a server
that died mid-turn — Termux killed, a crash — lost the in-flight
turn entirely: the request, every tool result so far, and any record
that side-effecting tools had already run. Now each tool loop files
a checkpoint after every tool_result (core/checkpoint_store), a
completed turn deletes it, and the next load of the session restores
it with a Burmese interruption note appended.

The crash tests simulate process death the only faithful way: drive
the real loop until it raises out of the loop entirely (the way a
dead process never reaches _try_cloud's catch), drop the instance,
and build a FRESH AgentMode over the same stores — disk state is
all that may carry over.

Run:  sh tests/run.sh -p "test_checkpoint.py"
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
from core import checkpoint_store  # noqa: E402
from core.agent_backend import AgentBackend  # noqa: E402
from core.checkpoint_store import CheckpointStore  # noqa: E402
from core.key_manager import KeyManager  # noqa: E402
from modes import agent as agent_mod  # noqa: E402
from modes.agent import AgentMode  # noqa: E402
from sessions import SessionStore  # noqa: E402
from tools import ALL_TOOLS  # noqa: E402

ALPHA = "<redacted>"
CLOUD_TEXT = "CLOUD_REPLY_OK"
NOTE_MARK = "ကြားဖြတ်"          # "interrupted" — the note's fingerprint
CONTINUE = "ဆက်လုပ်ပါ"


@contextlib.contextmanager
def verification_off():
    """Keep the reviewer out of these turns (see test_agent_cloud.py)."""
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


class FakeLocalLLM:
    """Local model stub: an engine for the local loop, ask() unused here."""

    def __init__(self, engine=None):
        self.log = []
        self._engine = None
        self._given_engine = engine

    def load(self):
        self.log.append(("load", None))
        self._engine = self._given_engine or FakeEngine(self.log)

    def ask(self, prompt):
        return ""


class FakeConversation:
    def __init__(self, log):
        self.log = log

    def send_message(self, msg):
        self.log.append(("message", msg))
        return {"content": [{"text": "LOCAL_TEXT"}]}


class FakeEngine:
    def __init__(self, log):
        self.log = log

    def create_conversation(self, **kw):
        self.log.append(("create_conversation", kw))
        return FakeConversation(self.log)


def tool_call_response(name, args, call_id="call_1"):
    return {"choices": [{"message": {
        "content": "",
        "tool_calls": [{"id": call_id, "type": "function",
                        "function": {"name": name,
                                     "arguments": json.dumps(args)}}],
    }}], "usage": {"prompt_tokens": 4, "completion_tokens": 2}}


class CrashTransport:
    """Provider stub: one tool round per request, then it 'dies'.

    Each request asks for list_files on the next directory; the
    request after the last directory raises a plain RuntimeError —
    the shape a dead process leaves behind: _handle_cloud does not
    catch it, the in-memory transcript is simply gone.
    """

    def __init__(self, dirs):
        self.dirs = list(dirs)
        self.calls = []

    def __call__(self, url, body, headers, timeout=30):
        self.calls.append((url, body, headers))
        n = len(self.calls)
        if n > len(self.dirs):
            raise RuntimeError("simulated crash: process died mid-turn")
        return tool_call_response(
            "list_files", {"directory": self.dirs[n - 1]},
            call_id="call_%d" % n)


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


class CrashConversation:
    """Local engine stub: one tool round, then the process 'dies'."""

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
        raise RuntimeError("simulated crash: process died mid-turn")


class CrashEngine:
    def __init__(self, conv):
        self._conv = conv

    def create_conversation(self, **kw):
        return self._conv


class FakeStreamBackend:
    """step_stream stub for _try_cloud_stream: tool rounds, then death."""

    def __init__(self, dirs):
        self.dirs = list(dirs)
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
        n = self.rounds
        if n > len(self.dirs):
            raise RuntimeError("simulated crash: stream died mid-turn")
        yield "done", {
            "text": "",
            "tool_calls": [{"function": {
                "name": "list_files",
                "arguments": {"directory": self.dirs[n - 1]}}}],
            "usage": {}, "model": "stub-model", "provider": "stub",
            "reasoning": "",
        }


class CheckpointBase(unittest.TestCase):
    """A private session store injected into every AgentMode built
    here, a tmp jail root for the file tools, and a keyed cloud rig."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="checkpoint-"))
        self.dir_a = self.tmp / "a"
        self.dir_a.mkdir()
        (self.dir_a / "alpha.txt").write_text("x", "utf-8")
        self.dir_b = self.tmp / "b"
        self.dir_b.mkdir()
        (self.dir_b / "beta.txt").write_text("x", "utf-8")
        # Release 16: file tools are jailed to config.PROJECT_ROOT
        # when no project is active.
        self._saved_root = config.PROJECT_ROOT
        config.PROJECT_ROOT = self.tmp
        self.addCleanup(setattr, config, "PROJECT_ROOT", self._saved_root)
        # AgentMode builds SessionStore() itself — hand it this
        # suite's private store instance instead.
        self.store = SessionStore(self.tmp / "sessions.json")
        saved_factory = agent_mod.SessionStore
        agent_mod.SessionStore = lambda: self.store
        self.addCleanup(setattr, agent_mod, "SessionStore", saved_factory)
        # Keys for the cloud rig (same shape as test_agent_cloud).
        self.km = KeyManager()
        if self.km.path.exists():
            self.km.path.unlink()
        self.km.add_key("openai", ALPHA, "alpha")
        cfg = self.km.load()
        cfg["prefer_cloud"] = True
        cfg["cost_ack"] = True
        self.km.save()

    def new_mode(self, llm=None):
        mode = AgentMode(llm or FakeLocalLLM(), None)
        mode._tools = [FunctionTool(fn) for fn in ALL_TOOLS]
        return mode

    def prep_cloud(self, mode, transport):
        self.km.set_mode("cloud")
        self.km.save()
        be = AgentBackend(keys=self.km, net_state=lambda: "online",
                          http=transport)
        mode._backend = be
        return be

    def filed(self, sid):
        """The checkpoint currently on disk for sid (or None)."""
        return CheckpointStore.load().get(sid)

    @staticmethod
    def roles(msgs):
        return [m.get("role") for m in msgs]

    @staticmethod
    def notes(msgs):
        return [m for m in msgs if NOTE_MARK in (m.get("text") or "")]


class CloudCrashResumeCase(CheckpointBase):
    def test_crash_mid_turn_resumes_from_checkpoint(self):
        """(a) Two tool rounds land, the third step kills the
        'process'; a fresh AgentMode restores the partial turn with
        the interruption note, and the checkpoint is consumed."""
        mode = self.new_mode()
        sid = mode.session_id
        self.assertTrue(sid)
        be = self.prep_cloud(mode, CrashTransport(
            [str(self.dir_a), str(self.dir_b)]))
        with self.assertRaises(RuntimeError):
            mode._handle_cloud("TASK_TEXT", be)

        entry = self.filed(sid)
        self.assertIsNotNone(entry, "no checkpoint survived the crash")
        self.assertEqual(entry["user_text"], "TASK_TEXT")
        self.assertEqual(entry["steps"], 2)
        self.assertEqual(self.roles(entry["messages"]),
                         ["user", "tool_call", "tool_result",
                          "tool_call", "tool_result"])
        self.assertIn("alpha.txt", entry["messages"][2]["text"])
        self.assertIn("beta.txt", entry["messages"][4]["text"])
        # The session store itself never saw the turn — turn-end only.
        self.assertEqual(self.store.get_session(sid)["messages"], [])

        fresh = self.new_mode()
        self.assertEqual(fresh.session_id, sid)
        msgs = fresh._messages
        self.assertEqual(self.roles(msgs),
                         ["user", "tool_call", "tool_result",
                          "tool_call", "tool_result", "assistant"])
        self.assertEqual(msgs[0]["text"], "TASK_TEXT")
        note = msgs[-1]["text"]
        self.assertIn(NOTE_MARK, note)
        self.assertIn("2", note, "note does not name the step reached")
        self.assertIn(CONTINUE, note)
        # Consumed, and the restored transcript was persisted into
        # the session itself, so it is the durable state now.
        self.assertIsNone(self.filed(sid))
        self.assertEqual(self.store.get_session(sid)["messages"], msgs)
        # A third load restores nothing a second time.
        third = self.new_mode()
        self.assertEqual(third._messages, msgs)
        self.assertEqual(len(self.notes(third._messages)), 1)


class StreamCrashResumeCase(CheckpointBase):
    def test_stream_crash_resumes_from_checkpoint(self):
        """(a-stream) The streaming loop files the same checkpoints:
        two yielded tool rounds survive the generator dying."""
        mode = self.new_mode()
        sid = mode.session_id
        self.km.set_mode("cloud")
        self.km.save()
        mode._backend = FakeStreamBackend([str(self.dir_a),
                                           str(self.dir_b)])
        # The route contract: begin_turn + user append happen before
        # the generator is iterated.
        mode.begin_turn()
        mode._messages.append({"role": "user", "text": "STREAM_TASK"})
        with self.assertRaises(RuntimeError):
            for _frame in mode._try_cloud_stream():
                pass
        entry = self.filed(sid)
        self.assertIsNotNone(entry)
        self.assertEqual(entry["steps"], 2)

        fresh = self.new_mode()
        self.assertEqual(self.roles(fresh._messages),
                         ["user", "tool_call", "tool_result",
                          "tool_call", "tool_result", "assistant"])
        self.assertIn(NOTE_MARK, fresh._messages[-1]["text"])
        self.assertIsNone(self.filed(sid))


class LocalCrashResumeCase(CheckpointBase):
    def test_local_crash_resumes_from_checkpoint(self):
        """(a-local) The on-device loop checkpoints too: one executed
        tool round survives the engine 'dying' on the next call."""
        self.km.set_mode("local")
        self.km.save()
        conv = CrashConversation(str(self.dir_a))
        mode = self.new_mode(FakeLocalLLM(engine=CrashEngine(conv)))
        # _system_prompt probes the backend — give it a harmless one.
        mode._backend = AgentBackend(keys=self.km,
                                     net_state=lambda: "offline",
                                     http=ToolThenAnswerTransport(
                                         str(self.dir_a)))
        sid = mode.session_id
        with self.assertRaises(RuntimeError):
            mode.handle("LOCAL_TASK")
        entry = self.filed(sid)
        self.assertIsNotNone(entry)
        self.assertEqual(entry["steps"], 1)
        self.assertEqual(entry["user_text"], "LOCAL_TASK")

        fresh = self.new_mode()
        self.assertEqual(self.roles(fresh._messages),
                         ["user", "tool_call", "tool_result",
                          "assistant"])
        self.assertIn(NOTE_MARK, fresh._messages[-1]["text"])
        self.assertIsNone(self.filed(sid))


class CompletedTurnCase(CheckpointBase):
    def test_completed_turn_leaves_no_checkpoint(self):
        """(b) A turn that finishes is saved by _save_turn and its
        checkpoint is gone — it must never be 'resumed'."""
        mode = self.new_mode()
        sid = mode.session_id
        self.prep_cloud(mode, ToolThenAnswerTransport(str(self.dir_a)))
        with verification_off():
            out = mode.handle("list my files")
        self.assertEqual(out, CLOUD_TEXT)
        self.assertIsNone(self.filed(sid))
        saved = self.store.get_session(sid)
        self.assertEqual(self.roles(saved["messages"]),
                         ["user", "tool_call", "tool_result",
                          "assistant"])


class CorruptCheckpointCase(CheckpointBase):
    def test_corrupt_checkpoint_file_loads_normally(self):
        """(c) A garbage checkpoints.json is quarantined, the session
        loads exactly as saved, and nothing raises."""
        mode = self.new_mode()
        sid = mode.session_id
        mode._messages = [{"role": "user", "text": "earlier"},
                          {"role": "assistant", "text": "done"}]
        mode._save_turn()

        path = checkpoint_store.default_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{not json at all")

        fresh = self.new_mode()
        self.assertEqual(fresh._messages, mode._messages)
        self.assertEqual(self.notes(fresh._messages), [])
        self.assertIsNone(self.filed(sid))
        quarantined = list(path.parent.glob("checkpoints.json.corrupt-*"))
        self.assertTrue(quarantined, "corrupt file was not preserved")


class CheckpointFailureCase(CheckpointBase):
    def test_checkpoint_write_failure_never_breaks_the_turn(self):
        """(d) The store's save raises on every call — the turn
        still completes, answers, and persists normally."""
        mode = self.new_mode()
        sid = mode.session_id
        self.prep_cloud(mode, ToolThenAnswerTransport(str(self.dir_a)))
        saved_save = CheckpointStore.save

        def boom(self_):
            raise OSError("disk full")

        CheckpointStore.save = boom
        self.addCleanup(setattr, CheckpointStore, "save", saved_save)
        with verification_off():
            out = mode.handle("list my files")
        self.assertEqual(out, CLOUD_TEXT)
        self.assertEqual(self.roles(mode._messages),
                         ["user", "tool_call", "tool_result",
                          "assistant"])
        saved = self.store.get_session(sid)
        self.assertEqual(saved["messages"], mode._messages)

    def test_save_checkpoint_wrapper_swallows_store_errors(self):
        """(d2) Even if the store module itself exploded, the loop's
        helper must not let it reach the turn."""
        mode = self.new_mode()
        saved_fn = checkpoint_store.save_checkpoint

        def boom(*a, **k):
            raise RuntimeError("store exploded")

        checkpoint_store.save_checkpoint = boom
        self.addCleanup(setattr, checkpoint_store, "save_checkpoint",
                        saved_fn)
        mode._messages = [{"role": "user", "text": "q"},
                          {"role": "tool_result", "tool": "t",
                           "text": "r"}]
        mode._save_checkpoint()   # must not raise


class ForeignCheckpointCase(CheckpointBase):
    def test_checkpoint_for_session_a_is_not_restored_into_b(self):
        """(e) Checkpoints are keyed by session id: B's load neither
        sees nor consumes A's; switching to A resumes it there."""
        sid_a = self.store.create_session("agent")
        checkpoint_store.save_checkpoint(
            sid_a, "A_TASK",
            [{"role": "user", "text": "A_TASK"},
             {"role": "tool_call", "tool": "list_files", "args": {}},
             {"role": "tool_result", "tool": "list_files",
              "text": "A_RESULT"}],
            1)
        sid_b = self.store.create_session("agent")   # B becomes active
        self.assertNotEqual(sid_a, sid_b)

        mode = self.new_mode()
        self.assertEqual(mode.session_id, sid_b)
        self.assertEqual(self.notes(mode._messages), [])
        self.assertIsNotNone(self.filed(sid_a),
                             "B's load consumed A's checkpoint")

        self.assertTrue(mode.reload_session(sid_a))
        self.assertEqual(mode._messages[0]["text"], "A_TASK")
        self.assertEqual(mode._messages[-1]["role"], "assistant")
        self.assertIn(NOTE_MARK, mode._messages[-1]["text"])
        self.assertIsNone(self.filed(sid_a))


class StoreShapeCase(unittest.TestCase):
    """The store itself, with an explicit path (usage_store style)."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="checkpoint-store-"))
        self.path = self.dir / "checkpoints.json"

    def load(self):
        return CheckpointStore.load(self.path)

    def test_round_trip(self):
        s = self.load()
        msgs = [{"role": "user", "text": "task"},
                {"role": "tool_result", "tool": "t", "text": "r"}]
        s.put("s1", "task", msgs, 2)
        s.save()
        e = self.load().get("s1")
        self.assertEqual(e["session_id"], "s1")
        self.assertEqual(e["user_text"], "task")
        self.assertEqual(e["steps"], 2)
        self.assertEqual(e["messages"], msgs)
        self.assertTrue(e["saved_at"])
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(raw["version"], 1)

    def test_discard(self):
        s = self.load()
        s.put("s1", "t", [{"role": "user", "text": "t"}], 0)
        s.save()
        s2 = self.load()
        self.assertTrue(s2.discard("s1"))
        self.assertFalse(s2.discard("s1"))
        s2.save()
        self.assertIsNone(self.load().get("s1"))

    def test_malformed_entries_are_invisible(self):
        self.path.write_text(json.dumps({
            "version": 1,
            "checkpoints": {
                "bad-shape": {"messages": "nope"},
                "empty": {"messages": []},
                "foreign": {"session_id": "other",
                            "messages": [{"role": "user",
                                          "text": "x"}]},
                "no-role": {"messages": [{"text": "x"}]},
            },
        }), encoding="utf-8")
        s = self.load()
        for sid in ("bad-shape", "empty", "foreign", "no-role",
                    "missing"):
            self.assertIsNone(s.get(sid), sid)

    def test_corrupt_file_is_quarantined_not_fatal(self):
        self.path.write_bytes(b"\x00\x01 garbage")
        s = self.load()
        self.assertIsNone(s.get("s1"))
        side = list(self.dir.glob("checkpoints.json.corrupt-*"))
        self.assertTrue(side, "corrupt bytes were not preserved")
        # And the store still works afterwards.
        s.put("s1", "t", [{"role": "user", "text": "t"}], 0)
        s.save()
        self.assertIsNotNone(self.load().get("s1"))


if __name__ == "__main__":
    unittest.main()
