"""Release 18 — auto-compaction must summarize, never silently drop.

Before R18, `_trim_messages` capped the agent transcript at 40 entries
by dropping the oldest ones without a trace: a long session lost its
early decisions and facts mid-conversation. Now the turn-end cap
(`_cap_history`) first summarizes the head that would be dropped into
one rolling summary pair — the same pair shape manual `compress_session`
writes — and only falls back to the plain trim when summarization is
off, unnecessary, or fails.

These tests run the real `AgentMode.handle()` (cloud and local). The
summarizer matches the backend serving the turn (Release 18 revision):
a cloud turn's summary comes from a stub transport that recognizes
the summary request by its prompt and answers it with a known text —
while the fake local model's `ask()` records every prompt so a test
can prove the local engine was never asked — and a local turn's
summary comes from that same `ask()`. Direct `_cap_history` drives
cover the pairing boundary. Nothing loads weights or opens a socket.

Run:  sh tests/run.sh -p "test_auto_compact.py"
"""
from __future__ import annotations

import contextlib
import unittest

from _harness import bootstrap

bootstrap()

import config  # noqa: E402
from core.agent_backend import AgentBackend  # noqa: E402
from core.key_manager import KeyManager  # noqa: E402
from modes.agent import AgentMode, _SUMMARY_MARKER  # noqa: E402

ALPHA = "<redacted>"
CLOUD_TEXT = "CLOUD_REPLY_OK"
LOCAL_TEXT = "LOCAL_MODEL_ANSWER"
KNOWN = "KNOWN_SUMMARY_TEXT"
KNOWN2 = "SECOND_SUMMARY_TEXT"
BACKEND_SUMMARY = "BACKEND_SUMMARY_TEXT"
SUMMARY_PHRASE = "Summarize the following conversation"
CAP = 40


@contextlib.contextmanager
def verification_off():
    """Keep the reviewer out of these turns (see test_agent_cloud.py)."""
    saved = config.AGENT_VERIFY_ENABLED
    config.AGENT_VERIFY_ENABLED = False
    try:
        yield
    finally:
        config.AGENT_VERIFY_ENABLED = saved


class RecordingTransport:
    """Stands in for a provider: always answers CLOUD_TEXT."""

    def __init__(self):
        self.calls = []

    def __call__(self, url, body, headers, timeout=30):
        self.calls.append((url, body, headers))
        return {"choices": [{"message": {"content": CLOUD_TEXT}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2}}


class SummaryTransport:
    """Provider stub that can tell the summary step apart from a turn
    step: the summary request's body carries the bullet-summary
    prompt. Turn steps get CLOUD_TEXT; the summary step gets
    BACKEND_SUMMARY — or raises, when ``fail_summary`` is set."""

    def __init__(self, fail_summary=False):
        self.calls = []
        self.fail_summary = fail_summary

    def __call__(self, url, body, headers, timeout=30):
        self.calls.append((url, body, headers))
        if SUMMARY_PHRASE in str(body):
            if self.fail_summary:
                raise RuntimeError("summary step exploded")
            return {"choices": [{"message": {"content": BACKEND_SUMMARY}}],
                    "usage": {"prompt_tokens": 9, "completion_tokens": 4}}
        return {"choices": [{"message": {"content": CLOUD_TEXT}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2}}

    def summary_calls(self):
        return [c for c in self.calls if SUMMARY_PHRASE in str(c[1])]


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
    """Local model stub: conversations for the local loop, ask() for
    the summarizer. ask() records every prompt, then returns
    ``ask_reply`` or raises ``ask_error``."""

    def __init__(self):
        self.log = []
        self._engine = None
        self.ask_prompts = []
        self.ask_reply = KNOWN
        self.ask_error = None

    @property
    def loaded(self):
        return any(k == "load" for k, _ in self.log)

    def load(self):
        self.log.append(("load", None))
        self._engine = FakeEngine(self.log)

    def ask(self, prompt):
        self.ask_prompts.append(prompt)
        if self.ask_error is not None:
            raise self.ask_error
        return self.ask_reply


def pairs(n, start=0):
    """n user/assistant entries with unique, ordered texts."""
    out = []
    for i in range(start, start + n):
        role = "user" if i % 2 == 0 else "assistant"
        out.append({"role": role, "text": f"seed-{i}"})
    return out


def expected_plain_trim(full):
    """What v1.15.0 would have kept: _trim_messages on the full list."""
    class _Scratch:
        pass
    s = _Scratch()
    s._messages = list(full)
    AgentMode._trim_messages(s)
    return s._messages


class AutoCompactBase(unittest.TestCase):
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

    def prep(self, mode="cloud", transport=None):
        self.km.set_mode(mode)
        self.km.save()
        be = AgentBackend(keys=self.km, net_state=lambda: "online",
                          http=transport or RecordingTransport())
        self.mode._backend = be
        return be

    def markers(self, msgs):
        return [m for m in msgs
                if m.get("role") == "user"
                and m.get("text") == _SUMMARY_MARKER]


class AutoCompactCloudCase(AutoCompactBase):
    def test_over_cap_turn_compacts_to_one_summary_pair(self):
        """(a) Over-cap cloud turn: the BACKEND writes the summary —
        the local engine is never asked. Head summarized, tail kept."""
        transport = SummaryTransport()
        self.prep(mode="cloud", transport=transport)
        seed = pairs(44)
        self.mode._messages = list(seed)
        with verification_off():
            out = self.mode.handle("hello")
        self.assertEqual(out, CLOUD_TEXT)
        msgs = self.mode._messages
        self.assertEqual(len(msgs), CAP)
        # Exactly one summary pair, at the front, carrying the text
        # only the backend stub produces.
        self.assertEqual(len(self.markers(msgs)), 1)
        self.assertEqual(msgs[0], {"role": "user", "text": _SUMMARY_MARKER})
        self.assertEqual(msgs[1], {"role": "assistant",
                                   "text": BACKEND_SUMMARY})
        # The backend saw exactly one summary step, carrying the
        # dropped head; the local engine was never touched.
        summaries = transport.summary_calls()
        self.assertEqual(len(summaries), 1)
        self.assertIn("seed-0", str(summaries[0][1]))
        self.assertEqual(self.llm.ask_prompts, [])
        # The kept tail is the newest history, ending with this turn.
        full = seed + [{"role": "user", "text": "hello"}, msgs[-1]]
        self.assertEqual(msgs[2:], full[-(CAP - 2):])
        self.assertEqual(msgs[-2], {"role": "user", "text": "hello"})
        self.assertEqual(msgs[-1]["text"], CLOUD_TEXT)

    def test_pairing_boundary_never_starts_tail_with_tool_result(self):
        """(a2) The split advances past a tool_result whose call drops."""
        seed = pairs(3)  # idx 0..2
        seed.append({"role": "tool_call", "tool": "read_file",
                     "args": {"path": "x"}})                    # idx 3
        seed.append({"role": "tool_result", "tool": "read_file",
                     "text": "TOOL_RESULT_BODY"})               # idx 4
        seed += pairs(37, start=5)                              # idx 5..41
        self.assertEqual(len(seed), 42)
        self.mode._messages = list(seed)
        self.mode._cap_history()
        msgs = self.mode._messages
        # Naive cut is 42 - 38 = 4, which lands on the tool_result; the
        # boundary must advance to 5 so the pair is summarized together.
        self.assertEqual(msgs[2], seed[5])
        self.assertNotEqual(msgs[2].get("role"), "tool_result")
        self.assertIn("TOOL_RESULT_BODY", self.llm.ask_prompts[0])
        self.assertLessEqual(len(msgs), CAP)

    def test_second_compaction_rolls_summary_forward(self):
        """(b) The next compaction summarizes the previous summary too."""
        self.mode._messages = pairs(44)
        self.mode._cap_history()
        self.assertEqual(self.mode._messages[1]["text"], KNOWN)
        # Grow past the cap again; the old pair is now part of the head.
        self.mode._messages += pairs(10, start=100)
        self.llm.ask_reply = KNOWN2
        self.mode._cap_history()
        msgs = self.mode._messages
        self.assertEqual(len(self.markers(msgs)), 1,
                         "summaries stacked instead of rolling forward")
        self.assertEqual(msgs[1]["text"], KNOWN2)
        self.assertEqual(len(self.llm.ask_prompts), 2)
        self.assertIn(KNOWN, self.llm.ask_prompts[1],
                      "previous summary never reached the summarizer")
        self.assertLessEqual(len(msgs), CAP)

    def test_backend_summarizer_failure_falls_back_to_plain_trim(self):
        """(b) The backend summary step raises → the turn still lands
        and the result is identical to the v1.15.0 trim."""
        transport = SummaryTransport(fail_summary=True)
        self.prep(mode="cloud", transport=transport)
        seed = pairs(44)
        self.mode._messages = list(seed)
        with verification_off():
            out = self.mode.handle("hello")
        self.assertEqual(out, CLOUD_TEXT)
        # The summary was attempted on the backend exactly once…
        self.assertEqual(len(transport.summary_calls()), 1)
        self.assertEqual(self.llm.ask_prompts, [])
        # …and its failure cost nothing but the summary itself.
        msgs = self.mode._messages
        full = seed + [{"role": "user", "text": "hello"}, msgs[-1]]
        self.assertEqual(msgs, expected_plain_trim(full))
        self.assertEqual(len(self.markers(msgs)), 0)

    def test_empty_summary_falls_back_to_plain_trim(self):
        """(c2) An empty summary is a failure, not a pair worth keeping."""
        self.mode._messages = pairs(44)
        self.llm.ask_reply = "   "
        before = list(self.mode._messages)
        self.mode._cap_history()
        self.assertEqual(self.mode._messages, expected_plain_trim(before))
        self.assertEqual(len(self.markers(self.mode._messages)), 0)

    def test_opt_out_uses_plain_trim_and_never_asks(self):
        """(d) AI_AGENT_AUTO_COMPACT=off → v1.15.0 behaviour, no call."""
        saved = config.AUTO_COMPACT_ENABLED
        config.AUTO_COMPACT_ENABLED = False
        self.addCleanup(setattr, config, "AUTO_COMPACT_ENABLED", saved)
        self.prep(mode="cloud")
        seed = pairs(44)
        self.mode._messages = list(seed)
        with verification_off():
            self.mode.handle("hello")
        self.assertEqual(self.llm.ask_prompts, [])
        msgs = self.mode._messages
        full = seed + [{"role": "user", "text": "hello"}, msgs[-1]]
        self.assertEqual(msgs, expected_plain_trim(full))

    def test_under_cap_turn_never_calls_summarizer(self):
        """(e) Under the cap: no summarizer call on either engine,
        transcript untouched apart from the turn itself."""
        transport = SummaryTransport()
        self.prep(mode="cloud", transport=transport)
        seed = pairs(10)
        self.mode._messages = list(seed)
        with verification_off():
            self.mode.handle("hello")
        self.assertEqual(self.llm.ask_prompts, [])
        self.assertEqual(transport.summary_calls(), [])
        self.assertEqual(len(transport.calls), 1,
                         "the turn step itself is the only provider call")
        msgs = self.mode._messages
        self.assertEqual(len(msgs), 12)
        self.assertEqual(msgs[:10], seed)
        self.assertEqual(len(self.markers(msgs)), 0)

    def test_locked_turn_summarizes_on_pinned_provider_only(self):
        """(c) Locked mode: the summary step runs on the pinned
        provider alone — be.step's route is the turn's own route, so
        no other provider is ever asked to summarize."""
        transport = SummaryTransport()
        self.prep(mode="cloud", transport=transport)
        self.km.add_key("deepseek", ALPHA, "beta")
        cfg = self.km.load()
        cfg["auto_switch"] = False
        cfg["pinned_provider"] = "openai"
        self.km.save()
        self.mode._messages = pairs(44)
        with verification_off():
            out = self.mode.handle("hello")
        self.assertEqual(out, CLOUD_TEXT)
        summaries = transport.summary_calls()
        self.assertEqual(len(summaries), 1)
        self.assertIn("api.openai.com", summaries[0][0])
        self.assertFalse(
            any("api.deepseek.com" in url for url, _, _ in transport.calls),
            "a non-pinned provider was called in locked mode")
        self.assertEqual(self.llm.ask_prompts, [])
        msgs = self.mode._messages
        self.assertEqual(len(self.markers(msgs)), 1)
        self.assertEqual(msgs[1], {"role": "assistant",
                                   "text": BACKEND_SUMMARY})

    def test_cloud_only_turn_never_touches_local_engine(self):
        """(d) Cloud Only: the summary comes from the backend; the
        local engine is neither asked nor even loaded — it may not
        exist in this mode at all."""
        transport = SummaryTransport()
        self.prep(mode="cloud", transport=transport)
        self.mode._messages = pairs(44)
        with verification_off():
            out = self.mode.handle("hello")
        self.assertEqual(out, CLOUD_TEXT)
        self.assertEqual(len(transport.summary_calls()), 1)
        self.assertEqual(self.llm.ask_prompts, [])
        self.assertFalse(self.llm.loaded,
                         "cloud-only turn loaded the local engine")
        msgs = self.mode._messages
        self.assertEqual(len(self.markers(msgs)), 1)
        self.assertEqual(msgs[1], {"role": "assistant",
                                   "text": BACKEND_SUMMARY})


class AutoCompactLocalCase(AutoCompactBase):
    def test_over_cap_local_turn_compacts(self):
        """(a-local) The local loop's turn end compacts the same way,
        summarized by the local engine that served the turn."""
        self.prep(mode="local")
        self.mode._messages = pairs(44)
        out = self.mode.handle("hello")
        self.assertEqual(out, LOCAL_TEXT)
        msgs = self.mode._messages
        self.assertEqual(len(msgs), CAP)
        self.assertEqual(len(self.markers(msgs)), 1)
        self.assertEqual(msgs[1], {"role": "assistant", "text": KNOWN})
        self.assertEqual(len(self.llm.ask_prompts), 1)
        self.assertEqual(msgs[-1]["text"], LOCAL_TEXT)

    def test_local_summarizer_failure_falls_back_to_plain_trim(self):
        """(c-local) The local summarizer raises → result identical
        to the v1.15.0 trim, transcript otherwise untouched."""
        self.mode._messages = pairs(44)
        self.llm.ask_error = RuntimeError("engine gone")
        before = list(self.mode._messages)
        self.mode._cap_history()
        self.assertEqual(self.mode._messages, expected_plain_trim(before))
        self.assertEqual(len(self.markers(self.mode._messages)), 0)


if __name__ == "__main__":
    unittest.main()
