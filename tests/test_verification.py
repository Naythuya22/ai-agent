"""Verification pass (cloud only).

After a tool turn the model has committed to an answer. One cheap reviewer
request checks that answer against the tool output while the evidence is still
in the transcript.

Two gates: a tool must actually have run this turn, and the backend must be
cloud. The local 4B shares one conversation object, so a review turn there
would cost phone tokens for nothing.

Run:  sh tests/run.sh -p "test_verification.py"
"""
from __future__ import annotations

import unittest

from _harness import bootstrap, get_app  # noqa: F401

bootstrap()

import config  # noqa: E402
from modes import agent as agent_mod  # noqa: E402


class _Step:
    """Stands in for AgentBackend.Step."""

    def __init__(self, text="", tool_calls=None, reasoning=""):
        self.text = text
        self.tool_calls = tool_calls or []
        self.reasoning = reasoning
        self.model = "test-model"
        self.provider = "test"


class _Backend:
    """A backend whose reviewer reply is scripted."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def step(self, system=None, messages=None, tools=None, max_tokens=None):
        self.calls.append({"system": system, "messages": messages,
                           "tools": tools, "max_tokens": max_tokens})
        reply = self.replies.pop(0) if self.replies else ""
        if isinstance(reply, Exception):
            raise reply
        return _Step(text=reply)


def _mode_with_results():
    m = agent_mod.AgentMode.__new__(agent_mod.AgentMode)
    m._messages = [{"role": "user", "text": "q"},
                   {"role": "tool_result", "tool": "read_file", "text": "evidence"}]
    m._turn_cache = {}
    m._turn_cache_order = []
    return m


def _steps(*texts):
    """A step_fn replaying scripted texts (no tool calls)."""
    seq = list(texts)

    def _fn():
        return _Step(text=seq.pop(0) if seq else "")
    return _fn


class DecideVerifyTests(unittest.TestCase):
    def test_decide_ok(self):
        self.assertEqual(agent_mod._decide_verify("OK", "cand"), ("OK", "cand"))
        self.assertEqual(agent_mod._decide_verify("ok — supported", "cand"),
                         ("OK", "cand"))

    def test_decide_fix(self):
        self.assertEqual(agent_mod._decide_verify("FIX: real answer", "cand"),
                         ("FIX", "real answer"))

    def test_decide_fix_without_a_body_keeps_the_candidate(self):
        self.assertEqual(agent_mod._decide_verify("FIX:", "cand"), ("OK", "cand"))

    def test_decide_missing(self):
        self.assertEqual(agent_mod._decide_verify("MISSING: which year?", "cand"),
                         ("MISSING", "which year?"))

    def test_unreadable_reply_keeps_the_candidate(self):
        for junk in ("", "   ", None, "maybe?", "I think it is fine"):
            with self.subTest(junk=junk):
                self.assertEqual(agent_mod._decide_verify(junk, "cand"),
                                 ("OK", "cand"))

    def test_fix_wins_over_a_leading_ok(self):
        self.assertEqual(agent_mod._decide_verify("FIX: no", "cand"), ("FIX", "no"))


class VerifyCloudTests(unittest.TestCase):
    def test_verification_pass_ok_uses_candidate(self):
        m = _mode_with_results()
        be = _Backend(["OK"])
        out = m._verify_cloud(be, "the answer", _steps())
        self.assertEqual(out, "the answer")
        self.assertEqual(len(be.calls), 1)

    def test_verification_pass_fix_replaces_candidate(self):
        m = _mode_with_results()
        be = _Backend(["FIX: the corrected answer"])
        out = m._verify_cloud(be, "wrong answer", _steps())
        self.assertEqual(out, "the corrected answer")

    def test_verification_pass_missing_triggers_one_more_round(self):
        m = _mode_with_results()
        be = _Backend(["MISSING: which year?"])
        step_fn = _steps("the answer with the missing detail")
        out = m._verify_cloud(be, "partial answer", step_fn)
        self.assertEqual(out, "the answer with the missing detail")
        self.assertEqual(len(be.calls), 1, "exactly one review")

    def test_verification_skipped_when_no_tools_called(self):
        m = agent_mod.AgentMode.__new__(agent_mod.AgentMode)
        m._messages = [{"role": "user", "text": "hi"}]
        m._turn_cache = {}
        m._turn_cache_order = []
        be = _Backend(["FIX: should never be used"])
        out = m._verify_cloud(be, "the answer", _steps())
        self.assertEqual(out, "the answer")
        self.assertEqual(be.calls, [], "pure chat must cost no extra request")

    def test_verification_disabled_by_flag(self):
        m = _mode_with_results()
        be = _Backend(["FIX: should never be used"])
        saved = config.AGENT_VERIFY_ENABLED
        config.AGENT_VERIFY_ENABLED = False
        try:
            out = m._verify_cloud(be, "the answer", _steps())
            self.assertEqual(out, "the answer")
            self.assertEqual(be.calls, [])
        finally:
            config.AGENT_VERIFY_ENABLED = saved

    def test_verification_failure_falls_back_to_candidate(self):
        m = _mode_with_results()
        be = _Backend([RuntimeError("network down")])
        out = m._verify_cloud(be, "the answer", _steps())
        self.assertEqual(out, "the answer", "a broken review must not lose the answer")

    def test_verification_does_not_recurse(self):
        """A MISSING round that asks for tools must not trigger a second review."""
        m = _mode_with_results()
        be = _Backend(["MISSING: need more"])

        def _fn():
            # asks for a tool, then answers
            if not getattr(_fn, "seen", False):
                _fn.seen = True
                return _Step(tool_calls=[{"function": {
                    "name": "read_file", "arguments": {"path": "/x"}}}])
            return _Step(text="final answer after the extra round")

        out = m._verify_cloud(be, "partial answer", _fn)
        self.assertEqual(out, "final answer after the extra round")
        self.assertEqual(len(be.calls), 1, "the reviewer is called exactly once")
        roles = [x["role"] for x in m._messages]
        self.assertIn("tool_call", roles, "the round keeps the transcript shape")
        self.assertIn("tool_result", roles)

    def test_empty_candidate_is_never_reviewed(self):
        m = _mode_with_results()
        be = _Backend(["FIX: something"])
        self.assertEqual(m._verify_cloud(be, "", _steps()), "")
        self.assertEqual(be.calls, [])

    def test_reviewer_request_is_cheap_and_isolated(self):
        m = _mode_with_results()
        be = _Backend(["OK"])
        before = len(m._messages)
        m._verify_cloud(be, "the answer", _steps())
        call = be.calls[0]
        self.assertEqual(call["max_tokens"], config.AGENT_VERIFY_MAX_TOKENS)
        self.assertLessEqual(config.AGENT_VERIFY_MAX_TOKENS, 1024)
        self.assertEqual(call["tools"], [], "the reviewer gets no tools")
        self.assertEqual(call["system"], agent_mod._VERIFY_SYSTEM)
        self.assertEqual(len(m._messages), before,
                         "the review must not write to the transcript")
        body = call["messages"][0]["text"]
        self.assertIn("evidence", body)
        self.assertIn("the answer", body)

    def test_compact_tool_results_are_capped(self):
        m = _mode_with_results()
        # 12 entries x ~800 chars is comfortably past the 4000-char ceiling.
        for i in range(12):
            m._messages.append({"role": "tool_result", "tool": "web_fetch",
                                "text": "%02d%s" % (i, "y" * 900)})
        out = agent_mod._compact_tool_results(m._messages)
        self.assertLessEqual(len(out), agent_mod._VERIFY_TOTAL_CHARS + 4)
        # each entry is capped too, so one tool cannot crowd out the others
        self.assertIn("read_file", out)
        self.assertIn("web_fetch", out)
        per_entry = out.split("\n")
        self.assertTrue(all(len(line) <= agent_mod._VERIFY_ENTRY_CHARS + 40
                            for line in per_entry), per_entry[:2])

    def test_compact_tool_results_redacts(self):
        m = _mode_with_results()
        m._messages[1]["text"] = "sk-live-ABCDEFGHIJKLMNOP"
        out = agent_mod._compact_tool_results(m._messages)
        self.assertNotIn("sk-live-ABCDEFGHIJKLMNOP", out)


class CloudTurnCallCountTests(unittest.TestCase):
    """The whole point of the review is that it costs one more request.

    The tool-loop tests in test_agent_cloud.py / test_agent_stream_cloud.py
    count provider requests with the reviewer switched off; this is the other
    half — with verification on, the same turn must make exactly one more call.
    """

    def setUp(self):
        agent_mod.record_usage("")

    def tearDown(self):
        agent_mod.record_usage("")

    def test_cloud_turn_adds_verification_call_when_enabled(self):
        import json
        import tempfile
        from pathlib import Path

        from core.agent_backend import AgentBackend
        from core.key_manager import KeyManager
        from modes.agent import AgentMode

        class _Transport:
            """Call 1 asks for a tool, call 2 answers, call 3 reviews."""

            def __init__(self):
                self.calls = []

            def __call__(self, url, body, headers, timeout=30):
                self.calls.append((url, body, headers))
                if len(self.calls) == 1:
                    return {"choices": [{"message": {
                        "content": "",
                        "tool_calls": [{"id": "c1", "type": "function",
                                        "function": {"name": "list_files",
                                                     "arguments": json.dumps(
                                                         {"directory": self.tmp})}}],
                    }}], "usage": {"prompt_tokens": 4, "completion_tokens": 2}}
                if len(self.calls) == 2:
                    return {"choices": [{"message": {"content": "THE ANSWER"}}],
                            "usage": {"prompt_tokens": 6, "completion_tokens": 3}}
                return {"choices": [{"message": {"content": "OK"}}],
                        "usage": {"prompt_tokens": 6, "completion_tokens": 1}}

        tr = _Transport()
        tr.tmp = tempfile.mkdtemp(prefix="verify-tools-")
        Path(tr.tmp, "seen-by-the-tool.txt").write_text("hello", "utf-8")

        km = KeyManager()
        if km.path.exists():
            km.path.unlink()
        km.add_key("openai", "sk-verify-aaaaaaaaaaaaaaa1111", "alpha")
        cfg = km.load()
        cfg["prefer_cloud"] = True
        cfg["cost_ack"] = True
        km.save()
        km.set_mode("cloud")
        km.save()

        # Real constructor: it wires up the tool wrappers and the turn cache
        # exactly as the phone does. Only the store is detached so the test
        # never reads or writes a session file.
        mode = AgentMode(None, None)
        mode._store = None
        mode._messages = []
        mode.turns = 0
        mode._backend = AgentBackend(keys=km, net_state=lambda: "online", http=tr)

        saved = config.AGENT_VERIFY_ENABLED
        config.AGENT_VERIFY_ENABLED = True
        # Release 16: the file tools are jailed to config.PROJECT_ROOT
        # when no project is active — point the root resolver at the
        # transport's tmp tree so list_files really lists it.
        saved_root = config.PROJECT_ROOT
        config.PROJECT_ROOT = Path(tr.tmp)
        try:
            out = mode.handle("list my files")
        finally:
            config.AGENT_VERIFY_ENABLED = saved
            config.PROJECT_ROOT = saved_root

        self.assertEqual(out, "THE ANSWER")
        self.assertEqual(len(tr.calls), 3,
                         "the turn must be: tool call, answer over the result, review")

        # The third request is the reviewer: no tools, a tiny budget, the
        # reviewer prompt as the system message, and the tool output quoted.
        review = tr.calls[2][1]
        self.assertNotIn("tools", review, "the reviewer gets no tools")
        self.assertEqual(review.get("max_tokens"),
                         config.AGENT_VERIFY_MAX_TOKENS)
        self.assertLessEqual(review.get("max_tokens", 10 ** 9), 1024)
        self.assertEqual(review["messages"][0]["role"], "system")
        self.assertEqual(review["messages"][0]["content"], agent_mod._VERIFY_SYSTEM)
        self.assertIn("TOOL RESULTS:", review["messages"][1]["content"])
        self.assertIn("seen-by-the-tool.txt", review["messages"][1]["content"])


class ShouldSkipVerifyTests(unittest.TestCase):
    """Batch 4.2i: capability nudge turns skip the reviewer."""

    def test_skip_when_nudged(self):
        m = _mode_with_results()
        m._cap_nudged = True
        self.assertTrue(m._should_skip_verify())

    def test_no_skip_by_default(self):
        m = _mode_with_results()
        m._cap_nudged = False
        self.assertFalse(m._should_skip_verify())

    def test_no_skip_when_attribute_missing(self):
        # test stubs build AgentMode via __new__ (no __init__)
        m = agent_mod.AgentMode.__new__(agent_mod.AgentMode)
        self.assertFalse(m._should_skip_verify())


if __name__ == "__main__":
    unittest.main()