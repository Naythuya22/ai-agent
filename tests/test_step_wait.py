"""F14 — step-level wait-and-retry for transient cloud failures.

The policy lives in AgentBackend.step()/step_stream() around the
fail-fast _step_once/_stream_once walks. These tests drive the wrapper
with scripted walks and a recorded (never real) sleep: a rate-limited
route inside the budget is waited out and retried; quota/auth/network
failures, an oversized wait, a spent budget and a stop request all
fail fast with the original exception.
"""
import time
import unittest
from unittest import mock

import config
from core.agent_backend import AgentBackend, BackendExhausted, Step


def _exc(failures, msg="providers exhausted"):
    e = BackendExhausted(msg)
    e.failures = failures
    return e


class _FakeBackend(AgentBackend):
    """AgentBackend with the walks and key-state reads scripted."""

    def __init__(self, script, overview=None, stream_script=None):
        # Deliberately no super().__init__: no keys, no network.
        self._script = list(script)          # Step or Exception, per call
        self._overview = overview or []
        self._stream_script = list(stream_script or [])
        self.once_calls = 0
        self.stream_calls = 0

    def _step_once(self, **kw):
        self.once_calls += 1
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def _stream_once(self, **kw):
        self.stream_calls += 1
        frames = self._stream_script.pop(0)
        yield from frames

    def failure_overview(self):
        return list(self._overview)

    def route_providers(self):
        return ["gemini"]


class _SleepRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, seconds):
        self.calls.append(seconds)

    @property
    def total(self):
        return sum(self.calls)


COOLING = [{"provider": "gemini", "reason": "cooling", "wait": 30}]
RL_FAIL = [{"provider": "gemini", "reason": "rate_limited",
            "status": 429, "kind": "rate_limit", "retry_after": 30}]


class TestStepWait(unittest.TestCase):
    def setUp(self):
        self.sleep = _SleepRecorder()
        patcher = mock.patch.object(time, "sleep", self.sleep)
        patcher.start()
        self.addCleanup(patcher.stop)
        cap = mock.patch.object(config, "step_wait_cap", lambda: 120)
        cap.start()
        self.addCleanup(cap.stop)

    def test_waits_out_rate_limit_then_succeeds(self):
        be = _FakeBackend([_exc(RL_FAIL), Step(text="ok")],
                          overview=COOLING)
        step = be.step(system="", messages=[], tools=[], wait_budget=150)
        self.assertEqual(step.text, "ok")
        self.assertEqual(be.once_calls, 2)
        self.assertAlmostEqual(step.waited_s, 31.0)
        self.assertAlmostEqual(self.sleep.total, 31.0)

    def test_default_budget_is_fail_fast(self):
        be = _FakeBackend([_exc(RL_FAIL)], overview=COOLING)
        with self.assertRaises(BackendExhausted):
            be.step(system="", messages=[], tools=[])
        self.assertEqual(self.sleep.total, 0)
        self.assertEqual(be.once_calls, 1)

    def test_quota_only_never_waits(self):
        # A daily-quota park reads as "cooling" in the overview, but its
        # wait is hours — far over the cap — so the step fails at once.
        be = _FakeBackend(
            [_exc([{"provider": "gemini", "reason": "quota_exhausted",
                    "status": 429, "kind": "rate_limit",
                    "retry_after": None}])],
            overview=[{"provider": "gemini", "reason": "cooling",
                       "wait": 20000}])
        with self.assertRaises(BackendExhausted):
            be.step(system="", messages=[], tools=[], wait_budget=150)
        self.assertEqual(self.sleep.total, 0)

    def test_network_never_waits(self):
        be = _FakeBackend(
            [_exc([{"provider": "gemini", "reason": "network",
                    "status": 0, "kind": "network", "retry_after": None}],
                  msg="network")],
            overview=COOLING)
        with self.assertRaises(BackendExhausted):
            be.step(system="", messages=[], tools=[], wait_budget=150)
        self.assertEqual(self.sleep.total, 0)

    def test_wait_over_budget_fails_fast(self):
        be = _FakeBackend([_exc(RL_FAIL)], overview=COOLING)
        with self.assertRaises(BackendExhausted):
            be.step(system="", messages=[], tools=[], wait_budget=10)
        self.assertEqual(self.sleep.total, 0)

    def test_budget_debits_across_retries(self):
        be = _FakeBackend([_exc(RL_FAIL), _exc(RL_FAIL), Step(text="ok")],
                          overview=COOLING)
        with self.assertRaises(BackendExhausted) as cm:
            be.step(system="", messages=[], tools=[], wait_budget=40)
        # First wait (31s) fits; the second (31s) does not fit the 9s left.
        self.assertAlmostEqual(cm.exception.waited_s, 31.0)
        self.assertEqual(be.once_calls, 2)

    def test_should_stop_aborts_the_wait(self):
        be = _FakeBackend([_exc(RL_FAIL), Step(text="ok")],
                          overview=COOLING)
        calls = {"n": 0}

        def stop():
            calls["n"] += 1
            return calls["n"] > 3

        with self.assertRaises(BackendExhausted) as cm:
            be.step(system="", messages=[], tools=[], wait_budget=150,
                    should_stop=stop)
        self.assertLess(cm.exception.waited_s, 31.0)
        self.assertEqual(be.once_calls, 1)   # never retried

    def test_on_wait_hook_fires_before_the_sleep(self):
        be = _FakeBackend([_exc(RL_FAIL), Step(text="ok")],
                          overview=COOLING)
        seen = []
        step = be.step(system="", messages=[], tools=[], wait_budget=150,
                       on_wait=seen.append)
        self.assertEqual(step.text, "ok")
        self.assertEqual(seen, [31.0])

    def test_no_waitable_route_raises_with_failures_intact(self):
        failures = [{"provider": "gemini", "reason": "key_invalid",
                     "status": 401, "kind": "auth", "retry_after": None}]
        be = _FakeBackend([_exc(failures)],
                          overview=[{"provider": "gemini",
                                     "reason": "key_invalid",
                                     "wait": None}])
        with self.assertRaises(BackendExhausted) as cm:
            be.step(system="", messages=[], tools=[], wait_budget=150)
        self.assertEqual(cm.exception.failures, failures)
        self.assertEqual(self.sleep.total, 0)


class TestStreamWait(unittest.TestCase):
    def setUp(self):
        self.sleep = _SleepRecorder()
        patcher = mock.patch.object(time, "sleep", self.sleep)
        patcher.start()
        self.addCleanup(patcher.stop)
        cap = mock.patch.object(config, "step_wait_cap", lambda: 120)
        cap.start()
        self.addCleanup(cap.stop)

    def test_stream_waits_then_yields_done(self):
        be = _FakeBackend([], overview=COOLING, stream_script=[
            [],     # first walk: exhausted, nothing yielded
            [("delta", "hi"), ("done", {"provider": "gemini"})],
        ])
        frames = list(be.step_stream(system="", messages=[], tools=[],
                                     wait_budget=150))
        self.assertEqual(frames[0], ("delta", "hi"))
        self.assertEqual(frames[1][0], "done")
        self.assertAlmostEqual(frames[1][1]["waited_s"], 31.0)
        self.assertEqual(be.stream_calls, 2)

    def test_stream_emit_wait_yields_a_wait_frame(self):
        be = _FakeBackend([], overview=COOLING, stream_script=[
            [],
            [("done", {"provider": "gemini"})],
        ])
        frames = list(be.step_stream(system="", messages=[], tools=[],
                                     wait_budget=150, emit_wait=True))
        self.assertEqual(frames[0], ("wait", {"seconds": 31.0}))
        self.assertEqual(frames[1][0], "done")

    def test_stream_in_band_error_never_waits(self):
        be = _FakeBackend([], overview=COOLING, stream_script=[
            [("delta", "partial"), ("error", {"message": "died"})],
        ])
        frames = list(be.step_stream(system="", messages=[], tools=[],
                                     wait_budget=150))
        self.assertEqual(frames[-1][0], "error")
        self.assertEqual(self.sleep.total, 0)
        self.assertEqual(be.stream_calls, 1)

    def test_stream_default_budget_returns_empty(self):
        be = _FakeBackend([], overview=COOLING, stream_script=[[]])
        frames = list(be.step_stream(system="", messages=[], tools=[]))
        self.assertEqual(frames, [])
        self.assertEqual(self.sleep.total, 0)


if __name__ == "__main__":
    unittest.main()
