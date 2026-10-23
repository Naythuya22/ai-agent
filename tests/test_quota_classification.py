"""Release 26 — 429 messaging vs reality (FIX 5, tester evaluation #5).

The eval replies said "wait ~60 seconds" while Gemini's DAILY quota
was exhausted (429 for hours). The classifier now distinguishes:

  (a) a short rate limit — keeps the (~Ns) wait wording;
  (b) daily/quota exhaustion — detected from the provider's own
      payload (RESOURCE_EXHAUSTED / PerDay quota ids / "exceeded your
      current quota") — whose Burmese line says today's quota is used
      up, to retry tomorrow or switch provider/model, with NO
      misleading short wait, and which is never a mere "park".

A bare 429 or the words "quota exceeded … slow down" (a per-key
throttle) must stay class (a): quota alone is not exhaustion.

Run:  sh tests/run.sh -p "test_quota_classification.py"
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from _harness import bootstrap

bootstrap()

import config  # noqa: E402
from core.agent_backend import (  # noqa: E402
    AgentBackend, BackendExhausted, CloudError, failure_reason)
from core.key_manager import KeyManager  # noqa: E402
from modes import agent as agent_mod  # noqa: E402
from modes.agent import AgentMode  # noqa: E402
from sessions import SessionStore  # noqa: E402
from tools import ALL_TOOLS  # noqa: E402

GEMINI_DAILY = (
    '{"error": {"code": 429, "message": "You exceeded your current '
    'quota, please check your plan and billing details.", "status": '
    '"RESOURCE_EXHAUSTED", "details": [{"quotaId": '
    '"GenerateRequestsPerDayPerProjectPerModel-FreeTier", '
    '"quotaValue": "20"}]}}')
OPENAI_QUOTA = ('{"error": {"message": "You exceeded your current '
                'quota, please check your plan and billing details.", '
                '"type": "insufficient_quota", "code": '
                '"insufficient_quota"}}')
SHORT_LIMIT = ('{"error": {"message": "Rate limit exceeded. Please '
               'retry after a moment."}}')
KEY_THROTTLE = '{"error":{"message":"quota exceeded for this key, slow down"}}'


class FunctionTool:
    def __init__(self, fn):
        self.fn = fn


class FakeLocalLLM:
    def __init__(self):
        self._engine = None

    def load(self):
        raise AssertionError("cloud-only turn must not load local")


class FailureReasonCase(unittest.TestCase):
    def test_gemini_daily_quota_payload_is_quota_exhausted(self):
        err = CloudError("rate_limit", GEMINI_DAILY, status=429)
        self.assertEqual(failure_reason(err), "quota_exhausted")

    def test_openai_style_quota_is_quota_exhausted(self):
        err = CloudError("rate_limit", OPENAI_QUOTA, status=429)
        self.assertEqual(failure_reason(err), "quota_exhausted")

    def test_short_rate_limit_stays_rate_limited(self):
        err = CloudError("rate_limit", SHORT_LIMIT, status=429,
                         retry_after=45)
        self.assertEqual(failure_reason(err), "rate_limited")

    def test_per_key_throttle_stays_rate_limited(self):
        # "quota" alone is not exhaustion — the Release 24 wording.
        err = CloudError("rate_limit", KEY_THROTTLE, status=429,
                         retry_after=60)
        self.assertEqual(failure_reason(err), "rate_limited")


def classify(records):
    mode = AgentMode.__new__(AgentMode)

    class _BE:
        @staticmethod
        def failure_overview():
            return []

    exc = BackendExhausted("providers exhausted")
    exc.failures = records
    return mode._classify_failure(_BE(), exc)


def record(provider, err):
    # The exact record shape AgentBackend.step() stores per failure.
    return {"provider": provider, "reason": failure_reason(err),
            "status": err.status, "kind": err.kind,
            "retry_after": err.retry_after}


class ClassifiedLineCase(unittest.TestCase):
    def test_quota_line_says_tomorrow_and_has_no_short_wait(self):
        out = classify([record("gemini", CloudError(
            "rate_limit", GEMINI_DAILY, status=429))])
        self.assertEqual(len(out["lines"]), 1)
        line = out["lines"][0]
        self.assertIn("ဒီနေ့ အတွက် quota ကုန်သွားပြီ", line)
        self.assertIn("မနက်ဖြန် ပြန်စမ်းပါ", line)
        self.assertNotIn("(~", line)
        self.assertFalse(out["park_only"])
        self.assertIsNone(out["wait"])

    def test_short_limit_line_keeps_the_wait(self):
        out = classify([record("gemini", CloudError(
            "rate_limit", SHORT_LIMIT, status=429, retry_after=45))])
        line = out["lines"][0]
        self.assertIn("Limit ပြည့်သွားပြီ (429)", line)
        self.assertIn("(~45 စက္ကန့်)", line)
        self.assertTrue(out["park_only"])
        self.assertEqual(out["wait"], 45)


class QuotaEndToEndCase(unittest.TestCase):
    """Cloud Only turn whose only provider dies on the daily quota:
    the user-facing reply carries the quota line, not a 60s wait."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="quota-e2e-"))
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
        self.km.add_key("gemini", "<redacted>", "g")
        cfg = self.km.load()
        cfg["cost_ack"] = True
        self.km.save()
        self.km.set_mode("cloud")
        self.km.save()

    def test_daily_quota_reply(self):
        def transport(url, body, headers, timeout=30):
            raise CloudError("rate_limit", GEMINI_DAILY, status=429)

        mode = AgentMode(FakeLocalLLM(), None)
        mode._tools = [FunctionTool(fn) for fn in ALL_TOOLS]
        mode._backend = AgentBackend(keys=self.km,
                                     net_state=lambda: "online",
                                     http=transport)
        out = mode.handle("မင်္ဂလာပါ၊ ဒီနေ့ ရာသီဥတု ဘယ်လိုလဲ")
        self.assertIn("ဒီနေ့ အတွက် quota ကုန်သွားပြီ", out)
        self.assertIn("မနက်ဖြန် ပြန်စမ်းပါ", out)
        self.assertNotIn("(~", out)
        self.assertNotIn("RESOURCE_EXHAUSTED", out)   # no raw payload


if __name__ == "__main__":
    unittest.main()
