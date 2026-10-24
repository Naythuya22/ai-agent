"""Release 30 — a daily-quota 429 parks the key until the quota
resets; a short rate limit keeps its minute.

Release 26 taught failure_reason() to SAY "today's quota is used up"
for a daily-quota 429, but _handle_error still parked the key for
the 60-second default — so every later step and turn re-selected the
spent key and re-paid the same refusal. Now the park matches the
words: mark_quota_exhausted() parks until the provider's next daily
reset moment (Gemini: midnight Pacific; Groq/OpenRouter: 00:00 UTC),
while a per-minute 429 keeps the Retry-After / 60s park.

Every test drives the backend through an injected fake transport, so
the suite never opens a socket and never sees a real key.

Run:  sh tests/run.sh -p "test_quota_parking.py"
"""
from __future__ import annotations

import time
import unittest

from _harness import bootstrap

bootstrap()

from core import key_manager as KM  # noqa: E402
from core.agent_backend import AgentBackend, CloudError, Step  # noqa: E402
from core.key_manager import KeyManager  # noqa: E402
from modes.agent import AgentMode, _wait_phrase  # noqa: E402

GEM1 = "gem-quota-a-1111111111111"
GEM2 = "gem-quota-b-2222222222222"

# The live Gemini free-tier daily-quota shape (same body as
# tests/test_quota_classification.py — RESOURCE_EXHAUSTED + a PerDay
# quota id, detail in the body, no Retry-After header).
GEMINI_DAILY = (
    '{"error": {"code": 429, "message": "You exceeded your current '
    'quota, please check your plan and billing details.", "status": '
    '"RESOURCE_EXHAUSTED", "details": [{"quotaId": '
    '"GenerateRequestsPerDayPerProjectPerModel-FreeTier", '
    '"quotaValue": "20"}]}}')
SHORT_LIMIT = ('{"error": {"message": "Rate limit exceeded. Please '
               'retry after a moment."}}')

GEMINI_OK = {"candidates": [{"content": {"parts": [
    {"text": "second key worked"}]}}],
    "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2}}


class QuotaParkingCase(unittest.TestCase):
    """Two Gemini keys; key 1 fails, key 2 answers."""

    def setUp(self):
        self.km = KeyManager()
        if self.km.path.exists():
            self.km.path.unlink()
        self.km.add_key("gemini", GEM1, "one")
        self.km.add_key("gemini", GEM2, "two")
        cfg = self.km.load()
        cfg["prefer_cloud"] = True
        cfg["cost_ack"] = True
        self.km.save()

    def backend(self, http):
        return AgentBackend(keys=self.km,
                            net_state=lambda: "online", http=http)

    def entry(self, raw_key):
        return [e for e in self.km.entries("gemini")
                if e["key"] == raw_key][0]

    def test_daily_quota_parks_until_reset_and_rotation_moves_on(self):
        calls = []

        def http(url, body, headers, timeout=30):
            calls.append(url)
            if "key=" + GEM1 in url:
                raise CloudError("rate_limit", GEMINI_DAILY, 429)
            return dict(GEMINI_OK)

        be = self.backend(http)
        before = time.time()
        step = be.step(system="s", messages=[], tools=[], max_tokens=8)
        after = time.time()

        self.assertIsInstance(step, Step)
        self.assertEqual(step.provider, "gemini")
        self.assertEqual(step.text, "second key worked")
        self.assertEqual(len(calls), 2, "the second key never got a turn")

        parked = self.entry(GEM1)
        self.assertEqual(parked["status"], "rate_limited")
        self.assertEqual(parked["last_error"], "quota exhausted (daily)")
        # The park ends at the next Pacific midnight — the bounded
        # equality pins it exactly (both bounds name the same reset
        # instant unless the test itself straddles one), which is
        # what "not the ~60s default" means here.
        self.assertGreaterEqual(
            parked["cooldown_until"], KM.quota_reset_after("gemini", before))
        self.assertLessEqual(
            parked["cooldown_until"], KM.quota_reset_after("gemini", after))

    def test_parked_quota_key_is_not_retried_on_the_next_step(self):
        def http1(url, body, headers, timeout=30):
            if "key=" + GEM1 in url:
                raise CloudError("rate_limit", GEMINI_DAILY, 429)
            return dict(GEMINI_OK)

        self.backend(http1).step(system="s", messages=[], tools=[],
                                  max_tokens=8)
        calls = []

        def http2(url, body, headers, timeout=30):
            calls.append(url)
            if "key=" + GEM1 in url:
                raise AssertionError("the quota-parked key was retried")
            return dict(GEMINI_OK)

        step = self.backend(http2).step(system="s", messages=[], tools=[],
                                        max_tokens=8)
        self.assertEqual(step.text, "second key worked")
        self.assertTrue(calls and all("key=" + GEM2 in u for u in calls))

    def test_short_rate_limit_still_parks_about_a_minute(self):
        def http(url, body, headers, timeout=30):
            if "key=" + GEM1 in url:
                raise CloudError("rate_limit", SHORT_LIMIT, 429)
            return dict(GEMINI_OK)

        be = self.backend(http)
        step = be.step(system="s", messages=[], tools=[], max_tokens=8)
        self.assertEqual(step.text, "second key worked")
        parked = self.entry(GEM1)
        self.assertEqual(parked["status"], "rate_limited")
        self.assertEqual(parked["last_error"], "429 rate limited")
        wait = parked["cooldown_until"] - time.time()
        self.assertGreater(wait, 50)
        self.assertLessEqual(wait, KM.DEFAULT_COOLDOWN + 1)


class _OverviewBackend:
    """Just enough backend for _classify_failure: an overview list."""

    def __init__(self, overview):
        self._overview = overview

    def failure_overview(self):
        return self._overview


class WaitRenderingCase(unittest.TestCase):
    """A quota park lasts hours — the failure line must not print it
    as a raw seconds figure."""

    def classify(self, wait):
        mode = AgentMode.__new__(AgentMode)
        be = _OverviewBackend([{"provider": "gemini", "reason": "cooling",
                                "wait": wait}])
        return mode._classify_failure(be, None)["lines"][0]

    def test_short_wait_stays_seconds(self):
        self.assertEqual(_wait_phrase(45), "(~45 စက္ကန့်)")
        self.assertIn("(~45 စက္ကန့်)", self.classify(45))

    def test_long_wait_reads_in_hours(self):
        self.assertEqual(_wait_phrase(7200), "(~2 နာရီ)")
        self.assertEqual(_wait_phrase(73800), "(~20.5 နာရီ)")
        line = self.classify(73800)
        self.assertIn("(~20.5 နာရီ)", line)
        self.assertNotIn("73800", line)
        self.assertNotIn("စက္ကန့်)", line)


if __name__ == "__main__":
    unittest.main()
