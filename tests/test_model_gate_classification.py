"""Release 27.2 — failure-classification correctness.

Two live-captured misclassifications, both from KoHtet's phone
(2026-10-23) and reproduced on the VM the same day:

  (a) A model GATE read as a key failure. Locked to OpenRouter with
      model thinkingmachines/inkling-small:free, a chat turn is
      refused HTTP 403 (body code 404): the model "is only available
      on agentic harnesses". The key is active and healthy, but the
      adapter maps every 403 to kind "auth", failure_reason() turned
      that into key_invalid, AND _handle_error buried the healthy key
      as dead. The retained detail also rendered as colon-soup
      ("error : message : ... , code :40") because the body was
      truncated mid-JSON before the one-liner could parse it.

  (b) OpenRouter's DAILY free cap read as a short rate limit. With the
      day's 50 free requests spent, chat calls answer 429
      "free-models-per-day" — a phrasing the Release 26 quota detector
      (Gemini shapes) does not match, so the user was told to wait
      ~60 seconds for a cap that resets tomorrow.

The matrix below pins every class with representative payloads and
proves no cross-contamination: gate ⇒ model_unavailable, 401 ⇒
key_invalid, daily caps ⇒ quota_exhausted, short 429 ⇒ rate_limited.

Run:  sh tests/run.sh -p "test_model_gate_classification.py"
"""
from __future__ import annotations

import json
import unittest

from _harness import bootstrap

bootstrap()

from core import agent_backend as AB  # noqa: E402
from core.agent_backend import (  # noqa: E402
    AgentBackend, BackendExhausted, CloudError, failure_reason)
from core.key_manager import KeyManager  # noqa: E402
from modes.agent import AgentMode  # noqa: E402

# Live-captured OpenRouter gate body (HTTP 403, code 404 in the body).
INKLING_BODY = (
    '{"error":{"message":"thinkingmachines/inkling-small:free is only '
    'available on agentic harnesses. Try plugging it into a coding '
    'agent or productivity app listed on '
    'https://openrouter.ai/apps","code":404}}')
INKLING_MESSAGE = (
    "thinkingmachines/inkling-small:free is only available on agentic "
    "harnesses. Try plugging it into a coding agent or productivity "
    "app listed on https://openrouter.ai/apps")
# Live-captured OpenRouter daily free cap (HTTP 429).
OR_FREE_DAY = (
    '{"error":{"message":"Rate limit exceeded: free-models-per-day. '
    'Add 10 credits to unlock 1000 free model requests per '
    'day","code":429}}')
NO_ACCESS_MODEL = (
    '{"error":{"message":"You do not have access to model '
    'acme/private-1.","code":403}}')
MODEL_GONE = (
    '{"error":{"message":"The model `llama-3.3-70b-versatile` does not '
    'exist or you do not have access to it.",'
    '"type":"invalid_request_error","code":"model_not_found"}}')
INVALID_KEY = (
    '{"error":{"message":"Incorrect API key provided. You can find '
    'your API key in the dashboard.","code":401}}')
GEMINI_DAILY = (
    '{"error": {"code": 429, "message": "You exceeded your current '
    'quota, please check your plan and billing details.", "status": '
    '"RESOURCE_EXHAUSTED", "details": [{"quotaId": '
    '"GenerateRequestsPerDayPerProjectPerModel-FreeTier", '
    '"quotaValue": "20"}]}}')
SHORT_LIMIT = ('{"error": {"message": "Rate limit exceeded. Please '
               'retry after a moment."}}')
KEY_THROTTLE = ('{"error":{"message":"quota exceeded for this key, '
                'slow down"}}')
OR_KEY = "sk-or-v1-2727272727272727272727272727272"


class FailureReasonMatrixCase(unittest.TestCase):
    """Errors are built through _status_error — the real adapter
    mapping — so the matrix tests the whole classification path."""

    def test_gate_403_is_model_unavailable(self):
        err = AB._status_error(403, INKLING_BODY, None)
        self.assertEqual(err.kind, "auth")   # the trap, documented
        self.assertEqual(failure_reason(err), "model_unavailable")

    def test_gate_404_is_model_unavailable(self):
        err = AB._status_error(404, INKLING_BODY, None)
        self.assertEqual(failure_reason(err), "model_unavailable")

    def test_no_access_to_model_403_is_model_unavailable(self):
        err = AB._status_error(403, NO_ACCESS_MODEL, None)
        self.assertEqual(failure_reason(err), "model_unavailable")

    def test_model_not_found_stays_model_unavailable(self):
        err = AB._status_error(404, MODEL_GONE, None)
        self.assertEqual(failure_reason(err), "model_unavailable")

    def test_invalid_key_401_stays_key_invalid(self):
        err = AB._status_error(401, INVALID_KEY, None)
        self.assertEqual(failure_reason(err), "key_invalid")

    def test_plain_403_without_model_words_stays_key_invalid(self):
        err = AB._status_error(
            403, '{"error":{"message":"Forbidden for this '
                 'account."}}', None)
        self.assertEqual(failure_reason(err), "key_invalid")

    def test_openrouter_free_day_429_is_quota_exhausted(self):
        err = AB._status_error(429, OR_FREE_DAY, None)
        self.assertEqual(err.kind, "rate_limit")
        self.assertEqual(failure_reason(err), "quota_exhausted")

    def test_gemini_daily_429_stays_quota_exhausted(self):
        err = CloudError("rate_limit", GEMINI_DAILY, status=429)
        self.assertEqual(failure_reason(err), "quota_exhausted")

    def test_short_429_stays_rate_limited(self):
        err = CloudError("rate_limit", SHORT_LIMIT, status=429,
                         retry_after=45)
        self.assertEqual(failure_reason(err), "rate_limited")

    def test_per_key_throttle_stays_rate_limited(self):
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
    """Auto mode composes from the same records — it must agree."""

    def test_gate_line_names_the_model_remedy(self):
        out = classify([record("openrouter", AB._status_error(
            403, INKLING_BODY, None))])
        self.assertEqual(len(out["lines"]), 1)
        line = out["lines"][0]
        self.assertIn("အဲဒီ model ကို အခု မရနိုင်ဘူး", line)
        self.assertIn("Settings မှာ model ပြောင်းရွေးပါ", line)
        self.assertNotIn("API key", line)
        self.assertFalse(out["park_only"])

    def test_free_day_line_says_tomorrow_with_no_short_wait(self):
        out = classify([record("openrouter", AB._status_error(
            429, OR_FREE_DAY, None))])
        line = out["lines"][0]
        self.assertIn("ဒီနေ့ အတွက် quota ကုန်သွားပြီ", line)
        self.assertIn("မနက်ဖြန် ပြန်စမ်းပါ", line)
        self.assertNotIn("(~", line)
        self.assertFalse(out["park_only"])
        self.assertIsNone(out["wait"])


class OnelineExtractionCase(unittest.TestCase):
    def test_inkling_body_gives_the_exact_message(self):
        out = AB._provider_message_oneline(INKLING_BODY)
        self.assertEqual(out, INKLING_MESSAGE)
        self.assertNotIn("{", out)
        self.assertNotIn("error :", out)

    def test_inkling_body_at_the_adapters_400_cap(self):
        # The real pipeline: _status_error keeps 400 chars; step()
        # passes them on; the one-liner must still extract cleanly.
        err = AB._status_error(403, INKLING_BODY, None)
        out = AB._provider_message_oneline(
            AB._sanitize(err.message or "", 400))
        self.assertEqual(out, INKLING_MESSAGE)

    def test_truncated_body_still_extracts_the_message(self):
        # The pre-fix phone shape: body cut at 200 chars, mid-message.
        out = AB._provider_message_oneline(INKLING_BODY[:200])
        self.assertTrue(out.startswith(
            "thinkingmachines/inkling-small:free is only available on "
            "agentic harnesses."))
        self.assertNotIn("error : message", out)
        self.assertNotIn("{", out)
        self.assertNotIn("code :40", out)

    def test_stringified_json_message_is_unwrapped(self):
        body = json.dumps({"error": {"message": json.dumps(
            {"error": {"message": "inner reason: model is gated"}})}})
        out = AB._provider_message_oneline(body)
        self.assertEqual(out, "inner reason: model is gated")

    def test_stringified_detail_is_unwrapped(self):
        body = json.dumps({"error": {"message": json.dumps(
            {"detail": "provider refused this model"})}})
        out = AB._provider_message_oneline(body)
        self.assertEqual(out, "provider refused this model")

    def test_key_material_in_message_stays_masked(self):
        # The pipeline's mask shape is "sk-" + a long alphanumeric run
        # (the contract retained from Release 27) — fixture matches it.
        body = ('{"error":{"message":"auth failed for key '
                'sk-999888777666555444333222111000"}}')
        out = AB._provider_message_oneline(body)
        self.assertNotIn("sk-999888777666555444333222111000", out)
        self.assertIn("auth failed for key", out)


class LockedGateEndToEndCase(unittest.TestCase):
    """Locked step() against the gate: reason, detail and key state."""

    def setUp(self):
        self.km = KeyManager()
        if self.km.path.exists():
            self.km.path.unlink()
        self.km.add_key("openrouter", OR_KEY, "main")
        cfg = self.km.load()
        cfg["cost_ack"] = True
        cfg["auto_switch"] = False
        cfg["pinned_provider"] = "openrouter"
        self.km.save()

    def _step_with(self, err):
        def http_fail(url, body, headers, timeout=30):
            raise err

        be = AgentBackend(keys=self.km, net_state=lambda: "online",
                          http=http_fail)
        with self.assertRaises(AB.BackendExhausted) as ctx:
            be.step(system="s",
                    messages=[{"role": "user", "content": "hi"}],
                    tools=[], max_tokens=16)
        return ctx.exception

    def _key_status(self):
        entries = self.km.entries("openrouter")
        self.assertEqual(len(entries), 1)
        return entries[0].get("status")

    def test_gate_turn_classifies_model_and_spares_the_key(self):
        exc = self._step_with(AB._status_error(403, INKLING_BODY, None))
        self.assertEqual(exc.locked_provider, "openrouter")
        self.assertEqual(exc.locked_reason, "model_unavailable")
        self.assertEqual(exc.locked_detail, INKLING_MESSAGE)
        self.assertNotIn("key_invalid", str(exc))
        self.assertNotIn("error : message", str(exc))
        self.assertNotEqual(self._key_status(), "dead")

    def test_true_401_still_kills_the_key(self):
        exc = self._step_with(AB._status_error(401, INVALID_KEY, None))
        self.assertEqual(exc.locked_reason, "key_invalid")
        self.assertEqual(self._key_status(), "dead")


if __name__ == "__main__":
    unittest.main()
