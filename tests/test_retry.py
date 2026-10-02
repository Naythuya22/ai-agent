"""Retry/backoff — core.retry.

The search tools hit the open internet from a phone, where a single blip used
to mean an empty result set for the whole turn. These tests pin the contract:
transient failures retry, permanent ones do not, Retry-After is respected, and
the whole thing is bounded so a dead link cannot stall a streamed reply.
"""
from __future__ import annotations

import unittest
import urllib.error

from _harness import bootstrap

bootstrap()

from core.retry import RetryError, call_with_retry, should_retry  # noqa: E402


def _http_error(code: int, headers: dict | None = None) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://x.test/", code, "err", headers or {}, None)


class TestShouldRetry(unittest.TestCase):
    def test_transport_failures_are_retryable(self):
        for exc in (ConnectionResetError("x"), TimeoutError("x"),
                    ConnectionRefusedError("x"), OSError("x")):
            self.assertTrue(should_retry(exc), type(exc).__name__)

    def test_rate_limit_and_server_errors_are_retryable(self):
        for code in (408, 429, 500, 502, 503, 504):
            self.assertTrue(should_retry(_http_error(code)), code)

    def test_client_errors_are_permanent(self):
        for code in (400, 401, 403, 404, 410):
            self.assertFalse(should_retry(_http_error(code)), code)

    def test_programming_errors_are_permanent(self):
        for exc in (ValueError("x"), TypeError("x"), KeyError("x")):
            self.assertFalse(should_retry(exc), type(exc).__name__)


class TestCallWithRetry(unittest.TestCase):
    def test_recovers_after_transient_failures(self):
        calls, sleeps = [], []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise ConnectionResetError("blip")
            return "ok"

        out = call_with_retry(flaky, attempts=3, sleep=sleeps.append)
        self.assertEqual(out, "ok")
        self.assertEqual(len(calls), 3, "should have retried twice")
        self.assertEqual(len(sleeps), 2, "one backoff between each attempt")

    def test_backoff_grows_exponentially(self):
        sleeps = []

        def always_down():
            raise ConnectionError("down")

        with self.assertRaises(RetryError):
            call_with_retry(always_down, attempts=4, base_delay=1.0,
                            max_delay=100.0, sleep=sleeps.append, budget=1e9)
        self.assertEqual(len(sleeps), 3)
        self.assertLess(sleeps[0], sleeps[1])
        self.assertLess(sleeps[1], sleeps[2])

    def test_permanent_failure_is_raised_immediately(self):
        calls, sleeps = [], []

        def broken():
            calls.append(1)
            raise ValueError("my bug")

        with self.assertRaises(ValueError):
            call_with_retry(broken, attempts=5, sleep=sleeps.append)
        self.assertEqual(len(calls), 1, "must not retry a programming error")
        self.assertEqual(sleeps, [])

    def test_exhausted_attempts_raise_retry_error_with_cause(self):
        def down():
            raise ConnectionError("still down")

        with self.assertRaises(RetryError) as cm:
            call_with_retry(down, attempts=3, sleep=lambda s: None, what="probe")
        err = cm.exception
        self.assertEqual(err.attempts, 3)
        self.assertIsInstance(err.__cause__, ConnectionError)
        self.assertIn("probe", str(err))

    def test_budget_stops_the_backoff_early(self):
        """A stalled backend must not eat the whole streaming budget."""
        sleeps = []

        def down():
            raise ConnectionError("down")

        with self.assertRaises(RetryError):
            call_with_retry(down, attempts=10, base_delay=5.0, max_delay=30.0,
                            budget=1.0, sleep=sleeps.append)
        self.assertLessEqual(len(sleeps), 1,
                             "budget must cut the retry loop short")

    def test_retry_after_is_used_as_a_floor(self):
        """429 with Retry-After: 5 must not be retried after 0.4s."""
        sleeps = []
        state = {"n": 0}

        def rate_limited():
            state["n"] += 1
            if state["n"] == 1:
                raise _http_error(429, {"Retry-After": "5"})
            return "ok"

        out = call_with_retry(rate_limited, attempts=3, base_delay=0.1,
                              max_delay=30.0, sleep=sleeps.append, budget=1e9)
        self.assertEqual(out, "ok")
        self.assertEqual(len(sleeps), 1)
        self.assertGreaterEqual(sleeps[0], 5.0, "Retry-After must be honoured")

    def test_server_wait_is_capped(self):
        """A hostile Retry-After must not park the request for a minute."""
        sleeps = []
        state = {"n": 0}

        def rate_limited():
            state["n"] += 1
            if state["n"] == 1:
                raise _http_error(429, {"Retry-After": "3600"})
            return "ok"

        call_with_retry(rate_limited, attempts=2, max_delay=6.0,
                        sleep=sleeps.append, budget=1e9)
        self.assertLessEqual(sleeps[0], 6.0 * 1.25)  # cap + jitter headroom

    def test_keyboard_interrupt_is_never_swallowed(self):
        def boom():
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            call_with_retry(boom, attempts=3, sleep=lambda s: None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
