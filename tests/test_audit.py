"""Log-stats regex in tools.audit must not invent errors from JSON bodies.

Session 2026-10-10: self_audit reported E=5 on the phone log tail while the
file contained zero bare ERROR lines. Every hit was either a JSON key
(`"error":` inside a 402 WARNING body) or a reflection note that quoted
`"Error: duplicate fact"`. The bare-token ERROR lines the logger itself
writes must still count.
"""
from __future__ import annotations

import unittest

from _harness import bootstrap

bootstrap()

from tools.audit import _ERR_RE, _WARN_RE  # noqa: E402


class ErrRegexTests(unittest.TestCase):
    def test_bare_error_tokens_still_count(self):
        for line in (
            "2026-10-10 ERROR   [MainThread] aiagent.server: boom",
            "2026-10-10 CRITICAL [MainThread] x",
            "2026-10-10 FATAL [MainThread] y",
            "2026-10-10 EXCEPTION [MainThread] z",
            "2026-10-10 TRACEBACK (most recent call last):",
        ):
            self.assertIsNotNone(_ERR_RE.search(line), msg=repr(line))

    def test_json_error_key_is_not_an_error(self):
        line = (
            'WARNING openrouter step failed (bad_response, status 402): '
            '{"error":{"message":"This request requires more credits"}}'
        )
        self.assertIsNone(_ERR_RE.search(line))
        # …but the WARNING itself still counts as a warning.
        self.assertIsNotNone(_WARN_RE.search(line))

    def test_quoted_error_message_is_not_an_error(self):
        line = (
            'reflect saved: The memory_write tool will return an '
            '"Error: duplicate fact" message'
        )
        self.assertIsNone(_ERR_RE.search(line))

    def test_e_slash_still_counts(self):
        # The compact syslog form "E/aiagent" / "W/aiagent" both scanners accept.
        self.assertIsNotNone(_ERR_RE.search("E/aiagent memory save failed"))
        self.assertIsNotNone(_WARN_RE.search("W/aiagent soft warning"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
