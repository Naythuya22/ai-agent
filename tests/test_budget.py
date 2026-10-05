"""Budget awareness (Agent Mode).

``AgentMode._budget_hint()`` is one line of context/iteration state, and
``_system_prompt()`` appends it to the frozen prompt when
``config.AGENT_BUDGET_HINT`` is on. With the flag off the prompt must be
byte-for-byte what it was before this feature existed — that is what the
`_unchanged_when_flag_off` test pins down.

Everything here runs on a bare ``AgentMode.__new__`` with ``_backend_obj``
stubbed: no session file, no backend build, no socket, no weights.

Run:  sh tests/run.sh -p "test_budget.py"
"""
from __future__ import annotations

import unittest

from _harness import bootstrap

bootstrap()

import config  # noqa: E402
from core import prompts as P  # noqa: E402
from modes import agent as agent_mod  # noqa: E402
from modes.agent import AgentMode  # noqa: E402


class _FakeBackend:
    """The one member the hint reads: ``context_state()`` (or its failure)."""

    def __init__(self, pct=0, raises=False):
        self.pct = pct
        self.raises = raises

    def context_state(self):
        if self.raises:
            raise RuntimeError("context state unreadable")
        return {"pct": self.pct}


def _bare_mode(turn_iter=0, backend=None) -> AgentMode:
    """An AgentMode with no ``__init__`` — the shape the other suites use.

    ``_backend_obj`` is stubbed so a test can say exactly what context state
    the hint is looking at, and so no real backend is ever built.
    """
    m = AgentMode.__new__(AgentMode)
    m._messages = []
    m._turn_cache = {}
    m._turn_cache_order = []
    m._turn_iter = turn_iter
    m._backend_obj = lambda: backend
    return m


def _hint(turn_iter=0, backend=None) -> str:
    return _bare_mode(turn_iter=turn_iter, backend=backend)._budget_hint()


def _expected(pct: int, used: int, maxi: int) -> str:
    """The note the code is contracted to build, spelled out here."""
    return "[BUDGET: ctx %d%% iter %d/%d]" % (pct, used, maxi)


class BudgetHintTests(unittest.TestCase):
    """``_budget_hint()`` on its own — flag, numbers and warnings."""

    def setUp(self):
        self._saved = config.AGENT_BUDGET_HINT
        config.AGENT_BUDGET_HINT = True      # the state the suite assumes

    def tearDown(self):
        config.AGENT_BUDGET_HINT = self._saved

    # ---------- the flag ----------

    def test_budget_hint_off_returns_empty(self):
        """Off is empty even when every number would have warned."""
        config.AGENT_BUDGET_HINT = False
        m = _bare_mode(turn_iter=999, backend=_FakeBackend(pct=99))
        self.assertEqual(m._budget_hint(), "")

    def test_budget_hint_on_returns_a_note(self):
        m = _bare_mode(backend=_FakeBackend(pct=10))
        self.assertEqual(m._budget_hint(), _expected(10, 0,
                                                     config.agent_max_iter("agent")))

    # ---------- the two numbers ----------

    def test_budget_hint_includes_ctx_pct(self):
        m = _bare_mode(backend=_FakeBackend(pct=42))
        self.assertEqual(m._budget_hint(),
                         _expected(42, 0, config.agent_max_iter("agent")))
        self.assertIn("ctx 42%", m._budget_hint())

    def test_budget_hint_includes_iter_count(self):
        maxi = config.agent_max_iter("agent")
        self.assertEqual(_hint(turn_iter=3, backend=_FakeBackend(pct=0)),
                         _expected(0, 3, maxi))
        self.assertIn("iter 3/%d" % maxi, _hint(turn_iter=3))
        # a turn that has not spent anything yet says so
        self.assertIn("iter 0/%d" % maxi, _hint())

    # ---------- the two warnings ----------

    def test_budget_hint_warns_when_ctx_high(self):
        warn = config.AGENT_BUDGET_WARN_CTX
        # exactly at the line already counts as high
        self.assertIn("be concise", _hint(backend=_FakeBackend(pct=warn)))
        self.assertIn("be concise", _hint(backend=_FakeBackend(pct=100)))
        if warn > 0:
            self.assertNotIn("be concise",
                             _hint(backend=_FakeBackend(pct=warn - 1)))
        # the warning is an addition, never a replacement
        note = _hint(backend=_FakeBackend(pct=100))
        self.assertTrue(note.startswith("[BUDGET: ctx 100%"), note)

    def test_budget_hint_warns_when_iters_almost_used(self):
        maxi = config.agent_max_iter("agent")
        warn = config.AGENT_BUDGET_WARN_ITER
        self.assertGreater(maxi, warn, "the cap must leave room to warn")
        # one call left, exactly at the threshold, and past the cap
        for used in (maxi - 1, maxi - warn, maxi):
            with self.subTest(used=used):
                self.assertIn("wrap up soon",
                              _hint(turn_iter=used, backend=_FakeBackend()))
        # a fresh turn is nowhere near the line
        self.assertNotIn("wrap up soon", _hint(turn_iter=0))
        self.assertNotIn("wrap up soon", _hint(turn_iter=maxi - warn - 1))

    # ---------- things that break in the field ----------

    def test_budget_hint_survives_missing_backend(self):
        # no backend object at all
        self.assertEqual(_hint(backend=None),
                         _expected(0, 0, config.agent_max_iter("agent")))
        # a backend whose context state raises
        boom = _hint(backend=_FakeBackend(raises=True))
        self.assertTrue(boom.startswith("[BUDGET: ctx 0%"), boom)
        # a backend with no context_state() member
        memberless = _hint(backend=object())
        self.assertTrue(memberless.startswith("[BUDGET: ctx 0%"), memberless)
        # nothing stubbed and no _turn_iter attribute either: a bare
        # __new__ object must not raise
        bare = AgentMode.__new__(AgentMode)
        self.assertTrue(bare._budget_hint().startswith("[BUDGET: ctx 0%"))
        self.assertIn("iter 0/", bare._budget_hint())


class SystemPromptHintTests(unittest.TestCase):
    """The hint inside ``_system_prompt()`` — appended, or absent when off."""

    def setUp(self):
        self._saved = config.AGENT_BUDGET_HINT
        config.AGENT_BUDGET_HINT = True

    def tearDown(self):
        config.AGENT_BUDGET_HINT = self._saved
        # _system_prompt() resolves a tier into the status-bar record on every
        # call; leave the module as it was found.
        agent_mod._USAGE.update({"tier": "", "tier_source": ""})

    def test_system_prompt_appends_budget_hint(self):
        m = _bare_mode(backend=_FakeBackend(pct=42))
        m.begin_turn()
        prompt, tier = m._system_prompt()
        self.assertIn(tier, P.TIERS)
        want = _expected(42, 0, config.agent_max_iter(m.name))
        self.assertTrue(prompt.endswith("\n\n" + want), prompt[-160:])
        # the tier prompt itself is untouched underneath the suffix
        self.assertTrue(prompt.startswith(P.load_prompt(tier)))

    def test_system_prompt_unchanged_when_flag_off(self):
        config.AGENT_BUDGET_HINT = False
        m = _bare_mode(backend=_FakeBackend(pct=99))
        m.begin_turn()
        prompt, tier = m._system_prompt()
        self.assertNotIn("[BUDGET", prompt)
        self.assertEqual(prompt, P.load_prompt(tier))

    def test_system_prompt_still_freezes_with_the_hint(self):
        """The hint must not become a second, movable part of the prompt."""
        m = _bare_mode(backend=_FakeBackend(pct=42))
        m.begin_turn()
        first = m._system_prompt()
        m._turn_iter = 9                        # the loop runs mid-turn
        self.assertEqual(m._system_prompt(), first)


class TurnIterTests(unittest.TestCase):
    """``_turn_iter`` lives with the other per-turn state."""

    def test_turn_iter_resets_on_begin_turn(self):
        m = AgentMode.__new__(AgentMode)
        m._turn_iter = 5
        m.begin_turn()
        self.assertEqual(m._turn_iter, 0)
        # it is a plain counter between resets
        m._turn_iter = 3
        self.assertEqual(m._turn_iter, 3)
        m.begin_turn()
        self.assertEqual(m._turn_iter, 0)

    def test_turn_iter_exists_on_a_constructed_mode(self):
        mode = AgentMode(None, None)
        self.assertEqual(mode._turn_iter, 0)
        mode._turn_iter = 2
        mode.begin_turn()
        self.assertEqual(mode._turn_iter, 0)


if __name__ == "__main__":
    unittest.main()
