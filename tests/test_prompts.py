"""Tier-aware Agent Mode system prompts.

The contract this pins down:

* ``classify_tier`` reads substrings, not a model list, and checks the fast
  patterns first — "gpt-4o-mini" contains "gpt-4o", and the small model is the
  one that must win;
* ``load_prompt`` composes ``prompts/_base.txt`` with one tier file and falls
  back to ``AGENT_SYSTEM`` rather than to no prompt at all;
* the tier is resolved **once per turn** — a turn that falls back to the phone
  keeps the prompt it started with;
* a tier pinned in Settings wins over AI_AGENT_PROMPT_TIER, and the reported
  source says which one actually decided.

Run:  sh tests/run.sh -p "test_prompts.py"
"""
from __future__ import annotations

import os
import unittest
from contextlib import closing

from _harness import bootstrap, get_app

MODEL = bootstrap()
app = get_app()
client = app.test_client()

import config  # noqa: E402
from core import prompts as P  # noqa: E402
from core.key_manager import KeyManager  # noqa: E402
from modes.agent import AGENT_SYSTEM, AgentMode  # noqa: E402


def clear_env():
    os.environ.pop(config.PROMPT_TIER_ENV, None)


def fresh_store() -> KeyManager:
    """A key file with no tier in it — the state a new install is in."""
    km = KeyManager()
    if km.path.exists():
        km.path.unlink()
    return km


def store_tier(value: str) -> KeyManager:
    km = fresh_store()
    if value is not None:
        cfg = km.load()
        cfg["prompt_tier_mode"] = value
        km.save()
    return km


class ClassifierTests(unittest.TestCase):
    def test_local_backend_is_always_local(self):
        for model in ("", "gpt-4o", "claude-opus-5", "gemma-4b"):
            self.assertEqual(P.classify_tier("local", model), "local")

    def test_fast_names(self):
        for model in ("gpt-4o-mini", "gemini-2.5-flash", "claude-3-5-haiku-latest",
                      "llama-3.1-8b-instruct", "qwen3-4b", "nova-nano-v2",
                      "gpt-4.1-nano", "some-model-lite", "tinyair-3b",
                      "gpt-oss-20b-small"):
            self.assertEqual(P.classify_tier("cloud", model), "cloud_fast", model)

    def test_strong_names(self):
        for model in ("gpt-4o", "gpt-4.1", "gpt-5", "claude-opus-4",
                      "claude-sonnet-4-5", "claude-3-5-sonnet-latest",
                      "gemini-2.5-pro", "deepseek-reasoner", "o3", "gpt-4-turbo"):
            self.assertEqual(P.classify_tier("cloud", model), "cloud_strong", model)

    def test_fast_beats_strong(self):
        # "gpt-4o-mini" contains "gpt-4o"; the small model is the right one.
        self.assertEqual(P.classify_tier("cloud", "gpt-4o-mini"), "cloud_fast")
        self.assertEqual(P.classify_tier("cloud", "gpt-4.1-mini"), "cloud_fast")
        # "o3-mini" is the same shape of trap, from the other side.
        self.assertEqual(P.classify_tier("cloud", "o3-mini"), "cloud_fast")

    def test_unknown_cloud_is_the_safe_default(self):
        for model in ("", "some-new-model-2099", "llama-99", "mystery-1"):
            self.assertEqual(P.classify_tier("cloud", model), "cloud_fast", model)

    def test_case_and_whitespace_do_not_matter(self):
        self.assertEqual(P.classify_tier("  LOCAL ", " GPT-4O "), "local")
        self.assertEqual(P.classify_tier("cloud", "  GPT-4O-MINI "), "cloud_fast")


class LoaderTests(unittest.TestCase):
    def setUp(self):
        P.clear_cache()

    def test_every_tier_loads_its_own_file(self):
        for tier in P.TIERS:
            text = P.load_prompt(tier)
            self.assertIn("Burmese", text, tier)
            with open(P.PROMPTS_DIR / ("agent_%s.txt" % tier),
                      encoding="utf-8") as fh:
                body = fh.read().strip()
            self.assertIn(body, text, tier)

    def test_base_is_in_every_prompt(self):
        with open(P.PROMPTS_DIR / P.BASE_NAME, encoding="utf-8") as fh:
            base = fh.read().strip()
        for tier in P.TIERS:
            self.assertIn(base, P.load_prompt(tier), tier)

    def test_unknown_tier_falls_back_to_agent_system(self):
        self.assertEqual(P.load_prompt("cloud_medium"), AGENT_SYSTEM)
        self.assertEqual(P.load_prompt(""), AGENT_SYSTEM)

    def test_traversal_is_impossible(self):
        # The whitelist is checked before the tier is ever part of a path.
        for evil in ("../../etc/passwd", "..%2f..%2fetc", "/etc/passwd",
                     "local/../../secret"):
            self.assertEqual(P.load_prompt(evil), AGENT_SYSTEM, evil)

    def test_missing_file_falls_back(self):
        saved = P.PROMPTS_DIR
        try:
            P.PROMPTS_DIR = saved.parent / "no-such-prompts-dir"
            P.clear_cache()
            self.assertEqual(P.load_prompt("local"), AGENT_SYSTEM)
        finally:
            P.PROMPTS_DIR = saved
            P.clear_cache()

    def test_empty_file_falls_back(self):
        saved = P.PROMPTS_DIR
        tmp = saved.parent / "prompts-test-empty"
        tmp.mkdir(exist_ok=True)
        try:
            (tmp / P.BASE_NAME).write_text("base only\n", encoding="utf-8")
            (tmp / "agent_local.txt").write_text("   \n", encoding="utf-8")
            P.PROMPTS_DIR = tmp
            P.clear_cache()
            # Whitespace is not a prompt: the half-deployed folder degrades to
            # the prompt that worked before tiers existed.
            self.assertEqual(P.load_prompt("local"), AGENT_SYSTEM)
        finally:
            P.PROMPTS_DIR = saved
            P.clear_cache()

    def test_cache_returns_the_same_object(self):
        self.assertIs(P.load_prompt("local"), P.load_prompt("local"))

    def test_prompts_fit_their_budget(self):
        # 2026-10-10: budgets re-tuned after the Burmese-native rewrite.
        # Burmese tokenizes dense (~1.5 chars/token), so the same rules cost
        # more tokens than the old English prompts. Caps keep ~10% headroom
        # so minor wording edits do not fail, but runaway growth still does.
        budgets = {"cloud_strong": 1050, "cloud_fast": 1150, "local": 950}
        for tier, cap in budgets.items():
            n = P.estimate_tokens(P.load_prompt(tier))
            self.assertLessEqual(n, cap, "%s is %d tokens (budget %d)"
                                 % (tier, n, cap))
        base = P.estimate_tokens(
            (P.PROMPTS_DIR / P.BASE_NAME).read_text(encoding="utf-8"))
        self.assertLessEqual(base, 700, "_base.txt is %d tokens" % base)


class ResolveTests(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.get(config.PROMPT_TIER_ENV)
        clear_env()
        store_tier(None)

    def tearDown(self):
        if self._env is None:
            clear_env()
        else:
            os.environ[config.PROMPT_TIER_ENV] = self._env

    def test_auto_uses_the_classifier(self):
        self.assertEqual(P.resolve_tier("cloud", "gpt-4o"), ("cloud_strong", "auto"))
        self.assertEqual(P.resolve_tier("local", "gemma"), ("local", "auto"))

    def test_settings_win_over_env(self):
        store_tier("local")
        os.environ[config.PROMPT_TIER_ENV] = "cloud_strong"
        self.assertEqual(config.prompt_tier_mode(), "local")
        self.assertEqual(P.resolve_tier("cloud", "gpt-4o", mode="local"),
                         ("local", "settings"))

    def test_env_is_used_when_settings_are_silent(self):
        os.environ[config.PROMPT_TIER_ENV] = "cloud_strong"
        self.assertEqual(config.prompt_tier_mode(), "cloud_strong")
        self.assertEqual(
            P.resolve_tier("local", "gemma", mode=config.prompt_tier_mode()),
            ("cloud_strong", "env"))

    def test_settings_auto_still_beats_env(self):
        # An explicit "Auto" in Settings is a decision, not an absence.
        store_tier("auto")
        os.environ[config.PROMPT_TIER_ENV] = "cloud_strong"
        self.assertEqual(config.prompt_tier_mode(), "auto")
        self.assertEqual(P.resolve_tier("cloud", "gpt-4o", mode="auto"),
                         ("cloud_strong", "auto"))

    def test_junk_falls_back_to_auto(self):
        store_tier("cloud_medium")
        clear_env()
        self.assertEqual(config.prompt_tier_mode(), "auto")
        os.environ[config.PROMPT_TIER_ENV] = "nonsense"
        self.assertEqual(config.prompt_tier_mode(), "auto")

    def test_describe_reports_what_the_next_turn_uses(self):
        view = P.describe("cloud", "gpt-4o-mini")
        self.assertEqual(view["mode"], "auto")
        self.assertEqual(view["resolved"], "cloud_fast")
        self.assertEqual(view["source"], "auto")
        self.assertEqual(view["model"], "gpt-4o-mini")


class UsageTests(unittest.TestCase):
    def tearDown(self):
        from modes import agent as A
        A._USAGE.update({"tier": "", "tier_source": ""})

    def test_record_tier_is_kept_across_usage_updates(self):
        from modes import agent as A
        A.record_tier("cloud_strong", "settings")
        A.record_usage("cloud", "gpt-4o", "openai", "auto")
        self.assertEqual(A.last_usage()["tier"], "cloud_strong")
        self.assertEqual(A.last_usage()["tier_source"], "settings")
        self.assertEqual(A.last_usage()["model"], "gpt-4o")

    def test_usage_defaults_carry_the_new_keys(self):
        from modes import agent as A
        A._USAGE.update({"tier": "", "tier_source": ""})
        self.assertIn("tier", A.last_usage())
        self.assertIn("tier_source", A.last_usage())


class TurnBoundaryTests(unittest.TestCase):
    """The tier is decided once per turn and then frozen."""

    def setUp(self):
        self._env = os.environ.get(config.PROMPT_TIER_ENV)
        clear_env()
        store_tier("auto")
        P.clear_cache()
        self.mode = AgentMode(None, None)

    def tearDown(self):
        if self._env is not None:
            os.environ[config.PROMPT_TIER_ENV] = self._env
        else:
            clear_env()
        store_tier(None)

    def test_prompt_is_frozen_for_the_turn(self):
        self.mode.begin_turn()
        first_prompt, first_tier = self.mode._system_prompt()
        # The user changes the setting mid-turn — nothing about this turn moves.
        store_tier("cloud_strong")
        self.assertEqual(self.mode._system_prompt(), (first_prompt, first_tier))
        # The next turn picks it up.
        self.mode.begin_turn()
        _, tier = self.mode._system_prompt()
        self.assertEqual(tier, "cloud_strong")

    def test_fallback_keeps_the_prompt_it_started_with(self):
        self.mode.begin_turn()
        prompt = self.mode._system_prompt()[0]
        # Cloud dies and the route answers locally: same prompt, no re-resolve.
        self.assertEqual(self.mode._system_prompt()[0], prompt)
        self.assertIn("Burmese", prompt)

    def test_tier_is_recorded_for_the_status_bar(self):
        from modes import agent as A
        A._USAGE.update({"tier": "", "tier_source": ""})
        store_tier("local")
        self.mode.begin_turn()
        _, tier = self.mode._system_prompt()
        self.assertEqual(tier, "local")
        self.assertEqual(A.last_usage()["tier"], "local")
        self.assertEqual(A.last_usage()["tier_source"], "settings")


class SettingsApiTests(unittest.TestCase):
    def setUp(self):
        clear_env()
        store_tier(None)

    def tearDown(self):
        clear_env()

    def call(self, method, path, body=None, expect=None):
        fn = getattr(client, method)
        kw = {"json": body} if body is not None else {}
        with closing(fn(path, **kw)) as r:
            data = r.get_json()
            if expect is not None:
                self.assertEqual(r.status_code, expect,
                                 "%s %s -> %s" % (method, path, r.status_code))
            return data

    def test_get_reports_the_tier_it_would_use(self):
        j = self.call("get", "/api/settings/agent", expect=200)
        self.assertIn("prompt_tier", j)
        view = j["prompt_tier"]
        for field in ("mode", "resolved", "source", "model"):
            self.assertIn(field, view)
        self.assertIn(view["resolved"], P.TIERS)
        self.assertIn(j["prompt_tier_modes"][0], config.PROMPT_TIER_MODES)
        # No key was added, so this turn is the on-device one.
        self.assertEqual(view["backend"], "local")
        self.assertEqual(view["resolved"], "local")
        self.assertFalse(view["mismatch"])

    def test_post_pins_a_tier_and_survives_a_reload(self):
        j = self.call("post", "/api/settings/agent",
                      {"prompt_tier_mode": "cloud_fast"}, expect=200)
        self.assertIn("prompt_tier_mode", j["changed"])
        self.assertEqual(j["prompt_tier"]["mode"], "cloud_fast")
        self.assertEqual(j["prompt_tier"]["resolved"], "cloud_fast")
        self.assertEqual(j["prompt_tier"]["source"], "settings")
        # Written to disk, not just echoed back.
        again = self.call("get", "/api/settings/agent", expect=200)
        self.assertEqual(again["config"]["prompt_tier_mode"], "cloud_fast")
        self.assertEqual(config.prompt_tier_mode(), "cloud_fast")
        # …and a pinned tier on a local turn is flagged as a mismatch.
        self.assertTrue(again["prompt_tier"]["mismatch"])

    def test_post_rejects_a_tier_that_does_not_exist(self):
        j = self.call("post", "/api/settings/agent",
                      {"prompt_tier_mode": "../../etc/passwd"})
        self.assertIn("error", j)
        self.assertIn("prompt_tier_mode", j["error"])
        self.call("get", "/api/settings/agent", expect=200)

    def test_auto_saves_as_auto(self):
        store_tier("local")
        self.call("post", "/api/settings/agent", {"prompt_tier_mode": "auto"},
                  expect=200)
        self.assertEqual(config.prompt_tier_mode(), "auto")


if __name__ == "__main__":
    unittest.main()