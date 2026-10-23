"""Release 28 — the level-up bundle (tester evaluation round 3, A–I).

A. The status key label names only an ENABLED provider's HEALTHY key.
B. Media models (Lyria music, *-image generation, Veo, Voxtral) no
   longer classify as chat; OpenRouter's modality output side rules.
C. web_search falls back once (Wikipedia) when DDG fails, and no raw
   HTTP status code reaches a tool result any more.
D. Mid-turn failure summaries quote the latest read-only tool result.
E. Fragment facts ("User prefers answers to be in") are rejected.
F. The no-key guidance explains "API key" and names the cost-ack step.
G. Fresh installs start with run_command disabled; existing installs
   (no stored value) keep it enabled; the stamp never rewrites a file.
H. A no-balance 402 stating an affordable count earns ONE retry at a
   reduced max_tokens — only when the count covers prompt + 256.
I. Steering hints are UI-text only — pinned in the JS suites
   (ui_model_info_test.js / ui_settings_test.js), not here.

Run:  sh tests/run.sh -p "test_release28.py"
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from _harness import bootstrap

bootstrap()

from core import agent_backend as AB  # noqa: E402
from core import model_discovery as MD  # noqa: E402
from core import safety  # noqa: E402
from core.agent_backend import AgentBackend, BackendExhausted  # noqa: E402
from core.key_manager import KeyManager, mask  # noqa: E402
from core.search.base import Result  # noqa: E402
from memory import Memory  # noqa: E402
from modes.agent import AgentMode  # noqa: E402
from routes._helpers import active_key_label  # noqa: E402
import importlib  # noqa: E402

# tools/__init__ re-exports the smart_search FUNCTION under the same
# name as the submodule, so the module itself needs importlib.
ss_mod = importlib.import_module("tools.smart_search")  # noqa: E402
from tools import shell as shell_tool  # noqa: E402
from tools import web_search as ws  # noqa: E402

GEM_KEY = "AIza-r28-gemini-aaaaaaaaaaaa"
DUMMY_KEY = "sk-r28-dummy-000000000000"
DEAD_KEY = "sk-r28-dead-1111111111111"
COOL_KEY = "sk-r28-cool-2222222222222"
PIN_KEY_A = "sk-r28-pin-a-333333333333"
PIN_KEY_B = "sk-r28-pin-b-444444444444"


def wiped_km() -> KeyManager:
    """A KeyManager on the harness default path, starting empty."""
    km = KeyManager()
    if km.path.exists():
        km.path.unlink()
    km.invalidate()
    return km


class KeyLabelCase(unittest.TestCase):
    """(A) Only an enabled provider with a healthy key may be labelled."""

    def setUp(self):
        self.km = wiped_km()

    def tearDown(self):
        wiped_km()

    def _disable(self, provider):
        cfg = self.km.load()
        cfg["providers"][provider]["enabled"] = False
        self.km.save()

    def test_disabled_providers_dummy_key_is_never_labelled(self):
        # The round-3 repro: openai sorts first in the fallback order
        # and holds only a dummy key on a DISABLED provider, while
        # gemini actually serves.
        self.km.add_key("openai", DUMMY_KEY, "dummy")
        self._disable("openai")
        self.km.add_key("gemini", GEM_KEY, "gem")
        self.assertEqual(active_key_label(),
                         "Key: %s (Active)" % mask(GEM_KEY))

    def test_dead_key_is_never_labelled_active(self):
        e = self.km.add_key("openai", DEAD_KEY, "dead")
        self.km.mark_auth_failed("openai", e["id"], "401")
        self.assertEqual(active_key_label(), "")

    def test_cooling_key_is_not_labelled_active(self):
        e = self.km.add_key("openai", COOL_KEY, "cool")
        self.km.mark_rate_limited("openai", e["id"], retry_after=3600)
        self.assertEqual(active_key_label(), "")

    def test_pinned_healthy_key_wins_within_its_provider(self):
        self.km.add_key("gemini", PIN_KEY_A, "a")
        b = self.km.add_key("gemini", PIN_KEY_B, "b")
        self.km.set_active_key("gemini", b["id"])
        self.assertEqual(active_key_label(),
                         "Key: %s (Active)" % mask(PIN_KEY_B))


class ModelKindCase(unittest.TestCase):
    """(B) Media models are not chat rows; modality output side rules."""

    @staticmethod
    def _kinds(entries):
        out = MD.normalise("openrouter", {"data": entries})
        return {m["id"]: m["kind"] for m in out}

    def test_modality_output_side_decides(self):
        kinds = self._kinds([
            {"id": "google/lyria-3-pro-preview",
             "architecture": {"modality": "text->audio"}},
            {"id": "google/gemini-3-pro-image",
             "architecture": {"modality": "text+image->image"}},
            {"id": "google/veo-3.1",
             "architecture": {"modality": "text->video"}},
            {"id": "acme/embed-1",
             "architecture": {"modality": "text->embedding"}},
            {"id": "acme/vision-chat-1",
             "architecture": {"modality": "text+image->text"}},
            {"id": "acme/plain-chat-1",
             "architecture": {"modality": "text->text"}},
        ])
        self.assertEqual(kinds["google/lyria-3-pro-preview"], "audio")
        self.assertEqual(kinds["google/gemini-3-pro-image"], "image")
        self.assertEqual(kinds["google/veo-3.1"], "other")
        self.assertEqual(kinds["acme/embed-1"], "embed")
        self.assertEqual(kinds["acme/vision-chat-1"], "chat")
        self.assertEqual(kinds["acme/plain-chat-1"], "chat")

    def test_transcription_modality_keeps_id_say(self):
        # audio->text: the output side is text, so modality has no
        # verdict — the id heuristic (whisper ⇒ audio) still applies.
        kinds = self._kinds([
            {"id": "openai/whisper-1",
             "architecture": {"modality": "audio->text"}},
        ])
        self.assertEqual(kinds["openai/whisper-1"], "audio")

    def test_id_hints_cover_the_media_families(self):
        self.assertEqual(MD.kind_from_id("google/lyria-3-clip-preview"),
                         "audio")
        self.assertEqual(MD.kind_from_id("mistralai/voxtral-small-24b"),
                         "audio")
        self.assertEqual(MD.kind_from_id("openai/gpt-5-image"), "image")
        self.assertEqual(MD.kind_from_id("google/gemini-2.5-flash-image"),
                         "image")
        self.assertEqual(MD.kind_from_id("black-forest/flux-pro"), "image")
        self.assertEqual(MD.kind_from_id("google/veo-3"), "other")
        self.assertEqual(MD.kind_from_id("acme/suno-music-1"), "audio")

    def test_unknown_ids_stay_chat(self):
        self.assertEqual(MD.kind_from_id("acme/brand-new-chat-9000"),
                         "chat")
        self.assertEqual(MD.kind_from_id("qwen/qwen3.8-27b"), "chat")


_DDG_HTML = (
    "<table><tr><td><a class='result-link' href='https://x.dev/p'>"
    "X Dev Page</a></td></tr>"
    "<tr><td class='result-snippet'>asyncio snippet</td></tr></table>")


class SearchResilienceCase(unittest.TestCase):
    """(C) One Wikipedia fallback behind DDG; no raw HTTP statuses."""

    def setUp(self):
        self._saved = {
            "_http_get": ws._http_get,
            "_fetch_page": ws._fetch_page,
            "offline_notice": ws.offline_notice,
        }
        ws.offline_notice = lambda *a, **k: None
        # The chain's HTML engines (Bing/Brave) fail by default here —
        # these tests are about the DDG↔Wikipedia ends of the chain;
        # test_search_chain.py covers the middle. Cooldown memory is
        # cleared so one test's failure cannot skip a source in the
        # next.
        ws._fetch_page = lambda *a, **k: ws._Response(202, [], b"")
        ws._down_until.clear()
        from core.search import backends as backends_mod
        self._wiki_search = backends_mod.WikipediaBackend.search
        backends_mod.WikipediaBackend.search = (
            lambda self_, q, max_results=3: [])

    def tearDown(self):
        ws._http_get = self._saved["_http_get"]
        ws._fetch_page = self._saved["_fetch_page"]
        ws.offline_notice = self._saved["offline_notice"]
        ws._down_until.clear()
        from core.search import backends as backends_mod
        backends_mod.WikipediaBackend.search = self._wiki_search

    def _wiki(self, results):
        from core.search import backends as backends_mod
        backends_mod.WikipediaBackend.search = (
            lambda self_, q, max_results=3: results)

    def test_ddg_block_falls_back_to_wikipedia(self):
        ws._http_get = lambda *a, **k: ws._Response(202, [], b"")
        self._wiki([Result(title="Yangon", url="https://en.wikipedia.org/wiki/Yangon",
                           snippet="Largest city.")])
        out = ws.web_search("yangon", 3)
        self.assertIn("Yangon", out)
        self.assertIn("Wikipedia", out)
        self.assertIn("<untrusted>", out)
        self.assertNotIn("202", out)

    def test_both_down_is_an_honest_error_without_a_status(self):
        ws._http_get = lambda *a, **k: ws._Response(202, [], b"")
        out = ws.web_search("yangon", 3)
        self.assertTrue(out.startswith("Error: search is unavailable"))
        self.assertNotIn("202", out)

    def test_network_failure_also_falls_back(self):
        def boom(*a, **k):
            raise ws.NetworkError("URLError: timed out")
        ws._http_get = boom
        self._wiki([Result(title="Naypyidaw",
                           url="https://en.wikipedia.org/wiki/Naypyidaw",
                           snippet="Capital.")])
        out = ws.web_search("capital of myanmar", 3)
        self.assertIn("Naypyidaw", out)

    def test_ddg_success_path_unchanged(self):
        ws._http_get = lambda *a, **k: ws._Response(
            200, [], _DDG_HTML.encode("utf-8"))
        out = ws.web_search("asyncio", 3)
        self.assertIn("X Dev Page", out)
        self.assertIn("https://x.dev/p", out)
        self.assertNotIn("Wikipedia", out)

    def test_ddg_empty_still_reports_no_results(self):
        ws._http_get = lambda *a, **k: ws._Response(
            200, [], b"<html><body>nope</body></html>")
        self.assertEqual(ws.web_search("zzz-nothing", 3),
                         "No results found for: zzz-nothing")

    def test_fetch_error_carries_no_status_code(self):
        ws._http_get = lambda *a, **k: ws._Response(503, [], b"")
        out = ws.web_fetch("https://example.com/page")
        self.assertTrue(out.startswith("Error:"))
        self.assertNotIn("503", out)

    def test_smart_search_error_text_scrubs_status_codes(self):
        out = ss_mod._error_text(RuntimeError("HTTP Error 202: blocked"))
        self.assertNotIn("202", out)
        self.assertIn("HTTP error", out)


def _entries(*pairs):
    out = []
    for name, text in pairs:
        out.append({"role": "tool_call", "tool": name, "args": {}})
        out.append({"role": "tool_result", "tool": name, "text": text})
    return out


class FailureQuoteCase(unittest.TestCase):
    """(D) The summary quotes the latest successful read-only result."""

    def setUp(self):
        self.mode = AgentMode.__new__(AgentMode)

    def test_memory_read_result_is_quoted(self):
        out = self.mode._work_summary(_entries(
            ("memory_read", "User prefers coffee with no sugar.")))
        self.assertIn("memory_read", out)
        self.assertIn("User prefers coffee with no sugar.", out)

    def test_write_result_is_never_quoted(self):
        out = self.mode._work_summary(_entries(
            ("write_file", "Wrote secret-draft.txt with DRAFT_BODY_XYZ")))
        self.assertNotIn("DRAFT_BODY_XYZ", out)
        self.assertNotIn("ဖတ်လို့ ရခဲ့တဲ့ ရလဒ်", out)

    def test_shell_result_is_never_quoted(self):
        out = self.mode._work_summary(_entries(
            ("run_command", "SHELL_OUTPUT_XYZ")))
        self.assertNotIn("SHELL_OUTPUT_XYZ", out)

    def test_latest_read_only_result_wins(self):
        out = self.mode._work_summary(_entries(
            ("read_file", "FIRST_RESULT"),
            ("memory_read", "SECOND_RESULT")))
        self.assertIn("SECOND_RESULT", out)
        self.assertNotIn("FIRST_RESULT", out.split("ရလဒ် (")[-1])

    def test_failed_read_is_not_quoted(self):
        out = self.mode._work_summary(_entries(
            ("read_file", "Error: no such file")))
        self.assertNotIn("ဖတ်လို့ ရခဲ့တဲ့ ရလဒ်", out)

    def test_banner_wrapper_is_stripped_from_the_quote(self):
        guarded = (safety.INJECTION_BANNER_OPEN + "\n"
                   "Meeting notes: budget approved.\n"
                   + safety.INJECTION_BANNER_CLOSE)
        out = self.mode._work_summary(_entries(("read_file", guarded)))
        self.assertIn("budget approved", out)
        self.assertNotIn("UNTRUSTED", out)

    def test_quote_is_capped(self):
        out = self.mode._work_summary(_entries(
            ("read_file", "x" * 5000)))
        quote = out.split("ရလဒ် (read_file):\n", 1)[-1]
        self.assertLessEqual(len(quote), AgentMode._QUOTE_CAP + 1)
        self.assertTrue(quote.endswith("…"))


class FragmentFactCase(unittest.TestCase):
    """(E) Half a sentence is not a fact."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="r28-memory-"))
        self.mem = Memory.load(self.tmp / "memory.json")

    def test_the_round3_fragment_is_rejected(self):
        ok, reason = self.mem.add_fact("User prefers answers to be in")
        self.assertFalse(ok)
        self.assertEqual(reason, "fragment")

    def test_other_dangling_endings_are_rejected(self):
        for text in ("User works at the", "User is going to",
                     "User prefers coffee with"):
            ok, reason = self.mem.add_fact(text)
            self.assertFalse(ok, text)
            self.assertEqual(reason, "fragment", text)

    def test_complete_facts_still_save(self):
        for text in ("User prefers answers to be in Burmese",
                     "User prefers coffee with no sugar",
                     "User drinks tea in the morning"):
            ok, reason = self.mem.add_fact(text)
            self.assertTrue(ok, (text, reason))

    def test_burmese_fact_is_untouched_by_the_english_rule(self):
        ok, _ = self.mem.add_fact("အကို ရေသောက်ရတာ ကြိုက်တယ်")
        self.assertTrue(ok)


class GuidanceTextCase(unittest.TestCase):
    """(F) The first-run guide explains the key and the next step."""

    def test_no_key_guidance_explains_what_a_key_is(self):
        self.assertIn("လျှို့ဝှက် သော့", AgentMode.NO_KEY_GUIDANCE)
        self.assertIn("မမျှဝေ", AgentMode.NO_KEY_GUIDANCE)

    def test_no_key_guidance_names_the_cost_ack_step(self):
        self.assertIn("ကုန်ကျစရိတ်", AgentMode.NO_KEY_GUIDANCE)
        self.assertIn("Save", AgentMode.NO_KEY_GUIDANCE)


class ShellGateCase(unittest.TestCase):
    """(G) Beginner default: fresh installs only, existing untouched."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="r28-keys-"))
        self.path = self.tmp / "agent_config.json"

    def test_absent_value_means_legacy_enabled(self):
        km = KeyManager(self.path)
        self.assertTrue(km.shell_enabled())
        self.assertTrue(km.public()["shell_enabled"])

    def test_fresh_install_stamp_disables_shell(self):
        km = KeyManager(self.path)
        self.assertTrue(km.apply_fresh_install_defaults())
        self.assertTrue(self.path.exists())
        stored = json.loads(self.path.read_text("utf-8"))
        self.assertIs(stored["shell_enabled"], False)
        self.assertFalse(km.shell_enabled())
        self.assertFalse(km.public()["shell_enabled"])
        # Second call: the file exists now — no longer a fresh install.
        self.assertFalse(km.apply_fresh_install_defaults())

    def test_existing_config_file_is_byte_identical_after_stamp(self):
        km = KeyManager(self.path)
        km.add_key("openai", "sk-r28-existing-5555555555", "keep")
        before = self.path.read_bytes()
        self.assertFalse(km.apply_fresh_install_defaults())
        self.assertEqual(self.path.read_bytes(), before)
        self.assertTrue(km.shell_enabled())

    def test_reset_preserves_an_explicit_gate(self):
        km = KeyManager(self.path)
        km.apply_fresh_install_defaults()
        km.reset_to_defaults()
        self.assertFalse(km.shell_enabled())

    def test_settings_value_round_trips(self):
        km = KeyManager(self.path)
        cfg = km.load()
        cfg["shell_enabled"] = False
        km.save()
        self.assertFalse(KeyManager(self.path).shell_enabled())
        cfg = km.load()
        cfg["shell_enabled"] = True
        km.save()
        self.assertTrue(KeyManager(self.path).shell_enabled())


class ShellToolGateCase(unittest.TestCase):
    """(G) run_command itself refuses while the gate is off."""

    def setUp(self):
        self.km = wiped_km()

    def tearDown(self):
        wiped_km()

    def test_disabled_install_blocks_run_command(self):
        cfg = self.km.load()
        cfg["shell_enabled"] = False
        self.km.save()
        out = shell_tool.run_command("echo r28-should-not-run")
        self.assertTrue(out.startswith("Error: BLOCKED"), out)
        self.assertIn("disabled", out)

    def test_legacy_install_runs_commands(self):
        out = shell_tool.run_command("echo r28-legacy-ok")
        self.assertEqual(out, "r28-legacy-ok")


_AFFORD_BODY = ('{"error":{"message":"This request requires more '
                'credits, or fewer max_tokens. You requested up to '
                '8192 tokens, but can only afford %d."}}')


class _AffordTransport:
    """Fails with a 402 afford body, then answers; records bodies."""

    def __init__(self, afford, always_fail=False):
        self.afford = afford
        self.always_fail = always_fail
        self.calls = []

    def __call__(self, url, body, headers, timeout=30):
        self.calls.append(body)
        if self.always_fail or len(self.calls) == 1:
            raise AB.CloudError("bad_response",
                                _AFFORD_BODY % self.afford, 402)
        return {"choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3}}


class BalanceRetryCase(unittest.TestCase):
    """(H) The constrained 402 retry, at the guard and end to end."""

    def setUp(self):
        self.km = wiped_km()
        self.km.add_key("openai", "sk-r28-balance-6666666666", "bal")
        cfg = self.km.load()
        cfg["cost_ack"] = True
        self.km.save()

    def tearDown(self):
        wiped_km()

    def _backend(self, transport):
        return AgentBackend(keys=self.km, net_state=lambda: "online",
                            http=transport)

    def test_guard_matrix(self):
        err = AB.CloudError("bad_response", _AFFORD_BODY % 6000, 402)
        self.assertEqual(AB._balance_retry_tokens(err, 2698, 8192), 3302)
        # The probe shape: prompt bigger than the affordable count.
        self.assertIsNone(AB._balance_retry_tokens(err, 6000, 8192))
        low = AB.CloudError("bad_response", _AFFORD_BODY % 241, 402)
        self.assertIsNone(AB._balance_retry_tokens(low, 2698, 8192))
        # Exactly prompt + 256 is the smallest rescuable shape.
        edge = AB.CloudError("bad_response", _AFFORD_BODY % 2954, 402)
        self.assertEqual(AB._balance_retry_tokens(edge, 2698, 8192), 256)
        under = AB.CloudError("bad_response", _AFFORD_BODY % 2953, 402)
        self.assertIsNone(AB._balance_retry_tokens(under, 2698, 8192))
        # A giant balance is not a "reduction" — nothing to retry.
        rich = AB.CloudError("bad_response", _AFFORD_BODY % 999999, 402)
        self.assertIsNone(AB._balance_retry_tokens(rich, 100, 8192))
        # Not a 402 / not the afford shape → no retry.
        rl = AB.CloudError("rate_limit", _AFFORD_BODY % 6000, 429)
        self.assertIsNone(AB._balance_retry_tokens(rl, 100, 8192))
        pl = AB.CloudError("bad_response",
                           "Prompt tokens limit exceeded: 2698 > 2012",
                           402)
        self.assertIsNone(AB._balance_retry_tokens(pl, 100, 8192))

    def test_step_retries_once_with_reduced_max_tokens(self):
        transport = _AffordTransport(6000)
        be = self._backend(transport)
        step = be.step(system="You are helpful.",
                       messages=[{"role": "user", "text": "hi"}],
                       tools=[])
        self.assertEqual(step.text, "ok")
        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(transport.calls[0]["max_tokens"], 8192)
        # The reduction is affordable − estimated prompt tokens; the
        # estimate for this tiny prompt is small but positive. (The
        # published _ctx["used"] is NOT compared here: after a
        # successful call it reflects the provider's actual usage,
        # while the retry was computed from the estimator.)
        spent_on_prompt = 6000 - transport.calls[1]["max_tokens"]
        self.assertGreater(spent_on_prompt, 0)
        self.assertLess(spent_on_prompt, 100)
        self.assertGreaterEqual(transport.calls[1]["max_tokens"], 256)
        self.assertLess(transport.calls[1]["max_tokens"], 8192)

    def test_step_does_not_retry_a_negative_balance_account(self):
        transport = _AffordTransport(241, always_fail=True)
        be = self._backend(transport)
        with self.assertRaises(BackendExhausted) as ctx:
            be.step(system="You are helpful.",
                    messages=[{"role": "user", "text": "hi"}], tools=[])
        self.assertEqual(len(transport.calls), 1)
        reasons = [f["reason"] for f in ctx.exception.failures]
        self.assertIn("no_balance", reasons)

    def test_step_does_not_retry_the_prompt_limit_shape(self):
        class PromptLimitTransport:
            def __init__(self):
                self.calls = 0

            def __call__(self, url, body, headers, timeout=30):
                self.calls += 1
                raise AB.CloudError(
                    "bad_response",
                    "Prompt tokens limit exceeded: 2698 > 2012", 402)

        transport = PromptLimitTransport()
        be = self._backend(transport)
        with self.assertRaises(BackendExhausted):
            be.step(system="You are helpful.",
                    messages=[{"role": "user", "text": "hi"}], tools=[])
        self.assertEqual(transport.calls, 1)

    def test_stream_retries_once_with_reduced_max_tokens(self):
        bodies = []

        def fake_stream(url, body, headers, timeout=30):
            bodies.append(body)
            if len(bodies) == 1:
                raise AB.CloudError("bad_response",
                                    _AFFORD_BODY % 6000, 402)
            yield 'data: {"choices": [{"delta": {"content": "ok"}}]}'
            yield "data: [DONE]"

        be = self._backend(_AffordTransport(6000))
        frames = list(be.step_stream(
            system="You are helpful.",
            messages=[{"role": "user", "text": "hi"}], tools=[],
            http_stream=fake_stream))
        kinds = [k for k, _ in frames]
        self.assertIn("done", kinds)
        self.assertEqual(len(bodies), 2)
        spent_on_prompt = 6000 - bodies[1]["max_tokens"]
        self.assertGreater(spent_on_prompt, 0)
        self.assertLess(spent_on_prompt, 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
