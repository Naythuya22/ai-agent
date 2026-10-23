"""Release 26 — Burmese quality gate on final cloud replies (FIX 1).

Tester evaluation #1/#7: the bare free router rotates models per turn
and many answer in script salad. The gate (core.burmese.reply_quality)
judges the FINAL text; modes.agent._quality_retry re-asks ONCE through
the next provider in the route when it fails. These tests pin:

  * the gate's verdicts (salad fails, clean Burmese passes, Burmese
    with ASCII tech terms passes, short replies unjudged, a
    repetition loop fails — the gpt-oss-20b Groq shape);
  * garbled → clean: the clean retry is what the user reads;
  * garbled → garbled: the better (original) reply is delivered;
  * clean: NO retry — provider call counts asserted;
  * an English turn is never gated;
  * the shipped Groq default is the evidence-based qwen model.

Run:  sh tests/run.sh -p "test_quality_gate.py"
"""
from __future__ import annotations

import contextlib
import tempfile
import unittest
from pathlib import Path

from _harness import bootstrap

bootstrap()

import config  # noqa: E402
from core import burmese  # noqa: E402
from core.agent_backend import AgentBackend, BackendExhausted  # noqa: E402
from core.key_manager import KeyManager  # noqa: E402
from modes import agent as agent_mod  # noqa: E402
from modes.agent import AgentMode  # noqa: E402
from sessions import SessionStore  # noqa: E402
from tools import ALL_TOOLS  # noqa: E402

KEY_A = "<redacted>"
KEY_B = "<redacted>"

ASK = "လျှပ်စစ်ကား အကြောင်း ရှင်းပြပါ"
GARBLED = ("လျှပ်စစ် ကားများ၏ alternatively Hóa chất đầu tiên trong "
           "các သမိုင်း ရှေးခေတ် အကြောင်း ပြောပြခြင်း ဖြစ်သည်")
CLEAN = ("လျှပ်စစ်ကား ဆိုတာ လျှပ်စစ် မော်တာနဲ့ မောင်းနှင်တဲ့ ကားပါ။ "
         "ဘက်ထရီထဲမှာ လျှပ်စစ် စွမ်းအင် သိုလှောင်ထားပြီး "
         "မောင်းနှင်ချိန်မှာ သုံးပါတယ်။")
SALAD_EN = ("This is a totally English answer about electric vehicles "
            "and their long history in the modern world of transport "
            "and energy today for everyone")
LOOP = " ".join(["လျှပ်စစ်ကား"] * 100)


@contextlib.contextmanager
def verification_off():
    saved = config.AGENT_VERIFY_ENABLED
    config.AGENT_VERIFY_ENABLED = False
    try:
        yield
    finally:
        config.AGENT_VERIFY_ENABLED = saved


class FunctionTool:
    def __init__(self, fn):
        self.fn = fn


class FakeLocalLLM:
    """Local stub — a cloud turn must never touch it."""

    def __init__(self):
        self.log = []
        self._engine = None

    @property
    def loaded(self):
        return any(k == "load" for k, _ in self.log)

    def load(self):
        self.log.append(("load", None))


class PerProviderTransport:
    """Answers per provider (dispatched on the request URL) and
    records every call so retry counts can be asserted."""

    def __init__(self, replies):
        self.replies = replies        # {"openai": text, "openrouter": text}
        self.calls = []               # (provider, body)

    def __call__(self, url, body, headers, timeout=30):
        provider = "openrouter" if "openrouter" in url else "openai"
        self.calls.append((provider, body))
        return {"choices": [{"message": {
            "content": self.replies[provider]}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2}}

    def count(self, provider=None):
        return sum(1 for p, _ in self.calls
                   if provider is None or p == provider)


class GateUnitCase(unittest.TestCase):
    def test_script_salad_fails(self):
        ok, score = burmese.reply_quality(GARBLED)
        self.assertFalse(ok)
        self.assertLess(score, 0.45)

    def test_clean_burmese_passes(self):
        ok, score = burmese.reply_quality(CLEAN)
        self.assertTrue(ok)
        self.assertGreater(score, 0.9)

    def test_burmese_with_ascii_tech_terms_passes(self):
        text = ("EV ကား battery ကို အိမ်မှာ အားသွင်း လို့ ရပါတယ်။ "
                "ဆီဖိုး သက်သာပြီး ထိန်းသိမ်း စရိတ်လည်း နည်းပါတယ်။")
        ok, _ = burmese.reply_quality(text)
        self.assertTrue(ok)

    def test_short_reply_is_not_judged(self):
        ok, _ = burmese.reply_quality("ဟုတ်ကဲ့ ရပါတယ်")
        self.assertTrue(ok)

    def test_repetition_loop_fails(self):
        # The gpt-oss-20b live shape: pure Burmese, one token ~100×.
        ok, _ = burmese.reply_quality(LOOP)
        self.assertFalse(ok)

    def test_repetitive_but_legit_listing_passes(self):
        text = ("စျေးနှုန်းများ — ကော်ဖီ ၃၀၀၀ ကျပ်၊ လက်ဖက်ရည် ၂၀၀၀ ကျပ်၊ "
                "မုန့်ဟင်းခါး ၂၅၀၀ ကျပ်၊ ရှမ်းခေါက်ဆွဲ ၃၀၀၀ ကျပ်၊ "
                "ထမင်းကြော် ၃၅၀၀ ကျပ်၊ ကြက်သားဟင်း ၅၀၀၀ ကျပ် ဖြစ်ပါတယ်။")
        ok, _ = burmese.reply_quality(text)
        self.assertTrue(ok)


class GateRetryCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="quality-gate-"))
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
        self.km.add_key("openai", KEY_A, "alpha")
        self.km.add_key("openrouter", KEY_B, "beta")
        cfg = self.km.load()
        cfg["prefer_cloud"] = True
        cfg["cost_ack"] = True
        self.km.save()
        self.km.set_mode("cloud")
        self.llm = FakeLocalLLM()

    def run_turn(self, replies, ask=ASK):
        transport = PerProviderTransport(replies)
        be = AgentBackend(keys=self.km, net_state=lambda: "online",
                          http=transport)
        mode = AgentMode(self.llm, None)
        mode._tools = [FunctionTool(fn) for fn in ALL_TOOLS]
        mode._backend = be
        with verification_off():
            out = mode.handle(ask)
        return out, transport

    def test_garbled_then_clean_delivers_clean(self):
        out, t = self.run_turn({"openai": GARBLED, "openrouter": CLEAN})
        self.assertEqual(out, CLEAN)
        self.assertEqual(t.count("openai"), 1)
        self.assertEqual(t.count("openrouter"), 1)
        # The retry is a final-text re-ask: it carries no tools.
        retry_body = [b for p, b in t.calls if p == "openrouter"][0]
        self.assertFalse(retry_body.get("tools"))

    def test_garbled_then_garbled_delivers_original(self):
        out, t = self.run_turn({"openai": GARBLED, "openrouter": SALAD_EN})
        self.assertEqual(out, GARBLED)
        self.assertEqual(t.count(), 2)   # exactly one retry, never a loop

    def test_clean_answer_is_not_retried(self):
        out, t = self.run_turn({"openai": CLEAN, "openrouter": SALAD_EN})
        self.assertEqual(out, CLEAN)
        self.assertEqual(t.count(), 1)

    def test_loop_answer_is_retried(self):
        out, t = self.run_turn({"openai": LOOP, "openrouter": CLEAN})
        self.assertEqual(out, CLEAN)
        self.assertEqual(t.count(), 2)

    def test_english_turn_is_never_gated(self):
        out, t = self.run_turn({"openai": SALAD_EN, "openrouter": CLEAN},
                               ask="Explain electric vehicles please")
        self.assertEqual(out, SALAD_EN)
        self.assertEqual(t.count(), 1)

    def test_step_only_restriction_cannot_widen_route(self):
        # only= narrows the route; a provider outside it is unreachable
        # and one outside the route entirely exhausts immediately.
        t = PerProviderTransport({"openai": CLEAN, "openrouter": CLEAN})
        be = AgentBackend(keys=self.km, net_state=lambda: "online", http=t)
        step = be.step(system="s", messages=[{"role": "user",
                                             "text": "hi"}],
                       tools=[], only=["openrouter"])
        self.assertEqual(step.provider, "openrouter")
        self.assertEqual(t.count("openai"), 0)
        with self.assertRaises(BackendExhausted):
            be.step(system="s", messages=[{"role": "user", "text": "hi"}],
                    tools=[], only=["gemini"])


class GroqDefaultCase(unittest.TestCase):
    def test_shipped_groq_default_is_the_evidence_based_qwen(self):
        # Live battery 2026-10-23: the Groq account serves no Llama
        # chat models; qwen/qwen3.8-27b passed 5/5 Burmese prompts.
        from core.key_manager import DEFAULT_CONFIG
        from core.agent_backend import MODELS
        self.assertEqual(DEFAULT_CONFIG["providers"]["groq"]["model"],
                         "qwen/qwen3.8-27b")
        self.assertEqual(MODELS["groq"][0], "qwen/qwen3.8-27b")


if __name__ == "__main__":
    unittest.main()
