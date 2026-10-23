"""Release 29 — cloud vision, clarify routing, compaction integrity,
second-key steering.

Four behaviours, each pinned the way the round-5 evaluation found it:

1. Vision: discovery records an ``image_input`` flag per model (the
   OpenRouter modality INPUT side, plus the two provider-controlled
   chat families; unknown ⇒ False). A user turn carrying an image is
   normalised to canonical parts and each adapter renders its own
   wire shape (OpenAI image_url data URI / Gemini inline_data /
   Anthropic base64 block). Routing respects capability: locked mode
   answers an incapable pin with Burmese guidance (the image is never
   dropped silently), Auto narrows the turn to vision-capable
   providers, and the image payload never reaches a durable store.
2. F2: a compaction whose summary drops the checkable facts of the
   dropped entries is a FAILED compaction — the fallback keeps the
   newest raw history and marks the loss with a visible pair, exactly
   like a summarizer failure does. Nothing about the failure path is
   ever worse than the v1.26.0 plain trim.
3. F3: locked-mode and quota failure replies gain the second-free-key
   steering line only when the user has exactly one live provider.

Stub transports only; no socket, no weights.

Run:  sh tests/run.sh -p "test_release29.py"
"""
from __future__ import annotations

import base64
import contextlib
import json
import unittest
from pathlib import Path

from _harness import bootstrap

bootstrap()

import config  # noqa: E402
from core import model_discovery as MD  # noqa: E402
from core.agent_backend import (  # noqa: E402
    ADAPTERS, AgentBackend, CloudError, normalize)
from core.key_manager import KeyManager  # noqa: E402
from modes.agent import (  # noqa: E402
    AgentMode, _SUMMARY_MARKER, _TRIM_MARKER, _TRIM_NOTE,
    _summary_keeps_facts)
from sessions import SessionStore  # noqa: E402

KEY_A = "<redacted>"
KEY_B = "<redacted>"
CLOUD_TEXT = "CLOUD_REPLY_OK"
B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 32).decode("ascii")
QUOTA_BODY = ('{"error":{"message":"You exceeded your current quota, '
              'please check your plan and billing details"}}')
RAW_402 = ('{"error":{"message":"This request requires more credits, '
           'or fewer max_tokens."}}')


@contextlib.contextmanager
def verification_off():
    saved = config.AGENT_VERIFY_ENABLED
    config.AGENT_VERIFY_ENABLED = False
    try:
        yield
    finally:
        config.AGENT_VERIFY_ENABLED = saved


class FakeLocalLLM:
    """Local engine stub: the cloud paths under test never call it;
    ask() exists so a stray summarizer call is visible, not fatal."""

    def __init__(self):
        self.ask_prompts = []

    @property
    def loaded(self):
        return False

    def load(self):
        pass

    def ask(self, prompt):
        self.ask_prompts.append(prompt)
        return ""


class OkTransport:
    """OpenAI-dialect stub: records calls, always answers CLOUD_TEXT."""

    def __init__(self):
        self.calls = []

    def __call__(self, url, body, headers, timeout=30):
        self.calls.append((url, body, headers))
        return {"choices": [{"message": {"content": CLOUD_TEXT}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2}}


class FailTransport:
    """OpenAI-dialect stub: every call raises the same CloudError."""

    def __init__(self, err):
        self.err = err
        self.calls = []

    def __call__(self, url, body, headers, timeout=30):
        self.calls.append(url)
        raise self.err


def vision_entry(mid, capable):
    return {"id": mid, "name": mid, "context": None, "kind": "chat",
            "free": None, "image_input": capable}


class R29Base(unittest.TestCase):
    """Fresh keys in the harness HOME; no real session store unless a
    test wires one."""

    def setUp(self):
        self.km = KeyManager()
        if self.km.path.exists():
            self.km.path.unlink()
        cache = MD.default_path()
        if cache.exists():
            cache.unlink()
        self.llm = FakeLocalLLM()

    def tearDown(self):
        cache = MD.default_path()
        if cache.exists():
            cache.unlink()

    def add(self, provider, key, model=None):
        self.km.add_key(provider, key, provider)
        if model:
            cfg = self.km.load()
            cfg["providers"][provider]["model"] = model
            self.km.save()

    def base_cfg(self):
        cfg = self.km.load()
        cfg["prefer_cloud"] = True
        cfg["cost_ack"] = True
        self.km.save()

    def backend(self, transport):
        return AgentBackend(keys=self.km, net_state=lambda: "online",
                            http=transport)

    def mode(self, be, backend_mode="auto"):
        self.km.set_mode(backend_mode)
        self.km.save()
        m = AgentMode(self.llm, None)
        m._store = None
        m._tools = []
        m._messages = []
        m._backend = be
        return m


# ── 1a. discovery flags ─────────────────────────────────────────────

class DiscoveryImageInputCase(unittest.TestCase):
    def by_id(self, provider, payload):
        return {m["id"]: m for m in MD.normalise(provider, payload)}

    def test_openrouter_modality_input_side(self):
        models = self.by_id("openrouter", {"data": [
            {"id": "a/vision-chat", "architecture":
             {"modality": "text+image->text"}},
            {"id": "a/text-chat", "architecture":
             {"modality": "text->text"}},
            {"id": "a/image-gen", "architecture":
             {"modality": "text+image->image"}},
            {"id": "a/no-meta"},
        ]})
        self.assertTrue(models["a/vision-chat"]["image_input"])
        self.assertFalse(models["a/text-chat"]["image_input"])
        # An image generator accepts an image as input too — the flag
        # is about the input side; kind keeps it out of chat rows.
        self.assertTrue(models["a/image-gen"]["image_input"])
        self.assertEqual(models["a/image-gen"]["kind"], "image")
        self.assertFalse(models["a/no-meta"]["image_input"])

    def test_gemini_family_flag_and_exclusions(self):
        models = self.by_id("gemini", {"models": [
            {"name": "models/gemini-2.5-flash",
             "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/text-embedding-004",
             "supportedGenerationMethods": ["embedContent"]},
            {"name": "models/gemini-2.5-flash-preview-tts",
             "supportedGenerationMethods": ["generateContent"]},
        ]})
        self.assertTrue(models["gemini-2.5-flash"]["image_input"])
        self.assertFalse(models["text-embedding-004"]["image_input"])
        self.assertFalse(
            models["gemini-2.5-flash-preview-tts"]["image_input"])

    def test_anthropic_family_flag(self):
        models = self.by_id("anthropic", {"data": [
            {"id": "claude-sonnet-4-5", "type": "model"},
        ]})
        self.assertTrue(models["claude-sonnet-4-5"]["image_input"])

    def test_open_namespaces_stay_conservative(self):
        # No metadata ⇒ never flagged, however famous the model is:
        # an unflagged model costs a fallback, a wrongly flagged one
        # costs a silently dropped photo.
        models = self.by_id("openai", {"data": [{"id": "gpt-4o"}]})
        self.assertFalse(models["gpt-4o"]["image_input"])
        models = self.by_id("groq", {"data": [
            {"id": "meta-llama/llama-4-scout-17b-16e-instruct"}]})
        self.assertFalse(
            models["meta-llama/llama-4-scout-17b-16e-instruct"]
            ["image_input"])

    def test_presets_carry_the_flag_false(self):
        from core.agent_backend import MODELS
        for entry in MD.fallback_models("openrouter", MODELS):
            self.assertIn("image_input", entry)
            self.assertIs(entry["image_input"], False)


# ── 1b. canonical shape + per-provider payloads ─────────────────────

IMAGE_MSG = {"role": "user", "text": "ဒီပုံမှာ ဘာတွေ့လဲ",
             "image": {"data": B64, "mime": "image/png"}}


class PayloadShapeCase(unittest.TestCase):
    def canonical(self):
        msgs = normalize([dict(IMAGE_MSG),
                          {"role": "assistant", "text": "အရင် အဖြေ"}])
        self.assertEqual(len(msgs), 2)
        return msgs

    def test_normalize_builds_parts_only_with_an_image(self):
        msgs = self.canonical()
        self.assertEqual(
            msgs[0]["content"],
            [{"type": "text", "text": "ဒီပုံမှာ ဘာတွေ့လဲ"},
             {"type": "image", "mime": "image/png", "data": B64}])
        plain = normalize([{"role": "user", "text": "မင်္လာပါ"}])
        self.assertEqual(plain[0]["content"], "မင်္လာပါ")

    def test_openai_dialect_data_uri(self):
        for provider in ("openai", "openrouter", "groq", "deepseek",
                         "siliconflow", "custom"):
            _url, body, _h = ADAPTERS[provider]().request(
                key="k", model="m", system="", messages=self.canonical(),
                tools=[], max_tokens=64)
            user = body["messages"][0]
            self.assertEqual(
                user["content"][1],
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + B64}},
                provider)

    def test_openai_text_only_shape_unchanged(self):
        _url, body, _h = ADAPTERS["openai"]().request(
            key="k", model="m", system="",
            messages=normalize([{"role": "user", "text": "hi"}]),
            tools=[], max_tokens=64)
        self.assertEqual(body["messages"][0]["content"], "hi")

    def test_gemini_inline_data(self):
        msgs = self.canonical() + normalize(
            [{"role": "user", "text": "နောက်ထပ် မေးခွန်း"}])
        _url, body, _h = ADAPTERS["gemini"]().request(
            key="k", model="gemini-2.5-flash", system="",
            messages=msgs, tools=[], max_tokens=64)
        # Round 6 (defect D1): exactly one contents entry per
        # transcript entry. The list-content branch once appended its
        # entry AND the shared loop-end append added it again, so a
        # 3-turn image transcript shipped 4 entries with the image
        # twice. Assert the count and the single inline_data entry —
        # parts-only assertions cannot see a duplicated entry.
        self.assertEqual(len(body["contents"]), 3)
        inline = [c for c in body["contents"]
                  if any("inline_data" in p for p in c["parts"])]
        self.assertEqual(len(inline), 1)
        self.assertIs(inline[0], body["contents"][0])
        parts = body["contents"][0]["parts"]
        self.assertEqual(parts[0], {"text": "ဒီပုံမှာ ဘာတွေ့လဲ"})
        self.assertEqual(parts[1], {"inline_data": {
            "mime_type": "image/png", "data": B64}})
        self.assertEqual(body["contents"][2]["parts"],
                         [{"text": "နောက်ထပ် မေးခွန်း"}])

    def test_anthropic_base64_block(self):
        _url, body, _h = ADAPTERS["anthropic"]().request(
            key="k", model="claude-sonnet-4-5", system="",
            messages=self.canonical(), tools=[], max_tokens=64)
        blocks = body["messages"][0]["content"]
        self.assertEqual(blocks[0],
                         {"type": "text", "text": "ဒီပုံမှာ ဘာတွေ့လဲ"})
        self.assertEqual(blocks[1], {"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": B64}})


# ── 1c. backend capability + AgentMode routing ──────────────────────

class VisionRoutingCase(R29Base):
    def setUp(self):
        super().setUp()
        self.add("openai", KEY_A, "vision-model-x")
        self.add("groq", KEY_B, "text-model-y")
        self.base_cfg()
        MD.store("openai", [vision_entry("vision-model-x", True)])
        MD.store("groq", [vision_entry("text-model-y", False)])

    def test_backend_capability_answers(self):
        be = self.backend(OkTransport())
        self.assertTrue(be.model_image_input("openai"))
        self.assertFalse(be.model_image_input("groq"))
        self.assertEqual(be.cloud_vision_providers(), ["openai"])

    def test_auto_image_turn_uses_only_the_capable_provider(self):
        transport = OkTransport()
        mode = self.mode(self.backend(transport))
        with verification_off():
            out = mode.handle("ဒီပုံမှာ ဘာတွေ့လဲ",
                              image={"data": B64, "mime": "image/png"})
        self.assertEqual(out, CLOUD_TEXT)
        self.assertTrue(transport.calls)
        for url, body, _h in transport.calls:
            self.assertIn("api.openai.com", url)
            self.assertNotIn("groq", url)
        sent = json.dumps(transport.calls[0][1])
        self.assertIn("data:image/png;base64," + B64[:24], sent)

    def test_auto_without_a_capable_route_declines_to_local(self):
        MD.store("openai", [vision_entry("vision-model-x", False)])
        transport = OkTransport()
        mode = self.mode(self.backend(transport))
        out = mode._try_cloud("ဒီပုံမှာ ဘာတွေ့လဲ",
                              image={"data": B64, "mime": "image/png"})
        self.assertIsNone(out)
        self.assertEqual(transport.calls, [])

    def test_locked_incapable_pin_answers_guidance_not_silence(self):
        cfg = self.km.load()
        cfg["auto_switch"] = False
        cfg["pinned_provider"] = "groq"
        self.km.save()
        transport = OkTransport()
        mode = self.mode(self.backend(transport))
        with verification_off():
            out = mode.handle("ဒီပုံမှာ ဘာတွေ့လဲ",
                              image={"data": B64, "mime": "image/png"})
        self.assertIn("မဖတ်နိုင်", out)
        self.assertIn("📷", out)
        self.assertEqual(transport.calls, [])
        # The guidance is the turn's reply in the transcript too.
        self.assertEqual(mode._messages[-1]["text"], out)

    def test_locked_capable_pin_serves_the_image(self):
        cfg = self.km.load()
        cfg["auto_switch"] = False
        cfg["pinned_provider"] = "openai"
        self.km.save()
        transport = OkTransport()
        mode = self.mode(self.backend(transport))
        with verification_off():
            out = mode.handle("ဒီပုံမှာ ဘာတွေ့လဲ",
                              image={"data": B64, "mime": "image/png"})
        self.assertEqual(out, CLOUD_TEXT)
        self.assertIn("data:image/png;base64,",
                      json.dumps(transport.calls[0][1]))

    def test_cloud_only_without_a_capable_route_says_so(self):
        MD.store("openai", [vision_entry("vision-model-x", False)])
        MD.store("groq", [vision_entry("text-model-y", False)])
        transport = OkTransport()
        mode = self.mode(self.backend(transport), "cloud")
        with verification_off():
            out = mode.handle("ဒီပုံမှာ ဘာတွေ့လဲ",
                              image={"data": B64, "mime": "image/png"})
        self.assertIn("Cloud Only mode", out)
        self.assertIn("ပုံကို မဖတ်နိုင်", out)
        self.assertEqual(transport.calls, [])

    def test_image_payload_never_reaches_the_session_store(self):
        store = SessionStore(Path(config.DATA_DIR) / "sessions.json")
        transport = OkTransport()
        mode = self.mode(self.backend(transport))
        mode._store = store
        with verification_off():
            out = mode.handle("ဒီပုံမှာ ဘာတွေ့လဲ",
                              image={"data": B64, "mime": "image/png"})
        self.assertEqual(out, CLOUD_TEXT)
        raw = (Path(config.DATA_DIR) / "sessions.json").read_text(
            encoding="utf-8")
        self.assertNotIn(B64, raw)
        self.assertNotIn('"image"', raw)
        # …while the live transcript kept it for the turn itself.
        self.assertIn("image", mode._messages[0])


# ── 2. compaction integrity ─────────────────────────────────────────

FACTS = ("Project codename is Zebra-42 and the budget is 450000 "
         "kyats for the hall rental")
FAITHFUL = "Codename Zebra-42 — budget 450000 kyats, hall rental project"
LOSSY = "Various topics were discussed at length"


def seeded_transcript():
    msgs = [{"role": "user", "text": FACTS},
            {"role": "assistant", "text": "noted, will remember"}]
    for i in range(42):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "text": f"filler message number {i}"})
    return msgs


class CompactionIntegrityCase(R29Base):
    def mode(self):
        m = AgentMode(self.llm, None)
        m._store = None
        m._tools = []
        m._messages = []
        return m

    def test_fact_check_units(self):
        dropped = [{"role": "user", "text": FACTS}]
        self.assertTrue(_summary_keeps_facts(dropped, FAITHFUL))
        self.assertFalse(_summary_keeps_facts(dropped, LOSSY))
        # Nothing checkable in the dropped entries ⇒ trusted.
        self.assertTrue(_summary_keeps_facts(
            [{"role": "user", "text": "ok"}], LOSSY))
        self.assertTrue(_summary_keeps_facts([], LOSSY))

    def test_faithful_summary_compacts_and_keeps_facts(self):
        mode = self.mode()
        mode._messages = seeded_transcript()
        mode._cap_history(summarize=lambda msgs: FAITHFUL)
        msgs = mode._messages
        self.assertEqual(msgs[0]["text"], _SUMMARY_MARKER)
        self.assertIn("Zebra-42", msgs[1]["text"])
        self.assertIn("450000", msgs[1]["text"])
        self.assertFalse(any(m.get("text") == _TRIM_MARKER
                             for m in msgs))
        # The summary pair carries the visible condensed framing.
        self.assertIn("အကျဉ်းချုပ်", _SUMMARY_MARKER)

    def test_lossy_summary_falls_back_to_marked_raw_history(self):
        mode = self.mode()
        before = seeded_transcript()
        mode._messages = list(before)
        mode._cap_history(summarize=lambda msgs: LOSSY)
        msgs = mode._messages
        self.assertEqual(len(msgs), 40)
        self.assertEqual(msgs[0]["text"], _TRIM_MARKER)
        self.assertEqual(msgs[1]["text"], _TRIM_NOTE)
        self.assertFalse(any(m.get("text") == _SUMMARY_MARKER
                             for m in msgs))
        # The newest raw entries survive, bounded by the cap.
        self.assertEqual(msgs[2:], before[-38:])
        self.assertEqual(msgs[-1], before[-1])

    def test_raising_summarizer_falls_back_to_marked_raw_history(self):
        def boom(msgs):
            raise RuntimeError("summarizer gone")

        mode = self.mode()
        mode._messages = seeded_transcript()
        mode._cap_history(summarize=boom)
        msgs = mode._messages
        self.assertEqual(msgs[0]["text"], _TRIM_MARKER)
        self.assertEqual(len(msgs), 40)


# ── 3. second-key steering ──────────────────────────────────────────

class SecondKeyHintCase(R29Base):
    def test_hint_counts_live_providers(self):
        self.add("openai", KEY_A, "gpt-4o-mini")
        self.base_cfg()
        be = self.backend(OkTransport())
        mode = self.mode(be)
        self.assertEqual(mode._second_key_hint(be),
                         AgentMode.SECOND_KEY_HINT)
        self.add("groq", KEY_B, "qwen/qwen3.8-27b")
        self.assertEqual(mode._second_key_hint(be), "")

    def test_locked_reply_carries_the_hint_for_a_solo_provider(self):
        self.add("openai", KEY_A, "gpt-4o-mini")
        self.base_cfg()
        cfg = self.km.load()
        cfg["auto_switch"] = False
        cfg["pinned_provider"] = "openai"
        self.km.save()
        transport = FailTransport(
            CloudError("bad_response", RAW_402, status=402))
        mode = self.mode(self.backend(transport))
        with verification_off():
            out = mode.handle("မင်္လာပါ")
        self.assertIn("🔒 Lock ချထားတဲ့", out)
        self.assertIn(AgentMode.SECOND_KEY_HINT, out)

    def test_quota_reply_carries_the_hint_for_a_solo_provider(self):
        self.add("openai", KEY_A, "gpt-4o-mini")
        self.base_cfg()
        transport = FailTransport(
            CloudError("rate_limit", QUOTA_BODY, status=429))
        mode = self.mode(self.backend(transport), "cloud")
        with verification_off():
            out = mode.handle("မင်္လာပါ")
        self.assertIn("Cloud Only mode", out)
        self.assertIn(AgentMode.SECOND_KEY_HINT, out)

    def test_no_hint_with_two_providers(self):
        self.add("openai", KEY_A, "gpt-4o-mini")
        self.add("groq", KEY_B, "qwen/qwen3.8-27b")
        self.base_cfg()
        transport = FailTransport(
            CloudError("rate_limit", QUOTA_BODY, status=429))
        mode = self.mode(self.backend(transport), "cloud")
        with verification_off():
            out = mode.handle("မင်္လာပါ")
        self.assertIn("Cloud Only mode", out)
        self.assertNotIn(AgentMode.SECOND_KEY_HINT, out)

    def test_no_hint_for_a_mere_cooldown(self):
        self.add("openai", KEY_A, "gpt-4o-mini")
        self.base_cfg()
        transport = FailTransport(CloudError(
            "rate_limit", "slow down", status=429, retry_after=60))
        mode = self.mode(self.backend(transport), "cloud")
        with verification_off():
            out = mode.handle("မင်္လာပါ")
        self.assertNotIn(AgentMode.SECOND_KEY_HINT, out)


if __name__ == "__main__":
    unittest.main()
