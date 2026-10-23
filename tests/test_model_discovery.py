"""Release 27 — live model discovery (core/model_discovery.py,
AgentBackend.list_models, GET /api/settings/models/<provider>) plus the
locked-detail one-liner rider.

Every transport is a stub: no test opens a socket or sees a real key.
Fixture payloads mirror each provider's real /models answer shape.

Run:  sh tests/run.sh -p "test_model_discovery.py"
"""
from __future__ import annotations

import json
import tempfile
import time
import unittest
from contextlib import closing
from pathlib import Path

from _harness import bootstrap, get_app

bootstrap()

import config  # noqa: E402
from core import agent_backend as AB  # noqa: E402
from core import model_discovery as MD  # noqa: E402
from core.agent_backend import AgentBackend, CloudError  # noqa: E402
from core.key_manager import KeyManager  # noqa: E402

ALPHA = "<redacted>"
GEM = "AIzaSyDUMMYKEY00000000000000000000000x"
ORKEY = "<redacted>"

# ── provider payload fixtures, in each provider's real shape ──

OPENAI_MODELS = {"object": "list", "data": [
    {"id": "gpt-4o", "object": "model", "created": 1715367049,
     "owned_by": "openai"},
    {"id": "gpt-4o-mini", "object": "model", "created": 1721172741,
     "owned_by": "openai"},
    {"id": "text-embedding-3-small", "object": "model",
     "created": 1705948997, "owned_by": "openai"},
    {"id": "whisper-1", "object": "model", "created": 1677532384,
     "owned_by": "openai"},
    {"id": "dall-e-3", "object": "model", "created": 1698785189,
     "owned_by": "openai"},
]}

GEMINI_MODELS = {"models": [
    {"name": "models/gemini-2.5-flash", "displayName": "Gemini 2.5 Flash",
     "inputTokenLimit": 1048576, "outputTokenLimit": 65536,
     "supportedGenerationMethods": ["generateContent", "countTokens"]},
    {"name": "models/gemini-2.5-pro", "displayName": "Gemini 2.5 Pro",
     "inputTokenLimit": 2097152, "outputTokenLimit": 65536,
     "supportedGenerationMethods": ["generateContent"]},
    {"name": "models/text-embedding-004", "displayName": "Text Embedding",
     "inputTokenLimit": 2048,
     "supportedGenerationMethods": ["embedContent"]},
]}

OPENROUTER_MODELS = {"data": [
    {"id": "openai/gpt-4o", "name": "OpenAI: GPT-4o", "created": 100,
     "context_length": 128000,
     "architecture": {"modality": "text+image->text"},
     "pricing": {"prompt": "0.000005", "completion": "0.000015"}},
    {"id": "nvidia/nemotron-3.5-embed-8b:free",
     "name": "NVIDIA: Nemotron 3.5 Embed 8B", "created": 300,
     "context_length": 131072,
     "architecture": {"modality": "text->embedding"},
     "pricing": {"prompt": "0", "completion": "0"}},
    {"id": "deepseek/deepseek-chat", "name": "DeepSeek V3",
     "created": 200, "context_length": 65536,
     "architecture": {"modality": "text->text"},
     "pricing": {"prompt": "0", "completion": "0"}},
]}

ANTHROPIC_MODELS = {"data": [
    {"id": "claude-sonnet-4-5", "display_name": "Muse",
     "created_at": "2025-09-29T00:00:00Z", "type": "model"},
    {"id": "claude-3-5-haiku-latest", "display_name": "Muse Spark",
     "created_at": "2024-10-22T00:00:00Z", "type": "model"},
], "has_more": False}

GROQ_MODELS = {"object": "list", "data": [
    {"id": "qwen/qwen3.8-27b", "object": "model", "owned_by": "qwen"},
    {"id": "whisper-large-v3", "object": "model", "owned_by": "openai"},
    {"id": "llama-prompt-guard-2-86m", "object": "model",
     "owned_by": "meta"},
]}

RAW_402 = ('{"error":{"message":"This request requires more credits, or '
           'your account has no balance. You requested up to 8192 tokens '
           'but can only afford 241.","code":402}}')
RAW_MODEL_GONE = ('{"error":{"message":"The model '
                  '`llama-3.3-70b-versatile` does not exist or you do '
                  'not have access to it.","type":"invalid_request_error",'
                  '"code":"model_not_found"}}')


def get_stub(payload, calls=None, fail=None):
    def _f(url, headers, timeout=30):
        if calls is not None:
            calls.append((url, headers))
        if fail is not None:
            raise fail
        return payload
    return _f


class NormaliseTests(unittest.TestCase):
    def test_openai_style_kinds_by_heuristic(self):
        out = {m["id"]: m for m in MD.normalise("openai", OPENAI_MODELS)}
        self.assertEqual(out["gpt-4o"]["kind"], "chat")
        self.assertEqual(out["text-embedding-3-small"]["kind"], "embed")
        self.assertEqual(out["whisper-1"]["kind"], "audio")
        self.assertEqual(out["dall-e-3"]["kind"], "image")
        # No pricing metadata ⇒ free stays unknown, never guessed.
        self.assertIsNone(out["gpt-4o"]["free"])
        self.assertIsNone(out["gpt-4o"]["context"])

    def test_groq_guard_model_is_not_chat(self):
        out = {m["id"]: m for m in MD.normalise("groq", GROQ_MODELS)}
        self.assertEqual(out["qwen/qwen3.8-27b"]["kind"], "chat")
        self.assertEqual(out["whisper-large-v3"]["kind"], "audio")
        self.assertEqual(out["llama-prompt-guard-2-86m"]["kind"], "other")

    def test_gemini_method_filter(self):
        out = MD.normalise("gemini", GEMINI_MODELS)
        by_id = {m["id"]: m for m in out}
        # The "models/" prefix is stripped; displayName rides along.
        self.assertEqual(by_id["gemini-2.5-flash"]["name"],
                         "Gemini 2.5 Flash")
        self.assertEqual(by_id["gemini-2.5-flash"]["kind"], "chat")
        self.assertEqual(by_id["gemini-2.5-flash"]["context"], 1048576)
        # embedContent-only ⇒ embed, never offered as chat.
        self.assertEqual(by_id["text-embedding-004"]["kind"], "embed")

    def test_openrouter_free_flag_and_embed_modality(self):
        out = MD.normalise("openrouter", OPENROUTER_MODELS)
        by_id = {m["id"]: m for m in out}
        embed = by_id["nvidia/nemotron-3.5-embed-8b:free"]
        self.assertEqual(embed["kind"], "embed")   # modality wins
        self.assertIs(embed["free"], True)          # 0 + 0 pricing
        self.assertEqual(embed["context"], 131072)
        self.assertIs(by_id["openai/gpt-4o"]["free"], False)
        self.assertIs(by_id["deepseek/deepseek-chat"]["free"], True)
        self.assertEqual(by_id["openai/gpt-4o"]["kind"], "chat")

    def test_openrouter_newest_first(self):
        out = MD.normalise("openrouter", OPENROUTER_MODELS)
        self.assertEqual([m["id"] for m in out],
                         ["nvidia/nemotron-3.5-embed-8b:free",
                          "deepseek/deepseek-chat", "openai/gpt-4o"])

    def test_anthropic_shape(self):
        out = MD.normalise("anthropic", ANTHROPIC_MODELS)
        self.assertEqual([m["id"] for m in out],
                         ["claude-sonnet-4-5", "claude-3-5-haiku-latest"])
        self.assertTrue(all(m["kind"] == "chat" for m in out))

    def test_garbage_payload_normalises_to_empty(self):
        self.assertEqual(MD.normalise("openai", {"weird": 1}), [])
        self.assertEqual(MD.normalise("openai", ["not", "a", "dict"]), [])
        self.assertEqual(MD.normalise("gemini", {"models": "nope"}), [])

    def test_fallback_models_are_unbadged_chat(self):
        out = MD.fallback_models("groq", AB.MODELS)
        self.assertEqual([m["id"] for m in out], AB.MODELS["groq"])
        self.assertTrue(all(m["kind"] == "chat" and m["free"] is None
                            and m["context"] is None for m in out))


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="models-cache-"))
        self.path = self.dir / "models_cache.json"

    def test_store_load_round_trip(self):
        models = MD.normalise("openai", OPENAI_MODELS)
        MD.store("openai", models, self.path)
        entry = MD.cached("openai", self.path)
        self.assertEqual(entry["models"], models)
        self.assertTrue(MD.is_fresh(entry))
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(raw["version"], MD.VERSION)

    def test_providers_do_not_collide(self):
        MD.store("openai", [{"id": "a"}], self.path)
        MD.store("gemini", [{"id": "b"}], self.path)
        self.assertEqual(MD.cached("openai", self.path)["models"],
                         [{"id": "a"}])
        self.assertEqual(MD.cached("gemini", self.path)["models"],
                         [{"id": "b"}])

    def test_corrupt_cache_is_ignored(self):
        self.path.write_text("{not json", encoding="utf-8")
        self.assertEqual(MD.load_cache(self.path), {})
        self.assertIsNone(MD.cached("openai", self.path))
        self.path.write_text('["a", "list"]', encoding="utf-8")
        self.assertEqual(MD.load_cache(self.path), {})

    def test_stale_entry_is_not_fresh(self):
        MD.store("openai", [{"id": "a"}], self.path)
        entry = MD.cached("openai", self.path)
        entry["fetched_at"] = time.time() - MD.TTL_SECONDS - 5
        self.assertFalse(MD.is_fresh(entry))


class ListModelsCase(unittest.TestCase):
    """AgentBackend.list_models with a stub GET transport."""

    def setUp(self):
        self.km = KeyManager()
        if self.km.path.exists():
            self.km.path.unlink()
        self.km.add_key("openai", ALPHA, "alpha")
        self.km.add_key("gemini", GEM, "gem")
        self.km.add_key("openrouter", ORKEY, "or")
        self.km.save()
        self.cache_path = MD.default_path()
        if self.cache_path.exists():
            self.cache_path.unlink()

    def tearDown(self):
        if self.cache_path.exists():
            self.cache_path.unlink()

    def backend(self, http_get):
        return AgentBackend(keys=self.km,
                            net_state=lambda: "online",
                            http_get=http_get)

    def set_model(self, provider, model):
        cfg = self.km.load()
        cfg["providers"][provider]["model"] = model
        self.km.save()

    def backdate_cache(self, provider, seconds):
        raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
        raw["providers"][provider]["fetched_at"] = time.time() - seconds
        self.cache_path.write_text(json.dumps(raw), encoding="utf-8")

    # ── live fetch ──
    def test_live_fetch_openai_shape_and_auth(self):
        calls = []
        be = self.backend(get_stub(OPENAI_MODELS, calls))
        out = be.list_models("openai")
        self.assertEqual(out["source"], "live")
        self.assertEqual(out["provider"], "openai")
        self.assertEqual(len(out["models"]), 5)
        self.assertEqual(len(calls), 1)
        url, headers = calls[0]
        self.assertEqual(url, "https://api.openai.com/v1/models")
        self.assertEqual(headers["Authorization"], "Bearer " + ALPHA)
        self.assertTrue(out["fetched_at"])

    def test_live_fetch_gemini_url_and_kinds(self):
        calls = []
        be = self.backend(get_stub(GEMINI_MODELS, calls))
        out = be.list_models("gemini")
        self.assertEqual(out["source"], "live")
        url, _headers = calls[0]
        self.assertIn("/v1beta/models?key=" + GEM, url)
        kinds = {m["id"]: m["kind"] for m in out["models"]}
        self.assertEqual(kinds["text-embedding-004"], "embed")

    def test_live_fetch_openrouter(self):
        calls = []
        be = self.backend(get_stub(OPENROUTER_MODELS, calls))
        out = be.list_models("openrouter")
        url, headers = calls[0]
        self.assertEqual(url, "https://openrouter.ai/api/v1/models")
        self.assertEqual(headers["Authorization"], "Bearer " + ORKEY)
        embed = [m for m in out["models"] if m["kind"] == "embed"]
        self.assertEqual(len(embed), 1)
        self.assertIs(embed[0]["free"], True)

    def test_fresh_cache_hit_skips_fetch(self):
        calls = []
        be = self.backend(get_stub(OPENAI_MODELS, calls))
        first = be.list_models("openai")
        self.assertEqual(first["source"], "live")
        second = be.list_models("openai")
        self.assertEqual(second["source"], "cache")
        self.assertEqual(len(calls), 1)  # TTL hit — no second fetch

    def test_refresh_forces_refetch(self):
        calls = []
        be = self.backend(get_stub(OPENAI_MODELS, calls))
        be.list_models("openai")
        out = be.list_models("openai", refresh=True)
        self.assertEqual(out["source"], "live")
        self.assertEqual(len(calls), 2)

    def test_stale_cache_triggers_refetch(self):
        calls = []
        be = self.backend(get_stub(OPENAI_MODELS, calls))
        be.list_models("openai")
        self.backdate_cache("openai", MD.TTL_SECONDS + 60)
        out = be.list_models("openai")
        self.assertEqual(out["source"], "live")
        self.assertEqual(len(calls), 2)

    # ── degradation order: live → cache → fallback ──
    def test_fetch_failure_uses_stale_cache(self):
        be = self.backend(get_stub(OPENAI_MODELS))
        be.list_models("openai")
        self.backdate_cache("openai", MD.TTL_SECONDS + 60)
        failing = self.backend(get_stub(
            None, fail=CloudError("network", "down", 0)))
        out = failing.list_models("openai")
        self.assertEqual(out["source"], "cache")
        self.assertEqual(len(out["models"]), 5)

    def test_fetch_failure_without_cache_uses_fallback(self):
        failing = self.backend(get_stub(
            None, fail=CloudError("network", "down", 0)))
        out = failing.list_models("openai")
        self.assertEqual(out["source"], "fallback")
        self.assertEqual([m["id"] for m in out["models"]],
                         AB.MODELS["openai"])
        self.assertIsNone(out["fetched_at"])

    def test_no_key_uses_fallback_without_fetching(self):
        calls = []
        be = self.backend(get_stub(OPENAI_MODELS, calls))
        out = be.list_models("deepseek")  # no deepseek key in the fixture
        self.assertEqual(out["source"], "fallback")
        self.assertEqual(calls, [])

    def test_bad_payload_degrades_to_fallback(self):
        be = self.backend(get_stub({"unexpected": "shape"}))
        out = be.list_models("openai")
        self.assertEqual(out["source"], "fallback")

    def test_corrupt_cache_then_live(self):
        self.cache_path.write_text("garbage{", encoding="utf-8")
        be = self.backend(get_stub(OPENAI_MODELS))
        out = be.list_models("openai")
        self.assertEqual(out["source"], "live")

    # ── stored_model_status ──
    def test_stored_model_ok(self):
        self.set_model("openai", "gpt-4o")
        be = self.backend(get_stub(OPENAI_MODELS))
        out = be.list_models("openai")
        self.assertEqual(out["stored_model"], "gpt-4o")
        self.assertEqual(out["stored_model_status"], "ok")

    def test_stored_model_missing(self):
        # The 2026-10-23 incident shape: the stored id was retired.
        self.set_model("openai", "gpt-gone-1")
        be = self.backend(get_stub(OPENAI_MODELS))
        out = be.list_models("openai")
        self.assertEqual(out["stored_model_status"], "missing")

    def test_stored_model_missing_against_cache(self):
        self.set_model("openrouter", "nvidia/nemotron-3.5-embed-8b:free")
        be = self.backend(get_stub(OPENROUTER_MODELS))
        be.list_models("openrouter")  # fills the cache
        self.set_model("openrouter", "some/retired-model")
        out = be.list_models("openrouter")
        self.assertEqual(out["source"], "cache")
        self.assertEqual(out["stored_model_status"], "missing")

    def test_stored_model_unknown_on_fallback(self):
        self.set_model("deepseek", "deepseek-chat")
        be = self.backend(get_stub(OPENAI_MODELS))
        out = be.list_models("deepseek")
        self.assertEqual(out["source"], "fallback")
        # The preset list CONTAINS deepseek-chat, but a fallback list is
        # not evidence about the account — status stays unknown.
        self.assertEqual(out["stored_model_status"], "unknown")


class EndpointTests(unittest.TestCase):
    """The route contract: shape, degradation, no key material."""

    @classmethod
    def setUpClass(cls):
        cls.client = get_app().test_client()

    def setUp(self):
        self.km = KeyManager()
        if self.km.path.exists():
            self.km.path.unlink()
        self.km.save()
        self.cache_path = MD.default_path()
        if self.cache_path.exists():
            self.cache_path.unlink()

    def tearDown(self):
        if self.cache_path.exists():
            self.cache_path.unlink()

    def call(self, path, expect=200):
        with closing(self.client.get(path)) as r:
            self.assertEqual(r.status_code, expect,
                             "%s -> %s" % (path, r.status_code))
            return r.get_json()

    def test_fallback_shape_for_keyless_provider(self):
        data = self.call("/api/settings/models/deepseek")
        self.assertTrue(data["ok"])
        self.assertEqual(data["provider"], "deepseek")
        self.assertEqual(data["source"], "fallback")
        self.assertEqual(data["stored_model_status"], "unknown")
        self.assertEqual([m["id"] for m in data["models"]],
                         AB.MODELS["deepseek"])
        for m in data["models"]:
            # Release 29: the entry shape gains image_input; presets
            # (no catalogue metadata) always carry it False.
            self.assertEqual(set(m), {"id", "name", "context", "kind",
                                      "free", "image_input"})
            self.assertIs(m["image_input"], False)

    def test_cache_shape_is_served(self):
        MD.store("openai", MD.normalise("openai", OPENAI_MODELS))
        data = self.call("/api/settings/models/openai")
        self.assertEqual(data["source"], "cache")
        self.assertEqual(len(data["models"]), 5)
        self.assertTrue(data["fetched_at"])

    def test_refresh_param_is_accepted(self):
        MD.store("openai", MD.normalise("openai", OPENAI_MODELS))
        data = self.call("/api/settings/models/openai?refresh=1")
        # Refresh forces a refetch attempt; with no openai key stored it
        # degrades back to the cache it already had — never an error.
        self.assertEqual(data["source"], "cache")

    def test_unknown_provider_is_a_clean_error(self):
        data = self.call("/api/settings/models/nonsense", expect=400)
        self.assertIn("error", data)

    def test_no_key_material_in_answer(self):
        MD.store("openai", MD.normalise("openai", OPENAI_MODELS))
        with closing(self.client.get("/api/settings/models/openai")) as r:
            body = r.get_data(as_text=True)
        self.assertNotIn("sk-", body)


class LockedDetailOnelineTests(unittest.TestCase):
    """Rider: locked replies show the provider's message, not its JSON."""

    def test_json_error_message_is_extracted(self):
        out = AB._provider_message_oneline(RAW_MODEL_GONE)
        self.assertEqual(
            out,
            "The model `llama-3.3-70b-versatile` does not exist or you "
            "do not have access to it.")
        self.assertNotIn("{", out)
        self.assertNotIn('"error"', out)

    def test_plain_text_passes_through(self):
        out = AB._provider_message_oneline("Service unavailable now")
        self.assertEqual(out, "Service unavailable now")

    def test_truncated_json_loses_its_braces(self):
        out = AB._provider_message_oneline(RAW_402[:60])
        self.assertNotIn("{", out)
        self.assertNotIn("}", out)
        self.assertIn("requires more credits", out)

    def test_cap_is_200_chars(self):
        out = AB._provider_message_oneline("x" * 500)
        self.assertLessEqual(len(out), 200)

    def test_echoed_key_is_masked(self):
        body = ('{"error":{"message":"Incorrect API key provided: '
                '<redacted>. You can find your API key..."}}')
        out = AB._provider_message_oneline(body)
        self.assertNotIn("sk-abc123def456ghi789", out)
        self.assertIn("Incorrect API key provided", out)

    def test_locked_exhausted_carries_the_oneline(self):
        km = KeyManager()
        if km.path.exists():
            km.path.unlink()
        km.add_key("openai", ALPHA, "alpha")
        cfg = km.load()
        cfg["cost_ack"] = True
        cfg["auto_switch"] = False
        cfg["pinned_provider"] = "openai"
        km.save()

        def http_fail(url, body, headers, timeout=30):
            raise CloudError("bad_response", RAW_402, 402)

        be = AgentBackend(keys=km, net_state=lambda: "online",
                          http=http_fail)
        with self.assertRaises(AB.BackendExhausted) as ctx:
            be.step(system="s",
                    messages=[{"role": "user", "content": "hi"}],
                    tools=[], max_tokens=16)
        exc = ctx.exception
        self.assertEqual(exc.locked_provider, "openai")
        self.assertEqual(exc.locked_reason, "no_balance")
        self.assertTrue(exc.locked_detail.startswith(
            "This request requires more credits"))
        self.assertIn("241", exc.locked_detail)
        self.assertNotIn('{"error"', str(exc))
        self.assertNotIn('{"error"', exc.locked_detail)


if __name__ == "__main__":
    unittest.main(verbosity=2)
