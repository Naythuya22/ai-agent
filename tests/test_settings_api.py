"""routes/settings_api.py — the Settings endpoints behind the ⚙️ page.

Two contracts are pinned here:

* every answer is JSON, including failures (`{"error": "..."}`) — the bug the
  UI reported was `Unexpected token 'T', "The reques… is not valid JSON`;
* no raw API key is ever serialised back to the client.

Run:  sh tests/run.sh -p "test_settings_api.py"
"""
from __future__ import annotations

import json
import unittest
from contextlib import closing

from _harness import bootstrap, get_app

MODEL = bootstrap()
app = get_app()
client = app.test_client()

from core import agent_backend as AB  # noqa: E402
from core.key_manager import PROVIDERS, PROVIDER_LABELS, KeyManager  # noqa: E402

RAW_OPENAI = "sk-live-settings-test-1234567890"
RAW_DEEPSEEK = "sk-deep-settings-test-abcdef"


def call(method, path, body=None, expect=None):
    fn = getattr(client, method)
    kw = {"json": body} if body is not None else {}
    with closing(fn(path, **kw)) as r:
        data = r.get_json() if r.content_type and "json" in r.content_type else None
        if expect is not None:
            assert r.status_code == expect, (
                "%s %s -> %s (want %s): %s" % (method.upper(), path, r.status_code,
                                               expect, r.get_data(as_text=True)[:200]))
        return r, data


def fresh_keys() -> KeyManager:
    km = KeyManager()
    if km.path.exists():
        km.path.unlink()
    return km


class SettingsApiCase(unittest.TestCase):
    def setUp(self):
        self.km = fresh_keys()
        self.cfg = self.km.load()
        self.cfg["prefer_cloud"] = True
        self.cfg["cost_ack"] = True
        self.km.save()   # the route opens its own KeyManager — persist it


class GetSettingsTests(SettingsApiCase):
    def test_get_returns_json_with_every_section(self):
        _, j = call("get", "/api/settings/agent", expect=200)
        for key in ("ok", "config", "backend", "costs", "providers", "modes",
                    "models", "fallback_order", "local_fallback",
                    "provider_labels"):
            self.assertIn(key, j, msg=key)
        self.assertTrue(j["ok"])

    def test_provider_dropdown_labels(self):
        _, j = call("get", "/api/settings/agent", expect=200)
        self.assertEqual(sorted(j["providers"]), sorted(PROVIDERS))
        self.assertEqual(j["provider_labels"], PROVIDER_LABELS)
        self.assertEqual(j["provider_labels"]["openai"], "OpenAI")
        self.assertEqual(j["provider_labels"]["gemini"], "Google Gemini")
        self.assertEqual(j["provider_labels"]["openrouter"], "OpenRouter")
        self.assertEqual(j["provider_labels"]["anthropic"], "Anthropic")
        self.assertEqual(j["provider_labels"]["deepseek"], "DeepSeek")

    def test_default_model_presets_per_provider(self):
        _, j = call("get", "/api/settings/agent", expect=200)
        models = j["models"]
        for m in ("gpt-4o", "gpt-4o-mini"):
            self.assertIn(m, models["openai"])
        for m in ("gemini-2.5-flash", "gemini-2.5-pro"):
            self.assertIn(m, models["gemini"])
        self.assertNotIn("google/gemini-2.5-flash:free", models["openrouter"])
        self.assertIn("google/gemini-2.5-flash", models["openrouter"])
        self.assertIn("deepseek-chat", models["deepseek"])
        self.assertTrue(models["anthropic"])

    def test_backend_modes_offered(self):
        _, j = call("get", "/api/settings/agent", expect=200)
        self.assertEqual(sorted(j["modes"]), ["auto", "cloud", "local"])
        self.assertEqual(j["config"]["backend_mode"], "auto")

    def test_costs_are_reported_per_provider(self):
        _, j = call("get", "/api/settings/agent", expect=200)
        self.assertEqual(j["costs"]["deepseek"], AB.cost_per_1k(
            "deepseek", j["config"]["providers"]["deepseek"]["model"]))
        self.assertIsNotNone(j["costs"]["openai"])

    def test_no_raw_key_is_ever_returned(self):
        self.km.add_key("openai", RAW_OPENAI, "main")
        self.km.add_key("deepseek", RAW_DEEPSEEK, "ds")
        _, j = call("get", "/api/settings/agent", expect=200)
        blob = json.dumps(j)
        self.assertNotIn(RAW_OPENAI, blob)
        self.assertNotIn(RAW_DEEPSEEK, blob)
        masked = j["config"]["keys"]["openai"][0]["key_masked"]
        self.assertTrue(masked.endswith("7890"), msg=masked)
        self.assertIn("key_masked", j["config"]["keys"]["deepseek"][0])

    def test_key_file_is_0600(self):
        self.km.add_key("openai", RAW_OPENAI)
        self.assertEqual(KeyManager().path.stat().st_mode & 0o777, 0o600)

    def test_config_path_is_reported_and_on_disk(self):
        """The fixed, HOME-independent location is visible to the client —
        the "keys vanished after restart" report needs a path to check."""
        self.km.add_key("openai", RAW_OPENAI)
        _, j = call("get", "/api/settings/agent", expect=200)
        self.assertEqual(j["config_path"], str(KeyManager().path))
        self.assertTrue(j["config_path"].endswith("agent_config.json"))
        self.assertTrue(j["config_exists"])


class PostSettingsTests(SettingsApiCase):
    def test_backend_mode_roundtrip(self):
        for mode in ("local", "cloud", "auto"):
            _, j = call("post", "/api/settings/agent",
                        {"backend_mode": mode}, expect=200)
            self.assertEqual(j["config"]["backend_mode"], mode, msg=mode)
            self.assertEqual(j["config"]["prefer_cloud"], mode != "local", msg=mode)
        self.assertEqual(KeyManager().mode(), "auto")

    def test_unknown_backend_mode_is_a_json_error(self):
        r, j = call("post", "/api/settings/agent",
                    {"backend_mode": "sometimes"}, expect=400)
        self.assertTrue(r.is_json)
        self.assertIn("error", j)

    def test_legacy_prefer_cloud_maps_to_a_mode(self):
        _, j = call("post", "/api/settings/agent",
                    {"prefer_cloud": False}, expect=200)
        self.assertEqual(j["config"]["backend_mode"], "local")

    def test_fallback_is_validated_and_completed(self):
        r, j = call("post", "/api/settings/agent",
                    {"fallback": "not-a-list"}, expect=400)
        self.assertIn("error", j)
        _, j = call("post", "/api/settings/agent",
                    {"fallback": ["deepseek", "deepseek", "nope"]}, expect=200)
        fb = j["config"]["fallback"]
        self.assertEqual(fb[0], "deepseek")
        self.assertEqual(len(fb), len(set(fb)))
        for p in PROVIDERS:
            self.assertIn(p, fb, msg="no provider may fall out of the order")

    def test_provider_edits(self):
        _, j = call("post", "/api/settings/agent", {"providers": {
            "deepseek": {"model": "deepseek-reasoner", "enabled": False,
                         "base_url": "https://api.deepseek.com/"}}}, expect=200)
        ds = j["config"]["providers"]["deepseek"]
        self.assertEqual(ds["model"], "deepseek-reasoner")
        self.assertFalse(ds["enabled"])
        self.assertEqual(ds["base_url"], "https://api.deepseek.com")

    def test_unknown_provider_is_rejected(self):
        r, j = call("post", "/api/settings/agent",
                    {"providers": {"huggingface": {"model": "x"}}}, expect=400)
        self.assertIn("error", j)

    def test_bad_base_url_is_rejected(self):
        r, j = call("post", "/api/settings/agent",
                    {"providers": {"openai": {"base_url": "ftp://nope"}}},
                    expect=400)
        self.assertIn("error", j)

    def test_oversized_model_name_is_rejected(self):
        r, j = call("post", "/api/settings/agent",
                    {"providers": {"openai": {"model": "x" * 200}}}, expect=400)
        self.assertIn("error", j)

    def test_cloud_mode_records_the_price_acknowledgement(self):
        call("post", "/api/settings/agent", {"backend_mode": "cloud"}, expect=200)
        self.assertTrue(KeyManager().load()["cost_ack"])
        call("post", "/api/settings/agent", {"backend_mode": "local"}, expect=200)
        # local mode never needs the acknowledgement; it just must not break
        self.assertTrue(KeyManager().load()["prefer_cloud"] is False)

    def test_cost_ack_flag_can_be_set_explicitly(self):
        _, j = call("post", "/api/settings/agent", {"cost_ack": False},
                    expect=200)
        self.assertFalse(j["config"]["cost_ack"])
        self.assertFalse(AB.AgentBackend(keys=KeyManager(),
                                         net_state=lambda: "online",
                                         http=lambda *a, **k: {}).should_use_cloud())


class KeyEndpointsTests(SettingsApiCase):
    def test_add_key_returns_masked_entry(self):
        r, j = call("post", "/api/settings/agent/keys",
                    {"provider": "deepseek", "key": RAW_DEEPSEEK, "label": "ds"},
                    expect=200)
        self.assertTrue(j["ok"])
        self.assertEqual(j["key"]["label"], "ds")
        self.assertNotIn(RAW_DEEPSEEK, json.dumps(j))
        self.assertTrue(j["key"]["key_masked"].startswith("sk-d"))
        self.assertEqual(j["config"]["providers"]["deepseek"]["model"],
                         "deepseek-chat")
        # The write is confirmed at a real path, not just accepted in memory.
        self.assertEqual(j["config_path"], str(KeyManager().path))
        self.assertTrue(KeyManager().path.exists())

    def test_short_key_and_unknown_provider_are_json_errors(self):
        for body in ({"provider": "openai", "key": "short"},
                     {"provider": "nope", "key": RAW_OPENAI},
                     {"provider": "openai", "key": ""}):
            r, j = call("post", "/api/settings/agent/keys", body, expect=400)
            self.assertTrue(r.is_json, msg=str(body))
            self.assertIn("error", j)

    def test_duplicate_key_is_stored_once(self):
        for _ in range(3):
            call("post", "/api/settings/agent/keys",
                 {"provider": "openai", "key": RAW_OPENAI}, expect=200)
        self.assertEqual(len(KeyManager().entries("openai")), 1)

    def test_delete_key(self):
        _, j = call("post", "/api/settings/agent/keys",
                    {"provider": "openai", "key": RAW_OPENAI}, expect=200)
        kid = j["key"]["id"]
        _, j = call("delete", "/api/settings/agent/keys/openai/%s" % kid,
                    expect=200)
        self.assertTrue(j["ok"])
        self.assertEqual(j["config"]["keys"]["openai"], [])
        r, j = call("delete", "/api/settings/agent/keys/openai/%s" % kid,
                    expect=404)
        self.assertIn("error", j)

    def test_delete_unknown_provider_is_json(self):
        r, j = call("delete", "/api/settings/agent/keys/nope/k_1", expect=400)
        self.assertTrue(r.is_json)
        self.assertIn("error", j)

    def test_test_endpoint_success(self):
        def fake(url, body, headers, timeout=30):
            return {"choices": [{"message": {"content": "ok"}}]}

        old = AB._http_json
        AB._http_json = fake
        try:
            _, added = call("post", "/api/settings/agent/keys",
                            {"provider": "openai", "key": RAW_OPENAI},
                            expect=200)
            _, j = call("post", "/api/settings/agent/keys/test",
                        {"provider": "openai", "key_id": added["key"]["id"]},
                        expect=200)
        finally:
            AB._http_json = old
        self.assertTrue(j["ok"], msg=str(j))
        self.assertIn("latency_ms", j)
        self.assertNotIn(RAW_OPENAI, json.dumps(j))

    def test_test_endpoint_reports_failure_as_json(self):
        def fake(url, body, headers, timeout=30):
            raise AB.CloudError("auth", "invalid api key " + RAW_OPENAI, 401)

        old = AB._http_json
        AB._http_json = fake
        try:
            _, added = call("post", "/api/settings/agent/keys",
                            {"provider": "openai", "key": RAW_OPENAI},
                            expect=200)
            _, j = call("post", "/api/settings/agent/keys/test",
                        {"provider": "openai", "key_id": added["key"]["id"]},
                        expect=200)
        finally:
            AB._http_json = old
        self.assertFalse(j["ok"])
        self.assertEqual(j["kind"], "auth")
        self.assertNotIn(RAW_OPENAI, json.dumps(j))

    def test_test_endpoint_needs_a_key(self):
        r, j = call("post", "/api/settings/agent/keys/test",
                    {"provider": "openai"}, expect=400)
        self.assertIn("error", j)

    def test_test_endpoint_rejects_unknown_provider(self):
        r, j = call("post", "/api/settings/agent/keys/test",
                    {"provider": "nope", "key": "sk-abcdefgh"}, expect=400)
        self.assertIn("error", j)


class ActivateKeyTests(SettingsApiCase):
    """POST /api/settings/agent/keys/<provider>/<key_id>/activate.

    Manual key selection: one key per provider is "the one in play", the
    Settings radio shows it, and the status bar names it (masked).
    """

    def add(self, provider="openai", key=RAW_OPENAI, label="a"):
        _, j = call("post", "/api/settings/agent/keys",
                    {"provider": provider, "key": key, "label": label},
                    expect=200)
        return j["key"]["id"]

    def test_activate_marks_exactly_one_key(self):
        first = self.add(key=RAW_OPENAI)
        second = self.add(key="sk-live-second-key-9876543210", label="b")

        _, j = call("post",
                    "/api/settings/agent/keys/openai/%s/activate" % second,
                    expect=200)
        self.assertTrue(j["ok"])
        self.assertTrue(j["key"]["active"])
        self.assertEqual(j["key"]["id"], second)

        flags = {e["id"]: e["active"] for e in j["config"]["keys"]["openai"]}
        self.assertEqual(flags[second], True)
        self.assertEqual(flags[first], False)
        # The tick follows the write, not just this response.
        self.assertEqual(KeyManager().active_entry("openai")["id"], second)
        self.assertNotIn(RAW_OPENAI, json.dumps(j))

    def test_get_settings_shows_the_active_key_masked(self):
        kid = self.add()
        call("post", "/api/settings/agent/keys/openai/%s/activate" % kid,
             expect=200)
        _, j = call("get", "/api/settings/agent", expect=200)
        rows = j["config"]["keys"]["openai"]
        self.assertEqual([e["id"] for e in rows if e["active"]], [kid])
        self.assertTrue(rows[0]["key_masked"].startswith("sk-l"))
        self.assertNotIn(RAW_OPENAI, json.dumps(j))

    def test_activate_a_second_key_moves_the_tick(self):
        first = self.add()
        second = self.add(key="sk-live-mover-key-0000000001", label="b")
        call("post", "/api/settings/agent/keys/openai/%s/activate" % first,
             expect=200)
        _, j = call("post",
                    "/api/settings/agent/keys/openai/%s/activate" % second,
                    expect=200)
        active = [e["id"] for e in j["config"]["keys"]["openai"] if e["active"]]
        self.assertEqual(active, [second])
        self.assertNotEqual(active[0], first)

    def test_unknown_key_is_404_json(self):
        self.add()
        r, j = call("post", "/api/settings/agent/keys/openai/k_nope/activate",
                    expect=404)
        self.assertTrue(r.is_json)
        self.assertIn("error", j)

    def test_unknown_provider_is_400_json(self):
        r, j = call("post", "/api/settings/agent/keys/nope/k_1/activate",
                    expect=400)
        self.assertTrue(r.is_json)
        self.assertIn("error", j)


class ErrorShapeTests(unittest.TestCase):
    """The bug the user hit: a non-JSON body reaching `res.json()`."""

    def test_unknown_api_path_is_json_not_html(self):
        r, j = call("get", "/api/settings/agent/nope", expect=404)
        self.assertTrue(r.is_json, msg=r.content_type)
        self.assertIn("error", j)
        self.assertNotIn("<html", j["error"].lower())

    def test_unknown_api_path_anywhere_is_json(self):
        r, j = call("get", "/api/definitely-not-a-route", expect=404)
        self.assertTrue(r.is_json)
        self.assertIn("error", j)

    def test_wrong_method_on_a_settings_route_is_json(self):
        r, j = call("delete", "/api/settings/agent", expect=405)
        self.assertTrue(r.is_json, msg=r.content_type)
        self.assertIn("error", j)

    def test_malformed_json_body_is_not_a_500(self):
        with closing(client.post("/api/settings/agent",
                                 data="{not json",
                                 content_type="application/json")) as r:
            # json_body() is lenient: {} means "nothing to change"
            self.assertEqual(r.status_code, 200)
            self.assertTrue(r.is_json)

    def test_pwa_and_static_404s_keep_their_html(self):
        """The JSON rule is for API paths only — the shell still renders."""
        with closing(client.get("/nope-page")) as r:
            self.assertEqual(r.status_code, 404)
            self.assertFalse(r.is_json, msg=r.content_type)

    def test_404_body_starts_with_the_old_plaintext_symptom(self):
        """Documents the original bug: this is what the UI tried to parse."""
        with closing(client.get("/nope-page")) as r:
            body = r.get_data(as_text=True)
        self.assertTrue(body.startswith("The requested URL"), msg=body[:40])
        # …while the API twin of that same 404 is JSON.
        r2, j2 = call("get", "/api/settings/agent/nope", expect=404)
        self.assertTrue(r2.is_json)
        self.assertIn("requested URL", j2["error"])




class ModelIndicatorTests(SettingsApiCase):
    """GET /api/settings/model — what the next turn would use (config).

    The status bar itself reads /api/agent/status now; this endpoint stays as
    the prediction the Settings page shows before a turn is sent.
    """

    def use_backend(self, online=True, mode="auto", with_key=True):
        """Pin a backend with a deterministic network answer for this test."""
        import routes.settings_api as SA
        km = KeyManager()
        if with_key:
            km.add_key("openai", "sk-indicator-000000000000", "i")
            cfg = km.load()
            cfg["cost_ack"] = True
            cfg["prefer_cloud"] = mode != "local"
            cfg["backend_mode"] = mode
            km.save()
        else:
            km.set_mode(mode)
            km.save()
        be = AB.AgentBackend(keys=km,
                             net_state=(lambda: "online" if online else "offline"),
                             http=lambda *a, **k: {})
        old, old_name = SA._backend, SA.friendly_local_name
        SA._backend = lambda: be
        SA.friendly_local_name = lambda: "Gemma 4"
        self.addCleanup(lambda: (setattr(SA, "_backend", old),
                                 setattr(SA, "friendly_local_name", old_name)))
        return be

    def test_local_label(self):
        self.use_backend(mode="local")
        _, j = call("get", "/api/settings/model?mode=agent", expect=200)
        self.assertEqual(j["label"], "Model: Gemma 4 (Local)")
        self.assertEqual(j["backend"], "local")
        self.assertEqual(j["provider"], "")
        self.assertFalse(j["cloud"])

    def test_cloud_label_carries_the_model_and_provider(self):
        self.use_backend(mode="cloud")
        _, j = call("get", "/api/settings/model?mode=agent", expect=200)
        self.assertEqual(j["backend"], "cloud")
        self.assertEqual(j["provider"], "openai")
        self.assertEqual(j["model"], "gpt-4o-mini")
        self.assertEqual(j["label"], "Model: gpt-4o-mini (Cloud)")
        self.assertTrue(j["cloud"])

    def test_auto_mode_reports_the_cloud_when_it_will_be_used(self):
        self.use_backend(mode="auto")
        _, j = call("get", "/api/settings/model?mode=agent", expect=200)
        self.assertEqual(j["backend"], "cloud", j)

    def test_offline_auto_reports_the_local_model(self):
        self.use_backend(mode="auto", online=False)
        _, j = call("get", "/api/settings/model?mode=agent", expect=200)
        self.assertEqual(j["backend"], "local", j)
        self.assertEqual(j["label"], "Model: Gemma 4 (Local)")

    def test_no_key_means_local_in_auto(self):
        self.use_backend(mode="auto", with_key=False)
        _, j = call("get", "/api/settings/model?mode=agent", expect=200)
        self.assertEqual(j["backend"], "local", j)

    def test_chat_is_always_local(self):
        self.use_backend(mode="cloud")
        _, j = call("get", "/api/settings/model?mode=chat", expect=200)
        self.assertEqual(j["backend"], "local")
        self.assertEqual(j["label"], "Model: Gemma 4 (Local)")

    def test_label_format_is_pinned(self):
        self.use_backend(mode="cloud")
        _, j = call("get", "/api/settings/model", expect=200)
        self.assertRegex(j["label"], r"^Model: .+ \((Local|Cloud)\)$")

    def test_no_key_material_in_the_indicator(self):
        self.use_backend(mode="cloud")
        raw = call("get", "/api/settings/model?mode=agent", expect=200)[1]
        flat = json.dumps(raw, ensure_ascii=False)
        self.assertNotIn("sk-indicator-000000000000", flat)

    def test_missing_endpoint_still_answers_json(self):
        _, j = call("get", "/api/settings/nope", expect=404)
        self.assertIn("error", j)

if __name__ == "__main__":
    unittest.main()
