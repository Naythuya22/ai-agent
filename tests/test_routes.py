"""Route-level smoke tests: every endpoint answers, and answers sanely.

Run:  python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import unittest

from contextlib import closing

from _harness import bootstrap, get_app

MODEL = bootstrap()
app = get_app()
client = app.test_client()

# Imported after bootstrap(): HOME and the litert_lm stub must be in place.
import litert_lm  # noqa: E402
from core.llm import (  # noqa: E402
    build_vision_message,
    clear_vision_cache,
    supports_vision,
)
from routes.chat import IMAGE_MAX_BYTES  # noqa: E402

# (method, path, query, json-body, expected-status)
# expected None = "any status, but must not raise"
CASES = [
    ("GET", "/", None, None, 200),
    ("GET", "/api/health", None, None, 200),
    ("GET", "/api/sw-version", None, None, 200),
    ("GET", "/api/thermal", None, None, 200),
    ("GET", "/api/net", None, None, 200),
    ("GET", "/api/net", {"refresh": "1"}, None, 200),
    ("GET", "/manifest.json", None, None, 200),
    ("GET", "/sw.js", None, None, 200),
    ("GET", "/v1/models", None, None, 200),
    ("GET", "/api/mode", None, None, 200),
    ("GET", "/api/agent/status", None, None, 200),
    ("GET", "/api/ctx", None, None, 200),
    ("GET", "/api/memory", None, None, 200),
    ("GET", "/api/profile", None, None, 200),
    ("GET", "/api/sessions", None, None, 200),
    ("GET", "/api/sessions", {"mode": "chat"}, None, 200),
    ("GET", "/api/sessions/nope", None, None, 404),
    ("GET", "/api/history", None, None, 200),
    ("GET", "/api/skills", None, None, 200),
    ("GET", "/api/skills/nope", None, None, 404),
    ("GET", "/api/code/projects", None, None, 200),
    ("GET", "/api/code/browse", None, None, 400),  # default dir absent in test HOME
    ("GET", "/api/code/browse", {"path": "/definitely/not/here"}, None, 400),
    ("GET", "/api/code/web/history", None, None, 200),
    ("GET", "/api/code/web/status", None, None, 200),
    ("GET", "/icons/icon-192.png", None, None, 200),
    ("GET", "/static/style.css", None, None, 200),  # served from web/
    ("GET", "/static/../../server.py", None, None, 404),  # no path traversal
    ("GET", "/nope-does-not-exist", None, None, 404),

    # Missing / empty payload must be a clean 400, never a 500.
    ("POST", "/api/chat", None, {}, 400),
    ("POST", "/api/chat", {"message": "   "}, None, 400),
    ("POST", "/api/agent", None, {}, 400),
    ("POST", "/api/agent/stream", None, {}, 400),
    ("POST", "/api/chat/stream", None, {}, 400),
    ("POST", "/api/remember", None, {}, 400),
    ("POST", "/api/profile", None, {}, 400),
    ("POST", "/api/code", None, {}, 400),
    ("POST", "/api/code/web/run", None, {}, 400),
    ("POST", "/api/code/web/run", {"message": "/project list"}, None, None),
    ("POST", "/api/code/web/run/stream", None, {}, 400),
    ("POST", "/api/skills/create", None, {}, 400),
    ("POST", "/api/skills/create", {"name": "x"}, None, 400),
    ("POST", "/api/skills/x/run", None, {}, 400),
    ("POST", "/api/skills/x/run", None, {"args": "a", "mode": "bogus"}, 400),
    ("POST", "/api/skills/x/run", None, {"args": "a", "mode": "chat"}, 404),
    ("POST", "/api/context/1234", None, None, 400),
    ("POST", "/api/context/abc", None, None, 400),
    ("POST", "/v1/chat/completions", None, {}, 400),
    ("POST", "/v1/chat/completions", None, {"messages": [{"role": "user", "content": "hi"}]}, 200),

    # Write endpoints
    ("POST", "/api/clear", None, None, 200),
    ("POST", "/api/summary", None, {}, 400),  # empty session: "too few messages"
    ("POST", "/api/new", None, {}, None),
    ("POST", "/api/code/web/clear", None, None, 200),
    ("POST", "/api/code/projects", None, {}, 400),
    ("POST", "/api/code/projects/zz/activate", None, None, 404),
    ("DELETE", "/api/code/projects/zz", None, None, None),
    ("DELETE", "/api/skills/nope", None, None, 404),

    # Bulk session delete (sidebar multi-select): bad payloads are 400, never
    # 500, and the 201-id case proves the cap is enforced after de-duplication.
    ("POST", "/api/sessions/bulk_delete", None, {}, 400),
    ("POST", "/api/sessions/bulk_delete", None, {"ids": "chat"}, 400),
    ("POST", "/api/sessions/bulk_delete", None, {"ids": [1, 2]}, 400),
    ("POST", "/api/sessions/bulk_delete", None, {"ids": []}, 400),
    ("POST", "/api/sessions/bulk_delete", None, {"ids": [None]}, 400),
    ("POST", "/api/sessions/bulk_delete", None,
     {"ids": ["sid-%d" % i for i in range(201)]}, 400),

    # Wrong method must be 405, never silently 200.
    ("POST", "/api/health", None, None, 405),
    ("POST", "/api/net", None, None, 405),
    ("GET", "/api/agent", None, None, 405),
]


class TestRoutes(unittest.TestCase):
    def test_all_cases(self):
        failures = []
        for method, path, query, body, expect in CASES:
            kwargs = {}
            if query:
                kwargs["query_string"] = query
            if body is not None:
                kwargs["json"] = body
            label = f"{method} {path}" + (f" ?{query}" if query else "")
            try:
                with closing(client.open(path, method=method, **kwargs)) as r:
                    if expect is not None and r.status_code != expect:
                        failures.append(
                            f"{label}: expected {expect}, got {r.status_code} "
                            f"body={r.get_data(as_text=True)[:120]!r}"
                        )
            except Exception as e:  # noqa: BLE001
                failures.append(f"{label}: RAISED {type(e).__name__}: {e}")
                continue
        self.assertEqual(failures, [], "\n".join(failures))

    def test_unknown_route_is_404_json_or_html(self):
        with closing(client.get("/totally/unknown")) as r:
            self.assertEqual(r.status_code, 404)

    def test_api_net_shape(self):
        with closing(client.get("/api/net")) as r:
            self.assertEqual(r.status_code, 200)
            body = r.get_json()
        for key in ("state", "online", "checked_at", "age_s"):
            self.assertIn(key, body)
        self.assertIn(body["state"], ("online", "offline", "unknown"))

    def test_health_is_ok(self):
        with closing(client.get("/api/health")) as r:
            body = r.get_json()
        self.assertEqual(body.get("status"), "ok", body)

    def test_sw_version_matches_sw_js(self):
        """The service worker must advertise the same cache name it serves."""
        import re

        with closing(client.get("/api/sw-version")) as r:
            version = r.get_json()["version"]
        with closing(client.get("/sw.js")) as r:
            js = r.get_data(as_text=True)
        m = re.search(r"const CACHE = 'ai-agent-([^']+)'", js)
        self.assertIsNotNone(m, "sw.js has no CACHE constant")
        self.assertEqual(version, m.group(1))

    # ── B1: service-worker cache invalidation on deploy ──────────────
    @staticmethod
    def _index_hash():
        import hashlib
        from pathlib import Path

        web = Path(__file__).resolve().parent.parent / "web"
        return hashlib.sha256((web / "index.html").read_bytes()).hexdigest()[:8]

    def test_sw_cache_name_is_index_html_hash(self):
        """A frontend deploy changes index.html, hence sw.js, hence CACHE."""
        expected = self._index_hash()
        with closing(client.get("/sw.js")) as r:
            js = r.get_data(as_text=True)
        self.assertNotIn("__CACHE_VERSION__", js,
                         "placeholder shipped unsubstituted — B1 is back")
        self.assertIn(f"const CACHE = 'ai-agent-{expected}';", js,
                      "CACHE is not derived from index.html content")
        with closing(client.get("/api/sw-version")) as r:
            self.assertEqual(r.get_json()["version"], expected)

    def test_sw_js_is_always_revalidated(self):
        """/sw.js must reach the browser on every navigation (no stale SW)."""
        with closing(client.get("/sw.js")) as r:
            self.assertEqual(r.status_code, 200)
            self.assertIn("javascript", r.headers.get("Content-Type", ""))
            cache_control = r.headers.get("Cache-Control", "")
        self.assertIn("no-cache", cache_control)
        self.assertIn("must-revalidate", cache_control)

    def test_sw_drops_stale_caches_and_fetches_html_from_network(self):
        """Activate purges old ai-agent-* caches; documents are network-first."""
        with closing(client.get("/sw.js")) as r:
            js = r.get_data(as_text=True)

        self.assertIn(".filter((k) => k.startsWith(CACHE_PREFIX) && k !== CACHE)",
                      js, "activate handler does not purge stale caches")
        self.assertIn("self.clients.claim()", js)

        head, _, tail = js.partition("const isDocument")
        self.assertTrue(head and tail, "sw.js has no document branch")
        body, _, _ = tail.partition("return;")
        net = body.find("fetch(req)")
        cached = body.find("caches.match(req)")
        self.assertNotEqual(net, -1, "documents are not fetched from the network")
        self.assertNotEqual(cached, -1, "no offline fallback for documents")
        self.assertLess(net, cached, "documents are served cache-first (B1)")

    def test_debug_mode_rereads_index_hash_per_request(self):
        """AI_AGENT_DEBUG=1 must bypass the startup snapshot, else dev edits
        never reach the browser without a restart."""
        import config
        from routes import pwa

        saved_debug, saved_start = config.DEBUG, pwa._STARTUP_HASH
        try:
            pwa._STARTUP_HASH = "stale000"
            config.DEBUG = False
            with closing(client.get("/api/sw-version")) as r:
                self.assertEqual(r.get_json()["version"], "stale000")

            config.DEBUG = True
            with closing(client.get("/api/sw-version")) as r:
                self.assertEqual(r.get_json()["version"], self._index_hash())
        finally:
            config.DEBUG, pwa._STARTUP_HASH = saved_debug, saved_start

    def test_index_html_is_served(self):
        with closing(client.get("/")) as r:
            body = r.get_data(as_text=True)
        self.assertIn("<html", body.lower())
        self.assertIn("/api/net", body, "UI should poll /api/net")

    def test_every_json_error_is_parseable(self):
        """A client parsing errors must never hit invalid JSON."""
        for method, path, _q, body, _e in CASES:
            kwargs = {"json": body} if body is not None else {}
            with closing(client.open(path, method=method, **kwargs)) as r:
                ctype = (r.headers.get("Content-Type") or "")
                if "json" in ctype:
                    try:
                        json.loads(r.get_data(as_text=True))
                    except Exception as e:  # noqa: BLE001
                        self.fail(f"{method} {path}: bad JSON ({e})")


class TestAgentImageUpload(unittest.TestCase):
    """POST /api/agent/upload — mime, size, emptiness and the happy path."""

    # Content is never decoded by the server (only the mime type is checked).
    PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64

    def _post(self, filename, mimetype, data):
        from io import BytesIO

        return client.post(
            "/api/agent/upload",
            data={"file": (BytesIO(data), filename, mimetype)},
            content_type="multipart/form-data",
        )

    def test_bad_mime_is_400(self):
        with closing(self._post("notes.txt", "text/plain", b"hello")) as r:
            self.assertEqual(r.status_code, 400)
            body = r.get_json()
        self.assertIn("unsupported", body["error"])

    def test_oversize_is_400(self):
        too_big = b"x" * (IMAGE_MAX_BYTES + 1)
        with closing(self._post("big.png", "image/png", too_big)) as r:
            self.assertEqual(r.status_code, 400)
        self.assertIn("5MB", r.get_json()["error"])

    def test_empty_is_400(self):
        with closing(self._post("empty.png", "image/png", b"")) as r:
            self.assertEqual(r.status_code, 400)
        self.assertIn("empty", r.get_json()["error"])

    def test_valid_is_200_with_id(self):
        with closing(self._post("pic.png", "image/png", self.PNG)) as r:
            self.assertEqual(r.status_code, 200)
            body = r.get_json()
        self.assertTrue(body["id"].endswith(".png"))
        self.assertEqual(body["name"], "pic.png")
        self.assertEqual(body["size"], len(self.PNG))
        self.assertTrue(body["vision"])

    def test_missing_file_is_400(self):
        with closing(client.post(
            "/api/agent/upload", data={}, content_type="multipart/form-data"
        )) as r:
            self.assertEqual(r.status_code, 400)

    def test_vision_unavailable_reports_reason_and_ocr(self):
        """vision=False → 400 that says why, plus an OCR fallback."""
        litert_lm.VISION_SUPPORTED = False
        clear_vision_cache()
        try:
            with closing(self._post("pic.png", "image/png", self.PNG)) as r:
                self.assertEqual(r.status_code, 400)
                body = r.get_json()
            self.assertIn("vision unavailable", body["error"])
            self.assertIn("vision=False", body["error"])
            self.assertIn("OCR", body["suggestion"])
        finally:
            litert_lm.VISION_SUPPORTED = True
            clear_vision_cache()

    def test_agent_stream_unknown_image_is_400(self):
        with closing(client.post(
            "/api/agent/stream",
            json={"message": "hi", "image_id": "nope.png"},
        )) as r:
            self.assertEqual(r.status_code, 400)

    def test_agent_stream_without_image_still_works(self):
        """image_id is optional — plain agent turns must be unaffected."""
        with closing(client.post(
            "/api/agent/stream", json={"message": "", "image_id": "x.png"}
        )) as r:
            self.assertEqual(r.status_code, 400)  # empty message, not image


class TestSupportsVision(unittest.TestCase):
    """core.llm.supports_vision / build_vision_message helpers."""

    def tearDown(self):
        litert_lm.VISION_SUPPORTED = True
        clear_vision_cache()

    def test_missing_model_path(self):
        ok, reason = supports_vision("")
        self.assertFalse(ok)
        self.assertIn("no model file", reason)
        ok, reason = supports_vision("/definitely/missing.litertlm")
        self.assertFalse(ok)
        self.assertIn("not found", reason)

    def test_stub_without_capabilities_is_not_fatal(self):
        """The litert_lm import is guarded: no Capabilities → no crash."""
        saved = litert_lm.Capabilities
        litert_lm.Capabilities = None
        clear_vision_cache()
        try:
            ok, reason = supports_vision(MODEL)
            self.assertFalse(ok)
            self.assertIn("Capabilities unavailable", reason)
        finally:
            litert_lm.Capabilities = saved
            clear_vision_cache()

    def test_vision_true_by_default(self):
        ok, reason = supports_vision(MODEL)
        self.assertTrue(ok, reason)
        self.assertEqual(reason, "")

    def test_vision_false_is_explained(self):
        litert_lm.VISION_SUPPORTED = False
        clear_vision_cache()
        ok, reason = supports_vision(MODEL)
        self.assertFalse(ok)
        self.assertIn("vision=False", reason)

    def test_build_vision_message_carries_the_image(self):
        import tempfile
        from pathlib import Path

        img = Path(tempfile.mkdtemp()) / "x.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")
        msg = build_vision_message(MODEL, str(img), "မြင်လား")
        self.assertIsNotNone(msg)
        payload = msg.to_json()
        self.assertEqual(payload[0], {"type": "image", "path": str(img)})
        self.assertEqual(payload[1], {"type": "text", "text": "မြင်လား"})

    def test_build_vision_message_is_none_without_vision(self):
        litert_lm.VISION_SUPPORTED = False
        clear_vision_cache()
        self.assertIsNone(build_vision_message(MODEL, "/tmp/x.png", "hi"))


class TestVisionEngineLoad(unittest.TestCase):
    """LLM.load() must create the vision executor.

    Regression: sending an image used to raise
    ``RuntimeError: INVALID_ARGUMENT: Vision executor should not be null,
    please TryLoadingVisionExecutor() first.`` because Engine() was built
    without a ``vision_backend``.
    """

    def tearDown(self):
        litert_lm.VISION_SUPPORTED = True
        clear_vision_cache()

    def _load(self):
        from config import Config
        from core.llm import LLM

        llm = LLM(Config.load())
        llm.load()
        return llm

    def test_engine_gets_a_vision_backend(self):
        llm = self._load()
        try:
            last = litert_lm.LOADED[-1]
            self.assertEqual(
                last["vision_backend"], "CPU",
                "vision executor must be requested at load time",
            )
        finally:
            llm.unload()

    def test_text_only_model_skips_the_vision_backend(self):
        litert_lm.VISION_SUPPORTED = False
        clear_vision_cache()
        llm = self._load()
        try:
            self.assertIsNone(litert_lm.LOADED[-1]["vision_backend"])
        finally:
            llm.unload()


class TestSessionBulkDelete(unittest.TestCase):
    """POST /api/sessions/bulk_delete — one round trip for the multi-select."""

    def _create(self, mode="chat", title="bulk"):
        with closing(client.post("/api/sessions",
                                 json={"mode": mode, "title": title})) as r:
            self.assertEqual(r.status_code, 200, r.get_data(as_text=True))
            return r.get_json()["id"]

    def _bulk(self, ids, expect=200):
        with closing(client.post("/api/sessions/bulk_delete",
                                 json={"ids": ids})) as r:
            self.assertEqual(r.status_code, expect, r.get_data(as_text=True))
            return r.get_json()

    def _listed_ids(self):
        with closing(client.get("/api/sessions")) as r:
            return [s["id"] for s in r.get_json()["sessions"]]

    def test_deletes_every_requested_id(self):
        ids = [self._create() for _ in range(3)]
        body = self._bulk(ids)
        self.assertTrue(body["ok"])
        self.assertEqual(sorted(body["deleted"]), sorted(ids))
        self.assertEqual(body["missing"], [])
        left = self._listed_ids()
        for sid in ids:
            self.assertNotIn(sid, left)

    def test_missing_ids_are_reported_not_raised(self):
        sid = self._create()
        body = self._bulk([sid, "does-not-exist"])
        self.assertEqual(body["deleted"], [sid])
        self.assertEqual(body["missing"], ["does-not-exist"])

    def test_duplicate_ids_delete_once(self):
        sid = self._create()
        body = self._bulk([sid, sid, sid])
        self.assertEqual(body["deleted"], [sid])
        self.assertEqual(body["missing"], [])

    def test_active_thread_moves_to_a_live_session(self):
        from routes._state import get_chat_mode

        self._create(title="keep")
        gone = self._create(title="gone")  # last created == active
        mode = get_chat_mode()
        self.assertEqual(mode.session_id, gone)
        body = self._bulk([gone])
        self.assertNotEqual(body["active_chat"], gone)
        # the mode must be showing the session the store now calls active
        self.assertEqual(mode.session_id, body["active_chat"])

    def test_deleting_the_last_session_resets_the_mode(self):
        from routes._state import get_chat_mode

        chat_ids = [s["id"] for s in _list_json("chat")]
        if not chat_ids:
            chat_ids = [self._create()]
        body = self._bulk(chat_ids)
        self.assertEqual(sorted(body["deleted"]), sorted(chat_ids))
        # reset() creates a fresh thread, so active_chat is never None — what
        # matters is that it never points at a session that no longer exists.
        self.assertNotIn(body["active_chat"], chat_ids)
        self.assertNotIn(get_chat_mode().session_id, chat_ids)


def _list_json(mode):
    with closing(client.get("/api/sessions", query_string={"mode": mode})) as r:
        return r.get_json()["sessions"]


class TestAgentUsageIndicator(unittest.TestCase):
    """/api/agent/status — the status-bar chip, from what really ran.

    The old chip read the settings config and printed a cloud model name even
    when no key had ever been used, so these tests pin the record itself.
    """

    def setUp(self):
        from modes import agent as agent_mod
        self.agent_mod = agent_mod
        self.saved = dict(agent_mod._USAGE)
        # Every test starts from "no turn yet" — earlier modules (the cloud
        # integration tests) leave a real record behind.
        self.agent_mod._USAGE.update({"backend": "", "model": "", "provider": "",
                                      "mode": "", "label": "", "at": 0.0})

    def tearDown(self):
        self.agent_mod._USAGE.clear()
        self.agent_mod._USAGE.update(self.saved)

    def _get(self):
        with closing(client.get("/api/agent/status")) as r:
            self.assertEqual(r.status_code, 200)
            return r.get_json()

    def test_labels_describe_what_ran(self):
        from modes.agent import usage_label

        self.assertEqual(usage_label("cloud", "deepseek-chat", "auto"),
                         "Model: deepseek-chat (Cloud, last used)")
        self.assertEqual(usage_label("local", "Gemma 4", "auto"),
                         "Model: Gemma 4 (Local, fallback)")
        self.assertEqual(usage_label("local", "Gemma 4", "local"),
                         "Model: Gemma 4 (Local)")
        # Nothing ran: never a name lifted from the config.
        self.assertEqual(usage_label("", "", "cloud"), "")

    def test_status_reports_the_record(self):
        from modes.agent import record_usage

        record_usage("cloud", "deepseek-chat", "deepseek", "auto")
        j = self._get()
        self.assertEqual(j["last_model"], "deepseek-chat")
        self.assertEqual(j["last_backend"], "cloud")
        self.assertEqual(j["provider"], "deepseek")
        self.assertEqual(j["label"], "Model: deepseek-chat (Cloud, last used)")

    def test_status_is_blank_before_any_turn(self):
        j = self._get()
        self.assertEqual(j["label"], "")
        self.assertEqual(j["last_backend"], "")
        self.assertEqual(j["last_model"], "")

    def test_record_is_module_state_so_a_rebuild_keeps_it(self):
        """rebuild_modes() throws the AgentMode away — the chip must not go."""
        from modes.agent import last_usage, record_usage

        record_usage("local", "Gemma 4", "", "local")
        # No instance involved: the record is read straight off the module.
        self.assertEqual(last_usage()["label"], "Model: Gemma 4 (Local)")
        self.assertEqual(self._get()["label"], "Model: Gemma 4 (Local)")

    # ---------- the second chip: which API key is in play ----------

    def _wipe_keys(self):
        from core.key_manager import PROVIDERS, KeyManager

        km = KeyManager()
        for p in PROVIDERS:
            for e in km.entries(p):
                km.remove_key(p, e["id"])
        return km

    def _no_keys(self):
        """Start from an empty key file — and put it back afterwards.

        The cleanup must not re-register itself: a cleanup that schedules
        another copy of itself runs until the suite is killed.
        """
        km = self._wipe_keys()
        self.addCleanup(self._wipe_keys)
        return km

    def test_status_names_the_key_that_is_selected(self):
        from core.key_manager import mask

        key = "sk-status-live-abcdef123456"
        self._no_keys().add_key("deepseek", key, "ds")
        j = self._get()
        # Masked through KeyManager — the same mask the Settings page shows.
        self.assertEqual(j["key_label"], "Key: %s (Active)" % mask(key))
        self.assertTrue(j["key_label"].startswith("Key: sk-s"))
        self.assertNotIn(key, json.dumps(j))

    def test_status_is_silent_about_keys_when_there_are_none(self):
        self._no_keys()
        self.assertEqual(self._get()["key_label"], "")

    def test_status_names_the_pinned_key_not_just_any_key(self):
        km = self._no_keys()
        km.add_key("deepseek", "sk-pin-one-aaaaaaaaaaaaaaa")
        pinned = km.add_key("deepseek", "sk-pin-two-bbbbbbbbbbbbbbbb")
        km.set_active_key("deepseek", pinned["id"])
        j = self._get()
        self.assertTrue(j["key_label"].endswith("bbbb (Active)"),
                        j["key_label"])
        self.assertNotIn("sk-pin-one", json.dumps(j))
        self.assertNotIn("sk-pin-two", json.dumps(j))


if __name__ == "__main__":
    unittest.main(verbosity=2)