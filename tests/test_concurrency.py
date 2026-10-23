"""Concurrency tests.

The server runs threaded=True, so several requests can be inside the Flask app
at once. The LiteRT-LM engine and the cached Conversation objects are shared
mutable state, so model work must be serialised. These tests fail loudly if
someone removes the lock.

Without a model these still work: generation is stubbed, and the stub is slow
enough (via sleep) that an overlap would be observable.
"""
from __future__ import annotations

import threading
import time
import unittest

from _harness import bootstrap, get_app

bootstrap()
app = get_app()
client = app.test_client()

WORKERS = 6


class _OverlapDetector:
    """Wraps the agent stream's conversation seam to record concurrent entry.

    Release 22: this used to patch ChatSession.send_stream behind
    /api/chat/stream; Chat Mode is gone, so the probe now patches
    routes.chat.stream_conversation — the seam the agent stream's local
    branch drives — behind /api/agent/stream. Same contract: if the
    turn locks were removed, concurrent turns would overlap here.
    """

    def __init__(self, delay=0.15):
        self.delay = delay
        self.active = 0
        self.peak = 0
        self.overlaps: list[str] = []
        self._lock = threading.Lock()
        self._orig = None

    def install(self):
        import routes.chat as rc

        self._orig = rc.stream_conversation
        me = self

        def probe(conv, msg, *a, **kw):
            with me._lock:
                me.active += 1
                me.peak = max(me.peak, me.active)
                if me.active > 1:
                    me.overlaps.append(str(msg))
            try:
                time.sleep(me.delay)
                for i in range(3):
                    yield ("text", f"tok{i} ")
                yield ("end", {"text": f"reply to {msg}", "tool_calls": None})
            finally:
                with me._lock:
                    me.active -= 1

        rc.stream_conversation = probe
        return self

    def remove(self):
        import routes.chat as rc

        if self._orig is not None:
            rc.stream_conversation = self._orig
            self._orig = None


class TestInferenceSerialisation(unittest.TestCase):
    def setUp(self):
        self.detector = _OverlapDetector().install()
        self.results: dict[int, object] = {}
        self.errors: dict[int, str] = {}

    def tearDown(self):
        self.detector.remove()

    def _fire(self, n=WORKERS, path="/api/agent/stream", key="message"):
        def hit(i):
            try:
                r = client.post(path, json={key: f"msg-{i}"})
                # MUST drain the body. With TESTING=True Werkzeug does not
                # buffer, so a response that is never read leaves the SSE
                # generator suspended on its last yield — still holding the
                # inference lock. A real HTTP client always reads to EOF, and
                # a real disconnect closes the iterator, which releases it.
                self.results[i] = r.get_data(as_text=True)
            except Exception as e:  # noqa: BLE001
                self.errors[i] = f"{type(e).__name__}: {e}"

        threads = [threading.Thread(target=hit, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        return threads

    def test_streams_do_not_interleave(self):
        threads = self._fire()
        self.assertFalse(any(t.is_alive() for t in threads), "a request hung (deadlock)")
        self.assertEqual(self.errors, {}, f"request raised: {self.errors}")
        self.assertEqual(
            self.detector.peak,
            1,
            f"peak concurrent generations = {self.detector.peak}; "
            f"overlapped on: {self.detector.overlaps}",
        )
        self.assertEqual(self.detector.overlaps, [])

    def test_all_clients_get_a_complete_answer(self):
        """Each client must get its own answer, unmixed."""
        threads = self._fire()
        self.assertFalse(any(t.is_alive() for t in threads))
        for i, body in self.results.items():
            self.assertIn('"end"', body, f"request {i} never finished:\n{body[:300]}")
            self.assertIn(f"reply to msg-{i}", body, "answer mixed up between clients")
            other = [j for j in self.results if j != i]
            for j in other:
                self.assertNotIn(
                    f"reply to msg-{j}", body,
                    f"request {i} also contains request {j}'s answer",
                )

    def test_non_stream_agent_does_not_deadlock(self):
        """The non-SSE path shares the same locks (Release 22: was /api/chat)."""
        def hit(i):
            try:
                r = client.post("/api/agent", json={"message": f"n{i}"})
                self.results[i] = r.status_code
            except Exception as e:  # noqa: BLE001
                self.errors[i] = f"{type(e).__name__}: {e}"

        threads = [threading.Thread(target=hit, args=(i,)) for i in range(WORKERS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        self.assertFalse(any(t.is_alive() for t in threads), "deadlock on /api/agent")
        self.assertEqual(self.errors, {})
        self.assertEqual(sorted(self.results.values()), [200] * WORKERS)


class TestNoDeadlock(unittest.TestCase):
    def test_first_request_from_many_threads_loads_once(self):
        """Lazy init must not double-load the engine or deadlock."""
        from routes import _state

        errors = {}

        def hit(i):
            try:
                client.get("/api/ctx")
            except Exception as e:  # noqa: BLE001
                errors[i] = f"{type(e).__name__}: {e}"

        threads = [threading.Thread(target=hit, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        self.assertFalse(any(t.is_alive() for t in threads), "init deadlocked")
        self.assertEqual(errors, {})
        self.assertIsNotNone(_state._llm)
        self.assertIsNotNone(_state._agent_mode)

    def test_lock_is_released_after_use(self):
        from routes._state import inference_lock, inference_guard

        with inference_guard():
            pass
        self.assertFalse(inference_lock._is_owned(), "lock left held")

    def test_lock_is_released_after_client_disconnect(self):
        """An aborted stream must not wedge the lock for the next client.

        Release 22: re-pointed from /api/chat/stream + /api/chat to the
        agent stream + sync agent endpoint (the chat endpoints are gone).
        """
        from routes._state import inference_lock

        r = client.post("/api/agent/stream", json={"message": "x"}, buffered=False)
        next(r.iter_encoded(), None)
        r.close()
        self.assertFalse(inference_lock._is_owned())
        self.assertEqual(
            client.post("/api/agent", json={"message": "after"}).status_code, 200
        )

    def test_nested_guard_is_reentrant(self):
        from routes._state import inference_guard

        with inference_guard():
            with inference_guard():
                pass
        self.assertEqual(client.get("/api/health").status_code, 200)


class TestContextResize(unittest.TestCase):
    def test_resize_works_and_is_restorable(self):
        # Release 22: /api/ctx now reports the AGENT context, whose "total"
        # is the window that would answer next (a provider's, when cloud
        # is configured). The resize changes the LOCAL engine's window,
        # which the meter reports as "local_total" — assert on that.
        original = None
        try:
            original = client.get("/api/ctx").get_json()["local_total"]
        except Exception:  # noqa: BLE001
            original = None

        r = client.post("/api/context/16384")
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True))
        self.assertTrue(r.get_json()["ok"])
        self.assertEqual(client.get("/api/ctx").get_json()["local_total"], 16384)

        client.post(f"/api/context/{original or 8192}")

    def test_resize_rejects_bad_size(self):
        for bad in ("4096", "abc", "0", "-1"):
            r = client.post(f"/api/context/{bad}")
            self.assertEqual(r.status_code, 400, f"/api/context/{bad}")

    def test_same_size_is_a_noop(self):
        before = client.get("/api/ctx").get_json()["local_total"]
        r = client.post(f"/api/context/{before}")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json().get("note"), "already set")

    def test_local_total_reads_the_engine_config(self):
        """Release 22 pin: _local_total() comes off the shared LLM handle.

        Agent Mode's context meter depends on this value; it used to be
        read through the ChatMode object, so the pin is that the number
        tracks the engine config directly.
        """
        from routes._helpers import _local_total
        from routes._state import get_llm

        llm = get_llm(load=False)
        saved = llm.config.context_size
        try:
            llm.config.context_size = 12288
            self.assertEqual(_local_total(), 12288)
        finally:
            llm.config.context_size = saved


if __name__ == "__main__":
    unittest.main(verbosity=2)