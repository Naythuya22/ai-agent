"""Release 32 — video generation via Google Flow (human-in-the-loop).

A. video_provider gate: absent/garbage resolves "off"; only the
   stored value "flow" enables the tools; public() reports it.
B. With the gate off, prepare/import refuse and create nothing.
C. Credit estimates are the values measured on-device 2026-10-27 —
   unmeasured combos report None, never an invented number.
D. Prompt crafting: exact Burmese dialogue embedded; Veo Lite +
   dialogue refused (its audio cannot do Burmese); no-dialogue jobs
   ask for music only.
E. Job lifecycle: prepare records awaiting_human; import picks the
   newest post-job .mp4 from Download, copies it into the library,
   marks the job imported and prints the [video:<id>] marker; a
   taken source file is never reused by a second job.
F. Serving route: known id streams video/mp4; unknown/traversal 404.
G. Settings POST validates video_provider; the PWA carries the
   select, the load/save wiring and the [video:] player rendering.

Run:  sh tests/run.sh -p "test_release32.py"
"""
from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from _harness import bootstrap, get_app

bootstrap()

from core.key_manager import KeyManager  # noqa: E402
from tools import video as V  # noqa: E402
from tools import ALL_TOOLS  # noqa: E402

INDEX = (Path(__file__).resolve().parent.parent / "web" / "index.html")


def set_provider(value):
    km = KeyManager()
    cfg = km.load()
    if value is None:
        cfg.pop("video_provider", None)
    else:
        cfg["video_provider"] = value
    km.save()
    return km


class VideoEnv(unittest.TestCase):
    """Isolate the job store, the library and the Download folder."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="r32-video-")
        root = Path(self._tmp.name)
        self.jobs = root / "jobs.json"
        self.lib = root / "videos"
        self.dl = root / "Download"
        self.dl.mkdir()
        self._env = {
            "AI_AGENT_VIDEO_JOBS": str(self.jobs),
            "AI_AGENT_VIDEOS_DIR": str(self.lib),
            "AI_AGENT_DOWNLOAD_DIRS": str(self.dl),
        }
        self._saved = {k: os.environ.get(k) for k in self._env}
        os.environ.update(self._env)
        self.addCleanup(self._restore)

    def _restore(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._tmp.cleanup()


class TestGate(VideoEnv):
    def tearDown(self):
        set_provider(None)

    def test_absent_defaults_off(self):
        km = set_provider(None)
        self.assertEqual(km.video_provider(), "off")
        self.assertEqual(km.public()["video_provider"], "off")

    def test_garbage_resolves_off(self):
        km = set_provider("banana")
        self.assertEqual(km.video_provider(), "off")

    def test_flow_enables(self):
        km = set_provider("flow")
        self.assertEqual(km.video_provider(), "flow")
        self.assertEqual(km.public()["video_provider"], "flow")

    def test_tools_refuse_when_off(self):
        set_provider(None)
        out = V.prepare_video("a cartoon farmer saving coins")
        self.assertIn("OFF", out)
        self.assertFalse(self.jobs.exists())
        out2 = V.import_video("whatever123")
        self.assertIn("OFF", out2)


class TestEstimates(unittest.TestCase):
    def test_measured_omni_360p(self):
        self.assertEqual(V._estimate_credits("omni", "360p", 6), 5)
        self.assertEqual(V._estimate_credits("omni", "360p", 8), 6)
        self.assertEqual(V._estimate_credits("omni", "360p", 10), 7)

    def test_measured_omni_720p(self):
        self.assertEqual(V._estimate_credits("omni", "720p", 6), 10)
        self.assertEqual(V._estimate_credits("omni", "720p", 8), 12)
        self.assertEqual(V._estimate_credits("omni", "720p", 10), 15)

    def test_veo_lite_flat(self):
        self.assertEqual(V._estimate_credits("veo_lite", "720p", 8), 10)

    def test_unmeasured_is_none_not_invented(self):
        self.assertIsNone(V._estimate_credits("omni", "360p", 4))
        self.assertIsNone(V._estimate_credits("omni", "720p", 4))


class TestPrepare(VideoEnv):
    def setUp(self):
        super().setUp()
        set_provider("flow")
        self.addCleanup(lambda: set_provider(None))

    def test_dialogue_job_recorded(self):
        out = V.prepare_video(
            "a young farmer drops a coin into a clay pot",
            dialogue="ရရင် အရင် စုပါနော်",
        )
        self.assertIn("OK: video job", out)
        jobs = V._load_jobs()
        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        self.assertEqual(job["state"], "awaiting_human")
        self.assertIn("ရရင် အရင် စုပါနော်", job["prompt"])
        self.assertEqual(job["est_credits"], 6)  # omni 360p 8s default
        self.assertIn(job["prompt"], out)  # no clipboard on this host

    def test_veo_lite_dialogue_refused(self):
        out = V.prepare_video("a tea shop", dialogue="မင်္ဂလာပါ",
                              model="veo_lite")
        self.assertIn("cannot generate Burmese speech", out)
        self.assertEqual(V._load_jobs(), [])

    def test_no_dialogue_asks_music_only(self):
        V.prepare_video("waves on a beach", style="cinematic",
                        model="veo_lite")
        job = V._load_jobs()[0]
        self.assertIn("no dialogue", job["prompt"])
        self.assertEqual(job["est_credits"], 10)

    def test_validation(self):
        self.assertIn("Error", V.prepare_video(""))
        self.assertIn("Error", V.prepare_video("x", model="sora"))
        self.assertIn("Error", V.prepare_video("x", duration_s=7))
        self.assertIn("Error", V.prepare_video("x", resolution="4k"))
        self.assertEqual(V._load_jobs(), [])

    def test_status_lists_job(self):
        V.prepare_video("a market scene")
        out = V.video_status()
        self.assertIn("awaiting_human", out)


class TestImport(VideoEnv):
    def setUp(self):
        super().setUp()
        set_provider("flow")
        self.addCleanup(lambda: set_provider(None))
        V.prepare_video("a farmer saving coins")
        self.job = V._load_jobs()[0]

    def _drop_mp4(self, name, when):
        f = self.dl / name
        f.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"x" * 2048)
        os.utime(f, (when, when))
        return f

    def test_import_happy_path(self):
        self._drop_mp4("flow-old.mp4", self.job["created_at"] - 3600)
        new = self._drop_mp4("flow-new.mp4", time.time())
        out = V.import_video(self.job["id"])
        self.assertIn("OK: imported", out)
        self.assertIn(f"[video:{self.job['id']}]", out)
        dest = self.lib / f"{self.job['id']}.mp4"
        self.assertTrue(dest.is_file())
        self.assertEqual(dest.read_bytes(), new.read_bytes())
        job = V._load_jobs()[0]
        self.assertEqual(job["state"], "imported")

    def test_import_none_found(self):
        self._drop_mp4("too-old.mp4", self.job["created_at"] - 3600)
        out = V.import_video(self.job["id"])
        self.assertIn("no new .mp4", out)
        self.assertEqual(V._load_jobs()[0]["state"], "awaiting_human")

    def test_import_unknown_job(self):
        self.assertIn("Error", V.import_video("nope123"))

    def test_source_not_reused_by_second_job(self):
        self._drop_mp4("only-one.mp4", time.time())
        self.assertIn("OK", V.import_video(self.job["id"]))
        V.prepare_video("another scene")
        job2 = V._load_jobs()[1]
        out = V.import_video(job2["id"])
        self.assertIn("no new .mp4", out)


class TestRoute(VideoEnv):
    def test_serves_known_and_404s_rest(self):
        client = get_app().test_client()
        clip = self.lib / "abc123.mp4"
        clip.parent.mkdir(parents=True, exist_ok=True)
        clip.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"y" * 1024)
        r = client.get("/api/video/abc123")
        self.assertEqual(r.status_code, 200)
        self.assertIn("video/mp4", r.headers.get("Content-Type", ""))
        self.assertEqual(client.get("/api/video/missing9").status_code, 404)
        r3 = client.get("/api/video/..%2F..%2Fconfig")
        self.assertEqual(r3.status_code, 404)


class TestSettingsAndPwa(unittest.TestCase):
    def tearDown(self):
        set_provider(None)

    def test_settings_post_roundtrip(self):
        client = get_app().test_client()
        r = client.post("/api/settings/agent",
                        json={"video_provider": "flow"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(KeyManager().video_provider(), "flow")
        r2 = client.post("/api/settings/agent",
                         json={"video_provider": "bogus"})
        self.assertEqual(r2.status_code, 400)
        self.assertEqual(KeyManager().video_provider(), "flow")

    def test_tools_registered(self):
        names = {fn.__name__ for fn in ALL_TOOLS}
        self.assertTrue(
            {"prepare_video", "video_status", "import_video"} <= names)

    def test_pwa_wiring(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn('id="set-video-provider"', html)
        self.assertIn("cfg.video_provider", html)
        self.assertIn("video_provider:", html)
        self.assertIn("[video:", html)
        self.assertIn("msg-video", html)
        self.assertIn("/api/video/", html)


if __name__ == "__main__":
    unittest.main()
