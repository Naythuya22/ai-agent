"""Release 33 — Flow handoff fits how the owner actually uses Flow.

Found on-device 2026-10-27 during the first real prepare_video run:
the owner is signed in to Flow inside the installed Flow app, not
the browser — the browser tab the agent opened asked for sign-in and
confused him. And the calling model passed duration_s=8 +
resolution="720p" (12 credits) because the docstring listed options
with no guidance; the pilot-proven cheap dialogue default is Omni
360p 6s = 5 credits.

A. Defaults: preparing with no duration/resolution records 6s/360p
   and the 5-credit estimate.
B. Handoff: the returned text presents BOTH routes — the browser
   tab and the Flow app — whether or not the browser intent fired.
C. Credit guard: a combo measured above 5 credits carries an
   explicit cost note naming the cheaper default; the default job
   carries no such note.
D. R32 refusals unbroken: Veo Lite + dialogue refused; provider
   off refuses and records nothing.

Run:  sh tests/run.sh -p "test_release33.py"
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from _harness import bootstrap

bootstrap()

from core.key_manager import KeyManager  # noqa: E402
from tools import video as V  # noqa: E402

DIALOGUE = "ရရင် အရင် စုပါနော်"


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
        self._tmp = tempfile.TemporaryDirectory(prefix="r33-video-")
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
        set_provider("flow")
        self.addCleanup(lambda: set_provider(None))

    def _restore(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._tmp.cleanup()


class TestDefaults(VideoEnv):
    def test_no_args_records_cheap_default(self):
        out = V.prepare_video(
            "a young farmer drops a coin into a clay pot",
            dialogue=DIALOGUE,
        )
        self.assertIn("OK: video job", out)
        jobs = V._load_jobs()
        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        self.assertEqual(job["duration_s"], 6)
        self.assertEqual(job["resolution"], "360p")
        self.assertEqual(job["est_credits"], 5)


class TestHandoffRoutes(VideoEnv):
    def test_browser_intent_failed_still_mentions_both_routes(self):
        # On this host _open_flow() cannot fire an Android intent,
        # so prepare takes the fallback branch — it must still
        # present the browser AND the Flow app route.
        out = V.prepare_video("a market scene", dialogue=DIALOGUE)
        self.assertIn("browser", out)
        self.assertIn("Flow app", out)

    def test_browser_intent_ok_mentions_both_routes(self):
        orig = V._open_flow
        V._open_flow = lambda: True
        self.addCleanup(setattr, V, "_open_flow", orig)
        out = V.prepare_video("a market scene", dialogue=DIALOGUE)
        self.assertIn("browser tab", out)
        self.assertIn("Flow app", out)
        self.assertIn("sign in", out)


class TestCreditGuard(VideoEnv):
    def test_explicit_720p_8s_carries_cost_note(self):
        out = V.prepare_video(
            "a young farmer drops a coin into a clay pot",
            dialogue=DIALOGUE, duration_s=8, resolution="720p",
        )
        job = V._load_jobs()[0]
        self.assertEqual(job["est_credits"], 12)
        self.assertIn("Cost note", out)
        self.assertIn("12 credits", out)
        self.assertIn("360p", out)
        self.assertIn("5 credits", out)

    def test_default_job_has_no_cost_note(self):
        out = V.prepare_video("a market scene", dialogue=DIALOGUE)
        self.assertNotIn("Cost note", out)

    def test_unmeasured_combo_gets_no_invented_note(self):
        # omni 360p 4s was never measured: estimate stays unknown,
        # so no cost note may appear either way.
        out = V.prepare_video("a market scene", dialogue=DIALOGUE,
                              duration_s=4)
        job = V._load_jobs()[0]
        self.assertIsNone(job["est_credits"])
        self.assertIn("credits shown in Flow", out)
        self.assertNotIn("Cost note", out)


class TestRefusalsUnbroken(VideoEnv):
    def test_veo_lite_dialogue_still_refused(self):
        out = V.prepare_video("a tea shop", dialogue="မင်္ဂလာပါ",
                              model="veo_lite")
        self.assertIn("cannot generate Burmese speech", out)
        self.assertEqual(V._load_jobs(), [])

    def test_provider_off_still_refused(self):
        set_provider(None)
        out = V.prepare_video("a cartoon farmer saving coins")
        self.assertIn("OFF", out)
        self.assertFalse(self.jobs.exists())


if __name__ == "__main__":
    unittest.main()
