"""Built-in Code backend: module surface, runner loop, event dialect.

The runner is exercised in-process with a fake AgentBackend — the engine's
cloud ladder is not under test here, only that the runner drives it,
executes the tools it asks for inside the project jail, speaks the
opencode event dialect core.code_stream parses, and persists history.
"""
from __future__ import annotations

import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from _harness import bootstrap, current_home

bootstrap()

from core import builtin as bi  # noqa: E402
from core import builtin_runner as runner  # noqa: E402
from core import code_stream  # noqa: E402


class TestModuleSurface(unittest.TestCase):
    def test_always_installed(self):
        self.assertTrue(bi.is_installed())
        self.assertEqual(bi.version(), "built-in engine")

    def test_never_stale(self):
        self.assertFalse(bi.session_is_stale("anything at all"))

    def test_build_command_shape(self):
        cmd = bi.build_command("do it", "/some/project")
        self.assertEqual(cmd[:3], [sys.executable, "-m",
                                   "core.builtin_runner"])
        self.assertIn("--workdir", cmd)
        self.assertIn("/some/project", cmd)
        self.assertIn("--prompt", cmd)
        self.assertIn("do it", cmd)
        self.assertNotIn("--session-id", cmd)

    def test_build_command_carries_the_session(self):
        cmd = bi.build_command("do it", "/p", session_id="abc12345")
        self.assertIn("--session-id", cmd)
        self.assertIn("abc12345", cmd)

    def test_env_puts_the_app_root_on_pythonpath(self):
        env = bi.env()
        root = str(Path(bi.__file__).resolve().parent.parent)
        self.assertTrue(env["PYTHONPATH"].startswith(root))
        self.assertNotIn("LD_PRELOAD", env)

    def test_doctor_shape(self):
        doc = bi.doctor()
        self.assertTrue(doc["installed"])
        self.assertEqual(doc["version"], "built-in engine")
        self.assertIsNone(doc["hint"])
        self.assertIn("sessions", doc)


class TestSessionMap(unittest.TestCase):
    def test_round_trip(self):
        workdir = current_home() / "bi-proj"
        workdir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: bi.forget_session(workdir))
        self.assertIsNone(bi.get_session(workdir))
        bi.remember_session(workdir, "sess-1234")
        self.assertEqual(bi.get_session(workdir), "sess-1234")
        self.assertEqual(bi.forget_session(workdir), "sess-1234")
        self.assertIsNone(bi.get_session(workdir))

    def test_empty_session_is_not_stored(self):
        workdir = current_home() / "bi-proj2"
        workdir.mkdir(parents=True, exist_ok=True)
        bi.remember_session(workdir, "")
        self.assertIsNone(bi.get_session(workdir))


class _Step:
    def __init__(self, text="", tool_calls=None, usage=None, reasoning=""):
        self.text = text
        self.tool_calls = tool_calls or []
        self.usage = usage or {}
        self.reasoning = reasoning
        self.model = "fake-model"
        self.provider = "fake"
        self.key_id = ""


class _FakeBackend:
    """First step asks for one write_file; second answers with text."""

    instances = []

    def __init__(self):
        self.calls = []
        _FakeBackend.instances.append(self)

    def step(self, *, system, messages, tools, **kw):
        self.calls.append([dict(m) for m in messages])
        if len(self.calls) == 1:
            return _Step(tool_calls=[{
                "function": {
                    "name": "write_file",
                    "arguments": json.dumps({
                        "path": "hello.py",
                        "content": "print('hi')\n",
                    }),
                },
            }], usage={"input": 10, "output": 5})
        return _Step(text="Done — created hello.py.",
                     usage={"input": 20, "output": 7})


class _ExhaustedBackend:
    def step(self, *, system, messages, tools, **kw):
        from core.agent_backend import BackendExhausted
        raise BackendExhausted("no route")


class _SilentBackend:
    """First step writes a file with no text; second step ends the run
    with neither text nor tool calls — the model did the work and said
    nothing, which the UI used to render as "(no output)"."""

    def __init__(self):
        self.calls = 0

    def step(self, *, system, messages, tools, **kw):
        self.calls += 1
        if self.calls == 1:
            return _Step(tool_calls=[{
                "function": {
                    "name": "write_file",
                    "arguments": json.dumps({
                        "path": "quiet.txt",
                        "content": "shh\n",
                    }),
                },
            }], usage={"input": 8, "output": 3})
        return _Step(usage={"input": 9, "output": 0})


class TestRunner(unittest.TestCase):
    def setUp(self):
        from core import projects as ps
        self.ps = ps
        self._prev_active = ps.load().get("active")
        self.workdir = current_home() / "runner-proj"
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.proj = ps.add("runner-proj", str(self.workdir))
        ps.set_active(self.proj["id"])
        self.addCleanup(self._restore)
        _FakeBackend.instances = []

    def _restore(self):
        data = self.ps.load()
        if self._prev_active and any(p.get("id") == self._prev_active
                                     for p in data["projects"]):
            self.ps.set_active(self._prev_active)
        self.ps.remove(self.proj["id"])

    def _run(self, backend_cls, prompt="create hello.py", sid=None):
        buf = io.StringIO()
        with patch("core.agent_backend.AgentBackend", backend_cls):
            with redirect_stdout(buf):
                rc = runner.run(prompt, str(self.workdir), sid)
        events = [code_stream.parse_event(line)
                  for line in buf.getvalue().splitlines()]
        return rc, [e for e in events if e is not None]

    def test_a_run_writes_the_file_and_speaks_the_dialect(self):
        rc, events = self._run(_FakeBackend)
        self.assertEqual(rc, 0)
        written = self.workdir / "hello.py"
        self.assertTrue(written.exists())
        self.assertEqual(written.read_text(encoding="utf-8"),
                         "print('hi')\n")

        summary = code_stream.summarize(events)
        self.assertIn("Done", summary["reply"])
        self.assertIsNone(summary["error"])
        self.assertTrue(summary["session_id"])
        self.assertEqual(len(summary["files"]), 1)
        self.assertEqual(summary["files"][0]["op"], "create")
        self.assertTrue(summary["files"][0]["path"].endswith("hello.py"))
        kinds = [e["type"] for e in events]
        self.assertIn("step_start", kinds)
        self.assertIn("tool_use", kinds)
        self.assertIn("text", kinds)

    def test_history_persists_and_the_next_run_resumes(self):
        rc, events = self._run(_FakeBackend)
        self.assertEqual(rc, 0)
        sid = code_stream.summarize(events)["session_id"]
        rc2, _events2 = self._run(_FakeBackend, prompt="and again", sid=sid)
        self.assertEqual(rc2, 0)
        second = _FakeBackend.instances[-1]
        seen = second.calls[0]
        texts = [m.get("text", "") for m in seen]
        self.assertIn("create hello.py", texts)
        self.assertIn("and again", texts)

    def test_pre_loop_crash_speaks_as_error_event(self):
        # A child that dies before its first event is reported by the
        # caller as "exited 1 with no output" — the runner must turn ANY
        # early failure (import, engine construction, tool wrapping)
        # into one error event carrying the real exception instead.
        def boom():
            raise RuntimeError("kaboom-import")

        with patch.object(runner, "_code_tools", boom):
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = runner.run("hi", str(self.workdir), None)
        self.assertEqual(rc, 1)
        events = [json.loads(x) for x in buf.getvalue().splitlines() if x]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "error")
        self.assertIn("kaboom-import",
                      events[0]["error"]["data"]["message"])

    def test_silent_model_still_gets_a_closing_line(self):
        rc, events = self._run(_SilentBackend, prompt="write quiet.txt")
        self.assertEqual(rc, 0)
        self.assertTrue((self.workdir / "quiet.txt").exists())
        summary = code_stream.summarize(events)
        self.assertIsNone(summary["error"])
        self.assertNotEqual(summary["reply"], "(no output)")
        self.assertIn("quiet.txt", summary["reply"])

    def test_no_cloud_route_is_one_friendly_error(self):
        rc, events = self._run(_ExhaustedBackend)
        self.assertEqual(rc, 1)
        summary = code_stream.summarize(events)
        self.assertIsNotNone(summary["error"])
        self.assertIn("Settings", summary["error"])


if __name__ == "__main__":
    unittest.main()
