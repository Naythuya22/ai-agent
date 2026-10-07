"""Code-mode tests: opencode wiring, event parsing, and the endpoints.

No opencode binary and no model are needed — the command builder, the managed
config and the event parser are all pure functions, and the endpoints are
exercised with opencode forced "not installed" (the state a fresh phone is in)
plus a stubbed run for the happy path.
"""
from __future__ import annotations

import io
import json
import os
import shlex
import subprocess
import tempfile
import threading
import time
import unittest
from contextlib import closing, contextmanager
from pathlib import Path
from unittest.mock import patch

from _harness import bootstrap, current_home, get_app

bootstrap()

from core import code_stream as cs  # noqa: E402
from core import opencode as oc  # noqa: E402
from core import projects as ps  # noqa: E402
from routes import _state as rstate  # noqa: E402
from routes import code as rc  # noqa: E402

app = get_app()
client = app.test_client()

# What a phone has to run to get opencode (see core/opencode.INSTALL_HINT).
INSTALL_CMD = "npm install -g --allow-scripts=opencode-ai opencode-ai"


def _oc_argv(cmd):
    """The `opencode ...` argv hidden inside the proot + `bash -lc` wrapper.

    build_command() returns PROOT_CMD + ["bash", "-lc", script], and the whole
    run — exports, cd, flags, prompt — lives in that script. Tests asserting on
    flags have to pull the argv back out first.
    """
    tokens = shlex.split(cmd[-1])
    return tokens[tokens.index("opencode"):]


# ── a real `opencode run --format json` transcript ─────────────────────────
# Captured from opencode 1.18.33; the shapes are what the code must handle.

EVENT_WRITE = {
    "type": "tool_use", "timestamp": 1790770481785, "sessionID": "ses_abc",
    "part": {"type": "tool", "tool": "write", "callID": "call-1",
             "state": {"status": "completed",
                       "input": {"filePath": "/w/hello.txt", "content": "hi"},
                       "output": "Wrote file successfully.",
                       "metadata": {"filepath": "/w/hello.txt", "exists": False},
                       "title": "w/hello.txt",
                       "time": {"start": 1, "end": 2}},
             "id": "prt_1", "sessionID": "ses_abc", "messageID": "msg_1"},
}

EVENT_EDIT_EXISTING = {
    "type": "tool_use", "sessionID": "ses_abc",
    "part": {"type": "tool", "tool": "edit", "callID": "call-2",
             "state": {"status": "completed",
                       "input": {"filePath": "/w/hello.txt"},
                       "metadata": {"exists": True},
                       "title": "w/hello.txt"},
             "id": "prt_2"},
}

EVENT_BASH = {
    "type": "tool_use", "sessionID": "ses_abc",
    "part": {"type": "tool", "tool": "bash", "callID": "call-3",
             "state": {"status": "completed",
                       "input": {"command": "ls", "workdir": "/w"},
                       "output": "hello.txt",
                       "title": "ls"},
             "id": "prt_3"},
}

EVENT_TEXT = {
    "type": "text", "sessionID": "ses_abc",
    "part": {"type": "text", "text": "Created hello.txt with `hi`.",
             "time": {"start": 1, "end": 2}, "id": "prt_4"},
}

EVENT_FINISH = {
    "type": "step_finish", "sessionID": "ses_abc",
    "part": {"type": "step-finish", "reason": "stop",
             "tokens": {"total": 100, "input": 90, "output": 10,
                        "reasoning": 0, "cache": {"write": 0, "read": 0}},
             "cost": 0, "id": "prt_5"},
}

FULL_TRANSCRIPT = [EVENT_WRITE, EVENT_BASH, EVENT_FINISH, EVENT_TEXT, EVENT_FINISH]


class TestParseEvent(unittest.TestCase):
    def test_parses_a_real_event(self):
        self.assertEqual(cs.parse_event(json.dumps(EVENT_TEXT))["type"], "text")

    def test_blank_and_warning_lines_are_ignored(self):
        for junk in ("", "   ", "\n", "npm warn deprecated", "not json {"):
            self.assertIsNone(cs.parse_event(junk), junk)

    def test_strips_ansi_before_parsing(self):
        self.assertIsNotNone(cs.parse_event("\x1b[32m" + json.dumps(EVENT_TEXT)))

    def test_json_scalar_is_not_an_event(self):
        for scalar in ("42", '"a string"', "[1,2]", "null"):
            self.assertIsNone(cs.parse_event(scalar), scalar)


class TestSummarize(unittest.TestCase):
    def test_joins_text_into_the_reply(self):
        s = cs.summarize([EVENT_TEXT, dict(EVENT_TEXT, part={**EVENT_TEXT["part"],
                                                            "text": " and more"})])
        self.assertEqual(s["reply"], "Created hello.txt with `hi`. and more")

    def test_reports_created_files(self):
        s = cs.summarize([EVENT_WRITE])
        self.assertEqual(s["files"], [{"tool": "write", "status": "completed",
                                       "title": "w/hello.txt", "op": "create",
                                       "path": "/w/hello.txt"}])

    def test_write_over_an_existing_file_counts_as_an_edit(self):
        s = cs.summarize([EVENT_EDIT_EXISTING])
        self.assertEqual(s["files"][0]["op"], "edit")

    def test_a_file_touched_twice_is_listed_once(self):
        """Listed once, keeping 'create' — it did not exist before this run."""
        s = cs.summarize([EVENT_WRITE, EVENT_EDIT_EXISTING])
        self.assertEqual(len(s["files"]), 1)
        self.assertEqual(s["files"][0]["op"], "create")

    def test_non_edit_tools_stay_out_of_files(self):
        """A bash call must not be reported as a file change."""
        s = cs.summarize([EVENT_BASH])
        self.assertEqual(s["files"], [])
        self.assertEqual([t["tool"] for t in s["tools"]], ["bash"])

    def test_carries_session_tokens_and_cost(self):
        s = cs.summarize(FULL_TRANSCRIPT)
        self.assertEqual(s["session_id"], "ses_abc")
        self.assertEqual(s["tokens"]["total"], 100)
        self.assertEqual(s["cost"], 0)

    def test_empty_stream_is_not_an_error(self):
        s = cs.summarize([])
        self.assertEqual(s["reply"], "")
        self.assertEqual(s["files"], [])

    def test_commits_key_stays_for_the_ui(self):
        """The UI renders j.commits; it must exist even though opencode
        does not commit on its own."""
        self.assertEqual(cs.summarize(FULL_TRANSCRIPT)["commits"], [])


class TestBuildCommand(unittest.TestCase):
    """The run happens inside Ubuntu: PROOT_CMD + `bash -lc <script>`."""

    def test_runs_inside_ubuntu_proot(self):
        cmd = oc.build_command("x", "/w")
        self.assertEqual(cmd[:6], oc.PROOT_CMD + ["bash", "-lc"])

    def test_uses_json_format_and_auto(self):
        argv = _oc_argv(oc.build_command("do it", "/w"))
        self.assertEqual(argv[:4], ["opencode", "run", "--format", "json"])
        self.assertIn("--auto", argv)

    def test_auto_can_be_turned_off(self):
        self.assertNotIn("--auto", _oc_argv(oc.build_command("x", "/w", auto=False)))

    def test_default_model_is_the_native_provider(self):
        argv = _oc_argv(oc.build_command("x", "/w"))
        self.assertEqual(argv[argv.index("--model") + 1], oc.MODEL_REF)

    def test_session_is_resumed_by_id(self):
        argv = _oc_argv(oc.build_command("x", "/w", session_id="ses_1"))
        self.assertIn("--session", argv)
        self.assertEqual(argv[argv.index("--session") + 1], "ses_1")
        self.assertNotIn("--continue", argv)

    def test_first_run_starts_a_fresh_session_not_the_last_one(self):
        """No --continue: a lost session map must not adopt an unrelated session
        that happens to live in the same directory."""
        argv = _oc_argv(oc.build_command("x", "/w"))
        self.assertNotIn("--continue", argv)
        self.assertNotIn("--session", argv)

    def test_prompt_is_last_and_dash_safe(self):
        prompt = "--not-a-flag please"
        argv = _oc_argv(oc.build_command(prompt, "/w"))
        self.assertEqual(argv[-2:], ["--", prompt])

    def test_files_are_attached(self):
        argv = _oc_argv(oc.build_command("x", "/w", files=["/w/a.py", "/w/b.py"]))
        self.assertEqual(argv.count("--file"), 2)


class TestBridgeConfig(unittest.TestCase):
    """No managed bridge config: opencode resolves the provider itself.

    The old design wrote a `bridge` provider plus `model` into a managed
    opencode.json. With the native provider there is nothing to declare, so
    the only thing this module owns is the path OPENCODE_CONFIG points at.
    """

    def test_no_bridge_provider_is_declared(self):
        self.assertEqual(oc.bridge_config(), {})

    def test_config_path_is_the_managed_file(self):
        path = oc.write_bridge_config()
        self.assertEqual(path, oc.config_file())
        self.assertEqual(str(path), oc.env()["OPENCODE_CONFIG"])

    def test_rewrite_is_idempotent(self):
        first = oc.write_bridge_config()
        mtime = first.stat().st_mtime_ns if first.exists() else None
        oc.write_bridge_config()
        self.assertEqual(first.stat().st_mtime_ns if first.exists() else None,
                         mtime, "unchanged config should not be rewritten")

    def test_env_hands_the_config_to_opencode(self):
        env = oc.env()
        self.assertEqual(env["OPENCODE_CONFIG"], str(oc.config_file()))
        self.assertEqual(env["TERM"], "dumb")
        self.assertEqual(env["NO_COLOR"], "1")


class TestEnvChildEnvironment(unittest.TestCase):
    """What env() gives a child process, and what it refuses to give.

    LD_PRELOAD must never travel: the opencode-termux wrapper exports a musl
    loader, Termux exports its own bionic one, and handing either loader to a
    binary built for the other libc segfaults it. LD_LIBRARY_PATH must travel
    — opencode's own binary needs it.
    """

    KEYS = ("LD_PRELOAD", "LD_LIBRARY_PATH")

    def setUp(self):
        self._saved = {key: os.environ.get(key) for key in self.KEYS}
        for key in self.KEYS:
            os.environ.pop(key, None)

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_env_strips_ld_preload(self):
        os.environ["LD_PRELOAD"] = "/sentinel/libfake.so"
        self.assertNotIn("LD_PRELOAD", oc.env())

    def test_env_restores_after_call(self):
        os.environ["LD_PRELOAD"] = "/sentinel/libfake.so"
        oc.env()
        self.assertEqual(os.environ["LD_PRELOAD"], "/sentinel/libfake.so")

    def test_env_keeps_ld_library_path(self):
        os.environ["LD_LIBRARY_PATH"] = "/sentinel/libdir"
        self.assertEqual(oc.env().get("LD_LIBRARY_PATH"), "/sentinel/libdir")


class TestSessionStore(unittest.TestCase):
    def setUp(self):
        oc._write_sessions({})
        self.addCleanup(oc._write_sessions, {})

    def test_round_trip(self):
        oc.remember_session("/w/a", "ses_1")
        self.assertEqual(oc.get_session("/w/a"), "ses_1")

    def test_workspaces_are_separate(self):
        oc.remember_session("/w/a", "ses_1")
        self.assertIsNone(oc.get_session("/w/b"))

    def test_forget_returns_what_it_removed(self):
        oc.remember_session("/w/a", "ses_1")
        self.assertEqual(oc.forget_session("/w/a"), "ses_1")
        self.assertIsNone(oc.get_session("/w/a"))
        self.assertIsNone(oc.forget_session("/w/a"))

    def test_corrupt_file_does_not_crash(self):
        oc.session_file().parent.mkdir(parents=True, exist_ok=True)
        oc.session_file().write_text("{not json")
        self.assertIsNone(oc.get_session("/w/a"))

    def test_empty_session_id_is_not_stored(self):
        oc.remember_session("/w/a", "")
        self.assertIsNone(oc.get_session("/w/a"))


class TestCodeEndpoints(unittest.TestCase):
    """The endpoints, with opencode forced present or absent."""

    def setUp(self):
        # Present unless a test forces absence. The real is_installed() shells
        # out to `proot-distro ... which opencode`, so leaving it live here
        # would make the endpoints depend on this machine having the distro.
        self._installed = oc.is_installed
        oc.is_installed = lambda: True
        self.addCleanup(lambda: setattr(oc, "is_installed", self._installed))

    def test_doctor_always_answers(self):
        with closing(client.get("/api/code/doctor")) as r:
            self.assertEqual(r.status_code, 200)
            body = r.get_json()
        for key in ("installed", "version", "config_file", "model", "bridge_base"):
            self.assertIn(key, body)

    def test_doctor_explains_a_missing_binary(self):
        oc.is_installed = lambda: False
        with closing(client.get("/api/code/doctor")) as r:
            body = r.get_json()
        self.assertFalse(body["ok"])
        self.assertIn(INSTALL_CMD, body["hint"])

    def test_run_without_opencode_says_how_to_install(self):
        oc.is_installed = lambda: False
        with closing(client.post("/api/code/web/run", json={"message": "hi"})) as r:
            self.assertEqual(r.status_code, 500)
            self.assertIn(INSTALL_CMD, r.get_json()["error"])

    def test_stream_without_opencode_says_how_to_install(self):
        oc.is_installed = lambda: False
        with closing(client.post("/api/code/web/run/stream", json={"message": "hi"})) as r:
            self.assertEqual(r.status_code, 500)
            self.assertIn(INSTALL_CMD, r.get_json()["error"])

    def test_api_code_without_opencode_says_how_to_install(self):
        oc.is_installed = lambda: False
        with closing(client.post("/api/code", json={"task": "hi"})) as r:
            self.assertEqual(r.status_code, 500)
            self.assertIn(INSTALL_CMD, r.get_json()["error"])

    def test_web_start_without_opencode_says_how_to_install(self):
        oc.is_installed = lambda: False
        with closing(client.post("/api/code/web/start")) as r:
            self.assertEqual(r.status_code, 500)
            self.assertIn(INSTALL_CMD, r.get_json()["error"])

    def test_web_start_runs_opencode_directly_with_the_bridge_env(self):
        """No ~/bin/opencode-web wrapper: ttyd must launch opencode itself, in
        the active project, with the managed config in its environment."""
        import subprocess as sp
        from _harness import current_home
        from core import projects as ps

        work = current_home() / "ttydproj"
        work.mkdir(parents=True, exist_ok=True)
        ps.add("ttyd", str(work))
        ps.set_active(ps.list_all()[-1]["id"])

        seen = {}

        class P:
            pid = 4242
            returncode = None

            def poll(self):
                return None

        def fake_popen(cmd, **kw):
            seen["cmd"] = cmd
            seen["env"] = kw.get("env") or {}
            seen["cwd"] = kw.get("cwd")
            return P()

        saved = sp.Popen
        sp.Popen = fake_popen
        self.addCleanup(lambda: setattr(sp, "Popen", saved))
        self.addCleanup(lambda: setattr(rc, "_ttyd_proc", None))

        with closing(client.post("/api/code/web/start")) as r:
            body = r.get_json()
        self.assertTrue(body["ok"], body)
        self.assertEqual(seen["cmd"][:6],
                         ["ttyd", "-W", "-i", "127.0.0.1", "-p", "8081"])
        self.assertIn(str(oc.BINARY), seen["cmd"])
        self.assertEqual(seen["cmd"][-1], str(work))
        self.assertEqual(seen["cwd"], str(work))
        self.assertEqual(seen["env"].get("OPENCODE_CONFIG"),
                         str(oc.config_file()))

    def test_project_commands_work_without_opencode(self):
        """/project is handled by us, so it must not need the binary."""
        from _harness import current_home

        oc.is_installed = lambda: False
        made = current_home() / "proj"
        made.mkdir(parents=True, exist_ok=True)

        for msg, needle in (("/project list", "No projects yet"),
                            (f"/project add {made}", "Added project"),
                            ("/project use", "Usage"),
                            ("/project", "Project commands")):
            with closing(client.post("/api/code/web/run", json={"message": msg})) as r:
                self.assertEqual(r.status_code, 200, msg)
                self.assertIn(needle, r.get_json()["reply"], msg)

    def test_missing_message_is_checked_before_the_binary(self):
        with closing(client.post("/api/code/web/run", json={"message": "  "})) as r:
            self.assertEqual(r.status_code, 400)

    def test_history_reports_the_backing_session(self):
        with closing(client.get("/api/code/web/history")) as r:
            body = r.get_json()
        self.assertIn("count", body)
        self.assertIn("session_id", body)

    def test_stream_commits_the_files_it_reported(self):
        """A streamed run must leave git in the same state a one-shot run does."""
        import io
        import subprocess as sp
        from _harness import current_home
        from core import projects as ps

        repo = current_home() / "repo"
        repo.mkdir(parents=True, exist_ok=True)
        for args in (["init", "-q"], ["config", "user.email", "t@t"],
                     ["config", "user.name", "T"], ["add", "-A"]):
            sp.run(["git", "-C", str(repo)] + args, capture_output=True)
        (repo / "keep.txt").write_text("x\n")
        sp.run(["git", "-C", str(repo), "commit", "-qm", "base"], capture_output=True)
        ps.add("repo", str(repo))
        ps.set_active(ps.list_all()[-1]["id"])

        write = {
            "type": "tool_use", "sessionID": "ses_stream",
            "part": {"tool": "write", "state": {"status": "completed",
                    "title": "wrote NEW.md",
                    "input": {"filePath": "NEW.md", "content": "hi"}}},
        }
        text = {"type": "text", "sessionID": "ses_stream",
                "part": {"text": "made NEW.md"}}

        class P:
            returncode = 0
            pid = 99

            def __init__(self, cwd=None, **kw):
                # Stand in for opencode: actually write the file on disk, the
                # way the real child would, so the route's commit path has
                # something real to commit.
                import pathlib
                pathlib.Path(cwd, "NEW.md").write_text("hi\n")
                self.stdout = io.StringIO(json.dumps(text) + "\n"
                                          + json.dumps(write) + "\n")

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                return self.returncode

            def terminate(self):
                self.returncode = 0

            def kill(self):
                self.returncode = -9

        saved = cs.subprocess.Popen

        def fake_popen(cmd, **kw):
            # Only stand in for opencode. cs.subprocess is the global subprocess
            # module, so an unconditional fake would also hijack the git calls
            # the route makes when it commits.
            if not any("opencode" in str(a) for a in cmd):
                return saved(cmd, **kw)
            return P(**kw)

        cs.subprocess.Popen = fake_popen
        self.addCleanup(lambda: setattr(cs.subprocess, "Popen", saved))

        with closing(client.post("/api/code/web/run/stream",
                                 json={"message": "add NEW.md"})) as r:
            frames = [json.loads(l[6:]) for l in r.get_data(as_text=True).splitlines()
                      if l.startswith("data: ")]
        end = next(f for f in frames if f["type"] == "end")

        # The route committed on its own; nothing here calls commit_changes.
        self.assertEqual((repo / "NEW.md").read_text(), "hi\n")
        log = sp.run(["git", "-C", str(repo), "log", "--oneline"],
                     capture_output=True, text=True).stdout
        self.assertTrue(log.strip(), "no commit was made by the streaming route")
        tracked = sp.run(["git", "-C", str(repo), "show", "--name-only",
                          "--format=", "HEAD"], capture_output=True, text=True).stdout
        self.assertIn("NEW.md", tracked)
        self.assertEqual([f["path"] for f in end["files"]], ["NEW.md"])


class TestStreamOpencode(unittest.TestCase):
    """Drive the streaming generator against a fake opencode process."""

    def setUp(self):
        self.words = ["opencode", "run", "--help"]
        self._popen = cs.subprocess.Popen

        def fake_popen(cmd, **kwargs):
            payload = "\n".join(json.dumps(e) for e in FULL_TRANSCRIPT) + "\n"

            class FakeProc:
                returncode = 0
                pid = 1234

                def __init__(self):
                    import io
                    self.stdout = io.StringIO(payload)

                def poll(self):
                    return self.returncode

                def wait(self, timeout=None):
                    return self.returncode

                def terminate(self):
                    self.returncode = 0

                def kill(self):
                    self.returncode = -9

            proc = FakeProc()
            self.last_cmd = cmd
            return proc

        cs.subprocess.Popen = fake_popen
        self.addCleanup(lambda: setattr(cs.subprocess, "Popen", self._popen))

    def test_emits_sse_for_text_files_and_end(self):
        frames = list(cs.stream_opencode(["opencode"], "/w", {}, threading.Lock()))
        kinds = [json.loads(f[6:])["type"] for f in frames]
        self.assertIn("chunk", kinds)
        self.assertIn("file", kinds)
        self.assertEqual(kinds[-1], "end")

    def test_end_frame_carries_the_summary(self):
        frames = list(cs.stream_opencode(["opencode"], "/w", {}, threading.Lock()))
        end = json.loads(frames[-1][6:])
        self.assertEqual(end["reply"], "Created hello.txt with `hi`.")
        self.assertEqual(end["files"][0]["op"], "create")
        self.assertEqual(end["session_id"], "ses_abc")
        self.assertEqual(end["rc"], 0)

    def test_session_is_reported_as_soon_as_it_is_known(self):
        seen = []
        list(cs.stream_opencode(["opencode"], "/w", {}, threading.Lock(),
                                on_session=seen.append))
        self.assertEqual(seen, ["ses_abc"] * len(FULL_TRANSCRIPT))

    def test_bash_tool_is_announced_but_not_as_a_file(self):
        frames = list(cs.stream_opencode(["opencode"], "/w", {}, threading.Lock()))
        events = [json.loads(f[6:]) for f in frames]
        tools = [e for e in events if e["type"] == "tool"]
        self.assertEqual([t["tool"] for t in tools], ["bash"])
        files = [e for e in events if e["type"] == "file"]
        self.assertEqual([f["path"] for f in files], ["/w/hello.txt"])

    def test_non_json_lines_are_surfaced_as_logs(self):
        def fake_popen(cmd, **kwargs):
            import io
            self.last_cmd = cmd

            class FakeProc:
                returncode = 0
                pid = 1
                stdout = io.StringIO("npm WARN deprecated\n"
                                     + json.dumps(EVENT_TEXT) + "\n")

                def poll(self): return 0
                def wait(self, timeout=None): return 0
                def terminate(self): pass
                def kill(self): pass

            return FakeProc()

        cs.subprocess.Popen = fake_popen
        frames = list(cs.stream_opencode(["opencode"], "/w", {}, threading.Lock()))
        events = [json.loads(f[6:]) for f in frames]
        self.assertIn("npm WARN deprecated", [e.get("text") for e in events])
        self.assertEqual(events[-1]["type"], "end")


EVENT_STEP_START = {
    "type": "step_start", "sessionID": "ses_abc",
    "part": {"id": "prt_step_1", "messageID": "msg_1"},
}


class TestLiveFrames(unittest.TestCase):
    """What the stream announces *while* opencode is still working.

    `run --format json` emits nothing but a step start until the step ends,
    so without the step frame and the unbuffered child the UI would sit blank
    for the whole generation and then jump.
    """

    def _run(self, lines, env=None):
        import io
        seen = {}

        def fake_popen(cmd, **kwargs):
            seen["env"] = kwargs.get("env") or {}

            class FakeProc:
                returncode = 0
                pid = 1
                stdout = io.StringIO("".join(
                    l if l.endswith("\n") else l + "\n" for l in lines))

                def poll(self): return 0
                def wait(self, timeout=None): return 0
                def terminate(self): pass
                def kill(self): pass

            return FakeProc()

        saved = cs.subprocess.Popen
        cs.subprocess.Popen = fake_popen
        try:
            frames = list(cs.stream_opencode(
                ["opencode"], "/w", env if env is not None else {},
                threading.Lock()))
        finally:
            cs.subprocess.Popen = saved
        events = [json.loads(f[6:]) for f in frames]
        return events, seen

    def test_step_start_becomes_a_step_frame(self):
        events, _ = self._run([json.dumps(EVENT_STEP_START) + "\n",
                               json.dumps(EVENT_TEXT) + "\n"])
        self.assertEqual([e["type"] for e in events[:2]], ["step", "chunk"])
        self.assertEqual(events[0]["n"], 1)

    def test_steps_are_numbered_in_order(self):
        events, _ = self._run([
            json.dumps(EVENT_STEP_START) + "\n",
            json.dumps(EVENT_BASH) + "\n",
            json.dumps(EVENT_STEP_START) + "\n",
        ])
        self.assertEqual([e["n"] for e in events if e["type"] == "step"],
                         [1, 2])

    def test_the_child_is_unbuffered_even_with_an_empty_env(self):
        """A bare {} (tests, ad-hoc callers) must not fall back to block
        buffering: that is what made the stream arrive in one lump."""
        _, seen = self._run([json.dumps(EVENT_TEXT) + "\n"])
        self.assertEqual(seen["env"].get("PYTHONUNBUFFERED"), "1")

    def test_an_explicit_env_is_not_overridden(self):
        _, seen = self._run([json.dumps(EVENT_TEXT) + "\n"],
                            env={"PYTHONUNBUFFERED": "0"})
        self.assertEqual(seen["env"]["PYTHONUNBUFFERED"], "0")

    def test_a_doubled_tool_call_is_announced_once(self):
        first = json.dumps(dict(EVENT_BASH)) + "\n"
        repeat = json.dumps(dict(
            EVENT_BASH,
            part={**EVENT_BASH["part"],
                  "state": {**EVENT_BASH["part"]["state"],
                            "status": "running"}})) + "\n"
        events, _ = self._run([first, repeat])
        self.assertEqual([e["type"] for e in events if e["type"] == "tool"],
                         ["tool"])

    def test_the_run_still_ends_with_end(self):
        events, _ = self._run([json.dumps(EVENT_STEP_START) + "\n",
                               json.dumps(EVENT_TEXT) + "\n"])
        self.assertEqual(events[-1]["type"], "end")


class TestPrintLogs(unittest.TestCase):
    """--print-logs is what gives the status bar anything to show before the
    model answers; it is opt-in so one-shot runs keep their output clean."""

    def test_the_streaming_command_asks_for_logs(self):
        cmd = oc.build_command("hi", "/w", logs=True)
        self.assertIn("--print-logs", cmd[-1])
        self.assertIn("--log-level", cmd[-1])

    def test_the_one_shot_command_does_not(self):
        self.assertNotIn("--print-logs", oc.build_command("hi", "/w")[-1])

    def test_json_format_is_still_first(self):
        # cmd[-1] is the bash script the runner executes, not argv: the
        # opencode call lives inside it, behind the exports and the cd.
        script = oc.build_command("hi", "/w")[-1]
        self.assertRegex(script, r"&& opencode run --format json --auto\b")

    def test_pythonunbuffered_is_exported_by_the_script(self):
        script = oc.build_command("hi", "/w")[-1]
        self.assertIn("PYTHONUNBUFFERED=1", script)

    def test_paths_with_spaces_are_quoted(self):
        script = oc.build_command("hi", "/tmp/a b")
        self.assertIn("'/tmp/a b'", script[-1])


class TestAbortedStream(unittest.TestCase):
    """A browser that navigates away mid-run must not leave opencode alive."""

    def test_generator_exit_terminates_the_child(self):
        import io

        state = {"alive": True}

        class FakeProc:
            returncode = None
            pid = 42
            stdout = io.StringIO(json.dumps(EVENT_WRITE) + "\n")

            def poll(self):
                return None if state["alive"] else 0

            def wait(self, timeout=None):
                return 0

            def terminate(self):
                state["alive"] = False

            def kill(self):
                state["alive"] = False

        original = cs.subprocess.Popen
        cs.subprocess.Popen = lambda cmd, **kw: FakeProc()
        self.addCleanup(lambda: setattr(cs.subprocess, "Popen", original))

        gen = cs.stream_opencode(["opencode"], "/w", {}, threading.Lock())
        self.assertTrue(next(gen).startswith("data: "))  # first frame
        gen.close()  # what Flask does when the client hangs up
        self.assertFalse(state["alive"], "opencode survived an aborted stream")


class TestStreamTimeouts(unittest.TestCase):
    """A hung child must not pin the worker thread, code_lock and proot.

    The read loop used to block on readline() forever, so a stalled provider
    call or a frozen proot mount left the Flask worker alive, code_lock held
    and the opencode + proot children running. It now polls with select and
    kills a child that goes silent (idle) or outlives the run limit (hard).
    """

    def _drive(self, lines=(), select_ready=False, idle=90, hard=300):
        """Run the stream against a fake child; return (proc, events).

        select_ready=True reports the pipe readable on every call, so the
        fake stdout is drained line by line; False never reports it, which is
        exactly what a child that stopped writing looks like.
        """

        class FakeStdout:
            def __init__(self, owner, lines):
                self.owner = owner
                self.lines = list(lines)

            def readline(self):
                if self.lines:
                    return self.lines.pop(0)
                # EOF: nothing left to read, and the child has exited.
                self.owner.returncode = 0
                return ""

        class FakeProc:
            pid = 4242

            def __init__(self):
                self.returncode = None
                self.terminated = False
                self.stdout = FakeStdout(self, lines)

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                if self.returncode is None:
                    self.returncode = 0
                return self.returncode

            def terminate(self):
                self.terminated = True
                self.returncode = -15

            def kill(self):
                self.terminated = True
                self.returncode = -9

        proc = FakeProc()

        def fake_select(rlist, wlist, xlist, timeout=None):
            if select_ready:
                return (list(rlist), [], [])
            # The ceilings are wall-clock seconds: nap rather than spin so the
            # test still finishes in about a second.
            time.sleep(min(timeout or 0, 0.1))
            return ([], [], [])

        with patch.object(cs.subprocess, "Popen", lambda cmd, **kw: proc), \
             patch.object(cs.select, "select", fake_select), \
             patch.object(cs, "STREAM_IDLE_TIMEOUT", idle), \
             patch.object(cs, "STREAM_HARD_TIMEOUT", hard):
            frames = list(cs.stream_opencode(
                ["opencode"], "/w", {}, threading.Lock()))
        return proc, [json.loads(f[6:]) for f in frames]

    def test_stream_opencode_idle_timeout_kills_the_child(self):
        proc, events = self._drive(select_ready=False, idle=1, hard=5)
        self.assertIn("idle_timeout", [e.get("reason") for e in events])
        self.assertTrue(proc.terminated, "the hung child was never terminated")
        self.assertEqual(events[-1]["type"], "end")

    def test_stream_opencode_hard_timeout_fires_first_when_no_output(self):
        proc, events = self._drive(select_ready=False, idle=999, hard=1)
        self.assertIn("hard_timeout", [e.get("reason") for e in events])
        self.assertNotIn("idle_timeout", [e.get("reason") for e in events])
        self.assertTrue(proc.terminated, "the hung child was never terminated")
        self.assertEqual(events[-1]["type"], "end")

    def test_stream_opencode_still_ends_normally_on_eof(self):
        proc, events = self._drive([json.dumps(EVENT_TEXT) + "\n"],
                                   select_ready=True)
        self.assertIn("chunk", [e["type"] for e in events])
        self.assertEqual(events[-1]["type"], "end")
        self.assertFalse([e for e in events if e.get("reason")],
                         "a clean EOF must not report a timeout")
        self.assertFalse(proc.terminated, "a finished child must not be killed")


# ── a faked opencode child ─────────────────────────────────────────────────

class _FakeOpencode:
    """Stands in for `opencode run`, handing out one canned body per call.

    git is left to the real subprocess module: the routes shell out to it to
    commit, and that side of things has to keep working for the test to mean
    anything.
    """

    def __init__(self, bodies=(), writes=(), returncodes=()):
        self.bodies = list(bodies)
        self.writes = list(writes)
        self.returncodes = list(returncodes)
        self.commands = []
        # Captured now, while still the real thing. Reading them off the module
        # at call time would hand back our own patch and recurse.
        self._real_popen = subprocess.Popen
        self._real_run = subprocess.run

    def _next(self):
        text = self.bodies.pop(0) if self.bodies else ""
        code = self.returncodes.pop(0) if self.returncodes else 0
        return text, code

    def _is_ours(self, cmd):
        return isinstance(cmd, (list, tuple)) and any(
            "opencode" in str(a) for a in cmd)

    def popen(self, cmd, **kw):
        if not self._is_ours(cmd):
            return self._real_popen(cmd, **kw)
        self.commands.append(list(cmd))
        text, code = self._next()
        for name in self.writes:
            (Path(kw.get("cwd") or ".") / name).write_text("hi\n")

        class FakeProc:
            def __init__(self):
                self.stdout = io.StringIO(text)
                # Real Popen only fills this in once wait()/poll() has seen the
                # child finish, and code_stream reads it after wait().
                self.returncode = code

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                return self.returncode

            def terminate(self):
                pass

            def kill(self):
                pass

        return FakeProc()

    def run(self, cmd, **kw):
        if not self._is_ours(cmd):
            return self._real_run(cmd, **kw)
        self.commands.append(list(cmd))
        text, code = self._next()

        class Done:
            returncode = code
            stdout = text
            stderr = ""

        return Done()


@contextmanager
def fake_opencode(bodies=(), writes=(), returncodes=()):
    """Patch both subprocess entry points for the duration of the block.

    opencode is also forced "installed": the real is_installed() shells out to
    `proot-distro ... which opencode`, that probe would match the fake above,
    and it would swallow a canned body meant for the run itself.
    """
    real_popen, real_run = subprocess.Popen, subprocess.run
    real_installed = oc.is_installed
    fake = _FakeOpencode(bodies, writes, returncodes)
    subprocess.Popen = fake.popen
    subprocess.run = fake.run
    oc.is_installed = lambda: True
    try:
        yield fake
    finally:
        subprocess.Popen = real_popen
        subprocess.run = real_run
        oc.is_installed = real_installed


_repo_seq = 0


def _make_repo(name):
    """A git repo with one commit in it.

    A fresh directory per call: these tests assert on the repo's log, and a
    shared one would let an earlier test's commit satisfy a later one (or make
    the next run a no-op, so nothing gets committed at all).
    """
    global _repo_seq
    _repo_seq += 1
    repo = current_home() / f"{name}-{_repo_seq}"
    repo.mkdir(parents=True, exist_ok=True)
    for args in (["init", "-q"], ["config", "user.email", "t@t"],
                 ["config", "user.name", "T"]):
        subprocess.run(["git", "-C", str(repo)] + args, capture_output=True)
    (repo / "keep.txt").write_text("x\n")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], capture_output=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"],
                   capture_output=True)
    return repo


def _use_repo(name):
    """Make a repo the active project for the code routes."""
    repo = _make_repo(name)
    ps.add(name, str(repo))
    ps.set_active(ps.list_all()[-1]["id"])
    return repo


def _stream(message="do a thing", **fake_kw):
    """POST the streaming endpoint with a faked child.

    Returns (frames, fake) where `frames` is the parsed SSE dicts in the order
    the client would see them and `fake.commands` is the argv of every faked
    opencode run.
    """
    with fake_opencode(**fake_kw) as fake:
        with closing(client.post("/api/code/web/run/stream",
                                 json={"message": message})) as r:
            raw = r.get_data(as_text=True)
    frames = [json.loads(line[6:]) for line in raw.splitlines()
              if line.startswith("data: ")]
    return frames, fake


def _once(message="do a thing", **fake_kw):
    """POST the one-shot endpoint with a faked child; returns (body, fake)."""
    with fake_opencode(**fake_kw) as fake:
        with closing(client.post("/api/code/web/run",
                                 json={"message": message})) as r:
            return r.get_json(), fake


def _log_len(repo):
    log = subprocess.run(["git", "-C", str(repo), "log", "--oneline"],
                         capture_output=True, text=True).stdout
    return len(log.strip().splitlines())


def _lines(*events):
    return "".join(json.dumps(e) + "\n" for e in events)


class TestCommitChanges(unittest.TestCase):
    """commit_changes must always say what happened, never hand back None.

    Returning None on every failure path meant the UI could only show "no
    commit" with no reason — not a repo, nothing staged and a failed `git
    commit` all looked identical.
    """

    def test_a_folder_that_is_not_a_repo_reports_it(self):
        tmp = Path(tempfile.mkdtemp(prefix="oc-nogit-"))
        self.assertEqual(
            rc.commit_changes(tmp, [{"path": "a.txt"}], "msg"),
            {"ok": False, "reason": "not a git repository"})

    def test_no_paths_is_reported_not_swallowed(self):
        self.assertEqual(
            rc.commit_changes(Path(tempfile.gettempdir()), [], "msg"),
            {"ok": False, "reason": "no files to commit"})

    def test_a_real_commit_reports_ok_with_hash_and_message(self):
        repo = Path(tempfile.mkdtemp(prefix="oc-commit-"))
        for args in (["init", "-q"], ["config", "user.email", "t@t"],
                     ["config", "user.name", "T"]):
            subprocess.run(["git", "-C", str(repo)] + args, capture_output=True)
        (repo / "new.txt").write_text("hi\n")

        res = rc.commit_changes(repo, [{"path": "new.txt"}], "add new.txt")

        self.assertTrue(res["ok"], res)
        self.assertTrue(res["hash"])
        self.assertTrue(res["message"])


class TestCommitFrames(unittest.TestCase):
    """A streamed commit has to reach the UI the way the UI reads it.

    web/index.html draws a commit from a *top-level* `commit` event
    (`evt.type==='commit'`). Nesting it on the end payload as `end.commit` —
    which this route used to do — still committed to git while showing the user
    nothing, and left the streamed path disagreeing with the one-shot path,
    which reports `commits` at the top level.
    """

    def setUp(self):
        self.repo = _use_repo("commitframes")
        events = [
            {"type": "tool_use", "sessionID": "ses_c",
             "part": {"tool": "write",
                      "state": {"status": "completed",
                                "title": "wrote NEW.md",
                                "input": {"filePath": "NEW.md",
                                          "content": "hi"}}}},
            {"type": "text", "sessionID": "ses_c",
             "part": {"text": "made NEW.md"}},
        ]
        self.frames, _ = _stream(bodies=[_lines(*events)], writes=["NEW.md"])
        self.kinds = [f["type"] for f in self.frames]

    def _commits(self):
        return [f for f in self.frames if f["type"] == "commit"]

    def test_commit_is_its_own_top_level_frame(self):
        commits = self._commits()
        self.assertEqual(len(commits), 1, self.kinds)
        self.assertRegex(commits[0]["hash"], r"^[0-9a-f]{7,}$")
        self.assertEqual(commits[0]["message"], "do a thing")

    def test_commit_frame_precedes_end(self):
        """Ordering matters: the UI appends bubbles in arrival order."""
        self.assertLess(self.kinds.index("commit"), self.kinds.index("end"))

    def test_the_commit_is_really_in_git(self):
        log = subprocess.run(["git", "-C", str(self.repo), "log", "--oneline"],
                             capture_output=True, text=True).stdout
        self.assertTrue(log.strip(), "no commit was made")
        shown = subprocess.run(["git", "-C", str(self.repo), "show",
                                "--name-only", "--format=", "HEAD"],
                               capture_output=True, text=True).stdout
        self.assertIn("NEW.md", shown)

    def test_end_also_carries_commits_for_one_shot_parity(self):
        end = next(f for f in self.frames if f["type"] == "end")
        self.assertEqual(len(end["commits"]), 1)
        self.assertEqual(end["commits"][0], {"hash": self._commits()[0]["hash"],
                                             "message": self._commits()[0]["message"]})

    def test_commit_is_not_nested_on_end(self):
        """The old shape, guarded so it cannot creep back in."""
        end = next(f for f in self.frames if f["type"] == "end")
        self.assertNotIn("commit", end)

    def test_a_failed_run_commits_nothing(self):
        events = [{"type": "tool_use", "sessionID": "ses_c",
                   "part": {"tool": "write",
                            "state": {"status": "completed",
                                      "input": {"filePath": "NEW.md"}}}}]
        # setUp already committed once, so count against that rather than
        # against an absolute number: what matters is that the failed run adds
        # nothing on top.
        before = _log_len(self.repo)
        frames, fake = _stream(bodies=[_lines(*events)], writes=["NEW.md"],
                               returncodes=[1])
        self.assertEqual([f["type"] for f in frames].count("commit"), 0)
        self.assertEqual(_log_len(self.repo), before, "a failed run committed")


class TestStaleSessionRecovery(unittest.TestCase):
    """A --session id opencode no longer knows must not wedge code mode.

    opencode validates the id before anything else: an unrecognised one exits
    1 with a bare `Error: Session not found` on stderr and emits no JSON at
    all. Since remember_session() refuses to store an empty id, the bad entry
    used to survive and every later message failed identically — the only way
    out was /api/code/web/clear, which nothing on screen suggests.
    """

    STALE = "Error: Session not found\n"

    def setUp(self):
        self.work = _use_repo("stalerun")
        oc._write_sessions({})
        self.addCleanup(oc._write_sessions, {})
        oc.remember_session(self.work, "ses_gone")

    def test_session_is_stale_recognises_the_error(self):
        self.assertTrue(oc.session_is_stale(self.STALE))
        self.assertTrue(oc.session_is_stale("Error: \x1b[0mSession not found"))
        for other in ("", "session resumed", "error: model not found",
                      "connection refused", "no session id given"):
            self.assertFalse(oc.session_is_stale(other), other)

    def test_stream_retries_without_the_stale_id(self):
        frames, fake = _stream(bodies=[self.STALE, _lines(EVENT_TEXT)],
                               returncodes=[1, 0])
        self.assertEqual(len(fake.commands), 2, "the run was not retried")
        first, second = (_oc_argv(c) for c in fake.commands)
        self.assertIn("--session", first)
        self.assertEqual(first[first.index("--session") + 1], "ses_gone")
        self.assertNotIn("--session", second,
                         "the retry must start a fresh session")
        self.assertEqual(frames[-1]["type"], "end")
        self.assertEqual(frames[-1]["rc"], 0)

    def test_the_dead_end_is_never_shown_to_the_user(self):
        frames, _ = _stream(bodies=[self.STALE, _lines(EVENT_TEXT)],
                            returncodes=[1, 0])
        self.assertNotIn("Session not found",
                         " ".join(str(f.get("text", "")) for f in frames))
        self.assertNotIn("error", [f["type"] for f in frames])

    def test_the_map_is_repaired_by_the_retry(self):
        _stream(bodies=[self.STALE, _lines(EVENT_TEXT)], returncodes=[1, 0])
        self.assertEqual(oc.get_session(self.work), "ses_abc")

    def test_only_one_retry_is_attempted(self):
        frames, fake = _stream(bodies=[self.STALE, self.STALE, self.STALE],
                               returncodes=[1, 1, 1])
        self.assertEqual(len(fake.commands), 2, "retried more than once")

    def test_a_log_line_after_real_output_is_not_a_stale_session(self):
        """Only a refusal arriving with nothing in front of it counts —
        otherwise a healthy run that merely mentions it gets run twice."""
        body = _lines(EVENT_TEXT) + self.STALE
        frames, fake = _stream(bodies=[body], returncodes=[0])
        self.assertEqual(len(fake.commands), 1, "retried a healthy run")
        self.assertEqual(frames[0]["type"], "chunk")

    def test_a_healthy_run_is_not_retried(self):
        frames, fake = _stream(bodies=[_lines(EVENT_TEXT)], returncodes=[0])
        self.assertEqual(len(fake.commands), 1)
        self.assertEqual(frames[-1]["type"], "end")

    def test_one_shot_run_also_recovers(self):
        body, fake = _once(bodies=[self.STALE, _lines(EVENT_TEXT)],
                           returncodes=[1, 0])
        self.assertEqual(len(fake.commands), 2)
        self.assertNotIn("--session", _oc_argv(fake.commands[1]))
        self.assertNotEqual(oc.get_session(self.work), "ses_gone")
        self.assertEqual(body["rc"], 0)

    def test_one_shot_run_recovers_and_still_commits(self):
        _once(bodies=[self.STALE, _lines(
            {"type": "tool_use", "sessionID": "ses_c",
             "part": {"tool": "write", "state": {"status": "completed",
                     "input": {"filePath": "NEW.md"}}}},
            EVENT_TEXT)], returncodes=[1, 0])
        self.assertEqual(oc.get_session(self.work), "ses_abc")


class TestSessionMapKeys(unittest.TestCase):
    """One real folder must be one entry, however it was spelled.

    ~/storage/shared is a symlink into /storage/emulated/0 on Android, so keying
    on the raw string filed the same directory twice and made the conversation
    look lost the moment you moved between the default workdir and that folder
    added as a project.
    """

    def setUp(self):
        oc._write_sessions({})
        self.addCleanup(oc._write_sessions, {})
        self.tmp = Path(tempfile.mkdtemp(prefix="oc-keys-"))
        self.real = self.tmp / "real"
        self.real.mkdir()
        self.link = self.tmp / "link"
        self.link.symlink_to(self.real)

    def test_symlink_and_target_share_one_entry(self):
        oc.remember_session(self.real, "ses_1")
        self.assertEqual(oc.get_session(self.link), "ses_1")
        self.assertEqual(len(oc._read_sessions()), 1)

    def test_trailing_slash_and_dot_dot_are_the_same_key(self):
        oc.remember_session(str(self.real) + "/", "ses_1")
        self.assertEqual(oc.get_session(self.tmp / "real" / ".." / "real"),
                         "ses_1")
        self.assertEqual(len(oc._read_sessions()), 1)

    def test_forget_through_a_symlink_removes_the_entry(self):
        oc.remember_session(self.real, "ses_1")
        self.assertEqual(oc.forget_session(self.link), "ses_1")
        self.assertEqual(oc._read_sessions(), {})

    def test_different_folders_stay_separate(self):
        other = self.tmp / "other"
        other.mkdir()
        oc.remember_session(self.real, "ses_1")
        oc.remember_session(other, "ses_2")
        self.assertEqual(oc.get_session(self.real), "ses_1")
        self.assertEqual(oc.get_session(other), "ses_2")

    def test_a_path_that_cannot_be_resolved_does_not_raise(self):
        """A folder deleted between runs must not break the map."""
        gone = self.tmp / "never-existed"
        self.assertIsNone(oc.get_session(gone))
        self.assertIsNone(oc.forget_session(gone))



class TestOneShotRunsHoldCodeLock(unittest.TestCase):
    """The one-shot routes must serialise through code_lock like the stream.

    api_code and api_code_web_run called _run() with no lock, while
    stream_opencode held code_lock for a whole streamed run — so a one-shot
    run and a streamed run on the same folder could drive opencode at once
    and corrupt files through its write/edit tools.
    """

    def _fake_run(self):
        """A stand-in for _run that records the lock state it saw.

        Returns (0, []) — _run's (returncode, lines) contract.
        """
        seen = {}

        def fake(workdir, prompt, *args, **kwargs):
            seen["locked"] = rstate.code_lock.locked()
            return 0, []

        return seen, fake

    def _assert_locked(self, seen, resp):
        self.assertEqual(resp.status_code, 200, resp.get_data(as_text=True))
        self.assertIn("locked", seen, "the fake _run was never called")
        self.assertTrue(seen["locked"],
                        "the one-shot run did not hold code_lock")

    def test_api_code_runs_under_code_lock(self):
        seen, fake = self._fake_run()
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(rc, "_run", new=fake), \
                 patch.object(rc, "_require_opencode", return_value=None), \
                 patch.object(rc, "commit_changes",
                              return_value={"ok": False, "reason": "test"}), \
                 app.test_request_context(
                     "/api/code", method="POST",
                     json={"task": "x", "workdir": tmp}):
                resp = rc.api_code()
        self._assert_locked(seen, resp)

    def test_api_code_web_run_runs_under_code_lock(self):
        seen, fake = self._fake_run()
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(rc, "_run", new=fake), \
                 patch.object(rc, "_require_opencode", return_value=None), \
                 patch.object(rc, "_resolve_workdir",
                              return_value=(Path(tmp), None)), \
                 patch.object(rc, "ensure_git_identity"), \
                 patch.object(rc, "_ensure_safe_directory"), \
                 patch.object(rc, "commit_changes",
                              return_value={"ok": False, "reason": "test"}), \
                 app.test_request_context(
                     "/api/code/web/run", method="POST",
                     json={"message": "x"}):
                resp = rc.api_code_web_run()
        self._assert_locked(seen, resp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
