"""openclaude backend tests: command, config, env, sessions, stream parsing.

No network and no logged-in CLI — the command builder, the session map and
the stream-json normalizer are all pure functions, and the subprocess-based
ones are driven against a fake child, exactly as tests/test_code.py does for
opencode.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from _harness import bootstrap, current_home

bootstrap()

from core import opencode as oc  # noqa: E402
from core import openclaude as ocl  # noqa: E402
from core import openclaude_stream as ocs  # noqa: E402

PROMPT = "do the thing"


# ── build_command ─────────────────────────────────────────────────────────

class TestBuildCommand(unittest.TestCase):
    def test_runs_directly_without_proot(self):
        cmd = ocl.build_command(PROMPT, "/w")
        self.assertEqual(ocl.PROOT_CMD, [])
        self.assertEqual(cmd[0], "openclaude")

    def test_stream_json_flags_are_present(self):
        cmd = ocl.build_command(PROMPT, "/w")
        self.assertIn("-p", cmd)
        self.assertIn("--output-format", cmd)
        self.assertEqual(cmd[cmd.index("--output-format") + 1], "stream-json")
        self.assertIn("--include-partial-messages", cmd)
        # Without --verbose the CLI refuses stream-json under --print.
        self.assertIn("--verbose", cmd)

    def test_auto_bypasses_permissions(self):
        cmd = ocl.build_command(PROMPT, "/w", auto=True)
        self.assertEqual(cmd[cmd.index("--permission-mode") + 1],
                         "bypassPermissions")

    def test_auto_off_leaves_the_cli_default(self):
        cmd = ocl.build_command(PROMPT, "/w", auto=False)
        self.assertNotIn("--permission-mode", cmd)

    def test_model_defaults_to_sonnet(self):
        cmd = ocl.build_command(PROMPT, "/w")
        self.assertEqual(cmd[cmd.index("--model") + 1], "sonnet")

    def test_model_can_be_named(self):
        cmd = ocl.build_command(PROMPT, "/w", model="opus")
        self.assertEqual(cmd[cmd.index("--model") + 1], "opus")

    def test_session_id_is_passed_through(self):
        sid = "6f0f2a4c-9e1b-4a53-8a9a-1f0c9a2b7c11"
        cmd = ocl.build_command(PROMPT, "/w", session_id=sid)
        self.assertEqual(cmd[cmd.index("--session-id") + 1], sid)

    def test_agent_is_passed_through(self):
        cmd = ocl.build_command(PROMPT, "/w", agent="reviewer")
        self.assertEqual(cmd[cmd.index("--agent") + 1], "reviewer")

    def test_a_file_adds_its_folder(self):
        tmp = Path(current_home()) / "cl-files"
        (tmp / "sub").mkdir(parents=True, exist_ok=True)
        cmd = ocl.build_command(PROMPT, "/w",
                                files=[str(tmp / "sub" / "a.py")])
        self.assertEqual(cmd[cmd.index("--add-dir") + 1], str(tmp / "sub"))

    def test_a_folder_adds_itself_and_duplicates_collapse(self):
        tmp = Path(current_home()) / "cl-dir"
        tmp.mkdir(parents=True, exist_ok=True)
        cmd = ocl.build_command(PROMPT, "/w", files=[str(tmp), str(tmp)])
        self.assertEqual(cmd.count("--add-dir"), 1)
        self.assertEqual(cmd[cmd.index("--add-dir") + 1], str(tmp))

    def test_prompt_is_the_last_argument(self):
        cmd = ocl.build_command(PROMPT, "/w", files=["/tmp"])
        self.assertEqual(cmd[-1], PROMPT)

    def test_logs_flag_is_accepted_for_signature_parity(self):
        cmd = ocl.build_command(PROMPT, "/w", logs=True)
        self.assertEqual(cmd[-1], PROMPT)


# ── config / session file ─────────────────────────────────────────────────

class TestConfigPaths(unittest.TestCase):
    def test_config_path_is_the_cli_own_file(self):
        self.assertEqual(ocl.config_file(),
                         Path.home() / ".openclaude" / ".openclaude.json")

    def test_session_store_is_separate_from_opencode(self):
        self.assertEqual(ocl.session_file(),
                         Path("/sdcard/.ai-agent/code_sessions_openclaude.json"))
        self.assertNotEqual(ocl.session_file(), oc.session_file())
        self.assertTrue(str(oc.session_file()).endswith("code_sessions.json"))

    def test_writing_the_bridge_config_touches_nothing(self):
        target = ocl.config_file()
        self.assertEqual(ocl.write_bridge_config(base="http://x"), target)
        self.assertFalse(target.exists())

    def test_bridge_config_is_always_empty(self):
        self.assertEqual(ocl.bridge_config(base="http://x"), {})
        self.assertEqual(ocl.bridge_config(), {})


# ── env ───────────────────────────────────────────────────────────────────

class TestEnv(unittest.TestCase):
    def test_ld_preload_never_reaches_the_child(self):
        with patch.dict(os.environ, {"LD_PRELOAD": "/sentinel/musl.so"}):
            self.assertIsNone(ocl.env().get("LD_PRELOAD"))

    def test_ld_preload_is_restored_in_the_parent(self):
        with patch.dict(os.environ, {"LD_PRELOAD": "/sentinel/musl.so"}):
            ocl.env()
            self.assertEqual(os.environ.get("LD_PRELOAD"), "/sentinel/musl.so")

    def test_ld_library_path_is_kept(self):
        with patch.dict(os.environ, {"LD_LIBRARY_PATH": "/sentinel/libdir"}):
            self.assertEqual(ocl.env().get("LD_LIBRARY_PATH"),
                             "/sentinel/libdir")

    def test_anthropic_key_is_handed_to_the_child(self):
        with patch.object(ocl, "anthropic_api_key", return_value="sk-test"):
            self.assertEqual(ocl.env().get("ANTHROPIC_API_KEY"), "sk-test")

    def test_an_env_key_wins_over_the_stored_one(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-from-env"}):
            self.assertEqual(ocl.env().get("ANTHROPIC_API_KEY"), "sk-from-env")

    def test_extra_keys_are_applied(self):
        self.assertEqual(ocl.env(PYTHONUNBUFFERED="1").get("PYTHONUNBUFFERED"),
                         "1")

    def test_no_key_means_no_env_entry(self):
        env = {k: v for k, v in os.environ.items()
               if k != "ANTHROPIC_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with patch.object(ocl, "anthropic_api_key", return_value=None):
                self.assertNotIn("ANTHROPIC_API_KEY", ocl.env())


# ── session map ───────────────────────────────────────────────────────────

class TestSessionStore(unittest.TestCase):
    """Runs against a temp store: /sdcard is phone storage, not a fixture."""

    def setUp(self):
        self._target = Path(current_home()) / "cl-sessions.json"
        patcher = patch.object(ocl, "session_file", return_value=self._target)
        patcher.start()
        self.addCleanup(patcher.stop)
        ocl._write_sessions({})
        self.addCleanup(ocl._write_sessions, {})

    def test_round_trip(self):
        sid = "6f0f2a4c-9e1b-4a53-8a9a-1f0c9a2b7c11"
        ocl.remember_session("/w/a", sid)
        self.assertEqual(ocl.get_session("/w/a"), sid)

    def test_workspaces_are_separate(self):
        ocl.remember_session("/w/a", "ses_1")
        self.assertIsNone(ocl.get_session("/w/b"))

    def test_opencode_store_is_untouched(self):
        oc._write_sessions({})
        self.addCleanup(oc._write_sessions, {})
        ocl.remember_session("/w/shared", "ses-cl")
        self.assertIsNone(oc.get_session("/w/shared"))
        oc.remember_session("/w/shared", "ses-oc")
        self.assertEqual(ocl.get_session("/w/shared"), "ses-cl")
        self.assertEqual(oc.get_session("/w/shared"), "ses-oc")

    def test_forget_returns_what_it_removed(self):
        ocl.remember_session("/w/a", "ses_1")
        self.assertEqual(ocl.forget_session("/w/a"), "ses_1")
        self.assertIsNone(ocl.get_session("/w/a"))
        self.assertIsNone(ocl.forget_session("/w/a"))

    def test_corrupt_file_does_not_crash(self):
        self._target.parent.mkdir(parents=True, exist_ok=True)
        self._target.write_text("{not json")
        self.assertIsNone(ocl.get_session("/w/a"))

    def test_empty_session_id_is_not_stored(self):
        ocl.remember_session("/w/a", "")
        self.assertIsNone(ocl.get_session("/w/a"))

    def test_keys_resolve_like_opencodes(self):
        self.assertEqual(ocl._key("/w/a/../a"), oc._key("/w/a/../a"))


# ── install / version / doctor ────────────────────────────────────────────

class TestInstall(unittest.TestCase):
    def test_is_installed_follows_path(self):
        with patch.object(ocl.shutil, "which",
                          return_value="/usr/bin/openclaude"):
            self.assertTrue(ocl.is_installed())
        with patch.object(ocl.shutil, "which", return_value=None):
            self.assertFalse(ocl.is_installed())

    def test_version_is_none_when_the_binary_is_gone(self):
        def boom(*a, **k):
            raise FileNotFoundError("openclaude")
        with patch.object(ocl.subprocess, "run", side_effect=boom):
            self.assertIsNone(ocl.version())

    def test_missing_message_names_the_binary(self):
        self.assertIn("openclaude", ocl.missing_install_message())

    def test_session_delete_has_no_command_to_run(self):
        self.assertIsNone(ocl.session_delete_command("ses_1"))

    def test_stale_session_recognises_the_error(self):
        self.assertTrue(
            ocl.session_is_stale("No conversation found with session ID x"))
        self.assertTrue(ocl.session_is_stale("Error: session not found"))
        self.assertFalse(ocl.session_is_stale("everything is fine"))

    def test_tui_command_is_a_bare_binary(self):
        self.assertEqual(ocl.tui_command("/w"), ["openclaude"])


class TestDoctor(unittest.TestCase):
    def test_doctor_answers_and_reports_installed(self):
        with patch.object(ocl.shutil, "which",
                          return_value="/usr/bin/openclaude"):
            with patch.object(ocl.subprocess, "run"):
                info = ocl.doctor()
        self.assertTrue(info["installed"])
        self.assertEqual(info["binary"], "openclaude")
        self.assertEqual(info["doctor_cmd"], "openclaude doctor")
        self.assertIn("version", info)
        self.assertIsInstance(info["sessions"], dict)

    def test_doctor_never_runs_the_interactive_subcommand(self):
        """`openclaude doctor` is an Ink TUI: without a TTY it spins forever."""
        calls = []

        def record(cmd, **kw):
            calls.append(list(cmd))

            class Done:
                returncode = 0
                stdout = "0.32.0"
                stderr = ""
            return Done()

        with patch.object(ocl.shutil, "which",
                          return_value="/usr/bin/openclaude"):
            with patch.object(ocl.subprocess, "run", side_effect=record):
                ocl.doctor()
        self.assertTrue(all("doctor" not in c for c in calls), calls)

    def test_doctor_explains_a_missing_binary(self):
        with patch.object(ocl.shutil, "which", return_value=None):
            info = ocl.doctor()
        self.assertFalse(info["installed"])
        self.assertIn("openclaude", info["hint"])

    def test_is_installed_survives_a_broken_probe(self):
        with patch.object(ocl.shutil, "which", side_effect=OSError("boom")):
            self.assertFalse(ocl.is_installed())


# ── stream-json parsing ───────────────────────────────────────────────────

INIT = {
    "type": "system", "subtype": "init",
    "cwd": "/w", "session_id": "6f0f2a4c-9e1b-4a53-8a9a-1f0c9a2b7c11",
    "model": "claude-sonnet-4-6", "permissionMode": "bypassPermissions",
}

MESSAGE_START = {
    "type": "stream_event", "session_id": "ses_cl",
    "event": {"type": "message_start",
              "message": {"id": "msg_1", "role": "assistant",
                          "usage": {"input_tokens": 10, "output_tokens": 0}}},
}

TEXT_DELTA = {
    "type": "stream_event", "session_id": "ses_cl",
    "event": {"type": "content_block_delta", "index": 0,
              "delta": {"type": "text_delta", "text": "Hello "}},
}

TEXT_DELTA_2 = {
    "type": "stream_event", "session_id": "ses_cl",
    "event": {"type": "content_block_delta", "index": 0,
              "delta": {"type": "text_delta", "text": "world"}},
}

ASSISTANT_TEXT = {
    "type": "assistant", "session_id": "ses_cl",
    "message": {"role": "assistant",
                "content": [{"type": "text", "text": "Hello world"}]},
}

ASSISTANT_WRITE = {
    "type": "assistant", "session_id": "ses_cl",
    "message": {"role": "assistant", "content": [
        {"type": "tool_use", "id": "toolu_1", "name": "Write",
         "input": {"file_path": "/w/hello.txt", "content": "hi"}},
    ]},
}

ASSISTANT_EDIT = {
    "type": "assistant", "session_id": "ses_cl",
    "message": {"role": "assistant", "content": [
        {"type": "tool_use", "id": "toolu_2", "name": "Edit",
         "input": {"file_path": "/w/hello.txt"}},
    ]},
}

ASSISTANT_BASH = {
    "type": "assistant", "session_id": "ses_cl",
    "message": {"role": "assistant", "content": [
        {"type": "tool_use", "id": "toolu_3", "name": "Bash",
         "input": {"command": "ls -la"}},
    ]},
}

TOOL_RESULT = {
    "type": "user", "session_id": "ses_cl",
    "message": {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"},
    ]},
}

RESULT = {
    "type": "result", "subtype": "success", "is_error": False,
    "duration_ms": 825, "num_turns": 1,
    "result": "Created hello.txt",
    "session_id": "ses_cl",
    "total_cost_usd": 0.0042,
    "usage": {"input_tokens": 100, "output_tokens": 40,
              "cache_read_input_tokens": 10, "cache_creation_input_tokens": 5},
}

RESULT_ERROR = {
    "type": "result", "subtype": "error_during_execution", "is_error": True,
    "result": "Not logged in · Please run /login",
    "session_id": "ses_cl", "total_cost_usd": 0,
}


def _lines(*events):
    return "".join(json.dumps(e) + "\n" for e in events)


class TestParseEvent(unittest.TestCase):
    def test_parses_a_real_event(self):
        self.assertEqual(ocs.parse_event(json.dumps(INIT))["type"], "system")

    def test_blank_and_warning_lines_are_ignored(self):
        for junk in ("", "   ", "\n", "npm warn deprecated", "not json {"):
            self.assertIsNone(ocs.parse_event(junk), junk)

    def test_strips_ansi_before_parsing(self):
        self.assertIsNotNone(ocs.parse_event("\x1b[32m" + json.dumps(INIT)))

    def test_json_scalar_is_not_an_event(self):
        for scalar in ("42", '"a string"', "[1,2]", "null"):
            self.assertIsNone(ocs.parse_event(scalar), scalar)


class TestNormalize(unittest.TestCase):
    def _feed(self, *raws):
        state = ocs.Normalizer()
        out = []
        for raw in raws:
            out.extend(state.feed(raw))
        return out

    def test_message_start_becomes_a_step(self):
        events = self._feed(MESSAGE_START)
        self.assertEqual(events, [{"type": "step", "n": 1,
                                   "session_id": "ses_cl"}])

    def test_steps_are_numbered_in_order(self):
        events = self._feed(MESSAGE_START, MESSAGE_START)
        self.assertEqual([e["n"] for e in events], [1, 2])

    def test_text_deltas_become_chunks(self):
        events = self._feed(MESSAGE_START, TEXT_DELTA, TEXT_DELTA_2)
        chunks = [e["text"] for e in events if e["type"] == "chunk"]
        self.assertEqual(chunks, ["Hello ", "world"])

    def test_a_message_is_not_reported_twice(self):
        events = self._feed(MESSAGE_START, TEXT_DELTA, TEXT_DELTA_2,
                            ASSISTANT_TEXT)
        chunks = [e["text"] for e in events if e["type"] == "chunk"]
        self.assertEqual(chunks, ["Hello ", "world"])

    def test_text_without_deltas_still_streams(self):
        events = self._feed(ASSISTANT_TEXT)
        chunks = [e["text"] for e in events if e["type"] == "chunk"]
        self.assertEqual(chunks, ["Hello world"])

    def test_a_later_message_streams_again(self):
        events = self._feed(MESSAGE_START, TEXT_DELTA, ASSISTANT_TEXT,
                            MESSAGE_START, ASSISTANT_TEXT)
        chunks = [e["text"] for e in events if e["type"] == "chunk"]
        self.assertEqual(chunks, ["Hello ", "Hello world"])

    def test_write_becomes_a_file_event(self):
        events = self._feed(ASSISTANT_WRITE)
        self.assertEqual(events[0]["type"], "file")
        self.assertEqual(events[0]["op"], "create")
        self.assertEqual(events[0]["path"], "/w/hello.txt")

    def test_edit_becomes_an_edit_file_event(self):
        events = self._feed(ASSISTANT_EDIT)
        self.assertEqual((events[0]["type"], events[0]["op"]),
                         ("file", "edit"))

    def test_a_non_edit_tool_stays_a_tool_event(self):
        events = self._feed(ASSISTANT_BASH)
        self.assertEqual(events[0]["type"], "tool")
        self.assertEqual(events[0]["tool"], "Bash")
        self.assertEqual(events[0]["title"], "ls -la")
        self.assertNotIn("path", events[0])

    def test_a_tool_is_announced_once(self):
        events = self._feed(ASSISTANT_BASH, ASSISTANT_BASH)
        self.assertEqual(len([e for e in events if e["type"] == "tool"]), 1)

    def test_tool_results_add_nothing(self):
        self.assertEqual(self._feed(TOOL_RESULT), [])

    def test_init_adds_nothing(self):
        self.assertEqual(self._feed(INIT), [])

    def test_result_becomes_an_end_event(self):
        end = self._feed(RESULT)[0]
        self.assertEqual(end["type"], "end")
        self.assertEqual(end["reply"], "Created hello.txt")
        self.assertEqual(end["cost"], 0.0042)
        self.assertEqual(end["session_id"], "ses_cl")
        self.assertIsNone(end["error"])
        self.assertEqual(end["tokens"]["input"], 100)
        self.assertEqual(end["tokens"]["total"], 155)

    def test_an_error_result_carries_the_reason(self):
        end = self._feed(RESULT_ERROR)[0]
        self.assertEqual(end["type"], "end")
        self.assertIn("Not logged in", end["error"])

    def test_an_unknown_type_is_ignored(self):
        self.assertEqual(self._feed({"type": "hook_event", "x": 1}), [])

    def test_non_dicts_are_ignored(self):
        self.assertEqual(self._feed(None), [])
        self.assertEqual(ocs.normalize("nope"), [])


class TestParseLines(unittest.TestCase):
    def test_a_clean_transcript_normalizes_in_order(self):
        events = ocs.parse_lines(
            _lines(INIT, MESSAGE_START, TEXT_DELTA, ASSISTANT_WRITE, RESULT))
        self.assertEqual([e["type"] for e in events],
                         ["step", "chunk", "file", "end"])

    def test_non_json_lines_become_logs(self):
        events = ocs.parse_lines(["npm warn deprecated x\n",
                                  json.dumps(RESULT) + "\n"])
        self.assertEqual(events[0]["type"], "log")
        self.assertEqual(events[0]["text"], "npm warn deprecated x")
        self.assertEqual(events[-1]["type"], "end")

    def test_blank_lines_are_dropped(self):
        self.assertEqual(ocs.parse_lines(["", "   ", "\n"]), [])

    def test_ansi_is_stripped_from_log_lines(self):
        events = ocs.parse_lines(["\x1b[32mwarning\x1b[0m\n"])
        self.assertEqual(events[0]["text"], "warning")


class TestSummarize(unittest.TestCase):
    def test_reply_comes_from_the_chunks(self):
        events = ocs.parse_lines(
            _lines(MESSAGE_START, TEXT_DELTA, TEXT_DELTA_2, RESULT))
        summary = ocs.summarize(events)
        self.assertEqual(summary["reply"], "Hello world")
        self.assertEqual(summary["session_id"], "ses_cl")
        self.assertEqual(summary["cost"], 0.0042)
        self.assertIsNone(summary["error"])
        self.assertEqual(summary["commits"], [])

    def test_result_text_is_the_fallback_reply(self):
        summary = ocs.summarize(ocs.parse_lines(json.dumps(RESULT) + "\n"))
        self.assertEqual(summary["reply"], "Created hello.txt")

    def test_files_carry_op_and_path(self):
        summary = ocs.summarize(ocs.parse_lines(
            json.dumps(ASSISTANT_WRITE) + "\n"))
        self.assertEqual(summary["files"][0]["path"], "/w/hello.txt")
        self.assertEqual(summary["files"][0]["op"], "create")
        self.assertEqual(summary["tools"][0]["path"], "/w/hello.txt")

    def test_a_file_touched_twice_is_listed_once(self):
        summary = ocs.summarize(ocs.parse_lines(
            _lines(ASSISTANT_WRITE, ASSISTANT_EDIT)))
        self.assertEqual(len(summary["files"]), 1)
        # First sighting wins, same as core/code_stream.summarize(): a file
        # this run created is reported as created even if it was edited next.
        self.assertEqual(summary["files"][0]["op"], "create")

    def test_non_edit_tools_stay_out_of_files(self):
        summary = ocs.summarize(ocs.parse_lines(
            json.dumps(ASSISTANT_BASH) + "\n"))
        self.assertEqual(summary["files"], [])
        self.assertEqual(summary["tools"][0]["tool"], "Bash")

    def test_an_error_run_reports_it(self):
        summary = ocs.summarize(ocs.parse_lines(
            json.dumps(RESULT_ERROR) + "\n"))
        self.assertIn("Not logged in", summary["error"])

    def test_empty_stream_is_not_an_error(self):
        summary = ocs.summarize([])
        self.assertEqual(summary["reply"], "")
        self.assertIsNone(summary["error"])
        self.assertEqual(summary["files"], [])


# ── the streaming generator ───────────────────────────────────────────────

class _FakeClaude:
    """Stands in for `openclaude -p`, handing out one canned body per run.

    git/git-less callers are left to the real subprocess module: only the
    child whose argv names openclaude is faked.
    """

    def __init__(self, body="", returncode=0, alive=False):
        self.body = body
        self.returncode = returncode
        self.alive = alive
        self.commands = []
        self.terminated = 0
        self._real_popen = __import__("subprocess").Popen

    def _ours(self, cmd):
        return isinstance(cmd, (list, tuple)) and any(
            "openclaude" in str(a) for a in cmd)

    def popen(self, cmd, **kw):
        if not self._ours(cmd):
            return self._real_popen(cmd, **kw)
        self.commands.append(list(cmd))
        body, code, alive = self.body, self.returncode, self.alive

        class FakeProc:
            def __init__(self, inner):
                self.stdout = io.StringIO(body)
                self.inner = inner
                # A live child answers None from poll() until it is reaped.
                self.returncode = None if alive else code

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                return self.returncode

            def terminate(self):
                self.inner.terminated += 1
                self.returncode = 0

            def kill(self):
                self.returncode = -9

        return FakeProc(self)


def _stream(body, returncode=0, cmd=None, alive=False, **kw):
    """Run stream_openclaude against a fake child; returns (frames, fake)."""
    fake = _FakeClaude(body, returncode, alive=alive)
    real_popen = subprocess.Popen
    subprocess.Popen = fake.popen
    try:
        frames = list(ocs.stream_openclaude(
            cmd or ["openclaude", "-p", "hi"], workdir=current_home(),
            env={"PATH": os.environ.get("PATH", "")},
            lock=threading.Lock(), **kw))
    finally:
        subprocess.Popen = real_popen
    parsed = [json.loads(f[6:]) for f in frames if f.startswith("data: ")]
    return parsed, fake


class TestStreamOpenclaude(unittest.TestCase):
    def test_emits_step_chunk_file_and_end(self):
        body = _lines(MESSAGE_START, TEXT_DELTA, ASSISTANT_WRITE, RESULT)
        frames, fake = _stream(body)
        kinds = [f["type"] for f in frames]
        self.assertEqual(kinds, ["step", "chunk", "file", "end"])
        self.assertEqual(frames[1]["text"], "Hello ")
        self.assertEqual(frames[2]["path"], "/w/hello.txt")
        self.assertEqual(frames[-1]["rc"], 0)

    def test_end_frame_carries_the_summary(self):
        frames, _ = _stream(_lines(MESSAGE_START, TEXT_DELTA, RESULT))
        end = frames[-1]
        self.assertEqual(end["reply"], "Hello")
        self.assertEqual(end["session_id"], "ses_cl")
        self.assertEqual(end["cost"], 0.0042)
        self.assertEqual(end["tokens"]["input"], 100)
        self.assertIn("elapsed", end)

    def test_session_is_reported_as_soon_as_it_is_known(self):
        seen = []
        _stream(_lines(INIT, RESULT), on_session=seen.append)
        self.assertEqual(seen[0], "6f0f2a4c-9e1b-4a53-8a9a-1f0c9a2b7c11")

    def test_non_json_lines_are_surfaced_as_logs(self):
        frames, _ = _stream("npm warn deprecated x\n" + _lines(RESULT))
        self.assertEqual(frames[0], {"type": "log",
                                     "text": "npm warn deprecated x"})

    def test_a_failed_run_explains_itself(self):
        frames, _ = _stream(_lines(RESULT_ERROR), returncode=1)
        kinds = [f["type"] for f in frames]
        self.assertIn("error", kinds)
        err = next(f for f in frames if f["type"] == "error")
        self.assertIn("Not logged in", err["error"])
        self.assertEqual(kinds[-1], "end")

    def test_a_silent_failure_still_ends(self):
        frames, _ = _stream("", returncode=3)
        kinds = [f["type"] for f in frames]
        self.assertIn("error", kinds)
        self.assertEqual(kinds[-1], "end")
        self.assertEqual(frames[-1]["rc"], 3)

    def test_an_empty_success_ends_with_no_output(self):
        frames, _ = _stream(_lines(RESULT), returncode=0)
        self.assertEqual(frames[-1]["type"], "end")
        self.assertEqual(frames[-1]["reply"], "Created hello.txt")

    def test_on_event_sees_the_raw_event(self):
        seen = []
        _stream(_lines(RESULT), on_event=seen.append)
        self.assertEqual(seen[0]["type"], "result")

    def test_idle_timeout_kills_the_child(self):
        real_select = ocs.select.select

        def never_ready(*a, **k):
            return ([], [], [])

        ocs.select.select = never_ready
        saved_timeout = ocs.STREAM_IDLE_TIMEOUT
        ocs.STREAM_IDLE_TIMEOUT = 0.05
        try:
            frames, fake = _stream(_lines(RESULT), alive=True)
        finally:
            ocs.select.select = real_select
            ocs.STREAM_IDLE_TIMEOUT = saved_timeout
        kinds = [f["type"] for f in frames]
        self.assertIn("error", kinds)
        err = next(f for f in frames if f["type"] == "error")
        self.assertEqual(err["reason"], "idle_timeout")
        self.assertGreaterEqual(fake.terminated, 1)
        self.assertEqual(kinds[-1], "end")

    def test_the_argv_reaches_the_child_unchanged(self):
        cmd = ocl.build_command(PROMPT, "/w", session_id="ses_1")
        _, fake = _stream(_lines(RESULT), cmd=cmd)
        self.assertEqual(fake.commands[0], cmd)
        self.assertEqual(cmd[0], "openclaude")
        self.assertIn("--session-id", cmd)
