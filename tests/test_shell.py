"""tools/shell.py — Session 23 (Q1): allowlist-only, shell=False execution."""
from _harness import bootstrap

bootstrap()

import os  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402

from tools import shell  # noqa: E402


class TestValidation(unittest.TestCase):
    """Rejections — every input below must never reach a process."""

    def test_rejects_empty(self):
        self.assertEqual(shell.run_command(""), "Error: empty command")
        self.assertEqual(shell.run_command("   "), "Error: empty command")

    def test_interpreters_are_off_the_allowlist(self):
        for cmd in (
            "python3 -c print(1)",
            "python -c print(1)",
            "python3 -m http.server 9999",
            "node script.js",
            "awk 'BEGIN{system(\"id\")}'",
            "sed s/a/b/f",
        ):
            with self.subTest(cmd=cmd):
                self.assertIsNotNone(shell._validate(cmd))
                self.assertIn("command ခွင့်မပြု", shell.run_command(cmd))

    def test_shell_syntax_is_rejected(self):
        for cmd in (
            "ls; id",
            "ls && id",
            "ls || id",
            "ls\nid",
            "ls\rmkdir x",
            "echo `id`",
            "echo $(id)",
            "echo ${HOME}",
            "echo hi > /tmp/f",
            "echo hi >f",
            "cat < /etc/passwd",
            "echo x | sh",
        ):
            with self.subTest(cmd=cmd):
                self.assertIsNotNone(shell._validate(cmd))
                self.assertTrue(shell.run_command(cmd).startswith("Error:"))

    def test_write_and_network_commands_are_rejected(self):
        for cmd in (
            "rm -rf /",
            "mv a b",
            "cp a b",
            "mkdir /tmp/x",
            "chmod 777 f",
            "sudo ls",
            "curl http://evil",
            "wget http://evil",
            "apt install x",
            "kill 1",
        ):
            with self.subTest(cmd=cmd):
                self.assertIsNotNone(shell._validate(cmd))
                self.assertTrue(shell.run_command(cmd).startswith("Error:"))

    def test_git_config_write_is_blocked_but_read_is_allowed(self):
        self.assertIsNone(shell._validate("git log --oneline -3"))
        self.assertIsNone(shell._validate("git status"))
        self.assertIsNone(shell._validate("git config --get user.name"))
        self.assertIsNone(shell._validate("git config --list"))
        for cmd in ("git config core.pager 'sh -c id'",
                    "git config --global core.sshCommand curl",
                    "git push", "git commit -m x"):
            with self.subTest(cmd=cmd):
                self.assertIsNotNone(shell._validate(cmd))

    def test_pip_only_read_only_subcommands(self):
        self.assertIsNone(shell._validate("pip list"))
        self.assertIsNone(shell._validate("pip3 show flask"))
        for cmd in ("pip install evil", "pip3 uninstall x"):
            with self.subTest(cmd=cmd):
                self.assertIsNotNone(shell._validate(cmd))

    def test_allowed_read_commands_still_pass(self):
        for cmd in ("ls -la", "cat README.md", "echo one", "free -h",
                    "grep -r foo .", "find . -name '*.py'", "wc -l",
                    "du -h", "uname -a"):
            with self.subTest(cmd=cmd):
                self.assertIsNone(shell._validate(cmd))


class TestRunCommand(unittest.TestCase):
    """Actual execution: argv list, scrubbed env, pipes, caps."""

    def test_echo_runs_and_returns_stdout(self):
        self.assertEqual(shell.run_command("echo hello"), "hello")

    def test_pipeline_is_chained_without_a_shell(self):
        out = shell.run_command("echo one two | wc -w")
        self.assertEqual(out.strip().split()[0], "2")

    def test_pipeline_honours_the_allowlist_on_every_segment(self):
        self.assertIn("command ခွင့်မပြု",
                      shell.run_command("echo x | python3 -c print(1)"))

    def test_child_env_is_scrubbed(self):
        os.environ["AI_AGENT_SECRET_TEST"] = "leaked-secret-42"
        try:
            out = shell.run_command("env")
            self.assertNotIn("leaked-secret-42", out)
            self.assertIn("PATH=", out)
        finally:
            os.environ.pop("AI_AGENT_SECRET_TEST", None)

    def test_output_is_truncated_to_the_cap(self):
        out = shell.run_command("echo " + "x" * 5000)
        self.assertIn("truncated", out)
        self.assertLess(len(out.splitlines()[0]), 5000)

    def test_timeout_kills_the_run(self):
        out = shell.run_command("sleep 30", timeout=1)
        self.assertIn("timed out", out)

    def test_cwd_exists(self):
        self.assertTrue(Path(shell._cwd()).is_dir())

    def test_stderr_of_the_last_stage_is_reported(self):
        out = shell.run_command("cat /nonexistent-ai-agent-file")
        self.assertIn("No such file", out)


if __name__ == "__main__":
    unittest.main()
