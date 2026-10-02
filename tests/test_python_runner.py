"""tools/python_runner.py — Session 23 (Q1): project-only Python runs."""
from _harness import bootstrap

bootstrap()

import tempfile  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402

import tools.python_runner as pr  # noqa: E402
from tools.python_runner import run_python  # noqa: E402


class _TempProject(unittest.TestCase):
    """Base: run against a throwaway 'project root' instead of the real repo."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="pyrunner-")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()
        self._real_root = pr.PROJECT_ROOT
        pr.PROJECT_ROOT = self.root
        self.addCleanup(setattr, pr, "PROJECT_ROOT", self._real_root)

    def script(self, name: str, body: str) -> str:
        p = self.root / name
        p.write_text(body, encoding="utf-8")
        return name


class TestPathRules(_TempProject):
    def test_project_root_is_the_repo(self):
        self.assertEqual(self._real_root, Path(pr.__file__).resolve().parent.parent)

    def test_runs_a_project_script(self):
        name = self.script("ok.py", 'print("hello-from-script")\n')
        self.assertEqual(run_python(name), "hello-from-script")

    def test_relative_paths_resolve_from_the_project_root(self):
        (self.root / "sub").mkdir()
        (self.root / "sub" / "nested.py").write_text('print("nested")\n')
        self.assertEqual(run_python("sub/nested.py"), "nested")

    def test_file_outside_the_project_is_refused(self):
        with tempfile.TemporaryDirectory(prefix="outside-") as other:
            p = Path(other) / "evil.py"
            p.write_text("print('nope')\n")
            out = run_python(str(p))
            self.assertIn("outside the project folder", out)

    def test_absolute_system_path_is_refused(self):
        self.assertIn("outside the project folder", run_python("/etc/passwd"))

    def test_non_python_file_is_refused(self):
        (self.root / "notes.txt").write_text("hi\n")
        self.assertIn("only .py files", run_python("notes.txt"))

    def test_missing_file_is_refused(self):
        self.assertIn("file not found", run_python("ghost_12345.py"))


class TestGuards(_TempProject):
    def test_socket_network_is_blocked(self):
        name = self.script(
            "net.py",
            "import socket\n"
            "socket.create_connection(('127.0.0.1', 9))\n",
        )
        out = run_python(name)
        self.assertIn("network access is disabled", out)

    def test_urllib_network_is_blocked(self):
        name = self.script(
            "url.py",
            "import urllib.request\n"
            "urllib.request.urlopen('http://127.0.0.1:1', timeout=1)\n",
        )
        out = run_python(name)
        self.assertIn("network access is disabled", out)

    def test_shell_out_is_blocked(self):
        name = self.script("sh.py", "import os\nos.system('echo leaked')\n")
        out = run_python(name)
        self.assertIn("shell commands are disabled", out)

    def test_network_binary_through_subprocess_is_blocked(self):
        name = self.script(
            "curl.py",
            "import subprocess\nsubprocess.run(['curl', 'http://x'])\n",
        )
        out = run_python(name)
        self.assertIn("network access is disabled", out)

    def test_ctypes_import_is_blocked(self):
        name = self.script("ctypes.py", "import ctypes\n")
        out = run_python(name)
        self.assertIn("ctypes halted", out)

    def test_timeout_kills_the_run(self):
        name = self.script("slow.py", "import time\ntime.sleep(30)\n")
        out = run_python(name, timeout=1)
        self.assertIn("timed out (1s)", out)

    def test_timeout_is_clamped_to_the_maximum(self):
        self.assertEqual(pr._clamp_timeout(9999), pr.MAX_TIMEOUT)
        self.assertEqual(pr._clamp_timeout(0), 1)
        self.assertEqual(pr._clamp_timeout(-5), 1)
        self.assertEqual(pr._clamp_timeout(pr.TIMEOUT_SEC), pr.TIMEOUT_SEC)
        name = self.script("ok.py", 'print("ok")\n')
        self.assertEqual(run_python(name, timeout="abc"),
                         "Error: timeout must be an integer")

    def test_output_is_truncated_to_the_cap(self):
        name = self.script("big.py", "print('y' * 20000)\n")
        out = run_python(name)
        self.assertIn("truncated", out)

    def test_silent_failure_reports_the_exit_code(self):
        name = self.script("fail.py", "raise SystemExit(3)\n")
        self.assertIn("exit code 3", run_python(name))


if __name__ == "__main__":
    unittest.main()
