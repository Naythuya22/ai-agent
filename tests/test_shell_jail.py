"""Release 26 — run_command joins the file jail (FIX 4, eval #4).

The eval: run_command executed `cat /etc/hostname` — an absolute path
outside the root — because Release 16 jailed only the file tools.
Now (while AI_AGENT_FILE_JAIL is on): the child runs with cwd pinned
to the current jail root, and any path-shaped argument resolving
outside the allowed roots (the jail root, $PREFIX, the system temp
dir, PATH directories) is refused BEFORE execution with a BLOCKED
tool result. Full shell confinement is impossible (an allowlisted
binary can be argued into many shapes) — this is the same pragmatic
gate as R16, and the notes say so.

Also pinned: the one interpreter exception — a bare
`python3 --version` probe runs; every other interpreter use stays
refused — and the AI_AGENT_FILE_JAIL=off opt-out restoring the old
behaviour wholesale.

Run:  sh tests/run.sh -p "test_shell_jail.py"
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from _harness import bootstrap

bootstrap()

import config  # noqa: E402
from tools import shell  # noqa: E402

SECRET = "OUTSIDE-JAIL-SECRET-7f3a"


def _real_home() -> Path:
    """The account's real home directory — from the passwd entry.

    NOT ``Path.home()`` and NOT ``$HOME``: tests/run.sh exports a
    throwaway HOME under the system temp dir and
    ``_harness.bootstrap`` re-points it again, so by the time any
    test runs, the environment's idea of "home" sits INSIDE the
    temp dir — one of the shell gate's allowed roots. The passwd
    entry is immune to that redirection and names the real home on
    both platforms (the VM user's home; on Termux
    ``/data/data/com.termux/files/home`` — a sibling of $PREFIX,
    not inside it, not the temp dir, not a PATH dir). Falls back to
    ``Path.home()`` when no passwd entry exists.
    """
    try:
        import pwd
        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except Exception:
        return Path.home()


class ShellJailCase(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp(prefix="shell-jail-"))
        self.jail = self.base / "jail"
        self.jail.mkdir()
        (self.jail / "note.txt").write_text("inside the jail",
                                            encoding="utf-8")
        (self.jail / "sub").mkdir()
        (self.base / "secret.txt").write_text(SECRET, encoding="utf-8")
        # The "outside" secret must sit outside EVERY allowed root —
        # including the system temp dir, which the spec allows and
        # which is where mkdtemp (and the harness HOME) live. The
        # real home dir is none of the four roots on either platform,
        # so it hosts the outside files; everything is removed in
        # tearDown.
        #
        # PORTABILITY (R26.1): the parent MUST be _real_home() (the
        # passwd entry), never a hardcoded path and never $HOME /
        # Path.home(). R26 hardcoded one machine's home (/home/…)
        # and the class errored in setUp on every other machine —
        # the phone run failed all 16 tests before any of them ran.
        # The env-derived home is equally wrong here: the harness
        # redirects it under the temp dir (see _real_home), i.e.
        # INSIDE an allowed root, which would silently disarm the
        # BLOCKED assertions instead. The assert below fails loudly,
        # in setUp, if a future platform ever lands the fixture
        # inside a root again.
        import shutil
        self.outside = Path(tempfile.mkdtemp(
            prefix="shell-jail-out-", dir=_real_home()))
        self.addCleanup(shutil.rmtree, self.outside, True)
        from tools._fs_guard import is_safe_path
        for _r in shell._allowed_roots(str(self.jail)):
            assert not is_safe_path(str(self.outside), _r), (
                f"outside fixture {self.outside} landed inside an "
                f"allowed root ({_r}) — the BLOCKED tests below "
                "would be meaningless; see _real_home()")
        (self.outside / "secret.txt").write_text(SECRET,
                                                 encoding="utf-8")
        self._saved_root = config.PROJECT_ROOT
        config.PROJECT_ROOT = self.jail
        self.addCleanup(setattr, config, "PROJECT_ROOT", self._saved_root)
        self._saved_jail = config.FILE_JAIL_ENABLED
        config.FILE_JAIL_ENABLED = True
        self.addCleanup(setattr, config, "FILE_JAIL_ENABLED",
                        self._saved_jail)
        self._saved_prefix = os.environ.get("PREFIX")
        self.addCleanup(self._restore_prefix)

    def _restore_prefix(self):
        if self._saved_prefix is None:
            os.environ.pop("PREFIX", None)
        else:
            os.environ["PREFIX"] = self._saved_prefix

    # ── refusals (nothing executes) ──────────────────────────────

    def test_eval_command_cat_etc_hostname_is_blocked(self):
        out = shell.run_command("cat /etc/hostname")
        self.assertIn("BLOCKED", out)

    def test_absolute_path_outside_is_blocked_and_not_executed(self):
        out = shell.run_command(f"cat {self.outside}/secret.txt")
        self.assertIn("BLOCKED", out)
        self.assertNotIn(SECRET, out)

    def test_blocked_token_anywhere_in_pipeline_refuses_all(self):
        out = shell.run_command(f"ls | cat {self.outside}/secret.txt")
        self.assertIn("BLOCKED", out)
        self.assertNotIn(SECRET, out)

    def test_traversal_out_of_the_jail_is_blocked(self):
        out = shell.run_command("cat ../secret.txt")
        self.assertIn("BLOCKED", out)
        self.assertNotIn(SECRET, out)

    def test_tilde_expansion_outside_the_jail_is_blocked(self):
        # HOME pointed at the outside dir for this one command: the
        # gate expands ~ itself, so what matters is where it lands.
        saved = os.environ.get("HOME")
        os.environ["HOME"] = str(self.outside)
        self.addCleanup(os.environ.__setitem__, "HOME", saved)
        out = shell.run_command("cat ~/secret.txt")
        self.assertIn("BLOCKED", out)
        self.assertNotIn(SECRET, out)

    def test_storage_style_path_is_blocked(self):
        out = shell.run_command("ls /storage/emulated/0/DCIM")
        self.assertIn("BLOCKED", out)

    # ── what still works ─────────────────────────────────────────

    def test_relative_read_inside_root_works(self):
        self.assertEqual(shell.run_command("cat note.txt"),
                         "inside the jail")

    def test_absolute_read_inside_root_works(self):
        self.assertEqual(shell.run_command(f"cat {self.jail}/note.txt"),
                         "inside the jail")

    def test_traversal_that_stays_inside_works(self):
        self.assertEqual(shell.run_command("cat sub/../note.txt"),
                         "inside the jail")

    def test_cwd_is_pinned_to_the_jail_root(self):
        out = shell.run_command("pwd")
        self.assertEqual(Path(out.strip()), self.jail.resolve())

    def test_ls_lists_the_jail_root(self):
        out = shell.run_command("ls")
        self.assertIn("note.txt", out)

    def test_python3_version_probe_works(self):
        out = shell.run_command("python3 --version")
        self.assertIn("Python 3", out)

    def test_other_interpreter_uses_stay_refused(self):
        self.assertIn("ခွင့်မပြု",
                      shell.run_command("python3 -c 'print(1)'"))
        self.assertIn("ခွင့်မပြု",
                      shell.run_command("python3 note.py"))
        self.assertIn("ခွင့်မပြု",
                      shell.run_command("python3 --version extra"))

    def test_prefix_root_is_allowed(self):
        prefix = self.base / "prefix"
        (prefix / "share").mkdir(parents=True)
        (prefix / "share" / "x.txt").write_text("from prefix",
                                                encoding="utf-8")
        os.environ["PREFIX"] = str(prefix)
        self.assertEqual(
            shell.run_command(f"cat {prefix}/share/x.txt"), "from prefix")
        # The $PREFIX spelling: the gate judges the EXPANDED path and
        # must not block it. (The child itself receives the token
        # literally — shell=False never expanded variables, before
        # this release either — so execution is not asserted here.)
        out = shell.run_command("cat $PREFIX/share/x.txt")
        self.assertNotIn("BLOCKED", out)

    def test_temp_dir_is_allowed(self):
        f = Path(tempfile.gettempdir()) / "shell-jail-tmp-ok.txt"
        f.write_text("from temp", encoding="utf-8")
        self.addCleanup(f.unlink, missing_ok=True)
        self.assertEqual(shell.run_command(f"cat {f}"), "from temp")

    # ── opt-out ──────────────────────────────────────────────────

    def test_opt_out_restores_old_behaviour(self):
        config.FILE_JAIL_ENABLED = False
        out = shell.run_command(f"cat {self.outside}/secret.txt")
        self.assertEqual(out, SECRET)          # executed, as before
        out = shell.run_command("cat /etc/hostname")
        self.assertNotIn("BLOCKED", out)


if __name__ == "__main__":
    unittest.main()
