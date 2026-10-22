"""P0 File Security & Explorer — path jail, backups, tree/content APIs.

Covers the three pieces added by the P0 roadmap item:

  tools/_fs_guard.py   is_safe_path() / create_backup()
  tools/file.py        the jail on read_file/list_files/write_file and the
                       pre-overwrite backup line in write_file's reply
  routes/files.py      GET /api/files/tree and GET /api/files/content

Run:  sh tests/run.sh -p "test_file.py"
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from _harness import bootstrap, get_app

MODEL = bootstrap()
app = get_app()
client = app.test_client()

import config  # noqa: E402
from core import projects as ps  # noqa: E402
from tools import _fs_guard as fg  # noqa: E402
from tools import file as file_mod  # noqa: E402


class _TmpCase(unittest.TestCase):
    """A throwaway directory, removed after every test."""

    def setUp(self):
        super().setUp()
        self.tmp = Path(tempfile.mkdtemp(prefix="fs-guard-"))
        self.addCleanup(shutil.rmtree, str(self.tmp), True)

    def write(self, name, data=b""):
        p = self.tmp / name
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, str):
            p.write_text(data, encoding="utf-8")
        else:
            p.write_bytes(data)
        return p


class _ProjectStoreCase(_TmpCase):
    """Point the project store at a throwaway file and restore it after.

    The store is process-global, so any case that adds a project must put
    the previous file back — otherwise the 'active project' leaks into the
    file tools of the next suite (that leak is what tests/_harness.py now
    prevents for every other module).
    """

    def setUp(self):
        super().setUp()
        self._saved_store = ps.PROJECTS_FILE
        self.store_file = self.tmp / "code_projects.json"
        ps.PROJECTS_FILE = self.store_file
        self.addCleanup(setattr, ps, "PROJECTS_FILE", self._saved_store)
        self.root = self.tmp / "proj"
        self.root.mkdir()


# ── is_safe_path ────────────────────────────────────────────────────────

class IsSafePathTests(_TmpCase):
    def setUp(self):
        super().setUp()
        self.root = self.tmp / "root"
        self.root.mkdir()
        (self.root / "inside.txt").write_text("ok", encoding="utf-8")

    def test_a_file_inside_the_root_is_safe(self):
        self.assertTrue(fg.is_safe_path(str(self.root / "inside.txt"),
                                        str(self.root)))

    def test_the_root_itself_is_safe(self):
        self.assertTrue(fg.is_safe_path(str(self.root), str(self.root)))

    def test_dotdot_traversal_cannot_escape(self):
        target = str(self.root / ".." / ".." / "etc" / "passwd")
        self.assertFalse(fg.is_safe_path(target, str(self.root)))

    def test_an_absolute_path_outside_is_refused(self):
        self.assertFalse(fg.is_safe_path("/etc/passwd", str(self.root)))

    def test_a_sibling_directory_is_refused(self):
        sibling = self.tmp / "other"
        sibling.mkdir()
        (sibling / "x.txt").write_text("x", encoding="utf-8")
        self.assertFalse(fg.is_safe_path(str(sibling / "x.txt"),
                                         str(self.root)))

    def test_blank_arguments_are_unsafe(self):
        self.assertFalse(fg.is_safe_path("", str(self.root)))
        self.assertFalse(fg.is_safe_path(str(self.root / "inside.txt"), ""))
        self.assertFalse(fg.is_safe_path("", ""))

    def test_a_symlink_out_of_the_root_is_refused(self):
        outside = self.tmp / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        link = self.root / "link.txt"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable here")
        # resolve() follows the link before the comparison, so the escape
        # is seen even though the path text sits inside the root.
        self.assertFalse(fg.is_safe_path(str(link), str(self.root)))


# ── create_backup ───────────────────────────────────────────────────────

class CreateBackupTests(_TmpCase):
    def setUp(self):
        super().setUp()
        self.src = self.write("note.txt", "original")

    def test_backup_copies_the_file_into_the_backup_dir(self):
        dest = Path(fg.create_backup(str(self.src)))
        self.assertTrue(dest.is_file())
        self.assertEqual(dest.read_text(encoding="utf-8"), "original")
        self.assertEqual(dest.parent, fg._backup_dir())
        self.assertRegex(dest.name, r"^\d{8}-\d{6}_note\.txt(\.\d+)?$")

    def test_the_source_is_left_alone(self):
        fg.create_backup(str(self.src))
        self.assertEqual(self.src.read_text(encoding="utf-8"), "original")

    def test_a_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            fg.create_backup(str(self.tmp / "absent.txt"))

    def test_two_backups_in_one_second_do_not_collide(self):
        a = Path(fg.create_backup(str(self.src)))
        b = Path(fg.create_backup(str(self.src)))
        self.assertNotEqual(a, b, "the same second must not overwrite a copy")
        self.assertTrue(a.is_file() and b.is_file())

    def test_pruning_keeps_only_the_newest_backups(self):
        saved = fg.BACKUP_KEEP
        fg.BACKUP_KEEP = 2
        self.addCleanup(setattr, fg, "BACKUP_KEEP", saved)
        for i in range(5):
            self.src.write_text("v%d" % i, encoding="utf-8")
            fg.create_backup(str(self.src))
        left = [p for p in fg._backup_dir().iterdir() if p.is_file()]
        self.assertEqual(len(left), 2, "BACKUP_KEEP is a cap, not a target")


# ── the jail inside tools/file.py ───────────────────────────────────────

class FileToolsJailTests(_ProjectStoreCase):
    def setUp(self):
        super().setUp()
        self.inside = self.write(self.root / "a.txt", "hello-inside")
        self.outside = self.write(self.tmp / "outside.txt", "outside-data")

    def test_traversal_is_refused_even_with_no_project(self):
        # No active project → the jail root falls back to the repo root
        # (Release 16), but a raw `..` segment is refused before the root
        # comparison either way: it is never a real file path here.
        for out in (file_mod.read_file("../../etc/passwd"),
                    file_mod.list_files("../.."),
                    file_mod.write_file("../escape.txt", "x")):
            self.assertTrue(out.startswith("Error: unsafe path"), out)
        self.assertFalse((self.tmp / "escape.txt").exists())

    def test_with_a_project_reads_inside_the_root(self):
        proj = ps.add("proj", str(self.root))
        ps.set_active(proj["id"])
        self.assertIn("hello-inside",
                      file_mod.read_file(str(self.inside)))
        self.assertIn("a.txt", file_mod.list_files(str(self.root)))

    def test_with_a_project_refuses_to_read_outside(self):
        proj = ps.add("proj", str(self.root))
        ps.set_active(proj["id"])
        out = file_mod.read_file(str(self.outside))
        self.assertTrue(out.startswith("Error: unsafe path"), out)
        self.assertIn("outside allowed root", out)
        self.assertIn(str(self.root.resolve()), out)
        listed = file_mod.list_files(str(self.tmp))
        self.assertTrue(listed.startswith("Error: unsafe path"), listed)

    def test_with_a_project_refuses_to_write_outside(self):
        proj = ps.add("proj", str(self.root))
        ps.set_active(proj["id"])
        target = self.tmp / "planted.txt"
        out = file_mod.write_file(str(target), "evil")
        self.assertTrue(out.startswith("Error: unsafe path"), out)
        self.assertFalse(target.exists())

    def test_writes_inside_the_project_still_work(self):
        proj = ps.add("proj", str(self.root))
        ps.set_active(proj["id"])
        target = self.root / "sub" / "new.txt"
        out = file_mod.write_file(str(target), "made")
        self.assertTrue(out.startswith("Wrote "), out)
        self.assertEqual(target.read_text(encoding="utf-8"), "made")


# ── the always-on jail (Release 16) ─────────────────────────────────────

class AlwaysOnJailTests(_ProjectStoreCase):
    """No project selected: the jail root falls back to the repo root.

    Before Release 16 an empty project selection meant no jail at all —
    Agent Mode could list/read/write anywhere the process could reach.
    These cases run against the case's empty project store, so
    _jail_root() must be answering config.PROJECT_ROOT.
    """

    def setUp(self):
        super().setUp()
        self.repo_root = Path(config.PROJECT_ROOT)
        self.outside = self.write(self.tmp / "outside.txt", "outside-data")

    def inside_tmp(self):
        """A throwaway dir inside the repo root, cleaned up with the case."""
        d = Path(tempfile.mkdtemp(prefix="jail-inside-", dir=str(self.repo_root)))
        self.addCleanup(shutil.rmtree, str(d), True)
        return d

    def test_no_project_root_is_the_repo_root(self):
        self.assertEqual(file_mod._jail_root(), str(self.repo_root))

    def test_no_project_write_outside_is_refused_and_names_the_root(self):
        target = self.tmp / "planted.txt"
        out = file_mod.write_file(str(target), "evil")
        self.assertTrue(out.startswith("Error: unsafe path"), out)
        self.assertIn("outside allowed root", out)
        self.assertIn(str(self.repo_root), out)
        self.assertFalse(target.exists())

    def test_no_project_read_and_list_outside_are_refused(self):
        out = file_mod.read_file(str(self.outside))
        self.assertTrue(out.startswith("Error: unsafe path"), out)
        self.assertIn(str(self.repo_root), out)
        listed = file_mod.list_files(str(self.tmp))
        self.assertTrue(listed.startswith("Error: unsafe path"), listed)

    def test_no_project_paths_inside_the_repo_root_work(self):
        d = self.inside_tmp()
        target = d / "note.txt"
        out = file_mod.write_file(str(target), "inside-repo")
        self.assertTrue(out.startswith("Wrote "), out)
        self.assertIn("inside-repo", file_mod.read_file(str(target)))
        self.assertIn("note.txt", file_mod.list_files(str(d)))

    def test_no_project_symlink_escape_is_refused(self):
        d = self.inside_tmp()
        link = d / "link.txt"
        try:
            link.symlink_to(self.outside)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable here")
        # The link text sits inside the root; resolve() follows it out.
        out = file_mod.read_file(str(link))
        self.assertTrue(out.startswith("Error: unsafe path"), out)

    def test_opt_out_reopens_outside_paths(self):
        # AI_AGENT_FILE_JAIL=off lands in config.FILE_JAIL_ENABLED; the
        # tools read it per call (the seam test_flag_off_returns_floor
        # uses for AGENT_ADAPTIVE_READ in test_read_cap.py).
        saved = config.FILE_JAIL_ENABLED
        config.FILE_JAIL_ENABLED = False
        self.addCleanup(setattr, config, "FILE_JAIL_ENABLED", saved)
        target = self.tmp / "opted-out.txt"
        out = file_mod.write_file(str(target), "free")
        self.assertTrue(out.startswith("Wrote "), out)
        self.assertIn("outside-data", file_mod.read_file(str(self.outside)))

    def test_opt_out_still_refuses_traversal(self):
        # The `..` refusal is a separate standing rule, not part of the
        # jail the opt-out switches off.
        saved = config.FILE_JAIL_ENABLED
        config.FILE_JAIL_ENABLED = False
        self.addCleanup(setattr, config, "FILE_JAIL_ENABLED", saved)
        out = file_mod.write_file("../escape.txt", "x")
        self.assertTrue(out.startswith("Error: unsafe path"), out)


# ── the backup line in write_file ───────────────────────────────────────

class WriteBackupTests(_ProjectStoreCase):
    def setUp(self):
        super().setUp()
        # Release 16: the file tools are jailed even with no project
        # (root = this repo), so this case names its tmp tree as the
        # active project — the same standing choice a user makes in
        # Code mode — instead of relying on the old open door.
        proj = ps.add("proj", str(self.root))
        ps.set_active(proj["id"])

    def test_overwriting_takes_a_backup_of_the_old_contents(self):
        target = self.write(self.root / "cfg.txt", "old-contents")
        out = file_mod.write_file(str(target), "new-contents")
        self.assertTrue(out.startswith("Wrote "), out)
        self.assertIn("\nBackup saved at: ", out)
        self.assertEqual(target.read_text(encoding="utf-8"), "new-contents")

        backup = Path(out.split("Backup saved at: ", 1)[1].strip())
        self.assertTrue(backup.is_file(), "the backup path must exist")
        self.assertEqual(backup.read_text(encoding="utf-8"), "old-contents")
        self.assertEqual(backup.parent, fg._backup_dir())

    def test_a_brand_new_file_needs_no_backup(self):
        target = self.root / "fresh.txt"
        out = file_mod.write_file(str(target), "hello")
        self.assertTrue(out.startswith("Wrote "), out)
        self.assertNotIn("Backup saved at:", out)
        self.assertEqual(target.read_text(encoding="utf-8"), "hello")


# ── /api/files/tree and /api/files/content ──────────────────────────────

class FilesApiTests(_ProjectStoreCase):
    def setUp(self):
        super().setUp()
        self.hello = self.write(self.root / "hello.py", "print('hi')\n")
        (self.root / "sub").mkdir()
        (self.root / "sub" / "deep.md").write_text("# deep\n",
                                                   encoding="utf-8")
        # The first project becomes the active one, which is what the UI
        # panel sends no `project` parameter for.
        self.proj = ps.add("proj", str(self.root))
        ps.set_active(self.proj["id"])

    def get_json(self, url):
        with closing(client.get(url)) as r:
            return r.status_code, r.get_json()

    def test_tree_returns_folders_then_files_with_size_and_mtime(self):
        code, body = self.get_json("/api/files/tree")
        self.assertEqual(code, 200, body)
        self.assertTrue(body["ok"])
        self.assertEqual(body["project"]["id"], self.proj["id"])
        tree = body["tree"]
        self.assertEqual(tree["type"], "dir")
        self.assertEqual(tree["path"], "")
        kinds = [(c["type"], c["name"]) for c in tree["children"]]
        self.assertIn(("dir", "sub"), kinds)
        self.assertIn(("file", "hello.py"), kinds)
        # directories sort before files
        self.assertEqual(kinds[0], ("dir", "sub"))
        file_node = [c for c in tree["children"]
                     if c["name"] == "hello.py"][0]
        self.assertEqual(file_node["path"], "hello.py")
        self.assertEqual(file_node["size"], self.hello.stat().st_size)
        self.assertGreater(file_node["mtime"], 0)
        sub = [c for c in tree["children"] if c["type"] == "dir"][0]
        self.assertEqual([c["path"] for c in sub["children"]],
                         ["sub/deep.md"])

    def test_tree_accepts_an_explicit_project_id(self):
        code, body = self.get_json(
            "/api/files/tree?project=" + self.proj["id"])
        self.assertEqual(code, 200, body)
        self.assertEqual(body["project"]["path"], str(self.root.resolve()))

    def test_tree_unknown_project_is_404(self):
        code, body = self.get_json("/api/files/tree?project=nope")
        self.assertEqual(code, 404)
        self.assertIn("not found", body["error"])

    def test_tree_without_any_active_project_is_400(self):
        saved = ps.PROJECTS_FILE
        ps.PROJECTS_FILE = self.tmp / "empty-store.json"
        try:
            code, body = self.get_json("/api/files/tree")
        finally:
            ps.PROJECTS_FILE = saved
        self.assertEqual(code, 400)
        self.assertIn("no active project", body["error"])

    def test_content_returns_text_language_and_size(self):
        code, body = self.get_json("/api/files/content?path=hello.py")
        self.assertEqual(code, 200, body)
        self.assertTrue(body["ok"])
        self.assertEqual(body["path"], "hello.py")
        self.assertEqual(body["name"], "hello.py")
        self.assertEqual(body["language"], "python")
        self.assertEqual(body["content"], "print('hi')\n")
        self.assertEqual(body["size"], self.hello.stat().st_size)

    def test_content_reads_a_nested_relative_path(self):
        code, body = self.get_json("/api/files/content?path=sub/deep.md")
        self.assertEqual(code, 200, body)
        self.assertEqual(body["content"], "# deep\n")
        self.assertEqual(body["language"], "markdown")

    def test_content_requires_a_path(self):
        code, body = self.get_json("/api/files/content")
        self.assertEqual(code, 400)
        self.assertIn("path required", body["error"])

    def test_content_rejects_dotdot_traversal(self):
        code, body = self.get_json(
            "/api/files/content?path=../../etc/passwd")
        self.assertEqual(code, 403)
        self.assertIn("unsafe path", body["error"])

    def test_content_rejects_an_absolute_path_outside_the_project(self):
        code, body = self.get_json("/api/files/content?path=/etc/passwd")
        self.assertEqual(code, 403)
        self.assertIn("unsafe path", body["error"])

    def test_content_of_a_missing_file_is_404(self):
        code, body = self.get_json("/api/files/content?path=ghost.txt")
        self.assertEqual(code, 404)
        self.assertIn("not found", body["error"])

    def test_content_refuses_a_binary_file(self):
        self.write(self.root / "blob.bin", b"\x00\x01\x02binary")
        code, body = self.get_json("/api/files/content?path=blob.bin")
        self.assertEqual(code, 400)
        self.assertIn("binary file", body["error"])

    def test_content_refuses_a_file_over_1mb(self):
        big = self.root / "big.txt"
        big.write_bytes(b"a" * (1024 * 1024 + 1))
        code, body = self.get_json("/api/files/content?path=big.txt")
        self.assertEqual(code, 413)
        self.assertIn("file too large", body["error"])
        self.assertEqual(body["limit"], 1024 * 1024)

    def test_content_with_an_unknown_project_is_404(self):
        code, body = self.get_json(
            "/api/files/content?project=nope&path=hello.py")
        self.assertEqual(code, 404)
        self.assertIn("not found", body["error"])

    def test_content_survives_a_directory_path(self):
        code, body = self.get_json("/api/files/content?path=sub")
        self.assertEqual(code, 400)
        self.assertIn("not a file", body["error"])


if __name__ == "__main__":
    unittest.main()
