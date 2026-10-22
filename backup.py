"""Backup / restore for ai-agent user data.

The code in this repo is replaceable; the data is not. Memory, sessions,
provider keys and skills accumulate over months under ``~/.ai-agent`` (plus
``data/agent_config.json`` beside the project, and the code-mode session
stores on ``/sdcard``). This script packs all of it into one zip with a
checksummed manifest, and restores it again — taking a safety snapshot of
whatever is currently in place before it overwrites anything.

    python backup.py export [--out PATH] [--no-secrets]
    python backup.py list  <backup.zip>
    python backup.py import <backup.zip> [--yes]

``--no-secrets`` leaves out the files that carry credentials
(``agent_config.json`` with the provider keys, and the server's
``auth_token``). A backup that includes them is written 0600 and must be
treated like the keys themselves. Model weights (``*.litertlm``) are never
included — they are large and re-downloadable; the manifest notes the
configured model path instead.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

FORMAT_VERSION = 1
SECRET_NAMES = {"agent_config.json", "auth_token"}
SKIP_DIRS = {"logs", "__pycache__"}
SDCARD_FILES = ("code_sessions.json", "code_sessions_openclaude.json",
                "opencode.json")


class BackupError(Exception):
    """A backup file that must not be trusted / cannot be applied."""


# ── roots ────────────────────────────────────────────────────────────

def default_roots() -> dict[str, Path]:
    """Live locations, read at call time so tests can substitute their own."""
    import config
    return {
        "home": Path.home() / ".ai-agent",
        "data": Path(config.DATA_DIR),
        "sdcard": Path("/sdcard/.ai-agent"),
    }


# ── export ───────────────────────────────────────────────────────────

def _iter_files(roots: dict[str, Path]):
    """Yield (archive-name, disk-path) for every file a backup covers."""
    home = roots["home"]
    if home.is_dir():
        for p in sorted(home.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(home)
            if rel.parts and rel.parts[0] in SKIP_DIRS:
                continue
            if p.suffix in (".pyc", ".litertlm"):
                continue
            yield f"home/{rel.as_posix()}", p
    agent_cfg = roots["data"] / "agent_config.json"
    if agent_cfg.is_file():
        yield "project-data/agent_config.json", agent_cfg
    sdcard = roots["sdcard"]
    if sdcard.is_dir():
        for name in SDCARD_FILES:
            p = sdcard / name
            if p.is_file():
                yield f"sdcard/{name}", p


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def do_export(out_path: Path | None = None, *, include_secrets: bool = True,
              roots: dict[str, Path] | None = None) -> Path:
    roots = roots or default_roots()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = Path(out_path) if out_path else \
        Path.home() / f"ai-agent-backup-{stamp}.zip"

    entries = []
    payloads = []
    for arc, disk in _iter_files(roots):
        if not include_secrets and disk.name in SECRET_NAMES:
            continue
        data = disk.read_bytes()
        entries.append({"entry": arc, "size": len(data),
                        "sha256": _sha256(data)})
        payloads.append((arc, data))

    manifest = {
        "format": FORMAT_VERSION,
        "app": "ai-agent",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "includes_secrets": include_secrets and any(
            Path(e["entry"]).name in SECRET_NAMES for e in entries),
        "files": entries,
    }
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
        for arc, data in payloads:
            z.writestr(arc, data)
    tmp.replace(out_path)
    if manifest["includes_secrets"]:
        out_path.chmod(0o600)
    return out_path


# ── automatic backup ─────────────────────────────────────────────

AUTO_BACKUP_PREFIX = "ai-agent-backup-"
AUTO_BACKUP_MAX_AGE_DAYS = 7
AUTO_BACKUP_KEEP = 4


def _auto_backup_files(home_dir: Path) -> list[Path]:
    """Backup zips in *home_dir*, newest first.

    Only the ``ai-agent-backup-*.zip`` names :func:`do_export` writes by
    default are managed here — pre-restore snapshots
    (``ai-agent-pre-restore-*.zip``) are safety copies, never pruned.
    """
    if not home_dir.is_dir():
        return []
    files = [p for p in home_dir.glob(AUTO_BACKUP_PREFIX + "*.zip")
             if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def maybe_auto_backup(now: datetime | None = None, *,
                      home_dir: Path | None = None,
                      roots: dict[str, Path] | None = None,
                      max_age_days: int = AUTO_BACKUP_MAX_AGE_DAYS,
                      keep: int = AUTO_BACKUP_KEEP) -> Path | None:
    """Write a fresh backup when the newest one is *max_age_days* old.

    Returns the new backup's path, or ``None`` when the newest existing
    backup is still fresh. After writing, backups beyond the newest
    *keep* are deleted. Raises on failure — the caller (server startup)
    treats a failed backup as a warning, never a reason not to start.
    """
    now = now or datetime.now()
    home_dir = Path(home_dir) if home_dir else Path.home()
    existing = _auto_backup_files(home_dir)
    if existing:
        newest_age = now - datetime.fromtimestamp(
            existing[0].stat().st_mtime)
        if newest_age < timedelta(days=max_age_days):
            return None
    stamp = now.strftime("%Y%m%d-%H%M%S")
    out = do_export(home_dir / f"{AUTO_BACKUP_PREFIX}{stamp}.zip",
                    include_secrets=True, roots=roots)
    for stale in _auto_backup_files(home_dir)[keep:]:
        stale.unlink()
    return out


# ── list / import ────────────────────────────────────────────────────

def read_manifest(backup_path: Path) -> dict:
    try:
        with zipfile.ZipFile(backup_path) as z:
            raw = z.read("manifest.json")
    except (zipfile.BadZipFile, KeyError, OSError) as e:
        raise BackupError(f"not a readable ai-agent backup: {e}") from e
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as e:
        raise BackupError(f"corrupt manifest: {e}") from e
    if manifest.get("format") != FORMAT_VERSION or \
            manifest.get("app") != "ai-agent":
        raise BackupError("unsupported backup format")
    return manifest


def _target_for(entry: str, roots: dict[str, Path]) -> Path:
    """Map an archive name back to its disk location, refusing escapes."""
    parts = Path(entry).parts
    if not parts or any(p in ("..", "") for p in parts):
        raise BackupError(f"unsafe entry in backup: {entry!r}")
    head, rest = parts[0], parts[1:]
    bases = {"home": roots["home"], "project-data": roots["data"],
             "sdcard": roots["sdcard"]}
    if head not in bases or not rest:
        raise BackupError(f"unknown entry root in backup: {entry!r}")
    target = bases[head].joinpath(*rest)
    if bases[head].resolve() not in target.resolve().parents:
        raise BackupError(f"entry escapes its root: {entry!r}")
    return target


def do_import(backup_path: Path, *, roots: dict[str, Path] | None = None,
              snapshot_dir: Path | None = None) -> dict:
    roots = roots or default_roots()
    manifest = read_manifest(backup_path)

    # Verify everything before touching a single file on disk.
    with zipfile.ZipFile(backup_path) as z:
        names = set(z.namelist())
        staged = []
        for item in manifest["files"]:
            entry = item["entry"]
            if entry not in names:
                raise BackupError(f"backup is missing {entry}")
            data = z.read(entry)
            if len(data) != item["size"] or _sha256(data) != item["sha256"]:
                raise BackupError(f"checksum mismatch for {entry} — "
                                    "the backup is damaged, not restoring")
            staged.append((_target_for(entry, roots), data))

    # Never overwrite the only copy: snapshot the current state first.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    snap_dir = Path(snapshot_dir) if snapshot_dir else Path.home()
    snapshot = do_export(snap_dir / f"ai-agent-pre-restore-{stamp}.zip",
                         include_secrets=True, roots=roots)

    for target, data in staged:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(target)

    return {"restored": len(staged), "snapshot": str(snapshot),
            "created_at": manifest.get("created_at", "?"),
            "includes_secrets": bool(manifest.get("includes_secrets"))}


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ai-agent data backup / restore")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_exp = sub.add_parser("export", help="write a backup zip")
    p_exp.add_argument("--out", type=Path, default=None)
    p_exp.add_argument("--no-secrets", action="store_true",
                       help="exclude provider keys and the server token")

    p_list = sub.add_parser("list", help="show what a backup contains")
    p_list.add_argument("file", type=Path)

    p_imp = sub.add_parser("import", help="restore a backup zip")
    p_imp.add_argument("file", type=Path)
    p_imp.add_argument("--yes", action="store_true",
                       help="do not ask for confirmation")

    args = ap.parse_args(argv)
    try:
        if args.cmd == "export":
            out = do_export(args.out, include_secrets=not args.no_secrets)
            print(f"backup written: {out}")
            manifest = read_manifest(out)
            print(f"  {len(manifest['files'])} files, "
                  f"created {manifest['created_at']}")
            if manifest["includes_secrets"]:
                print("  ⚠ contains provider keys + server token — "
                      "keep this file private")
            return 0
        if args.cmd == "list":
            manifest = read_manifest(args.file)
            print(f"ai-agent backup, format {manifest['format']}, "
                  f"created {manifest.get('created_at', '?')}, "
                  f"secrets: {'yes' if manifest['includes_secrets'] else 'no'}")
            for item in manifest["files"]:
                print(f"  {item['entry']}  ({item['size']} bytes)")
            return 0
        if args.cmd == "import":
            if not args.yes:
                print("Stop the server first — restoring under a running "
                      "server can lose the restored state.")
                answer = input(f"Restore {args.file}? Current data is "
                               "snapshotted first. [y/N] ")
                if answer.strip().lower() != "y":
                    print("aborted")
                    return 1
            result = do_import(args.file)
            print(f"restored {result['restored']} files "
                  f"(backup from {result['created_at']})")
            print(f"previous state snapshot: {result['snapshot']}")
            return 0
    except BackupError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
