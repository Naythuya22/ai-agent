"""Path jail and pre-write backups for the file tools (P0).

Two jobs, both about making an agent edit contained and reversible:

  is_safe_path(target, root)  -- the one answer to "may this path be
      touched?". Path.resolve() runs first, so `../`, a symlink and a `~`
      segment all collapse to the same real path before is_relative_to()
      compares it: nothing escapes the root by spelling its way out.

  create_backup(path)         -- copy the current contents to
      <CONFIG_DIR>/backups/<timestamp>_<name> before write_file overwrites
      it, so a bad edit is one `cp` away from undone. The copy is made
      *before* the write and a failed copy stops the write.

Neither function logs, prompts or raises-and-hides: is_safe_path answers a
bool, create_backup raises so the caller has to decide.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

# Newest backups kept before the oldest are pruned. One copy per overwrite
# per second per name adds up slowly; 50 is weeks of agent work on a phone.
BACKUP_KEEP = 50

# Read as a module attribute on every prune so a test can lower it without
# rebuilding the world.
_BACKUP_SUBDIR = "backups"


def _backup_dir() -> Path:
    """Where backups land: <CONFIG_DIR>/backups, i.e. ~/.ai-agent/backups.

    Resolved per call, not at import: tests re-point config.CONFIG_DIR at a
    throwaway HOME after this module may already be imported, and a stale
    import-time path would write into the previous run's directory.
    """
    base = None
    try:
        import config
        base = getattr(config, "CONFIG_DIR", None)
    except Exception:
        base = None
    if not base:
        base = Path.home() / ".ai-agent"
    return Path(base) / _BACKUP_SUBDIR


def is_safe_path(target_path: str, project_root: str) -> bool:
    """True when target_path resolves inside project_root.

    Both sides are expanded and resolved (symlinks and `..` included) before
    the comparison, so a link planted inside the root cannot point the read
    (or the write) at /etc. A blank argument is unsafe, never "allow".
    """
    if not target_path or not project_root:
        return False
    try:
        root = Path(project_root).expanduser().resolve()
        target = Path(target_path).expanduser().resolve()
    except (OSError, RuntimeError, ValueError, TypeError):
        return False
    if target == root:
        return True
    try:
        return target.is_relative_to(root)
    except (AttributeError, ValueError, OSError):
        # AttributeError: python without is_relative_to (pre-3.9) — the
        # caller asked for a jail, so answer "no" rather than "yes".
        return False


def create_backup(file_path: str) -> str:
    """Copy file_path into the backup folder and return the copy's path.

    Raises FileNotFoundError when there is nothing to back up (write_file
    only calls this on a file that exists) and OSError when the copy fails —
    a failed backup must stop the overwrite, not be swallowed.
    """
    src = Path(file_path).expanduser()
    if not src.is_file():
        raise FileNotFoundError("no file to back up: %s" % file_path)
    dest_dir = _backup_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = dest_dir / ("%s_%s" % (stamp, src.name))
    n = 1
    while dest.exists():
        # Two writes inside one second must not clobber each other's copy.
        dest = dest_dir / ("%s_%s.%d" % (stamp, src.name, n))
        n += 1
    shutil.copy2(str(src), str(dest))
    _prune(dest_dir)
    return str(dest)


def _prune(dest_dir: Path, keep: int | None = None) -> None:
    """Leave only the newest `keep` backups. Never raises: pruning is a
    courtesy, a full disk must not fail the write it is protecting."""
    limit = BACKUP_KEEP if keep is None else keep
    entries = []
    try:
        candidates = list(dest_dir.iterdir())
    except OSError:
        return
    for path in candidates:
        try:
            if path.is_file():
                entries.append((path.stat().st_mtime, path))
        except OSError:
            continue
    if len(entries) <= limit:
        return
    entries.sort(key=lambda item: item[0], reverse=True)
    for _, old in entries[limit:]:
        try:
            old.unlink()
        except OSError:
            pass
