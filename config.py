"""Configuration management for ai-agent."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from core.log import get_logger

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = Path.home() / ".ai-agent"
CONFIG_FILE = CONFIG_DIR / "config.json"

# Assigned after CONFIG_DIR exists: core.log resolves the log directory from
# this module, and configuring the logger first would make it fall back to a
# guessed path for the rest of the process.
log = get_logger("config")

# Bundled model directory. A model dropped in here is picked up automatically,
# so the app no longer depends on an absolute path outside the project.
MODELS_DIR = PROJECT_ROOT / "models"

# Where models used to live. Kept as fallbacks so older installs still work;
# we search recursively, so the per-model hash folder doesn't matter.
LEGACY_MODEL_DIRS = [
    Path("/storage/emulated/0/Download/gemmamodel"),
    Path.home() / "storage" / "shared" / "Download" / "gemmamodel",
]

MODEL_SUFFIX = ".litertlm"

# The model is often kept *outside* the project so the repo stays small — e.g.
# alongside it as ../ai-agent-model/. Project root may sit one or two levels
# deep, so we walk up and pick up any sibling directory named like a model
# store. Bounded by SIBLING_WALK_LEVELS so we never scan the whole filesystem.
SIBLING_WALK_LEVELS = 3

# Context options (only these 3)
CONTEXT_OPTIONS = [8192, 12288, 16384]
CONTEXT_LABELS = {
    8192:  "8K  — RAM သက်သာ (agent ~5 turns)",
    12288: "12K — Balance (agent ~8 turns)",
    16384: "16K — Long context (agent ~11 turns)",
}

# Empty means "auto-discover": the newest *.litertlm found under models/
# (or a legacy location) is used. A stale saved path is repaired on load.
DEFAULT_MODEL = ""


def sibling_model_dirs() -> list[Path]:
    """Model directories sitting near PROJECT_ROOT, e.g. ../ai-agent-model/.

    Only child directories whose name looks like a model store ("model" in the
    name) are considered, and only a few levels up. Skips anything inside
    PROJECT_ROOT itself — MODELS_DIR already covers that.
    """
    found: list[Path] = []
    seen: set[Path] = set()
    node = PROJECT_ROOT
    for _ in range(SIBLING_WALK_LEVELS):
        node = node.parent
        try:
            children = sorted(node.iterdir())
        except OSError:
            continue
        for child in children:
            if child == PROJECT_ROOT or child in seen:
                continue
            if child.name.startswith("."):
                continue
            if "model" not in child.name.lower():
                continue
            try:
                if not child.is_dir():
                    continue
            except OSError:
                continue
            seen.add(child)
            found.append(child)
    return found


def model_search_dirs() -> list[Path]:
    """Directories scanned for a model, most-preferred first."""
    return [MODELS_DIR, *LEGACY_MODEL_DIRS, *sibling_model_dirs()]


def _litertlm_files(root: Path) -> list[Path]:
    """*.litertlm files under root, newest first. Empty list if root is gone."""
    if not root.is_dir():
        return []
    try:
        files = [p for p in root.rglob(f"*{MODEL_SUFFIX}") if p.is_file()]
    except OSError:
        return []
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def find_model_file() -> Path | None:
    """Locate a model file, or None if none of the search dirs has one."""
    for directory in model_search_dirs():
        found = _litertlm_files(directory)
        if found:
            return found[0]
    return None


def nearby_model_files(limit: int = 3) -> list[Path]:
    """Any *.litertlm reachable from a model dir, for diagnostics.

    Same scan as find_model_file, but returns every hit instead of the first,
    so a 'model not found' error can point at a file that *does* exist rather
    than just listing directories the user already knows are empty.
    """
    seen: set[Path] = set()
    hits: list[Path] = []
    for directory in model_search_dirs():
        for p in _litertlm_files(directory):
            if p not in seen:
                seen.add(p)
                hits.append(p)
            if len(hits) >= limit:
                return hits
    return hits


def model_missing_message(path: str = "") -> str:
    """Human-readable 'no model' error listing everywhere we looked."""
    searched = "\n    ".join(str(d) for d in model_search_dirs())
    msg = (
        f"Model not found: {path or '(unset)'}\n"
        f"  searched:\n    {searched}\n"
        f"  fix: copy a {MODEL_SUFFIX} file into {MODELS_DIR}"
    )
    nearby = nearby_model_files()
    if nearby:
        listing = "\n    ".join(str(p) for p in nearby)
        msg += (
            f"\n  a model does exist nearby — point config at it:"
            f"\n    {listing}\n"
            f"    (~/.ai-agent/config.json: \"model_path\": \"<path above>\")"
        )
    return msg


@dataclass
class Config:
    model_path: str = DEFAULT_MODEL
    backend: str = "GPU"
    context_size: int = 8192
    temperature: float = 0.7
    max_output_tokens: int = 1024

    def save(self) -> None:
        """Atomic save to config file."""
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            tmp = CONFIG_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))
            tmp.replace(CONFIG_FILE)
        except OSError as e:
            log.error("config save failed for %s: %s: %s",
                      CONFIG_FILE, type(e).__name__, e)
            raise

    @classmethod
    def load(cls) -> "Config":
        """Load config; fallback to defaults if missing/corrupt.

        A model path that no longer exists (model moved, or config predates
        auto-discovery) is repaired here and written back to disk.
        """
        data = None
        if CONFIG_FILE.exists():
            try:
                data = json.loads(CONFIG_FILE.read_text())
            except (json.JSONDecodeError, OSError) as e:
                log.warning("config unreadable (%s: %s) — using defaults",
                            type(e).__name__, e)
                data = None

        if data is None:
            cfg = cls()
        else:
            # Filter unknown keys
            known = set(cls.__dataclass_fields__)
            filtered = {k: v for k, v in data.items() if k in known}
            cfg = cls(**filtered)

        cfg.resolve_model_path(persist=True)
        if data is None:
            cfg.save()
        return cfg

    def resolve_model_path(self, *, persist: bool = False) -> str:
        """Return the model path to actually use.

        The configured path wins when it still exists. Otherwise we fall back
        to auto-discovery so a relocated model keeps working without the user
        editing config.json by hand. With `persist`, a repaired path is saved.
        """
        configured = (self.model_path or "").strip()
        if configured and Path(configured).is_file():
            return configured

        discovered = find_model_file()
        if discovered is None:
            return configured

        resolved = str(discovered)
        if persist and resolved != configured:
            self.model_path = resolved
            try:
                self.save()
            except OSError as e:
                # Return the discovered path anyway — only the write-back
                # failed, so the next run will rediscover it.
                log.warning("could not persist the repaired model path: %s: %s",
                            type(e).__name__, e)
        return resolved

    def validate(self) -> list[str]:
        """Return list of errors (empty = OK)."""
        errors: list[str] = []
        resolved = self.resolve_model_path()
        if not resolved or not Path(resolved).is_file():
            errors.append(model_missing_message(resolved))
        if self.context_size not in CONTEXT_OPTIONS:
            errors.append(
                f"context_size must be one of {CONTEXT_OPTIONS}, "
                f"got {self.context_size}"
            )
        if self.backend not in ("GPU", "CPU"):
            errors.append(f"backend must be GPU or CPU, got {self.backend}")
        return errors
