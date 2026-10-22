"""Configuration management for ai-agent."""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from core.log import get_logger

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = Path.home() / ".ai-agent"
CONFIG_FILE = CONFIG_DIR / "config.json"

# Fixed, HOME-independent data directory — Agent Mode's API keys live here.
#
# The phone runs the server with HOME=/data/data/com.termux/files/home while
# the shell has HOME=/root, so anything derived from Path.home() is written to
# two different places and the keys "disappear" after a restart. PROJECT_ROOT
# is absolute and identical for both, so this path never moves.
#
# AI_AGENT_DATA_DIR overrides it (the test harness points it at a throwaway
# directory). DEFAULT_DATA_DIR stays fixed on purpose: it is the only location
# the one-time legacy migration below will read from.
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR = (Path(os.environ["AI_AGENT_DATA_DIR"])
            if os.environ.get("AI_AGENT_DATA_DIR") else DEFAULT_DATA_DIR)
AGENT_CONFIG_FILE = DATA_DIR / "agent_config.json"

# HOME dirs the key file may already sit in from before this fix, most
# specific first. Both the Termux server HOME and the root shell HOME are
# listed — that mismatch is the whole reason the path became absolute.
LEGACY_HOMES = (
    "/data/data/com.termux/files/home",
    "/root",
    os.environ.get("HOME") or "",
    str(Path.home()),
)


def legacy_config_dirs() -> list[Path]:
    """``<home>/.ai-agent`` for every HOME this install may have run under."""
    seen: set[str] = set()
    out: list[Path] = []
    for home in LEGACY_HOMES:
        if not home:
            continue
        try:
            d = Path(home) / ".ai-agent"
        except (OSError, ValueError):
            continue
        if str(d) not in seen:
            seen.add(str(d))
            out.append(d)
    return out


def migrate_legacy_file(name: str, dest: Path) -> Path | None:
    """Copy ``name`` out of a legacy HOME dir into ``dest`` (one-time).

    Returns the file it was copied from, or None. Only runs while ``dest`` is
    missing and only for the real fixed data dir — a harness that retargeted
    DATA_DIR must never pull a developer's live keys into a test run.
    """
    try:
        if dest.exists() or Path(dest).parent != DEFAULT_DATA_DIR:
            return None
    except (OSError, ValueError):
        return None
    for src_dir in legacy_config_dirs():
        src = src_dir / name
        try:
            if not src.is_file():
                continue
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(src), str(dest))
            try:
                os.chmod(str(dest), 0o600)
            except OSError:
                pass        # storage that ignores unix modes still keeps the data
            return src
        except OSError:
            continue
    return None


# ── private keys location (v1.9.0) ─────────────────────────────────
#
# The keys used to live only in the project/data file above. On the
# phone the project sits on shared storage, where any app holding the
# storage permission can read them — mode bits do not apply there.
# The canonical home is now the app-private .ai-agent dir. On the
# phone that is the Termux files dir, named outright because it is a
# device fact, identical under both of the phone's HOMEs; elsewhere
# it is CONFIG_DIR, beside config.json / memory / sessions.
TERMUX_HOME = Path("/data/data/com.termux/files/home")


def private_keys_file() -> Path:
    """Canonical private location of agent_config.json."""
    if TERMUX_HOME.is_dir():
        return TERMUX_HOME / ".ai-agent" / "agent_config.json"
    return Path(CONFIG_DIR) / "agent_config.json"


def keys_file() -> Path:
    """The agent_config.json to read and write right now.

    Only the real fixed install resolves privately; a retargeted
    AGENT_CONFIG_FILE (test harness, tools) is returned untouched.
    Read precedence: the project file while it exists — it is the
    only copy pre-v1.9.0 code ever wrote, so it is the live one
    until the move completes — then the private file, then a copy
    in any known legacy HOME. A fresh install is born private.
    """
    project = Path(AGENT_CONFIG_FILE)
    if project.parent != DEFAULT_DATA_DIR:
        return project
    if project.exists():
        return project
    private = private_keys_file()
    if private.exists():
        return private
    for d in legacy_config_dirs():
        cand = d / "agent_config.json"
        if cand.is_file():
            return cand
    return private


def migrate_keys_to_private() -> Path | None:
    """Move the keys from shared storage into private_keys_file().

    Copy → byte-verify → delete the shared original; any failure
    leaves the original in place and returns None, so the caller
    keeps reading it where it was. Returns the path moved FROM.
    Only runs for the real fixed data dir, like migrate_legacy_file.

    A file already sitting at the private path is a leftover from
    the legacy-HOME era, never the live copy (nothing wrote there
    while the project file existed). It is never destroyed: an
    identical one is simply retired with the project file, a
    differing one is renamed to ``agent_config.json.superseded-*``
    before the live copy takes its place.
    """
    project = Path(AGENT_CONFIG_FILE)
    private = private_keys_file()
    try:
        if project.parent != DEFAULT_DATA_DIR:
            return None
        if not project.is_file():
            # Keys may still sit in a pre-absolute-path HOME dir:
            # pull them into the project file first, then move on.
            migrate_legacy_file(project.name, project)
            if not project.is_file():
                return None
        src = project.read_bytes()
        if private.is_file() and private.read_bytes() == src:
            project.unlink()
            return project
        private.parent.mkdir(parents=True, exist_ok=True)
        if private.is_file():
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            private.replace(private.with_name(
                "agent_config.json.superseded-" + stamp))
        tmp = private.with_name(private.name + ".tmp")
        shutil.copyfile(str(project), str(tmp))
        if tmp.read_bytes() != src:
            tmp.unlink(missing_ok=True)
            return None
        os.replace(str(tmp), str(private))
        try:
            os.chmod(str(private), 0o600)
        except OSError:
            pass        # the data matters more than the mode bit
        project.unlink()
        return project
    except OSError:
        return None


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


# ── output budget and tool-loop cap ───────────────────────────────────────
#
# Both numbers used to be hardcoded where they were used — 1024 output tokens,
# 8 tool iterations — so an online Agent turn was capped by the source and
# nothing else could raise it. They live here now, and two of them can be
# moved from the environment without touching a file: AI_AGENT_MAX_TOKENS and
# AI_AGENT_MAX_ITER.
#
# Module constants rather than Config fields on purpose: Config.context_size
# is the local engine's KV cache, and a cloud request's output budget has
# nothing to do with it. A malformed or out-of-range environment value falls
# back to the default rather than raising — a typo in a shell profile must not
# take the agent down.

# Nothing sent to a provider may ask for more than this, whatever a client
# asked for. It is the ceiling on the cost of a single request.
MAX_TOKENS_CEILING = 65_536


def _env_int(name: str, default: int, *, minimum: int = 1,
             maximum: int | None = None) -> int:
    """An integer from the environment, or ``default`` when it is not sane."""
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        n = int(str(raw).strip())
    except (TypeError, ValueError):
        log.warning("%s=%r is not a number — using %d", name, raw, default)
        return default
    if n < minimum or (maximum is not None and n > maximum):
        log.warning("%s=%d is outside [%d, %s] — using %d", name, n, minimum,
                    maximum if maximum is not None else "any", default)
        return default
    return n


def _env_bool(name: str, default: bool) -> bool:
    """A boolean from the environment — ``1/true/yes/on``, else ``default``.

    Deliberately conservative: an unrecognised value is not "off", it is a
    typo, and a typo in a shell profile must not quietly switch off something
    the user believes is on. It warns and keeps the default instead.
    """
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return bool(default)
    text = str(raw).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    log.warning("%s=%r is not a boolean — using %s", name, raw,
                "True" if default else "False")
    return bool(default)


# Output tokens per request.
AGENT_MAX_TOKENS_DEFAULT = _env_int("AI_AGENT_MAX_TOKENS", 8_192,
                                    maximum=MAX_TOKENS_CEILING)
CHAT_MAX_TOKENS_DEFAULT = 2_048      # the OpenAI-compatible route
CODE_MAX_TOKENS_DEFAULT = 16_384     # a code turn wants whole files back

# Which Code-mode backend a fresh run starts on. A project can override it
# (projects.backend), the UI can switch it (/model, POST /api/code/backend),
# and code_backends.default_name() falls back to "opencode" if this names
# something that is not registered — an env var must never break code mode.
CODE_BACKEND_DEFAULT = os.environ.get("AI_AGENT_CODE_BACKEND", "opencode")
# The key-validation probe: one cheap round-trip that proves a key works.
# Not environment-overridable on purpose — this is the one number that must
# never be inflated by accident, and 8 tokens is enough for the "ok" it asks.
PROBE_MAX_TOKENS = 8

# Tool-loop iterations per user turn. The loop also exits the moment the model
# asks for no tools, and again on a repeated identical call, so this is a
# backstop against a model that calls tools forever — not the normal exit.
AGENT_MAX_ITER_DEFAULT = _env_int("AI_AGENT_MAX_ITER", 30, maximum=100)
AGENT_MAX_ITER_CODE = 50


# ── Agent Mode robustness (each one reversible from the shell) ─────────────
#
# Four behaviours that change what the model is shown, all on by default and
# all individually switchable. To get the old behaviour back, set the variable
# to 0 / false / no / off:
#
#   AI_AGENT_TOOL_ERRORS=0   a search backend that fails is silent again
#   AI_AGENT_VERIFY=0        no reviewer round-trip after a tool turn
#   AI_AGENT_CACHE=0         no reuse of a tool result within one turn
#   AI_AGENT_INJECTION=0     no untrusted banner on scanned tool output
#
# A typo is not read as "off" — _env_bool warns and keeps the default, so a
# broken shell profile cannot quietly disable a guard.

# Task 1 — a tool that fails internally says so, instead of leaving the model
# to invent an answer from an empty result set.
TOOL_ERRORS_VISIBLE = _env_bool("AI_AGENT_TOOL_ERRORS", True)

# Task 2 — one cheap reviewer pass over the tool results before the answer is
# returned. Cloud only: the local 4B shares a single conversation object, and a
# review turn there would cost phone tokens and thermal headroom for nothing.
AGENT_VERIFY_ENABLED = _env_bool("AI_AGENT_VERIFY", True)
AGENT_VERIFY_MAX_TOKENS = _env_int("AI_AGENT_VERIFY_TOKENS", 512, maximum=1024)

# Task 3 — reuse a read-only tool result inside one user turn.
AGENT_TURN_CACHE = _env_bool("AI_AGENT_CACHE", True)

# Task 4 — scan tool output for instruction-shaped text and mark it untrusted.
AGENT_INJECTION_GUARD = _env_bool("AI_AGENT_INJECTION", True)

# Dev only — /sw.js re-reads web/index.html on every request instead of
# serving the hash computed at startup, so a frontend edit reaches the
# browser without a server restart (Lesson #3). Off by default.
DEBUG = _env_bool("AI_AGENT_DEBUG", False)

# ── read_file budget (Batch 1.5) ──────────────────────────────
# Model-aware char cap. Floor 4K protects small local models, ceiling
# 100K protects cost, the 25%-of-remaining rule keeps 3-4 reads alive.
# Revert with AI_AGENT_ADAPTIVE_READ=0.
AGENT_ADAPTIVE_READ = _env_bool("AI_AGENT_ADAPTIVE_READ", True)
AGENT_READ_FLOOR    = 4_000
AGENT_READ_CEILING  = _env_int("AI_AGENT_READ_CEILING",
                               100_000, maximum=500_000)
AGENT_READ_RESERVE  = 1_000


# ── budget awareness (Agent Mode) ────────────────────────────────
AGENT_BUDGET_HINT = _env_bool("AI_AGENT_BUDGET_HINT", True)
AGENT_BUDGET_WARN_CTX = _env_int("AI_AGENT_BUDGET_WARN_CTX",
                                 80, maximum=100)
AGENT_BUDGET_WARN_ITER = 5      # remaining calls threshold

# ── cloud token streaming (Batch 2) ──────────────────────────────
# Cloud replies arrive as SSE deltas instead of one whole message. The
# non-streaming routes are unchanged either way.
# Revert with AI_AGENT_STREAM=0.
AGENT_CLOUD_STREAM = _env_bool("AI_AGENT_STREAM", True)

# ── parallel tool execution (Batch 4.2) ──────────────────────
# Only used when a single model response carries >1 tool_call and
# none of the tools are in SERIAL_ONLY_TOOLS (write_file,
# run_python). Read-only tools are safe to run concurrently.
# Revert with AI_AGENT_PARALLEL=0.
AGENT_PARALLEL_TOOLS = _env_bool("AI_AGENT_PARALLEL", True)

# ── reflection loop (Batch 3.2) ──────────────────────────────
# After each turn, ask the model whether a lesson is worth saving
# to memory. Only durable lessons get saved (never one-time facts).
# Revert with AI_AGENT_REFLECT=0.
AGENT_REFLECT_ENABLED = _env_bool("AI_AGENT_REFLECT", True)
AGENT_REFLECT_MAX_TOKENS = _env_int(
    "AI_AGENT_REFLECT_TOKENS", 256, maximum=512)


# ── prompt tier ────────────────────────────────────────────────────────────
#
# "auto" (the default) lets core.prompts.classify_tier() pick the system prompt
# from the backend and the model name. The three tier names pin it instead —
# useful when a provider's model name says nothing useful about its size, or
# when a local phone answers better with the short on-device prompt.
#
# Two places can set the mode and they are not equal: the Settings store wins,
# and AI_AGENT_PROMPT_TIER is the fallback for a shell that has not been given
# a Settings page yet. The tier is resolved once per turn — see
# core.prompts.resolve_tier.

PROMPT_TIER_MODE_DEFAULT = "auto"
PROMPT_TIER_MODES = ("auto", "cloud_strong", "cloud_fast", "local")
PROMPT_TIER_ENV = "AI_AGENT_PROMPT_TIER"


def stored_prompt_tier_mode() -> str:
    """The tier pinned in the Settings store, or '' when none is set.

    Separate from prompt_tier_mode() so the resolver can tell a Settings choice
    from the environment one and report the right source.
    """
    try:
        from core.key_manager import KeyManager
        stored = (KeyManager().load() or {}).get("prompt_tier_mode")
    except Exception as e:
        log.warning("prompt tier in settings unreadable: %s", type(e).__name__)
        return ""
    stored = str(stored or "").strip()
    return stored if stored in PROMPT_TIER_MODES else ""


def prompt_tier_mode() -> str:
    """The effective tier mode: Settings store, then environment, then auto."""
    stored = stored_prompt_tier_mode()
    if stored:
        return stored
    env = str(os.environ.get(PROMPT_TIER_ENV) or "").strip()
    if env in PROMPT_TIER_MODES:
        return env
    return PROMPT_TIER_MODE_DEFAULT


def agent_max_tokens(mode: str = "agent") -> int:
    """Per-request output budget for ``mode``, never above the ceiling."""
    base = (CODE_MAX_TOKENS_DEFAULT if (mode or "") == "code"
            else AGENT_MAX_TOKENS_DEFAULT)
    return max(1, min(int(base), MAX_TOKENS_CEILING))


def agent_max_iter(mode: str = "agent") -> int:
    """Tool-loop iteration cap for ``mode``."""
    if (mode or "") == "code":
        return AGENT_MAX_ITER_CODE
    return AGENT_MAX_ITER_DEFAULT


def clamp_max_tokens(value, default: int) -> int:
    """A caller's ``max_tokens``: honoured, but bounded by the ceiling.

    ``None``, a non-number or a nonsense value falls back to ``default``, so a
    bad request body can neither ask for no tokens at all nor for a million.
    """
    try:
        n = int(value)
    except (TypeError, ValueError):
        return int(default)
    if n < 1:
        return int(default)
    return min(n, MAX_TOKENS_CEILING)


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
            if not configured:
                return ""
            # The configured path is stale and nothing can be discovered to
            # replace it. Report "unset" instead of pointing the loader at a
            # file that is not there (the missing-model message then lists
            # the searched dirs instead of blaming a moved file), and with
            # `persist` clear it on disk so the dead path stops shadowing
            # auto-discovery on every later load.
            if persist:
                self.model_path = ""
                try:
                    self.save()
                except OSError as e:
                    log.warning(
                        "could not persist the cleared model path: %s: %s",
                        type(e).__name__, e)
            return ""

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
