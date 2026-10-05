"""Tier-aware Agent Mode system prompts.

One prompt per *tier*, never per model: a hardcoded list of model names would
need editing every time a provider ships something new, so the tier is guessed
from substrings in the model name and every unrecognised cloud model lands on
the safe default (``cloud_fast`` — the shorter, blunter instruction set).

Precedence, first match wins:

    Settings (a forced tier)  →  AI_AGENT_PROMPT_TIER  →  the classifier

The tier is resolved **once per user turn** and frozen for that turn; see
``modes.agent.AgentMode._system_prompt``. Nothing here reads or writes the
network, and nothing here can be pointed at a file outside ``prompts/`` — the
tier is checked against a whitelist before it ever becomes part of a path.
"""
from __future__ import annotations

import os
from pathlib import Path

import config
from core.log import get_logger

log = get_logger("core.prompts")

# prompts/ sits next to config.py, not inside core/ — same rule as the models
# directory: one obvious place to drop a new prompt in.
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
BASE_NAME = "_base.txt"

# The only three prompt files that exist. A tier is checked against this tuple
# before it is used to build a filename, so a caller-supplied "../.." can never
# reach the filesystem.
TIERS = ("cloud_strong", "cloud_fast", "local")

# Substrings, not model names. "mini" catches gpt-4o-mini, gemini-*-flash-lite,
# llama-3.1-8b and every future "-mini" without anyone editing this file.
_FAST_PATTERNS = (
    "mini", "flash", "haiku", "small", "lite", "instant",
    "nano", "air", "8b", "7b", "3b", "4b", "o3-mini", "gpt-oss-20b-small",
)

_STRONG_PATTERNS = (
    "opus", "sonnet", "-pro", "-max", "-ultra", "-large",
    "reasoner", "gpt-4o", "gpt-4.1", "gpt-5",
    "gemini-2.5-pro", "claude-sonnet-4", "claude-3-5-sonnet",
    "o1", "o3", "turbo", "gemini-2.5-pro",
)

# Only used if even the deferred AGENT_SYSTEM import fails — which would mean
# modes.agent is mid-import. Never silently empty: a turn with no system prompt
# at all is worse than one with a plain instruction.
_EMERGENCY_SYSTEM = (
    "You are a helpful assistant. Answer honestly and never invent facts."
)

# tier -> assembled prompt. Populated on first use; cleared by clear_cache().
_CACHE: dict[str, str] = {}


def estimate_tokens(text: str) -> int:
    """Rough token count for a prompt file — UTF-8 bytes / 3, rounded up.

    Deliberately pessimistic for English (a real tokenizer manages ~4 chars per
    token) and about right for Burmese, where one character is usually one
    token. A budget check that errs low is a budget check worth having.
    """
    raw = len(str(text or "").encode("utf-8"))
    return -(-raw // 3)


def _fallback_prompt() -> str:
    """AGENT_SYSTEM — the prompt that worked before tiers existed.

    Imported lazily because modes.agent imports this module.
    """
    try:
        from modes.agent import AGENT_SYSTEM
        return AGENT_SYSTEM
    except Exception as e:
        log.error("AGENT_SYSTEM unreachable, using the emergency prompt: %s",
                  type(e).__name__)
        return _EMERGENCY_SYSTEM



def _has_token(name: str, pattern: str) -> bool:
    """True if pattern appears as a bounded token in name.

    Boundaries: start/end of string, or one of -_.
    Prevents "mini" from matching "gemini".
    """
    i = name.find(pattern)
    while i != -1:
        before_ok = i == 0 or name[i - 1] in "-_."
        after = i + len(pattern)
        after_ok = after == len(name) or name[after] in "-_."
        if before_ok and after_ok:
            return True
        i = name.find(pattern, i + 1)
    return False


def classify_tier(backend: str, model: str) -> str:
    """Which prompt tier suits this backend/model pair.

    ``local`` is decided by the backend alone. For cloud, fast patterns
    are checked first to avoid matching a strong substring in a fast
    name (e.g. "gpt-4o-mini" contains "gpt-4o").
    """
    if (backend or "").strip().lower() == "local":
        return "local"
    m = (model or "").strip().lower()
    # Prevent false-positives: don't let short tokens like "air" or "lite"
    # inside "gemini-2.5-pro" or other names flip to fast. Require a
    # separator or explicit suffix for a few potentially ambiguous tokens.
    if any(_has_token(m, p) for p in _FAST_PATTERNS):
        return "cloud_fast"
    if any(_has_token(m, p) for p in _STRONG_PATTERNS):
        return "cloud_strong"
    return "cloud_fast"  # unknown cloud → the safe default


def _read(path: Path) -> str | None:
    """File text, or None when it is missing, unreadable or empty."""
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except OSError as e:
        log.warning("prompt file unreadable: %s: %s", path.name,
                    type(e).__name__)
        return None
    return text or None


def load_prompt(tier: str) -> str:
    """The system prompt for ``tier``: `_base.txt` + `agent_<tier>.txt`.

    Unknown tier, missing file or empty file → ``AGENT_SYSTEM`` with a warning,
    so a half-deployed prompts/ folder degrades to the old behaviour instead of
    to no prompt at all.
    """
    tier = (tier or "").strip()
    if tier not in TIERS:
        log.warning("unknown prompt tier %r — using AGENT_SYSTEM", tier)
        return _fallback_prompt()
    cached = _CACHE.get(tier)
    if cached is not None:
        return cached

    base = _read(PROMPTS_DIR / BASE_NAME)
    body = _read(PROMPTS_DIR / ("agent_%s.txt" % tier))
    if base is None or body is None:
        missing = BASE_NAME if base is None else ("agent_%s.txt" % tier)
        log.warning("prompt file %s missing or empty — using AGENT_SYSTEM",
                    missing)
        return _fallback_prompt()

    out = base + "\n\n" + body
    # Plain dict assignment: two threads racing here both read the same bytes
    # and both write the same value, which is why this needs no lock.
    _CACHE[tier] = out
    return out


def clear_cache() -> None:
    """Forget every assembled prompt — for tests and for a reload."""
    _CACHE.clear()


def resolve_tier(backend: str, model: str, *, mode: str = "auto") -> tuple[str, str]:
    """``(tier, source)`` for one turn.

    ``mode`` is the *effective* mode — what :func:`config.prompt_tier_mode`
    already decided by looking at the Settings store first and the environment
    second. ``source`` then says which of the two actually asked for it, so a
    tier pinned in Settings is never reported as an environment override.
    """
    mode = (mode or "auto").strip()
    if mode not in config.PROMPT_TIER_MODES:
        if mode:
            log.warning("prompt tier mode %r is not a tier — falling back to auto",
                        mode)
        mode = config.PROMPT_TIER_MODE_DEFAULT
    if mode in TIERS:
        if config.stored_prompt_tier_mode() == mode:
            return mode, "settings"
        if (os.environ.get(config.PROMPT_TIER_ENV) or "").strip() == mode:
            return mode, "env"
        return mode, "settings"
    return classify_tier(backend, model), "auto"


def local_model_name() -> str:
    """The on-device model's display name, or '' when it cannot be read."""
    try:
        from core.agent_backend import friendly_local_name
        return friendly_local_name() or ""
    except Exception as e:
        log.debug("local model name unreadable: %s", type(e).__name__)
        return ""


def plan_target(backend) -> tuple[str, str]:
    """``(backend, model)`` the next Agent turn would use, from config only.

    Read-only: it asks the backend object what it *would* do (mode, whether the
    cloud is usable, which provider comes first) and never changes it.
    """
    mode = ""
    try:
        mode = backend.mode
    except Exception as e:
        log.debug("backend mode unreadable for prompt tier: %s", type(e).__name__)
    if mode == "local":
        return "local", local_model_name()
    use_cloud = mode == "cloud"
    if not use_cloud:
        try:
            use_cloud = bool(backend.should_use_cloud())
        except Exception as e:
            log.debug("should_use_cloud failed for prompt tier: %s",
                      type(e).__name__)
            use_cloud = False
    if not use_cloud:
        return "local", local_model_name()
    model = ""
    try:
        model = (backend.planned_context() or {}).get("model") or ""
    except Exception as e:
        log.debug("planned_context failed for prompt tier: %s", type(e).__name__)
    return "cloud", str(model)


def describe(backend: str, model: str, mode: str | None = None) -> dict:
    """What the Settings page shows: the mode, the resolved tier and its source."""
    m = (mode or "").strip()
    if m not in config.PROMPT_TIER_MODES:
        m = config.prompt_tier_mode()
    tier, source = resolve_tier(backend, model, mode=m)
    return {"mode": m, "resolved": tier, "source": source, "model": model}