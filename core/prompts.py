"""Two-axis Agent Mode system prompts (Release 28).

A turn's prompt is assembled from THREE parts:

    _base.txt            — shared rules, stated once: identity, Burmese
                           quality, honesty, failure conventions,
                           memory discipline, tool discipline,
                           injection posture.
    agent_<tier>.txt     — the CAPABILITY tier: what the model running
                           the turn can handle (cloud_strong /
                           cloud_fast / local). One prompt per tier,
                           never per model: a hardcoded list of model
                           names would need editing every time a
                           provider ships something new, so the tier
                           is guessed from substrings in the model
                           name and every unrecognised cloud model
                           lands on the safe default (``cloud_fast``).
    effort_<effort>.txt  — the EFFORT overlay: how hard this turn
                           should work (fast / balanced / pro), chosen
                           per turn from the task in Auto.

Effort is a SEPARATE axis from payment and runtime. Free/Paid is a
property of the key; Local is where inference runs; neither says how
much effort a task deserves. A free model can serve Fast or Balanced;
Pro asks for deep work from the strongest available target and says
its limits honestly when the target is small.

Precedence, first match wins, for both axes independently:

    Settings (a forced mode)  →  AI_AGENT_PROMPT_TIER  →  auto routing

Stored legacy modes map onto the two axes (see config.py):
``cloud_fast`` pins tier cloud_fast + effort fast, ``cloud_strong``
pins tier cloud_strong + effort pro, ``local`` pins tier local +
effort fast. The effort labels (fast / balanced / pro) pin only the
effort; the capability tier still comes from the classifier.

Tier and effort are resolved **once per user turn** and frozen for
that turn; see ``modes.agent.AgentMode._system_prompt``. Nothing here
reads or writes the network, and nothing here can be pointed at a
file outside ``prompts/`` — tier and effort are checked against
whitelists before they ever become part of a path.
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

# The effort overlays (Release 28) — same whitelist discipline.
EFFORTS = ("fast", "balanced", "pro")
EFFORT_DEFAULT = "balanced"

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

# (tier, effort) -> assembled prompt. Populated on first use; cleared
# by clear_cache().
_CACHE: dict = {}


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


# ── Effort routing (Release 28) ──
# Task signals, not model signals: the SAME model should work a
# translation at Fast and a debugging session at Pro. Single English
# words match on word boundaries (so "plan" cannot fire inside
# "explain"); phrases and Burmese markers match as substrings.
_PRO_WORDS = (
    "code", "function", "script", "bug", "debug", "refactor",
    "implement", "sql", "database", "compare", "analyze", "analyse",
    "analysis", "report", "essay", "document", "plan",
)
_PRO_PHRASES = (
    "```", "traceback", "stack trace", "step by step", "step-by-step",
    "ကုဒ်", "ပရိုဂရမ်", "အဆင့်ဆင့်", "နှိုင်းယှဉ်", "စီမံချက်",
)
_FAST_PHRASES = (
    "translate", "what is", "what's", "who is", "when is",
    "when did", "capital of", "meaning of", "mean in",
    "how do you say", "ဘာသာပြန်", "ဆိုတာ ဘာ", "မြို့တော်",
    "အဓိပ္ပာယ်", "ဘယ်လို ခေါ်",
)
_LONG_INPUT = 1500     # chars — a pasted document is never a Fast task
_SHORT_INPUT = 200     # chars — with a fast signal, a Fast task


def _word_in(text: str, word: str) -> bool:
    """True when `word` appears in `text` bounded by non-letters."""
    i = text.find(word)
    while i != -1:
        before_ok = i == 0 or not text[i - 1].isalpha()
        after = i + len(word)
        after_ok = after == len(text) or not text[after].isalpha()
        if before_ok and after_ok:
            return True
        i = text.find(word, i + 1)
    return False


def classify_effort(text: str, backend: str = "") -> str:
    """Route one turn's effort from its task signals.

    Long inputs and code/multi-step/high-stakes markers -> pro; short
    simple-fact or translation asks -> fast; everything else ->
    balanced. The local runtime caps effort at balanced: an 8K-window
    on-device model asked to work at Pro depth runs out of window
    before it runs out of instructions, so its ceiling is balanced
    (and fast for simple asks)."""
    t = (text or "").strip()
    local = (backend or "").strip().lower() == "local"
    if not t:
        return EFFORT_DEFAULT
    low = t.lower()
    if len(t) >= _LONG_INPUT \
            or any(ph in low for ph in _PRO_PHRASES) \
            or any(_word_in(low, w) for w in _PRO_WORDS):
        return "balanced" if local else "pro"
    if len(t) <= _SHORT_INPUT and any(ph in low for ph in _FAST_PHRASES):
        return "fast"
    if (len(t) <= 60 and "?" in t) or len(t) <= 40:
        # A short bare question or a greeting-length message: the
        # Fast overlay still allows tools and search for live data.
        return "fast"
    return EFFORT_DEFAULT


def resolve_effort(backend: str, user_text: str, *, mode: str = "auto") -> tuple[str, str]:
    """``(effort, source)`` for one turn — the effort twin of
    :func:`resolve_tier`. An explicit effort label pins the effort;
    a legacy tier mode implies one (config.PROMPT_TIER_LEGACY_EFFORT);
    otherwise the task signals decide."""
    mode = (mode or "auto").strip()
    if mode in EFFORTS:
        if config.stored_prompt_tier_mode() == mode:
            return mode, "settings"
        if (os.environ.get(config.PROMPT_TIER_ENV) or "").strip() == mode:
            return mode, "env"
        return mode, "settings"
    legacy = config.PROMPT_TIER_LEGACY_EFFORT.get(mode)
    if legacy:
        if config.stored_prompt_tier_mode() == mode:
            return legacy, "settings"
        if (os.environ.get(config.PROMPT_TIER_ENV) or "").strip() == mode:
            return legacy, "env"
        return legacy, "settings"
    return classify_effort(user_text, backend), "auto"


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


def load_prompt(tier: str, effort: str = EFFORT_DEFAULT) -> str:
    """The system prompt for ``tier`` at ``effort``:
    `_base.txt` + `agent_<tier>.txt` + `effort_<effort>.txt`.

    Unknown tier, missing file or empty file → ``AGENT_SYSTEM`` with a warning,
    so a half-deployed prompts/ folder degrades to the old behaviour instead of
    to no prompt at all. A missing EFFORT overlay degrades one step
    less far — base + tier is still a complete prompt — so it is
    logged and skipped rather than discarding the whole assembly.
    An unknown effort is treated as the default rather than trusted.
    """
    tier = (tier or "").strip()
    if tier not in TIERS:
        log.warning("unknown prompt tier %r — using AGENT_SYSTEM", tier)
        return _fallback_prompt()
    effort = (effort or "").strip()
    if effort not in EFFORTS:
        effort = EFFORT_DEFAULT
    key = (tier, effort)
    cached = _CACHE.get(key)
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
    overlay = _read(PROMPTS_DIR / ("effort_%s.txt" % effort))
    if overlay is None:
        log.warning("effort overlay effort_%s.txt missing or empty — "
                    "prompt ships without it", effort)
    else:
        out = out + "\n\n" + overlay
    # Plain dict assignment: two threads racing here both read the same bytes
    # and both write the same value, which is why this needs no lock.
    _CACHE[key] = out
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
    """What the Settings page shows: the mode, the resolved tier and
    its source, plus the effort the next turn would get (Release 28).

    The effort shown for Auto is the one an empty task signal yields
    — the balanced default or the pinned effort; a real turn's effort
    depends on its message, which Settings cannot know in advance."""
    m = (mode or "").strip()
    if m not in config.PROMPT_TIER_MODES:
        m = config.prompt_tier_mode()
    tier, source = resolve_tier(backend, model, mode=m)
    effort, effort_source = resolve_effort(backend, "", mode=m)
    return {"mode": m, "resolved": tier, "source": source, "model": model,
            "effort": effort, "effort_source": effort_source}