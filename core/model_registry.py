"""Max context window per model — one table, four resolution layers.

The context meter used to divide by ``Config.context_size`` (the *local*
engine's KV cache, 8192 by default) for every request, so a cloud turn on a
1M-token Gemini model reported 0.1% and a DeepSeek turn that really was near
its 64K ceiling reported nothing at all. This module answers the only question
that matters per request: how many tokens can *this* model actually take?

Resolution order (first hit wins):

0. runtime   — a limit the provider itself reported, remembered in
   ``~/.ai-agent/model_limits.json`` (an unknown model that answered "maximum
   context length is N" is learned once and never guessed again)
1. exact     — the model name is in :data:`MODEL_CONTEXT_LIMITS`
2. prefix    — the longest registry key that is a prefix of the model name
   (``deepseek-flash-2026`` → ``deepseek-flash``, ``gpt-4o-2024-08-06`` →
   ``gpt-4o``)
3. endpoint  — nothing known about the model, so the endpoint decides:
   local → :data:`LOCAL_CONTEXT_DEFAULT`, cloud → :data:`UNKNOWN_CLOUD_DEFAULT`,
   unknown host → :data:`UNKNOWN_CLOUD_DEFAULT` (a cloud-shaped guess is far
   safer than pretending a provider on a private LAN can take 1M tokens)

Nothing here is user-tunable and nothing here ever sets the local engine size:
``Config.context_size`` stays the KV-cache choice, this is only the request
budget. Chat Mode never imports this module.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from core.log import get_logger

log = get_logger("model_registry")

# ── the table ────────────────────────────────────────────────────────────
#
# Keys are the bare model name as providers report it (OpenRouter's vendor
# prefix and Gemini's ``models/`` prefix are stripped before lookup, and a
# trailing ``:free`` is too — see :func:`lookup_names`).
MODEL_CONTEXT_LIMITS: dict[str, int] = {
    # Google
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.5-flash-lite": 1_048_576,
    "gemini-1.5-pro": 2_097_152,
    "gemini-3-flash-preview": 1_048_576,
    # DeepSeek
    "deepseek-v4-pro": 1_000_000,
    "deepseek-v4-flash": 1_000_000,
    "deepseek-chat": 64_000,
    # Not one of DeepSeek's published API names (the presets in `MODELS` are
    # deepseek-chat and deepseek-reasoner). This install's agent_config.json
    # points at it, so the entry stays — without it the meter would have
    # nothing to describe the running model — but it is unverified against the
    # provider, and AgentBackend warns when a configured model is off-preset.
    "deepseek-flash": 64_000,
    "deepseek-reasoner": 64_000,
    # Anthropic
    "claude-sonnet-4.6": 1_000_000,
    "claude-3-5-sonnet": 200_000,
    "claude-3-5-haiku": 200_000,
    "claude-3-opus": 200_000,
    # OpenAI
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    # OpenRouter free tier (vendor-prefixed names, as OpenRouter sends them)
    "minimax/minimax-m3:free": 1_048_576,
    "thinkingmachines/inkling:free": 1_048_576,
    "meituan/longcat-2.0:free": 1_048_576,
    # Groq
    "llama-3.3-70b-versatile": 128_000,
    "llama-3.1-8b-instant": 128_000,
    # Qwen / NVIDIA / Arcee on the free routes
    "qwen/qwen3.6-plus:free": 1_000_000,
    "nvidia/nemotron-3-super-120b-a12b:free": 262_000,
    "arcee-ai/trinity-large-preview:free": 131_000,
    # Qwen, GLM, Kimi, Mistral, Nemotron
    "qwen3.8-flash": 1_000_000,
    "qwen3-coder-plus": 1_000_000,
    "glm-4.7-flash": 200_000,
    "glm-5.2": 1_000_000,
    "kimi-k3": 1_000_000,
    "kimi-k2.5": 262_000,
    "mistral-small-4": 256_000,
    "devstral-2": 256_000,
    "command-a": 256_000,
    "nemotron-3.5-lightning": 1_000_000,
    # Local / custom
    "space-bunny-alpha": 1_000_000,
    "gemma:2b": 8_000,
    "gemma:4b": 8_000,
    "phi-3-mini": 4_000,
    "llama3:8b": 8_000,
}

# Used when the model is unknown *and* the endpoint is this device. The local
# engine's own ceiling is the truth here; this is the floor used by callers that
# never load it (an HTTP client, a test).
LOCAL_CONTEXT_DEFAULT = 8_000
# Unknown model on a cloud endpoint. 128K is the industry floor and, unlike a
# 1M guess, cannot turn a too-long request into an immediate 400.
UNKNOWN_CLOUD_DEFAULT = 128_000

# ── endpoint classification ──────────────────────────────────────────────
LOCAL_ENDPOINTS = ("localhost", "127.0.0.1", "0.0.0.0", "::1", "unix:")
CLOUD_ENDPOINTS = (
    "api.deepseek.com",
    "api.openai.com",
    "generativelanguage.googleapis.com",
    "api.anthropic.com",
    "api.mistral.ai",
    "openrouter.ai",
    "api.groq.com",
)

# A reported limit outside this range is a parser accident, not a window size.
MIN_WINDOW = 1_024
MAX_LEARNED = 10_000_000
_MIN_LEARNED = MIN_WINDOW
# Bound on the persisted file: it is a cache of provider answers, not a
# database, and it must never grow without limit on a device.
_MAX_ENTRIES = 500

# Where learned limits live. ``None`` until first use — the test harness
# re-points ``config.CONFIG_DIR`` at a temp HOME *after* import, so resolving
# the path lazily (and re-reading CONFIG_DIR each time) is what keeps a test
# run from writing into the real ~/.ai-agent.
_PATH_OVERRIDE: Path | None = None
_CACHE: dict | None = None


# ── names ────────────────────────────────────────────────────────────────

def normalize_model(model: str) -> str:
    """Lowercase, trim, and drop the decorations providers add to a name.

    ``models/gemini-2.5-flash`` → ``gemini-2.5-flash``; that is Gemini's own
    spelling in the REST path and would otherwise miss every entry.
    """
    name = str(model or "").strip().lower()
    if name.startswith("models/"):
        name = name[len("models/"):]
    return name


def lookup_names(model: str) -> list[str]:
    """Candidate names to try, most specific first.

    OpenRouter sends ``google/gemini-2.5-flash:free`` while the table is keyed
    on ``gemini-2.5-flash``; both the vendor prefix and the routing suffix are
    therefore peeled off as extra candidates.
    """
    base = normalize_model(model)
    if not base:
        return []
    names = [base]
    if "/" in base:
        tail = base.rsplit("/", 1)[-1]
        if tail and tail not in names:
            names.append(tail)
    for name in list(names):
        head, sep, tail = name.rpartition(":")
        # Only a routing suffix (:free, :beta, :nitro) — a port or a colon
        # inside a real name is left alone.
        if sep and head and len(tail) <= 12 and tail.isalnum():
            if head not in names:
                names.append(head)
    return names


def classify_endpoint(base_url: str = "") -> str:
    """``local`` | ``cloud`` | ``unknown`` for a provider base URL."""
    url = str(base_url or "").strip().lower()
    if not url:
        return "unknown"
    if any(marker in url for marker in LOCAL_ENDPOINTS):
        return "local"
    if any(marker in url for marker in CLOUD_ENDPOINTS):
        return "cloud"
    return "unknown"


# ── the learned-limit store ──────────────────────────────────────────────

def limits_path() -> Path:
    """``<config dir>/model_limits.json`` — resolved on every call."""
    if _PATH_OVERRIDE is not None:
        return Path(_PATH_OVERRIDE)
    try:
        import config
        return Path(config.CONFIG_DIR) / "model_limits.json"
    except Exception as e:                       # config unimportable
        log.debug("config dir unavailable for model limits: %s: %s",
                  type(e).__name__, e)
        return Path.home() / ".ai-agent" / "model_limits.json"


def load_limits(path: Path | None = None) -> dict:
    """Learned ``{key: limit}`` map; empty on a missing or corrupt file.

    A half-written file must never take the agent down, so every failure here
    is logged once and answered with "no learned limits".
    """
    global _CACHE
    if path is None and _CACHE is not None:
        return _CACHE
    target = Path(path) if path is not None else limits_path()
    data: dict = {}
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            entries = raw.get("limits") if isinstance(raw.get("limits"), dict) \
                else raw
            for key, value in (entries or {}).items():
                n = _as_limit(value)
                if n:
                    data[str(key)] = n
    except FileNotFoundError:
        data = {}
    except (OSError, ValueError) as e:
        log.warning("model limits file unreadable (%s: %s) — ignoring it",
                    type(e).__name__, e)
        data = {}
    if path is None:
        _CACHE = data
    return data


def save_limits(data: dict, path: Path | None = None) -> bool:
    """Atomically persist the learned map. Never raises."""
    target = Path(path) if path is not None else limits_path()
    payload = {"version": 1, "limits": dict(list(data.items())[:_MAX_ENTRIES])}
    tmp = target.with_name(target.name + ".tmp")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(target)
        return True
    except (OSError, ValueError) as e:
        log.warning("could not save model limits to %s: %s: %s",
                    target, type(e).__name__, e)
        try:
            tmp.unlink()
        except OSError:
            pass
        return False


def reset_limits(path: Path | None = None) -> None:
    """Forget every learned limit (tests / a manual reset)."""
    global _CACHE
    _CACHE = {} if path is None else None
    try:
        p = limits_path() if path is None else Path(path)
        if p.exists():
            p.unlink()
    except OSError as e:
        log.debug("could not remove %s: %s", path, e)


def _as_limit(value) -> int:
    """A sane positive int, or 0 — 131_072 but not ``"131072 tokens"``."""
    try:
        n = int(str(value).strip().replace("_", "").replace(",", ""))
    except (TypeError, ValueError):
        return 0
    if _MIN_LEARNED <= n <= MAX_LEARNED:
        return n
    return 0


def store_key(model: str, base_url: str = "") -> str:
    """Key a learned limit is filed under.

    Qualified by host when there is one, so a self-hosted model named like a
    cloud one cannot inherit its window size. With no base URL configured — the
    usual case, since each adapter carries its own default — the bare model name
    is the key, which is also what :func:`learned_limit` falls back to.
    """
    name = normalize_model(model)
    if not name:
        return ""
    url = str(base_url or "").strip().lower()
    host = ""
    if "://" in url:
        host = url.split("://", 1)[1]
    else:
        host = url
    host = host.split("/", 1)[0].split("@")[-1].split(":")[0]
    if not host:
        return name
    return "%s::%s" % (host, name)


def learned_limit(model: str, base_url: str = "") -> int | None:
    """Runtime limit for this model, or None when nothing was learned."""
    data = load_limits()
    for key in (store_key(model, base_url), normalize_model(model)):
        n = data.get(key)
        if n:
            return int(n)
    return None


def record_limit(model: str, base_url: str = "", limit=None,
                 text: str = "", path: Path | None = None) -> int | None:
    """Remember a limit the provider just told us. Returns what was stored.

    ``limit`` wins over ``text`` (the caller may already have parsed the
    number); when neither yields one, nothing is written.
    """
    name = normalize_model(model)
    if not name:
        return None
    n = _as_limit(limit) if limit is not None else parse_limit(text)
    if not n:
        return None
    key = store_key(model, base_url)
    data = load_limits(path)
    data.pop(key, None)                # re-insert so the cap keeps the newest
    data[key] = n
    if len(data) > _MAX_ENTRIES:
        for stale in list(data)[: len(data) - _MAX_ENTRIES]:
            data.pop(stale, None)
    save_limits(data, path)
    return n


# ── error text ───────────────────────────────────────────────────────────

# "This model's maximum context length is 16384 tokens"
_LIMIT_PATTERNS = (
    re.compile(r"maximum context length is\s*([0-9][0-9,_]*)", re.I),
    re.compile(r"context length of\s*([0-9][0-9,_]*)", re.I),
    re.compile(r"maximum of\s*([0-9][0-9,_]*)\s*tokens", re.I),
    # "… 130000 tokens > 65536 tokens" — the limit is the *last* group.
    re.compile(r"[0-9][0-9,_]*\s*tokens\s*>\s*([0-9][0-9,_]*)", re.I),
)

# Words that mean "the request did not fit", not "you sent bad JSON".
_CONTEXT_WORDS = (
    "maximum context length",
    "context length",
    "context_length_exceeded",
    "context window",
    "too many tokens",
    "reduce the length",
    "prompt is too long",
    "input is too long",
    "request too large",
    "exceeds the context",
    # Gemini: "The input token count (250000) exceeds the maximum number of
    # tokens allowed (200000)."
    "exceeds the maximum",
    "input token count",
)


def is_context_error(text: str) -> bool:
    """Does this provider complaint mean "too much history"?"""
    msg = str(text or "").lower()
    if not msg:
        return False
    # A reasoning_content 400 is a different fault and has its own retry;
    # it must not be counted as a context overflow.
    if "reasoning_content" in msg:
        return False
    return any(word in msg for word in _CONTEXT_WORDS)


def parse_limit(text: str) -> int | None:
    """Pull the window size out of a provider error message."""
    msg = str(text or "")
    for pattern in _LIMIT_PATTERNS:
        m = pattern.search(msg)
        if m:
            n = _as_limit(m.group(1))
            if n:
                return n
    return None


# ── resolution ───────────────────────────────────────────────────────────

def _exact(name: str) -> int | None:
    return MODEL_CONTEXT_LIMITS.get(name)


def _prefix(name: str) -> tuple[str, int] | None:
    """Longest registry key that is a prefix of ``name``."""
    best = None
    for key, limit in MODEL_CONTEXT_LIMITS.items():
        if name == key or name.startswith(key):
            if best is None or len(key) > len(best[0]):
                best = (key, limit)
    return best


def describe_limit(model: str, base_url: str = "",
                   limits: dict | None = None) -> dict:
    """Every layer's answer, plus which one won — for logs and the UI tooltip."""
    names = lookup_names(model)
    endpoint = classify_endpoint(base_url)
    info = {
        "model": str(model or ""),
        "base_url": str(base_url or ""),
        "endpoint": endpoint,
        "limit": UNKNOWN_CLOUD_DEFAULT,
        "source": "endpoint_unknown",
        "matched": "",
    }
    runtime = None if limits is not None else learned_limit(model, base_url)
    if limits is not None:
        for key in (store_key(model, base_url), normalize_model(model)):
            n = _as_limit(limits.get(key))
            if n:
                runtime = n
                info["matched"] = key
                break
    if runtime:
        info.update(limit=int(runtime), source="runtime")
        return info
    for name in names:
        limit = _exact(name)
        if limit:
            info.update(limit=limit, source="exact", matched=name)
            return info
    for name in names:
        hit = _prefix(name)
        if hit:
            info.update(limit=hit[1], source="prefix", matched=hit[0])
            return info
    if endpoint == "local":
        info.update(limit=LOCAL_CONTEXT_DEFAULT, source="endpoint_local")
    elif endpoint == "cloud":
        info.update(limit=UNKNOWN_CLOUD_DEFAULT, source="endpoint_cloud")
    return info


def resolve_limit(model: str, base_url: str = "",
                  limits: dict | None = None) -> int:
    """Max context window for this model+endpoint, in tokens."""
    return int(describe_limit(model, base_url, limits)["limit"])