"""Live model discovery (Release 27) — what models a key can really use.

The built-in ``MODELS`` presets in :mod:`core.agent_backend` are a frozen
snapshot: providers retire ids (Groq removed ``llama-3.3-70b-versatile``,
which then failed only at message time with ``model_not_found``) and ship
new ones the list never learns about. This module normalises each
provider's live ``/models`` answer into one shape —

    {"id", "name", "context" (int|None), "kind", "free" (True|False|None)}

with ``kind`` in ``chat | embed | audio | image | other`` — and keeps a
per-provider on-disk cache so Settings stays fast and works offline.

Honesty rules, mirroring the other stores:

* classification is provider metadata first (Gemini's
  ``supportedGenerationMethods``, OpenRouter's ``architecture.modality``
  and ``pricing``), then conservative id heuristics. A model is never
  hidden: the worst case is ``kind == "chat"`` with no badge, so an
  unknown new chat model stays selectable.
* ``free`` is only ever True/False when the provider actually reported
  pricing (OpenRouter); everywhere else it stays None — unknown, never
  guessed.
* the cache is a convenience, never a source of truth failures: a
  corrupt cache file is ignored, a failed save is logged, and neither
  ever raises into a request.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import config
from core import data_version
from core.log import get_logger

log = get_logger("model_discovery")

VERSION = 1
FILE_NAME = "models_cache.json"

# How long a fetched list is served from disk before a refetch. Model
# catalogues change on the order of days, and the Settings ↻ button
# forces a refresh, so 24h keeps Settings instant without going stale
# in any way that matters.
TTL_SECONDS = 24 * 3600

KINDS = ("chat", "embed", "audio", "image", "other")

# Conservative id heuristics, applied only when provider metadata says
# nothing. Each family is a thing that cannot hold a chat conversation.
# Release 28: the round-3 evaluation found a dozen media models in the
# live OpenRouter list wearing chat rows (Lyria music, the *-image
# generation families, Voxtral transcription) — the hint lists grew to
# name those families. Still conservative: a hint only ever fires on a
# family that is media by construction, and an unknown id stays chat.
_EMBED_HINTS = ("embed",)
_AUDIO_HINTS = ("whisper", "tts", "transcribe", "lyria", "voxtral",
                "suno", "music")
_IMAGE_HINTS = ("dall-e", "gpt-image", "imagen", "-image", "flux")
_OTHER_HINTS = ("moderation", "rerank", "guard", "veo")


def kind_from_id(model_id: str) -> str:
    """Best-effort kind from the id alone. Unknown ⇒ ``chat``."""
    mid = (model_id or "").lower()
    if any(h in mid for h in _EMBED_HINTS):
        return "embed"
    if any(h in mid for h in _AUDIO_HINTS):
        return "audio"
    if any(h in mid for h in _IMAGE_HINTS):
        return "image"
    if any(h in mid for h in _OTHER_HINTS):
        return "other"
    return "chat"


def _kind_from_modality(modality: str) -> str | None:
    """Kind from OpenRouter's ``architecture.modality``, or None.

    The string is ``<inputs>-><outputs>`` ("text+image->text",
    "text->audio"). Only the OUTPUT side decides what a model can be
    used for: a model whose outputs contain no text cannot hold a chat
    conversation, whatever its id looks like — that is decisive, and
    overrides the id heuristics. A text output says nothing the id
    heuristics don't (an audio->text transcription model still isn't a
    chat model), so it returns None and the heuristics keep their say.
    Video has no kind of its own in KINDS; it lands in "other".
    """
    if "->" not in modality:
        return None
    outputs = [t.strip() for t in modality.rsplit("->", 1)[1].split("+")]
    if "embedding" in outputs:
        return "embed"
    if "text" in outputs or not outputs:
        return None
    if "audio" in outputs:
        return "audio"
    if "image" in outputs:
        return "image"
    if "video" in outputs:
        return "other"
    return None


def _entry(mid, name=None, context=None, kind="chat", free=None):
    return {
        "id": mid,
        "name": name or mid,
        "context": int(context) if isinstance(context, (int, float))
                   and not isinstance(context, bool) and context > 0 else None,
        "kind": kind if kind in KINDS else "chat",
        "free": free if isinstance(free, bool) else None,
    }


def _openrouter_free(entry: dict):
    """True/False from OpenRouter pricing; None when pricing is absent."""
    pricing = entry.get("pricing")
    if not isinstance(pricing, dict):
        return None
    prompt, completion = pricing.get("prompt"), pricing.get("completion")
    if prompt is None or completion is None:
        return None
    return str(prompt) == "0" and str(completion) == "0"


def _normalise_list_entry(provider: str, entry) -> dict | None:
    if not isinstance(entry, dict):
        return None
    mid = entry.get("id")
    if not isinstance(mid, str) or not mid:
        return None
    kind = kind_from_id(mid)
    free = None
    context = entry.get("context_length")
    arch = entry.get("architecture")
    if isinstance(arch, dict):
        modality = str(arch.get("modality") or "").lower()
        by_modality = _kind_from_modality(modality)
        if by_modality is not None:
            kind = by_modality
    if provider == "openrouter":
        free = _openrouter_free(entry)
    return _entry(mid, name=entry.get("name"), context=context,
                  kind=kind, free=free)


def _normalise_gemini_entry(entry) -> dict | None:
    if not isinstance(entry, dict):
        return None
    raw = entry.get("name")
    if not isinstance(raw, str) or not raw:
        return None
    mid = raw[len("models/"):] if raw.startswith("models/") else raw
    methods = entry.get("supportedGenerationMethods")
    methods = [str(m) for m in methods] if isinstance(methods, list) else []
    if "generateContent" in methods:
        kind = "chat"
    elif "embedContent" in methods:
        kind = "embed"
    else:
        kind = kind_from_id(mid)
    return _entry(mid, name=entry.get("displayName"),
                  context=entry.get("inputTokenLimit"), kind=kind)


def normalise(provider: str, data) -> list:
    """A provider's /models payload → the normalised list.

    Payload shapes: OpenAI-compatible and Anthropic carry the list under
    ``data``; Gemini under ``models``. Anything unexpected yields what
    could be salvaged (possibly an empty list) — never an exception, so
    a weird payload degrades to the fallback path instead of an error.
    OpenRouter entries carry a ``created`` epoch; when present the list
    is sorted newest-first (the catalogue order his reference page
    uses). Other providers' order is kept as sent.
    """
    if not isinstance(data, dict):
        return []
    out = []
    if provider == "gemini":
        raw_list = data.get("models")
        if isinstance(raw_list, list):
            for entry in raw_list:
                item = _normalise_gemini_entry(entry)
                if item:
                    out.append(item)
        return out
    raw_list = data.get("data")
    if isinstance(raw_list, list):
        created = []
        for entry in raw_list:
            item = _normalise_list_entry(provider, entry)
            if item:
                out.append(item)
                c = entry.get("created") if isinstance(entry, dict) else None
                created.append(c if isinstance(c, (int, float)) else None)
        if any(c is not None for c in created):
            order = sorted(range(len(out)),
                           key=lambda i: created[i] or 0, reverse=True)
            out = [out[i] for i in order]
    return out


def fallback_models(provider: str, presets: dict) -> list:
    """The built-in preset list as normalised entries (kinds unknown —
    presets are plain ids, so they stay chat-eligible and unbadged)."""
    return [_entry(mid) for mid in (presets or {}).get(provider) or []]


# ───────────────────────────── cache ─────────────────────────────

def default_path() -> Path:
    """Resolved per call: tests re-point CONFIG_DIR after import."""
    return Path(config.CONFIG_DIR) / FILE_NAME


def _stamp(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).isoformat(timespec="seconds")


def fetched_at_iso(entry) -> str | None:
    if not entry:
        return None
    ts = entry.get("fetched_at")
    if isinstance(ts, (int, float)) and ts > 0:
        return _stamp(float(ts))
    return None


def load_cache(path: Path | None = None) -> dict:
    """{provider: {"fetched_at": epoch, "models": [...]}} — {} when the
    file is missing or unreadable. Corrupt ⇒ ignored, by design."""
    path = Path(path) if path else default_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        log.info("models cache unreadable (%s) — ignoring it", e)
        return {}
    if not isinstance(raw, dict):
        return {}
    data_version.warn_if_newer(FILE_NAME, raw, VERSION, log)
    providers = raw.get("providers")
    if not isinstance(providers, dict):
        return {}
    out = {}
    for provider, entry in providers.items():
        if (isinstance(entry, dict)
                and isinstance(entry.get("models"), list)
                and isinstance(entry.get("fetched_at"), (int, float))):
            models = [m for m in entry["models"]
                      if isinstance(m, dict) and isinstance(m.get("id"), str)]
            out[provider] = {"fetched_at": float(entry["fetched_at"]),
                             "models": models}
    return out


def cached(provider: str, path: Path | None = None) -> dict | None:
    return load_cache(path).get(provider)


def is_fresh(entry, now: float | None = None) -> bool:
    if not entry:
        return False
    return (now if now is not None else time.time()) \
        - float(entry.get("fetched_at") or 0) < TTL_SECONDS


def store(provider: str, models: list, path: Path | None = None) -> None:
    """Merge one provider's fresh list into the cache, atomically.

    Never raises: a failed cache write must not fail the request that
    fetched the list.
    """
    path = Path(path) if path else default_path()
    try:
        data = load_cache(path)
        data[provider] = {"fetched_at": time.time(), "models": models}
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": VERSION, "providers": data}
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(path)
    except OSError as e:
        log.warning("models cache save failed: %s", e)
