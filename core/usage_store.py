"""Persisted token-usage totals for Agent Mode (Release 17).

R13's Usage view could only show key counts and per-key success
counters, because no token counts were stored anywhere: the adapters
already parse each provider's usage block onto the Step (and the
backend already logs it), but nothing kept it. This store is the
missing ledger — one JSON file under CONFIG_DIR, fed by the same
points that call ``record_usage()`` in modes/agent.py, so a call is
metered exactly when the status-bar chip says it ran.

Honesty rules, mirroring the adapters' own:

* a call is recorded exactly once, when its step completes — the
  backend's internal key retries return a single Step, so they
  cannot double-count;
* token numbers are *measured* values copied off the provider's
  response. When a response carries no usage block the call is
  still counted, with its token fields left ``None`` (unknown) —
  never estimated, never zero-filled;
* local (Gemma) turns are counted as calls under the ``local``
  provider with token fields ``None``: core/tokens.py is a BM25
  tokenizer for memory search, not a usage meter, so there is no
  measured local count to store and an estimate is not dressed up
  as one.

Shape (version 1)::

    {"version": 1,
     "providers": {"openai": {"models": {"gpt-4o-mini": {
         "calls": 3, "measured_calls": 3,
         "prompt_tokens": 1200, "completion_tokens": 300,
         "updated": "2026-10-22T19:20:00"}}}},
     "updated": "2026-10-22T19:20:00"}

``prompt_tokens`` / ``completion_tokens`` are sums over measured
calls only; each stays ``None`` while no call has reported that
field. The file sits under CONFIG_DIR, so backup.py's home walk
covers it with no extra wiring.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

import config
from core import data_version
from core.log import get_logger

log = get_logger("usage_store")

VERSION = 1
FILE_NAME = "usage.json"

_lock = threading.Lock()


def default_path() -> Path:
    """Where the ledger lives, resolved per call like _fs_guard's
    backup dir: tests re-point config.CONFIG_DIR after import, and a
    stale import-time path would write into the previous run's dir."""
    return Path(config.CONFIG_DIR) / FILE_NAME


def _stamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _as_count(value) -> int | None:
    """A token count off the wire, or None when it is not one.

    Providers send ints; a couple of compatible endpoints send
    numeric strings/floats. Booleans, negatives and anything else
    are not counts — the field is simply unreported.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        return int(value) if value >= 0 and value == int(value) else None
    if isinstance(value, str):
        try:
            n = int(value.strip())
        except ValueError:
            return None
        return n if n >= 0 else None
    return None


def _quarantine(path: Path) -> None:
    """Copy a broken ledger aside so a later save cannot destroy it."""
    try:
        if not path.exists():
            return
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        side = path.with_name(f"{path.name}.corrupt-{ts}")
        side.write_bytes(path.read_bytes())
        log.error("corrupt usage store preserved at %s", side)
    except OSError as e:
        log.error("could not preserve corrupt usage store %s: %s: %s",
                  path, type(e).__name__, e)


class UsageStore:
    """Load/save/record for usage.json. Mirrors Memory's IO shape."""

    VERSION = VERSION

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path is not None else default_path()
        self.data = self._empty()

    @staticmethod
    def _empty() -> dict:
        return {"version": VERSION, "providers": {}, "updated": ""}

    # ---------- IO ----------

    @classmethod
    def load(cls, path: Path | None = None) -> "UsageStore":
        store = cls(path)
        if not store.path.exists():
            return store
        try:
            raw = json.loads(store.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            # Totals vanish silently here if we just return empty —
            # and the next save() would overwrite the file for good.
            log.error("usage store unreadable (%s: %s) — starting empty",
                      type(e).__name__, e)
            _quarantine(store.path)
            return store
        if not isinstance(raw, dict):
            log.error("usage store is not a JSON object — starting empty")
            _quarantine(store.path)
            return store
        data_version.warn_if_newer("usage store", raw, VERSION, log)
        merged = cls._empty()
        if isinstance(raw.get("providers"), dict):
            merged["providers"] = raw["providers"]
        if isinstance(raw.get("updated"), str):
            merged["updated"] = raw["updated"]
        store.data = merged
        return store

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.data["version"] = VERSION
            self.data["updated"] = _stamp()
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(self.data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(self.path)
        except OSError as e:
            log.error("usage save failed for %s: %s: %s",
                      self.path, type(e).__name__, e)
            raise

    # ---------- mutation ----------

    def record(self, provider: str, model: str, usage=None) -> None:
        """Add one completed call. ``usage`` is the Step's usage dict
        (``prompt_tokens`` / ``completion_tokens``), or None/empty when
        the provider reported nothing."""
        provider = (provider or "").strip()
        if not provider:
            return
        model = (model or "").strip() or "unknown"
        prompt = completion = None
        if isinstance(usage, dict):
            prompt = _as_count(usage.get("prompt_tokens"))
            completion = _as_count(usage.get("completion_tokens"))
        measured = prompt is not None or completion is not None

        providers = self.data.setdefault("providers", {})
        entry = providers.setdefault(provider, {"models": {}})
        models = entry.setdefault("models", {})
        agg = models.setdefault(model, {
            "calls": 0, "measured_calls": 0,
            "prompt_tokens": None, "completion_tokens": None,
            "updated": "",
        })
        agg["calls"] = int(agg.get("calls") or 0) + 1
        if measured:
            agg["measured_calls"] = int(agg.get("measured_calls") or 0) + 1
            if prompt is not None:
                agg["prompt_tokens"] = int(agg.get("prompt_tokens") or 0) + prompt
            if completion is not None:
                agg["completion_tokens"] = \
                    int(agg.get("completion_tokens") or 0) + completion
        agg["updated"] = _stamp()

    def totals(self) -> dict:
        """Deep-enough copy of the per-provider aggregates."""
        out = {}
        for provider, entry in (self.data.get("providers") or {}).items():
            models = {}
            for model, agg in ((entry or {}).get("models") or {}).items():
                models[model] = dict(agg or {})
            out[provider] = {"models": models}
        return out


# ── module-level recorder (the choke point modes/agent.py calls) ──

def record_call(backend: str, model: str = "", provider: str = "",
                usage=None) -> bool:
    """Record one completed call in the ledger. NEVER raises.

    ``backend`` is record_usage()'s vocabulary: ``"cloud"`` records
    under the call's provider (a cloud call with no provider cannot
    be attributed and is skipped), ``"local"`` records under the
    ``local`` provider with no token data. Anything else is not a
    completed model call and is ignored.

    The store is re-read from disk on every call: the terminal app
    and the web server are separate processes sharing this file, and
    a cached copy would silently drop the other process's counts.
    The amounts are tiny and steps are network-bound, so the read
    costs nothing that matters.
    """
    try:
        backend = (backend or "").strip()
        if backend == "cloud":
            provider = (provider or "").strip()
            if not provider:
                return False
        elif backend == "local":
            provider = "local"
        else:
            return False
        with _lock:
            store = UsageStore.load()
            store.record(provider, model, usage)
            store.save()
        return True
    except Exception as e:
        log.warning("usage record failed: %s: %s", type(e).__name__, e)
        return False


def totals() -> dict:
    """Current aggregates for Settings; {} when unreadable."""
    try:
        with _lock:
            return UsageStore.load().totals()
    except Exception as e:
        log.warning("usage totals failed: %s: %s", type(e).__name__, e)
        return {}
