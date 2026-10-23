"""API key storage and rotation for Agent Mode's cloud backends.

Keys live in ``~/.ai-agent/agent_config.json`` (``config.private_keys_file()``)
with mode 0600 — app-private storage, beside config/memory/sessions. They
used to sit in ``<project>/data/agent_config.json``; on the phone the project
is on shared storage, where any app with the storage permission could read
them, so on first load the file is copied to the private location, verified
byte for byte, and the shared original deleted (see
:func:`config.migrate_keys_to_private`). The private location is canonical
under both of the phone's HOMEs (the server runs with
HOME=/data/data/com.termux/files/home, the shell with HOME=/root), and the
older legacy-HOME migration (``config.migrate_legacy_file``) still feeds into
the move. The path is resolved per instance via ``config.keys_file()`` — a
retargeted ``AGENT_CONFIG_FILE`` (tests, tools) keeps the old behaviour
exactly.

Two rules that nothing here may break:

* a raw key never reaches a log line or an HTTP response — every caller goes
  through :func:`mask` / :meth:`redact`;
* the file is rewritten atomically with 0600, so a half-written config can
  never be left world-readable (on a filesystem that cannot express unix
  modes — /storage/emulated/0 — the chmod is best-effort and logged, because
  losing the keys is worse than a loose mode bit).
"""
from __future__ import annotations

import json
import os
import secrets
import time
from pathlib import Path

import config
from core.data_version import warn_if_newer
from core.log import get_logger

log = get_logger("key_manager")

CONFIG_PATH = Path(config.AGENT_CONFIG_FILE)
FILE_MODE = 0o600

# On-disk schema version of agent_config.json, stamped on every save
# and checked on every load (see core/data_version.py). The version
# lives only in the file — the in-memory config never carries it.
DATA_VERSION = 1

# Set once a filesystem has refused a chmod (FUSE/sdcardfs on /storage does),
# so every later load does not log the same failure again.
_chmod_warned = False

# Set once per process after the first mode tightening, so the log is not
# flooded with the same warning on every request (FUSE keeps reporting
# 0660 even after a successful chmod).
_mode_warned = False

# A 429'd key sits out for a minute (or whatever Retry-After said) before the
# rotation is allowed to pick it again.
DEFAULT_COOLDOWN = 60.0
# A key that came back 401/403 is dead until the user re-adds it — retrying it
# just burns the provider's failure budget.
STATUS_DEAD = "dead"
STATUS_LIMITED = "rate_limited"
STATUS_OK = "ok"

PROVIDERS = ("openai", "gemini", "openrouter", "anthropic", "deepseek",
             "groq", "siliconflow", "custom")

# What the Settings dropdown shows. The value stays the provider id so every
# endpoint, config field and adapter key keeps the short name.
PROVIDER_LABELS = {
    "openai": "OpenAI",
    "gemini": "Google Gemini",
    "openrouter": "OpenRouter",
    "anthropic": "Anthropic",
    "deepseek": "DeepSeek",
    "groq": "Groq",
    "siliconflow": "SiliconFlow",
    "custom": "Custom (OpenAI-compatible)",
}

# How a turn is routed: auto = cloud when possible, local = never cloud,
# cloud = never silent local (the user asked for a provider explicitly).
MODES = ("auto", "local", "cloud")
DEFAULT_MODE = "auto"

DEFAULT_CONFIG: dict = {
    "version": 1,
    # auto | local | cloud — see MODES. `prefer_cloud` is kept alongside as
    # the boolean view older callers/tests already read (mode != "local").
    "backend_mode": DEFAULT_MODE,
    "prefer_cloud": True,
    # Provider order tried after each provider runs out of usable keys.
    "fallback": ["openai", "gemini", "openrouter", "anthropic", "deepseek",
                 "groq", "siliconflow", "custom"],
    # The UI must show the price once before a cloud model is used.
    "cost_ack": False,
    # Release 9 — provider switching policy. auto_switch True is the
    # historical behaviour: step() walks the fallback order and a failed
    # provider silently hands the turn to the next one. False is the
    # locked mode the user asked for: only `pinned_provider` may answer,
    # and when it cannot, the turn stops with the reason instead of
    # quietly switching models behind the user's back.
    "auto_switch": True,
    "pinned_provider": None,
    # prompt_tier_mode (auto | cloud_strong | cloud_fast | local) is deliberately
    # NOT in the defaults: an absent key has to stay absent, or the "auto"
    # default would look like a Settings choice and the
    # AI_AGENT_PROMPT_TIER environment fallback could never be reached. See
    # config.stored_prompt_tier_mode() and public() below.
    "providers": {
        "openai": {
            "model": "gpt-4o-mini",
            "base_url": "https://api.openai.com/v1",
            "enabled": True,
        },
        "gemini": {
            "model": "gemini-2.5-flash",
            "base_url": "https://generativelanguage.googleapis.com/v1beta",
            "enabled": True,
        },
        "openrouter": {
            "model": "google/gemini-2.5-flash",
            "base_url": "https://openrouter.ai/api/v1",
            "enabled": True,
        },
        "deepseek": {
            "model": "deepseek-chat",
            "base_url": "https://api.deepseek.com/v1",
            "enabled": True,
        },
        "anthropic": {
            "model": "claude-3-5-haiku-latest",
            "base_url": "https://api.anthropic.com/v1",
            "enabled": True,
        },
        # OpenAI-compatible services (Release 8). They ride the plain
        # OpenAI adapter; base_url + model are user-editable in Settings,
        # so this one `custom` slot can front any other compatible API.
        "groq": {
            # Release 26: evidence-based default (live Burmese battery,
            # 2026-10-23, the project owner's own Groq key): the account
            # lists no Llama chat models at all — qwen/qwen3.8-27b,
            # openai/gpt-oss-120b and openai/gpt-oss-20b are what it
            # serves — and qwen passed all five eval prompts (clean EV
            # explanation, correct change and sum-of-squares, natural
            # translations) at 0.8–1.5s per reply, where gpt-oss-20b
            # degenerated into a repetition loop. Fresh installs start
            # on qwen; existing configs keep the model they stored.
            "model": "qwen/qwen3.8-27b",
            "base_url": "https://api.groq.com/openai/v1",
            "enabled": True,
        },
        "siliconflow": {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "base_url": "https://api.siliconflow.com/v1",
            "enabled": True,
        },
        "custom": {
            "model": "",
            "base_url": "",
            "enabled": False,
        },
    },
    "keys": {"openai": [], "gemini": [], "openrouter": [],
             "anthropic": [], "deepseek": [], "groq": [],
             "siliconflow": [], "custom": []},
}


def chmod_0600(path: Path | str) -> bool:
    """Tighten ``path`` to 0600; False when the filesystem refuses.

    /storage/emulated/0 (FUSE/sdcardfs) has no unix mode bits and returns
    EPERM. Failing the surrounding save over that would lose the user's keys,
    so the failure is logged once and swallowed.
    """
    global _chmod_warned
    try:
        os.chmod(str(path), FILE_MODE)
        return True
    except OSError as e:
        if _chmod_warned:
            log.debug("chmod 0600 refused for %s: %s", path, e)
        else:
            _chmod_warned = True
            log.warning("cannot set 0600 on %s (%s: %s) — this filesystem "
                        "ignores unix modes; keys stay where they are written",
                        path, type(e).__name__, e)
        return False


def mask(key: str) -> str:
    """`sk-live-…a91c` — enough to tell keys apart, useless to an attacker."""
    if not key:
        return ""
    if len(key) <= 8:
        return "*" * len(key)
    return "%s…%s" % (key[:4], key[-4:])


def redact(text: str, keys: list[str] | None = None) -> str:
    """Replace every known raw key inside ``text`` with its mask.

    Providers echo keys back in error bodies (``Invalid API key: sk-…``), and
    those bodies are logged — so this runs before anything is written.
    """
    if not text:
        return text
    out = str(text)
    for k in keys or []:
        if k and k in out:
            out = out.replace(k, mask(k))
    return out


def _now() -> float:
    return time.time()


def _default_entry(key: str, label: str) -> dict:
    return {
        "id": "k_" + secrets.token_hex(4),
        "label": (label or "").strip()[:60],
        "key": key,
        "added": int(_now()),
        "status": STATUS_OK,
        "cooldown_until": 0.0,
        "last_error": None,
        "used": 0,
    }


class KeyManager:
    """Reads/writes agent_config.json and decides which key is next."""

    def __init__(self, path: Path | str | None = None):
        # Resolved per instance, not at import: the config paths are
        # re-pointed by the test harness and by any future config reload.
        # Default is config.keys_file() — the private location once the
        # keys have moved there, the project file while they have not.
        self.path = Path(path) if path else config.keys_file()
        self.data_version = DATA_VERSION
        self._cfg: dict | None = None
        self._cursor: dict = {}      # provider -> next index (round-robin)
        self.migrated_from: Path | None = None

    # ---------- file ----------
    def load(self) -> dict:
        if self._cfg is not None:
            return self._cfg
        # v1.9.0: move the keys out of shared storage into the private
        # location. One-time and verified; a failure simply leaves us
        # reading the project copy — never a reason to lose the keys.
        # migrate_keys_to_private() self-guards retargeted setups.
        if self.path != config.private_keys_file():
            try:
                moved = config.migrate_keys_to_private()
                if moved is not None:
                    self.migrated_from = moved
                    self.path = config.private_keys_file()
                    log.info("migrated %s from %s to %s",
                             self.path.name, moved, self.path)
            except Exception as e:
                log.warning("private keys migration failed: %s: %s",
                            type(e).__name__, e)
        if not self.path.exists():
            # First run after the path became absolute: pull the keys over
            # from whichever HOME they used to be written to.
            try:
                self.migrated_from = config.migrate_legacy_file(
                    self.path.name, self.path)
                if self.migrated_from is not None:
                    log.info("migrated %s from %s to %s",
                             self.path.name, self.migrated_from, self.path)
            except Exception as e:
                log.warning("legacy config migration failed: %s: %s",
                            type(e).__name__, e)
        raw = None
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    # The version belongs to the file, not the config:
                    # note it, warn when it is newer than this build,
                    # and keep it out of the merged config (and so out
                    # of every API response built from it).
                    self.data_version = warn_if_newer(
                        "agent_config.json", raw, DATA_VERSION, log)
                    raw = {k: v for k, v in raw.items() if k != "version"}
            except (OSError, json.JSONDecodeError) as e:
                # Never silently hand out defaults while a user's keys sit in
                # a broken file — keep the bytes for them and start fresh.
                side = self.path.with_suffix(".corrupt")
                try:
                    self.path.replace(side)
                    log.error("agent_config unreadable (%s); moved to %s",
                              type(e).__name__, side)
                except OSError:
                    log.error("agent_config unreadable: %s", e)
                raw = None
        cfg = _merge(DEFAULT_CONFIG, raw if isinstance(raw, dict) else {})
        self._cfg = cfg
        self._ensure_mode()
        return cfg

    def invalidate(self) -> None:
        """Drop the cached config so the next read comes from disk.

        Settings writes this file from a request while a long-lived backend
        holds an instance; without this, a mode switch (Cloud Only above all)
        would not be seen until the process restarts. The round-robin cursor
        is deliberately kept — only the parsed config is discarded.
        """
        self._cfg = None

    def set_mode(self, mode: str) -> str:
        """Set backend_mode and keep the boolean view consistent."""
        cfg = self.load()
        if mode not in MODES:
            raise ValueError("backend_mode must be one of %s" % (MODES,))
        cfg["backend_mode"] = mode
        cfg["prefer_cloud"] = (mode != "local")
        return mode

    def mode(self) -> str:
        """Resolved mode.

        `prefer_cloud=False` is the older, stricter switch: a config written
        before `backend_mode` existed (or by a caller that only knows the
        boolean) must keep meaning "never call out".
        """
        cfg = self.load()
        if not cfg.get("prefer_cloud", True):
            return "local"
        m = cfg.get("backend_mode")
        return m if m in MODES else DEFAULT_MODE

    def save(self) -> None:
        cfg = self.load()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(cfg)
        payload["version"] = DATA_VERSION
        data = json.dumps(payload, ensure_ascii=False, indent=2)
        # Unique temp name: Settings writes this file from several
        # request threads at once, and a shared `.tmp` made one
        # thread's os.replace() consume the file out from under
        # another (`FileNotFoundError: agent_config.tmp -> ...`).
        tmp = self.path.with_name(
            "%s.tmp.%s" % (self.path.name, secrets.token_hex(4)))
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, FILE_MODE)
        try:
            os.write(fd, data.encode("utf-8"))
        finally:
            os.close(fd)
        os.replace(str(tmp), str(self.path))
        # Best-effort: /storage/emulated/0 refuses chmod, and refusing to save
        # the user's keys over a mode bit would be the worse failure.
        chmod_0600(self.path)

    def reset_to_defaults(self) -> dict:
        """Restore every setting to DEFAULT_CONFIG — except the keys.

        The keys are the one thing a reset must never destroy: re-entering
        five providers' worth of secrets because a model name was mistyped
        is exactly the failure the Settings reset button exists to prevent.
        Returns the public view of the restored config.
        """
        cfg = self.load()
        keys = cfg.get("keys") or {}
        fresh = json.loads(json.dumps(DEFAULT_CONFIG))
        fresh["keys"] = {p: list(keys.get(p) or []) for p in PROVIDERS}
        self._cfg = fresh
        self.save()
        return self.public()

    def _ensure_mode(self) -> None:
        """chmod 600 every time the file is seen, even if it arrived loose."""
        global _mode_warned
        try:
            before = self.path.stat().st_mode & 0o777 if self.path.exists() else None
            if before is not None and before != FILE_MODE and chmod_0600(self.path):
                # /storage/emulated/0 always reports 0660, so this would
                # warn on every single request — once is enough.
                if not _mode_warned:
                    _mode_warned = True
                    log.warning("agent_config.json was mode %o — tightened to 0600",
                                before)
        except OSError as e:
            log.error("could not chmod agent_config.json: %s", e)

    # ---------- keys ----------
    def entries(self, provider: str) -> list[dict]:
        cfg = self.load()
        return list(cfg.setdefault("keys", {}).get(provider, []))

    def healthy(self, provider: str) -> list[dict]:
        now = _now()
        return [
            e for e in self.entries(provider)
            if e.get("status") != STATUS_DEAD
            and float(e.get("cooldown_until") or 0.0) <= now
        ]

    def next_key(self, provider: str) -> dict | None:
        usable = self.healthy(provider)
        if not usable:
            return None
        # A key the user pinned with set_active_key() wins over the rotation —
        # that is the point of selecting one. It only wins while it is still
        # usable: a dead or cooling key never blocks the turn.
        for e in usable:
            if e.get("active"):
                return e
        i = self._cursor.get(provider, 0) % len(usable)
        self._cursor[provider] = (i + 1) % len(usable)
        return usable[i]

    def active_entry(self, provider: str) -> dict | None:
        """The key this provider is currently on — what the UI ticks.

        The explicitly selected key first; otherwise the first healthy one,
        which is the key the rotation would hand out next. That keeps a
        config written before selection existed showing an honest tick.
        """
        bucket = self.entries(provider)
        for e in bucket:
            if e.get("active"):
                return e
        healthy = self.healthy(provider)
        if healthy:
            return healthy[0]
        return bucket[0] if bucket else None

    def set_active_key(self, provider: str, key_id: str) -> dict:
        """Pin one key for this provider (manual selection in Settings).

        Exactly one key per provider can be marked; the rotation cursor is
        reset so the next turn starts from the pinned key.
        """
        if provider not in PROVIDERS:
            raise ValueError("unknown provider: %s" % provider)
        cfg = self.load()
        bucket = cfg.setdefault("keys", {}).setdefault(provider, [])
        hit = None
        for e in bucket:
            on = (e.get("id") == key_id)
            if on:
                hit = e
            e["active"] = on
        if hit is None:
            raise KeyError("no such key: %s" % key_id)
        self._cursor[provider] = 0
        self.save()
        log.info("active key set: provider=%s id=%s", provider, key_id)
        return _public(hit, active=True)

    def add_key(self, provider: str, key: str, label: str = "") -> dict:
        key = (key or "").strip()
        if provider not in PROVIDERS:
            raise ValueError("unknown provider: %s" % provider)
        if len(key) < 8:
            raise ValueError("key too short")
        cfg = self.load()
        bucket = cfg.setdefault("keys", {}).setdefault(provider, [])
        # Same key added twice is still one key — rotation must not double it.
        for e in bucket:
            if e.get("key") == key:
                return _public(e, active=(e.get("id") ==
                                          (self.active_entry(provider) or {}).get("id")))
        first = not bucket
        entry = _default_entry(key, label)
        bucket.append(entry)
        self.save()
        # "active" is written only by set_active_key(): until the user picks
        # one, the rotation rotates (and active_entry() shows the healthy key
        # it would try first). Adding a key never silently pins it.
        return _public(entry, active=first)

    def remove_key(self, provider: str, key_id: str) -> bool:
        cfg = self.load()
        bucket = cfg.setdefault("keys", {}).get(provider, [])
        hit = None
        for e in bucket:
            if e.get("id") == key_id:
                hit = e
        keep = [e for e in bucket if e.get("id") != key_id]
        if len(keep) == len(bucket):
            return False
        cfg["keys"][provider] = keep
        if hit is not None and hit.get("active") and keep:
            # The pinned key just went away — promote the next healthy one so
            # the provider still has a key in play.
            nxt = next((e for e in keep if e.get("status") != STATUS_DEAD), keep[0])
            nxt["active"] = True
        self.save()
        return True

    # ---------- status transitions ----------
    def mark_ok(self, provider: str, key_id: str) -> None:
        self._patch(provider, key_id, status=STATUS_OK, cooldown_until=0.0,
                    last_error=None, used="+1")

    def mark_rate_limited(self, provider: str, key_id: str,
                          retry_after: float | None = None) -> float:
        until = _now() + (retry_after if retry_after and retry_after > 0
                          else DEFAULT_COOLDOWN)
        self._patch(provider, key_id, status=STATUS_LIMITED,
                    cooldown_until=until, last_error="429 rate limited")
        return until - _now()

    def mark_auth_failed(self, provider: str, key_id: str, error: str = "") -> None:
        self._patch(provider, key_id, status=STATUS_DEAD,
                    last_error=redact(error, self.all_keys())[:200])

    def _patch(self, provider: str, key_id: str, **fields) -> None:
        cfg = self.load()
        for e in cfg.setdefault("keys", {}).get(provider, []):
            if e.get("id") != key_id:
                continue
            for k, v in fields.items():
                if k == "used" and v == "+1":
                    e["used"] = int(e.get("used") or 0) + 1
                else:
                    e[k] = v
            self.save()
            return

    def all_keys(self, provider: str | None = None) -> list[str]:
        cfg = self.load()
        if provider:
            return [e.get("key", "") for e in self.entries(provider)]
        out = []
        for p in cfg.get("keys", {}):
            out.extend(e.get("key", "") for e in self.entries(p))
        return out

    # ---------- views ----------
    def public(self) -> dict:
        """Config minus raw keys — what /api/settings may return."""
        cfg = self.load()
        keys = {}
        for p, bucket in (cfg.get("keys") or {}).items():
            act = self.active_entry(p)
            keys[p] = [_public(e, active=bool(act) and e.get("id") == act.get("id"))
                       for e in bucket]
        return {
            "path": str(self.path),
            "mode": oct(self.path.stat().st_mode & 0o777) if self.path.exists() else None,
            "backend_mode": self.mode(),
            "prefer_cloud": bool(cfg.get("prefer_cloud", True)),
            "fallback": list(cfg.get("fallback") or []),
            "cost_ack": bool(cfg.get("cost_ack", False)),
            "auto_switch": bool(cfg.get("auto_switch", True)),
            "pinned_provider": cfg.get("pinned_provider") or None,
            # Report the stored value only. The environment fallback belongs to
            # config.prompt_tier_mode(), which routes/settings_api.py calls to
            # work out what this turn will actually use.
            "prompt_tier_mode": cfg.get("prompt_tier_mode") or "auto",
            "provider_labels": dict(PROVIDER_LABELS),
            "providers": {p: dict(v) for p, v in (cfg.get("providers") or {}).items()},
            "keys": keys,
        }

    def summary(self) -> dict:
        """Per-provider health for the Settings panel."""
        out = {}
        for p in PROVIDERS:
            entries = self.entries(p)
            healthy = self.healthy(p)
            act = self.active_entry(p)
            out[p] = {
                "total": len(entries),
                "healthy": len(healthy),
                "dead": sum(1 for e in entries if e.get("status") == STATUS_DEAD),
                "cooling": sum(1 for e in entries
                               if e.get("status") == STATUS_LIMITED),
                "active_id": (act or {}).get("id"),
                "keys": [_public(e, active=bool(act) and e.get("id") == act.get("id"))
                         for e in entries],
            }
        return out


def _public(entry: dict, active: bool = False) -> dict:
    return {
        "id": entry.get("id"),
        "label": entry.get("label") or "",
        "key_masked": mask(entry.get("key") or ""),
        "status": entry.get("status"),
        "cooldown_until": entry.get("cooldown_until"),
        "last_error": entry.get("last_error"),
        "used": entry.get("used") or 0,
        "added": entry.get("added"),
        # Is this the key the provider is on right now? Masked only — the raw
        # key never leaves this module.
        "active": bool(active),
    }


def _merge(base: dict, over: dict) -> dict:
    """deep-merge ``over`` onto ``base`` without ever returning the defaults
    object itself (a caller mutating the result must not corrupt the default)."""
    out = json.loads(json.dumps(base))
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out
