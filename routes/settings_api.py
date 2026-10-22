"""Settings endpoints for the agent's hybrid backend.

Agent Mode only: which cloud provider/model to use, how many keys to rotate
through, and whether a turn should go to the cloud at all. Chat Mode has no
endpoints here and never reads this config.

No response from this file ever contains a raw key — `KeyManager.public()` /
`_public()` hand back `key_masked` only, and error strings pass through
`redact()` before they are serialised.
"""
from __future__ import annotations

import shutil

from flask import Blueprint, jsonify, request
from werkzeug.exceptions import HTTPException

import config
from core.agent_backend import (ADAPTERS, MODELS, AgentBackend, cost_per_1k,
                               friendly_local_name)
from core.key_manager import MODES, PROVIDERS, KeyManager, redact
from core.log import get_logger
from core.prompts import classify_tier, describe, plan_target
from routes._helpers import json_body

log = get_logger("routes.settings")

bp = Blueprint("settings_api", __name__)


@bp.errorhandler(HTTPException)
def _settings_http_error(exc):
    """405/415/… raised inside this blueprint leave as JSON, never HTML."""
    return jsonify({"error": exc.description or exc.name}), (exc.code or 500)


@bp.errorhandler(Exception)
def _settings_unhandled(exc):
    log.error("settings error: %s", exc, exc_info=True)
    return jsonify({"error": "%s: %s" % (type(exc).__name__, exc)}), 500

MAX_KEYS = 20


def _keys() -> KeyManager:
    return KeyManager()


def _backend() -> AgentBackend:
    return AgentBackend(keys=_keys())


def _err(msg: str, status: int = 400):
    return jsonify({"error": redact(str(msg), _keys().all_keys())}), status


def _labels() -> dict:
    from core.key_manager import PROVIDER_LABELS
    return dict(PROVIDER_LABELS)


def _valid_url(url: str) -> bool:
    return url.startswith("http://") or url.startswith("https://")


def _prompt_tier_view(be) -> dict:
    """What the Prompt Tier panel shows about the *next* Agent turn.

    Reads the backend for what it would do and never changes it. `backend` is
    the extra field the UI needs to say "you pinned a cloud tier but this turn
    is running on the phone": without it a forced tier and the active backend
    cannot be compared, because the resolved tier *is* the forced one by
    definition.
    """
    backend, model = "local", friendly_local_name() or ""
    try:
        backend, model = plan_target(be)
    except Exception as e:
        log.warning("prompt tier view failed: %s: %s", type(e).__name__, e)
    view = describe(backend, model)
    view["backend"] = backend
    # What the classifier would have picked on its own — the comparison the
    # "pinned tier does not match this turn" badge is built from.
    view["auto"] = classify_tier(backend, model)
    view["mismatch"] = bool(view["mode"] != "auto"
                            and view["mode"] != view["auto"])
    return view


@bp.route("/api/settings/agent", methods=["GET"])
def api_settings_get():
    km = _keys()
    try:
        cfg = km.public()
        be = _backend()
        try:
            status = be.status()
        except Exception as e:
            log.warning("backend status failed: %s: %s", type(e).__name__, e)
            status = {"online": False, "error": str(e)}
        costs = {p: cost_per_1k(p, (cfg["providers"].get(p) or {}).get("model", ""))
                 for p in PROVIDERS}
        return jsonify({
            "ok": True,
            "config": cfg,
            # Fixed, HOME-independent location of the key file. Surfaced so a
            # "keys vanished after restart" report can be checked against the
            # real path instead of guessing which $HOME the server ran under.
            "config_path": str(km.path),
            "config_exists": km.path.exists(),
            "backend": status,
            "costs": costs,
            "providers": list(PROVIDERS),
            "provider_labels": _labels(),
            "modes": list(MODES),
            "models": MODELS,
            "fallback_order": be.fallback_order(),
            "local_fallback": be.fallback_to_local,
            # Which system prompt the next Agent turn would use, and why.
            "prompt_tier": _prompt_tier_view(be),
            "prompt_tier_modes": list(config.PROMPT_TIER_MODES),
        })
    except Exception as e:
        return _err("%s: %s" % (type(e).__name__, e), 500)


@bp.route("/api/settings/agent", methods=["POST"])
def api_settings_post():
    body = json_body()
    km = _keys()
    cfg = km.load()
    changed = []

    if "backend_mode" in body:
        try:
            km.set_mode(str(body["backend_mode"] or "").strip())
        except ValueError as e:
            return _err(str(e))
        changed.append("backend_mode")
    elif "prefer_cloud" in body:
        # Older callers only know the boolean — keep the two in step.
        try:
            km.set_mode("auto" if body["prefer_cloud"] else "local")
        except ValueError as e:
            return _err(str(e))
        changed.append("prefer_cloud")
    if "cost_ack" in body:
        cfg["cost_ack"] = bool(body["cost_ack"])
        changed.append("cost_ack")

    if "prompt_tier_mode" in body:
        mode = str(body["prompt_tier_mode"] or "").strip()
        if mode not in config.PROMPT_TIER_MODES:
            return _err("prompt_tier_mode must be one of %s"
                        % (config.PROMPT_TIER_MODES,))
        cfg["prompt_tier_mode"] = mode
        changed.append("prompt_tier_mode")

    if "fallback" in body:
        fb = body["fallback"]
        if not isinstance(fb, list) or not all(isinstance(p, str) for p in fb):
            return _err("fallback must be a list of provider names")
        fb = [p for p in fb if p in PROVIDERS]
        seen, ordered = set(), []
        for p in fb:                      # dedupe, keep order
            if p not in seen:
                seen.add(p)
                ordered.append(p)
        for p in PROVIDERS:               # fill gaps so no provider is lost
            if p not in ordered:
                ordered.append(p)
        cfg["fallback"] = ordered
        changed.append("fallback")

    if "providers" in body:
        incoming = body["providers"]
        if not isinstance(incoming, dict):
            return _err("providers must be an object")
        for p, vals in incoming.items():
            if p not in PROVIDERS:
                return _err("unknown provider: %s" % p)
            if not isinstance(vals, dict):
                return _err("providers.%s must be an object" % p)
            cur = cfg.setdefault("providers", {}).setdefault(p, {})
            if "model" in vals:
                model = str(vals["model"] or "").strip()
                if len(model) > 80:
                    return _err("model name too long")
                cur["model"] = model
            if "enabled" in vals:
                cur["enabled"] = bool(vals["enabled"])
            if "base_url" in vals:
                url = str(vals["base_url"] or "").strip().rstrip("/")
                if url and not _valid_url(url):
                    return _err("base_url must start with http:// or https://")
                cur["base_url"] = url
            changed.append("providers.%s" % p)

    try:
        km.save()
    except Exception as e:
        return _err("could not save: %s" % e, 500)
    if not km.path.exists():
        return _err("settings were not written to %s" % km.path, 500)
    log.info("agent settings updated: %s (config at %s)",
             ", ".join(changed) or "nothing", km.path)
    return jsonify({"ok": True, "changed": changed,
                    "config": km.public(),
                    "backend": _safe_status(),
                    # Re-read from disk: a tier pinned here has to be reflected
                    # in the panel without a second GET.
                    "prompt_tier": _prompt_tier_view(_backend())})


@bp.route("/api/settings/model", methods=["GET"])
def api_model_info():
    """Which model the *next* turn would use — a prediction read from config.

    The status-bar chip no longer uses this: it shows what actually ran
    (/api/agent/status plus the SSE `end` event), because a Cloud Only
    setting with no usable key used to print a cloud model name that nothing
    ever called.

    Chat Mode is always local and never reads this config, so a `mode=chat`
    request (or the default when the caller is unsure) always answers the
    local label.
    """
    which = (request.args.get("mode") or "agent").lower()
    km = _keys()
    try:
        cfg = km.load()
        be = _backend()
        mode = be.mode
        usable = be.usable()
        order = be.fallback_order()
        use_cloud = (which == "agent" and mode == "cloud") or (
            which == "agent" and mode == "auto" and be.should_use_cloud())
        local = friendly_local_name() or "Local model"
        if not use_cloud:
            return jsonify({
                "ok": True,
                "label": "Model: %s (Local)" % local,
                "model": local,
                "backend": "local",
                "provider": "",
                "mode": mode,
                "cloud": False,
            })
        provider = (usable or order or [""])[0]
        prov_cfg = (cfg.get("providers") or {}).get(provider) or {}
        model = (prov_cfg.get("model") or (MODELS.get(provider) or [""])[0] or "")
        return jsonify({
            "ok": True,
            "label": "Model: %s (Cloud)" % (model or provider),
            "model": model,
            "backend": "cloud",
            "provider": provider,
            "mode": mode,
            "cloud": True,
            "usable": bool(usable),
            "cost_per_1k": cost_per_1k(provider, model),
        })
    except Exception as e:
        log.warning("model info failed: %s: %s", type(e).__name__, e)
        local = friendly_local_name() or "Local model"
        return jsonify({
            "ok": True,
            "label": "Model: %s (Local)" % local,
            "model": local,
            "backend": "local",
            "provider": "",
            "mode": km.mode(),
            "cloud": False,
            "error": str(e),
        })


@bp.route("/api/settings/agent/keys", methods=["POST"])
def api_key_add():
    body = json_body()
    provider = str(body.get("provider") or "").strip()
    key = str(body.get("key") or "").strip()
    label = str(body.get("label") or "").strip()
    if provider not in PROVIDERS:
        return _err("unknown provider: %s" % provider)
    if not key:
        return _err("key is required")
    km = _keys()
    if len(km.entries(provider)) >= MAX_KEYS:
        return _err("too many keys for %s (max %d)" % (provider, MAX_KEYS))
    try:
        entry = km.add_key(provider, key, label)
    except ValueError as e:
        return _err(str(e))
    except Exception as e:
        return _err("could not write key: %s" % e, 500)
    if not km.path.exists():
        # Test OK but the key is not on disk — the restart bug in disguise.
        return _err("key was accepted but not persisted to %s" % km.path, 500)
    log.info("key added: provider=%s id=%s (%s)",
             provider, entry.get("id"), km.path)
    return jsonify({"ok": True, "key": entry, "config": km.public(),
                    "config_path": str(km.path)})


@bp.route("/api/settings/agent/keys/<provider>/<key_id>", methods=["DELETE"])
def api_key_delete(provider: str, key_id: str):
    if provider not in PROVIDERS:
        return _err("unknown provider: %s" % provider)
    km = _keys()
    if not km.remove_key(provider, key_id):
        return _err("no such key", 404)
    log.info("key removed: provider=%s id=%s", provider, key_id)
    return jsonify({"ok": True, "config": km.public()})


@bp.route("/api/settings/agent/keys/<provider>/<key_id>/activate",
          methods=["POST"])
def api_key_activate(provider: str, key_id: str):
    """Pin one key: every later turn on this provider starts with it.

    Manual selection — the rotation still takes over if the pinned key goes
    dead or hits a 429, so a bad choice can never strand the provider.
    """
    if provider not in PROVIDERS:
        return _err("unknown provider: %s" % provider)
    km = _keys()
    try:
        entry = km.set_active_key(provider, key_id)
    except KeyError:
        return _err("no such key", 404)
    except ValueError as e:
        return _err(str(e))
    except Exception as e:
        return _err("could not save active key: %s" % e, 500)
    log.info("key activated: provider=%s id=%s", provider, key_id)
    return jsonify({"ok": True, "key": entry, "config": km.public()})


@bp.route("/api/settings/agent/keys/test", methods=["POST"])
def api_key_test():
    """Prove a key works with one cheap round-trip (max_tokens=8)."""
    body = json_body()
    provider = str(body.get("provider") or "").strip()
    model = str(body.get("model") or "").strip()
    if provider not in PROVIDERS:
        return _err("unknown provider: %s" % provider)
    km = _keys()
    key = str(body.get("key") or "").strip()
    if not key and body.get("key_id"):
        for e in km.entries(provider):
            if e.get("id") == body.get("key_id"):
                key = e.get("key", "")
                break
        if not key:
            return _err("no such key", 404)
    if not key:
        return _err("key is required")
    result = _backend().test_key(provider, key, model)
    return jsonify(result)


@bp.route("/api/settings/agent/reset", methods=["POST"])
def api_settings_reset():
    """Restore the factory settings — mode, fallback order, provider
    models/URLs — while keeping every stored API key.

    This is the self-service recovery path: any Settings value a user can
    mistype can be undone in one tap, and the secrets they cannot retype
    from memory are never part of the blast radius.
    """
    km = _keys()
    try:
        pub = km.reset_to_defaults()
    except Exception as e:
        return _err("could not reset: %s" % e, 500)
    log.info("agent settings reset to defaults (keys kept, config at %s)",
             km.path)
    return jsonify({"ok": True, "config": pub,
                    "backend": _safe_status(),
                    "prompt_tier": _prompt_tier_view(_backend())})


@bp.route("/api/settings/doctor", methods=["GET"])
def api_settings_doctor():
    """One-tap system check for the Settings panel.

    Read-only roll-up of what a user otherwise has to verify in a terminal:
    backend state, per-provider key health (`KeyManager.summary()` — written
    for this panel and until now never wired to a route), whether the Code
    mode backend is installed, and free space where the config lives. No raw
    key is ever part of the answer.
    """
    km = _keys()
    out = {
        "ok": True,
        "config_path": str(km.path),
        "config_exists": km.path.exists(),
        "backend": _safe_status(),
        "keys": km.summary(),
        "disk_free_bytes": None,
        "code": None,
    }
    try:
        out["disk_free_bytes"] = shutil.disk_usage(str(km.path.parent)).free
    except Exception as e:
        out["disk_error"] = str(e)
    try:
        from core import code_backends
        b = code_backends.get("opencode")
        info = b.doctor()
        out["code"] = {
            "backend": "opencode",
            "installed": bool(info.get("installed")),
            "version": info.get("version"),
        }
    except Exception as e:
        out["code"] = {"backend": "opencode", "installed": False,
                       "error": str(e)}
    return jsonify(out)


@bp.route("/api/settings/backup", methods=["POST"])
def api_settings_backup():
    """Export a full backup zip from the Settings panel.

    Same engine as `python3 backup.py export` (Release 1): the zip carries
    the provider keys, so the answer reports that fact and the UI warns —
    the file itself stays in the app home, never in the response.
    """
    try:
        import backup as backup_mod
        path = backup_mod.do_export()
        man = backup_mod.read_manifest(path) or {}
    except Exception as e:
        return _err("backup failed: %s" % e, 500)
    log.info("settings backup exported: %s", path)
    return jsonify({"ok": True, "file": path.name,
                    "files": len(man.get("files") or []),
                    "includes_secrets": bool(man.get("includes_secrets"))})


def _safe_status() -> dict:
    try:
        return _backend().status()
    except Exception as e:
        return {"online": False, "error": str(e)}
