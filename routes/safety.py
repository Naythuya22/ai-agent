"""Thermal + idle safety guard (F4/F5).

Refuses LLM-heavy requests when the SoC is too hot, and hard-exits the
process after a long idle period so the phone is not left running.

`safety_guard` is registered app-wide via `routes.register()` — it must not
live in a blueprint `before_request`, which would only see its own routes.
"""
from __future__ import annotations

import os as _os_guard
import threading
import time
from pathlib import Path

from flask import Blueprint, jsonify, request

from core import net
from core.log import get_logger

log = get_logger("routes.safety")

bp = Blueprint("safety", __name__)

_THERMAL_LIMIT_C = float(_os_guard.environ.get("AI_AGENT_THERMAL_LIMIT", "45"))
_IDLE_TIMEOUT_S = int(_os_guard.environ.get("AI_AGENT_IDLE_TIMEOUT", "1800"))  # 30 min
_SAFETY_GUARD_ENABLED = _os_guard.environ.get("AI_AGENT_SAFETY_GUARD", "1") != "0"

_last_activity = time.time()
_activity_lock = threading.Lock()

# Endpoints NOT throttled by thermal guard
_GUARD_SKIP_PREFIXES = (
    "/api/health",
    "/api/thermal",
    "/api/sw-version",
    "/api/mode",
    "/api/net",
    "/sw.js",
    "/manifest.json",
    "/icons/",
    "/static/",
)

# Endpoints that count as LLM "activity" (for idle shutdown)
_LLM_PATH_PREFIXES = (
    "/api/chat",
    "/api/agent",
    "/api/skills/",
    "/api/code",
    "/v1/chat/completions",
)


def get_cpu_temp():
    """Read CPU temperature. Returns (temp_c, zone_name) or (None, None)."""
    try:
        base = Path("/sys/class/thermal")
        if not base.is_dir():
            return None, None
        priority = ("quiet_therm", "xo-therm", "conn_therm", "battery", "cpu_therm")
        candidates = []
        for z in sorted(base.glob("thermal_zone*")):
            try:
                name = (z / "type").read_text(encoding="utf-8").strip()
                raw = (z / "temp").read_text(encoding="utf-8").strip()
                temp_c = int(raw) / 1000.0
                if temp_c < 0 or temp_c > 150:
                    continue
                candidates.append((name, temp_c))
            except Exception:
                continue
        for p in priority:
            for name, t in candidates:
                if name == p:
                    return t, name
        if candidates:
            name, t = max(candidates, key=lambda kv: kv[1])
            return t, name
    except Exception as e:
        # A missing reading means "no thermal data" — callers degrade to
        # allowing the request, so this must not raise.
        log.debug("could not read a CPU temperature source: %s: %s",
                  type(e).__name__, e)
    return None, None


def _touch_activity():
    global _last_activity
    with _activity_lock:
        _last_activity = time.time()


def _seconds_since_activity():
    with _activity_lock:
        return time.time() - _last_activity


def _is_llm_path(path):
    for p in _LLM_PATH_PREFIXES:
        if path.startswith(p):
            return True
    return False


def _is_guard_skip(path):
    for p in _GUARD_SKIP_PREFIXES:
        if path.startswith(p):
            return True
    return False


def _idle_watchdog():
    """F5: background thread — hard exit if idle too long."""
    while True:
        try:
            time.sleep(60)
            if _IDLE_TIMEOUT_S <= 0:
                continue
            idle = _seconds_since_activity()
            if idle >= _IDLE_TIMEOUT_S:
                print(f"\n[F5] Idle {int(idle)}s — auto-shutdown", flush=True)
                _os_guard._exit(0)
        except Exception:
            continue


def start_idle_watchdog():
    if not _SAFETY_GUARD_ENABLED:
        print("  [SafetyGuard] DISABLED via env", flush=True)
        return
    if _IDLE_TIMEOUT_S <= 0:
        print("  [SafetyGuard] idle shutdown off", flush=True)
        return
    t = threading.Thread(target=_idle_watchdog, daemon=True, name="idle-watchdog")
    t.start()
    print(f"  [SafetyGuard] thermal>{_THERMAL_LIMIT_C}°C refuse | idle>{_IDLE_TIMEOUT_S}s exit", flush=True)


def safety_guard():
    """F4: refuse LLM requests if CPU is too hot. F5: touch activity on LLM paths."""
    if not _SAFETY_GUARD_ENABLED:
        return None
    path = request.path or ""
    if _is_guard_skip(path):
        return None
    # Only guard LLM-heavy paths
    if _is_llm_path(path):
        temp_c, zone = get_cpu_temp()
        if temp_c is not None and temp_c >= _THERMAL_LIMIT_C:
            return jsonify({
                "error": "thermal_throttle",
                "message": (
                    f"Phone too hot ({temp_c:.1f}°C > {_THERMAL_LIMIT_C}°C). "
                    f"Please wait for it to cool down."
                ),
                "temp_c": round(temp_c, 1),
                "zone": zone,
                "limit_c": _THERMAL_LIMIT_C,
            }), 503
        _touch_activity()
    return None


@bp.route("/api/thermal", methods=["GET"])
def api_thermal():
    """F4: report current CPU temp + guard status."""
    temp_c, zone = get_cpu_temp()
    return jsonify({
        "temp_c": round(temp_c, 1) if temp_c is not None else None,
        "zone": zone,
        "limit_c": _THERMAL_LIMIT_C,
        "over_limit": (temp_c is not None and temp_c >= _THERMAL_LIMIT_C),
        "idle_seconds": round(_seconds_since_activity(), 1),
        "idle_timeout_s": _IDLE_TIMEOUT_S,
        "guard_enabled": _SAFETY_GUARD_ENABLED,
    })


@bp.route("/api/net", methods=["GET"])
def api_net():
    """Report internet reachability so the UI can show online/offline.

    Pass ?refresh=1 to force a re-probe instead of using the 30s cache.
    """
    force = (request.args.get("refresh") or "").strip() in ("1", "true", "yes")
    return jsonify(net.describe(force=force))
