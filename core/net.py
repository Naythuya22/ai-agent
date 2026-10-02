"""Internet connectivity probe with a short TTL cache.

Only the search tools need the internet — the LLM runs on-device and the
OpenAI bridge is on 127.0.0.1, so chat/agent/code keep working with no
network at all. This module exists so the network tools can fail *fast*
instead of burning their full timeout against a dead link.

Tri-state on purpose:
  ONLINE  - a probe connected
  OFFLINE - a probe failed
  UNKNOWN - never probed, or sockets are unavailable on this host

`is_online()` fails OPEN: only a confirmed OFFLINE returns False, so a
broken or blocked probe can never disable a working network.
"""
from __future__ import annotations

import os
import socket
import threading
import time

ONLINE = "online"
OFFLINE = "offline"
UNKNOWN = "unknown"

# Cheap TCP targets, plain IPs first so the probe works without DNS.
TARGETS = (
    ("1.1.1.1", 443),
    ("8.8.8.8", 53),
    ("9.9.9.9", 443),
)

PROBE_TIMEOUT = float(os.environ.get("AI_AGENT_NET_PROBE_TIMEOUT", "1.5"))
CACHE_TTL = float(os.environ.get("AI_AGENT_NET_CACHE_TTL", "30"))

_lock = threading.Lock()
_state = UNKNOWN
_checked_mono = 0.0
_checked_wall = 0.0
_latency_ms = None
_last_error = None


def _probe() -> tuple[bool, float | None, str | None]:
    """Try each target in order. Returns (ok, latency_ms, last_error)."""
    err = None
    for host, port in TARGETS:
        t0 = time.monotonic()
        try:
            with socket.create_connection((host, port), timeout=PROBE_TIMEOUT):
                return True, (time.monotonic() - t0) * 1000.0, None
        except OSError as e:
            err = f"{type(e).__name__}: {e}"
    return False, None, err


def state(force: bool = False) -> str:
    """Return ONLINE / OFFLINE / UNKNOWN, refreshing the cache when stale."""
    global _state, _checked_mono, _checked_wall, _latency_ms, _last_error
    with _lock:
        fresh = _checked_mono and (time.monotonic() - _checked_mono) < CACHE_TTL
        if not force and fresh and _state != UNKNOWN:
            return _state
        try:
            ok, latency, err = _probe()
        except Exception as e:  # probe itself unusable -> fail open
            _state = UNKNOWN
            _checked_mono = time.monotonic()
            _checked_wall = time.time()
            _latency_ms = None
            _last_error = f"probe unavailable: {type(e).__name__}: {e}"
            return _state
        _state = ONLINE if ok else OFFLINE
        _checked_mono = time.monotonic()
        _checked_wall = time.time()
        _latency_ms = round(latency, 1) if latency is not None else None
        _last_error = err
        return _state


def is_online(force: bool = False) -> bool:
    """True unless we have a confirmed OFFLINE. Fails open."""
    return state(force=force) != OFFLINE


def describe(force: bool = False) -> dict:
    """Serializable snapshot for the UI / /api/net."""
    st = state(force=force)
    age = round(time.monotonic() - _checked_mono, 1) if _checked_mono else None
    return {
        "state": st,
        "online": st != OFFLINE,
        "latency_ms": _latency_ms,
        "age_s": age,
        "checked_at": _checked_wall or None,
        "last_error": _last_error,
    }


def reset() -> None:
    """Drop the cache so the next call re-probes. For tests."""
    global _state, _checked_mono, _checked_wall, _latency_ms, _last_error
    with _lock:
        _state = UNKNOWN
        _checked_mono = 0.0
        _checked_wall = 0.0
        _latency_ms = None
        _last_error = None


def _is_burmese(text: str) -> bool:
    return any("\u1000" <= ch <= "\u109F" for ch in (text or ""))


def offline_notice(tool: str, query: str = "") -> str | None:
    """None when online, else a notice for the model to act on.

    This text is tool output, so it is written for the model: state the
    fact, forbid a retry, and tell it to answer from its own knowledge and
    say plainly that it could not verify online.
    """
    if is_online():
        return None
    subject = f" for {query!r}" if query else ""
    if _is_burmese(query):
        shown = "အင်တာနက် မရှိလို့ မစစ်နိုင်ပါ။"
    else:
        shown = "I don't have an internet connection right now."
    return (
        f"OFFLINE — {tool} could not run{subject}: no internet connection.\n"
        "Do NOT retry this tool and do NOT call any other network tool.\n"
        "Answer from your own knowledge instead, and tell the user plainly "
        "that you could not check online.\n"
        f"User-facing line: {shown}"
    )


__all__ = [
    "ONLINE", "OFFLINE", "UNKNOWN", "state", "is_online", "describe",
    "offline_notice", "reset", "PROBE_TIMEOUT", "CACHE_TTL",
]
