"""Shared-secret auth for the HTTP API (Session 23, Q2/Q3).

Enforcement is opt-in so the test suite is untouched:

  - ``AI_AGENT_TOKEN`` env is set        → enforced
  - the token file exists                → enforced (``main()`` writes it
    on first start and prints a ``?token=`` URL)
  - neither                              → the guard is a no-op

A request is accepted when it carries the token in any of:
  ``X-Agent-Token: <t>`` header, ``Authorization: Bearer <t>`` (for /v1
  clients), or ``?token=<t>`` (EventSource-style / opened-from-URL loads).
"""
from __future__ import annotations

import hmac
import os
import secrets
from pathlib import Path

from flask import jsonify, request

from core.log import get_logger

log = get_logger("auth")

ENV_TOKEN = "AI_AGENT_TOKEN"
FILE_NAME = "auth_token"

# PWA shell + read-only probes the page loads before it can hold a token.
EXEMPT_EXACT = {
    "/", "/index.html", "/manifest.json", "/sw.js",
    "/api/health", "/api/thermal", "/api/sw-version",
    "/api/net", "/api/mode",
}
EXEMPT_PREFIX = ("/static/", "/icons/")

UNAUTH_BODY = {
    "error": "unauthorized",
    "hint": "send X-Agent-Token header, Authorization: Bearer, or ?token=",
}


def _token_file() -> Path:
    """config.CONFIG_DIR ကို ပဲ သုံး (test HOME retarget ကို လိုက်နိုင်အောင်)။"""
    try:
        import config
        base = Path(config.CONFIG_DIR)
    except Exception:
        base = Path(os.environ.get("HOME", str(Path.home()))) / ".ai-agent"
    return base / FILE_NAME


def configured_token() -> str | None:
    """The token in force: env first, then the persisted file."""
    tok = (os.environ.get(ENV_TOKEN) or "").strip()
    if tok:
        return tok
    try:
        tok = _token_file().read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return tok or None


def generate_token() -> str:
    """Create a token, persist it (0600) and return it."""
    tok = secrets.token_urlsafe(24)
    path = _token_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(tok + "\n", encoding="utf-8")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    except OSError as e:
        log.warning("could not persist the auth token: %s: %s",
                    type(e).__name__, e)
    return tok


def _provided() -> str:
    header = (request.headers.get("X-Agent-Token") or "").strip()
    if header:
        return header
    auth = (request.headers.get("Authorization") or "").strip()
    if auth[:7].lower() == "bearer ":
        return auth[7:].strip()
    return (request.args.get("token") or "").strip()


def _exempt(path: str) -> bool:
    if path in EXEMPT_EXACT:
        return True
    return any(path.startswith(p) for p in EXEMPT_PREFIX)


def auth_guard():
    """before_request — 401 unless a valid token arrives (opt-in)."""
    expected = configured_token()
    if not expected:
        return None
    path = request.path or ""
    if _exempt(path):
        return None
    provided = _provided()
    if provided and hmac.compare_digest(provided, expected):
        return None
    return jsonify(UNAUTH_BODY), 401
