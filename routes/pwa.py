"""PWA + static asset routes — served from `web/`."""
from __future__ import annotations

import hashlib
from pathlib import Path

from flask import Blueprint, Response, jsonify, send_from_directory

import config
from core.log import get_logger

log = get_logger("routes.pwa")

bp = Blueprint("pwa", __name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
INDEX_HTML = WEB_DIR / "index.html"

# Files whose presence the service worker covers — reported by
# /api/sw-version so a client can see what a version bump is about.
_VERSION_FILES = ("index.html", "sw.js", "manifest.json")

# Placeholder inside web/sw.js the served copy is rewritten with.
_CACHE_VERSION_TOKEN = "__CACHE_VERSION__"

_HASH_HEX_CHARS = 8


def _index_hash() -> str:
    """First 8 hex chars of sha256(web/index.html).

    Content-based, so a deploy that rewrites index.html always changes the
    SW cache name — unlike mtime sums, which two identical-but-rebuilt trees
    can share. "dev" only when the file is unreadable.
    """
    try:
        digest = hashlib.sha256(INDEX_HTML.read_bytes()).hexdigest()
        return digest[:_HASH_HEX_CHARS]
    except OSError as e:
        log.warning("could not hash %s: %s: %s", INDEX_HTML, type(e).__name__, e)
        return "dev"


# Computed once at import (server startup) and reused per request.
_STARTUP_HASH = _index_hash()


def _cache_version() -> str:
    """Version injected into the CACHE name of the served sw.js.

    Production serves the startup snapshot (one file read per boot).
    AI_AGENT_DEBUG=1 re-reads index.html per request so dev edits
    propagate without a restart.
    """
    return _index_hash() if config.DEBUG else _STARTUP_HASH


@bp.route("/", methods=["GET"])
def root():
    resp = send_from_directory(WEB_DIR, "index.html")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@bp.route("/api/sw-version", methods=["GET"])
def api_sw_version():
    """Return the cache-busting version injected into /sw.js.

    Client registers SW with ?v=<this> — when index.html changes, the hash
    changes, sw.js content changes, browser detects a new SW, update flow
    triggers. Must stay in lockstep with serve_sw() or the browser would
    register a cache name the server never serves.
    """
    try:
        return jsonify({"version": _cache_version(), "files": list(_VERSION_FILES)})
    except Exception as e:
        return jsonify({"version": "dev", "error": str(e)}), 500


@bp.route("/manifest.json", methods=["GET"])
def serve_manifest():
    """PWA manifest — must be at root for correct scope."""
    resp = send_from_directory(WEB_DIR, "manifest.json")
    resp.headers["Content-Type"] = "application/manifest+json; charset=utf-8"
    resp.headers["Cache-Control"] = "public, max-age=3600"
    return resp


@bp.route("/sw.js", methods=["GET"])
def serve_sw():
    """Service worker — served with a content-hash-injected CACHE name (B1).

    web/sw.js is a template: its `__CACHE_VERSION__` token is replaced with
    sha256(index.html)[:8]. Any deploy that rewrites index.html therefore
    changes sw.js itself, so the browser installs a new worker, the activate
    handler drops the stale caches, and index.html is no longer served from
    the previous deploy's cache.
    """
    sw_path = WEB_DIR / "sw.js"
    try:
        content = sw_path.read_text(encoding="utf-8")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    version = _cache_version()
    if _CACHE_VERSION_TOKEN in content:
        content = content.replace(_CACHE_VERSION_TOKEN, version)
    else:
        # A missing token would ship a literal '__CACHE_VERSION__' cache name
        # and silently resurrect B1; make it loud instead.
        log.warning("%s has no %s placeholder", sw_path, _CACHE_VERSION_TOKEN)

    resp = Response(content, mimetype="application/javascript; charset=utf-8")
    resp.headers["Service-Worker-Allowed"] = "/"
    # no-cache: the browser must revalidate sw.js on every navigation, so a
    # deploy is noticed without the user clearing site data.
    resp.headers["Cache-Control"] = "no-cache, must-revalidate"
    return resp


@bp.route("/icons/<path:filename>", methods=["GET"])
def serve_icon(filename):
    """PWA icons."""
    resp = send_from_directory(WEB_DIR / "icons", filename)
    resp.headers["Content-Type"] = "image/png"
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


@bp.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({"status": "ok", "service": "litert-openai-bridge"})
