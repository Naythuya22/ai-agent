"""PWA + static asset routes — served from `web/`."""
from __future__ import annotations

from pathlib import Path

from flask import Blueprint, Response, jsonify, send_from_directory

from core.log import get_logger

log = get_logger("routes.pwa")

bp = Blueprint("pwa", __name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# Files whose mtimes make up the service-worker cache-bust version.
_VERSION_FILES = ("index.html", "sw.js", "manifest.json")


def _version_sum():
    """Sum of mtimes over the tracked web files, plus whether any was found.

    The two callers format the empty case differently (sw-version renders
    "0", sw.js renders "dev"), so both facts are returned rather than
    collapsed into one string here.
    """
    mtimes = []
    for name in _VERSION_FILES:
        p = WEB_DIR / name
        try:
            if p.is_file():
                mtimes.append(int(p.stat().st_mtime))
        except Exception as e:
            # A missing file just renders "dev"/"0"; only a failing stat on
            # an existing file would hide real trouble.
            log.debug("could not stat %s: %s: %s", p, type(e).__name__, e)
    return (sum(mtimes) if mtimes else 0), bool(mtimes)


@bp.route("/", methods=["GET"])
def root():
    resp = send_from_directory(WEB_DIR, "index.html")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@bp.route("/api/sw-version", methods=["GET"])
def api_sw_version():
    """Return a version string based on web/ files mtime.

    Client registers SW with ?v=<this> — when files change, URL changes,
    browser detects new SW, update flow triggers.
    """
    try:
        total, _found = _version_sum()
        return jsonify({"version": f"{total:x}", "files": list(_VERSION_FILES)})
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
    """Service worker — served with dynamic version-injected CACHE name (PWAU2b).

    The `CACHE` constant in sw.js is rewritten with a version hash derived
    from web/ files mtime. When any web file changes, sw.js content changes,
    the browser detects a new SW, and the update flow triggers.
    """
    import re as _re
    sw_path = WEB_DIR / "sw.js"
    try:
        content = sw_path.read_text(encoding="utf-8")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    total, found = _version_sum()
    version = f"{total:x}" if found else "dev"

    # Inject version into the placeholder CACHE declaration
    content = _re.sub(
        r"const CACHE = '[^']*';",
        f"const CACHE = 'ai-agent-{version}';",
        content,
        count=1,
    )

    resp = Response(content, mimetype="application/javascript; charset=utf-8")
    resp.headers["Service-Worker-Allowed"] = "/"
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
