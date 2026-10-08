"""Same-origin HTTP proxy for ttyd (P1-B2B-1).

Backend-only. WebSocket relay is P1-B2B-2, a separate task.
ttyd runs at 127.0.0.1:8081 with --base-path /terminal.
"""
from __future__ import annotations

import urllib.error
import urllib.request

from flask import Blueprint, Response, jsonify, request, stream_with_context

bp = Blueprint("terminal", __name__)

TTYD_BASE = "http://127.0.0.1:8081"
_BASE_PATH = "/terminal"
_MAX_QUERY = 4096
_ALLOWED_METHODS = {"GET", "POST", "HEAD"}
_STRIP_REQ_HEADERS = {"host", "cookie", "x-agent-token", "content-length"}
_FORWARD_REQ_HEADERS = {"content-type", "accept", "cache-control"}
_FORWARD_RESP_HEADERS = ("Content-Type", "Content-Length", "Cache-Control")


def _validate_path(path: str):
    if path.startswith("/"):
        return "path must not start with /"
    parts = path.split("/") if path else []
    if ".." in parts:
        return "path must not contain .."
    if "\\" in path:
        return "path must not contain backslash"
    return None


def _validate_query():
    """Refuse an oversized query instead of silently truncating it."""
    if len(request.query_string) > _MAX_QUERY:
        return "query too long"
    return None


def _upstream_url(path: str) -> str:
    qs = request.query_string.decode("ascii", "replace")[:_MAX_QUERY]
    url = f"{TTYD_BASE}{_BASE_PATH}/" + path
    if qs:
        url += "?" + qs
    return url


def _forward_request_headers() -> dict:
    out = {}
    for k, v in request.headers.items():
        low = k.lower()
        if low in _STRIP_REQ_HEADERS:
            continue
        if low in _FORWARD_REQ_HEADERS:
            out[k] = v
    return out


@bp.route("/terminal/", defaults={"path": ""},
          methods=["GET", "POST", "HEAD"])
@bp.route("/terminal/<path:path>", methods=["GET", "POST", "HEAD"])
def terminal_proxy(path):
    err = _validate_path(path)
    if err:
        return jsonify({"error": err}), 400

    too_long = _validate_query()
    if too_long:
        return jsonify({"error": too_long}), 414

    method = request.method.upper()
    if method not in _ALLOWED_METHODS:
        return jsonify({"error": "method not allowed"}), 405

    body = None
    if method == "POST":
        body = request.get_data(cache=False)

    req = urllib.request.Request(
        _upstream_url(path),
        data=body,
        method=method,
        headers=_forward_request_headers(),
    )

    try:
        upstream = urllib.request.urlopen(req, timeout=30)
    except urllib.error.HTTPError as e:
        upstream = e
    except (urllib.error.URLError, ConnectionRefusedError, OSError):
        return jsonify({"error": "ttyd not running"}), 502

    def gen():
        try:
            while True:
                chunk = upstream.read(65536)
                if not chunk:
                    break
                yield chunk
        finally:
            try:
                upstream.close()
            except Exception:
                pass

    headers = {}
    for h in _FORWARD_RESP_HEADERS:
        v = upstream.headers.get(h)
        if v:
            headers[h] = v

    return Response(
        stream_with_context(gen()),
        status=upstream.status,
        headers=headers,
    )
