"""Same-origin HTTP proxy for ttyd (P1-B2B-1) + WebSocket relay (P1-B2B-3a).

Backend-only. ttyd runs at 127.0.0.1:8081 with --base-path /terminal.
"""
from __future__ import annotations

import threading
import urllib.error
import urllib.request

from flask import Blueprint, Response, jsonify, request, stream_with_context
from flask_sock import Sock
import simple_websocket

bp = Blueprint("terminal", __name__)
_sock = Sock()

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


TTYD_WS = "ws://127.0.0.1:8081/terminal/ws"


def terminal_ws(ws):
    """Relay browser <-> ttyd WebSocket.

    `ws` is a simple_websocket.Server (flask_sock's wrapper).
    Two daemon threads pipe bytes; either closing stops both.
    """
    qs = request.query_string.decode("ascii", "replace")
    url = TTYD_WS + (("?" + qs) if qs else "")
    try:
        upstream = simple_websocket.Client(
            url,
            subprotocols=["tty"],
            headers={
                "Origin": "http://127.0.0.1:8080",
            },
        )
    except Exception:
        try:
            ws.close(1011, "ttyd unavailable")
        except Exception:
            pass
        return

    stop = threading.Event()

    def _recv(server):
        try:
            return server.receive(timeout=0.5)
        except Exception:
            return None

    def browser_to_upstream():
        try:
            while not stop.is_set():
                data = _recv(ws)
                if data is None:
                    continue
                if isinstance(data, str):
                    data = data.encode("utf-8")
                upstream.send(data)
        except Exception:
            pass
        finally:
            stop.set()

    def upstream_to_browser():
        try:
            while not stop.is_set():
                data = _recv(upstream)
                if data is None:
                    continue
                ws.send(data)
        except Exception:
            pass
        finally:
            stop.set()

    t1 = threading.Thread(target=browser_to_upstream, daemon=True)
    t2 = threading.Thread(target=upstream_to_browser, daemon=True)
    t1.start(); t2.start()
    t1.join(); t2.join()

    for s in (upstream, ws):
        try:
            s.close()
        except Exception:
            pass


# flask_sock's decorator returns None, so register explicitly: the module
# must keep `terminal_ws` as the raw handler (tests call it directly).
_sock.route("/terminal/ws")(terminal_ws)


def init_ws(app):
    """Bind the Sock instance to the app. Call once from server.py."""
    _sock.init_app(app)
