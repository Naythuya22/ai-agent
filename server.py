"""OpenAI-compatible HTTP server wrapping LiteRT (for opencode).

Entry point only. Every route lives in a blueprint under `routes/`; shared
runtime state is in `routes/_state.py`. Run with `python server.py`.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from config import Config, model_missing_message
from core.log import get_logger
from routes import register
from routes._state import get_llm
from routes.safety import start_idle_watchdog

log = get_logger("server")


def create_app() -> Flask:
    """Build the Flask app and attach every blueprint + the safety guard."""
    # Point Flask's own /static at web/ and drop the duplicate rule in
    # routes/pwa.py. Previously both claimed "/static/<path:filename>" and the
    # built-in won, resolving against a non-existent project-root static/ — so
    # every /static request 404'd even though web/style.css exists.
    # static_url_path must be given explicitly: Flask derives it from the
    # folder's basename, so a folder named "web" would mount at "/web".
    web_dir = _root / "web"
    application = Flask(
        __name__, static_folder=str(web_dir), static_url_path="/static"
    )
    register(application)

    # Anything no route caught still has to leave a trace. The handler is keyed
    # on the base Exception class, so Flask routes HTTPExceptions (404/405/…)
    # here too — those are handed straight back with their own status so the
    # stock responses survive. Under TESTING, propagate is on and real errors
    # are re-raised before this runs, so tests still see the raw exception.
    @application.errorhandler(Exception)
    def _log_unhandled(exc):
        if isinstance(exc, HTTPException):
            return exc.description or exc.name, exc.code
        log.error("unhandled exception on %s %s",
                  request.method, request.path, exc_info=exc)
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    return application


app = create_app()


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()

    cfg = Config.load()
    model_path = cfg.resolve_model_path()
    if not model_path or not Path(model_path).is_file():
        print(model_missing_message(model_path), flush=True)
        return 1
    print("=" * 60, flush=True)
    print("  LiteRT OpenAI Bridge", flush=True)
    print("=" * 60, flush=True)
    print(f"  Model:   {Path(model_path).name}", flush=True)
    print(f"  From:    {Path(model_path).parent}", flush=True)
    print(f"  Context: {cfg.context_size} ({cfg.context_size // 1024}K)", flush=True)
    print(f"  Backend: {cfg.backend}", flush=True)
    print("\nPre-loading engine...", flush=True)

    t0 = time.time()
    get_llm()
    print(f"  ✅ Loaded in {time.time() - t0:.2f}s", flush=True)

    print(f"\nServer: http://{args.host}:{args.port}", flush=True)
    print("  GET  /v1/models", flush=True)
    print("  POST /v1/chat/completions", flush=True)
    print("\nPress Ctrl+C to stop.\n", flush=True)

    start_idle_watchdog()
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
