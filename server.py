# test 4
# test 3
"""OpenAI-compatible HTTP server wrapping LiteRT (for opencode).

Entry point only. Every route lives in a blueprint under `routes/`; shared
runtime state is in `routes/_state.py`. Run with `python server.py`.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from config import Config, model_missing_message
from core import auth
from core.llm import LLMError
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

    # A missing local model is an expected state on a cloud-only device,
    # not a server fault: hand the caller the actionable message (how to
    # add a model) with a 503 instead of a bare 500.
    @application.errorhandler(LLMError)
    def _llm_unavailable(exc):
        path = request.path or ""
        if path.startswith("/api/") or path.startswith("/v1/"):
            return jsonify({"error": str(exc)}), 503
        return str(exc), 503

    # Anything no route caught still has to leave a trace. The handler is keyed
    # on the base Exception class, so Flask routes HTTPExceptions (404/405/…)
    # here too — those are handed straight back with their own status so the
    # stock responses survive. Under TESTING, propagate is on and real errors
    # are re-raised before this runs, so tests still see the raw exception.
    @application.errorhandler(Exception)
    def _log_unhandled(exc):
        if isinstance(exc, HTTPException):
            code = exc.code or 500
            text = exc.description or exc.name
            # Every /api and /v1 caller parses the body as JSON — a plain-text
            # "The requested URL was not found…" reaches the UI as
            # `Unexpected token 'T' … is not valid JSON`. The PWA shell and
            # static files keep the stock HTML so they still render.
            path = request.path or ""
            if path.startswith("/api/") or path.startswith("/v1/"):
                return jsonify({"error": text}), code
            return text, code
        log.error("unhandled exception on %s %s",
                  request.method, request.path, exc_info=exc)
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    return application


app = create_app()

# flask_sock keeps a single Sock instance per app: bind it here, before the
# first request, so /terminal/ws lands in app.url_map.
from routes.terminal import init_ws  # noqa: E402

init_ws(app)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()

    cfg = Config.load()
    model_path = cfg.resolve_model_path()
    have_model = bool(model_path) and Path(model_path).is_file()
    print("=" * 60, flush=True)
    print("  LiteRT OpenAI Bridge", flush=True)
    print("=" * 60, flush=True)
    if have_model:
        print(f"  Model:   {Path(model_path).name}", flush=True)
        print(f"  From:    {Path(model_path).parent}", flush=True)
        print(f"  Context: {cfg.context_size} ({cfg.context_size // 1024}K)",
              flush=True)
        print(f"  Backend: {cfg.backend}", flush=True)
        print("\nPre-loading engine...", flush=True)

        t0 = time.time()
        get_llm()
        print(f"  ✅ Loaded in {time.time() - t0:.2f}s", flush=True)
    else:
        # Cloud-only start: refusing to boot here would also lock the user
        # out of the Settings page where provider keys are entered — the
        # exact thing a model-less device needs. Serve everything, and let
        # local turns report the model as unavailable (503, LLMError).
        print("  Model:   (none found)", flush=True)
        print(model_missing_message(model_path), flush=True)
        print("  Starting WITHOUT the local model — the UI, Settings, Code "
              "mode and\n  cloud-backed Agent turns work. Local Chat turns "
              "will report the model as\n  unavailable until a .litertlm "
              "file is added.", flush=True)

    # Q2: enforcement is opt-in — without a token the API is open exactly as
    # before (that is what keeps the test suite honest). main() guarantees a
    # token exists, so a server started this way is never accidentally open.
    tok = auth.configured_token()
    if not tok:
        tok = auth.generate_token()
        print(f"\n  Token: {tok}  (created, saved to {auth._token_file()})",
              flush=True)
    elif os.environ.get(auth.ENV_TOKEN):
        print(f"\n  Token: {tok}  (from ${auth.ENV_TOKEN})", flush=True)
    else:
        print(f"\n  Token: {tok}  (from {auth._token_file()})", flush=True)

    print(f"\nServer: http://{args.host}:{args.port}", flush=True)
    print(f"  Open:  http://{args.host}:{args.port}/?token={tok}", flush=True)
    print("  API:   X-Agent-Token header, or Authorization: Bearer for /v1",
          flush=True)
    print("  GET  /v1/models", flush=True)
    print("  POST /v1/chat/completions", flush=True)
    print("\nPress Ctrl+C to stop.\n", flush=True)

    # Data outlives code: when the newest backup in ~ is over a week
    # old, take a fresh one before serving. A failed backup must never
    # stop the server — warn and continue. Opt out with
    # AI_AGENT_NO_AUTO_BACKUP=1.
    if not os.environ.get("AI_AGENT_NO_AUTO_BACKUP"):
        try:
            import backup as _backup
            _made = _backup.maybe_auto_backup()
            if _made:
                print(f"  Auto-backup: {_made.name} written",
                      flush=True)
        except Exception as exc:  # startup must survive a bad backup
            log.warning("auto-backup failed: %s", exc)
            print(f"  ⚠ Auto-backup failed ({exc}) — server continues",
                  flush=True)

    start_idle_watchdog()
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
