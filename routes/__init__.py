"""Flask blueprints — route modules for the ai-agent server.

Each module here owns a self-contained slice of the HTTP API. Shared
runtime state lives in `_state`, shared pure helpers in `_helpers`.
`server.py` stays a thin entry point that registers these and runs the app.

Routes are declared with full paths (no `url_prefix`) so the route table can
be grepped end-to-end and cannot drift through double-prefixing.
"""
from __future__ import annotations

from routes import (
    safety,
    pwa,
    openai_compat,
    chat,
    memory_api,
    sessions_api,
    code,
    skills_api,
)

# Registration order is irrelevant to routing (rules are matched by URL), but
# kept in the same order the routes were originally declared in server.py.
BLUEPRINTS = (
    safety.bp,
    pwa.bp,
    openai_compat.bp,
    chat.bp,
    memory_api.bp,
    sessions_api.bp,
    code.bp,
    skills_api.bp,
)

__all__ = ["BLUEPRINTS", "register"]


def register(app) -> None:
    """Attach every blueprint plus the app-level safety hook to `app`."""
    # before_request is registered on the app (not a blueprint) on purpose:
    # a blueprint hook only fires for that blueprint's own requests, and the
    # thermal guard must cover every endpoint.
    app.before_request(safety.safety_guard)
    for bp in BLUEPRINTS:
        app.register_blueprint(bp)
