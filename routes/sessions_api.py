"""Session management + conversation history endpoints."""
from __future__ import annotations

from flask import Blueprint, jsonify, request

from routes._helpers import json_body
from routes._state import ensure_modes, get_agent_mode, get_chat_mode
from core.log import get_logger

log = get_logger("routes.sessions")

bp = Blueprint("sessions_api", __name__)


def _store():
    """SessionStore is imported lazily — it touches ~/.ai-agent on load."""
    from sessions import SessionStore
    return SessionStore()


@bp.route("/api/sessions", methods=["GET"])
def api_sessions_list():
    ensure_modes()
    mode = request.args.get("mode")
    try:
        store = _store()
        items = store.list_sessions(mode if mode else None)
        active_chat = store.get_active_id("chat")
        active_agent = store.get_active_id("agent")
        for s in items:
            if s["mode"] == "chat":
                s["active"] = (s["id"] == active_chat)
            else:
                s["active"] = (s["id"] == active_agent)
        return jsonify({"sessions": items, "active_chat": active_chat,
                        "active_agent": active_agent})
    except Exception as e:
        return jsonify({"error": str(e), "sessions": []}), 500


@bp.route("/api/sessions", methods=["POST"])
def api_sessions_create():
    ensure_modes()
    body = json_body()
    mode = (body.get("mode") or "chat").strip()
    title = (body.get("title") or "").strip()
    try:
        sid = _store().create_session(mode, title)
        # Reload active mode if same
        if mode == "chat":
            get_chat_mode().reload_session(sid)
        elif mode == "agent":
            get_agent_mode().reload_session(sid)
        return jsonify({"ok": True, "id": sid})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/sessions/<sid>", methods=["GET"])
def api_sessions_get(sid):
    ensure_modes()
    try:
        s = _store().get_session(sid)
        if not s:
            return jsonify({"error": "not found"}), 404
        return jsonify({"session": s})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/sessions/<sid>", methods=["DELETE"])
def api_sessions_delete(sid):
    ensure_modes()
    try:
        store = _store()
        s = store.get_session(sid)
        if not s:
            return jsonify({"error": "not found"}), 404
        mode = s.get("mode", "chat")
        ok = store.delete_session(sid)
        if not ok:
            return jsonify({"error": "delete failed"}), 500
        # If active deleted, reload new active
        new_active = store.get_active_id(mode)
        if mode == "chat" and get_chat_mode().session_id == sid:
            if new_active:
                get_chat_mode().reload_session(new_active)
            else:
                get_chat_mode().reset()
        elif mode == "agent" and get_agent_mode().session_id == sid:
            if new_active:
                get_agent_mode().reload_session(new_active)
            else:
                get_agent_mode().reset()
        return jsonify({"ok": True, "new_active": new_active})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/sessions/<sid>/activate", methods=["POST"])
def api_sessions_activate(sid):
    ensure_modes()
    try:
        store = _store()
        s = store.get_session(sid)
        if not s:
            return jsonify({"error": "not found"}), 404
        mode = s.get("mode", "chat")
        if mode == "chat":
            ok = get_chat_mode().reload_session(sid)
        elif mode == "agent":
            ok = get_agent_mode().reload_session(sid)
        else:
            ok = False
        if not ok:
            return jsonify({"error": "activate failed"}), 500
        return jsonify({"ok": True, "mode": mode})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/history", methods=["GET"])
def api_history():
    ensure_modes()
    mode = request.args.get("mode", "chat")
    try:
        if mode == "agent":
            msgs = list(get_agent_mode().get_messages())
        else:
            msgs = list(get_chat_mode().get_messages())
    except Exception:
        log.error("could not list sessions", exc_info=True)
        msgs = []
    recent = msgs[-40:]
    return jsonify({"mode": mode, "messages": recent})


@bp.route("/api/new", methods=["POST"])
def api_new():
    ensure_modes()
    body = json_body()
    mode = (body.get("mode") or "chat").strip()
    archived = None
    try:
        archived = _store().archive_mode(mode)
    except Exception:
        # The reset below still runs, so the old thread is dropped — say so
        # loudly instead of losing it without a trace.
        log.error("could not archive %s session before reset", mode, exc_info=True)
    try:
        if mode == "agent":
            get_agent_mode().reset()
        else:
            get_chat_mode().reset()
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"ok": True, "archived": archived, "mode": mode})
