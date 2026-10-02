"""Memory + relationship-profile endpoints."""
from __future__ import annotations

from flask import Blueprint, jsonify

from routes._helpers import json_body
from routes._state import get_memory

bp = Blueprint("memory_api", __name__)


@bp.route("/api/memory", methods=["GET"])
def api_memory():
    return jsonify({"text": get_memory().show()})


@bp.route("/api/remember", methods=["POST"])
def api_remember():
    text = (json_body().get("text") or "").strip()
    if not text:
        return jsonify({"error": "text required"}), 400
    ok = get_memory().add_fact(text, source="web")
    return jsonify({"ok": ok})


@bp.route("/api/profile", methods=["GET"])
def api_profile_get():
    return jsonify({"profile": get_memory().get_profile()})


@bp.route("/api/profile", methods=["POST"])
def api_profile_set():
    body = json_body()
    key = (body.get("key") or "").strip()
    val = (body.get("value") or "").strip()
    if not key or not val:
        return jsonify({"error": "key and value required"}), 400
    memory = get_memory()
    if memory.set_profile(key, val):
        return jsonify({"ok": True, "profile": memory.get_profile()})
    return jsonify({"error": "invalid key/value"}), 400
