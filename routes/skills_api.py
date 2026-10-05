"""Skills API — list, inspect, run, generate, delete user skill templates."""
from __future__ import annotations

from flask import Blueprint, jsonify

from core.skills import (
    _sanitize_name,
    default_template,
    delete_skill,
    list_skills,
    load_skill,
    render,
    save_skill,
)
from routes._helpers import json_body
from routes._state import (
    agent_turn_guard,
    get_agent_mode,
    get_chat_mode,
    ensure_modes,
    inference_guard,
)

bp = Blueprint("skills_api", __name__)

SKILL_GEN_PROMPT = (
    "Generate a skill template as PLAIN MARKDOWN (no code fences). "
    "Format EXACTLY:\n"
    "You are an expert <role>. <specific instructions about how to handle "
    "the task described below>. Keep the user's language.\n\n"
    "Input:\n\n"
    "{{args}}\n\n"
    "Rules:\n"
    "- Start with 'You are' \n"
    "- Include the literal placeholder {{args}} in the template\n"
    "- No YAML frontmatter, no code fences, no extra explanation\n"
    "- 2 to 5 sentences\n\n"
    "Skill name: {name}\n"
    "Description: {description}\n\n"
    "Output ONLY the template body:"
)


def _validate_template(body: str) -> bool:
    """A generated body must be non-trivial + contain {{args}}."""
    if not body or len(body.strip()) < 30:
        return False
    if "{{args}}" not in body:
        return False
    if "```" in body or "---" in body[:20]:
        return False
    return True


@bp.route("/api/skills", methods=["GET"])
def api_skills_list():
    """List all available skills."""
    try:
        return jsonify({"skills": list_skills()})
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@bp.route("/api/skills/<name>", methods=["GET"])
def api_skills_get(name):
    """Get a single skill's metadata + body."""
    sk = load_skill(name)
    if not sk:
        return jsonify({"error": f"skill not found: {name}"}), 404
    return jsonify(sk)


@bp.route("/api/skills/<name>/run", methods=["POST"])
def api_skills_run(name):
    """Run a skill with args. Body: {args: str, mode?: 'chat'|'agent'}."""
    body = json_body()
    args = (body.get("args") or "").strip()
    mode = (body.get("mode") or "agent").strip().lower()

    if not args:
        return jsonify({"error": "args required"}), 400
    if mode not in ("chat", "agent"):
        return jsonify({"error": "mode must be chat or agent"}), 400

    sk = load_skill(name)
    if not sk:
        return jsonify({"error": f"skill not found: {name}"}), 404

    target = get_agent_mode() if mode == "agent" else get_chat_mode()
    prompt = render(sk, args)

    try:
        # An Agent skill run goes through the same cloud-or-local loop as a
        # chat turn, so it takes the agent-turn lock before the engine lock.
        if mode == "agent":
            with agent_turn_guard():
                with inference_guard():
                    reply = target.handle(prompt)
        else:
            with inference_guard():
                reply = target.handle(prompt)
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    return jsonify({
        "skill": sk["name"],
        "mode": mode,
        "reply": reply,
    })


@bp.route("/api/skills/create", methods=["POST"])
def api_skills_create():
    """Create a skill from a name + description (LLM-generated body)."""
    body = json_body()
    raw_name = (body.get("name") or "").strip()
    description = (body.get("description") or "").strip()
    user_template = (body.get("template") or "").strip()

    name = _sanitize_name(raw_name)
    if not name:
        return jsonify({"error": "invalid or missing name (letters/digits/-/_)"}), 400
    if not description:
        return jsonify({"error": "description is required"}), 400
    if load_skill(name):
        return jsonify({"error": f"skill already exists: {name}"}), 409

    # 1) User-provided template → highest priority
    if user_template:
        try:
            meta = save_skill(name, description, user_template)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"ok": True, "source": "user", **meta})

    # 2) Try LLM-generated template
    ensure_modes()
    llm_source = "fallback"
    try:
        chat_mode = get_chat_mode()
        prompt = SKILL_GEN_PROMPT.format(name=name, description=description)
        with inference_guard():
            generated = chat_mode.llm.ask(prompt) if hasattr(chat_mode, "llm") else ""
        if generated and _validate_template(generated):
            save_skill(name, description, generated.strip())
            llm_source = "llm"
        else:
            save_skill(name, description, default_template(name, description))
    except Exception:
        save_skill(name, description, default_template(name, description))

    sk = load_skill(name)
    return jsonify({
        "ok": True,
        "source": llm_source,
        "name": name,
        "description": description,
        "args": sk.get("args", "text") if sk else "text",
    })


@bp.route("/api/skills/<name>", methods=["DELETE"])
def api_skills_delete(name):
    """Delete a skill by name."""
    if delete_skill(name):
        return jsonify({"ok": True, "name": name})
    return jsonify({"error": f"skill not found: {name}"}), 404
