"""OpenAI-compatible surface — consumed by opencode and any OpenAI SDK client."""
from __future__ import annotations

import json
import time

from flask import Blueprint, Response, jsonify, request

import config
from core.log import get_logger
from routes._helpers import extract_messages
from routes._state import get_llm, inference_guard

log = get_logger("routes.openai")

bp = Blueprint("openai_compat", __name__)


@bp.route("/v1/models", methods=["GET"])
def list_models():
    return jsonify({
        "object": "list",
        "data": [{
            "id": "gemma-4-e4b",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
        }],
    })


@bp.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    try:
        body = request.get_json(force=True, silent=False) or {}
    except Exception as e:
        return jsonify({"error": {"message": f"invalid JSON: {e}"}}), 400

    messages = body.get("messages", [])
    if not messages:
        return jsonify({"error": {"message": "messages required"}}), 400

    model = body.get("model", "gemma-4-e4b")
    # The client's own value is honoured; a missing or nonsensical one falls
    # back to the configured default, and nothing passes the ceiling.
    max_tokens = config.clamp_max_tokens(body.get("max_tokens"),
                                         config.CHAT_MAX_TOKENS_DEFAULT)
    stream = body.get("stream", False)

    system, prompt = extract_messages(messages)

    # 4.2i-C-H4: one unit of model work, same lock Chat mode takes
    # (routes/chat.py, modes/agent.py local branch). The engine, its KV
    # cache and every cached Conversation are shared mutable state, and
    # Flask runs threaded=True; without this an opencode /v1 call and a
    # chat turn can drive the same native engine at once and the LiteRT
    # bridge drops the process.
    try:
        with inference_guard():
            llm = get_llm()
            engine = llm._engine
            assert engine is not None
            conv = engine.create_conversation(
                system_message=system or None,
                max_output_tokens=max_tokens,
            )
            try:
                resp = conv.send_message(prompt)
            finally:
                try:
                    conv.__exit__(None, None, None)
                except Exception as e:
                    log.debug("could not close the bridge conversation: %s: %s",
                              type(e).__name__, e)
            reply = resp["content"][0]["text"]
    except Exception as e:
        return jsonify({
            "error": {"message": f"{type(e).__name__}: {e}"}
        }), 500

    created = int(time.time())
    response = {
        "id": f"chatcmpl-{created}",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": reply},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }

    if not stream:
        return jsonify(response)

    def gen():
        chunk = dict(response)
        chunk["object"] = "chat.completion.chunk"
        chunk["choices"] = [{
            "index": 0,
            "delta": {"role": "assistant", "content": reply},
            "finish_reason": None,
        }]
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        done = dict(response)
        done["object"] = "chat.completion.chunk"
        done["choices"] = [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }]
        yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return Response(gen(), mimetype="text/event-stream")
