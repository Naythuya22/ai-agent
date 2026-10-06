"""Chat + agent mode endpoints (context, summarisation, sync and SSE streams)."""
from __future__ import annotations

import json
import secrets

from pathlib import Path

from flask import Blueprint, Response, jsonify, request, stream_with_context

import config
from core.burmese import BurmeseValidator
from core.llm import (
    VISION_OCR_SUGGESTION,
    build_vision_message,
    stream_conversation,
    supports_vision,
)
from core.log import get_logger
from modes.agent import last_usage, record_usage
from routes._helpers import (
    active_key_label,
    agent_context,
    ctx_of,
    extract_and_apply,
    filter_scripts,
    is_repetition_loop,
    json_body,
    run_agent,
    run_chat,
    sanitize_text,
)
from routes._state import (
    agent_turn_guard,
    ensure_modes,
    get_agent_mode,
    get_chat_mode,
    get_memory,
    inference_guard,
    rebuild_modes,
)

bp = Blueprint("chat", __name__)

log = get_logger("routes.chat")

# Read-only, always "chat": no code path assigns to it (pre-existing).
_current_mode = "chat"


@bp.route("/api/ctx", methods=["GET"])
def api_ctx():
    mode = request.args.get("mode", "chat")
    return jsonify(ctx_of(mode))


@bp.route("/api/agent/context", methods=["GET"])
def api_agent_context():
    """Agent Mode's context usage against the active model's own window.

    Its own endpoint because Agent Mode can run on a cloud model whose window
    is nothing like the local engine's — the meter must not fall back to the
    8192 KV cache figure there. Counts and model name only, never the
    transcript.
    """
    return jsonify(agent_context())


@bp.route("/api/context/<size>", methods=["POST"])
def api_context_size(size):
    try:
        n = int(size)
    except (ValueError, TypeError):
        return jsonify({"error": "invalid size"}), 400
    if n not in (8192, 12288, 16384):
        return jsonify({"error": "must be 8192/12288/16384"}), 400

    # Unloading the engine underneath an in-flight generation would be fatal,
    # so the whole check-unload-save-rebuild cycle runs exclusively.
    with inference_guard():
        ensure_modes()
        chat_mode = get_chat_mode()

        # Already at target?
        try:
            current = chat_mode.llm.config.context_size
        except Exception:
            current = None
        if current == n:
            return jsonify({"ok": True, "size": n, "note": "already set"})

        # Unload engine + close modes (session data on disk)
        try:
            chat_mode.close()
        except Exception as e:
            log.warning("chat_mode.close() failed during ctx resize: %s: %s",
                        type(e).__name__, e)
        try:
            get_agent_mode().close()
        except Exception as e:
            log.warning("agent_mode.close() failed during ctx resize: %s: %s",
                        type(e).__name__, e)
        try:
            chat_mode.llm.unload()
        except Exception as e:
            log.warning("llm.unload() failed during ctx resize: %s: %s",
                        type(e).__name__, e)

        # Update config
        try:
            chat_mode.llm.config.context_size = n
            chat_mode.llm.config.save()
        except Exception as e:
            return jsonify({"error": f"save failed: {e}"}), 500

        # Reinit modes (lazy engine — next request loads at new size)
        try:
            rebuild_modes()
        except Exception as e:
            return jsonify({"error": f"reinit failed: {e}"}), 500

    return jsonify({"ok": True, "size": n, "reload": "next_message"})


@bp.route("/api/summary", methods=["POST"])
def api_summary():
    body = json_body()
    mode = (body.get("mode") or "chat").strip()
    try:
        with inference_guard():
            if mode == "agent":
                r = get_agent_mode().compress_session()
            else:
                r = get_chat_mode().compress_session()
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
    if "error" in r:
        return jsonify(r), 400
    return jsonify(r)


@bp.route("/api/chat", methods=["POST"])
def api_chat():
    msg = (json_body().get("message") or "").strip()
    if not msg:
        return jsonify({"error": "message required"}), 400
    remembered = extract_and_apply(msg)
    try:
        r = run_chat(msg)
        r["reply"] = sanitize_text(r.get("reply") or "")
        r["remembered"] = remembered
        r["status"] = "ready"
        r["ctx"] = ctx_of("chat")
        return jsonify(r)
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@bp.route("/api/chat/stream", methods=["POST"])
def api_chat_stream():
    msg = (json_body().get("message") or "").strip()
    if not msg:
        return jsonify({"error": "message required"}), 400
    ensure_modes()
    remembered = extract_and_apply(msg)
    chat_mode = get_chat_mode()

    def generate():
        if remembered:
            yield "data: " + json.dumps({"type": "remembered", "items": remembered}) + "\n\n"
        try:
            session = chat_mode.session
        except Exception as e:
            yield "data: " + json.dumps({"type": "error", "error": f"{type(e).__name__}: {e}"}) + "\n\n"
            return
        # Held for the whole stream: the session owns a cached Conversation
        # object, so a second client must not start generating on it mid-stream.
        # The `with` releases on normal completion and on GeneratorExit (client
        # disconnect), so an aborted stream cannot wedge the lock.
        with inference_guard():
            # Per-response sequence validator. Chunks split syllables, so the
            # validator holds a trailing mark run back until the next chunk (or
            # the final `end`) settles it — a bad boundary never cuts good text.
            # A cut must not break out of the loop: the generator is drained
            # first (F7b — a mid-stream break trips LiteRT sig11), and `end`
            # then carries the truncated text, which is what the client renders.
            vseq = BurmeseValidator()
            seq_cut = False
            skip = False
            streamed = ""
            try:
                for kind, data in session.send_stream(msg):
                    if skip and kind != "end":
                        continue
                    if kind == "text":
                        safe, cut = vseq.feed(filter_scripts(data))
                        streamed += safe
                        if safe:
                            yield "data: " + json.dumps({"type": "chunk", "text": safe}) + "\n\n"
                        if cut:
                            seq_cut = True
                            skip = True
                    elif kind == "error":
                        yield "data: " + json.dumps({"type": "error", "error": str(data)}) + "\n\n"
                        return
                    elif kind == "tool_calls":
                        yield "data: " + json.dumps({"type": "error", "error": "tool_calls not allowed in chat stream"}) + "\n\n"
                        return
                    elif kind == "end":
                        # Settle the held-back mark run so the chunks add up to
                        # the whole message instead of stopping short of it.
                        held = vseq.flush()
                        if held:
                            streamed += held
                            yield "data: " + json.dumps({"type": "chunk", "text": held}) + "\n\n"
                        if vseq.cut:
                            seq_cut = True
                        # The client replaces the streamed text with end.text,
                        # so the whole message gets one full validation pass.
                        full = data if isinstance(data, str) else ""
                        if seq_cut:
                            fin = vseq.accepted
                        elif full:
                            filtered = filter_scripts(full)
                            fin = sanitize_text(filtered)
                            seq_cut = fin != filtered
                        else:
                            fin = vseq.accepted
                        ctx = ctx_of("chat")
                        yield "data: " + json.dumps({"type": "end", "text": fin, "truncated": seq_cut, "ctx": ctx, "status": "ready"}) + "\n\n"
                        return
                # Drained after a cut, or the stream ended without `end`.
                vseq.flush()
                fin = vseq.accepted or sanitize_text(streamed)
                ctx = ctx_of("chat")
                yield "data: " + json.dumps({"type": "end", "text": fin, "truncated": seq_cut, "ctx": ctx, "status": "ready"}) + "\n\n"
            except GeneratorExit:
                return
            except Exception as e:
                yield "data: " + json.dumps({"type": "error", "error": f"{type(e).__name__}: {e}"}) + "\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


# ---------- Agent-mode image upload (vision input) ----------

# Spec: images up to 5MB; png/jpeg/webp/gif only.
IMAGE_MAX_BYTES = 5 * 1024 * 1024
IMAGE_MIMES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}


def _uploads_dir() -> Path:
    """Staging dir for uploaded images (inside the isolated config dir)."""
    d = Path(config.CONFIG_DIR) / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_image(image_id: str) -> Path | None:
    """Map a returned image id back to its staged file (no path traversal)."""
    if not image_id or Path(image_id).name != image_id:
        return None
    if not image_id.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
        return None
    path = _uploads_dir() / image_id
    return path if path.is_file() else None


def _model_path_for_vision() -> str:
    """Resolved model path without paying for an engine load."""
    try:
        return config.Config.load().resolve_model_path() or ""
    except Exception as e:
        log.debug("could not resolve model path for vision: %s: %s",
                  type(e).__name__, e)
        return ""


@bp.route("/api/agent/upload", methods=["POST"])
def api_agent_upload():
    """Validate + stage an image for vision input (Agent mode only)."""
    f = (request.files or {}).get("file")
    if f is None or not f.filename:
        return jsonify({"error": "file required"}), 400

    mimetype = (f.mimetype or "").split(";")[0].strip().lower()
    if mimetype not in IMAGE_MIMES:
        return jsonify({
            "error": "unsupported image type: %s (allowed: png, jpeg, webp, gif)"
                     % (mimetype or "unknown"),
        }), 400

    # Read one byte past the ceiling so an oversize body is detected without
    # buffering a huge file in memory.
    data = f.read(IMAGE_MAX_BYTES + 1)
    if len(data) > IMAGE_MAX_BYTES:
        return jsonify({"error": "image too large (max 5MB)"}), 400
    if not data:
        return jsonify({"error": "empty file"}), 400

    ok, reason = supports_vision(_model_path_for_vision())
    if not ok:
        return jsonify({
            "error": "vision unavailable — %s" % reason,
            "suggestion": VISION_OCR_SUGGESTION,
        }), 400

    image_id = secrets.token_hex(16) + IMAGE_MIMES[mimetype]
    (_uploads_dir() / image_id).write_bytes(data)
    return jsonify({
        "id": image_id,
        "name": Path(f.filename).name,
        "size": len(data),
        "vision": True,
    })


@bp.route("/api/agent/stream", methods=["POST"])
def api_agent_stream():
    body = json_body()
    msg = (body.get("message") or "").strip()
    if not msg:
        return jsonify({"error": "message required"}), 400
    image_id = (body.get("image_id") or "").strip()
    image_path = None
    vision_model_path = ""
    if image_id:
        # Fail before the SSE stream starts so the UI gets a plain 400 with
        # the reason instead of an error event.
        image_path = _resolve_image(image_id)
        if image_path is None:
            return jsonify({"error": "uploaded image not found — upload it again"}), 400
        vision_model_path = _model_path_for_vision()
        ok, reason = supports_vision(vision_model_path)
        if not ok:
            return jsonify({
                "error": "vision unavailable — %s" % reason,
                "suggestion": VISION_OCR_SUGGESTION,
            }), 400
    ensure_modes()
    remembered = extract_and_apply(msg)
    agent_mode = get_agent_mode()

    def _sse(obj):
        return "data: " + json.dumps(obj) + "\n\n"

    def _tool_events(new_msgs):
        """tool_call / tool_result events plus the trace carried by `end`.

        The cloud backend answers off-stream, so its whole trace is replayed
        here in transcript order — the same two shapes the UI already paints
        while a local turn is still streaming.
        """
        events, trace = [], []
        for m in new_msgs:
            if not isinstance(m, dict):
                continue
            role = m.get("role", "")
            name = m.get("tool", "")
            if role == "tool_call":
                args = m.get("args", {})
                events.append({"type": "tool_call", "tool": name, "args": args})
                trace.append({"type": "call", "tool": name, "args": args})
            elif role == "tool_result":
                txt = m.get("text", "")
                events.append({"type": "tool_result", "tool": name,
                               "result": txt[:500], "result_len": len(txt)})
                trace.append({"type": "result", "tool": name,
                              "result": txt[:500], "result_len": len(txt)})
        return events, trace

    def _end(text, tool_trace, usage, truncated=False):
        """Terminal SSE event — identical shape for cloud and local turns."""
        return _sse({
            "type": "end",
            "text": text,
            "tool_trace": tool_trace,
            "truncated": truncated,
            "ctx": ctx_of("agent"),
            "status": "ready",
            # What really answered — the UI paints the status-bar
            # chip straight from this instead of refetching config.
            "last_model": usage.get("model", ""),
            "last_backend": usage.get("backend", ""),
            "provider": usage.get("provider", ""),
            "model_label": usage.get("label", ""),
            # Which system prompt this turn was given, and where the choice
            # came from — settings / env / auto.
            "prompt_tier": usage.get("tier", ""),
            "prompt_tier_source": usage.get("tier_source", ""),
        })

    def _finish(final_text):
        # Exactly one call per turn, at every terminal path of _turn().
        # Reflection must never break the stream, so anything it raises dies here.
        try:
            agent_mode._finish_turn(msg, final_text)
        except Exception:
            pass

    def generate():
        if remembered:
            yield _sse({"type": "remembered", "items": remembered})
        # One Agent turn at a time, cloud or local: AgentMode._messages must
        # not interleave with a concurrent /api/agent or skill run. This is
        # deliberately NOT inference_lock — a provider call can block for 30s
        # and Chat mode must keep running while it does.
        with agent_turn_guard():
            try:
                for ev in _turn():
                    yield ev
            except GeneratorExit:
                return
            except Exception as e:
                yield _sse({"type": "error", "error": "%s: %s" % (type(e).__name__, e)})

    def _turn():
        before_len = len(agent_mode._messages)
        # This route drives the cloud and local branches itself instead of
        # calling handle(), so it has to open the turn too: that is what makes
        # the prompt tier resolve once and stay put if the turn falls back to
        # the on-device model half way through.
        agent_mode.begin_turn()

        def _sync_reply(text):
            """Keep the stored transcript in step with what the UI received.

            The cloud path appends the provider's raw text; without this a
            sequence the validator cut would still replay from the session.
            """
            msgs = agent_mode._messages
            if not text or not msgs or msgs[-1].get("role") != "assistant":
                return
            if msgs[-1].get("text") != text:
                msgs[-1]["text"] = text
                agent_mode._save_turn()

        # Cloud backend first (Auto / Cloud Only). A turn carrying an image
        # stays local: per-provider image payloads are not wired up yet.
        if image_path is None:
            cloud_text = None
            try:
                cloud_text = agent_mode._try_cloud(msg)
            except Exception as e:
                log.error("cloud attempt failed (%s: %s)", type(e).__name__, e)
                # _try_cloud rolls its own failures back; this covers anything
                # raised before it took a checkpoint.
                del agent_mode._messages[before_len:]
            if cloud_text is None:
                try:
                    mode = agent_mode._mode()
                except Exception:
                    mode = ""
                if mode == "cloud":
                    # Belt and braces, as in AgentMode.handle(): Cloud Only
                    # never reaches the local engine below.
                    cloud_text = agent_mode._cloud_only_reply(msg, "exhausted")
            if cloud_text is not None:
                events, trace = _tool_events(agent_mode._messages[before_len:])
                for ev in events:
                    yield _sse(ev)
                # Same order as the local path: filter per chunk, validate,
                # then the whole-string sanitiser on what survived.
                vcloud = BurmeseValidator()
                text, cut_now = vcloud.feed(filter_scripts(cloud_text))
                held = vcloud.flush()
                if held:
                    text += held
                truncated = bool(cut_now or vcloud.cut)
                if not truncated:
                    is_loop, cut = is_repetition_loop(text)
                    if is_loop:
                        text, truncated = text[:cut], True
                text = sanitize_text(text)
                _sync_reply(text)
                if text:
                    yield _sse({"type": "chunk", "text": text})
                # Frame first: last_usage()/ctx_of() must describe this turn, not
                # whatever reflection records while it runs.
                end_frame = _end(text, trace, last_usage(), truncated)
                _finish(text)
                yield end_frame
                return

        # Local engine — unchanged from here down.
        try:
            agent_mode._ensure_conv()
            conv = agent_mode._conv
        except Exception as e:
            _finish("")
            yield _sse({"type": "error", "error": "%s: %s" % (type(e).__name__, e)})
            return
        if conv is None:
            _finish("")
            yield _sse({"type": "error", "error": "conversation not ready"})
            return

        agent_mode._messages.append({"role": "user", "text": msg})
        current_input = msg
        if image_path is not None:
            # Attach the staged image to this turn only — session history
            # keeps the plain text so a replay never needs the file back.
            vm = build_vision_message(vision_model_path, str(image_path), msg)
            if vm is None:
                _finish("")
                yield _sse({
                    "type": "error",
                    "error": "vision unavailable — image not sent",
                    "suggestion": VISION_OCR_SUGGESTION,
                })
                return
            current_input = vm
        max_iter = 8
        final_text = ""
        done = False
        loop_detected = False
        seq_cut = False

        try:
            with inference_guard():
                for _ in range(max_iter):
                    tool_calls_this = None
                    text_this = ""
                    skip_remaining = False
                    # Fresh validator per response: each iteration is its own
                    # stream from the model.
                    vseq = BurmeseValidator()
                    for kind, data in stream_conversation(conv, current_input):
                        if skip_remaining:
                            continue
                        if kind == "text":
                            safe, cut_now = vseq.feed(filter_scripts(data))
                            if safe:
                                text_this += safe
                            if cut_now:
                                # Broken Myanmar sequence — stop feeding the
                                # model's output to the UI, then drain (F7b).
                                seq_cut = True
                                skip_remaining = True
                                continue
                            is_loop, cut = is_repetition_loop(text_this)
                            if is_loop:
                                text_this = text_this[:cut]
                                loop_detected = True
                                skip_remaining = True
                                continue
                            yield _sse({"type": "chunk", "text": safe})
                        elif kind == "error":
                            _finish("")
                            yield _sse({"type": "error", "error": str(data)})
                            return
                        elif kind == "tool_calls":
                            tool_calls_this = data
                        elif kind == "end":
                            # Flush the held-back mark run into the stream so
                            # chunks cover the whole message (see chat stream).
                            held = vseq.flush()
                            if held:
                                text_this += held
                                yield _sse({"type": "chunk", "text": held})
                            if vseq.cut:
                                seq_cut = True
                                skip_remaining = True
                                continue    # a cut response has no tool calls
                            if data.get("tool_calls"):
                                tool_calls_this = data["tool_calls"]
                            else:
                                if loop_detected:
                                    final_text = text_this
                                else:
                                    final_text = data.get("text") or text_this

                    if not tool_calls_this:
                        final_text = final_text or text_this
                        done = True
                        break

                    tc_list = tool_calls_this if isinstance(tool_calls_this, list) else [tool_calls_this]
                    for tc in tc_list:
                        if isinstance(tc, dict):
                            fn_info = tc.get("function", {}) or {}
                            name = fn_info.get("name", "")
                            args = fn_info.get("arguments", {})
                        else:
                            name = getattr(tc, "name", "")
                            args = getattr(tc, "arguments", {})
                        if not name:
                            continue
                        agent_mode._messages.append({"role": "tool_call", "tool": name, "args": args})
                        yield _sse({"type": "tool_call", "tool": name, "args": args})
                        try:
                            result = agent_mode._execute_tool(name, args)
                        except Exception as e:
                            result = "Error: %s: %s" % (type(e).__name__, e)
                        agent_mode._messages.append({"role": "tool_result", "tool": name, "text": result})
                        yield _sse({"type": "tool_result", "tool": name, "result": result[:500], "result_len": len(result)})
                        current_input = {"role": "tool", "name": name, "content": result}

                if not done:
                    _finish("")
                    yield _sse({"type": "error", "error": "max_iter (%d) reached" % max_iter})
                    return

                # One full validation pass over the finished message: covers the
                # `end`-text path (chunks may not equal it) and keeps corrupt
                # sequences out of the session replay.
                final_text = sanitize_text(final_text)
                agent_mode._messages.append({"role": "assistant", "text": final_text})
                agent_mode.turns += 1
                if getattr(agent_mode, "_store", None) and getattr(agent_mode, "session_id", None):
                    try:
                        agent_mode._store.save_mode("agent", agent_mode._messages)
                    except Exception as e:
                        # The stream already delivered the answer; without this
                        # line the whole tool trace would vanish silently.
                        log.error("could not save agent session %s: %s: %s",
                                  agent_mode.session_id, type(e).__name__, e)

                new_msgs = agent_mode._messages[before_len:]
                _, tool_trace = _tool_events(new_msgs)
                usage = _record_local_turn(agent_mode)
                end_frame = _end(final_text, tool_trace, usage,
                                 truncated=loop_detected or seq_cut)
                _finish(final_text)
                yield end_frame
        except GeneratorExit:
            return
        except Exception as e:
            _finish("")
            yield _sse({"type": "error", "error": "%s: %s" % (type(e).__name__, e)})

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform",
                 "X-Accel-Buffering": "no"},
    )


@bp.route("/api/agent/stream/v2", methods=["POST"])
def api_agent_stream_v2():
    """SSE cloud streaming. Falls back to v1 on 404."""
    if not config.AGENT_CLOUD_STREAM:
        # Feature off → 404, so the frontend re-POSTs to /api/agent/stream.
        return jsonify({"error": "cloud streaming disabled"}), 404
    # mirrors api_agent_stream at L332 — keep in sync
    body = json_body()
    msg = (body.get("message") or "").strip()
    if not msg:
        return jsonify({"error": "message required"}), 400
    # mirrors api_agent_stream at L336 — keep in sync. A staged image has no
    # cloud payload wired up yet, so that turn belongs to v1.
    image_id = (body.get("image_id") or "").strip()
    if image_id:
        return jsonify({"error": "image turns are handled by v1"}), 404
    # mirrors api_agent_stream at L352 — keep in sync
    ensure_modes()
    # mirrors api_agent_stream at L354 — keep in sync
    agent_mode = get_agent_mode()
    # Reject up front exactly what _try_cloud_stream declines at
    # modes/agent.py L854-L873 — keep in sync — so a non-cloud turn gets a
    # 404 before a single SSE byte is written.
    if agent_mode._mode() == "local":
        return jsonify({"error": "local mode"}), 404
    backend = agent_mode._backend_obj()
    if backend is None:
        return jsonify({"error": "cloud backend unavailable"}), 404
    try:
        want_cloud = backend.should_use_cloud()
    except Exception as e:
        log.warning("cloud check failed (%s)", e)
        want_cloud = False
    if not want_cloud:
        return jsonify({"error": "cloud unavailable"}), 404

    def _frame(event, data):
        """Named SSE event frames — v1's _sse at L356 emits data-only lines."""
        return "event: %s\ndata: %s\n\n" % (event, json.dumps(data))

    def _finish(final_text):
        # Exactly one call per turn — see the three outcomes in generate().
        try:
            agent_mode._finish_turn(msg, final_text)
        except Exception:
            pass

    def generate():
        # mirrors api_agent_stream at L420 — keep in sync: one Agent turn at
        # a time (deliberately NOT inference_lock — a cloud call can block
        # for 30s and Chat mode must keep running while it does).
        with agent_turn_guard():
            try:
                # mirrors api_agent_stream at L435 — keep in sync: open the
                # turn so the prompt tier resolves once and stays put.
                agent_mode.begin_turn()
                agent_mode._clear_turn_cache()
                # The same user-message shape v1's _turn() stores. We track
                # its index so the fallback path can roll it back: otherwise
                # v1's _try_cloud appends a second copy and the UI shows the
                # question twice.
                appended_idx = len(agent_mode._messages)
                agent_mode._messages.append({"role": "user", "text": msg})
                saw_end = False
                final_text = ""
                try:
                    for kind, payload in agent_mode._handle_cloud_stream():
                        if kind == "delta":
                            yield _frame("delta", {"t": payload})
                        elif kind == "tool":
                            yield _frame("tool", {
                                "name": payload.get("name", ""),
                                "args": payload.get("args", {}),
                                "result": payload.get("result", ""),
                            })
                        elif kind == "error":
                            yield _frame("error", payload)
                        elif kind == "end":
                            saw_end = True
                            final_text = payload if isinstance(payload, str) else ""
                            yield _frame("done", {"t": payload})
                    if not saw_end:
                        # Cloud declined → undo our user append so v1 starts
                        # from a clean transcript (no duplicate question).
                        if len(agent_mode._messages) > appended_idx:
                            del agent_mode._messages[appended_idx:]
                        yield _frame("fallback",
                                     {"reason": "cloud_unavailable"})
                        # (b) deliberately NO _finish_turn here: v1 runs its own
                        # reflection when the frontend re-POSTs this turn.
                    else:
                        # (a) end frame already delivered — nothing left to yield.
                        _finish(final_text)
                except Exception:
                    # (c) reflect while the turn is still in the transcript,
                    # then roll back; the outer handler yields the error frame.
                    _finish("")
                    if len(agent_mode._messages) > appended_idx:
                        del agent_mode._messages[appended_idx:]
                    raise
            except Exception as e:
                # mirrors api_agent_stream at L426 — keep in sync. The
                # generator already rolled back on its way out; GeneratorExit
                # is NOT caught, so a client disconnect lets it close and run
                # its own commit/rollback.
                yield _frame("error", {"error": "%s: %s" % (type(e).__name__, e)})

    # mirrors api_agent_stream at L644 — keep in sync
    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform",
                 "X-Accel-Buffering": "no"},
    )


def _record_local_turn(agent_mode) -> dict:
    """Record the on-device model the SSE stream just ran.

    The stream drives the conversation itself instead of going through
    ``AgentMode.handle()``, so without this the status bar would keep showing
    whatever the previous turn used.
    """
    try:
        from core.agent_backend import friendly_local_name
        name = friendly_local_name() or "Local model"
    except Exception:
        name = "Local model"
    try:
        mode = agent_mode._mode()
    except Exception:
        mode = ""
    return record_usage("local", name, "", mode)


@bp.route("/api/agent/status", methods=["GET"])
def api_agent_status():
    """What the last Agent turn actually ran — the status-bar indicator.

    Deliberately not derived from config: a Cloud Only setting with no usable
    key must not be able to print a cloud model name nothing ever called.
    An empty ``label`` means no turn has run yet in this process.
    """
    u = last_usage()
    return jsonify({
        "ok": True,
        "last_model": u.get("model", ""),
        "last_backend": u.get("backend", ""),
        "provider": u.get("provider", ""),
        "label": u.get("label", ""),
        "mode": u.get("mode", ""),
        "used_at": u.get("at", 0.0),
        # Which key is selected right now (masked) — the second chip in the
        # status bar. Selection, not usage; usage is what `label` reports.
        "key_label": active_key_label(),
        # Same request paints the context meter, so the UI does not have to
        # fetch a second endpoint every time it refreshes the chips.
        "ctx": agent_context(),
    })


@bp.route("/api/agent", methods=["POST"])
def api_agent():
    msg = (json_body().get("message") or "").strip()
    if not msg:
        return jsonify({"error": "message required"}), 400
    remembered = extract_and_apply(msg)
    try:
        r = run_agent(msg)
        r["reply"] = sanitize_text(r.get("reply") or "")
        r["remembered"] = remembered
        r["status"] = "ready"
        r["ctx"] = ctx_of("agent")
        u = last_usage()
        r["last_model"] = u.get("model", "")
        r["last_backend"] = u.get("backend", "")
        r["model_label"] = u.get("label", "")
        return jsonify(r)
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@bp.route("/api/clear", methods=["POST"])
def api_clear():
    ensure_modes()
    try:
        with inference_guard():
            get_chat_mode().reset()
            get_agent_mode().reset()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/mode", methods=["GET"])
def api_mode():
    ensure_modes()
    return jsonify({
        "mode": _current_mode,
        "context": get_memory().data.get("version", 0),
        "backend": "GPU",
    })
