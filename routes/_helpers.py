"""Shared helpers used across route blueprints.

Bodies here are carried over verbatim from server.py; only the direct reads
of the old module globals were replaced with `routes._state` accessors.
"""
from __future__ import annotations

import unicodedata

from flask import request

from core.burmese import sanitize as sanitize_burmese
from core.llm import estimate_messages_tokens
from core.log import get_logger
from memory_extract import extract as extract_profile
from routes._state import (
    agent_turn_guard,
    get_agent_mode,
    get_chat_mode,
    get_memory,
    inference_guard,
)

log = get_logger("routes.helpers")

# Only used when the config itself cannot be read — the value Chat Mode has
# always fallen back to.
LOCAL_TOTAL_FALLBACK = 8192

# Invisible / bidi / zero-width characters that must never reach the UI.
_DROP_CPS = {
    0x00AD,                                   # Soft Hyphen
    0x200B, 0x200C, 0x200D,                   # ZWSP/ZWNJ/ZWJ
    0x200E, 0x200F,                           # LRM/RLM
    0x2028, 0x2029,                           # LS/PS
    0x202A, 0x202B, 0x202C, 0x202D, 0x202E,   # Bidi embed
    0x2060, 0x2061, 0x2062, 0x2063, 0x2064,   # Word joiner
    0xFEFF,                                   # BOM
}

# Thai code points the model emits in place of their Myanmar twin while
# writing Burmese. The pairs are visually identical, but the allowlist below
# drops Thai outright — so the vowel simply vanished: "ခင်ဗျးာ။"
# (Thai SARA AA) came out of the UI as "ခင်ဗျ။". Normalise first, then filter.
# Only unambiguous mark-like pairs are listed; Thai consonants and tone marks
# are still dropped (mapping those would invent a mark that is not there).
_THAI_TO_MYANMAR = {
    0x0E32: 0x102C,   # SARA AA    -> MYANMAR VOWEL SIGN AA
    0x0E34: 0x102D,   # SARA I     -> MYANMAR VOWEL SIGN I
    0x0E35: 0x102E,   # SARA II    -> MYANMAR VOWEL SIGN II
    0x0E38: 0x102F,   # SARA U     -> MYANMAR VOWEL SIGN U
    0x0E39: 0x1030,   # SARA UU    -> MYANMAR VOWEL SIGN UU
    0x0E47: 0x103A,   # MAITAIKHU  -> MYANMAR SIGN ASAT
    0x0E4D: 0x1036,   # NIKHAHIT   -> MYANMAR SIGN ANUSVARA
    0x0E33: 0x1036,   # SARA AM    -> MYANMAR SIGN ANUSVARA
}


def normalize_confusables(text):
    """Replace Thai lookalikes with their Myanmar twin.

    Character by character — a streaming caller may split the text anywhere
    and still get exactly the same result as the whole-string call.
    """
    if not text:
        return text
    return "".join(
        chr(_THAI_TO_MYANMAR[ord(c)]) if ord(c) in _THAI_TO_MYANMAR else c
        for c in text
    )


# The model drops the final AA vowel from its own honorific even after the
# prompt spells the rule out — "…မလဲခင်ဗျ။" instead of "…မလဲခင်ဗျာ။", measured
# on the real model. Corrected in code as well, on the whole-string path only:
# a streaming caller cannot tell whether the next chunk starts with the vowel,
# so inserting there could produce "ခင်ဗျာာ".
_HONORIFIC = "ခင်ဗျ"

# Wrong whole-token spellings the model emits for the same honorific — seen
# five different ways inside ONE live agent reply: ခင်ဗျာ / ခင်ဗွာ / ခင်ျ /
# ခဗျ / ခင်ဗလျ. None of these is correct Burmese, so the token is rewritten.
# Ordered longest-first: ခင်ျ is a prefix of ခင်ျာ, ခင်ျ does not occur inside
# ခင်ဗျာ or ခင်ဗလျ (ဗ/လ follow ်, not ျ), and ခဗျ does not occur in ခင်ဗျာ.
_HONORIFIC_WRONG = (
    ("ခင်ဗလျ", "ခင်ဗျာ"),
    ("ခင်ဗွာ", "ခင်ဗျာ"),
    ("ခင်ျာ", "ခင်ဗျာ"),
    ("ခင်ျ", "ခင်ဗျာ"),
    ("ခဗျ", "ခင်ဗျာ"),
)


def _is_myanmar_mark(ch: str) -> bool:
    """A combining mark in the Myanmar block (same rule as core.burmese)."""
    return 0x1000 <= ord(ch) <= 0x109F and unicodedata.category(ch) in ("Mn", "Mc")


def _is_myanmar_letter(ch: str) -> bool:
    """A base letter in the Myanmar block."""
    return 0x1000 <= ord(ch) <= 0x109F and unicodedata.category(ch) == "Lo"


def _replace_wrong_honorific(text, wrong, right):
    """Rewrite each whole ``wrong`` token, keeping words that merely start with it.

    Replaced only when the next character is not a Myanmar letter or mark, so a
    correct ``ခင်ဗျာ``/``ခင်ဗျံး`` after the match is never half-rewritten, while
    a sentence end (``။``/``၊``), a space or Latin text still counts as a
    token end and does get rewritten.
    """
    out, start, n = [], 0, len(wrong)
    while True:
        i = text.find(wrong, start)
        if i < 0:
            out.append(text[start:])
            break
        end = i + n
        nxt = text[end] if end < len(text) else ""
        if nxt == "" or not (_is_myanmar_letter(nxt) or _is_myanmar_mark(nxt)):
            out.append(text[start:i])
            out.append(right)
        else:
            out.append(text[start:end])
        start = end
    return "".join(out)


def repair_honorific(text):
    """Repair the honorific: wrong spellings rewritten, missing AA vowel added.

    The vowel is only added when no Myanmar mark follows, so a correct
    ``ခင်ဗျာ`` is untouched and the real words ``ခင်ဗျို့`` / ``ခင်ဗျံး`` keep
    their spelling. The wrong-token table is stricter: it also requires that no
    Myanmar letter follows, so ``ခင်ျ`` inside a longer word is left alone.
    """
    if not text:
        return text
    for wrong, right in _HONORIFIC_WRONG:
        if wrong in text:
            text = _replace_wrong_honorific(text, wrong, right)
    if _HONORIFIC not in text:
        return text
    out = []
    n = len(_HONORIFIC)
    start = 0
    while True:
        i = text.find(_HONORIFIC, start)
        if i < 0:
            out.append(text[start:])
            break
        end = i + n
        out.append(text[start:end])
        if end >= len(text) or not _is_myanmar_mark(text[end]):
            out.append("ာ")
        start = end
    return "".join(out)


def json_body() -> dict:
    """Parse the request body as JSON, tolerating a malformed payload."""
    try:
        return request.get_json(force=True, silent=False) or {}
    except Exception as e:
        # Kept lenient (callers treat {} as "missing fields"), but the parse
        # failure itself is a client bug worth seeing server-side.
        log.debug("malformed JSON body on %s %s: %s: %s",
                  request.method, request.path, type(e).__name__, e)
        return {}


def extract_messages(messages):
    """Split OpenAI messages into (system, user_prompt)."""
    system = ""
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") for c in content if isinstance(c, dict)
            )
        if role == "system":
            system = content
        elif role == "user":
            parts.append(content)
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
    return system, "\n\n".join(parts)


def is_repetition_loop(text, min_repeat=15, max_pattern=15):
    """Detect repeated pattern at tail (handles multi-char patterns).
    Returns (is_loop, cut_index). If is_loop, truncate text[:cut_index].
    """
    if not text or len(text) < min_repeat:
        return False, len(text)
    for plen in range(1, max_pattern + 1):
        if len(text) < plen * min_repeat:
            break
        pattern = text[-plen:]
        count = 1
        i = len(text) - plen * 2
        while i >= 0 and text[i:i + plen] == pattern:
            count += 1
            i -= plen
        if count >= min_repeat:
            return True, i + plen
    return False, len(text)


def filter_scripts(text):
    """Normalise Thai lookalikes, then keep ONLY Burmese + Latin + digits +
    visible punctuation. Drops invisible/format chars (U+00AD, U+200B-F, U+2028-2E, U+2060-64, U+FEFF).
    """
    if not text:
        return text
    text = normalize_confusables(text)
    out = []
    for ch in text:
        cp = ord(ch)
        # 1) Explicit blacklist (invisible/format)
        if cp in _DROP_CPS:
            continue
        # 2) Any remaining format char (Cf category)
        if unicodedata.category(ch) == "Cf":
            continue
        # 3) Allowlist
        if 0x1000 <= cp <= 0x109F:      # Burmese
            out.append(ch)
        elif 0xAA60 <= cp <= 0xAA7F:    # Burmese Ext-A
            out.append(ch)
        elif 0x0020 <= cp <= 0x007E:    # ASCII printable
            out.append(ch)
        elif 0x00A0 <= cp <= 0x024F:    # Latin Extended
            out.append(ch)
        elif 0x2010 <= cp <= 0x2027:    # Visible dashes/quotes
            out.append(ch)
        elif 0x2030 <= cp <= 0x205E:    # Visible punctuation
            out.append(ch)
        elif ch in "\n\r\t":
            out.append(ch)
        # else: drop
    return "".join(out)


def sanitize_text(text):
    """filter_scripts + Burmese sequence validation + honorific repair.

    Range filtering alone cannot catch a structurally broken Myanmar sequence
    (repeated virama, orphaned medial) — every character in it is legal on its
    own. ``core.burmese`` checks the sequence, drops repeated marks and
    truncates at the first marker with no safe reading; order matters because
    ``filter_scripts`` may remove the character that sat between a base and
    its mark. The honorific repair runs last, on the surviving text.

    Whole-string only — the streaming path feeds ``filter_scripts`` per chunk
    and lets the ``end`` event carry this final, corrected text.
    """
    return repair_honorific(sanitize_burmese(filter_scripts(text)))


def active_key_label() -> str:
    """`Key: sk-f3…647b (Active)` — the key the backend is on right now.

    Masked through KeyManager, so a raw key can never reach the wire. Provider
    order follows the fallback list (the provider that would be tried first).
    Empty when no key exists: the chip then shows nothing instead of a name
    nothing could use.
    """
    try:
        from core.key_manager import KeyManager, mask
        km = KeyManager()
        cfg = km.load()
        seen = set()
        order = list(cfg.get("fallback") or []) + list(cfg.get("keys") or {})
        for p in order:
            if p in seen:
                continue
            seen.add(p)
            e = km.active_entry(p)
            if e and (e.get("key") or ""):
                return "Key: %s (Active)" % mask(e.get("key") or "")
    except Exception as e:
        log.warning("active key label unavailable: %s: %s",
                    type(e).__name__, e)
    return ""


def run_agent(user_msg: str) -> dict:
    # The tool loop drives the shared conversation, so the whole call — not
    # just mode.handle() — has to be exclusive: the trace is read back out of
    # the same mode object afterwards. agent_turn_guard first (this path can
    # spend 30s on a provider call), then inference_lock for the local model.
    with agent_turn_guard():
        with inference_guard():
            mode = get_agent_mode()
            # Fix (B): mark the tail by identity, not by length. handle()
            # calls _trim_messages() (keeps the newest 40) before returning,
            # which shifts every index — a [before_len:] slice on a long
            # history starts past the end of the trimmed list and silently
            # drops this turn's tool_call/tool_result entries (verified: a
            # 46-message session ran web_search but returned tool_trace: []).
            # The marker is the newest pre-turn message, so the trim keeps it
            # unless one turn appends 40+ messages; the id-set fallback below
            # still finds the new entries in that case.
            before_ids = {id(m) for m in mode._messages}
            marker = mode._messages[-1] if mode._messages else None
            reply = mode.handle(user_msg)
            msgs = mode._messages
            start = 0
            if marker is not None:
                for i, m in enumerate(msgs):
                    if m is marker:
                        start = i + 1
                        break
                else:
                    # marker trimmed away mid-turn — fall back to identity diff
                    start = 0
                    msgs = [m for m in msgs if id(m) not in before_ids]
            new_msgs = msgs[start:]
            tool_trace = []
            for m in new_msgs:
                if not isinstance(m, dict):
                    continue
                role = m.get("role", "")
                if role == "tool_call":
                    tool_trace.append({
                        "type": "call",
                        "tool": m.get("tool", ""),
                        "args": m.get("args", {}),
                    })
                elif role == "tool_result":
                    text = m.get("text", "")
                    tool_trace.append({
                        "type": "result",
                        "tool": m.get("tool", ""),
                        "result": text[:500],
                        "result_len": len(text),
                    })
            return {
                "reply": reply,
                "turns": mode.turns,
                "tool_trace": tool_trace,
            }


def run_chat(user_msg: str) -> dict:
    with inference_guard():
        mode = get_chat_mode()
        return {"reply": mode.handle(user_msg), "turns": mode.turns}


def extract_and_apply(user_msg: str) -> list:
    # memory_extract runs its own LLM pass over the message.
    with inference_guard():
        memory = get_memory()
        out = []
        try:
            ex = extract_profile(user_msg)
            prof = ex.get("profile", {})
            for k, v in prof.items():
                ok, _reason = memory.set_profile(k, v)
                if ok:
                    out.append(f"{k}={v}")
        except Exception as e:
            # Profile extraction is best-effort, but a crash here means the
            # user's facts are never captured — log it rather than hide it.
            log.warning("profile extraction failed: %s: %s", type(e).__name__, e)
        return out


def _local_total() -> int:
    """The on-device engine's window — the only answer Chat Mode can give."""
    try:
        return int(get_chat_mode().llm.config.context_size) or LOCAL_TOTAL_FALLBACK
    except Exception:
        return LOCAL_TOTAL_FALLBACK


def agent_context() -> dict:
    """Agent Mode's context usage, measured against the *active model's* window.

    Answers three questions the meter needs: which model is (or would be) used,
    how many tokens its window really holds, and how full it is. The transcript
    itself never leaves this process — only counts and the model name do.

    Before the first cloud turn there is no measured usage yet, so the window
    comes from `AgentBackend.planned_context()` (the provider that would answer
    next) and the usage from a fresh estimate. With no cloud backend at all —
    Local Only, or the backend module missing — this is the local engine's own
    `Config.context_size`, which is exactly what Chat Mode has always reported.
    """
    out = {"used": 0, "total": LOCAL_TOTAL_FALLBACK, "pct": 0.0, "model": "",
           "provider": "", "source": "local", "endpoint": "local",
           "trimmed": False, "dropped": 0, "summarized": 0,
           "over_budget": False, "at": 0.0}
    try:
        mode = get_agent_mode()
    except Exception as e:
        log.debug("agent context unavailable: %s: %s", type(e).__name__, e)
        out["total"] = _local_total()
        out["local_total"] = out["total"]
        return out
    # ── Local Only: total is the KV cache, never a cloud window ──
    try:
        if mode._mode() == "local":
            out["total"] = _local_total()
            out["local_total"] = out["total"]
            out["model"] = _friendly_local()
            out["source"] = "local"
            out["endpoint"] = "local"
            return out
    except Exception as e:
        log.debug("local-mode short-circuit failed: %s: %s",
                  type(e).__name__, e)

    backend = None
    try:
        backend = mode._backend_obj()
    except Exception as e:
        log.debug("agent backend unavailable for context: %s: %s",
                  type(e).__name__, e)
    state = {}
    if backend is not None:
        try:
            state = backend.context_state() or {}
        except Exception as e:
            log.debug("backend context record unreadable: %s: %s",
                      type(e).__name__, e)
            state = {}
    limit = int(state.get("max") or 0)
    if not limit and backend is not None:
        # No cloud step yet — plan it, so the meter is honest before the first
        # reply instead of jumping from 8192 to 64K after one.
        try:
            planned = backend.planned_context() or {}
        except Exception as e:
            log.debug("planned context unavailable: %s: %s",
                      type(e).__name__, e)
            planned = {}
        limit = int(planned.get("limit") or 0)
        for field in ("model", "provider", "source", "endpoint"):
            if not state.get(field) and planned.get(field):
                state[field] = planned[field]
    if not limit:
        out.update({
            "total": _local_total(),
            "model": _friendly_local(),
            "source": "local",
            "endpoint": "local",
        })
    else:
        out.update({
            "total": limit,
            "source": state.get("source") or "unknown",
            "endpoint": state.get("endpoint") or "",
            "trimmed": bool(state.get("trimmed")),
            "dropped": int(state.get("dropped") or 0),
            "summarized": int(state.get("summarized") or 0),
            "over_budget": bool(state.get("over_budget")),
            "at": float(state.get("at") or 0.0),
        })
    out["model"] = str(state.get("model") or "") or _friendly_local()
    out["provider"] = str(state.get("provider") or "")
    used = int(state.get("used") or 0)
    if not used:
        try:
            used = estimate_messages_tokens(mode.get_messages())
        except Exception as e:
            log.debug("agent transcript estimate failed: %s: %s",
                      type(e).__name__, e)
            used = 0
    out["used"] = used
    out["pct"] = round(used / out["total"] * 100, 1) if out["total"] else 0.0
    # The on-device window is reported alongside the cloud one so the drawer's
    # 8K/12K/16K buttons can still show which KV cache is loaded — Agent Mode
    # now answers with the cloud model's window, which is not one of those.
    out["local_total"] = _local_total()
    return out


def _friendly_local() -> str:
    """The on-device model's display name, or '' when it cannot be read."""
    try:
        from core.agent_backend import friendly_local_name
        return friendly_local_name() or ""
    except Exception:
        return ""


def ctx_of(mode: str) -> dict:
    if mode == "agent":
        return agent_context()
    try:
        chat_mode = get_chat_mode()
        msgs = chat_mode.get_messages()
        used = estimate_messages_tokens(msgs)
        total = chat_mode.llm.config.context_size
        pct = round(used / total * 100, 1) if total else 0
        return {"used": used, "total": total, "pct": pct,
                "model": _friendly_local(), "source": "local",
                "endpoint": "local", "trimmed": False, "dropped": 0,
                "summarized": 0, "over_budget": False, "at": 0.0,
                "local_total": total}
    except Exception as e:
        log.warning("context estimate unavailable: %s: %s", type(e).__name__, e)
        return {"used": 0, "total": LOCAL_TOTAL_FALLBACK, "pct": 0.0,
                "model": "", "source": "local", "endpoint": "local",
                "trimmed": False, "dropped": 0, "summarized": 0,
                "over_budget": False, "at": 0.0,
                "local_total": LOCAL_TOTAL_FALLBACK}
