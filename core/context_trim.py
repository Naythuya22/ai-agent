"""Keep a request inside the active model's window — estimator and trimmer.

Two jobs:

* **Estimate.** Burmese is not 4 characters per token like ASCII: a single
  grapheme cluster carries far more information per character, so English is
  counted at ``chars / 3.5`` and Burmese at ``chars / 2.0``. Under-counting here
  is what lets a transcript grow past the window and come back as a 400.
* **Trim.** When the estimate does not fit, old material goes first and in this
  order: tool calls/results and assistant turns (the bulk, and the least
  expensive to lose), then — if it is still too big — the remaining middle
  history collapses into one summary. The system prompt and the current turn
  (the last user message and everything after it) are never dropped.

Trimming always works on a deep copy and never mutates the caller's list, so
the stored transcript is only replaced when the caller accepts the result.
"""
from __future__ import annotations

import copy
import json

from core.log import get_logger

log = get_logger("context_trim")

# Chars per token. English/ASCII really is ~4; 3.5 leaves a little headroom.
ENGLISH_CHARS_PER_TOKEN = 3.5
# Burmese clusters cost more per character than Latin ones.
BURMESE_CHARS_PER_TOKEN = 2.0

# Held back for the answer plus provider framing, so a turn never fills the
# window with the prompt and then has nowhere to put the reply.
RESERVE = 1_000

# Losing more than this share of the history in one go is worth a log line —
# it is the difference between "the meter is high" and "your tools are eating
# the window", and only the log can say which.
LOST_WARNING_PCT = 20.0

# Each pass folds the oldest half of the middle history into one summary, so
# this bounds the work on a very long transcript. Three passes is 1/8 of it.
_MAX_SUMMARY_PASSES = 3

# Burmese, Burmese Extended-A and the Myanmar Extended-B block, so
# "ကိုကြီး" is not counted as English.
_BURMESE_RANGES = (
    (0x1000, 0x109F),
    (0xAA60, 0xAA7F),
    (0xA9E0, 0xA9FF),
)

_TOOL_ROLES = ("tool_call", "tool_result", "tool")
# Material the trimmer sacrifices first: the bulk of a tool-heavy transcript,
# and the cheapest thing to lose.
_FLESH_ROLES = ("tool_call", "tool_result", "tool", "assistant")


def is_burmese(ch: str) -> bool:
    if not ch:
        return False
    cp = ord(ch)
    for lo, hi in _BURMESE_RANGES:
        if lo <= cp <= hi:
            return True
    return False


# ── estimation ───────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """Rough token count for mixed Burmese/English text."""
    if not text:
        return 0
    burmese = 0
    for ch in text:
        if is_burmese(ch):
            burmese += 1
    other = len(text) - burmese
    tokens = (other / ENGLISH_CHARS_PER_TOKEN
              + burmese / BURMESE_CHARS_PER_TOKEN)
    return int(tokens) if tokens >= 1 else (1 if text else 0)


def _args_text(value) -> str:
    """Tool arguments as text — a big one is a big part of the window."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def message_text(m) -> str:
    """Everything a stored/normalized message contributes, as one string."""
    if not isinstance(m, dict):
        return str(m or "")
    parts = [m.get("text") or "", m.get("content") or "",
             m.get("reasoning_content") or m.get("reasoning") or ""]
    content = m.get("content")
    if isinstance(content, list):               # vision / multi-part content
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text") or "")
    args = m.get("args")
    if args:
        parts.append(_args_text(args))
    arguments = m.get("arguments")
    if arguments:
        parts.append(_args_text(arguments))
    for call in m.get("tool_calls") or []:
        if isinstance(call, dict):
            fn = call.get("function") or {}
            parts.append(str(fn.get("name") or ""))
            parts.append(_args_text(fn.get("arguments")))
    return "\n".join(p for p in parts if isinstance(p, str) and p)


def estimate_message_tokens(m) -> int:
    """One message: its text plus the per-message framing overhead."""
    return estimate_tokens(message_text(m)) + 4


def estimate_messages_tokens(messages) -> int:
    """Token estimate for a transcript, in either storage shape."""
    return sum(estimate_message_tokens(m) for m in (messages or []))


def _overhead(system: str, tools) -> int:
    """Prompt cost that is not a message: system prompt + tool schemas."""
    total = estimate_tokens(system or "") if system else 0
    for tool in tools or []:
        try:
            total += estimate_tokens(json.dumps(tool, ensure_ascii=False))
        except (TypeError, ValueError):
            total += estimate_tokens(str(tool))
    return total


def estimate_request_tokens(system: str = "", messages=None, tools=None) -> int:
    """Prompt as the provider sees it: system + history + tool schemas."""
    return _overhead(system, tools) + estimate_messages_tokens(messages)


# ── roles ────────────────────────────────────────────────────────────────

def _role(m) -> str:
    if not isinstance(m, dict):
        return "user"
    return str(m.get("role") or "user")


# ── the local summariser ─────────────────────────────────────────────────

def _clip(text: str, limit: int = 160) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def local_summary(entries: list) -> str:
    """Deterministic, offline digest of trimmed-out history.

    No model call on purpose: trimming must work on the failure path, when the
    provider is the thing that just refused, and it must not cost a round trip
    or need a key. Keeps what the next turn needs — what was asked, which tools
    ran, what came back.
    """
    entries = list(entries or [])
    if not entries:
        return ""
    asks, replies, tools = [], [], []
    for m in entries:
        role = _role(m)
        text = message_text(m)
        if role == "user" and text:
            asks.append(_clip(text))
        elif role == "assistant" and text:
            replies.append(_clip(text))
        elif role in _TOOL_ROLES:
            name = str(m.get("tool") or m.get("name") or "") if isinstance(m, dict) else ""
            tools.append((name or "tool", _clip(text, 80)))
    lines = ["[Earlier conversation — %d messages trimmed]" % len(entries)]
    if asks:
        lines.append("User asked: " + " → ".join(asks[:4]))
    if replies:
        lines.append("Assistant said: " + " → ".join(replies[-2:]))
    if tools:
        counts: dict = {}
        for name, _body in tools:
            counts[name] = counts.get(name, 0) + 1
        lines.append("Tools used: " + ", ".join(
            "%s×%d" % (k, v) for k, v in sorted(counts.items())))
    return "\n".join(lines)


# ── orphan repair ────────────────────────────────────────────────────────

def _drop_orphans(entries: list) -> list:
    """Remove tool entries that lost their partner, keeping order.

    Two storage shapes have to survive here: ours (``tool_call`` then
    ``tool_result``) and the normalized OpenAI one (an assistant carrying
    ``tool_calls`` then ``role: tool`` entries keyed by ``tool_call_id``).
    A ``tool`` message with no matching call — an assistant advertising calls
    whose answers were dropped, or a ``tool_call`` whose result never arrived —
    is what draws "messages with role tool must be a response to a preceding
    message" from a provider, so dropping an entry out of the middle must never
    leave that behind. The orphan goes, never the message around it.
    """
    out: list = []
    open_result = False        # ours: a tool_call still awaits its result
    open_ids: set = set()      # normalized: call ids still awaiting answers
    for m in entries:
        role = _role(m)
        if role == "tool_call":
            if open_result or open_ids:
                continue
            open_result = True
            out.append(m)
            continue
        if role == "tool_result":
            if not open_result:
                continue
            open_result = False
            out.append(m)
            continue
        if role == "tool":
            tid = str(m.get("tool_call_id") or "") if isinstance(m, dict) else ""
            if tid and tid in open_ids:
                open_ids.discard(tid)
                out.append(m)
            continue
        if open_result:
            # A tool_call whose result never arrived is the orphan — the
            # message behind it is real content (usually the next question) and
            # has to survive, exactly like the unanswered assistant calls below.
            while out and _role(out[-1]) == "tool_call":
                del out[-1]
            open_result = False
        if open_ids:
            # The answers to this message's calls were dropped, so the call
            # itself goes with them.
            while out and _role(out[-1]) == "assistant" and out[-1].get("tool_calls"):
                del out[-1]
            open_ids = set()
        open_ids = {str(c.get("id")) for c in ((m.get("tool_calls") or [])
                                              if isinstance(m, dict) else [])
                    if isinstance(c, dict) and c.get("id")}
        out.append(m)
    return out


# ── the trimmer ──────────────────────────────────────────────────────────

def _is_summary(m) -> bool:
    """Marker the trimmer puts on the entry it replaced history with."""
    return isinstance(m, dict) and bool(m.get("summary"))


def _protected_flags(entries: list) -> list:
    """True where an entry must survive: system prompts and the current turn.

    A summary entry counts as current turn: it is already compressed, and it is
    the only record left of everything that was dropped to make room for it.
    """
    last_user = -1
    for i, m in enumerate(entries):
        if _role(m) == "user":
            last_user = i
    return [(_role(m) == "system" or i >= last_user or _is_summary(m))
            for i, m in enumerate(entries)]


def _empty_result(entries: list, limit: int, before: int) -> dict:
    return {
        "messages": entries,
        "limit": int(limit or 0),
        "before": before,
        "after": before,
        "dropped": 0,
        "summarized": 0,
        "removed_pct": 0.0,
        "trimmed": False,
        "over_budget": False,
    }


def trim_messages(messages, *, limit: int, reserve: int = RESERVE,
                  system: str = "", tools=None, summarize=None) -> dict:
    """Fit a transcript into ``limit - reserve`` tokens.

    Returns a dict with the trimmed ``messages`` (a deep copy, possibly
    shorter, never reordered) plus what the UI and the tests need: ``before``
    and ``after`` token counts, ``dropped`` (entries removed outright),
    ``summarized`` (entries folded into a summary), ``removed_pct``,
    ``trimmed``, and ``over_budget``.

    Order of sacrifice: old tool/assistant material, then the middle history is
    compressed into one summary, then whatever is left of it is dropped. The
    system prompt and the current turn are never touched, so a transcript that
    cannot fit even after all of that comes back with ``over_budget`` set
    rather than quietly sending a request the provider will refuse.

    With ``summarize=None`` the offline :func:`local_summary` is used; pass a
    callable ``(entries) -> str`` to plug a model-backed summary in. That
    callable runs on the failure path, so it must not need a key of its own.
    """
    entries = copy.deepcopy(list(messages or []))
    # Every cost is computed once and tracked as a running total: the estimator
    # is per-message, so dropping an entry subtracts exactly its own cost.
    base = _overhead(system, tools)
    costs = [estimate_message_tokens(m) for m in entries]
    total = base + sum(costs)
    info = _empty_result(entries, limit, total)
    budget = max(0, int(limit or 0) - max(0, int(reserve or 0)))
    if not entries:
        return info
    if budget <= 0:
        info["over_budget"] = True
        log.error("context budget is zero (limit %s) — nothing can be sent; "
                  "raise the model's window or /clear", limit)
        return info
    if total <= budget:
        return info

    flags = _protected_flags(entries)

    def _drop(i: int) -> None:
        del entries[i]
        del costs[i]
        del flags[i]

    # 1. Old tool/assistant material goes first: it is the bulk of a
    #    tool-heavy transcript and the cheapest thing to lose. User turns are
    #    not touched here — they are compressed (step 2) before any are deleted,
    #    so the transcript keeps a record of what was asked.
    i = 0
    while i < len(entries) and total > budget:
        role = _role(entries[i])
        if not flags[i] and role in _FLESH_ROLES:
            total -= costs[i]
            _drop(i)
        else:
            i += 1

    # 2. Still over: compress the rest of the middle instead of deleting it.
    #    Each pass folds the oldest half into one summary, so what was asked
    #    survives at a fraction of the size. The summary from the previous pass
    #    goes back into the fold, so this converges to *one* summary however
    #    many turns were dropped instead of stacking summaries until they cost
    #    as much as the history they replaced.
    summary_at = None
    for _ in range(_MAX_SUMMARY_PASSES):
        if total <= budget:
            break
        middle = [i for i in range(len(entries)) if not flags[i]]
        if summary_at is not None:
            chunk = sorted({summary_at} | set(middle[: max(1, len(middle) // 2)]))
        elif middle:
            chunk = sorted(set(middle[: max(1, len(middle) // 2)]))
        else:
            break
        victim = [entries[i] for i in chunk]
        try:
            text = (summarize or local_summary)(copy.deepcopy(victim))
        except Exception as e:               # a bad summariser must not kill
            log.warning("context summary failed (%s: %s) — dropping instead",
                        type(e).__name__, e)
            text = None
        if not text:
            break
        at = chunk[0]
        for i in reversed(chunk):
            total -= costs[i]
            _drop(i)
        cost = estimate_tokens(str(text)) + 4
        entries.insert(at, {"role": "user", "text": str(text),
                            "summary": True})
        costs.insert(at, cost)
        flags.insert(at, True)           # a summary is already compressed
        summary_at = at
        total += cost
        folded = sum(1 for m in victim if not _is_summary(m))
        info["summarized"] += folded
        log.info("context trim: summarized %d middle messages (%d → %d "
                 "tokens of %s)", folded, info["before"], total, limit)

    # 3. Summarising was not enough (or not available): drop whatever middle
    #    history is left, oldest first.
    i = 0
    while i < len(entries) and total > budget:
        if not flags[i]:
            total -= costs[i]
            _drop(i)
        else:
            i += 1

    entries = _drop_orphans(entries)
    # Orphan repair can remove entries, so the running total and the flags are
    # both rebuilt from what is actually left. Zipping the shortened list against
    # the old flags would protect the wrong messages and drop the current turn.
    total = _overhead(system, tools) + estimate_messages_tokens(entries)
    # 4. Nothing left to give: say so rather than pretending it fits.
    if total > budget:
        entries = [m for m, keep in zip(entries, _protected_flags(entries))
                   if keep]
        entries = _drop_orphans(entries)
        info["over_budget"] = True
        log.error("context still over budget after trimming (%d > %d) — the "
                  "system prompt and current turn alone exceed this model's "
                  "%s token window", total, budget, limit)

    info["messages"] = entries
    info["dropped"] = len(messages or []) - len(entries)
    # Recomputed rather than trusting the running total: orphan repair may have
    # dropped more entries, and the number is what the meter shows.
    info["after"] = _overhead(system, tools) + estimate_messages_tokens(entries)
    if info["after"] < info["before"]:
        info["trimmed"] = True
        info["removed_pct"] = round(
            (info["before"] - info["after"]) / info["before"] * 100, 1)
    if info["trimmed"] and info["removed_pct"] >= LOST_WARNING_PCT:
        log.warning("context trim dropped %.1f%% of the prompt (%d → %d of %s "
                    "tokens)", info["removed_pct"], info["before"],
                    info["after"], limit)
    return info