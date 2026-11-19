"""Agent-facing memory tools (Phase 2 memory gate).

propose_memory — propose one durable fact for long-term memory.
                 The gate is mechanical: an owner/tool claim must
                 carry a verbatim evidence quote that the harness
                 itself recorded (the user's own words, or a tool
                 result). Your own replies are never evidence, so
                 quote from the tool output or the user's message —
                 not from anything you wrote. Inferences go to
                 quarantine automatically; say so honestly.
list_memory    — read what long-term memory currently holds, so
                 "what do you remember about X" is answered from
                 the store, not improvised.
"""
from __future__ import annotations

from core import memorygate


def propose_memory(key: str, text: str, source_kind: str = "tool",
                   evidence: str = "") -> str:
    """Propose one durable fact for long-term memory.

    Use this for facts that will still matter next week: something
    the user decided or prefers, a proven fact about their phone,
    files or projects, the verified outcome of a task. Do NOT use
    it for chat noise, guesses, or your own summaries of yourself.

    Args:
        key: short slug for the subject, e.g. "owner_name",
            "pos_printer", "shell_state". One active fact per key.
        text: the fact, one or two sentences, stated plainly.
        source_kind: "owner" (the user said it), "tool" (a tool
            result proved it), or "inference" (your own conclusion).
        evidence: for owner/tool, a VERBATIM quote from the user's
            message or the tool result that proves the fact. The
            gate searches the harness's own records for this quote;
            if it is not found, the fact is quarantined as
            unverified. Never quote your own reply.
    """
    try:
        out = memorygate.propose(key, text, source_kind,
                                 evidence=evidence)
    except Exception as e:
        return f"Memory proposal refused: {type(e).__name__}: {e}"
    entry = out["entry"]
    if out["verdict"] == "active":
        return (f"Remembered [{entry['key']}]: {entry['text']} "
                "(evidence verified — this is now long-term memory).")
    if out["verdict"] == "duplicate":
        return (f"Already remembered [{entry['key']}]: "
                f"{entry['text']} — nothing changed.")
    return (f"Stored as UNVERIFIED [{entry['key']}]: {entry['text']} "
            f"— reason: {entry['reason']}. It sits in quarantine "
            "and expires unless real evidence confirms it. Do not "
            "treat it as established fact.")


def list_memory() -> str:
    """List long-term memory: verified facts and the unverified
    quarantine section, exactly as stored."""
    try:
        entries = memorygate.list_entries()
    except Exception as e:
        return f"Memory unreadable: {type(e).__name__}: {e}"
    active = [e for e in entries if e.get("status") == "active"]
    quarantined = [e for e in entries
                   if e.get("status") == "quarantine"]
    if not active and not quarantined:
        return "Long-term memory is empty."
    lines = []
    for e in active:
        lines.append(f"- [{e.get('key', '?')}] {e.get('text', '')}")
    if quarantined:
        lines.append("Unverified (quarantine):")
        for e in quarantined:
            lines.append(f"- [{e.get('key', '?')}] "
                         f"{e.get('text', '')}")
    return "\n".join(lines)


ALL_TOOLS = [propose_memory, list_memory]
