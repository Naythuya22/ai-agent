"""Agent-facing skill tools (Phase 1 skill gate).

propose_skill — draft a new skill into the gated inbox. A draft never
                becomes a live skill on its own; the owner reviews the
                gate report and applies it by hand.
use_skill     — load a live skill's full procedure text and follow it.
skill_outcome — report whether a used skill actually worked. This
                feeds the audit log the owner reviews and the
                probation counters in `skillgate status`.
"""
from __future__ import annotations

from core import skillgate
from core import skills as _skills


def propose_skill(name: str, description: str, body: str,
                  source: str = "experience", evidence: str = "") -> str:
    """Propose a new reusable skill for the owner to approve.

    Use this after you complete a procedure successfully and it would
    help again: a multi-step phone task, a fixed workflow, a format
    the user likes. The draft goes to an inbox — it does NOT become a
    real skill until the owner approves it, so propose honestly and
    include your evidence.

    Args:
        name: short skill name (letters, digits, dash, underscore).
        description: one line saying when this skill applies.
        body: the procedure/template text. Must contain {{args}} where
            the task input goes.
        source: where the procedure came from — "experience" (you did
            it yourself and it worked), "web" (read it somewhere),
            "user" (the user taught it), or "other".
        evidence: how you know it works — what you ran, what happened.
    """
    try:
        out = skillgate.propose(name, description, body,
                                source=source, evidence=evidence)
    except Exception as e:
        return f"Proposal not recorded: {type(e).__name__}: {e}"
    gate = out["gate"]
    if gate["ok"] and not gate["warnings"]:
        return (f"Proposal '{out['name']}' is in the inbox and passed "
                "the gate. It becomes a real skill only when the owner "
                "approves it — carry on with your work.")
    if gate["ok"]:
        return (f"Proposal '{out['name']}' is in the inbox; the gate "
                "passed it with warnings: " + "; ".join(gate["warnings"])
                + ". The owner will decide.")
    return (f"Proposal '{out['name']}' is in the inbox but FAILED the "
            "gate: " + "; ".join(gate["failures"])
            + ". The owner can see the failure; a corrected re-proposal "
            "is welcome.")


def use_skill(name: str, args: str = "") -> str:
    """Load a learned skill's full procedure and follow it.

    Args:
        name: the skill name, from the Learned skills list.
        args: the task input to substitute into the skill's {{args}}.
    """
    sk = _skills.load_skill(name)
    if not sk:
        return (f"No live skill named '{name}'. Work without it, or "
                "check the Learned skills list for the right name.")
    skillgate.record_load(sk["name"])
    return ("Follow this skill procedure for the current task:\n\n"
            + _skills.render(sk, args))


def _parse_worked(value) -> bool | None:
    """Parse the worked flag robustly.

    Tool-call arguments sometimes arrive as strings ("False"), and
    bool("False") is True in Python — trusting the type silently
    recorded a failed skill use as a success on 2026-11-19. Accept
    real booleans, numbers, and the common string spellings; anything
    else returns None so the caller can refuse to record.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "yes", "1", "ok", "worked"):
            return True
        if v in ("false", "no", "0", "fail", "failed"):
            return False
    return None


def skill_outcome(name: str, worked: bool, note: str = "") -> str:
    """Report whether a skill you used actually worked.

    Call this after finishing a task where you followed a skill.
    Honest failure reports matter: they are how bad skills get found
    and removed, and how your approval record is built.

    Args:
        name: the skill name you used.
        worked: True if following the skill led to a good result.
        note: one short line about what happened.
    """
    parsed = _parse_worked(worked)
    if parsed is None:
        return ("Not recorded: 'worked' must be true or false "
                f"(got {worked!r}). Call skill_outcome again with a "
                "clear true/false.")
    skillgate.record_outcome(name, parsed, note)
    return (f"Recorded: skill '{name}' "
            f"{'worked' if worked else 'did not work'}.")


ALL_TOOLS = [propose_skill, use_skill, skill_outcome]
