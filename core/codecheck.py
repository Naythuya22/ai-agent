"""Release 31: deterministic Python-syntax gate for local-model replies.

The on-device model (Gemma 4 E4B) answers simple chat well but, on
multi-constraint code requests, produces code whose shape is broken —
classically a ``for`` loop and the final ``return`` dedented out of the
function, which is a SyntaxError before it is anything else (owner's
own phone test, 2026-10-25: two rounds, both unrunnable, and a spoken
correction from the user did not fix either). A human correction is a
long Burmese paragraph; a machine check is one fact. This module is
that fact, applied once:

* a finished LOCAL reply that is predominantly Python code and does
  not parse gets ONE repair attempt — the model is told the exact
  error line and asked for the corrected code only;
* the repair replaces the original only when it parses; a repair that
  still fails leaves the original untouched (the user sees what the
  model first said, never something silently worse);
* prose replies, valid code and fenced snippets inside prose are
  never touched, and the injected ``send`` is never called for them.

Local path only by construction: the two call sites are the local
tails of ``AgentMode.handle()`` and of the v1 SSE route (v2 is
cloud-only and 404s local turns). No cloud method calls this module,
so a cloud answer cannot be re-asked, delayed or rewritten by it.
``AI_AGENT_CODE_GATE=off`` disables it, mirroring the Release 26
Burmese quality gate's kill switch.
"""
from __future__ import annotations

import os
import re

from core.log import get_logger

log = get_logger("core.codecheck")

_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)
_OPENERS = ("def ", "import ", "from ", "class ", "@", "if ", "for ",
            "while ", "try:", "with ")


def gate_enabled() -> bool:
    return os.environ.get("AI_AGENT_CODE_GATE", "").lower() != "off"


def extract_code(text: str):
    """The Python code a reply carries, or None when it is not one.

    A fenced block wins. Otherwise the reply itself counts as code
    only when it *starts* as code and defines something — a prose
    answer that merely mentions ``def`` mid-paragraph is not a code
    reply and is never gated.
    """
    if not text or not text.strip():
        return None
    m = _FENCE.search(text)
    if m:
        return m.group(1).strip("\n") or None
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    first = lines[0].lstrip()
    if "def " not in text:
        return None
    if not first.startswith(_OPENERS):
        return None
    return text.strip()


def syntax_error(code: str):
    """``(lineno, message)`` of the first syntax error, else None.

    ``compile()``, not ``ast.parse()``: parsing alone accepts a
    top-level ``return`` (the check lives in the symbol-table pass),
    and the escaped ``return`` is exactly the failure this gate
    exists to catch.
    """
    try:
        compile(code, "<local-reply>", "exec")
    except SyntaxError as e:
        return (e.lineno or 0, e.msg or "invalid syntax")
    except (ValueError, MemoryError):
        return (0, "unparseable input")
    return None


def correction_prompt(err) -> str:
    line, msg = err
    where = " on line %d" % line if line else ""
    return (
        "The Python code you just gave has a syntax error%s: %s. "
        "Reply with the corrected code ONLY — no explanation, no "
        "markdown fences. Keep every line of the function body "
        "indented inside the function, and the final return inside "
        "the function, after the loop." % (where, msg)
    )


def repair(text: str, send) -> str:
    """One repair attempt for a broken local code reply.

    ``send(prompt)`` must return the model's next reply text (or
    None/empty on any trouble). Returns ``text`` unchanged unless the
    reply is broken code AND the repair parses.
    """
    if not gate_enabled() or not text:
        return text
    code = extract_code(text)
    if code is None:
        return text
    err = syntax_error(code)
    if err is None:
        return text
    try:
        cand = send(correction_prompt(err))
    except Exception as e:
        log.info("code gate: repair send failed (%s: %s) — keeping "
                 "the original", type(e).__name__, e)
        return text
    if not cand:
        return text
    cand_code = extract_code(cand)
    if cand_code is None or syntax_error(cand_code) is not None:
        log.info("code gate: repair still does not parse — keeping "
                 "the original")
        return text
    log.info("code gate: repaired a local code reply (was: line %s: %s)",
             err[0], err[1])
    return cand
