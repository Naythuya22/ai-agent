"""Agent mode — native tool calling (LiteRT automatic loop)."""
from __future__ import annotations

import base64
import json
import re
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import config
from litert_lm import tool_from_function
from core import prompts as prompt_tiers
from core import safety
from core.llm import LLM
from core.log import get_logger
from tools import ALL_TOOLS

log = get_logger("mode.agent")

try:
    from sessions import SessionStore
except ImportError:
    SessionStore = None

# ── what actually answered the last turn (drives the status-bar chip) ──
#
# Module state on purpose: the indicator must survive rebuild_modes() throwing
# the AgentMode instance away, and the SSE stream records its turn without one.
# It is only ever written from an Agent Mode path — Chat Mode never touches it.
#
# tier/tier_source are written by record_tier() once, when the turn's prompt is
# resolved. record_usage() updates the other keys in place and never touches
# them, so a cloud step that is retried on another provider cannot erase which
# prompt the turn was given.
_USAGE = {"backend": "", "model": "", "provider": "", "mode": "",
          "tier": "", "tier_source": "", "effort": "", "effort_source": "",
          "label": "", "at": 0.0}


def usage_label(backend: str, model: str, mode: str) -> str:
    """``Model: deepseek-chat (Cloud, last used)`` / ``(Local, fallback)``.

    Built from what ran, never from config: a Cloud Only setting with no key
    must not be able to print the name of a model nothing ever called.
    """
    backend = (backend or "").strip()
    if backend == "cloud":
        return "Model: %s (Cloud, last used)" % ((model or "").strip() or "cloud model")
    if backend == "local":
        name = (model or "").strip() or "Local model"
        suffix = "Local" if (mode or "") == "local" else "Local, fallback"
        return "Model: %s (%s)" % (name, suffix)
    return ""


def record_usage(backend: str, model: str = "", provider: str = "",
                 mode: str = "") -> dict:
    """Remember the backend/model that really produced a reply."""
    _USAGE.update({
        "backend": (backend or "").strip(),
        "model": (model or "").strip(),
        "provider": (provider or "").strip(),
        "mode": (mode or "").strip(),
        "label": usage_label(backend, model, mode),
        "at": time.time(),
    })
    return dict(_USAGE)


def last_usage() -> dict:
    """Copy of the record — safe to serialise straight into a response."""
    return dict(_USAGE)


def meter_usage(backend: str, model: str = "", provider: str = "",
                usage: dict | None = None) -> None:
    """Persist one completed call in the usage ledger (Release 17).

    Called beside record_usage() at exactly the points where a cloud
    step or a local turn completes — so a call is metered exactly when
    the chip says it ran, and never for the reviewer's private steps
    (record_usage is deliberately not called there either). The
    HTTP-transport branch of handle() does NOT meter: the server that
    answered already metered its own side, and counting both ends of
    that one turn would double it. Fully guarded — a broken ledger
    must never break a turn.
    """
    try:
        from core import usage_store
        usage_store.record_call(backend, model, provider, usage)
    except Exception as e:
        log.debug("usage metering failed: %s: %s", type(e).__name__, e)


def record_tier(tier: str, source: str) -> dict:
    """Remember which system prompt this turn was given, and why.

    Separate from record_usage() because it happens once, before any model
    runs: a tier that changed its mind half a turn through would be a lie, and
    the per-step usage updates must not be able to overwrite it.
    """
    _USAGE["tier"] = (tier or "").strip()
    _USAGE["tier_source"] = (source or "").strip()
    return dict(_USAGE)


def record_effort(effort: str, source: str) -> dict:
    """Remember the effort this turn was routed to (Release 28).

    The effort twin of record_tier(): written once when the turn's
    prompt is resolved, never overwritten by per-step usage updates.
    """
    _USAGE["effort"] = (effort or "").strip()
    _USAGE["effort_source"] = (source or "").strip()
    return dict(_USAGE)


AGENT_SYSTEM = """You are "ညီ" (Nyi), a personal AI assistant for "ကိုကြီး" (Ko Gyi).
Your tone must be warm, respectful, and brotherly.

## Language Rules
- Burmese is your mother tongue. Think, understand, and reply in Burmese.
- Use "ညီ" for yourself, "ကိုကြီး" for the user.
- English is ONLY for technical terms, code, file paths, and commands.
- No "ခင်ဗျာ". No "ရှင့်". Never mix languages within a sentence.
- Write natural Burmese — not English sentences translated word-by-word.

## Response Style
- No greetings. No "မင်္ဂလာပါ". Start with the answer.
- If acknowledgment needed: just "ဟုတ် ကိုကြီး".
- Answer as fully as the question deserves.
- Have opinions. Be genuinely helpful. Skip generic greetings.
- If unsure, say you don't know. Never make things up.

## Tool Use Rules (CRITICAL)
- If the user asks for counts, listings, file contents, or system info -> MUST use tools. NEVER guess.
- read/search/run files -> use read_file, run_command, run_python.
- search web/facts/news -> use web_search or smart_search.
- set alarm/timer/calendar -> call the matching tool immediately.
- After a tool returns, base your answer ONLY on the tool output.

## Analysis & Recommendation (ALLOWED)
- You ARE allowed to analyze tool output, compare it to what a
  good agent needs, and recommend concrete next steps. This is not
  a "tool" — it is reasoning, and you do it in plain text.
- For "what are your weaknesses / what should be added" questions:
    STEP 1 — Call capability_list() BEFORE writing any reply.
             If you answer without this tool, the answer is wrong.
    STEP 2 — Read its output verbatim.
    STEP 3 — List ONLY the entries it reports as not implemented.
             You may add one recommendation line per entry.
    STEP 4 — You may NOT list a capability as missing unless the
             tool output says so. No guessing.
- Do NOT claim you lack the ability to analyze or recommend. You
  have it. Use the tool to get facts, then reason on them.
- Never write a negative self-statement ("I cannot ...", "I am
  only able to ...") to memory as a fact. Facts are about the
  world, not about your own limits.

## Tool Output Fidelity (CRITICAL)
- A tool's return value is the only thing you know about its result.
  Copy exactly what it says. Never invent names, versions, reasons,
  or library requirements that the tool did not print.
- If capability_list() says "not implemented", say "not implemented".
  Do not add a cause ("missing library", "not imported") unless that
  cause appears in the tool output itself.
- Do not repeat the same answer across turns just because the user
  asked again with different wording. Give the shortest honest answer.

## Brevity (applies to every reply)
- Answer first, supporting lines after. Two lines beats ten.
- Do not restate the user's question.
- If a list was already shown this session, refer to it — do not
  repaste it.

## Tool Failure Rule (CRITICAL)
- If a tool call returns "Error:" or a failure message, you MUST NOT
  describe what its result "would have been". A failed read is a
  failed read — do not fill the gap from memory or training.
- Allowed responses to a failed tool:
    * say the file/content is unavailable,
    * ask the user for a correct path or target,
    * try a different tool that actually succeeds.
- Never paraphrase a file you did not see. Never invent file names.

## Task Handling
- Simple task: Just do it.
- Complex task: Break down and confirm plan first."""


# ── answer verification (cloud only) ───────────────────────────────────────
#
# After a tool turn the model has already committed to an answer. One cheap
# reviewer pass checks that answer against the tool output, which is the cheapest
# place to catch a hallucination: the evidence is still in the transcript.
#
# _VERIFY_EXTRA_ROUNDS is a hard 1 and the reviewer is called from exactly one
# place, never from inside itself, so recursion is impossible by construction.
_VERIFY_EXTRA_ROUNDS = 1

# One entry per tool result in the reviewer prompt, and a ceiling on the whole
# block — the reviewer is a check, not a second reading of the conversation.
_VERIFY_ENTRY_CHARS = 800
_VERIFY_TOTAL_CHARS = 4000

_VERIFY_SYSTEM = (
    "You are a reviewer. You see a candidate answer and the raw tool results. "
    "Reply with ONE of:\n"
    "  OK            -> the answer is fully supported\n"
    "  FIX: <answer> -> the corrected answer, using ONLY the tool results\n"
    "  MISSING: <q>  -> the answer needs information that is not present\n"
    "Never invent data. Never add facts that are not in the tool results. "
    "Reply with exactly one of the three forms and nothing else."
)

# The user-side half of the summary pair compress_session writes: a
# compressed transcript (manual or automatic) always opens with this
# marker entry followed by an assistant entry holding the summary text.
# Auto-compaction (Release 18) writes the same pair, so a transcript the
# two paths produced is indistinguishable downstream. Release 29: the
# marker carries the condensed framing in Burmese too — the summary it
# introduces is a compression, and both the user reading the history
# and the model answering later turns should know detail may be gone.
_SUMMARY_MARKER = (
    "[Earlier conversation — summary] "
    "(အကျဉ်းချုပ်: အစောပိုင်း စကားဝိုင်း ကို ချုံ့ထားတာ ဖြစ်လို့ "
    "အသေးစိတ် အချို့ မပါနိုင်ပါ)")

# Release 29: the pair _cap_history writes when auto-compaction could
# not summarize (summarizer failed, or its summary failed the fact
# check) and the plain trim dropped entries instead. Before R29 that
# loss was silent — the transcript simply started later. The pair
# makes it visible in the history itself, in the same shape as the
# summary pair so nothing downstream has to special-case it.
_TRIM_MARKER = "[Earlier conversation — trimmed without summary]"
_TRIM_NOTE = (
    "🗂️ မှတ်ချက်: မှတ်တမ်း ရှည်လာလို့ အစောပိုင်း စကားဝိုင်း အချို့ကို "
    "အကျဉ်းချုပ် မလုပ်နိုင်ဘဲ ဖြတ်လိုက်ရပါတယ် — အဲ့ဒီ အပိုင်းတွေ "
    "အကုန် မမှတ်မိနိုင်တော့ပါ။ လိုအပ်တာရှိရင် ပြန် ပြောပြပေးပါ။")

# System prompt for the cloud summarizer step (Release 18): a cloud
# turn's summary is written by the backend that served the turn, under
# the same faithfulness discipline as the reviewer — a loose summary
# would poison every later turn that reads it back as history.
_SUMMARIZE_SYSTEM = (
    "You are a careful summarizer. Summarize the conversation you are "
    "given faithfully and concisely. Preserve facts, decisions and "
    "open questions exactly as stated; never add, guess or embellish "
    "anything that is not in the conversation."
)

_REFLECT_PROMPT = """You are reviewing one finished turn to decide
whether a DURABLE lesson is worth saving to persistent memory.

CURRENT MEMORY (do not duplicate):
{memory_summary}

THIS TURN
User asked: {user_message}
Tools called: {tool_names}
Errors hit: {error_count}
Final reply (first 150 chars): {final_excerpt}

SAVE a NOTE only for a DURABLE lesson:
  * user preference or constraint ("prefers X", "wants Y only")
  * a repeated error pattern ("tool T fails when Z")
  * a project-level fact ("this repo uses W")
  * a workflow rule you plan to follow again
Anything that fits in the identity/profile sections belongs
THERE, not in facts.

NEVER save:
  * anything already in CURRENT MEMORY above (no paraphrase-
    duplicates)
  * a description of THIS conversation ("user is asking...",
    "user is requesting...", "user wants to learn...")
  * a one-time answer, timestamp, or question
  * a claim about who the user or assistant IS if it contradicts
    the identity/profile section (trust the profile)
  * the current turn's counts, tokens, or latency

Reply with EXACTLY one of:
  NOTE: <one line, <= 200 chars>
  OK
"""


# Batch 4.2h: markers that mean "the user is asking about capabilities"
_CAPABILITY_QUERY_MARKERS = (
    "အားနည်းချက်", "အားသာချက်", "လိုအပ်ချက်", "လုပ်ဆောင်နိုင်စွမ်း",
    "စွမ်းရည်", "ဘာလိုအပ်", "ဘာတွေလိုအပ်",
    "weakness", "weaknesses", "gap", "gaps", "limitation",
    "capability", "capabilities", "what can you", "what are your",
    "what should be added", "what is missing",
)
_TOOL_REQUIRED_FOR_CAPABILITY = ("capability_list", "self_audit")


def _is_capability_query(text: str) -> bool:
    """True when the user is asking about the agent's own abilities."""
    if not text:
        return False
    low = text.lower()
    return any(m.lower() in low for m in _CAPABILITY_QUERY_MARKERS)


def _capability_nudge() -> str:
    """Injected into the next user turn when no capability tool ran yet."""
    return (
        "\n\n[SYSTEM REMINDER — read this before answering]\n"
        "The previous reply answered a capability question WITHOUT "
        "calling the required tool. You must call capability_list() "
        "NOW and answer from its output. Do NOT list capabilities or "
        "weaknesses from memory. The tool is the only source of truth."
    )


def _decide_verify(reply: str, candidate: str) -> tuple:
    """``(verdict, value)`` from the reviewer's raw reply.

    verdict is ``OK``, ``FIX`` or ``MISSING``. Anything unreadable is ``OK``:
    a broken review must never cost the user their answer, so the candidate
    stands unless the reviewer clearly said otherwise.
    """
    text = str(reply or "").strip()
    if not text:
        return "OK", candidate
    low = text.lower()
    # FIX and MISSING are checked before OK so a reply that merely starts with
    # "OK ..." cannot shadow a correction.
    if low.startswith("fix:"):
        fixed = text[4:].strip()
        return ("FIX", fixed) if fixed else ("OK", candidate)
    if low.startswith("missing:"):
        return "MISSING", text[8:].strip()
    return "OK", candidate


def _compact_tool_results(messages: list) -> str:
    """This turn's tool results, in order, each capped — what a reviewer sees.

    Secrets are stripped here as well as in the transcript: this block is the
    one place tool output is deliberately re-quoted to a model.
    """
    parts = []
    for m in messages:
        if m.get("role") != "tool_result":
            continue
        txt = safety.redact_secrets(str(m.get("text", "")))
        if len(txt) > _VERIFY_ENTRY_CHARS:
            txt = txt[:_VERIFY_ENTRY_CHARS] + "…"
        parts.append("[%s] %s" % (m.get("tool", "?"), txt))
    out = "\n".join(parts)
    if len(out) > _VERIFY_TOTAL_CHARS:
        out = out[:_VERIFY_TOTAL_CHARS] + "\n…"
    return out


def _summary_prompt(msgs: list) -> str:
    """The one bullet-summary prompt every summarizer is fed (Release 18).

    Renders ``msgs`` as ``role: text`` lines and asks for a 3–5 bullet
    summary. The local summarizer (``_summarize_messages``) and the
    cloud one (``_summarize_messages_cloud``) share this builder, so
    which engine writes a summary never changes what it was asked for —
    and manual compress_session's prompt is byte-identical to before.
    """
    convo = "\n".join(
        f"{m.get('role', 'user')}: {m.get('text', '')}" for m in msgs
    )
    return (
        "Summarize the following conversation in 3-5 concise bullet points. "
        "Preserve key facts, user questions, decisions, and context needed to continue.\n\n"
        f"{convo}\n\nSummary:"
    )


# ── Release 29: compaction fact check + live-turn image payload ──
#
# Round 5 (P4) proved the machinery worked but nothing checked WHAT a
# summarizer handed back: a lossy one ("various topics") replaced the
# head and the early facts vanished inside a "successful" compaction.
# The check below is deliberately cheap and deterministic — salient
# tokens only, no second model call: numbers (3+ digits), Latin words,
# hyphenated codenames (Zebra-42) and long Burmese tokens found in the
# dropped entries must mostly survive into the summary. It is a floor
# against pathological loss, NOT a quality judge; a fluent summary
# that keeps the checkable facts passes untouched.
#
# Tokenisation notes: a codename like "Zebra-42" is checked through
# its alpha part ("zebra", from the Latin rule) — a dedicated
# codename token would make every member of a numbered series
# (seed-0, seed-1, …) a distinct must-keep token no summary could
# ever carry, and the 3-digit floor keeps dates' fragments and small
# counts out of the number rule the same way.

_SALIENT_MIN_OVERLAP = 0.2


def _salient_tokens(text: str) -> set:
    """The checkable tokens of one text (see the note above)."""
    text = str(text or "")
    out = set()
    for num in re.findall(r"\d{3,}(?:,\d{3})*", text):
        out.add(num.replace(",", ""))
    for word in re.findall(r"[A-Za-z]{3,}", text):
        out.add(word.lower())
    for token in re.split(r"\s+", text):
        burmese = sum(1 for ch in token if "\u1000" <= ch <= "\u109f")
        if burmese >= 5:
            out.add(token.strip(".,;:!?()[]\"'"))
    return {t for t in out if t}


def _summary_keeps_facts(dropped: list, summary: str) -> bool:
    """True when ``summary`` keeps a sane share of the dropped facts.

    No salient tokens in the dropped entries ⇒ nothing checkable ⇒
    pass (the summarizer is trusted, as before R29). Otherwise at
    least ``_SALIENT_MIN_OVERLAP`` of them must appear in the summary;
    below that the summary is treated as a compaction failure, so the
    caller falls back to raw history + the visible trim marker rather
    than entombing the loss inside a plausible-looking summary.
    """
    salient = set()
    for m in dropped or []:
        salient |= _salient_tokens((m or {}).get("text", ""))
    if not salient:
        return True
    flat = str(summary or "").lower().replace(",", "")
    kept = sum(1 for t in salient if t in flat)
    return kept / len(salient) >= _SALIENT_MIN_OVERLAP


def _wire_image(image) -> dict:
    """The live-turn image payload stored on the user entry.

    Accepts {"data", "mime"} ready-made or {"path", "mime"} (read and
    base64-encoded here, once, so later steps of the turn reuse the
    same bytes). The payload NEVER persists: _save_turn and
    _save_checkpoint write _durable() copies without it — stored
    sessions keep the text-only design, so a replay never needs the
    staged file back (Release 29).
    """
    if not isinstance(image, dict):
        raise ValueError("image must be a dict")
    data = image.get("data")
    mime = image.get("mime") or "image/png"
    if not data:
        path = image.get("path")
        if not path:
            raise ValueError("image carries neither data nor path")
        data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return {"data": data, "mime": mime}


def _wait_phrase(wait: int) -> str:
    """The "(~…)" suffix naming a park's remaining wait in a failure
    line.

    Under an hour the wait stays in seconds, as it has always
    rendered. A DAILY-quota park (Release 30) runs until the
    provider's next daily reset — hours away — and printing it as a
    raw seconds figure ("(~73800 စက္ကန့်)") is a number no user can
    act on, so long waits read in hours.
    """
    if wait < 3600:
        return "(~%d စက္ကန့်)" % wait
    hours = wait / 3600.0
    text = ("%d" % hours) if hours == int(hours) else ("%.1f" % hours)
    return "(~%s နာရီ)" % text


class AgentMode:
    """Agent mode using LiteRT native automatic tool calling."""

    name = "agent"

    def __init__(self, llm: LLM, memory=None):
        self.llm = llm
        self.memory = memory
        self.turns = 0
        self._conv = None
        # Cloud backend is built lazily: a device that never has a network
        # should not pay for building it, and the local path must stay the
        # default when anything at all is wrong.
        self._backend = None
        self._tools = [tool_from_function(fn) for fn in ALL_TOOLS]
        self._store = SessionStore() if SessionStore else None
        self.session_id = None
        self._messages: list = []
        self._server_ctx = None  # Bug #4: HTTP-mode server ctx cache
        # Per-turn tool result reuse. Instance state on purpose: an entry is
        # only ever true for the turn that created it, and it must die with the
        # object — a module global would leak one user's files into the next
        # user's turn.
        self._turn_cache: dict = {}
        self._turn_cache_order: list = []
        # The prompt tier resolved for the turn in flight — None between turns.
        # See _system_prompt().
        self._turn_prompt: tuple | None = None
        # This turn's user message, for the effort router. See begin_turn().
        self._turn_user_text: str = ""
        # Tool-loop iterations spent this turn — the "iter" half of the budget
        # hint. Reset alongside the prompt tier in begin_turn().
        self._turn_iter = 0
        self._turn_tools: list = []   # tool names used this turn
        self._turn_errors: int = 0    # "Error:" results this turn
        # C5a: In HTTP mode, server owns session state — do not touch store.
        if self._store and not self._is_http():
            try:
                self.session_id = self._store.get_active_id("agent")
                if self.session_id is None:
                    self.session_id = self._store.create_session("agent")
                self._messages = self._store.get_mode("agent")
                self.turns = len(self._messages) // 2
                # Release 19: a checkpoint left by a process that died
                # mid-turn resumes here — its transcript replaces the
                # last completed turn's (see _restore_checkpoint).
                self._restore_checkpoint()
            except Exception as e:
                log.error("could not load agent history: %s: %s", type(e).__name__, e)
                self._messages = []

    def _is_http(self) -> bool:
        from core.http_llm import HTTPLLM
        return isinstance(self.llm, HTTPLLM)

    # ── prompt tier, once per turn ───────────────────────────────────────
    #
    # The tier is a property of the turn, not of the step. A turn that starts
    # on a cloud model and falls back to the on-device one keeps the prompt it
    # started with: swapping the system message halfway through would mean two
    # different instructions were given for one question. The next turn
    # re-resolves, so a Settings change takes effect on the next message.

    def begin_turn(self, user_text: str = "") -> None:
        """Mark the start of a user turn.

        Called by handle() and by the SSE route, which drives the cloud/local
        branches itself and so never goes through handle(). Between turns this
        is None, which is what makes the first _system_prompt() of a turn
        resolve and every later one reuse.

        ``user_text`` is the turn's message, kept only so the effort
        router (Release 28) can read its task signals when the prompt
        resolves; callers that do not have it pass nothing and the
        router falls back to the balanced default.
        """
        self._turn_user_text = user_text or ""
        self._turn_prompt = None
        self._turn_iter = 0
        self._turn_tools = []
        self._turn_errors = 0
        self._cap_nudged = False   # Batch 4.2h: one nudge per turn

    def _system_prompt(self) -> tuple:
        """``(system_prompt, tier)`` for this turn — resolved once, then frozen.

        The backend is only asked what it *would* do (mode, cloud eligibility,
        which provider is first), never told to do anything: this must not be
        able to change which model answers.

        Budget note: when ``config.AGENT_BUDGET_HINT`` is on, ``_budget_hint()``
        is appended to the prompt string *before* the tuple is stored, so the
        note is frozen together with the prompt. Turn-freeze already freezes
        the resolved prompt — the hint inherits that behaviour automatically:
        the model gets one consistent budget line for the whole turn, and
        nothing that happens mid-turn (a Settings write, the iteration counter
        moving) can rewrite a prompt the model has already been given. The
        tier logic below is untouched by the hint.
        """
        if self._turn_prompt is None:
            backend, model = "", ""
            try:
                backend, model = prompt_tiers.plan_target(self._backend_obj())
            except Exception as e:
                log.warning("prompt tier: backend probe failed (%s: %s)",
                            type(e).__name__, e)
            mode = config.prompt_tier_mode()
            tier, source = prompt_tiers.resolve_tier(backend, model, mode=mode)
            auto_tier = prompt_tiers.classify_tier(backend, model)
            if mode in prompt_tiers.TIERS and mode != auto_tier:
                # Worth a log line and a badge in Settings: the user pinned a
                # tier that does not match what this turn is actually running.
                # (Effort pins — fast/balanced/pro — are not tiers and can
                # never mismatch this way; they apply to any tier.)
                log.warning("prompt tier %s is pinned but this turn looks like "
                            "%s (%s/%s) — using the pinned prompt", mode,
                            auto_tier, backend or "?", model or "?")
            # Release 28 — the second axis: effort, from the same mode
            # (an override) or from this turn's task signals (Auto).
            effort, effort_source = prompt_tiers.resolve_effort(
                backend, getattr(self, "_turn_user_text", ""), mode=mode)
            prompt = prompt_tiers.load_prompt(tier, effort)
            hint = self._budget_hint()
            if hint:
                prompt = prompt + "\n\n" + hint
            self._turn_prompt = (prompt, tier)
            record_tier(tier, source)
            record_effort(effort, effort_source)
            log.info("prompt tier %s (source %s), effort %s (source %s) "
                     "for %s/%s", tier, source, effort, effort_source,
                     backend or "?", model or "?")
        return self._turn_prompt

    def _budget_hint(self) -> str:
        """One-line budget note. Empty when AGENT_BUDGET_HINT is off.

        Numbers and fixed text only — no user input, no file contents, no
        credentials, and no newline can reach the prompt through it.
        """
        if not config.AGENT_BUDGET_HINT:
            return ""
        pct = 0
        try:
            backend = self._backend_obj()
            if backend is not None:
                st = backend.context_state() or {}
                pct = int(st.get("pct", 0) or 0)
        except Exception:
            pct = 0
        used = getattr(self, "_turn_iter", 0) or 0
        try:
            maxi = config.agent_max_iter(self.name)
        except Exception:
            maxi = 0
        note = "[BUDGET: ctx %d%% iter %d/%d]" % (pct, used, maxi)
        if pct >= getattr(config, "AGENT_BUDGET_WARN_CTX", 80):
            note += " — be concise"
        left = maxi - used if maxi else 0
        if maxi and left <= getattr(config, "AGENT_BUDGET_WARN_ITER", 5):
            note += " — wrap up soon"
        return note

    def _reflect_call(self, prompt: str) -> str:
        """One cheap LLM call for the reflection step.

        Cloud-only for now (Part 3.2a). Returns "" on any failure —
        never raises, never touches self._messages, never runs tools.
        Local reflection is deferred to a later part.
        """
        be = self._backend_obj()
        if be is None:
            return ""
        try:
            if not be.should_use_cloud():
                return ""
            step = be.step(
                system="",
                messages=[{"role": "user", "text": prompt}],
                tools=[],
                max_tokens=config.AGENT_REFLECT_MAX_TOKENS,
            )
            return (getattr(step, "text", "") or "").strip()
        except Exception as e:
            log.debug("reflect call failed: %s: %s", type(e).__name__, e)
            return ""

    def _reflect(self, *, user_message: str, tool_names: list,
                 error_count: int, final_text: str) -> str | None:
        """After a turn, decide whether a durable lesson should be saved.

        The prompt now embeds the CURRENT memory summary (identity/profile
        plus facts) so the model can see what is already stored and avoid
        paraphrase-duplicates, and the candidate NOTE passes a validation
        gate before memory_write: transient markers ("the user is ..."),
        reversed identity claims, and a cheap near-duplicate containment
        test against the existing summary all reject the lesson.

        Returns the saved lesson text (str) if memory_write accepted
        it, or None if nothing was saved. Never raises, never recurses,
        never changes the transcript.

        The caller (Part 3.2b) decides when to invoke this — it is NOT
        called anywhere in 3.2a.
        """
        if not config.AGENT_REFLECT_ENABLED:
            return None
        try:
            try:
                from tools import memory as _M  # local import: avoid cycle
                memory_summary = _M.memory_read()
            except Exception:
                memory_summary = "(unavailable)"
            memory_summary = str(memory_summary or "(unavailable)")[:800]
            excerpt = (final_text or "")[:150].replace("\n", " ")
            tools_str = ", ".join(tool_names) if tool_names else "(none)"
            prompt = _REFLECT_PROMPT.format(
                user_message=(user_message or "")[:200],
                tool_names=tools_str,
                error_count=int(error_count or 0),
                final_excerpt=excerpt,
                memory_summary=memory_summary,
            )
            reply = self._reflect_call(prompt)
            if not reply:
                return None
            up = reply.strip().upper()
            if up.startswith("NOTE:"):
                lesson = reply.strip()[5:].strip()[:200]
                if not lesson:
                    log.debug("reflect: empty NOTE, skipped")
                    return None
                reason = None
                low = lesson.lower()
                _TRANSIENT = (
                    "user is asking", "user is requesting",
                    "user wants to learn", "user asked about",
                    "user wants to", "the user is",
                    "the user has requested", "this turn",
                    # Batch 4.2d: more transient phrasings observed live
                    "user is interested", "user is curious",
                    "user would like", "user needs",
                    "user hopes", "user expects",
                    "user is looking", "the user wants",
                )
                for marker in _TRANSIENT:
                    if marker in low:
                        reason = "transient marker '%s'" % marker
                        break
                if reason is None and (
                        "ကိုကြီး to refer to the ai" in low
                        or "ညီ to refer to themselves" in low):
                    reason = "reversed identity"
                if reason is None:
                    head = lesson[:40]
                    if head:
                        for line in memory_summary.splitlines():
                            if head in line.strip():
                                reason = "near-duplicate"
                                break
                if reason:
                    log.debug("reflect rejected: %s — %s", reason,
                              lesson[:80])
                    return None
                from tools import memory as _M  # local import: avoid cycle
                result = _M.memory_write(lesson, "fact")
                if result.startswith("Error:"):
                    log.warning("reflect write refused: %s", result)
                    return None
                log.info("reflect saved: %s", lesson)
                return lesson
            # "OK" or anything else -> nothing saved
            log.debug("reflect: no lesson (%s)", up[:40])
            return None
        except Exception as e:
            log.debug("reflect failed: %s: %s", type(e).__name__, e)
            return None

    def _finish_turn(self, user_message: str, final_text: str) -> None:
        """Run the reflection step after a user turn. Never raises."""
        try:
            self._reflect(
                user_message=user_message,
                tool_names=list(self._turn_tools),
                error_count=self._turn_errors,
                final_text=final_text or "",
            )
        except Exception as e:
            log.debug("finish_turn reflect failed: %s: %s",
                      type(e).__name__, e)

    def _with_memory(self, system: str) -> str:
        """The user's profile hint appended to a prompt, when there is one."""
        if self.memory is None:
            return system
        try:
            hint = self.memory.get_system_hint() or ""
        except Exception as e:
            log.debug("memory hint unreadable: %s", type(e).__name__)
            return system
        return f"{system}\n\n{hint}" if hint else system

    def _ensure_conv(self, system: str = "") -> None:
        if self._is_http():
            return
        if self._conv is None:
            self.llm.load()
            assert self.llm._engine is not None
            sys_msg = self._with_memory(system or self._system_prompt()[0])
            replay = None
            if self._messages:
                replay = []
                for m in self._messages:
                    role = m.get("role", "user")
                    if role in ("user", "assistant"):
                        txt = m.get("text", "")
                        if txt:
                            replay.append({"role": role, "content": txt})
                    elif role == "tool_result":
                        # Tool result → role "tool" for replay
                        txt = m.get("text", "")
                        if txt:
                            replay.append({
                                "role": "tool",
                                "name": m.get("tool", ""),
                                "content": txt,
                            })
                    # Skip "tool_call" entries in replay (only results carry info)
                if not replay:
                    replay = None
            self._conv = self.llm._engine.create_conversation(
                messages=replay,
                tools=self._tools,
                automatic_tool_calling=False,  # Bug #3: manual loop for transparency
                system_message=sys_msg,
            )

    # ── per-turn tool result cache ──────────────────────────────────────
    #
    # The tool loops already refuse to execute the same call twice in a turn
    # (the _seen_calls guard) — but they answer the repeat with a bare
    # "(skipped: …)", so the model loses the data it asked for. This returns
    # the data instead, tagged so the model knows it is a repeat.
    #
    # Read-only tools only, listed by their real names in tools/: write_file
    # and run_python change the world and are never cached, and anything not on
    # this list is not cached either.
    _CACHEABLE_TOOLS = frozenset({
        "list_files",     # tools/file.py
        "read_file",      # tools/file.py
        "web_search",     # tools/web_search.py
        "web_fetch",      # tools/web_search.py
        "smart_search",   # tools/smart_search.py
        "run_command",    # tools/shell.py — read-only allowlist by construction
    })
    _CACHE_CAP = 64

    def _clear_turn_cache(self) -> None:
        """Drop every cached result. Called at the start of each user turn."""
        self._turn_cache.clear()
        self._turn_cache_order.clear()

    def _cache_key(self, tool_name: str, tool_args) -> str:
        """``(tool, canonical kwargs)`` — argument order never makes a new key."""
        try:
            return "%s|%s" % (tool_name, json.dumps(tool_args, sort_keys=True,
                                                   default=str))
        except Exception:
            return "%s|%s" % (tool_name, sorted((str(k), str(v))
                                                for k, v in dict(tool_args).items()))

    def _cache_peek(self, tool_name: str, tool_args):
        """The stored result for this exact call, or None."""
        if not config.AGENT_TURN_CACHE:
            return None
        if tool_name not in self._CACHEABLE_TOOLS:
            return None
        return self._turn_cache.get(self._cache_key(tool_name, tool_args))

    def _cache_put(self, tool_name: str, tool_args, value: str) -> None:
        """Remember a result, dropping the oldest once the cap is reached."""
        if not config.AGENT_TURN_CACHE:
            return
        if tool_name not in self._CACHEABLE_TOOLS:
            return
        key = self._cache_key(tool_name, tool_args)
        if key not in self._turn_cache:
            while len(self._turn_cache_order) >= self._CACHE_CAP:
                self._turn_cache.pop(self._turn_cache_order.pop(0), None)
            self._turn_cache_order.append(key)
        self._turn_cache[key] = value

    def _cached_text(self, tool_name: str, tool_args, fallback: str) -> str:
        """``[cached] …`` when we have it, otherwise the caller's own text."""
        hit = self._cache_peek(tool_name, tool_args)
        if hit is None:
            return fallback
        log.debug("tool cache hit: %s", tool_name)
        return "[cached] " + hit

    def _guard_result(self, tool_name: str, result: str) -> str:
        """Mark instruction-shaped tool output, then take credentials out.

        The banner goes on first so the warning names the markers the source
        really contained; redaction runs last so a secret can never end up
        inside the banner text. Two redaction passes run, in order: the
        shape-based one in core.safety (no key list needed), then a
        known-key pass so the user's own registered keys are stripped
        whatever shape they have.
        """
        text = str(result)
        if config.AGENT_INJECTION_GUARD:
            markers = safety.scan_injection(text)
            if markers:
                log.warning("injection markers in %s output: %s",
                            tool_name, ", ".join(markers))
                text = (safety.INJECTION_BANNER_OPEN + "\n"
                        + text + "\n" + safety.INJECTION_BANNER_CLOSE)
        return self._redact_known_keys(safety.redact_secrets(text))

    def _redact_known_keys(self, text: str) -> str:
        """Mask every registered provider key still present in ``text``.

        safety.redact_secrets works on shapes; it cannot know a key whose
        shape is not in its list (a Gemini ``AQ.…`` key, say). The key
        manager can: each registered key is replaced by its mask, exactly
        as error bodies are already treated in _try_cloud. Best-effort by
        construction — no backend, no keys, or any failure leaves
        ``text`` untouched, and a bare test instance without __init__
        (no ``_backend`` attribute) lands in the except branch too.
        """
        try:
            be = self._backend_obj()
            if be is None:
                return text
            keys = be.keys.all_keys()
            if not keys:
                return text
            from core.key_manager import redact
            return redact(text, keys)
        except Exception as e:
            log.debug("known-key redaction skipped: %s", e)
            return text

    def _execute_tool(self, tool_name: str, tool_args) -> str:
        """Execute a tool by name; return string result (Bug #3 manual loop).

        Everything a tool result goes through on its way to the model happens
        here — cache, then error record, then injection banner, then redaction.
        Both loops call this one method and both put its return value straight
        into the transcript, so there is no path that skips the guards.
        """
        # Batch 3.2b: test stubs build AgentMode via __new__ (no __init__),
        # so the per-turn reflection counters may not exist yet. Create them
        # on first use — real instances get them from __init__/begin_turn.
        if not hasattr(self, "_turn_tools"):
            self._turn_tools = []
            self._turn_errors = 0
        tool_map = {fn.__name__: fn for fn in ALL_TOOLS}
        fn = tool_map.get(tool_name)
        if fn is None:
            self._turn_errors += 1  # Batch 3.2b: Error: result, tool never ran
            return f"Error: unknown tool: {tool_name}"
        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except Exception as e:
                self._turn_errors += 1  # Batch 3.2b
                return f"Error: bad args JSON: {e}"
        if not isinstance(tool_args, dict):
            self._turn_errors += 1  # Batch 3.2b
            return f"Error: args must be dict, got {type(tool_args).__name__}"

        hit = self._cache_peek(tool_name, tool_args)
        if hit is not None:
            log.debug("tool cache hit: %s", tool_name)
            # Batch 3.2b: a cache hit is a real result this turn too — track
            # it here, not after the fn call, or hits would never count.
            if tool_name not in self._turn_tools:
                self._turn_tools.append(tool_name)
            return "[cached] " + hit

        # Inject read budget for read_file only (Batch 1.5).
        if tool_name == "read_file":
            try:
                from tools import file as _file_mod
                backend = self._backend_obj()
                if backend is not None:
                    st = backend.context_state() or {}
                    _file_mod.set_read_context(
                        model=getattr(backend, "model", "") or "",
                        base_url=getattr(backend, "base_url", "") or "",
                        used_tokens=int(st.get("used", 0) or 0),
                    )
                else:
                    _file_mod.set_read_context()  # reset to safe default
            except Exception as e:
                log.debug("read_context inject failed: %s", e)

        try:
            out = str(fn(**tool_args))
        except TypeError as e:
            # Batch 3.2b: dispatched but failed — count name and error.
            if tool_name not in self._turn_tools:
                self._turn_tools.append(tool_name)
            self._turn_errors += 1
            return self._guard_result(
                tool_name, f"Error: bad arguments for {tool_name}: {e}")
        except Exception as e:
            if tool_name not in self._turn_tools:
                self._turn_tools.append(tool_name)
            self._turn_errors += 1
            return self._guard_result(tool_name, f"Error: {type(e).__name__}: {e}")

        # Batch 3.2b: track what ran this turn (for reflection)
        if tool_name not in self._turn_tools:
            self._turn_tools.append(tool_name)
        if out.startswith("Error:"):
            self._turn_errors += 1

        # An "Error: …" body is a failed call, not a result worth remembering:
        # caching it would make a transient failure look permanent for the turn.
        # Batch 4.2i-B (H3): cache the GUARDED text, not the raw
        # result — mirrors the batch path (L893-897). Otherwise a
        # cache hit returns "[cached] " + raw and skips the
        # injection banner + secret redaction.
        guarded = self._guard_result(tool_name, out)
        if not out.startswith("Error:"):
            self._cache_put(tool_name, tool_args, guarded)
        return guarded

    def _batch_parallel_ok(self, calls) -> bool:
        """True when a batch of model-shape tool_calls can run concurrently.

        Requirements (all must hold):
          - config.AGENT_PARALLEL_TOOLS is on
          - at least 2 calls
          - backend exposes tools_are_parallel_safe and returns True
          - no duplicate (name, args) pair within the batch itself
        """
        if not getattr(config, "AGENT_PARALLEL_TOOLS", False):
            return False
        if not calls or len(calls) < 2:
            return False
        be = self._backend_obj()
        if be is None:
            return False
        check = getattr(be, "tools_are_parallel_safe", None)
        if not callable(check):
            return False
        names = []
        keys = set()
        for tc in calls:
            fn_info = tc.get("function", {}) if isinstance(tc, dict) else {}
            name = fn_info.get("name", "")
            args = fn_info.get("arguments", {}) or {}
            names.append(name)
            try:
                key = (name, json.dumps(args, sort_keys=True))
            except Exception:
                key = (name, str(args))
            if key in keys:
                return False
            keys.add(key)
        try:
            return bool(check(names))
        except Exception:
            return False

    def _execute_tools_batch(self, calls):
        """Run model-shape tool_calls concurrently; return per-call dicts.

        Returns a list of {"name": str, "args": dict, "result": str,
        "error": str|None} in the SAME order as `calls`. Never raises.
        Applies _guard_result to every result and updates _turn_tools /
        _turn_errors for the batch.
        """
        if not hasattr(self, "_turn_tools"):
            self._turn_tools = []
            self._turn_errors = 0

        tool_map = {fn.__name__: fn for fn in ALL_TOOLS}
        be = self._backend_obj()

        # Resolve names / args from model shape
        resolved = []
        for tc in calls:
            fn_info = tc.get("function", {}) if isinstance(tc, dict) else {}
            name = fn_info.get("name", "")
            args = fn_info.get("arguments", {}) or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if not isinstance(args, dict):
                args = {}
            resolved.append({"name": name, "args": args})

        def _serial_fallback():
            out = []
            for item in resolved:
                r = self._execute_tool(item["name"], item["args"])
                out.append({"name": item["name"], "args": item["args"],
                            "result": r, "error": None})
            return out

        if be is None or not callable(getattr(be, "run_tools_parallel", None)):
            return _serial_fallback()

        # Batch 4.2c: inject read_file budget before parallel — the
        # serial path does this inside _execute_tool, but parallel
        # bypasses it, so callers otherwise see a stale or default cap.
        if any(item["name"] == "read_file" for item in resolved):
            try:
                from tools import file as _file_mod
                st = be.context_state() or {}
                _file_mod.set_read_context(
                    model=getattr(be, "model", "") or "",
                    base_url=getattr(be, "base_url", "") or "",
                    used_tokens=int(st.get("used", 0) or 0),
                )
            except Exception as e:
                log.debug("batch read_context inject failed: %s", e)

        # Build parallel payload
        batch = []
        for item in resolved:
            fn = tool_map.get(item["name"])
            if fn is None:
                def _unknown(*args, _n=item["name"], **kwargs):
                    return f"Error: unknown tool: {_n}"
                batch.append({
                    "name": item["name"], "args": item["args"],
                    "fn": _unknown})
            else:
                batch.append({"name": item["name"], "args": item["args"],
                              "fn": fn})
        try:
            raw = be.run_tools_parallel(batch)
        except Exception as e:
            log.debug("parallel batch failed, serial fallback: %s", e)
            return _serial_fallback()

        # Post-process: guard each result, cache, track
        out = []
        for item, r in zip(resolved, raw):
            text = r.get("result", "") if isinstance(r, dict) else ""
            err = r.get("error") if isinstance(r, dict) else None
            if err:
                text = self._guard_result(item["name"], f"Error: {err}")
                self._turn_errors += 1
                # Batch 4.2c: make failure explicit so the model cannot
                # interpret a failure as content.
                text = "[FAILED] " + text
            else:
                text = self._guard_result(item["name"], text)
                if not text.startswith("Error:"):
                    try:
                        self._cache_put(item["name"], item["args"], text)
                    except Exception:
                        pass
            if item["name"] not in self._turn_tools:
                self._turn_tools.append(item["name"])
            out.append({"name": item["name"], "args": item["args"],
                        "result": text, "error": err})
        return out

    def _trim_messages(self, max_entries: int = 40) -> None:
        """F2: Cap message history to prevent unbounded memory growth.

        Keeps the most recent max_entries. Preserves user/assistant and
        tool_call/tool_result pairing — never starts with a dangling
        tool_result.
        """
        msgs = self._messages
        if len(msgs) <= max_entries:
            return
        excess = len(msgs) - max_entries
        # Don't start with a dangling tool_result
        while excess < len(msgs) and msgs[excess].get("role") == "tool_result":
            excess += 1
        # Safety: if we walked past everything, just keep the last max_entries
        if excess >= len(msgs):
            self._messages = msgs[-max_entries:]
        else:
            self._messages = msgs[excess:]

    def _summarize_messages(self, msgs: list) -> str:
        """Summarize a transcript slice with the on-device engine.

        Feeds ``_summary_prompt(msgs)`` to ``self.llm.ask`` — a one-shot
        on the local model — and returns the stripped summary text (""
        when the engine produced nothing); engine errors propagate.

        The summarizer matches the backend serving the turn (Release
        18, KoHtet's ruling): this local core serves local turns and
        manual compress_session; a cloud turn's history is summarized
        by _summarize_messages_cloud instead — same prompt, stronger
        model, and Cloud Only never has to wake an engine it may not
        even have loaded. Neither call is metered as a user-answer
        call: like the reviewer (see _reviewer_reply), the summarizer
        is not the model that produced the answer.
        """
        return (self.llm.ask(_summary_prompt(msgs)) or "").strip()

    def _summarize_messages_cloud(self, be, msgs: list) -> str:
        """Summarize a transcript slice through the cloud backend.

        The cloud twin of _summarize_messages: ONE tools-free backend
        step carrying the same bullet-summary prompt as a lone user
        message — exactly the _reviewer_reply shape, so the fresh
        one-message list neither reads the whole conversation nor
        writes anything into the transcript. Routing is the turn's
        own: be.step walks route_providers(), so in locked mode the
        summary is attempted on the pinned provider alone, and its
        failure propagates (→ _cap_history falls back to the plain
        trim) instead of quietly switching to another provider for a
        call the user never sees. Like the reviewer, the step is
        deliberately NOT passed through record_usage/meter_usage.
        Backend errors propagate; an empty reply returns "".
        """
        reply = be.step(
            system=_SUMMARIZE_SYSTEM,
            messages=[{"role": "user", "text": _summary_prompt(msgs)}],
            tools=[],
            max_tokens=config.AGENT_VERIFY_MAX_TOKENS,
        )
        return (getattr(reply, "text", "") or "").strip()

    def _cloud_summarizer(self, be):
        """The summarize callable for a cloud turn's _cap_history.

        None when there is no backend to summarize with — _cap_history
        then uses the local core, exactly the pre-revision behaviour.
        """
        if be is None:
            return None
        return lambda msgs: self._summarize_messages_cloud(be, msgs)

    def _auto_compact(self, max_entries: int = 40, summarize=None) -> bool:
        """Summarize the head a trim would drop; keep one summary pair.

        The kept tail is the newest ``max_entries - 2`` entries (the pair
        takes the other two slots), split at the same pairing boundary
        _trim_messages uses, so no kept tool_result is left dangling.
        The dropped head is rendered whole into the summarizer — a
        previous summary pair at its front is just more rendered lines,
        which is what makes summaries roll forward into the next one
        instead of stacking. Returns True when the head was replaced;
        False (transcript untouched) when there is nothing sensible to
        compact. Summarizer errors propagate to _cap_history, which
        falls back to the plain trim.

        ``summarize`` is the callable that writes the summary; it
        defaults to the on-device core (_summarize_messages). Cloud
        call sites pass _cloud_summarizer(be) so the summary is written
        by the backend that served the turn (Release 18 revision).
        """
        if summarize is None:
            summarize = self._summarize_messages
        msgs = self._messages
        keep = max_entries - 2
        if keep < 1 or len(msgs) <= max_entries:
            return False
        cut = len(msgs) - keep
        # Same rule as _trim_messages: never start with a dangling
        # tool_result — its tool_call belongs to the summarized head.
        while cut < len(msgs) and msgs[cut].get("role") == "tool_result":
            cut += 1
        if cut < 1 or cut >= len(msgs):
            return False
        summary = summarize(msgs[:cut])
        if not summary:
            return False
        if not _summary_keeps_facts(msgs[:cut], summary):
            # Release 29: a summary that drops the checkable facts is
            # a failed compaction, not a successful one — return False
            # so _cap_history keeps the newest raw history and marks
            # the loss visibly instead of entombing it in the summary.
            log.warning("auto-compact: summary failed the fact check "
                        "(salient tokens of %d dropped entries mostly "
                        "absent) — falling back to raw trim", cut)
            return False
        self._messages = (
            [{"role": "user", "text": _SUMMARY_MARKER},
             {"role": "assistant", "text": summary}]
            + msgs[cut:]
        )
        log.info("auto-compact: %d entries summarized, %d kept",
                 cut, len(msgs) - cut)
        return True

    def _cap_history(self, max_entries: int = 40, summarize=None) -> None:
        """Turn-end history cap: compact when over, trim as the net.

        Auto-compaction runs at most once per call and only when the
        transcript is over the cap; ANY failure in it (engine error,
        empty summary, no engine) falls back to the plain trim, so a
        compaction problem can never lose or block the user's turn.
        _trim_messages always runs last with its unchanged behaviour —
        after a successful compaction it is a no-op by construction.
        ``summarize`` is threaded to _auto_compact unchanged.

        Release 29: when compaction was on and attempted but did NOT
        produce a summary pair, the fallback trim no longer drops the
        head silently — it keeps two slots for a visible marker pair
        (_TRIM_MARKER + _TRIM_NOTE) saying the early conversation was
        cut without a summary. With compaction disabled by the user
        the trim is the designed behaviour and stays unmarked, exactly
        as before.
        """
        compacted = False
        if (len(self._messages) > max_entries
                and config.AUTO_COMPACT_ENABLED):
            try:
                compacted = self._auto_compact(max_entries, summarize)
            except Exception as e:
                log.warning("auto-compact failed, plain trim instead "
                            "(%s: %s)", type(e).__name__, e)
            if not compacted:
                before = len(self._messages)
                self._trim_messages(max(max_entries - 2, 1))
                if len(self._messages) < before:
                    self._messages = (
                        [{"role": "user", "text": _TRIM_MARKER},
                         {"role": "assistant", "text": _TRIM_NOTE}]
                        + self._messages)
                    return
        self._trim_messages(max_entries)

    # ── hybrid backend (cloud when WiFi + keys, local otherwise) ──

    def _backend_obj(self):
        try:
            from core.agent_backend import AgentBackend
        except Exception as e:
            log.warning("agent_backend unavailable: %s", e)
            return None
        if self._backend is None:
            try:
                self._backend = AgentBackend()
            except Exception as e:
                log.error("could not build agent backend: %s: %s",
                          type(e).__name__, e)
        if self._backend is not None:
            # Settings writes agent_config.json from another request while this
            # object lives for the whole process — re-read it every turn, or a
            # switch to Cloud Only stays invisible until the server restarts.
            try:
                self._backend.refresh()
            except Exception as e:
                log.warning("backend config refresh failed: %s", e)
        return self._backend

    def _mode(self) -> str:
        """backend_mode, always read fresh (auto | local | cloud)."""
        be = self._backend_obj()
        if be is None:
            return self._key_mode()
        try:
            return be.mode
        except Exception as e:
            log.warning("backend mode unreadable: %s", e)
            return self._key_mode()

    def _key_mode(self) -> str:
        """Mode straight from disk — used when the backend cannot be built."""
        try:
            from core.key_manager import KeyManager
            return KeyManager().mode()
        except Exception as e:
            log.warning("could not read backend_mode: %s", e)
            return "auto"

    @staticmethod
    def _local_label() -> str:
        """Friendly name of the on-device model, for the status-bar record."""
        try:
            from core.agent_backend import friendly_local_name
            return friendly_local_name() or "Local model"
        except Exception:
            return "Local model"

    # Why a Cloud Only turn could not run — Burmese, user-facing.
    CLOUD_ONLY_REASONS = {
        "offline": "Network မရှိလို့ Cloud ကို ချိတ်လို့မရသေးဘူး",
        "no_usable_key": "သုံးလို့ရတဲ့ API Key မရှိသေးဘူး",
        "cost_not_acknowledged":
            "Settings မှာ Model ကုန်ကျစရိတ်ကို အရင် အတည်ပြုပေးပါ",
        "local_only": "Local Only mode ဖြစ်နေလို့",
        "backend_unavailable": "Cloud backend ကို စတင်လို့မရတော့ဘူး",
        "exhausted": "Cloud provider အားလုံး မအောင်မြင်တော့ဘူး",
        # Release 24: every key merely parked — temporary, with a wait.
        "cooling": "Provider key တွေ အကုန် cooldown ထဲမှာ ရှိတယ်",
        # Release 29: Cloud Only and no routed model can read images.
        "vision_unavailable":
            "ရွေးထားတဲ့ cloud model တွေက ပုံကို မဖတ်နိုင်ဘူး — "
            "Settings → Models မှာ 📷 တံဆိပ် ပါတဲ့ model ပြောင်းရွေးပါ",
    }

    def _try_cloud(self, user_input: str, system: str = "", image=None):
        """One cloud attempt for this turn; None → answer locally.

        Every failure path rolls the transcript back to exactly where it was,
        so the local branch below never sees a half-written turn (and a cloud
        success never loads a local conversation).

        Cloud Only is the exception: it never falls back to the 4B model, it
        answers with the reason instead. Local Only never leaves this function
        with anything but None.

        ``system`` is passed down by handle() so a cloud attempt and the local
        fallback that may follow it are given the *same* prompt; the SSE route
        leaves it empty and the tier is resolved here instead.
        """
        mode = self._mode()          # fresh from disk, every turn
        if mode == "local":
            return None              # Local Only never leaves this function
        if self._is_http() and mode != "cloud":
            return None              # HTTP transport stays untouched in auto
        be = self._backend_obj()
        if be is None:
            # Cloud Only pinned by the user: say so rather than answering from
            # the local model the user just switched off.
            if mode == "cloud":
                return self._cloud_only_reply(user_input, "backend_unavailable")
            return None
        try:
            want = be.should_use_cloud()
            reason = "" if want else (be.why_not_cloud() or "no_usable_key")
        except Exception as e:
            log.warning("cloud check failed (%s)", e)
            if mode == "cloud":
                return self._cloud_only_reply(user_input, "backend_unavailable")
            return None
        if not want and reason == "no_usable_key":
            # Locked mode (Release 9): the global gate only knows that
            # SOME provider is unusable. With a pin set, step() must get
            # its chance to name the pinned provider's exact problem
            # instead of the turn silently dropping to the local model.
            try:
                if be.locked_pin():
                    want = True
            except Exception:
                pass
        if not want:
            if mode == "cloud":
                if reason == "no_usable_key":
                    # Release 24: "no usable key" can mean every key is
                    # merely parked — a temporary state the reply must
                    # name as such, with the wait, not as a missing key.
                    classified = self._classify_failure(be, None)
                    if classified["park_only"]:
                        return self._cloud_only_reply(
                            user_input, "cooling",
                            causes=classified["lines"], be=be,
                            classified=classified)
                return self._cloud_only_reply(user_input, reason)
            return None
        before = len(self._messages)
        reason = "exhausted"
        locked_reason = None
        locked_provider = ""
        locked_detail = ""
        failure_exc = None
        turn_only = None
        if image is not None:
            # Release 29: an image turn may only run where the model
            # can actually read the image. Locked mode: the pin alone
            # decides — incapable means the guidance reply below, never
            # a silent text-only answer that drops the photo, and never
            # a quiet switch away from the chosen pin. Auto / Cloud
            # Only: the turn's route narrows to the vision-capable
            # providers; with none capable, Cloud Only says so and
            # Auto declines (None) to the local engine, whose own
            # vision gate decides there — the same shape as a text
            # turn the cloud cannot serve.
            try:
                pin = be.locked_pin()
            except Exception:
                pin = None
            if pin is not None:
                try:
                    capable = be.model_image_input(pin)
                except Exception:
                    capable = False
                if not capable:
                    return self._locked_vision_reply(user_input, pin)
            else:
                try:
                    capable = be.cloud_vision_providers()
                except Exception:
                    capable = []
                if capable:
                    turn_only = capable
                elif mode == "cloud":
                    return self._cloud_only_reply(
                        user_input, "vision_unavailable")
                else:
                    return None
        try:
            return self._handle_cloud(user_input, be, system=system,
                                      image=image, turn_only=turn_only)
        except Exception as e:
            from core.agent_backend import BackendExhausted
            from core.key_manager import redact
            failure_exc = e
            if isinstance(e, BackendExhausted):
                log.warning("cloud backends exhausted (%s)", e)
                msg = str(e)
                reason = ("offline" if ("network" in msg.lower()
                                        or "no network" in msg.lower())
                          else "exhausted")
                if getattr(e, "locked_reason", None):
                    locked_reason = e.locked_reason
                    locked_provider = getattr(e, "locked_provider", "") or ""
                    try:
                        locked_detail = redact(
                            getattr(e, "locked_detail", "") or "",
                            be.keys.all_keys())[:220]
                    except Exception:
                        locked_detail = ""
            else:
                log.error("cloud step failed (%s: %s)", type(e).__name__, e)
        # Release 24: a turn that already executed tools must not be
        # rolled back into nothing — its tool_call/tool_result entries
        # are the only record that side effects happened (the live
        # v1.21.0 failure: the file was written, the reply said the
        # turn could not be answered, the transcript showed nothing).
        # Preserve the turn, close it with the failure reply (completed-
        # work summary + classified cause) and save it. Falling back to
        # the local model here would re-run the whole turn and repeat
        # the side effects, so a tool-using turn stops with the reply
        # in auto mode too. Turns with no tool work keep the rollback
        # (and auto's silent local fallback) exactly as before.
        turn_entries = self._messages[before:]
        if any(m.get("role") == "tool_result" for m in turn_entries):
            try:
                return self._preserve_failed_turn(
                    be, mode, turn_entries,
                    locked=(locked_provider, locked_reason, locked_detail)
                    if locked_reason else None,
                    exc=failure_exc)
            except Exception as e:
                log.error("failed-turn preservation failed (%s: %s) — "
                          "rolling back as before", type(e).__name__, e)
        del self._messages[before:]
        if locked_reason:
            return self._locked_reply(user_input, locked_provider,
                                      locked_reason, locked_detail, be=be)
        if mode == "cloud":
            # Release 24: the reply is composed from the classified
            # failure records — never from the exception's raw text
            # (provider JSON, masked key fragments, mid-word cuts).
            classified = self._classify_failure(be, failure_exc)
            if classified["park_only"]:
                return self._cloud_only_reply(
                    user_input, "cooling", causes=classified["lines"],
                    be=be, classified=classified)
            return self._cloud_only_reply(
                user_input, reason, causes=classified["lines"],
                be=be, classified=classified)
        return None

    # ── failure replies (Release 24) ───────────────────────────────
    #
    # A failed turn used to show the provider's raw error text —
    # redacted and cut at 220 chars, but still JSON fragments, masked
    # key material and sentences severed mid-word — and a turn that
    # died after running tools lost the tools from the transcript
    # entirely. The builders below compose every failure reply from
    # structured state only: the per-provider records step() left on
    # the exception, the key states the backend reads back, and the
    # turn's own tool entries. Deterministic, no provider call.

    # failure_reason() vocabulary → Burmese cause + one action hint.
    FAILURE_REASONS = {
        "key_invalid": "API key က အလုပ်မလုပ်တော့ဘူး",
        "no_balance": "အကောင့်မှာ ဘေ/credit မရှိတော့ဘူး",
        "rate_limited": "Limit ပြည့်သွားပြီ (429)",
        "quota_exhausted": "ဒီနေ့ အတွက် quota ကုန်သွားပြီ (429)",
        "cooling": "Key က cooldown ထဲမှာ ရှိတယ်",
        "model_unavailable": "အဲဒီ model ကို အခု မရနိုင်ဘူး",
        "no_keys": "Key မရှိဘူး",
        "off": "Provider ကို ပိတ်ထားတယ်",
        "network": "Network မရှိလို့ ချိတ်မရဘူး",
        "server_error": "Provider ဘက်က ပြဿနာ ရှိနေတယ်",
    }
    FAILURE_HINTS = {
        "key_invalid": "Settings မှာ key ကို ပြန်စစ်ပါ",
        "no_balance": "credit ဖြည့်ပါ၊ ဒါမှမဟုတ် free model ပြောင်းသုံးပါ",
        "rate_limited": "ခဏ စောင့်ပြီး ပြန်စမ်းပါ",
        "quota_exhausted": "မနက်ဖြန် ပြန်စမ်းပါ၊ ဒါမှမဟုတ် တခြား provider/model ပြောင်းသုံးပါ",
        "cooling": "ခဏ စောင့်ပြီး ပြန်စမ်းပါ",
        "model_unavailable": "Settings မှာ model ပြောင်းရွေးပါ",
        "no_keys": "Settings မှာ key ထည့်ပါ",
        "off": "Settings မှာ provider ပြန်ဖွင့်ပါ",
        "network": "network ချိတ်ဆက်မှု ပြန်စစ်ပါ",
        "server_error": "ခဏနေ ပြန်စမ်းပါ",
    }

    def _classify_failure(self, be, exc=None) -> dict:
        """Classified Burmese causes for a cloud failure.

        Merges the exception's per-provider records (providers step()
        actually tried) with the backend's key-state overview
        (providers it never reached — e.g. a key parked by an earlier
        step of the same turn, whose remaining wait the overview can
        still name). Returns {"lines", "park_only", "wait"}:
        park_only when every blocker is a cooling/rate-limit park with
        a known remaining time — a temporary state, not exhaustion.
        Never raises: classification trouble degrades to no lines and
        the caller's headline still stands.
        """
        merged = {}
        order = []
        try:
            for f in (getattr(exc, "failures", None) or []):
                p = f.get("provider") or ""
                if not p or p in merged:
                    continue
                wait = None
                ra = f.get("retry_after")
                if ra:
                    try:
                        wait = max(1, int(round(float(ra))))
                    except (TypeError, ValueError):
                        wait = None
                merged[p] = {"reason": f.get("reason") or "server_error",
                             "wait": wait}
                order.append(p)
            overview = []
            if be is not None:
                try:
                    overview = be.failure_overview() or []
                except Exception:
                    overview = []
            for o in overview:
                p = o.get("provider") or ""
                if not p:
                    continue
                if p not in merged:
                    merged[p] = {"reason": o.get("reason") or "server_error",
                                 "wait": o.get("wait")}
                    order.append(p)
                elif merged[p]["wait"] is None and o.get("wait"):
                    merged[p]["wait"] = o["wait"]
        except Exception as e:
            log.warning("failure classification failed (%s: %s)",
                        type(e).__name__, e)
            return {"lines": [], "park_only": False, "wait": None,
                    "reasons": set()}
        from core.key_manager import PROVIDER_LABELS
        lines = []
        for p in order:
            info = merged[p]
            cause = self.FAILURE_REASONS.get(
                info["reason"], self.FAILURE_REASONS["server_error"])
            line = "• %s: %s" % (PROVIDER_LABELS.get(p, p), cause)
            hint = self.FAILURE_HINTS.get(info["reason"], "")
            if hint:
                line += " — " + hint
                if info["reason"] in ("cooling", "rate_limited") \
                        and info["wait"]:
                    line += " " + _wait_phrase(info["wait"])
            lines.append(line)
        known = [merged[p]["wait"] for p in order if merged[p]["wait"]]
        park_only = bool(order) and all(
            merged[p]["reason"] in ("cooling", "rate_limited")
            and merged[p]["wait"] for p in order)
        return {"lines": lines, "park_only": park_only,
                "wait": min(known) if known else None,
                # Release 29: the raw reason vocabulary, so reply
                # builders can react to a class (quota_exhausted)
                # without re-parsing the rendered Burmese lines.
                "reasons": {merged[p]["reason"] for p in order}}

    # Args that name what a tool call acted on, in preference order.
    _TARGET_KEYS = ("path", "file", "filename", "filepath", "directory",
                    "dir", "query", "command", "cmd", "url", "name")

    # Release 28: the read-only tools whose result a failure summary
    # may quote back to the user. Read-only is the whole point — a
    # lookup turn's answer often IS the tool result (round 3, T6a2:
    # memory_read had already found the fact when the composing step
    # died, and the summary named the tool but never the answer).
    # Write/shell results are never quoted: their text can carry
    # command output the summary has no business repeating.
    _QUOTABLE_RESULT_TOOLS = frozenset(
        {"memory_read", "read_file", "list_files"})
    _QUOTE_CAP = 600

    @classmethod
    def _quote_result(cls, name: str, text: str) -> str:
        """The quotable form of one guarded read-only result.

        The transcript text already passed _guard_result on the way
        in (banner + redaction); this strips the model-facing wrapper
        (cache tag, banner lines) for a human reader, redacts once
        more for defence in depth, and caps the length.
        """
        body = str(text or "")
        if body.startswith("[cached] "):
            body = body[len("[cached] "):]
        lines = [ln for ln in body.splitlines()
                 if ln.strip() not in (safety.INJECTION_BANNER_OPEN,
                                       safety.INJECTION_BANNER_CLOSE)]
        body = safety.redact_secrets("\n".join(lines).strip())
        if len(body) > cls._QUOTE_CAP:
            body = body[:cls._QUOTE_CAP].rstrip() + "…"
        if not body:
            return ""
        return ("နောက်ဆုံး ဖတ်လို့ ရခဲ့တဲ့ ရလဒ် (%s):\n%s"
                % (name, body))

    @staticmethod
    def _shorten_target(target: str) -> str:
        """Fit a tool target into the summary's 48-char slot.

        Plain values (commands, queries) read best from the front, so
        they keep the existing head-truncation. A path reads the other
        way — the filename at the END is what identifies it — so a
        long path keeps its tail behind an ASCII "..." marker.
        Head-truncating a path loses the filename entirely wherever
        directory prefixes run long (on Termux the tmp prefix alone
        is longer than the whole slot), which is exactly the R24
        phone-only failure this replaces."""
        if len(target) <= 48:
            return target
        if "/" in target:
            return "..." + target[-45:]
        return target[:48]

    def _work_summary(self, entries) -> str:
        """Completed-work summary for a turn that died mid-work.

        Built from the turn's own tool_call/tool_result pairs, in
        execution order — deterministic, no provider call. A result
        counts as failed when its text carries the tools' own "Error:"
        prefix (the shape _execute_tool guarantees). "" when no tool
        completed a result this turn.
        """
        parts = []
        quote = ""
        pending = None
        for m in entries:
            role = m.get("role")
            if role == "tool_call":
                pending = m
            elif role == "tool_result":
                name = m.get("tool") or (pending or {}).get("tool") or "tool"
                args = (pending or {}).get("args") or {}
                pending = None
                target = ""
                if isinstance(args, dict):
                    for k in self._TARGET_KEYS:
                        v = args.get(k)
                        if isinstance(v, str) and v.strip():
                            target = self._shorten_target(
                                " ".join(v.split()))
                            break
                ok = not str(m.get("text") or "").startswith("Error")
                label = "%s (%s)" % (name, target) if target else name
                parts.append("%s — %s" % (label, "အောင်မြင်"
                                           if ok else "မအောင်မြင်"))
                if ok and name in self._QUOTABLE_RESULT_TOOLS:
                    q = self._quote_result(name, m.get("text") or "")
                    if q:
                        quote = q   # the LATEST read-only result wins
        if not parts:
            return ""
        out = ("ဒီနေရာအထိ ပြီးပြီးသားပါ: " + "၊ ".join(parts) + "။\n"
               "ဒီအလုပ်တွေကို မှတ်တမ်းမှာ သိမ်းထားပါတယ် — "
               "အစက ပြန်လုပ်စရာ မလိုပါဘူး။")
        if quote:
            out += "\n" + quote
        return out

    def _failure_turn_text(self, mode: str, classified: dict,
                           summary: str, hint: str = "") -> str:
        """The failure reply closing a preserved cloud turn."""
        lines = []
        if mode == "cloud":
            lines.append("☁️ Cloud Only mode ဖြစ်နေလို့ ဒီအလှည့်ကို "
                         "ဆက်မဖြေနိုင်တော့ဘူး။")
        else:
            lines.append("☁️ Cloud provider တွေ မဖြေနိုင်လို့ ဒီအလှည့်ကို "
                         "ဒီတင် ရပ်လိုက်ပါတယ်။")
        lines.extend(classified["lines"])
        if summary:
            lines.append(summary)
        if mode == "cloud":
            lines.append("⚙️ Settings → Backend Mode ကနေ Auto ပြောင်းရင် "
                         "Local Gemma 4B ကို အလိုအလျောက် fallback ဖြစ်မယ်။")
        if hint:
            lines.append(hint)
        lines.append("ဆက်လုပ်စေချင်ရင် \"ဆက်လုပ်ပါ\" လို့ ပြန်ပြောပါ — "
                     "ကျန်နေတဲ့ အလုပ်ကို ဆက်လုပ်ပေးပါမယ်။")
        return "\n".join(lines)

    def _local_failure_text(self, entries) -> str:
        """The failure reply closing a preserved local turn."""
        lines = ["⚠️ Local model က ကြားဖြတ် ရပ်သွားလို့ ဒီအလှည့်ကို "
                 "ဆက်မဖြေနိုင်တော့ဘူး။"]
        summary = self._work_summary(entries)
        if summary:
            lines.append(summary)
        lines.append("ဆက်လုပ်စေချင်ရင် \"ဆက်လုပ်ပါ\" လို့ ပြန်ပြောပါ — "
                     "ကျန်နေတဲ့ အလုပ်ကို ဆက်လုပ်ပေးပါမယ်။")
        return "\n".join(lines)

    def _preserve_failed_turn(self, be, mode: str, turn_entries,
                              locked=None, exc=None) -> str:
        """Close a tool-using turn that died on the cloud path.

        The transcript already holds the user message and the turn's
        tool_call/tool_result entries (the caller does NOT roll back);
        append the failure reply as the assistant entry, trim, save.
        _save_turn clears the mid-turn checkpoint — the Release 24
        invariant: completed tool work is recoverable from durable
        state, here the transcript itself, and the reply says so, so
        the checkpoint has nothing left to guard. record_usage("")
        mirrors the other generated replies: the text the user reads
        was composed here, not produced by a model (the real steps are
        already in the Release 17 ledger via meter_usage).
        """
        summary = self._work_summary(turn_entries)
        # NB for the hint lookups below: this method's ``be`` is the
        # BackendExhausted exception (the _classify_failure input), so
        # _second_key_hint is called bare — it resolves the mode's real
        # backend itself.
        if locked is not None:
            provider, reason, detail = locked
            text = self._locked_text(provider, reason, detail,
                                     hint=self._second_key_hint())
            if summary:
                text += "\n" + summary
        else:
            classified = self._classify_failure(be, exc)
            hint = (self._second_key_hint()
                    if "quota_exhausted" in classified["reasons"] else "")
            text = self._failure_turn_text(
                mode, classified, summary, hint=hint)
        self._messages.append({"role": "assistant", "text": text})
        self._trim_messages()
        self._save_turn()
        record_usage("")
        return text

    @staticmethod
    def _durable(messages) -> list:
        """The transcript as durable stores may keep it (Release 29).

        Identical to the live transcript except that a user entry's
        live-turn image payload is stripped: the image serves the one
        turn that carried it, and a stored session must never need the
        staged file (or hold megabytes of base64) to replay. Entries
        without an image pass through as the same objects.
        """
        out = []
        for m in messages or []:
            if isinstance(m, dict) and "image" in m:
                m = {k: v for k, v in m.items() if k != "image"}
            out.append(m)
        return out

    def _save_turn(self) -> None:
        if not self._store:
            return
        try:
            self._store.save_mode("agent", self._durable(self._messages))
        except Exception as e:
            log.error("could not save agent turn (session %s): %s: %s",
                      getattr(self, "session_id", None), type(e).__name__, e)
            return
        # Release 19: the turn is durably saved, so its mid-turn
        # checkpoint has done its job — a completed turn must never
        # be "resumed". Only on success: when the save failed, the
        # checkpoint is the only durable copy of the turn left.
        self._clear_checkpoint()

    # ── mid-turn checkpoints (Release 19) ──────────────────────────
    #
    # _save_turn persists at turn END only. While a turn runs, every
    # tool loop snapshots the working transcript to
    # core/checkpoint_store after each tool_result lands in it; if the
    # process dies, the next load of the session restores the snapshot
    # (with an interruption note) instead of losing the turn — and its
    # already-executed side effects — without a trace. Every helper
    # here is guarded end to end: the store functions never raise,
    # and these wrappers add a second net, because a checkpoint
    # problem must never affect the turn it only describes.

    def _save_checkpoint(self) -> None:
        """Snapshot the in-flight turn (loops call this per result)."""
        sid = getattr(self, "session_id", None)
        if not sid or getattr(self, "_store", None) is None:
            return
        # The in-flight request and the steps spent on it, read back
        # off the transcript itself — every loop (including the v1 SSE
        # route's hand-driven one, which keeps no counters of ours)
        # produces the same shape here.
        user_text = ""
        steps = 0
        for m in reversed(self._messages):
            if m.get("role") == "user":
                user_text = m.get("text", "") or ""
                break
            if m.get("role") == "tool_call":
                steps += 1
        try:
            from core import checkpoint_store
            checkpoint_store.save_checkpoint(
                sid, user_text, self._durable(self._messages), steps)
        except Exception as e:
            log.warning("checkpoint save failed (session %s): %s: %s",
                        sid, type(e).__name__, e)

    def _clear_checkpoint(self, sid: str | None = None) -> None:
        """Drop a session's checkpoint (its turn completed)."""
        try:
            from core import checkpoint_store
            checkpoint_store.clear_checkpoint(
                sid or getattr(self, "session_id", None) or "")
        except Exception as e:
            log.warning("checkpoint clear failed: %s: %s",
                        type(e).__name__, e)

    @staticmethod
    def _interrupted_note(steps: int) -> str:
        """Burmese transcript note closing a resumed turn (Release 19).

        Appended as an ordinary assistant entry, so it is visible to
        the user reading the history AND to the model answering the
        next turn: the turn was cut off mid-work, what it had done is
        still here, and continuing is one word away — the same
        closing line the Release 15 budget notice uses.
        """
        lines = ["⚠️ အရင် အလှည့်က ကြားဖြတ် ရပ်သွားခဲ့ပါတယ်"
                 " (server ပိတ်သွားတာ ဖြစ်နိုင်ပါတယ်)။"]
        if steps:
            lines[0] = ("⚠️ အရင် အလှည့်က အဆင့် (step) %d ဆင့် "
                        "လုပ်ပြီးချိန်မှာ ကြားဖြတ် ရပ်သွားခဲ့ပါတယ်"
                        " (server ပိတ်သွားတာ ဖြစ်နိုင်ပါတယ်)။" % steps)
        lines.append("လုပ်ပြီးသမျှ အလုပ်တွေ — tool ရလဒ်တွေ အပါအဝင် — "
                     "ဒီ မှတ်တမ်းမှာ အဲဒီ အတိုင်း ကျန်နေပါတယ်၊ "
                     "အစက ပြန်လုပ်စရာ မလိုပါဘူး။")
        lines.append("ဆက်လုပ်စေချင်ရင် \"ဆက်လုပ်ပါ\" လို့ ပြန်ပြောပါ — "
                     "ကျန်နေတဲ့ အလုပ်ကို ဆက်လုပ်ပေးပါမယ်။")
        return "\n".join(lines)

    def _restore_checkpoint(self) -> bool:
        """Resume an interrupted turn left by a previous process.

        Runs at session load (``__init__`` / ``reload_session``),
        after ``self._messages`` holds the last completed turn. A
        waiting checkpoint's transcript replaces it, closed by the
        interruption note, and is saved straight back through
        _save_turn so the restored state is itself durable and a
        second load finds nothing to restore (take consumes the
        checkpoint). Malformed or foreign checkpoints are ignored by
        the store. Returns True when a turn was resumed.
        """
        sid = getattr(self, "session_id", None)
        if not sid or getattr(self, "_store", None) is None:
            return False
        try:
            from core import checkpoint_store
            entry = checkpoint_store.take_checkpoint(sid)
        except Exception as e:
            log.warning("checkpoint restore failed (session %s): %s: %s",
                        sid, type(e).__name__, e)
            return False
        if not entry:
            return False
        msgs = [dict(m) for m in entry["messages"]]
        steps = entry.get("steps") or 0
        msgs.append({"role": "assistant",
                     "text": self._interrupted_note(steps)})
        self._messages = msgs
        self.turns = len(msgs) // 2
        self._save_turn()
        log.info("resumed interrupted turn for session %s "
                 "(%d steps had completed)", sid, steps)
        return True

    # Why a locked (pinned) provider could not answer — Burmese,
    # user-facing. The codes are core.agent_backend.failure_reason's
    # vocabulary plus the pin_block_reason set (off / no_keys / cooling).
    LOCKED_REASONS = {
        "key_invalid": "API key က အလုပ်မလုပ်တော့ဘူး (key invalid)",
        "no_balance": "အကောင့်မှာ ဘေ/credit မရှိတော့ဘူး",
        "rate_limited": "Limit ပြည့်သွားပြီ (429) — ခဏစောင့်ပြီး ပြန်စမ်းပါ",
        "quota_exhausted": "ဒီနေ့ အတွက် quota ကုန်သွားပြီ (429) — မနက်ဖြန် ပြန်စမ်းပါ၊ ဒါမှမဟုတ် တခြား provider/model ပြောင်းသုံးပါ",
        "model_unavailable": "အဲဒီ model ကို အခု မရနိုင်ဘူး",
        "cooling": "Key တွေ အကုန် cooldown ထဲမှာ ရှိတယ် — ခဏစောင့်ပါ",
        "no_keys": "အဲဒီ provider မှာ key မရှိဘူး",
        "off": "အဲဒီ provider ကို ပိတ်ထားတယ်",
        "network": "Network မရှိလို့ ချိတ်မရဘူး",
        "server_error": "Provider ဘက်က ပြဿနာ ရှိနေတယ်",
    }

    # Release 29 (round-5 finding F3): when the user's whole cloud
    # life is ONE provider, a locked stop or a dead daily quota leaves
    # them with nothing — and the old replies only said "switch
    # provider", never that a second free key is what makes switching
    # possible on a day like that. The line is deterministic text (no
    # provider call), appended only when exactly one provider is
    # usable; with two or more it would be noise. It extends the
    # Release 28 Settings steering (item I) into the failure reply
    # itself, where the user actually meets the dead end.
    SECOND_KEY_HINT = (
        "💡 အခမဲ့ key နောက်တစ်ခု (ဥပမာ Groq) ထည့်ထားရင် ဒီလို "
        "အခြေအနေမျိုးမှာ အဲ့ဒီ key နဲ့ ဆက် သုံးနိုင်ပါတယ် — "
        "Settings မှာ ထည့်နိုင်ပါတယ်။")

    def _second_key_hint(self, be=None) -> str:
        """The F3 steering line, or "" when it does not apply.

        Applies only when the user has exactly ONE live provider
        (backend.live_providers: a non-dead key exists; the failure
        being reported has usually just parked that key, so the
        cooldown-aware usable() would undercount exactly here). Any
        doubt — no backend, uncountable state — answers "": a
        steering line printed on a guess would be worse than none.
        """
        if be is None:
            be = self._backend_obj()
        if be is None:
            return ""
        try:
            live = be.live_providers()
        except Exception:
            return ""
        return self.SECOND_KEY_HINT if len(live) == 1 else ""

    def _locked_text(self, provider: str, reason: str,
                     detail: str = "", hint: str = "") -> str:
        """The locked-mode stop text (Release 9 wording, unchanged —
        Release 29 only appends the optional F3 hint line)."""
        from core.key_manager import PROVIDER_LABELS
        label = PROVIDER_LABELS.get(provider, provider or "provider")
        why = self.LOCKED_REASONS.get(reason, reason or "server_error")
        if detail:
            why = why + " — " + detail
        text = ("🔒 Lock ချထားတဲ့ " + label + " က မဖြေနိုင်ဘူး: " + why +
                "\nAuto switch ပိတ်ထားလို့ တခြား model ကို အလိုအလျောက် "
                "မပြောင်းဘူး။ ပြောင်းသုံးချင်ရင် model chip ကို နှိပ်ပြီး "
                "ရွေးပါ၊ ဒါမှမဟုတ် Settings → Backend မှာ Auto switch "
                "ပြန်ဖွင့်ပါ။")
        if hint:
            text += "\n" + hint
        return text

    def _locked_reply(self, user_input: str, provider: str,
                      reason: str, detail: str = "", be=None) -> str:
        """Locked mode (Release 9): the pinned provider failed and auto
        switch is off, so the turn stops here with the reason — the user
        pinned that model on purpose and asked to decide themselves,
        not to be switched quietly."""
        text = self._locked_text(provider, reason, detail,
                                 hint=self._second_key_hint(be))
        self._messages.append({"role": "user", "text": user_input})
        self._messages.append({"role": "assistant", "text": text})
        self._trim_messages()
        self._save_turn()
        # Nothing ran — same bookkeeping as _cloud_only_reply.
        record_usage("")
        return text

    def _locked_vision_reply(self, user_input: str, provider: str) -> str:
        """Locked mode + an image turn the pinned model cannot read
        (Release 29): stop with the limitation and the ways out. The
        image is never sent somewhere the user did not choose, and the
        turn never degrades to a text-only answer that silently drops
        the photo. Same transcript contract as _locked_reply."""
        from core.key_manager import PROVIDER_LABELS
        label = PROVIDER_LABELS.get(provider, provider or "provider")
        text = ("🔒 Lock ချထားတဲ့ " + label + " ရဲ့ model က ပုံကို "
                "မဖတ်နိုင်လို့ ဒီအလှည့်မှာ ပုံကို မပို့လိုက်ပါ။\n"
                "ပုံ မေးခွန်း သုံးချင်ရင် — Settings → Models မှာ 📷 "
                "တံဆိပ် ပါတဲ့ model တစ်ခု ရွေးပါ၊ ဒါမှမဟုတ် Auto "
                "switch ပြန်ဖွင့်ပါ၊ ဒါမှမဟုတ် local vision model "
                "ရှိရင် Backend Mode ကို Auto/Local ပြောင်းပါ။")
        self._messages.append({"role": "user", "text": user_input})
        self._messages.append({"role": "assistant", "text": text})
        self._trim_messages()
        self._save_turn()
        record_usage("")
        return text

    def _cloud_only_reply(self, user_input: str, reason: str,
                          detail: str = "", causes=None, be=None,
                          classified=None) -> str:
        """Cloud Only mode with no usable provider: say so instead of
        quietly answering from the local 4B model the user just switched off.

        ``causes`` (Release 24): the classified per-provider lines from
        _classify_failure, printed under the headline in place of the
        old raw ``detail``. With neither, the output is exactly the
        pre-R24 shape every simple-reason caller relies on.

        Release 29: when the classified failure includes a dead daily
        quota and the user has exactly one usable provider, the F3
        second-key steering line joins the reply (before the Settings
        line, with the causes it explains).
        """
        why = self.CLOUD_ONLY_REASONS.get(reason, reason or "exhausted")
        if detail:
            why = why + " — " + detail
        text = "☁️ Cloud Only mode ဖြစ်နေလို့ မဖြေရတော့ဘူး: " + why
        if causes:
            text += "\n" + "\n".join(causes)
        if classified is not None \
                and "quota_exhausted" in classified.get("reasons", set()):
            hint = self._second_key_hint(be)
            if hint:
                text += "\n" + hint
        text += ("\n⚙️ Settings → Backend Mode ကနေ Auto ပြောင်းရင် "
                 "Local Gemma 4B ကို အလိုအလျောက် fallback ဖြစ်မယ်။")
        self._messages.append({"role": "user", "text": user_input})
        self._messages.append({"role": "assistant", "text": text})
        self._trim_messages()
        self._save_turn()
        # Nothing ran: drop the record so the status bar cannot keep showing
        # a model from an earlier turn for an answer no model produced.
        record_usage("")
        return text

    # Release 25 — the fresh-install dead end. A brand-new user has no
    # keys and (usually) no local model file: cloud declines with
    # "no_usable_key" and the local engine then raises "Model not
    # found", which used to reach the user as a raw developer error
    # that never mentioned keys at all. Callers invoke this ONLY after
    # the local engine failed to start, so "nothing can answer" is
    # already established; when the blocker is keys, close the turn
    # with guidance instead of the exception. Same transcript contract
    # as _cloud_only_reply (append the pair, save, record no usage).
    # Release 28: the guide no longer stops at the key. Round 3's
    # public-user walk found two gaps for a first-time user: "API
    # key" was never explained in plain words, and the cost-ack step
    # waiting behind the key entry arrived as a second, unexplained
    # wall. Both are now named here, up front, in beginner Burmese.
    NO_KEY_GUIDANCE = (
        "🔑 သုံးလို့ရတဲ့ API key မရှိသေးလို့ အခုတော့ မဖြေနိုင်သေးပါ။\n"
        "API key ဆိုတာ အဲ့ဒီ AI ကုမ္ပဏီက ထုတ်ပေးတဲ့ ကိုယ်ပိုင် "
        "လျှို့ဝှက် သော့ လိုပါပဲ — ကိုယ် ထုတ်ယူ၊ ကိုယ်ပဲ သုံးရပြီး "
        "သူများကို မပြော၊ မမျှဝေရပါဘူး။\n"
        "Settings မှာ provider key တစ်ခု ထည့်လိုက်ရင် သုံးလို့ ရပါပြီ — "
        "Gemini key ကို Google AI Studio (aistudio.google.com) မှာ "
        "ကိုယ့် Google အကောင့်နဲ့ အခမဲ့ ထုတ်ယူနိုင်ပါတယ်။\n"
        "Key ထည့်ပြီးရင် နောက်တစ်ဆင့် အနေနဲ့ Settings မှာ "
        "ကုန်ကျစရိတ် စာကြောင်းကို ကြည့်ပြီး Save ခလုတ် နှိပ်ပေးဖို့ "
        "လိုပါသေးတယ် — အဲဒါပြီးရင် တန်း သုံးနိုင်ပါပြီ။\n"
        "Local model ဖိုင် (.litertlm) ထည့်ထားရင်လည်း "
        "key မလိုဘဲ သုံးနိုင်ပါတယ်။")

    # Release 26: the key exists but the one-time cost acknowledgement
    # was never completed. Adding a key in Settings posts to the keys
    # route alone; the acknowledgement is the Settings Save (the UI
    # shows the price line first, then Save sends cost_ack) — a fresh
    # user who only tapped "Add" landed on the raw local-model error
    # instead (tester evaluation #2). The guidance names the exact
    # taps; the gate itself is unchanged.
    COST_ACK_GUIDANCE = (
        "💰 API key ထည့်ပြီးပေမယ့် cloud model သုံးဖို့ "
        "ကုန်ကျစရိတ် အတည်ပြုချက် မလုပ်ရသေးလို့ အခုတော့ မဖြေနိုင်သေးပါ။\n"
        "Settings ကို ဖွင့်ပြီး Model အောက်မှာ ပြထားတဲ့ "
        "ကုန်ကျစရိတ် စာကြောင်းကို ကြည့်ပြီးရင် Save ခလုတ်ကို "
        "တစ်ချက် နှိပ်ပေးပါ — အဲဒါဆိုရင် cloud model နဲ့ "
        "ချက်ချင်း သုံးနိုင်ပါပြီ။")

    def _key_guidance(self, user_input: str):
        """The guidance reply for a settings blocker, or None when the
        blocker is not a settings state (offline, local-only, backend
        unavailable…). Covers both fresh-install states: no usable key
        (Release 25) and key-present-but-cost-not-acknowledged
        (Release 26)."""
        be = self._backend_obj()
        if be is None:
            return None
        try:
            blocker = be.why_not_cloud()
        except Exception as e:
            log.debug("key guidance check failed: %s: %s",
                      type(e).__name__, e)
            return None
        text = {"no_usable_key": self.NO_KEY_GUIDANCE,
                "cost_not_acknowledged": self.COST_ACK_GUIDANCE}.get(blocker)
        if text is None:
            return None
        self._messages.append({"role": "user", "text": user_input})
        self._messages.append({"role": "assistant", "text": text})
        self._trim_messages()
        self._save_turn()
        record_usage("")
        return text

    def _budget_notice(self, max_iter: int, partial_text: str = "") -> str:
        """Burmese notice for a turn the step budget cut short (Release 15).

        Deterministic — built only from this turn's transcript, never from
        a provider call, so it costs nothing and cannot itself fail. Which
        tools ran (and how often) is read back from the tool_call entries
        appended since the user's message, in first-use order; whatever
        text the model produced alongside its last, unanswered tool
        request rides along so no finished work is hidden.
        """
        counts = {}
        order = []
        for m in reversed(self._messages):
            role = m.get("role")
            if role == "user":
                break
            if role == "tool_call":
                name = m.get("tool") or ""
                if not name:
                    continue
                if name not in counts:
                    counts[name] = 0
                    order.append(name)
                counts[name] += 1
        order.reverse()   # scanned newest-first; present in first-use order
        lines = ["⏱️ အဆင့် (step) ကန့်သတ် %d ဆင့် ပြည့်သွားလို့ "
                 "ဒီအလှည့်ကို ဒီတင် ရပ်လိုက်ပါတယ်။" % max_iter]
        if order:
            used = "၊ ".join("%s %d ကြိမ်" % (name, counts[name])
                              for name in order)
            lines.append("ဒီအလှည့်မှာ သုံးခဲ့တဲ့ tool တွေ — " + used + "။")
        partial = (partial_text or "").strip()
        if partial:
            lines.append("မရပ်ခင် model နောက်ဆုံး ပြောထားတာ:\n" + partial)
        lines.append("ဆက်လုပ်စေချင်ရင် \"ဆက်လုပ်ပါ\" လို့ ပြန်ပြောပါ — "
                     "ကျန်နေတဲ့ အလုပ်ကို ဆက်လုပ်ပေးပါမယ်။")
        log.info("step budget exhausted: %d iterations, tools=%s",
                 max_iter, order)
        return "\n".join(lines)

    def _handle_cloud(self, user_input: str, be, system: str = "",
                      image=None, turn_only=None) -> str:
        """Mirror of the local tool loop, but every step goes to the provider.

        ``image`` (Release 29): the live turn's image payload source —
        {"path"/"data", "mime"} — attached to the appended user entry
        for this turn only. ``turn_only``: the vision-capable route
        subset _try_cloud resolved; the step closure applies it to
        every step that does not narrow the route itself, so the whole
        turn (image in history throughout) stays on models that can
        read the image.
        """
        import json as _json

        mode = self._mode()
        entry = {"role": "user", "text": user_input}
        if image is not None:
            entry["image"] = _wire_image(image)
        self._messages.append(entry)
        self.turns += 1
        system = self._with_memory(system or self._system_prompt()[0])

        def _step(only=None, tools=None):
            step = be.step(system=system, messages=self._messages,
                           tools=self._tools if tools is None else tools,
                           max_tokens=config.agent_max_tokens(self.name),
                           only=only if only is not None else turn_only)
            # Recorded per successful step, not once at the end: the provider
            # can change mid-turn when a key runs out, and the chip must show
            # the model that produced the answer the user is reading.
            record_usage("cloud", step.model or "", step.provider or "", mode)
            meter_usage("cloud", step.model or "", step.provider or "",
                        step.usage)
            return step

        step = _step()
        max_iter = config.agent_max_iter(self.name)
        final_text = ""
        _seen_calls = set()   # same dedup guard as the local loop

        # Batch 4.2i: capability nudge via shared helper (one per turn)
        _first_names = [
            (tc.get("function", {}) or {}).get("name", "")
            for tc in (step.tool_calls or [])
        ]
        if self._maybe_nudge_capability(_first_names):
            step = _step()

        for _ in range(max_iter):
            if not step.tool_calls:
                final_text = step.text or final_text
                break
            # Batch 4.2b: try parallel batch when safe and no dups.
            if self._batch_parallel_ok(step.tool_calls):
                results = self._execute_tools_batch(step.tool_calls)
                for tc, r in zip(step.tool_calls, results):
                    name = r["name"]
                    args = r["args"]
                    try:
                        _key = (name, _json.dumps(args, sort_keys=True))
                    except Exception:
                        _key = (name, str(args))
                    _seen_calls.add(_key)
                    self._messages.append({"role": "tool_call", "tool": name,
                                           "args": args,
                                           "reasoning": step.reasoning})
                    self._turn_iter += 1
                    self._messages.append({"role": "tool_result", "tool": name,
                                           "text": r["result"]})
                    self._save_checkpoint()   # Release 19: result is durable
            else:
                for tc in step.tool_calls:
                    fn_info = tc.get("function", {}) if isinstance(tc, dict) else {}
                    name = fn_info.get("name", "")
                    args = fn_info.get("arguments", {}) or {}
                    try:
                        _key = (name, _json.dumps(args, sort_keys=True))
                    except Exception:
                        _key = (name, str(args))
                    if _key in _seen_calls:
                        self._messages.append({
                            "role": "tool_result", "tool": name,
                            "text": self._cached_text(
                                name, args,
                                f"(skipped: {name} already called with "
                                f"identical args this turn)"),
                        })
                        self._save_checkpoint()   # Release 19
                        continue
                    _seen_calls.add(_key)
                    # `reasoning` rides along so the next provider step
                    # can pass it back — DeepSeek's thinking mode refuses
                    # the turn with a 400 otherwise.
                    self._messages.append({"role": "tool_call", "tool": name,
                                           "args": args,
                                           "reasoning": step.reasoning})
                    result = self._execute_tool(name, args)
                    self._turn_iter += 1   # budget: one more loop iteration spent
                    self._messages.append({"role": "tool_result", "tool": name,
                                           "text": result})
                    self._save_checkpoint()   # Release 19: result is durable
            step = _step()
        # Release 15: the loop only falls through with tool calls still
        # pending when the step budget ran out mid-work — the refreshed
        # step's request was never executed. Say so, with what did get
        # done, instead of returning its bare text (normally "").
        exhausted = bool(step.tool_calls)
        if exhausted:
            final_text = self._budget_notice(max_iter, step.text or "")
        elif not final_text:
            final_text = step.text or ""

        # One reviewer pass, cloud only (Decision 4) — and only when a tool
        # actually produced evidence this turn.
        if exhausted:
            # The notice is generated text, not a model answer — there is
            # no candidate for the reviewer to check.
            log.info("verify skipped: step budget exhausted (notice turn)")
        elif self._should_skip_verify():
            # Batch 4.2i: nudge-fired turns are tool-grounded.
            log.info("verify skipped: capability nudge turn (tool-grounded)")
        else:
            final_text = self._verify_cloud(be, final_text, _step)
        if not exhausted:
            if image is not None:
                # Release 29: the quality retry re-asks on the NEXT
                # route provider — for an image turn that may be a
                # text-only model, which would receive a photo it
                # cannot read. Image turns keep their single pass.
                log.info("quality retry skipped: image turn")
            else:
                # Release 26: Burmese quality gate on the final text
                # (the budget notice is generated text — never gated,
                # like the verify skip above).
                final_text = self._quality_retry(
                    be, final_text, _step, answerer=step.provider or "")
        self._messages.append({"role": "assistant", "text": final_text,
                               "reasoning": step.reasoning})
        # A cloud turn's history is summarized by this same backend
        # (Release 18 revision) — never by the on-device engine.
        self._cap_history(summarize=self._cloud_summarizer(be))
        self._save_turn()
        return final_text

    def _reviewer_reply(self, be, candidate: str) -> str:
        """Ask the provider to review ``candidate`` against the tool results.

        The request carries a fresh one-message list, not ``self._messages``, so
        the review neither reads the whole conversation nor writes anything into
        the transcript the user will see. ``record_usage`` is deliberately not
        called: the reviewer is not the model that produced the answer.
        """
        evidence = _compact_tool_results(self._messages)
        reply = be.step(
            system=_VERIFY_SYSTEM,
            messages=[{"role": "user", "text":
                       "TOOL RESULTS:\n%s\n\nCANDIDATE:\n%s" % (evidence, candidate)}],
            tools=[],
            max_tokens=config.AGENT_VERIFY_MAX_TOKENS,
        )
        return getattr(reply, "text", "") or ""

    def _should_skip_verify(self) -> bool:
        """Batch 4.2i: skip reviewer on capability-nudge turns.

        When the capability nudge fires, the follow-up answer is produced
        from capability_list output — it is tool-grounded. A reviewer pass
        adds noise and (with small models) sometimes replaces the analysis
        with a shorter rewrite. Deterministic structural fix; no prompt
        change, no length heuristic.
        """
        return getattr(self, "_cap_nudged", False)

    def _verify_cloud(self, be, candidate: str, step_fn) -> str:
        """Return the answer to show, after one review pass.

        Verification may improve an answer, never lose one: every failure path
        returns ``candidate`` untouched.
        """
        if not candidate or not config.AGENT_VERIFY_ENABLED:
            return candidate
        if not any(m.get("role") == "tool_result" for m in self._messages):
            return candidate          # pure chat — nothing to check against

        try:
            verdict, value = _decide_verify(
                self._reviewer_reply(be, candidate), candidate)
        except Exception as e:
            log.info("verification skipped (%s: %s)", type(e).__name__, e)
            return candidate
        log.info("verification: %s", verdict)

        if verdict == "FIX":
            return value or candidate
        if verdict != "MISSING":
            return candidate

        # MISSING → exactly _VERIFY_EXTRA_ROUNDS extra round(s). A round is the same
        # two-step shape the main loop uses: step, run whatever tools it asked
        # for, step again for the text. Anything further is ignored rather than
        # verified again — the reviewer is never called from inside itself, so
        # recursion is impossible by construction.
        for _ in range(_VERIFY_EXTRA_ROUNDS):
            try:
                step = step_fn()
            except Exception as e:
                log.info("verification follow-up failed (%s: %s)",
                         type(e).__name__, e)
                return candidate
            if not getattr(step, "tool_calls", None):
                return (getattr(step, "text", "") or "").strip() or candidate
            for tc in step.tool_calls:
                fn_info = tc.get("function", {}) if isinstance(tc, dict) else {}
                name = fn_info.get("name", "")
                args = fn_info.get("arguments", {}) or {}
                # Same message shape as the main loop, or the transcript would
                # carry a tool_result with no tool_call and the trimmer would
                # drop it as an orphan.
                self._messages.append({"role": "tool_call", "tool": name,
                                       "args": args,
                                       "reasoning": step.reasoning})
                result = self._execute_tool(name, args)
                self._messages.append({"role": "tool_result", "tool": name,
                                       "text": result})
            # The new evidence only helps if the model gets to read it.
            try:
                step = step_fn()
            except Exception as e:
                log.info("verification follow-up read-back failed (%s: %s)",
                         type(e).__name__, e)
                return candidate
            return (getattr(step, "text", "") or "").strip() or candidate
        return candidate

    def _quality_retry(self, be, final_text: str, step_fn,
                       answerer: str = "") -> str:
        """Release 26: one retry when a Burmese answer fails the gate.

        The free-model roulette (tester evaluation #1/#7): whichever
        model the free router picks this turn may answer in script
        salad. core.burmese.reply_quality judges the FINAL text only —
        never a tool-using turn mid-flight — and when it fails, the
        answer is re-asked ONCE through the next provider in the live
        route (tools=[], so the retry can only produce text; it is
        metered like any step because it goes through the callers'
        _step closure). A retry that also fails yields the
        better-scored of the two; any error yields the original. The
        gate is deterministic and script-level: well-formed Burmese
        that means the wrong thing is beyond it, and it does not
        pretend otherwise. AI_AGENT_QUALITY_GATE=off disables it.
        """
        import os as _os
        if _os.environ.get("AI_AGENT_QUALITY_GATE", "").lower() == "off":
            return final_text
        if not final_text:
            return final_text
        from core import burmese
        user_text = ""
        for m in reversed(self._messages):
            if m.get("role") == "user":
                user_text = m.get("text") or ""
                break
        if not burmese.has_myanmar(user_text):
            return final_text   # the turn was not asked in Burmese
        ok, score = burmese.reply_quality(final_text)
        if ok:
            return final_text
        try:
            route = be.route_providers()
        except Exception as e:
            log.debug("quality gate: route unavailable (%s: %s)",
                      type(e).__name__, e)
            return final_text
        candidates = [p for p in route if p and p != answerer]
        if not candidates:
            log.info("quality gate failed (score %.2f) but the route "
                     "has no other provider — keeping the original", score)
            return final_text
        target = candidates[0]
        log.warning("burmese quality gate failed (score %.2f) — one "
                    "retry via %s", score, target)
        try:
            step = step_fn(only=[target], tools=[])
        except Exception as e:
            log.info("quality retry failed (%s: %s) — keeping the "
                     "original", type(e).__name__, e)
            return final_text
        cand = (getattr(step, "text", "") or "").strip()
        if not cand:
            return final_text
        ok2, score2 = burmese.reply_quality(cand)
        if ok2:
            log.info("quality retry via %s passed the gate", target)
            return cand
        log.info("quality retry also failed the gate (score %.2f) — "
                 "keeping the better of the two", score2)
        return cand if score2 > score else final_text

    # ── streaming cloud attempt (Batch 2b) ────────────────────────────────
    #
    # The same cloud turn _try_cloud/_handle_cloud run; only the answer is
    # produced incrementally. Part 3's SSE route owns the turn setup —
    # begin_turn(), _clear_turn_cache() and the {"role": "user", ...} append
    # happen before it iterates _handle_cloud_stream() — because neither
    # method below takes user_input. No path here touches the local engine:
    # an empty stream is the signal for the route to fall back.

    def _maybe_nudge_capability(self, tool_names):
        """Append a capability nudge once per turn when needed.

        Returns True if a nudge was appended (caller should re-step).
        Never raises.
        """
        try:
            if getattr(self, "_cap_nudged", False):
                return False
            # Find the most recent non-nudge user message
            user_text = ""
            for m in reversed(self._messages):
                if m.get("role") == "user":
                    t = str(m.get("text", ""))
                    if "SYSTEM REMINDER" in t:
                        continue
                    user_text = t
                    break
            if not _is_capability_query(user_text):
                return False
            names = [n for n in (tool_names or []) if n]
            if any(t in _TOOL_REQUIRED_FOR_CAPABILITY for t in names):
                return False
            self._cap_nudged = True
            self._messages.append({"role": "user",
                                    "text": _capability_nudge()})
            log.info("capability nudge appended (no capability tool in first round)")
            return True
        except Exception as e:
            log.debug("capability nudge helper failed: %s", e)
            return False

    def _try_cloud_stream(self):
        """Generator. Yields:

            ("delta", str)
            ("tool", {"name": ..., "args": ..., "result": ...})
            ("error", {...})
            ("end", final_text)

        Yields nothing if cloud is unusable or declines — the caller then
        decides the fallback (Part 3, same job handle()'s cloud_text check
        does). Errors
        are forwarded verbatim and end the stream; the local engine is never
        called from here.
        """
        # mirrors _try_cloud at L599 — keep in sync on any change
        mode = self._mode()          # fresh from disk, every turn
        if mode == "local":
            return              # Local Only never leaves this function
        if self._is_http() and mode != "cloud":
            return              # mirrors _try_cloud at L602 — keep in sync on any change
        be = self._backend_obj()
        if be is None:
            return              # mirrors _try_cloud at L604 — keep in sync on any change
        try:
            want = be.should_use_cloud()
            reason = "" if want else (be.why_not_cloud() or "no_usable_key")
        except Exception as e:
            log.warning("cloud check failed (%s)", e)
            return              # mirrors _try_cloud at L611 — keep in sync on any change
        if not want:
            # mirrors _try_cloud at L619 — keep in sync on any change.
            # Cloud Only answers there with _cloud_only_reply(); a generator
            # cannot, so this yields nothing and Part 3 renders the refusal
            # (handle()'s Cloud Only branch).
            log.info("cloud stream declined: %s", reason)
            return
        # mirrors _handle_cloud at L688 — keep in sync on any change. The SSE
        # route passes no system, so the frozen per-turn tier is resolved
        # here exactly as it is there
        system = self._with_memory(self._system_prompt()[0])
        max_iter = config.agent_max_iter(self.name)   # mirrors _handle_cloud at L701 — keep in sync on any change
        _seen_calls = set()                           # mirrors _handle_cloud at L703 — keep in sync on any change
        # Read back by _handle_cloud_stream at commit time: only the last
        # ("done", ...) frame carries the reasoning the final answer was
        # produced with, and normalize() (core/agent_backend.py L607) hands
        # it back as reasoning_content — without it the NEXT turn's first
        # DeepSeek thinking request dies with a 400.
        self._stream_reasoning = ""
        self._stream_provider = ""
        # Release 15: read back by _handle_cloud_stream at commit time,
        # the same way _stream_reasoning is — True when the end frame
        # carries the budget notice instead of a model answer.
        self._stream_budget_exhausted = False
        final_text = ""

        def _pump_round():
            """One provider step: forward its deltas, then hand back
            ``(info, text)``. ``info`` is None when the stream came back
            empty (every provider failed) or died in-band mid-answer —
            both mean no ("end", ...) leaves this generator.
            """
            # mirrors _handle_cloud at L691 — keep in sync on any change.
            # Same system/messages/tools/max_tokens; http/http_stream stay
            # at step_stream's defaults, which is exactly what step() gets
            # at L691 (it passes neither).
            text = ""
            info = None
            stream = be.step_stream(
                system=system, messages=self._messages, tools=self._tools,
                max_tokens=config.agent_max_tokens(self.name))
            for kind, payload in stream:
                if kind == "delta":
                    text += payload
                    yield ("delta", payload)     # forwarded verbatim
                elif kind == "done":
                    info = payload
                    # Release 26: remember who answered — the quality
                    # gate's retry must go to a DIFFERENT provider.
                    self._stream_provider = info.get("provider") or ""
                    # mirrors _handle_cloud at L697 — keep in sync on any
                    # change. Metered inside the pump, so even the round
                    # refreshed after the last tools (the L736 step) is
                    # recorded — exactly as _step() records before the loop
                    # ever looks at its tool_calls.
                    record_usage("cloud", info.get("model") or "",
                                 info.get("provider") or "", mode)
                    meter_usage("cloud", info.get("model") or "",
                                info.get("provider") or "",
                                info.get("usage"))
                    self._stream_reasoning = info.get("reasoning") or ""
                elif kind == "error":
                    yield ("error", payload)     # forwarded verbatim
                    return None
            return (info, text)

        # mirrors _handle_cloud at L700 — the first step, before the loop
        info, round_text = (yield from _pump_round()) or (None, "")
        ended = False
        # Batch 4.2i: capability nudge (stream path) — if the user
        # asked about capabilities and the first round missed the
        # required tool, remind once and re-pump.
        if info is not None:
            _first_names = [
                (tc.get("function", {}) or {}).get("name", "")
                for tc in (info.get("tool_calls") or [])
            ]
            if self._maybe_nudge_capability(_first_names):
                info, round_text = (yield from _pump_round()) or (None, "")
        for _ in range(max_iter):                     # mirrors _handle_cloud at L704 — keep in sync on any change
            if info is None:
                # step_stream's empty-stream signal: every provider failed
                # before a single byte — no ("end", ...) → route falls back.
                return
            if not info.get("tool_calls"):
                final_text = round_text or final_text   # mirrors _handle_cloud at L706 — keep in sync on any change
                ended = True
                break
            # Batch 4.2b: stream parallel — when safe and no dups.
            if self._batch_parallel_ok(info["tool_calls"]):
                results = self._execute_tools_batch(info["tool_calls"])
                for tc, r in zip(info["tool_calls"], results):
                    name = r["name"]
                    args = r["args"]
                    try:
                        _key = (name, json.dumps(args, sort_keys=True))
                    except Exception:
                        _key = (name, str(args))
                    _seen_calls.add(_key)
                    self._messages.append({"role": "tool_call", "tool": name,
                                           "args": args,
                                           "reasoning": info.get("reasoning")})
                    self._turn_iter += 1
                    self._messages.append({"role": "tool_result", "tool": name,
                                           "text": r["result"]})
                    self._save_checkpoint()   # Release 19: before the yield
                    yield ("tool", {"name": name, "args": args,
                                    "result": r["result"]})
            else:
                for tc in info["tool_calls"]:
                    # mirrors _handle_cloud at L708 — keep in sync on any change
                    fn_info = tc.get("function", {}) if isinstance(tc, dict) else {}
                    name = fn_info.get("name", "")
                    args = fn_info.get("arguments", {}) or {}
                    try:
                        _key = (name, json.dumps(args, sort_keys=True))
                    except Exception:
                        _key = (name, str(args))
                    if _key in _seen_calls:
                        # mirrors _handle_cloud at L716 — keep in sync on any
                        # change. Skip the repeat, but hand the model its data
                        # back instead of losing it
                        self._messages.append({
                            "role": "tool_result", "tool": name,
                            "text": self._cached_text(
                                name, args,
                                f"(skipped: {name} already called with "
                                f"identical args this turn)"),
                        })
                        self._save_checkpoint()   # Release 19
                        continue
                    _seen_calls.add(_key)
                    # mirrors _handle_cloud at L729 — keep in sync on any change.
                    # `reasoning` rides along so the next provider step can pass
                    # it back (DeepSeek 400)
                    self._messages.append({"role": "tool_call", "tool": name,
                                           "args": args,
                                           "reasoning": info.get("reasoning")})
                    result = self._execute_tool(name, args)   # mirrors _handle_cloud at L732 — keep in sync on any change
                    self._turn_iter += 1   # budget: one more loop iteration spent
                    self._messages.append({"role": "tool_result", "tool": name,
                                           "text": result})   # mirrors _handle_cloud at L734 — keep in sync on any change
                    self._save_checkpoint()   # Release 19: before the yield
                    yield ("tool", {"name": name, "args": args,
                                    "result": result})
            # mirrors _handle_cloud at L736 — step = _step() for the next round
            info, round_text = (yield from _pump_round()) or (None, "")
        if not ended:
            if info is None:
                # mirrors _handle_cloud at L736 — the refresh after the last
                # tool round came back empty; there _step() raises and
                # _try_cloud rolls back (L648), so nothing commits here either
                return
            if info.get("tool_calls"):
                # Release 15: the budget ran out with work still pending —
                # the end frame carries the budget notice instead of the
                # last round's bare text (the client replaces its buffer
                # with the end text, so the partial round text rides
                # inside the notice), and the flag tells
                # _handle_cloud_stream to skip the verify pass.
                self._stream_budget_exhausted = True
                final_text = self._budget_notice(max_iter, round_text)
            else:
                # mirrors _handle_cloud at L737 — keep in sync on any change.
                # The last pumped round's text stands — exactly what
                # `final_text = step.text or ""` does there
                final_text = round_text
        yield ("end", final_text)

    def _handle_cloud_stream(self):
        """Wrap _try_cloud_stream with _handle_cloud's finalization
        (verify pass, sanitize, commit/rollback).

        Call contract (Part 3): the route performs begin_turn(),
        _clear_turn_cache() and the user-message append BEFORE iterating
        this generator — neither streaming method takes user_input. ``before``
        is captured after that append, so every rollback below leaves the
        question in the transcript for the local fallback to answer (the
        non-streaming rollback at L648 also removes it, because
        _handle_cloud appended it itself).

        Yields every ("delta"|"tool"|"error", ...) frame verbatim, then at
        most one ("end", final_text) carrying the *verified* text now in the
        transcript — the frontend replaces its buffer with it, never appends.
        """
        before = len(self._messages)
        any_output = False
        partial = ""
        final_text = None
        failed_commit = False   # Release 24: failure already committed
        try:
            for kind, payload in self._try_cloud_stream():
                if kind == "end":
                    final_text = payload
                    continue      # verified below, re-emitted once at the end
                if kind == "delta":
                    any_output = True
                    partial += payload
                elif kind == "tool":
                    any_output = True
                yield kind, payload    # forwarded verbatim
            if final_text is None:
                # Cloud declined, every provider failed, or the stream died
                # after the first delta: nothing committed, and no local call
                # here — Part 3 decides (the Cloud Only refusal stays Part
                # 3's job too, as in handle()'s Cloud Only branch). Same
                # rollback the non-streaming failure path performs.
                # mirrors _try_cloud at L648 — keep in sync on any change
                #
                # Release 24 exception: when tools already executed, the
                # rollback would erase their only record and Part 3's
                # fallback would re-run the whole turn, repeating the
                # side effects. Commit the turn with the failure reply
                # instead — the same preservation _try_cloud performs.
                turn_entries = self._messages[before:]
                if any(m.get("role") == "tool_result"
                       for m in turn_entries):
                    final_text = self._commit_failed_stream_turn(
                        turn_entries)
                    failed_commit = True
                else:
                    del self._messages[before:]
                    return
            if not failed_commit:
                be = self._backend_obj()
                if be is not None:
                    mode = self._mode()
                    system = self._with_memory(self._system_prompt()[0])

                    def _step(only=None, tools=None):
                        # mirrors _handle_cloud at L690 — keep in sync on any change
                        step = be.step(system=system, messages=self._messages,
                                       tools=self._tools if tools is None else tools,
                                       max_tokens=config.agent_max_tokens(self.name),
                                       only=only)
                        record_usage("cloud", step.model or "", step.provider or "",
                                     mode)
                        meter_usage("cloud", step.model or "",
                                    step.provider or "", step.usage)
                        return step

                    # mirrors _handle_cloud at L742 — keep in sync on any change.
                    # One reviewer pass, running on the unmodified
                    # _verify_cloud/_reviewer_reply
                    if getattr(self, "_stream_budget_exhausted", False):
                        # Release 15: the end text is the generated budget
                        # notice, not a model answer — nothing to verify.
                        log.info("verify skipped: step budget exhausted "
                                 "(notice turn)")
                    elif self._should_skip_verify():
                        # Batch 4.2i: nudge-fired turns are tool-grounded.
                        log.info("verify skipped: capability nudge turn (tool-grounded)")
                    else:
                        final_text = self._verify_cloud(be, final_text, _step)
                if not getattr(self, "_stream_budget_exhausted", False):
                    # Release 26: Burmese quality gate on the final
                    # text — mirrors _handle_cloud. The ("end", ...)
                    # frame below REPLACES the streamed buffer in the
                    # UI, so a retry here is what the user ends up
                    # reading even though deltas already flowed.
                    final_text = self._quality_retry(
                        be, final_text, _step,
                        answerer=getattr(self, "_stream_provider", ""))
                # Sanitize + commit, the mechanism _handle_cloud uses at L744-747:
                # tool output was already banner-ed and redacted inside
                # _execute_tool/_guard_result on the way in, so there is nothing
                # left to strip from the answer — then cap history, then persist.
                # mirrors _handle_cloud at L744 — keep in sync on any change
                self.turns += 1
                self._messages.append({"role": "assistant", "text": final_text,
                                       "reasoning": getattr(
                                           self, "_stream_reasoning", "")})
                # Same backend-matched summarizer as _handle_cloud.
                self._cap_history(summarize=self._cloud_summarizer(be))
                self._save_turn()
        except GeneratorExit:
            if any_output:
                # Commit the partial answer — only because real output
                # already reached the client. Those deltas and tool
                # round-trips happened and were rendered, and the
                # tool_call/tool_result pairs are already in the transcript:
                # rolling back now would leave history shorter than what the
                # user saw and one assistant turn short of balanced. The
                # verify pass is deliberately skipped — no client is left to
                # correct, and the reviewer would spend a provider call on it.
                # mirrors _handle_cloud at L744 — keep in sync on any change
                self.turns += 1
                self._messages.append({"role": "assistant", "text": partial,
                                       "reasoning": getattr(
                                           self, "_stream_reasoning", "")})
                # The stream ran on the cloud backend, so its history
                # is summarized there too (Release 18 revision);
                # _backend_obj() returns the cached object and cannot
                # raise — a None backend degrades to the local core.
                self._cap_history(
                    summarize=self._cloud_summarizer(self._backend_obj()))
                self._save_turn()
            else:
                # mirrors _try_cloud at L648 — keep in sync on any change
                del self._messages[before:]
            raise
        except Exception as e:
            # mirrors _try_cloud at L648 — keep in sync on any change. Same
            # rollback the non-streaming failure path performs, then the
            # error keeps travelling so the route can report it.
            #
            # Release 24 exception, same rule as the no-end-frame path
            # above: when tools already executed, preserve the turn
            # instead. The error is logged and the turn ends with the
            # failure reply as its end frame — re-raising would send the
            # route into a fallback that rolls this transcript back and
            # re-runs the turn, repeating the side effects. If the
            # preservation itself fails, the old behaviour stands.
            turn_entries = self._messages[before:]
            if any(m.get("role") == "tool_result" for m in turn_entries):
                log.error("cloud stream died mid-work (%s: %s) — "
                          "preserving the completed tool work",
                          type(e).__name__, e)
                try:
                    final_text = self._commit_failed_stream_turn(
                        turn_entries)
                except Exception:
                    del self._messages[before:]
                    raise
            else:
                del self._messages[before:]
                raise
        yield ("end", final_text)

    def _commit_failed_stream_turn(self, turn_entries) -> str:
        """Commit a streamed cloud turn that died after running tools.

        The streamed twin of _preserve_failed_turn: the transcript
        already holds the route's user append and the turn's
        tool_call/tool_result entries — append the failure reply
        (work summary + classified cause from key state; a stream
        failure raises no exception to classify from), trim, save.
        _save_turn clears the checkpoint: the transcript now carries
        the work and the reply says so.
        """
        be = self._backend_obj()
        mode = self._mode()
        classified = self._classify_failure(be, None)
        hint = (self._second_key_hint(be)
                if "quota_exhausted" in classified["reasons"] else "")
        text = self._failure_turn_text(
            mode, classified, self._work_summary(turn_entries),
            hint=hint)
        self.turns += 1
        self._messages.append({"role": "assistant", "text": text})
        self._trim_messages()
        self._save_turn()
        record_usage("")
        return text

    def handle(self, user_input: str, image=None) -> str:
        """One user turn. ``image`` (Release 29) is the live turn's
        image payload source for the CLOUD path; the on-device vision
        path lives on the v1 SSE route (build_vision_message), so an
        image the cloud declines is not re-attached here."""
        if not user_input or not user_input.strip():
            raise ValueError("user_input must be non-empty")
        self._clear_turn_cache()   # a new user turn starts a new cache
        self.begin_turn(user_input)  # …and a new prompt tier + effort
        mode = self._mode()   # read once: Cloud Only must be decided up front
        if self._is_http() and mode != "cloud":
            from core.http_llm import _post, CHAT_TIMEOUT, HTTPLLMError
            j = _post(
                f"{self.llm.server_url}/api/agent",
                {"message": user_input},
                timeout=CHAT_TIMEOUT,
            )
            if "error" in j:
                raise HTTPLLMError(str(j["error"]))
            text = j.get("reply", "")
            self._server_ctx = j.get("ctx")  # Bug #4: cache server ctx
            self.turns += 1
            self._messages.append({"role": "user", "text": user_input})
            self._messages.append({"role": "assistant", "text": text})
            self._trim_messages()   # F2: cap history
            # The transport is HTTP but the engine behind it is still the
            # on-device one — that is what answered, so that is what the
            # status bar must show.
            record_usage("local", self._local_label(), "", mode)
            # C5b: HTTP mode — server owns session; skip local save.
            self._finish_turn(user_input, text)
            return text
        # WiFi + cloud configured → answer from the provider; anything else
        # (no network, no usable key, prefer_cloud off, provider error) falls
        # through to the local Gemma 4B path below, unchanged.
        #
        # The prompt is resolved here, once, and handed to both branches: a
        # turn that falls back mid-way keeps the prompt it started with.
        system = self._system_prompt()[0]
        cloud_text = self._try_cloud(user_input, system=system,
                                     image=image)
        if cloud_text is not None:
            self._finish_turn(user_input, cloud_text)
            return cloud_text
        if mode == "cloud":
            # Belt and braces: Cloud Only never reaches the local model below.
            refusal = self._cloud_only_reply(user_input, "exhausted")
            self._finish_turn(user_input, refusal)
            return refusal
        try:
            self._ensure_conv(system=system)
        except Exception:
            # Release 25: fresh install (no keys, no model file) — the
            # raw "Model not found" error never told the user the real
            # fix is a key. Release 26: the same applies when a key
            # exists but the cost acknowledgement is missing. Guidance
            # applies only for those settings blockers; anything else
            # keeps the original error.
            guide = self._key_guidance(user_input)
            if guide is None:
                raise
            self._finish_turn(user_input, guide)
            return guide
        assert self._conv is not None

        # Bug #3: manual tool-calling loop (transparent + persistable)
        turn_start = len(self._messages)   # Release 24: this turn's slice
        self._messages.append({"role": "user", "text": user_input})
        resp = self._conv.send_message(user_input)
        self.turns += 1

        max_iter = config.agent_max_iter(self.name)
        final_text = ""
        _seen_calls = set()  # A8c: dedup guard
        _finished = False   # Release 15: the loop saw a tool-free response
        try:
            for _ in range(max_iter):
                tool_calls = None
                if isinstance(resp, dict):
                    tool_calls = resp.get("tool_calls")
                if not tool_calls:
                    # No more tool calls — extract final text
                    try:
                        final_text = resp["content"][0]["text"]
                    except (KeyError, IndexError, TypeError):
                        final_text = str(resp)
                    _finished = True
                    break

                # Save tool call to _messages
                for tc in tool_calls:
                    fn_info = tc.get("function", {}) if isinstance(tc, dict) else {}
                    name = fn_info.get("name", "")
                    args = fn_info.get("arguments", {})
                    # ── A8c: dedup guard ──
                    import json as _j
                    try:
                        _key = (name, _j.dumps(args, sort_keys=True))
                    except Exception:
                        _key = (name, str(args))
                    if _key in _seen_calls:
                        _skip = self._cached_text(
                            name, args,
                            f"(skipped: {name} already called with identical args this turn)")
                        self._messages.append({
                            "role": "tool_result",
                            "tool": name,
                            "text": _skip,
                        })
                        self._save_checkpoint()   # Release 19
                        resp = self._conv.send_message({
                            "role": "tool",
                            "name": name,
                            "content": _skip,
                        })
                        continue
                    _seen_calls.add(_key)
                    # ── end guard ──
                    self._messages.append({
                        "role": "tool_call",
                        "tool": name,
                        "args": args,
                    })
                    result = self._execute_tool(name, args)
                    self._turn_iter += 1   # budget: one more loop iteration spent
                    self._messages.append({
                        "role": "tool_result",
                        "tool": name,
                        "text": result,
                    })
                    self._save_checkpoint()   # Release 19: result is durable
                    # Send result back to LLM
                    resp = self._conv.send_message({
                        "role": "tool",
                        "name": name,
                        "content": result,
                    })
        except Exception as e:
            # Release 24: the engine died mid-work. With tool
            # results already in the transcript, the turn must not
            # die with the exception — its completed work would be
            # invisible in the saved transcript (the cloud path's
            # v1.21.0 failure, local edition). Close the turn with
            # the failure reply instead and let the common tail
            # below save it; with no tool work done, the old
            # behaviour (the exception travels) stands.
            if not any(m.get("role") == "tool_result"
                       for m in self._messages[turn_start:]):
                raise
            log.error("local turn died mid-work (%s: %s) — "
                      "preserving the completed tool work",
                      type(e).__name__, e)
            final_text = self._local_failure_text(
                self._messages[turn_start:])
            _finished = True

        if not _finished:
            # Release 15: the budget ran out before the loop could look at
            # the last response. If it still asks for tools, the turn was
            # cut short mid-work — say so, with what did get done. If it
            # carries the final text instead, the model finished exactly
            # on the last allowed round: recover the text the same way the
            # break branch above does (the cloud loop's step.text fallback
            # does the same).
            pending = (resp.get("tool_calls")
                       if isinstance(resp, dict) else None)
            if pending:
                partial = ""
                try:
                    partial = resp["content"][0]["text"] or ""
                except (KeyError, IndexError, TypeError):
                    partial = ""
                final_text = self._budget_notice(max_iter, partial)
            else:
                try:
                    final_text = resp["content"][0]["text"]
                except (KeyError, IndexError, TypeError):
                    final_text = str(resp)

        self._messages.append({"role": "assistant", "text": final_text})
        self._cap_history()   # F2: cap history (compact, then trim)
        # Only now that the local model really answered — a Cloud Only refusal
        # or a cloud success must never be overwritten by this line.
        record_usage("local", self._local_label(), "", mode)
        meter_usage("local", self._local_label(), "", None)
        if self._store:
            try:
                self._store.save_mode("agent", self._messages)
            except Exception as e:
                # A tool run can span many turns — losing the whole transcript
                # here would be invisible, so it goes to the log.
                log.error("could not save agent turn (session %s): %s: %s",
                          self.session_id, type(e).__name__, e)
            else:
                # Release 19: same rule as _save_turn — the turn is
                # durably saved, so its checkpoint is done.
                self._clear_checkpoint()
        self._finish_turn(user_input, final_text)
        return final_text

    def reload_session(self, sid: str) -> bool:
        """Switch to a different agent session."""
        if not self._store:
            return False
        self._clear_turn_cache()   # the old session's results do not carry over
        try:
            if not self._store.set_active("agent", sid):
                return False
            s = self._store.get_session(sid) or {}
            msgs = list(s.get("messages", []))
            # Close current conv; conversation state is in _messages
            if self._conv is not None:
                try:
                    self._conv.__exit__(None, None, None)
                except Exception as e:
                    log.debug("could not close conv on session reload: %s: %s",
                              type(e).__name__, e)
                self._conv = None
            self._messages = msgs
            self.session_id = sid
            self.turns = len(msgs) // 2
            # Release 19: switching to a session whose last turn was
            # interrupted resumes that turn here, same as at startup.
            self._restore_checkpoint()
            return True
        except Exception as e:
            log.error("could not switch to agent session %s: %s: %s",
                      sid, type(e).__name__, e)
            return False

    def get_messages(self) -> list:
        return list(self._messages)

    def get_server_ctx(self):
        """Return server-side context usage dict (HTTP mode) or None."""
        return self._server_ctx

    def compress_session(self) -> dict:
        """Summarize session, archive old, replace with summary."""
        if self._store is None or self.session_id is None:
            return {"error": "no session store / id"}

        msgs = list(self._messages)
        if len(msgs) < 4:
            return {"error": f"too few messages ({len(msgs)}, need 4+)"}

        before = 0
        try:
            from core.llm import estimate_messages_tokens
            before = estimate_messages_tokens(msgs)
        except Exception:
            before = 0

        try:
            summary = self._summarize_messages(msgs)
        except Exception as e:
            return {"error": f"summarize failed: {type(e).__name__}: {e}"}

        if not summary:
            return {"error": "empty summary"}

        try:
            archived = self._store.archive_mode("agent")
        except Exception as e:
            return {"error": f"archive failed: {e}"}

        new_msgs = [
            {"role": "user", "text": _SUMMARY_MARKER},
            {"role": "assistant", "text": summary},
        ]
        try:
            self._store.save_mode("agent", new_msgs)
        except Exception as e:
            log.error("could not save summarized agent chat: %s: %s",
                      type(e).__name__, e)

        # Force rebuild of conversation on next use
        if self._conv is not None:
            try:
                self._conv.__exit__(None, None, None)
            except Exception as e:
                log.debug("could not close conv after compress: %s: %s",
                          type(e).__name__, e)
            self._conv = None
        self._messages = new_msgs
        self.session_id = self._store.get_active_id("agent")
        self.turns = 1

        after = 0
        try:
            from core.llm import estimate_messages_tokens
            after = estimate_messages_tokens(new_msgs)
        except Exception:
            after = 0

        saved = 0.0
        if before > 0:
            saved = round((1 - after / before) * 100, 1)

        return {
            "summary": summary,
            "before": before,
            "after": after,
            "saved_pct": saved,
            "archived": archived,
        }

    def reset(self) -> None:
        # getattr: test stubs build AgentMode via __new__ (no __init__),
        # the same reason _save_turn reads session_id defensively.
        old_sid = getattr(self, "session_id", None)
        self._clear_turn_cache()
        if self._conv is not None:
            try:
                self._conv.__exit__(None, None, None)
            except Exception as e:
                log.debug("could not close conv on reset: %s: %s",
                          type(e).__name__, e)
            self._conv = None
        self.turns = 0
        self._messages = []
        if self._store:
            try:
                self._store.clear_mode("agent")
                self.session_id = self._store.get_active_id("agent")
            except Exception as e:
                log.error("could not clear agent session: %s: %s", type(e).__name__, e)
        # Release 19: reset abandons the old session — its in-flight
        # checkpoint goes with it, or a later switch back to that
        # session would resurrect a turn the user threw away.
        if old_sid and old_sid != getattr(self, "session_id", None):
            self._clear_checkpoint(old_sid)

    def close(self) -> None:
        self.reset()
