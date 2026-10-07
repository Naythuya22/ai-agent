"""Agent mode — native tool calling (LiteRT automatic loop)."""
from __future__ import annotations

import json
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
          "tier": "", "tier_source": "", "label": "", "at": 0.0}


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


def record_tier(tier: str, source: str) -> dict:
    """Remember which system prompt this turn was given, and why.

    Separate from record_usage() because it happens once, before any model
    runs: a tier that changed its mind half a turn through would be a lie, and
    the per-step usage updates must not be able to overwrite it.
    """
    _USAGE["tier"] = (tier or "").strip()
    _USAGE["tier_source"] = (source or "").strip()
    return dict(_USAGE)


AGENT_SYSTEM = """You are "ညီ" (Nyi), a personal AI assistant for "ကိုကြီး" (Ko Gyi).
Your tone must be warm, respectful, and brotherly.

## Language Rules
- Always reply in Burmese. Use "ညီ" for yourself.
- English is ONLY for technical terms, code, file paths, and commands.
- Never mix languages within a sentence.

## Response Style
- Be concise. Short answers are better than long, broken ones.
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

    def begin_turn(self) -> None:
        """Mark the start of a user turn.

        Called by handle() and by the SSE route, which drives the cloud/local
        branches itself and so never goes through handle(). Between turns this
        is None, which is what makes the first _system_prompt() of a turn
        resolve and every later one reuse.
        """
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
            if mode != "auto" and mode != auto_tier:
                # Worth a log line and a badge in Settings: the user pinned a
                # tier that does not match what this turn is actually running.
                log.warning("prompt tier %s is pinned but this turn looks like "
                            "%s (%s/%s) — using the pinned prompt", mode,
                            auto_tier, backend or "?", model or "?")
            prompt = prompt_tiers.load_prompt(tier)
            hint = self._budget_hint()
            if hint:
                prompt = prompt + "\n\n" + hint
            self._turn_prompt = (prompt, tier)
            record_tier(tier, source)
            log.info("prompt tier %s (source %s) for %s/%s", tier, source,
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
        inside the banner text.
        """
        text = str(result)
        if config.AGENT_INJECTION_GUARD:
            markers = safety.scan_injection(text)
            if markers:
                log.warning("injection markers in %s output: %s",
                            tool_name, ", ".join(markers))
                text = (safety.INJECTION_BANNER_OPEN + "\n"
                        + text + "\n" + safety.INJECTION_BANNER_CLOSE)
        return safety.redact_secrets(text)

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
    }

    def _try_cloud(self, user_input: str, system: str = ""):
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
        if not want:
            if mode == "cloud":
                return self._cloud_only_reply(user_input, reason)
            return None
        before = len(self._messages)
        reason, detail = "exhausted", ""
        try:
            return self._handle_cloud(user_input, be, system=system)
        except Exception as e:
            from core.agent_backend import BackendExhausted
            from core.key_manager import redact
            if isinstance(e, BackendExhausted):
                log.warning("cloud backends exhausted (%s)", e)
                msg = str(e)
                reason = ("offline" if ("network" in msg.lower()
                                        or "no network" in msg.lower())
                          else "exhausted")
                try:
                    detail = redact(msg, be.keys.all_keys())[:220]
                except Exception:
                    detail = msg[:220]
            else:
                log.error("cloud step failed (%s: %s)", type(e).__name__, e)
                detail = "%s: %s" % (type(e).__name__, e)
                try:
                    from core.key_manager import redact
                    detail = redact(detail, be.keys.all_keys())[:220]
                except Exception:
                    detail = detail[:220]
        del self._messages[before:]
        if mode == "cloud":
            return self._cloud_only_reply(user_input, reason, detail)
        return None

    def _save_turn(self) -> None:
        if not self._store:
            return
        try:
            self._store.save_mode("agent", self._messages)
        except Exception as e:
            log.error("could not save agent turn (session %s): %s: %s",
                      getattr(self, "session_id", None), type(e).__name__, e)

    def _cloud_only_reply(self, user_input: str, reason: str,
                          detail: str = "") -> str:
        """Cloud Only mode with no usable provider: say so instead of
        quietly answering from the local 4B model the user just switched off."""
        why = self.CLOUD_ONLY_REASONS.get(reason, reason or "exhausted")
        if detail:
            why = why + " — " + detail
        text = ("☁️ Cloud Only mode ဖြစ်နေလို့ မဖြေရတော့ဘူး: " + why +
                "\n⚙️ Settings → Backend Mode ကနေ Auto ပြောင်းရင် "
                "Local Gemma 4B ကို အလိုအလျောက် fallback ဖြစ်မယ်။")
        self._messages.append({"role": "user", "text": user_input})
        self._messages.append({"role": "assistant", "text": text})
        self._trim_messages()
        self._save_turn()
        # Nothing ran: drop the record so the status bar cannot keep showing
        # a model from an earlier turn for an answer no model produced.
        record_usage("")
        return text

    def _handle_cloud(self, user_input: str, be, system: str = "") -> str:
        """Mirror of the local tool loop, but every step goes to the provider."""
        import json as _json

        mode = self._mode()
        self._messages.append({"role": "user", "text": user_input})
        self.turns += 1
        system = self._with_memory(system or self._system_prompt()[0])

        def _step():
            step = be.step(system=system, messages=self._messages,
                           tools=self._tools,
                           max_tokens=config.agent_max_tokens(self.name))
            # Recorded per successful step, not once at the end: the provider
            # can change mid-turn when a key runs out, and the chip must show
            # the model that produced the answer the user is reading.
            record_usage("cloud", step.model or "", step.provider or "", mode)
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
            step = _step()
        if not final_text:
            final_text = step.text or ""

        # One reviewer pass, cloud only (Decision 4) — and only when a tool
        # actually produced evidence this turn.
        if self._should_skip_verify():
            # Batch 4.2i: nudge-fired turns are tool-grounded.
            log.info("verify skipped: capability nudge turn (tool-grounded)")
        else:
            final_text = self._verify_cloud(be, final_text, _step)
        self._messages.append({"role": "assistant", "text": final_text,
                               "reasoning": step.reasoning})
        self._trim_messages()   # F2: cap history
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
                    # mirrors _handle_cloud at L697 — keep in sync on any
                    # change. Metered inside the pump, so even the round
                    # refreshed after the last tools (the L736 step) is
                    # recorded — exactly as _step() records before the loop
                    # ever looks at its tool_calls.
                    record_usage("cloud", info.get("model") or "",
                                 info.get("provider") or "", mode)
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
            # mirrors _handle_cloud at L737 — keep in sync on any change.
            # The last pumped round's text stands, tool_calls or not —
            # exactly what `final_text = step.text or ""` does there
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
                del self._messages[before:]
                return
            be = self._backend_obj()
            if be is not None:
                mode = self._mode()
                system = self._with_memory(self._system_prompt()[0])

                def _step():
                    # mirrors _handle_cloud at L690 — keep in sync on any change
                    step = be.step(system=system, messages=self._messages,
                                   tools=self._tools,
                                   max_tokens=config.agent_max_tokens(self.name))
                    record_usage("cloud", step.model or "", step.provider or "",
                                 mode)
                    return step

                # mirrors _handle_cloud at L742 — keep in sync on any change.
                # One reviewer pass, running on the unmodified
                # _verify_cloud/_reviewer_reply
                if self._should_skip_verify():
                    # Batch 4.2i: nudge-fired turns are tool-grounded.
                    log.info("verify skipped: capability nudge turn (tool-grounded)")
                else:
                    final_text = self._verify_cloud(be, final_text, _step)
            # Sanitize + commit, the mechanism _handle_cloud uses at L744-747:
            # tool output was already banner-ed and redacted inside
            # _execute_tool/_guard_result on the way in, so there is nothing
            # left to strip from the answer — then cap history, then persist.
            # mirrors _handle_cloud at L744 — keep in sync on any change
            self.turns += 1
            self._messages.append({"role": "assistant", "text": final_text,
                                   "reasoning": getattr(
                                       self, "_stream_reasoning", "")})
            self._trim_messages()   # F2: cap history
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
                self._trim_messages()   # F2: cap history
                self._save_turn()
            else:
                # mirrors _try_cloud at L648 — keep in sync on any change
                del self._messages[before:]
            raise
        except Exception:
            # mirrors _try_cloud at L648 — keep in sync on any change. Same
            # rollback the non-streaming failure path performs, then the
            # error keeps travelling so the route can report it.
            del self._messages[before:]
            raise
        yield ("end", final_text)

    def handle(self, user_input: str) -> str:
        if not user_input or not user_input.strip():
            raise ValueError("user_input must be non-empty")
        self._clear_turn_cache()   # a new user turn starts a new cache
        self.begin_turn()          # …and a new prompt tier
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
        cloud_text = self._try_cloud(user_input, system=system)
        if cloud_text is not None:
            self._finish_turn(user_input, cloud_text)
            return cloud_text
        if mode == "cloud":
            # Belt and braces: Cloud Only never reaches the local model below.
            refusal = self._cloud_only_reply(user_input, "exhausted")
            self._finish_turn(user_input, refusal)
            return refusal
        self._ensure_conv(system=system)
        assert self._conv is not None

        # Bug #3: manual tool-calling loop (transparent + persistable)
        self._messages.append({"role": "user", "text": user_input})
        resp = self._conv.send_message(user_input)
        self.turns += 1

        max_iter = config.agent_max_iter(self.name)
        final_text = ""
        _seen_calls = set()  # A8c: dedup guard
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
                # Send result back to LLM
                resp = self._conv.send_message({
                    "role": "tool",
                    "name": name,
                    "content": result,
                })

        self._messages.append({"role": "assistant", "text": final_text})
        self._trim_messages()   # F2: cap history
        # Only now that the local model really answered — a Cloud Only refusal
        # or a cloud success must never be overwritten by this line.
        record_usage("local", self._local_label(), "", mode)
        if self._store:
            try:
                self._store.save_mode("agent", self._messages)
            except Exception as e:
                # A tool run can span many turns — losing the whole transcript
                # here would be invisible, so it goes to the log.
                log.error("could not save agent turn (session %s): %s: %s",
                          self.session_id, type(e).__name__, e)
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

        convo = "\n".join(
            f"{m.get('role', 'user')}: {m.get('text', '')}" for m in msgs
        )
        prompt = (
            "Summarize the following conversation in 3-5 concise bullet points. "
            "Preserve key facts, user questions, decisions, and context needed to continue.\n\n"
            f"{convo}\n\nSummary:"
        )

        try:
            summary = self.llm.ask(prompt)
        except Exception as e:
            return {"error": f"summarize failed: {type(e).__name__}: {e}"}

        summary = (summary or "").strip()
        if not summary:
            return {"error": "empty summary"}

        try:
            archived = self._store.archive_mode("agent")
        except Exception as e:
            return {"error": f"archive failed: {e}"}

        new_msgs = [
            {"role": "user", "text": "[Earlier conversation — summary]"},
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

    def close(self) -> None:
        self.reset()
