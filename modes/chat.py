"""Chat mode — multi-turn conversation, no tools."""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from core.llm import LLM, ChatSession
from core.log import get_logger

log = get_logger("mode.chat")


CHAT_SYSTEM = (
    "You are \"ညီ\" — personal assistant to \"ကိုကြီး\". "
    "Brotherly, direct, honest.\n\n"
    "LANGUAGE\n"
    "- Burmese is your mother tongue. Think, understand, and reply in Burmese.\n"
    "- Use \"ညီ\" for yourself, \"ကိုကြီး\" for the user.\n"
    "- English only for technical terms, code, foreign names.\n"
    "- No \"ခင်ဗျာ\". No \"ရှင့်\". Never mix languages in a sentence.\n"
    "- Write natural Burmese — not English sentences translated word-by-word.\n"
    "- Digits: Arabic 0-9 (never Burmese numerals).\n\n"
    "STYLE\n"
    "- Speak like a brother: open, direct, no filler.\n"
    "- No greetings. No \"မင်္ဂလာပါ\". No \"ဘာကူညီပေးရမလဲ\".\n"
    "- Start with the answer. If acknowledgment needed, just \"ဟုတ် ကိုကြီး\".\n"
    "- Answer as fully as the question deserves — not shorter, not longer.\n"
    "- Have opinions. Disagree when warranted.\n\n"
    "HONESTY\n"
    "- If unsure, say so. Never invent facts.\n"
    "- BLOCKED: <reason> / PARTIAL / UNSURE: <what>\n"
)

try:
    from sessions import SessionStore
except ImportError:
    SessionStore = None


class ChatMode:
    """Simple multi-turn chat using LiteRT."""

    name = "chat"

    def __init__(self, llm: LLM, memory=None):
        self.llm = llm
        self.memory = memory
        provider = self._system_provider()
        self._store = SessionStore() if SessionStore else None
        self.session_id = None
        initial = None
        _http = self._is_http()
        if self._store and not _http:
            try:
                self.session_id = self._store.get_active_id("chat")
                if self.session_id is None:
                    self.session_id = self._store.create_session("chat")
                initial = self._store.get_mode("chat")
            except Exception as e:
                log.error("could not load chat history: %s: %s", type(e).__name__, e)
                initial = None
        self.session: ChatSession = llm.create_chat(
            system_provider=provider,
            initial_messages=initial,
        )
        self.turns: int = len(self.session.get_messages()) // 2

    def _is_http(self) -> bool:
        from core.http_llm import HTTPLLM
        return isinstance(self.llm, HTTPLLM)

    def handle(self, user_input: str) -> str:
        """Process one user turn; returns assistant reply."""
        reply = self.session.send(user_input)
        self.turns += 1
        if self._store and not self._is_http():
            try:
                self._store.save_mode("chat", self.session.get_messages())
            except Exception as e:
                # The reply is already on screen — losing the save silently is
                # how "my history disappeared" happens. Record it.
                log.error("could not save chat turn (session %s): %s: %s",
                          self.session_id, type(e).__name__, e)
        return reply

    def _system_provider(self):
        """Chat persona + memory hint — single source of truth for
        every session lifecycle (Fix 4.2i-A).

        __init__ had the correct provider; reload_session,
        compress_session and reset each built a DIFFERENT provider
        that returned only the memory hint (or None), dropping
        CHAT_SYSTEM. After any session switch, /summary, /clear or
        /new the model lost the Burmese persona and its language
        rules. Returning the same lazy closure here restores them.
        """
        if self.memory is None:
            return None
        memory = self.memory

        def provider():
            hint = memory.get_system_hint() or ''
            return f"{CHAT_SYSTEM}\n\n{hint}" if hint else CHAT_SYSTEM
        return provider

    def reload_session(self, sid: str) -> bool:
        """Switch to a different session by id."""
        if not self._store:
            return False
        try:
            if not self._store.set_active("chat", sid):
                return False
            s = self._store.get_session(sid) or {}
            msgs = list(s.get("messages", []))
            self.session.clear()
            provider = self._system_provider()
            self.session = self.llm.create_chat(
                system_provider=provider,
                initial_messages=msgs,
            )
            self.session_id = sid
            self.turns = len(msgs) // 2
            return True
        except Exception as e:
            log.error("could not switch to session %s: %s: %s",
                      sid, type(e).__name__, e)
            return False

    def get_messages(self) -> list:
        try:
            return self.session.get_messages()
        except Exception as e:
            log.warning("could not read chat messages: %s: %s", type(e).__name__, e)
            return []

    def compress_session(self) -> dict:
        """Summarize session, archive old, replace with summary."""
        if self._store is None or self.session_id is None:
            return {"error": "no session store / id"}

        msgs = self.session.get_messages()
        if len(msgs) < 4:
            return {"error": f"too few messages ({len(msgs)}, need 4+)"}

        before = self.session.token_count()

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
            archived = self._store.archive_mode("chat")
        except Exception as e:
            return {"error": f"archive failed: {e}"}

        new_msgs = [
            {"role": "user", "text": "[Earlier conversation — summary]"},
            {"role": "assistant", "text": summary},
        ]
        try:
            self._store.save_mode("chat", new_msgs)
        except Exception as e:
            log.error("could not save summarized chat: %s: %s", type(e).__name__, e)

        provider = self._system_provider()
        self.session.clear()
        self.session = self.llm.create_chat(
            system_provider=provider,
            initial_messages=new_msgs,
        )
        self.session_id = self._store.get_active_id("chat")
        self.turns = 1

        after = self.session.token_count()
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
        """Clear conversation history; refresh memory binding."""
        self.session.clear()
        self.turns = 0
        if self._store:
            try:
                self._store.clear_mode("chat")
                self.session_id = self._store.get_active_id("chat")
            except Exception as e:
                log.error("could not clear chat session: %s: %s", type(e).__name__, e)
        provider = self._system_provider()
        self.session = self.llm.create_chat(system_provider=provider)

    def close(self) -> None:
        self.session.close()
