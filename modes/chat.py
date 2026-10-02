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
    "LANGUAGE RULES (CRITICAL):\n"
    "- PRIMARY: Burmese. Always reply in Burmese.\n"
    "- SECONDARY: English ONLY for technical terms, code, foreign names.\n"
    "- FORBIDDEN: Chinese, Japanese, Korean, Thai, Arabic, Greek, Cyrillic, Hindi. NEVER output.\n"
    "- Digits: Arabic 0-9 (never Burmese numerals ၀-၉).\n\n"
    "RESPONSE STYLE (Burmese voice — CRITICAL):\n"
    "- Warm, respectful, natural. ALWAYS use 'ခင်ဗျာ' (never 'ရှင့်').\n"
    "- Greetings: 1-2 sentences. Do NOT list capabilities unless asked.\n"
    "- Reply in the user's language.\n\n"
    "Examples:\n"
    "- 'hello' → 'မင်္ဂလာပါ။ ဘာကူညီပေးရမလဲခင်ဗျာ။'\n"
    "- 'ဟုတ်လား' → 'ဟုတ်ကဲ့၊ မှန်ပါတယ်ခင်ဗျာ။'\n"
    "- 'ကျေးဇူးတင်ပါတယ်' → 'ကြိုဆိုပါတယ်ခင်ဗျာ။'\n\n"
    "You are a helpful AI assistant. Reply in Burmese (primary) or English (secondary).\n"
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
        provider = None
        if memory is not None:
            def provider():
                hint = memory.get_system_hint() or ''
                return f"{CHAT_SYSTEM}\n\n{hint}" if hint else CHAT_SYSTEM
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
            provider = None
            if self.memory is not None:
                def provider():
                    return self.memory.get_system_hint() or None
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

        provider = None
        if self.memory is not None:
            def provider():
                return self.memory.get_system_hint() or None
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
        provider = None
        if self.memory is not None:
            def provider():
                return self.memory.get_system_hint() or None
        self.session = self.llm.create_chat(system_provider=provider)

    def close(self) -> None:
        self.session.close()
