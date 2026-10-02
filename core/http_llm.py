"""HTTP-based LLM proxy for Terminal mode.

Terminal must NOT load its own LiteRT engine (GPU/OpenCL conflict with
server). All LLM calls proxy via HTTP to the local server, which owns
the single engine instance.

Interface parity with core.llm.LLM (drop-in for App).
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from core.auth import configured_token
from core.log import get_logger

log = get_logger("http_llm")

DEFAULT_SERVER = "http://127.0.0.1:8080"
HEALTH_TIMEOUT = 3.0
CHAT_TIMEOUT = 300.0


class HTTPLLMError(Exception):
    """HTTP-LLM specific errors."""


def _auth_headers() -> dict:
    """Auth enabled ဖြစ်နေရင် Bearer token ထည့် (Q3: /v1 အပါအဝင် အားလုံး)။"""
    tok = configured_token()
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def _post(url: str, payload: dict, timeout: float) -> dict:
    """POST JSON; return parsed dict. Raises HTTPLLMError on failure."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **_auth_headers()},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8")
        except Exception as ex:
            log.debug("could not read the HTTP error body: %s: %s",
                      type(ex).__name__, ex)
        raise HTTPLLMError(f"HTTP {e.code}: {detail or e.reason}") from e
    except urllib.error.URLError as e:
        raise HTTPLLMError(f"Server unreachable: {e.reason}") from e
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPLLMError(f"Invalid JSON: {e}") from e


class HTTPLLM:
    """Terminal-side LLM proxy — no local engine."""

    def __init__(self, config, server_url: str = DEFAULT_SERVER):
        self.config = config
        self.server_url = server_url.rstrip("/")

    # ---------- Lifecycle (no-ops) ----------

    def load(self) -> None:
        return None

    def unload(self) -> None:
        return None

    def health_check(self) -> bool:
        try:
            with urllib.request.urlopen(
                f"{self.server_url}/api/health", timeout=HEALTH_TIMEOUT
            ) as r:
                return 200 <= getattr(r, "status", 200) < 300
        except Exception:
            return False

    # ---------- Inference ----------

    def ask(self, prompt: str) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be non-empty")
        j = _post(
            f"{self.server_url}/api/chat",
            {"message": prompt},
            timeout=CHAT_TIMEOUT,
        )
        if "error" in j:
            raise HTTPLLMError(str(j["error"]))
        return j.get("reply", "")

    def create_chat(self, system_provider=None, initial_messages=None):
        return _HTTPChatSession(self, list(initial_messages or []))

    def context_usage(self, messages) -> int:
        from core.llm import estimate_messages_tokens
        return estimate_messages_tokens(messages)

    # ---------- Properties ----------

    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def backend_used(self) -> str:
        return "HTTP"

    @property
    def load_time(self) -> float:
        return 0.0

    @property
    def context_size(self) -> int:
        return self.config.context_size


class _HTTPChatSession:
    """Chat session proxy — one HTTP call per turn."""

    def __init__(self, llm: HTTPLLM, initial_messages: list):
        self.llm = llm
        self.messages = list(initial_messages)

    def send(self, message: str) -> str:
        if not message or not message.strip():
            raise ValueError("message must be non-empty")
        j = _post(
            f"{self.llm.server_url}/api/chat",
            {"message": message},
            timeout=CHAT_TIMEOUT,
        )
        if "error" in j:
            raise HTTPLLMError(str(j["error"]))
        reply = j.get("reply", "")
        self.messages.append({"role": "user", "text": message})
        self.messages.append({"role": "assistant", "text": reply})
        return reply

    def get_messages(self) -> list:
        return list(self.messages)

    def token_count(self) -> int:
        from core.llm import estimate_messages_tokens
        return estimate_messages_tokens(self.messages)

    def clear(self) -> None:
        self.messages = []

    def close(self) -> None:
        self.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None
