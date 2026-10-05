"""LiteRT-LM engine wrapper with lazy loading + multi-turn chat."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import litert_lm
from config import Config, model_missing_message
from core.log import get_logger
from litert_lm.interfaces import (
    SamplerConfig,
    RepetitionPenaltyConfig,
    NoRepeatNgramConfig,
)

log = get_logger("llm")

# Silence native LiteRT logs (only ERROR+ shown)
try:
    litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
except Exception as e:
    log.debug("could not lower the LiteRT log level: %s: %s", type(e).__name__, e)


def _repetition_kwargs():
    """Build repetition guard kwargs for send_message calls."""
    return {
        "repetition_penalty_config": RepetitionPenaltyConfig(
            repetition_penalty=1.1,
            frequency_penalty=0.05,
        ),
        "no_repeat_ngram_config": NoRepeatNgramConfig(
            no_repeat_ngram_size=4,
        ),
    }


def estimate_tokens(text: str) -> int:
    """Rough token estimate for mixed Burmese/English text."""
    if not text:
        return 0
    myanmar = sum(1 for c in text if "\u1000" <= c <= "\u109F")
    other = len(text) - myanmar
    # Burmese: ~1.5 chars/token; ASCII/English: ~4 chars/token
    return max(1, int(myanmar / 1.5 + other / 4))


def estimate_messages_tokens(messages) -> int:
    """Estimate total tokens for a message list."""
    total = 0
    for m in messages:
        total += estimate_tokens(m.get("text", ""))
        total += 4  # role + formatting overhead
    return total


def stream_conversation(conv, message):
    """Yield streaming events from conv.send_message_async(message).

    Events:
      ('text', str)        - incremental text chunk
      ('tool_calls', list) - tool calls (stream ends here)
      ('end', {'text': str, 'tool_calls': list|None})
    """
    try:
        iterator = conv.send_message_async(
            message,
            **_repetition_kwargs(),
        )
    except Exception as e:
        yield ('error', f'{type(e).__name__}: {e}')
        return

    full_text = ""
    tool_calls = None
    try:
        for chunk in iterator:
            if not isinstance(chunk, dict):
                continue
            tc = chunk.get('tool_calls')
            if tc:
                tool_calls = tc
                yield ('tool_calls', tc)
                continue
            content = chunk.get('content')
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get('type') == 'text':
                        txt = part.get('text', '')
                        if txt:
                            full_text += txt
                            yield ('text', txt)
    except Exception as e:
        yield ('error', f'{type(e).__name__}: {e}')
        return

    yield ('end', {'text': full_text, 'tool_calls': tool_calls})


# ---------- Vision (image input) ----------

# Successful probes are cached per model file — Capabilities() opens the
# model and parses its metadata, which is expensive for multi-GB files.
_VISION_CACHE: dict[str, tuple[bool, str]] = {}

# Shown to the user whenever image input is unavailable: what failed, plus a
# workable substitute path (OCR → paste the text as a normal message).
VISION_OCR_SUGGESTION = (
    "OCR အစားထိုးနည်း: ပုံထဲက စာသားကို OCR tool (Google Lens / Text Scanner / "
    "tesseract-ocr) ဖြင့် ဖတ်ပြီး ရလဒ် စာသားကို ကူးထည့်ပြီး ပြန်မေးပါ — "
    "ပုံ upload မလိုဘဲ Chat မှာ တန်းလို့ရသည်။"
)


def clear_vision_cache() -> None:
    """Forget cached capability probes (tests / model swap)."""
    _VISION_CACHE.clear()


def supports_vision(model_path: str) -> tuple[bool, str]:
    """Probe whether the model accepts image input.

    Returns ``(supported, reason)``. ``reason`` is "" when supported, otherwise
    it says *why* the probe failed (the route appends VISION_OCR_SUGGESTION).

    Litert-LM is imported lazily and every attribute is getattr-guarded so the
    test stub (which has no ``Capabilities``) and remote HTTP backends degrade
    to "unsupported" instead of raising.
    """
    if not model_path:
        return False, "no model file configured"
    if not Path(model_path).is_file():
        return False, f"model file not found: {model_path}"

    cached = _VISION_CACHE.get(str(model_path))
    if cached is not None:
        return cached

    caps_cls = getattr(litert_lm, "Capabilities", None)
    if caps_cls is None:
        result = (False, "litert_lm.Capabilities unavailable in this build")
    else:
        caps = None
        try:
            caps = caps_cls(model_path)
            vision = bool(caps.input_modalities.vision)
            if vision:
                result = (True, "")
            else:
                result = (False, "model reports vision=False (text-only model)")
        except Exception as e:
            result = (False, f"capability probe failed: {type(e).__name__}: {e}")
        finally:
            close = getattr(caps, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    if result[0]:
        _VISION_CACHE[str(model_path)] = result
    return result


def build_vision_message(model_path: str, image_path: str, text: str):
    """Build a litert_lm ``Contents`` message carrying an image + text.

    Returns ``None`` when image input is unsupported or the Litert-LM message
    types are missing — callers must check :func:`supports_vision` first and
    report its reason to the user.
    """
    ok, _reason = supports_vision(model_path)
    if not ok:
        return None
    content_cls = getattr(litert_lm, "Content", None)
    contents_cls = getattr(litert_lm, "Contents", None)
    if content_cls is None or contents_cls is None:
        return None
    parts = [content_cls.ImageFile(absolute_path=image_path)]
    if text:
        parts.append(content_cls.Text(text))
    return contents_cls.of(*parts)


class LLMError(Exception):
    """LLM-specific errors."""


class ChatSession:
    """Multi-turn conversation session (persistent state)."""

    def __init__(self, llm: "LLM", system_provider=None, initial_messages=None):
        self.llm = llm
        self._conv = None  # litert_lm conversation object
        self._system_provider = system_provider
        self.messages: list = list(initial_messages or [])

    def _ensure_conv(self) -> None:
        if self._conv is None:
            self.llm.load()
            assert self.llm._engine is not None
            system = None
            if self._system_provider is not None:
                try:
                    system = self._system_provider()
                except Exception:
                    system = None
            replay = None
            if self.messages:
                replay = []
                for m in self.messages:
                    role = m.get("role", "user")
                    content = m.get("text", "")
                    if content:
                        replay.append({"role": role, "content": content})
                if not replay:
                    replay = None
            self._conv = self.llm._engine.create_conversation(
                messages=replay,
                system_message=system or None,
                sampler_config=SamplerConfig(
                    temperature=0.2,
                    top_p=0.95,
                    top_k=10,
                ),
            )

    def send(self, message: str) -> str:
        """Send message; returns assistant reply. Keeps history."""
        if not message or not message.strip():
            raise ValueError("message must be non-empty")
        self._ensure_conv()
        assert self._conv is not None
        resp = self._conv.send_message(
            message,
            **_repetition_kwargs(),
        )
        try:
            text = resp["content"][0]["text"]
        except (KeyError, IndexError, TypeError):
            text = str(resp)
        self.messages.append({"role": "user", "text": message})
        self.messages.append({"role": "assistant", "text": text})
        return text

    def send_stream(self, message: str):
        """Streaming variant of send(). Yields ('text', str) then ('end', str)."""
        if not message or not message.strip():
            raise ValueError("message must be non-empty")
        self._ensure_conv()
        assert self._conv is not None

        full = ""
        for kind, data in stream_conversation(self._conv, message):
            if kind == 'text':
                full += data
                yield ('text', data)
            elif kind == 'error':
                yield ('error', data)
                return
            elif kind == 'end':
                text = data.get('text', '') or full
                self.messages.append({"role": "user", "text": message})
                self.messages.append({"role": "assistant", "text": text})
                yield ('end', text)
                return
            elif kind == 'tool_calls':
                yield ('tool_calls', data)

    def get_messages(self) -> list:
        return list(self.messages)

    def token_count(self) -> int:
        """Estimated token usage of this session."""
        return estimate_messages_tokens(self.messages)

    def clear(self) -> None:
        """Reset conversation (drops history)."""
        if self._conv is not None:
            try:
                self._conv.__exit__(None, None, None)
            except Exception as e:
                log.debug("could not close conv on clear: %s: %s", type(e).__name__, e)
            self._conv = None
        self.messages = []

    def close(self) -> None:
        self.clear()

    def __enter__(self) -> "ChatSession":
        self._ensure_conv()
        return self

    def __exit__(self, *exc) -> None:
        self.clear()


class LLM:
    """Lazy-loading wrapper around LiteRT-LM Engine."""

    def __init__(self, config: Config):
        self.config = config
        self._engine: Optional[litert_lm.Engine] = None
        self._backend_used: Optional[str] = None
        self._load_time: float = 0.0
        self._model_path: str = ""

    # ---------- Lifecycle ----------

    def load(self) -> None:
        """Load engine (idempotent). Uses backend from config."""
        if self._engine is not None:
            return

        model_path = self.config.resolve_model_path()
        if not model_path or not Path(model_path).is_file():
            raise LLMError(model_missing_message(model_path))
        self._model_path = model_path

        backend_name = self.config.backend
        try:
            backend_obj = getattr(litert_lm.Backend, backend_name)()
        except AttributeError as e:
            raise LLMError(f"Unknown backend '{backend_name}': {e}")

        # Images need a vision executor, and litert_lm only creates one when a
        # vision_backend is passed — otherwise sending a Contents with an
        # ImageFile raises "Vision executor should not be null, please
        # TryLoadingVisionExecutor() first." CPU is used for the vision encoder
        # so it works whatever backend runs the language model.
        vision_backend = None
        vision_ok, vision_reason = supports_vision(model_path)
        if vision_ok:
            vision_backend = litert_lm.Backend.CPU()
        else:
            log.debug("vision executor skipped for %s: %s", model_path,
                      vision_reason)

        t0 = time.time()
        try:
            self._engine = litert_lm.Engine(
                model_path,
                backend=backend_obj,
                max_num_tokens=self.config.context_size,
                vision_backend=vision_backend,
            )
        except Exception as e:
            if backend_name == "GPU":
                # GPU init can fail on some builds (memory, unsupported op);
                # retry once on CPU before giving up.
                try:
                    self._engine = litert_lm.Engine(
                        model_path,
                        backend=litert_lm.Backend.CPU(),
                        max_num_tokens=self.config.context_size,
                        vision_backend=vision_backend,
                    )
                    backend_name = "CPU"
                except Exception as cpu_err:
                    raise LLMError(
                        f"Engine load failed on GPU ({e}) and on CPU "
                        f"({cpu_err}) for {model_path}"
                    )
            else:
                raise LLMError(f"Engine load failed: {e}")

        self._backend_used = backend_name
        self._load_time = time.time() - t0

    def unload(self) -> None:
        """Free engine memory."""
        if self._engine is not None:
            try:
                self._engine.__exit__(None, None, None)
            except Exception as e:
                log.warning("could not release the LiteRT engine: %s: %s",
                            type(e).__name__, e)
            self._engine = None
            self._backend_used = None
            self._load_time = 0.0

    # ---------- Inference ----------

    def ask(self, prompt: str) -> str:
        """One-shot prompt (no history)."""
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be non-empty")
        self.load()
        assert self._engine is not None
        with self._engine.create_conversation(
            sampler_config=SamplerConfig(
                temperature=0.2,
                top_p=0.95,
                top_k=10,
            ),
        ) as conv:
            resp = conv.send_message(
                prompt,
                **_repetition_kwargs(),
            )
        try:
            return resp["content"][0]["text"]
        except (KeyError, IndexError, TypeError):
            return str(resp)

    def context_usage(self, messages) -> int:
        """Estimate total tokens of message list (helper)."""
        return estimate_messages_tokens(messages)

    def create_chat(self, system_provider=None, initial_messages=None) -> ChatSession:
        """Create a new multi-turn chat session."""
        return ChatSession(
            self,
            system_provider=system_provider,
            initial_messages=initial_messages,
        )

    # ---------- Properties ----------

    @property
    def is_loaded(self) -> bool:
        return self._engine is not None

    @property
    def model_path(self) -> str:
        """Model file the engine was built from (resolved, not configured)."""
        return self._model_path or self.config.resolve_model_path()

    @property
    def backend_used(self) -> Optional[str]:
        return self._backend_used

    @property
    def load_time(self) -> float:
        return self._load_time

    @property
    def context_size(self) -> int:
        return self.config.context_size
