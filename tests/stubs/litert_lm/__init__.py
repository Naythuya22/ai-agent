"""Test stub for the Termux-only litert_lm package."""

import json

# Records what core.llm.LLM.load() asked for, so tests can assert on it.
LOADED = []
# Set to an exception instance to make the first Engine() construction fail.
FAIL_BACKENDS = set()
# Flip to False to make Capabilities().input_modalities.vision say "no image
# input" — exercises the vision-unsupported path end to end.
VISION_SUPPORTED = True
# Every message handed to a stub conversation (str or Contents).
LAST_MESSAGES = []


class LogSeverity:
    ERROR = "ERROR"
    INFO = "INFO"
    DEBUG = "DEBUG"


def set_min_log_severity(level):
    pass


def tool_from_function(fn):
    return {"name": fn.__name__, "fn": fn}


class _Interfaces:
    pass


class Backend:
    def __init__(self, name="CPU"):
        self.name = name

    def __call__(self):
        return self

    def __repr__(self):
        return f"Backend({self.name})"

    CPU = None
    GPU = None


Backend.CPU = Backend("CPU")
Backend.GPU = Backend("GPU")


class _SamplerConfig:
    def __init__(self, **kw):
        self.kw = kw

    def __repr__(self):
        return f"SamplerConfig({self.kw})"


class _Conv:
    def __init__(self, **kw):
        self.kw = kw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @staticmethod
    def _echo(message):
        LAST_MESSAGES.append(message)
        if hasattr(message, "to_json"):
            payload = json.dumps(message.to_json(), ensure_ascii=False)
        else:
            payload = str(message)
        return {"content": [{"type": "text", "text": f"stub-echo: {payload}"}]}

    def send_message(self, message, **kw):
        return self._echo(message)

    def send_message_async(self, message, **kw):
        yield self._echo(message)


class _Modalities:
    def __init__(self, text=True, vision=True, audio=False, video=False):
        self.text = text
        self.vision = vision
        self.audio = audio
        self.video = video


class Capabilities:
    """Mirrors litert_lm.Capabilities(model_path).input_modalities."""

    def __init__(self, model_path):
        self.model_path = str(model_path)
        self._closed = False

    @property
    def input_modalities(self):
        if self._closed:
            raise RuntimeError("stub: capabilities handle closed")
        return _Modalities(
            text=True, vision=VISION_SUPPORTED, audio=True, video=False
        )

    def supports_thinking(self):
        return False

    def close(self):
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class _Part:
    def __init__(self, payload):
        self._payload = payload

    def to_json(self):
        return dict(self._payload)

    def __repr__(self):
        return f"_Part({self._payload!r})"


class Content:
    """Mirrors litert_lm.Content.ImageFile / .Text factories."""

    @staticmethod
    def ImageFile(absolute_path):
        return _Part({"type": "image", "path": str(absolute_path)})

    @staticmethod
    def Text(text):
        return _Part({"type": "text", "text": str(text)})


class Contents:
    def __init__(self, parts):
        self.parts = list(parts)

    @classmethod
    def of(cls, *parts):
        return cls(parts)

    def to_json(self):
        return [p.to_json() for p in self.parts]

    def __repr__(self):
        return f"Contents({self.to_json()!r})"


class Engine:
    def __init__(self, model_path, backend=None, max_num_tokens=None,
                 vision_backend=None, audio_backend=None, **kw):
        backend_name = getattr(backend, "name", str(backend))
        LOADED.append(
            {"model_path": str(model_path), "backend": backend_name,
             "max_num_tokens": max_num_tokens,
             "vision_backend": getattr(vision_backend, "name", vision_backend)}
        )
        if backend_name in FAIL_BACKENDS:
            raise RuntimeError(f"stub: backend {backend_name} unavailable")
        self.model_path = str(model_path)
        self.backend = backend
        self.max_num_tokens = max_num_tokens
        self.vision_backend = vision_backend

    def create_conversation(self, **kw):
        return _Conv(**kw)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
