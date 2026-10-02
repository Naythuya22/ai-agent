"""Test stub for the Termux-only litert_lm package."""

# Records what core.llm.LLM.load() asked for, so tests can assert on it.
LOADED = []
# Set to an exception instance to make the first Engine() construction fail.
FAIL_BACKENDS = set()


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

    def send_message(self, message, **kw):
        return {"content": [{"type": "text", "text": f"stub-echo: {message}"}]}

    def send_message_async(self, message, **kw):
        yield {"content": [{"type": "text", "text": f"stub-echo: {message}"}]}


class Engine:
    def __init__(self, model_path, backend=None, max_num_tokens=None):
        backend_name = getattr(backend, "name", str(backend))
        LOADED.append(
            {"model_path": str(model_path), "backend": backend_name,
             "max_num_tokens": max_num_tokens}
        )
        if backend_name in FAIL_BACKENDS:
            raise RuntimeError(f"stub: backend {backend_name} unavailable")
        self.model_path = str(model_path)
        self.backend = backend
        self.max_num_tokens = max_num_tokens

    def create_conversation(self, **kw):
        return _Conv(**kw)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
