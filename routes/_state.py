"""Shared server singletons — the lazily-built LLM, memory, and chat/agent modes.

These were module globals in server.py. Blueprints cannot share module
globals, so they live here and everything imports this one module — which
keeps the dependency graph acyclic (`_state` imports only from `config`,
`core`, `memory`, `modes`).
"""
from __future__ import annotations

import sys
import threading
from contextlib import contextmanager
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import Config
from core.llm import LLM
from memory import Memory
from modes.chat import ChatMode
from modes.agent import AgentMode

_llm = None
_memory = None
_chat_mode = None
_agent_mode = None

# Serialises model work: generation, tool loops, engine load/unload/rebuild.
# The LiteRT-LM engine, its KV cache and the cached Conversation objects are
# shared mutable state, and the server runs threaded=True — so without this two
# clients would interleave turns on one conversation. Reentrant because the
# locked paths call each other (get_llm -> ensure_modes -> run_chat).
inference_lock = threading.RLock()

# Guards lazy construction of the singletons. Deliberately separate from
# inference_lock: a second request must wait for construction instead of racing
# it, or two threads would both call LLM.load() and pull the engine in twice.
init_lock = threading.RLock()

# Guards concurrent opencode runs (only used by the code blueprint).
code_lock = threading.Lock()

# Serialises Agent turns that touch AgentMode._messages. Separate from
# inference_lock on purpose: a cloud step blocks on a provider socket for up
# to 30s per attempt, and holding the engine lock for that would stall Chat
# mode even though no engine is involved. Lock order is always
# agent_turn_lock -> inference_lock (the local branch takes both); nothing
# takes it the other way round, so the pair cannot deadlock.
agent_turn_lock = threading.RLock()


@contextmanager
def inference_guard():
    """Hold the inference lock for one unit of model work."""
    with inference_lock:
        yield


@contextmanager
def agent_turn_guard():
    """Hold the agent-turn lock for one full Agent turn (cloud or local)."""
    with agent_turn_lock:
        yield


def get_llm(load: bool = True):
    """Return the process-wide LLM, loading the engine on first use.

    With ``load=False`` the LLM is only constructed: every engine entry
    point (``ask``, ``ChatSession`` sends, the agent local branch) loads
    lazily and raises ``LLMError`` with the missing-model message when no
    model file exists, so construction is safe on a cloud-only device.
    """
    global _llm
    if _llm is None:
        with init_lock:
            if _llm is None:  # re-check: another thread may have won the race
                cfg = Config.load()
                _llm = LLM(cfg)
                if load:
                    _llm.load()
    return _llm


def ensure_modes() -> None:
    """Build memory + chat/agent modes on first use. No-op afterwards.

    The modes are built around an *unloaded* LLM: building them must not
    require a local model, or a device with only cloud keys could not
    serve the UI, Settings or the agent's cloud path at all. The engine
    loads on the first turn that actually needs it.
    """
    global _chat_mode, _agent_mode, _memory
    if _chat_mode is None:
        with init_lock:
            if _chat_mode is None:
                _memory = Memory.load()
                llm = get_llm(load=False)
                _chat_mode = ChatMode(llm, _memory)
                _agent_mode = AgentMode(llm, _memory)


def get_memory():
    ensure_modes()
    return _memory


def get_chat_mode():
    ensure_modes()
    return _chat_mode


def get_agent_mode():
    ensure_modes()
    return _agent_mode


def rebuild_modes() -> None:
    """Reconstruct chat/agent modes around the existing LLM + memory objects.

    Used after a context-size change: the engine is unloaded and the next
    request reloads it at the new size. Session data already lives on disk.
    """
    global _chat_mode, _agent_mode
    with inference_guard():
        with init_lock:
            _chat_mode = ChatMode(_chat_mode.llm, _chat_mode.memory)
            _agent_mode = AgentMode(_agent_mode.llm, _agent_mode.memory)
