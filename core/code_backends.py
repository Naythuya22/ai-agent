"""Registry of Code-mode backends.

Code mode can drive either `opencode` or `openclaude`. Rather than branching
on a name throughout routes/code.py, every consumer resolves the name once
(`_backend()` in the routes) and then calls the same function names on the
module it gets back — both core/opencode.py and core/openclaude.py expose
is_installed(), version(), env(), build_command(), doctor() and the session
map.

The registry is name-keyed and import-lazy: `names()` is the contract the UI
and the tests rely on, and importing only the module asked for keeps a
backend that is not in play out of the process.

    from core import code_backends as b
    b.names()          # ['opencode', 'openclaude']
    b.get('openclaude')  # the core.openclaude module (None if unknown)
    b.available()      # the subset whose binary is on PATH right now
"""
from __future__ import annotations

from core.log import get_logger

log = get_logger("code_backends")

# Registration order is the display order: opencode stays first so every
# existing caller that assumed "the" backend still lands on it.
_ORDER = ("opencode", "openclaude")


def names() -> list[str]:
    """Every registered backend, in registration order."""
    return list(_ORDER)


def get(name) -> object | None:
    """The backend module for `name`, or None when it is not registered.

    Deliberately does *not* fall back to the default: a caller passing a
    typo must be able to tell "unknown backend" from "the default one".
    """
    key = str(name or "").strip()
    if key == "opencode":
        from core import opencode
        return opencode
    if key == "openclaude":
        from core import openclaude
        return openclaude
    return None


def default_name() -> str:
    """Configured default (AI_AGENT_CODE_BACKEND), or 'opencode'.

    An env var naming an unregistered backend must not take the whole mode
    down, so it degrades to the default instead of failing later with a
    ModuleNotFoundError.
    """
    import config

    name = str(getattr(config, "CODE_BACKEND_DEFAULT", "") or "").strip()
    return name if name in _ORDER else "opencode"


def resolve(name) -> str:
    """A registered name: `name` if it is one, else the configured default."""
    key = str(name or "").strip()
    return key if key in _ORDER else default_name()


def installed(name) -> bool:
    """Is this backend's binary present and callable right now?"""
    mod = get(name)
    if mod is None:
        return False
    try:
        return bool(mod.is_installed())
    except Exception as e:
        # A probe that throws (missing proot, PATH oddity) is the same as "not
        # installed" to a caller deciding what to offer the user.
        log.debug("is_installed(%s) failed: %s: %s",
                  name, type(e).__name__, e)
        return False


def available() -> list[str]:
    """The registered backends whose binary is on PATH right now."""
    return [n for n in _ORDER if installed(n)]


def availability() -> dict:
    """name -> bool, for endpoints that report per-backend health."""
    return {n: installed(n) for n in _ORDER}
