"""Central logging for ai-agent.

Every module logs through this one place — the tree deliberately has no other
``logging`` import, so destination, format and level stay configurable here.

Output goes to ``<CONFIG_DIR>/logs/agent.log`` (rotating), because the chat
REPL owns stdout/stderr and interleaving stack traces into it would corrupt
the display. Failures that are invisible while the app is running — a session
or memory save that silently failed — are therefore still recoverable after
the fact by reading the file.

Console logging is opt-in with ``AI_AGENT_LOG_CONSOLE=1``. Level is
``AI_AGENT_LOG_LEVEL`` (default INFO).

Setup is lazy and idempotent: it runs on the first log call, not at import
time, so callers that retarget ``config.CONFIG_DIR`` (the test harness) are
honoured, and a failure to create the log directory degrades to console-only
instead of crashing the process.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import threading
from pathlib import Path

LOGGER_NAME = "aiagent"

_lock = threading.RLock()
_configured = False
_logger = logging.getLogger(LOGGER_NAME)

# Fallback destination used before/instead of the rotating file handler.
_CONSOLE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_FILE_FORMAT = "%(asctime)s %(levelname)-7s [%(threadName)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_MAX_BYTES = 512 * 1024
_BACKUP_COUNT = 3


def _log_dir() -> Path | None:
    """Resolved lazily so a retargeted config.CONFIG_DIR is picked up."""
    try:
        import config
        base = Path(config.CONFIG_DIR)
    except Exception:
        base = Path.home() / ".ai-agent"
    return base / "logs"


def _ensure_configured() -> None:
    global _configured
    if _configured:
        return
    with _lock:
        if _configured:
            return
        # Claim before touching the handlers. Resolving the log directory can
        # import `config`, whose own module body calls get_logger() — that
        # nested call must be a no-op rather than a second (or deadlocking)
        # trip through this function.
        _configured = True
        level = getattr(logging, (os.environ.get("AI_AGENT_LOG_LEVEL") or "INFO").upper(),
                        logging.INFO)
        _logger.setLevel(level)
        _logger.propagate = False

        fmt = logging.Formatter(_FILE_FORMAT, datefmt=_DATE_FORMAT)

        # Console first — always safe, even if the log file cannot be created.
        if os.environ.get("AI_AGENT_LOG_CONSOLE"):
            console = logging.StreamHandler(sys.stderr)
            console.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT))
            console.setLevel(level)
            _logger.addHandler(console)

        try:
            log_dir = _log_dir()
            log_dir.mkdir(parents=True, exist_ok=True)
            handler = logging.handlers.RotatingFileHandler(
                log_dir / "agent.log",
                maxBytes=_MAX_BYTES,
                backupCount=_BACKUP_COUNT,
                encoding="utf-8",
            )
            handler.setFormatter(fmt)
            handler.setLevel(level)
            _logger.addHandler(handler)
        except OSError:
            # Read-only storage, bad permissions, full disk — logging must
            # never take the app down with it.
            pass

        if not _logger.handlers:
            _logger.addHandler(logging.NullHandler())


def get_logger(name: str = "") -> logging.Logger:
    """Return a child logger under the single ``aiagent`` namespace."""
    _ensure_configured()
    return _logger.getChild(name) if name else _logger


def log_path() -> Path:
    """Where the log file lives (may not exist yet if nothing was logged)."""
    return _log_dir() / "agent.log"


def tail(n: int = 40) -> list[str]:
    """Last ``n`` log lines, oldest first. Empty list if no log yet.

    Used by the REPL ``/logs`` command — read-only and best effort.
    """
    try:
        path = log_path()
        if not path.is_file():
            return []
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            return fh.readlines()[-n:]
    except OSError:
        return []


def reset_for_tests() -> None:
    """Drop handlers and re-apply them against the current config.CONFIG_DIR.

    Reconfigures immediately rather than leaving the logger bare: records are
    emitted through whatever handlers exist *now*, and a module that already
    imported its logger will never call get_logger() again to trigger setup.
    """
    global _configured
    with _lock:
        for h in list(_logger.handlers):
            try:
                h.close()
            except Exception:
                pass
            _logger.removeHandler(h)
        _configured = False
    _ensure_configured()


__all__ = ["get_logger", "log_path", "tail", "reset_for_tests", "LOGGER_NAME"]
