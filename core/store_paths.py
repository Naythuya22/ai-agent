"""Where the Code-mode session/config stores live.

The stores sit in /sdcard/.ai-agent on Android so BOTH sides of Code
mode can reach them: the Termux server and the opencode process inside
the proot Ubuntu. That only works where /sdcard can actually be
created — on a desktop or a CI runner it cannot (no permission to
mkdir at /), and every write was silently dropped, leaving the stores
permanently empty and the session-recovery tests red off-device.

So: prefer /sdcard/.ai-agent whenever it exists or can be created
(the phone — behaviour there is byte-for-byte what it always was);
otherwise fall back to ~/.ai-agent, which the backup tool already
covers as the "home" root.
"""
from __future__ import annotations

from pathlib import Path

_SDCARD_DIR = Path("/sdcard/.ai-agent")


def shared_store_dir(sdcard_dir: Path | None = None,
                     home_dir: Path | None = None) -> Path:
    """Return the shared store directory, creating it when possible."""
    sdcard = sdcard_dir if sdcard_dir is not None else _SDCARD_DIR
    if sdcard.is_dir():
        return sdcard
    try:
        sdcard.mkdir(parents=True, exist_ok=True)
        return sdcard
    except OSError:
        pass
    home = home_dir if home_dir is not None else Path.home() / ".ai-agent"
    try:
        home.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return home
