"""Tools package for agent mode."""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from tools.file import ALL_TOOLS as FILE_TOOLS
from tools.shell import run_command
from tools.android_intent import ALL_TOOLS as ANDROID_TOOLS
from tools.web_search import ALL_TOOLS as WEB_TOOLS
from tools.smart_search import smart_search

ALL_TOOLS = (
    list(FILE_TOOLS)
    + [run_command]
    + list(ANDROID_TOOLS)
    + list(WEB_TOOLS)
    + [smart_search]
)

__all__ = [
    "ALL_TOOLS", "run_command", "ANDROID_TOOLS", "WEB_TOOLS",
    "smart_search",
]
