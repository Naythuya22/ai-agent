"""Tools package for agent mode."""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from tools.file import ALL_TOOLS as FILE_TOOLS
from tools.shell import run_command
from tools.python_runner import run_python
from tools.android_intent import ALL_TOOLS as ANDROID_TOOLS
from tools.web_search import ALL_TOOLS as WEB_TOOLS
from tools.smart_search import smart_search
from tools.memory import (
    memory_read, memory_write, memory_search, memory_forget,
)
from tools.audit import self_audit, capability_list

ALL_TOOLS = (
    list(FILE_TOOLS)
    + [run_command, run_python]
    + list(ANDROID_TOOLS)
    + list(WEB_TOOLS)
    + [smart_search]
    + [memory_read, memory_write, memory_search, memory_forget]
    + [self_audit, capability_list]
)

__all__ = [
    "ALL_TOOLS", "run_command", "run_python", "ANDROID_TOOLS", "WEB_TOOLS",
    "smart_search",
    "memory_read", "memory_write", "memory_search", "memory_forget",
    "self_audit", "capability_list",
]
