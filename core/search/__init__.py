"""Smart search subsystem — code-deterministic pipeline + 1 LLM synthesis.

Architecture:
  L1 classifier  → category (code)
  L2 translator  → English (code + LLM fallback)
  L3 router      → backends (code)
  L4 fetch       → results (code)
  L5 cleaner     → deduped + ranked (code)
  L6 packer      → formatted <context> (code)
  L7 LLM         → synthesis (4B model)
"""
from core.search.classifier import classify, Classification, is_burmese
from core.search.translator import translate
from core.search.base import Result, BaseBackend
from core.search.router import route
from core.search.cleaner import clean
from core.search.packer import pack, sources_line

__all__ = [
    "classify", "Classification", "is_burmese",
    "translate",
    "Result", "BaseBackend",
    "route", "clean", "pack", "sources_line",
]
