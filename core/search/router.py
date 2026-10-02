"""L3 — Route category to ordered backend instances (code, no LLM)."""
from __future__ import annotations

from typing import List

from core.search.backends import (
    WikipediaBackend, StackExchangeBackend, OpenLibraryBackend,
    RSSBackend, DDGBackend,
)


def route(category: str) -> List:
    """Return ordered list of backend instances for a category.

    Rules:
      knowledge → Wikipedia (primary) + DDG (fallback)
      coding    → StackExchange (primary)
      book      → OpenLibrary (primary)
      news      → RSS (news) + RSS (major_myanmar) + RSS (regional)
      general   → Wikipedia + DDG (broad fallback)
    """
    c = (category or "general").strip().lower()

    if c == "knowledge":
        return [WikipediaBackend(), DDGBackend()]

    if c == "coding":
        return [StackExchangeBackend()]

    if c == "book":
        return [OpenLibraryBackend()]

    if c == "news":
        return [
            RSSBackend(category_key="news"),
            RSSBackend(category_key="major_myanmar"),
            RSSBackend(category_key="regional"),
        ]

    if c == "burmese_news":
        return [
            RSSBackend(category_key="news"),
            RSSBackend(category_key="major_myanmar"),
        ]

    # Fallback — try Wikipedia first (broad), then DDG
    return [WikipediaBackend(), DDGBackend()]


__all__ = ["route"]
