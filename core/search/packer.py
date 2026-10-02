"""L6 — Pack cleaned results into <context> for LLM consumption."""
from __future__ import annotations

from typing import List

from core.search.base import Result

_MAX_TOTAL_CHARS = 3200
_PER_SNIPPET = 500


def pack(results: List[Result], query: str, max_chars: int = _MAX_TOTAL_CHARS) -> str:
    """Pack results into a compact <context> block.

    Format:
        <context>
        Source: Wikipedia (EN) | Type: knowledge
        URL: https://...
        Title: Myanmar
        Content: Myanmar is a country in...

        Source: Stack Overflow | Type: coding
        ...
        </context>
    """
    if not results:
        return f"<context>\nNo results found for: {query}\n</context>"

    parts: List[str] = ["<context>"]
    total = 0

    for i, r in enumerate(results, 1):
        block = []
        src = r.source or "Unknown"
        cat = r.category or "general"
        block.append(f"Source: {src} | Type: {cat}")
        if r.url:
            block.append(f"URL: {r.url}")
        block.append(f"Title: {(r.title or '').strip()}")
        snip = (r.snippet or "").strip()
        if len(snip) > _PER_SNIPPET:
            snip = snip[:_PER_SNIPPET] + "..."
        if snip:
            block.append(f"Content: {snip}")

        text = "\n".join(block)
        if total + len(text) > max_chars:
            break
        parts.append(text)
        total += len(text)
        parts.append("")  # blank line between results

    parts.append("</context>")
    return "\n".join(parts)


def sources_line(results: List[Result]) -> str:
    """Compact source summary: 'Wikipedia (EN), Stack Overflow'."""
    if not results:
        return ""
    seen = []
    for r in results:
        s = r.source or "Unknown"
        if s not in seen:
            seen.append(s)
    return ", ".join(seen[:5])


__all__ = ["pack", "sources_line"]
