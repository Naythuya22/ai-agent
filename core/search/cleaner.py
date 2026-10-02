"""L5 — Clean, dedupe, rank results (code, no LLM)."""
from __future__ import annotations

import re
from typing import List

from core.search.base import Result

_MIN_SNIPPET_LEN = 20


def _norm_url(url: str) -> str:
    """Normalize URL for dedup: strip scheme, www., trailing slash, fragment."""
    if not url:
        return ""
    u = url.lower()
    u = re.sub(r"^https?://", "", u)
    u = re.sub(r"^www\.", "", u)
    u = u.split("#", 1)[0]
    u = u.rstrip("/")
    return u


def _is_duplicate(a: Result, b: Result) -> bool:
    """True if two results are likely the same."""
    ua, ub = _norm_url(a.url), _norm_url(b.url)
    if ua and ub and ua == ub:
        return True
    # Title similarity: same first 30 chars lowercase
    ta = (a.title or "").lower()[:30]
    tb = (b.title or "").lower()[:30]
    if ta and tb and ta == tb:
        return True
    return False


def _rank_score(r: Result) -> float:
    """Higher = better. Combines: backend score, snippet quality, url presence."""
    s = r.score or 0.0
    # Snippet quality
    sn = len(r.snippet or "")
    if sn >= 200:
        s += 2.0
    elif sn >= 80:
        s += 1.0
    elif sn < _MIN_SNIPPET_LEN:
        s -= 1.0
    # URL presence
    if r.url:
        s += 0.5
    # Title length penalty (very short titles less useful)
    if len(r.title or "") < 5:
        s -= 0.5
    return s


def clean(results: List[Result], max_keep: int = 8) -> List[Result]:
    """Dedupe, rank, and truncate result list."""
    if not results:
        return []

    # Remove empty / bad
    good = []
    for r in results:
        if not r or not (r.title or "").strip():
            continue
        good.append(r)

    # Dedupe (O(n^2) — small n)
    deduped: List[Result] = []
    for r in good:
        if any(_is_duplicate(r, k) for k in deduped):
            continue
        deduped.append(r)

    # Rank
    deduped.sort(key=_rank_score, reverse=True)

    return deduped[:max_keep]


__all__ = ["clean"]
