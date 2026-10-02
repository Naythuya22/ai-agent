"""L1 — Classify user query into a category using keyword rules (no LLM).

Categories:
  news      — current events, breaking, today
  coding    — programming, code, technical
  knowledge — facts, definitions, history, science
  book      — books, authors, chapters
  general   — fallback
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

# ── Burmese Unicode range ──
_BURMESE_RE = re.compile(r"[\u1000-\u109F]")


def is_burmese(text: str) -> bool:
    """Return True if text contains Burmese characters."""
    return bool(_BURMESE_RE.search(text or ""))


@dataclass
class Classification:
    category: str = "general"
    confidence: float = 0.0
    matched: List[str] = field(default_factory=list)


# ── Keyword rules (Burmese + English) ──
RULES = {
    "news": [
        # English
        "news", "breaking news", "latest news", "today's news",
        "current events", "what happened", "headline",
        # Burmese
        "သတင်း", "သတင်းတွေ", "သတင်းအသစ်", "ဒီနေ့", "မကြာသေးမီ",
        "ဘာဖြစ်လဲ", "ဖြစ်ရပ်",
    ],
    "coding": [
        # English
        "code", "coding", "programming", "python", "javascript",
        "typescript", "java", "rust", "golang", "ruby", "php",
        "function", "syntax", "error", "debug", "api", "library",
        "framework", "asyncio", "react", "vue", "django", "flask",
        "node", "sql", "database", "docker", "git",
        # Burmese
        "ကုဒ်", "ပရိုဂရမ်", "ဖန်ရှင်", "ကွန်ပျူတာ", "နည်းပညာ",
    ],
    "knowledge": [
        # English
        "what is", "what are", "who is", "who was", "who are",
        "history", "explain", "definition", "meaning", "define",
        "why is", "why does", "how does", "how is",
        "tell me about", "difference between",
        # Burmese
        "ဘာလဲ", "ဘာလဲဆိုတာ", "သမိုင်း", "ရှင်းပြ", "အကြောင်း",
        "ပြောပြ", "ဘယ်လို", "ဘာကြောင့်",
        "ဘယ်လောက်လဲ", "ဘယ်နှစ်", "ဘယ်တုန်း", "ဘယ်အချိန်",
        "ဘယ်သူလဲ", "ဘယ်မှာလဲ", "ဘယ်တော့လဲ",
        "ဆိုတာ", "ဆိုတာဘာလဲ", "အနှစ်ချုပ်",
        # English extra
        "tell me", "how many", "how much", "what",
    ],
    "book": [
        # English
        "book", "author", "chapter", "novel", "reading list",
        "summary", "review", "who wrote", "published",
        "best seller", "bestseller", "non-fiction", "fiction",
        "biography", "memoir", "manuscript",
        # Burmese
        "စာအုပ်", "စာရေးဆရာ", "အခန်း", "ဝတ္ထု",
        "စာအုပ်အနှစ်ချုပ်", "စာအုပ်အကြောင်း", "စာအုပ်လေ့လာ",
    ],
}


def _has_keyword(query: str, kw: str) -> bool:
    """Match: word-boundary for ASCII, substring for Burmese."""
    if is_burmese(kw):
        return kw in query
    pattern = r"\b" + re.escape(kw.lower()) + r"\b"
    return bool(re.search(pattern, query.lower()))


def classify(query: str) -> Classification:
    """Classify a query. Deterministic, no LLM."""
    if not query or not query.strip():
        return Classification(category="knowledge", confidence=0.0, matched=[])

    scores = {}
    matched_map = {}
    for cat, keywords in RULES.items():
        matched = [kw for kw in keywords if _has_keyword(query, kw)]
        if matched:
            scores[cat] = len(matched)
            matched_map[cat] = matched

    if not scores:
        # Default to knowledge — safer than DDG-fallback for unknown nouns
        return Classification(category="knowledge", confidence=0.3, matched=[])

    best = max(scores, key=scores.get)
    best_n = scores[best]

    # Confidence: 0.5 base, +0.1 per match, cap 0.95
    confidence = min(0.5 + 0.1 * best_n, 0.95)

    return Classification(
        category=best,
        confidence=round(confidence, 2),
        matched=matched_map[best],
    )


__all__ = ["classify", "Classification", "is_burmese"]
