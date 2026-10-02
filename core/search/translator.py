"""L2 — Burmese → English translation (dict-based + optional LLM fallback)."""
from __future__ import annotations

import re

from core.log import get_logger

log = get_logger("search.translator")

_BURMESE_RE = re.compile(r"[\u1000-\u109F]")


def is_burmese(text: str) -> bool:
    return bool(_BURMESE_RE.search(text or ""))


# ── High-frequency Burmese → English terms ──
# NOTE: Conservative list — only include terms we're confident about.
_TERM_MAP = {
    "မြန်မာနိုင်ငံ": "Myanmar",
    "မြန်မာသမိုင်း": "Myanmar history",
    "မြန်မာ": "Myanmar",
    "သမိုင်း": "history",
    "လူဦးရေ": "population",
    "နိုင်ငံရေး": "politics",
    "စီးပွားရေး": "economy",
    "ပညာရေး": "education",
    "ကျန်းမာရေး": "health",
    "နည်းပညာ": "technology",
    "သတင်း": "news",
    "စာအုပ်": "book",
    "ပရိုဂရမ်": "programming",
    "ပရိုဂရမ်မင်း": "programming",
    "ကုဒ်": "code",
    "ဖုန်း": "phone",
    "ကွန်ပျူတာ": "computer",
    "အင်တာနက်": "internet",
    "ဘာလဲ": "what is",
    "ဘာလဲဆိုတာ": "what is",
    "ဘယ်သူလဲ": "who is",
    "ဘယ်မှာလဲ": "where is",
    "ဘယ်တော့လဲ": "when",
    "ဘာလို့": "why",
    "ဘာကြောင့်": "why",
    "ဘယ်လို": "how",
    "ရှာပြ": "search",
    "ရှာပေး": "find",
    "ပြောပြ": "tell me about",
    "ရှင်းပြ": "explain",
    "အကြောင်း": "about",
    "ဆိုတာ": "",
    "ရဲ့": "",
    "ပါ": "",
}


def translate_burmese_to_english(text: str) -> str:
    """Rule-based partial translation. Strips leftover Burmese chars."""
    if not text:
        return ""
    out = text
    # Longer keys first (avoid partial overlap)
    for burmese in sorted(_TERM_MAP, key=len, reverse=True):
        out = out.replace(burmese, " " + _TERM_MAP[burmese] + " ")
    # Strip leftover Burmese chars
    out = _BURMESE_RE.sub(" ", out)
    # Collapse spaces + trim
    out = re.sub(r"\s+", " ", out).strip()
    # Remove leading/trailing punctuation
    out = out.strip(".,;:!?-")
    return out


def translate(text: str, llm=None) -> str:
    """Translate Burmese → English.

    Strategy:
      1. Not Burmese → return as-is
      2. Dict-based partial translation
      3. If result is mostly English → return
      4. Optional LLM fallback (llm.ask)
      5. Best-effort return
    """
    if not text:
        return ""
    text = text.strip()
    if not is_burmese(text):
        return text

    # Step 1: dict-based
    partial = translate_burmese_to_english(text)

    # Step 2: is it good enough?
    #   - Not Burmese anymore
    #   - At least 2 words
    #   - Preserved at least 30% of original chars (loss-guard)
    if (not is_burmese(partial)
            and len(partial.split()) >= 2
            and len(partial) >= len(text) * 0.3):
        return partial

    # Step 3: LLM fallback (optional)
    if llm is not None:
        try:
            prompt = (
                "Translate the following Burmese text into a short English "
                "search query (2 to 6 words). Output ONLY the English query, "
                "no explanation, no quotes:\n\n" + text
            )
            result = llm.ask(prompt)
            if result:
                result = result.strip().split("\n")[0].strip()
                result = result.strip("\"'`")
                if result and not is_burmese(result) and len(result) < 200:
                    return result
        except Exception as e:
            log.debug("LLM translation fallback failed: %s: %s",
                      type(e).__name__, e)

    # Step 4: best-effort
    return partial or text


__all__ = ["is_burmese", "translate", "translate_burmese_to_english"]
