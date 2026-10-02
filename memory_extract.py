"""Heuristic extractor for profile fields from user messages.

No LLM calls — pure regex/pattern matching.
"""
from __future__ import annotations

import re


# ---- Sensitive keywords (skip extraction if present) ----
SENSITIVE = [
    "password", "passwd", "pwd", "pin", "credit card",
    "otp", "cvv", "bank account", "ssn",
    "စကားဝှက်", "လျှို့ဝှက်နံပါတ်",
]


def _normalize(text: str) -> str:
    """Remove zero-width chars, collapse whitespace."""
    # ZWSP + ZWJ + ZWNJ
    text = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _has_sensitive(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in SENSITIVE)


# Words that indicate a question / non-fact (skip extraction)
QUESTION_MARKERS = [
    "ဘယ်သူ", "ဘာလဲ", "ဘာတွေ", "ဘယ်မှာ", "ဘယ်တော့", "ဘယ်လို",
    "လား", "လဲ", "သလား", "ပါသလား",
    "who", "what", "where", "when", "how", "why", "?",
]


def _is_question(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in QUESTION_MARKERS)


# ---- Pattern definitions ----
# Each entry: (field_name, [regex patterns])
PATTERNS = [
    ("ai_name", [
        r"မင်းကို\s*(.{1,30}?)\s*လို့\s*ခေါ်",
        r"နင်ကို\s*(.{1,30}?)\s*လို့\s*ခေါ်",
        r"your name is\s+(.{1,30}?)(?:[\.\,]|$)",
        r"i'?ll call you\s+(.{1,30}?)(?:[\.\,]|$)",
        r"call you\s+(.{1,30}?)(?:[\.\,]|$)",
    ]),
    ("user_name", [
        r"ငါ့ကို\s*(.{1,30}?)\s*လို့\s*ခေါ်",
        r"ကျွန်တော့်ကို\s*(.{1,30}?)\s*လို့\s*ခေါ်",
        r"ကျွန်မကို\s*(.{1,30}?)\s*လို့\s*ခေါ်",
        r"my name is\s+(.{1,30}?)(?:[\.\,]|$)",
        r"call me\s+(.{1,30}?)(?:[\.\,]|$)",
    ]),
    ("relationship", [
        r"မင်းနဲ့ငါ\s*(.{1,30}?)\s*လိုပဲ",
        r"နင်နဲ့ငါ\s*(.{1,30}?)\s*လိုပဲ",
        r"ငါတို့\s*(.{1,30}?)\s*လိုပဲ",
        r"we'?re like\s+(.{1,30}?)(?:[\.\,]|$)",
        r"we are like\s+(.{1,30}?)(?:[\.\,]|$)",
    ]),
    ("tone", [
        r"(.{1,30}?)\s+စွာ\s+ဆက်ဆံ",
        r"(.{1,30}?)\s*လိုပဲ\s*ဆက်ဆံ",
        r"(.{1,30}?)\s+ဆက်ဆံပါ",
        r"talk to me like\s+(.{1,30}?)(?:[\.\,]|$)",
        r"speak\s+(.{1,30}?)\s+to me",
    ]),
    ("language", [
        (r"မြန်မာလိုပဲ", "Burmese"),
        (r"အင်္ဂလိပ်လိုပဲ", "English"),
        (r"speak in\s+(burmese|english|myanmar)",
         None),  # dynamic value below
    ]),
]


# Special values (fixed-string mappings)
LANGUAGE_MAP = {
    "မြန်မာလိုပဲ": "Burmese",
    "မြန်မာလို": "Burmese",
    "အင်္ဂလိပ်လိုပဲ": "English",
    "အင်္ဂလိပ်လို": "English",
}


class Extractor:
    """Extract profile fields from user message (heuristic)."""

    def extract(self, text: str) -> dict:
        """Return {'profile': {field: value, ...}}.

        Empty dict if nothing extracted or sensitive.
        """
        if not text:
            return {"profile": {}}

        norm = _normalize(text)

        if _has_sensitive(norm):
            return {"profile": {}}

        if _is_question(norm):
            return {"profile": {}}

        result: dict[str, str] = {}

        # ai_name, user_name, relationship, tone (regex with capture)
        for field, pats in PATTERNS[:4]:
            for pat in pats:
                m = re.search(pat, norm)
                if m:
                    val = m.group(1).strip()
                    # Filter noise
                    if 1 <= len(val) <= 30 and val not in ("ပါ", "ဘူး", "တယ်"):
                        result[field] = val
                        break

        # Cleanup tone suffix "စွာ" (cosmetic)
        if "tone" in result:
            result["tone"] = result["tone"].replace("စွာ", "").strip()

        # language (fixed map)
        for kw, lang in LANGUAGE_MAP.items():
            if kw in norm:
                result["language"] = lang
                break

        # English "speak in X"
        m = re.search(r"speak in\s+(burmese|english|myanmar)", norm.lower())
        if m:
            result["language"] = m.group(1).capitalize()

        return {"profile": result}


# Module-level singleton
_default = Extractor()


def extract(text: str) -> dict:
    """Convenience function."""
    return _default.extract(text)
