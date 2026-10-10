"""Tokenizer for BM25 memory search. Pure Python — no deps.

Burmese is a space-less script, so word-splitting does not work there.
Myanmar characters (U+1000–U+109F) are tokenized as character n-grams
of length 2–4 — a query and a document that share any 2+ character
sequence will match. Latin/digit runs are split into lowercase word
unigrams. Mixed text (common in memory: Burmese sentences with English
tool names) works because each script is tokenized independently.
"""
from __future__ import annotations

NGRAM_SIZES = (2, 3, 4)

_MY_LO = 0x1000
_MY_HI = 0x109F


def _is_myanmar(ch: str) -> bool:
    return _MY_LO <= ord(ch) <= _MY_HI


def tokenize(text: str) -> list[str]:
    """Split ``text`` into BM25 tokens (lowercase).

    Myanmar runs -> char n-grams (2,3,4); single isolated Myanmar
    characters are kept as unigrams so they are not lost. Latin/digit
    runs -> word unigrams. Everything else (punctuation, spaces) is a
    separator.
    """
    if not text:
        return []
    s = text.lower()
    tokens: list[str] = []
    word: list[str] = []
    myan_run: list[str] = []

    def flush_word() -> None:
        if word:
            tokens.append("".join(word))
            word.clear()

    def flush_myan() -> None:
        if not myan_run:
            return
        if len(myan_run) == 1:
            tokens.append(myan_run[0])
        else:
            joined = "".join(myan_run)
            for n in NGRAM_SIZES:
                if n > len(joined):
                    break
                for i in range(len(joined) - n + 1):
                    tokens.append(joined[i : i + n])
        myan_run.clear()

    for ch in s:
        if _is_myanmar(ch):
            flush_word()
            myan_run.append(ch)
        elif ch.isalnum():
            flush_myan()
            word.append(ch)
        else:
            flush_word()
            flush_myan()
    flush_word()
    flush_myan()
    return tokens
