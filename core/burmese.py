"""Burmese combining-sequence validation.

``filter_scripts`` looks at one character at a time, so a structurally broken
Myanmar sequence — repeated virama, a virama with nothing to stack onto, an
orphaned medial — passes straight through: every character in it is
individually legal. The 4B model produces exactly that garbage under load
(``ထိင်္္္္န္``). This module validates the *sequence* instead and truncates
at the first invalid marker.

Rules (enforced here in code, never left to the prompt):

  B1  the same combining mark never repeats back to back
  B2  a Myanmar combining mark must sit inside a syllable that started with
      a Myanmar base — no orphan marks at the start of a message, after a
      space, after Latin text, …
  B3  U+1039 SIGN VIRAMA must be followed by a letter (a stacked consonant).
      At the end of the text there is nothing left to stack onto → invalid.
  B4  more than MAX_MARKS combining marks in a row cannot be a real Burmese
      syllable (backstop for alternating garbage that B1 does not catch).
      Worst mark run measured on real text is 4, so 6 leaves headroom.

Streaming: chunks split syllables all the time — a chunk can end on a lone
virama, which is only wrong if nothing follows. ``feed`` therefore holds the
trailing mark run back until the next chunk or ``flush()`` settles it, so a
bad chunk boundary never cuts valid text.
"""
from __future__ import annotations

import unicodedata

VIRAMA = 0x1039
BLOCK_LO = 0x1000
BLOCK_HI = 0x109F
MAX_MARKS = 6


def _is_mark(ch: str) -> bool:
    """True for a Myanmar combining mark (vowel sign, medial, virama, …)."""
    return BLOCK_LO <= ord(ch) <= BLOCK_HI and unicodedata.category(ch) in ("Mn", "Mc")


def _is_base(ch: str) -> bool:
    """True for a Myanmar non-combining char (consonant, digit, punctuation)."""
    return BLOCK_LO <= ord(ch) <= BLOCK_HI and not _is_mark(ch)


def _is_letter(ch: str) -> bool:
    """True for a Myanmar letter — the only thing a virama may stack onto."""
    return BLOCK_LO <= ord(ch) <= BLOCK_HI and unicodedata.category(ch) == "Lo"


def scan(text: str, open_syllable: bool = False) -> tuple[int, bool]:
    """Find the first invalid marker in ``text``.

    ``open_syllable`` says whether the character *before* ``text`` left a
    Myanmar syllable open (i.e. the first char of ``text`` may legally be a
    mark). Returns ``(index, state_after_index)``: ``index`` is ``len(text)``
    when everything is valid, and ``state_after_index`` is the value to pass
    as ``open_syllable`` for the next slice.
    """
    run = 0
    prev_mark = None
    for i, ch in enumerate(text):
        if _is_mark(ch):
            if not open_syllable:
                return i, False                      # B2 orphan mark
            if ch == prev_mark:
                return i, True                       # B1 duplicate mark
            if ord(ch) == VIRAMA and i + 1 < len(text) and not _is_letter(text[i + 1]):
                return i, True                       # B3 stack onto non-letter
            run += 1
            if run > MAX_MARKS:
                return i, True                       # B4 runaway mark run
            prev_mark = ch
        else:
            open_syllable = _is_base(ch)
            run = 0
            prev_mark = None
    return len(text), open_syllable


def _tail_run(text: str) -> tuple[str, str]:
    """Split ``text`` into (everything up to the last non-mark, trailing marks)."""
    i = len(text)
    while i > 0 and _is_mark(text[i - 1]):
        i -= 1
    return text[:i], text[i:]


class BurmeseValidator:
    """Incremental validator for a text stream.

    ``feed(chunk)`` returns ``(safe_text, cut)``: ``safe_text`` is what may be
    shown now, ``cut`` says an invalid marker was found and nothing further
    will ever be emitted. After a cut, every later call keeps returning
    ``("", True)`` — one bad syllable ends the message, it does not open a
    path back in.
    """

    def __init__(self) -> None:
        self._hold = ""          # trailing mark run whose fate is not decided yet
        self._open = False       # is a Myanmar syllable open right before _hold?
        self._dead = False
        self.accepted = ""       # everything handed out so far

    @property
    def cut(self) -> bool:
        return self._dead

    def reset(self) -> None:
        self._hold = ""
        self._open = False
        self._dead = False
        self.accepted = ""

    def feed(self, chunk: str) -> tuple[str, bool]:
        if self._dead:
            return "", True
        buf = self._hold + (chunk or "")
        if not buf:
            return "", False

        idx, _ = scan(buf, self._open)
        if idx < len(buf):
            # Invalid marker found: everything before it is clean and must be
            # handed out (it includes the held-back run), then the message ends.
            out = buf[:idx]
            self._hold = ""
            self._dead = True
            self.accepted += out
            return out, True

        # Valid so far. Emit everything except the trailing mark run — a mark
        # at the very end may still be waiting for its consonant.
        head, tail = _tail_run(buf)
        _, self._open = scan(head, self._open)
        self._hold = tail
        self.accepted += head
        return head, False

    def flush(self) -> str:
        """Settle the held-back run at end of stream. Returns newly safe text."""
        buf, self._hold = self._hold, ""
        if self._dead or not buf:
            return ""
        idx, _ = scan(buf, self._open)
        if idx < len(buf):
            out, self._dead = buf[:idx], True
        elif ord(buf[-1]) == VIRAMA:
            out, self._dead = buf[:-1], True         # B3 with nothing following
        else:
            out = buf
        self.accepted += out
        return out


def sanitize(text: str) -> str:
    """One-shot: return ``text`` with the first invalid marker and after cut off."""
    if not text:
        return text
    v = BurmeseValidator()
    v.feed(text)
    v.flush()
    return v.accepted
