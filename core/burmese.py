"""Burmese combining-sequence validation.

``filter_scripts`` looks at one character at a time, so a structurally broken
Myanmar sequence — repeated virama, a virama with nothing to stack onto, an
orphaned medial — passes straight through: every character in it is
individually legal. The 4B model produces exactly that garbage under load
(``ထိင်္္္္န္``). This module validates the *sequence* instead.

Rules (enforced here in code, never left to the prompt):

  B1  the same combining mark never repeats back to back — and this is the
      one defect that is *repaired*: the repeat is dropped and the message
      continues. The model writes doubled marks routinely (``ခင်ဗျျာ``), and
      truncating there cost the rest of the reply: 2 of 3 real replies were
      cut short by it. A doubled virama is excluded — a collapsed stack has
      no single correct reading, so it still ends the message.
  B2  a Myanmar combining mark must sit inside a syllable that started with
      a Myanmar base — no orphan marks at the start of a message, after a
      space, after Latin text, …
  B3  U+1039 SIGN VIRAMA must be followed by a letter (a stacked consonant).
      At the end of the text there is nothing left to stack onto → invalid.
  B4  more than MAX_MARKS combining marks in a row cannot be a real Burmese
      syllable (backstop for alternating garbage that B1 does not catch).
      Worst mark run measured on real text is 4, so 6 leaves headroom.

B2/B3/B4 have no safe reading, so they truncate at the first invalid marker
and everything after it is withheld from the UI.

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


def _is_repeat_mark(text: str, idx: int) -> bool:
    """True when the defect at ``idx`` is B1 — a mark repeated right after
    itself — and dropping the repeat is a safe repair.

    ``scan`` only reports B1 inside an open syllable (B2 returns first for an
    orphan), so the neighbour check is enough. A doubled virama is not
    repairable: it means a collapsed consonant stack with no single reading.
    """
    if idx <= 0 or idx >= len(text):
        return False
    ch = text[idx]
    return _is_mark(ch) and ch == text[idx - 1] and ord(ch) != VIRAMA


def _repair_repeat_marks(text: str, open_syllable: bool) -> tuple[str, int]:
    """Drop every repeated combining mark.

    Returns ``(text, fatal_index)``: ``fatal_index`` is ``-1`` once the whole
    slice validates, otherwise the index of the first defect that has no safe
    repair (B2/B3/B4), at which point the caller cuts.
    """
    while True:
        idx, _ = scan(text, open_syllable)
        if idx >= len(text):
            return text, -1
        if not _is_repeat_mark(text, idx):
            return text, idx
        text = text[:idx] + text[idx + 1:]


def _tail_run(text: str) -> tuple[str, str]:
    """Split ``text`` into (everything up to the last non-mark, trailing marks)."""
    i = len(text)
    while i > 0 and _is_mark(text[i - 1]):
        i -= 1
    return text[:i], text[i:]


class BurmeseValidator:
    """Incremental validator for a text stream.

    ``feed(chunk)`` returns ``(safe_text, cut)``: ``safe_text`` is what may be
    shown now, ``cut`` says an unrecoverable marker was found and nothing
    further will ever be emitted. After a cut, every later call keeps returning
    ``("", True)`` — one unreadable sequence ends the message, it does not
    open a path back in. Repeated marks never cause a cut: they are dropped
    and the message carries on.
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

        buf, fatal = _repair_repeat_marks(buf, self._open)
        if fatal >= 0:
            # Unreadable marker (orphan mark, stray virama, runaway run):
            # everything before it is clean and must be handed out (it
            # includes the held-back run and any repaired duplicates), then
            # the message ends.
            out = buf[:fatal]
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
        buf, fatal = _repair_repeat_marks(buf, self._open)
        if fatal >= 0:
            out, self._dead = buf[:fatal], True
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


# ── Release 26: whole-reply Burmese quality gate ─────────────────────
#
# sanitize() above judges combining-mark *sequences*. It cannot judge
# whether a finished reply is readable Burmese at all: the free-model
# roulette (tester evaluation, 2026-10-23) produced replies in mixed
# scripts and accented-Latin salad that are perfectly valid Unicode —
# ``Hóa chất``-style fragments, Hangul/Thai shards and ၃၀%-Burmese
# mixes — plus well-formed Burmese that means the opposite of the
# truth. The deterministic part (script composition) is checkable;
# the semantic part is not, and this gate does not pretend otherwise:
# it exists to catch script salad so the caller can retry the answer
# once through another provider.
#
# A reply is scored 0.0–1.0. ``ok`` is False only on a hard signal:
#   * Burmese letters are a minority of the letters (< 45%) in a reply
#     long enough to judge (≥ 30 letters);
#   * letters from scripts that never belong in a Burmese answer
#     (Han, kana, Hangul, Hebrew, Arabic, Cyrillic, Devanagari, Thai)
#     exceed 15% of the letters;
#   * two or more accented-Latin words (U+00C0–U+024F) — legitimate
#     technical English in this app is ASCII; Vietnamese-style salad
#     is the observed failure shape;
#   * the combining-mark sanitizer keeps < 70% of the text, i.e. the
#     Burmese it does contain is itself malformed;
#   * a degenerate repetition loop (one token repeated ~100× — the
#     shape openai/gpt-oss-20b produced live on Groq, 2026-10-23):
#     ≥ 10 identical consecutive tokens, or one token over half of a
#     ≥ 20-token reply. Pure Burmese script, so only its own detector
#     catches it.
# Digits, punctuation, emoji and ASCII English terms (``EV``,
# ``OpenRouter``) are not letters for this purpose and never fail a
# reply on their own.

_OTHER_SCRIPT_RANGES = (
    (0x0400, 0x04FF),   # Cyrillic
    (0x0590, 0x05FF),   # Hebrew
    (0x0600, 0x06FF),   # Arabic
    (0x0900, 0x097F),   # Devanagari
    (0x0E00, 0x0E7F),   # Thai
    (0x3040, 0x30FF),   # kana
    (0x3400, 0x4DBF),   # Han (ext A)
    (0x4E00, 0x9FFF),   # Han
    (0xAC00, 0xD7AF),   # Hangul syllables
    (0xF900, 0xFAFF),   # Han (compat ideographs)
)

_MIN_JUDGE_LETTERS = 30
_BURMESE_SHARE_FLOOR = 0.45
_OTHER_SHARE_CEIL = 0.15
_SANITIZE_KEEP_FLOOR = 0.70


def _is_burmese_char(ch: str) -> bool:
    return "\u1000" <= ch <= "\u109f"


def _is_other_script(ch: str) -> bool:
    o = ord(ch)
    return any(lo <= o <= hi for lo, hi in _OTHER_SCRIPT_RANGES)


def has_myanmar(text: str) -> bool:
    """True when the text contains any Myanmar-script character."""
    return any(_is_burmese_char(c) for c in text or "")


def _repetition_fail(text: str) -> bool:
    """True for a degenerate repetition loop — one token repeated
    ~100×, the shape openai/gpt-oss-20b produced live on the EV prompt
    (2026-10-23) while answering the maths correctly. Such a reply is
    pure Burmese script, so the composition checks below all pass it;
    it needs its own detector. Conservative by construction: the
    reply needs ≥ 20 tokens before anything is judged, then it fails
    only on a run of ≥ 10 identical consecutive tokens or a single
    token making up more than half the reply. Trailing punctuation is
    ignored when comparing tokens ("ကား၊" vs "ကား")."""
    toks = [(t.strip("၊။.,!?…\"'") or t) for t in (text or "").split()]
    if len(toks) < 20:
        return False
    run = 1
    for a, b in zip(toks, toks[1:]):
        run = run + 1 if a == b else 1
        if run >= 10:
            return True
    counts: dict = {}
    for t in toks:
        counts[t] = counts.get(t, 0) + 1
    return max(counts.values()) * 2 > len(toks)


def reply_quality(text: str) -> tuple[bool, float]:
    """Score a finished reply that was supposed to be Burmese.

    Returns ``(ok, score)``. Replies too short to judge (fewer than
    30 letters) pass with a neutral score — a one-line answer is not
    the roulette's failure shape, and gating it would retry perfectly
    good short answers. ``score`` ranks two failing replies: the
    caller keeps the better one when a retry also fails.
    """
    letters = [c for c in (text or "") if c.isalpha()]
    if len(letters) < _MIN_JUDGE_LETTERS:
        return True, 0.5
    burmese = sum(1 for c in letters if _is_burmese_char(c))
    other = sum(1 for c in letters if _is_other_script(c))
    share = burmese / len(letters)
    other_share = other / len(letters)
    score = share - other_share
    words = (text or "").split()
    accented = sum(1 for w in words
                   if any("\u00c0" <= c <= "\u024f" for c in w))
    if accented >= 2:
        score -= 0.3
    kept = len(sanitize(text)) / max(1, len(text or ""))
    if kept < _SANITIZE_KEEP_FLOOR:
        score -= 0.3
    rep = _repetition_fail(text)
    if rep:
        score -= 0.4
    ok = (share >= _BURMESE_SHARE_FLOOR
          and other_share <= _OTHER_SHARE_CEIL
          and accented < 2
          and kept >= _SANITIZE_KEEP_FLOOR
          and not rep)
    return ok, max(0.0, min(1.0, score))
