"""Unit tests for core.burmese (Session 21 P1 — sequence validator).

Run:  sh tests/run.sh -p "test_burmese.py"
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.burmese import MAX_MARKS, BurmeseValidator, sanitize, scan

# Written with an escape on purpose: the keyboard offers the asat (U+103A)
# where the stacking virama (U+1039) is meant, and they look identical.
TRAILING_VIRAMA = "မန္တလေး" + "\u1039"

# Real Burmese sentences — none of these may be truncated.
VALID = [
    "",
    "မြန်မာနိုင်ငံသည် အရှေ့တောင်အာရှတွင် တည်ရှိသည်။",
    "ကျွန်တော်က Python ဖြင့် ရေးသားထားသော စာသားဖြစ်သည်။",
    "အင်္ဂလိပ်ဘာသာစကားကို သင်ယူနေသည်။",          # kinzi: င ် ္ ဂ
    "မြောက်ကိုရီးယားနိုင်ငံ၏ လူဦးရေ ၂၅ သန်းခန့်ရှိသည်။",
    "သီဟိုဠ်ကျွန်းသည် လေးပတ်လည်ရှိသည်။",
    "ကျောင်းသားများသည် စာမေးပွဲဖြေဆိုနေကြသည်။",
    "ရန်ကုန်မြို့တွင် လူသွားလူလာများသည်။",
    "ဒေါ်လာ ၁၀၀ ဖြင့် ဝယ်ယူနိုင်သည်။",
    "ကျေးဇူးတင်ပါတယ် ခင်ဗျာ",
    "မင်္ဂလာပါ ခင်ဗျာ",
    "ဗုဒ္ဓဘာသာကို လူအများစု ကိုးကွယ်ကြသည်။",
    "ကွန်ပျူတာစကားပြောစက် (keyboard) ဖြင့် ရိုက်ထားသည်။",
    "ဧရာဝတီမြစ်သည် မြန်မာနိုင်ငံအတွက် အရေးကြီးသည်။",
    "Hello ကမ္ဘာ့ချစ်ခင်ရေးပေါ့ (123)",
]

class TestScan(unittest.TestCase):
    def test_valid_text_is_fully_accepted(self):
        for s in VALID:
            idx, _ = scan(s, False)
            self.assertEqual(idx, len(s), msg=repr(s))

    def test_orphan_mark_at_start(self):
        idx, _ = scan("ိက", False)
        self.assertEqual(idx, 0)

    def test_mark_after_latin_has_no_base(self):
        idx, _ = scan("abcိ", False)
        self.assertEqual(idx, 3)

    def test_duplicate_mark(self):
        idx, _ = scan("ကိိ", False)
        self.assertEqual(idx, 2)

    def test_virama_onto_non_letter(self):
        idx, _ = scan("က" + "\u1039" + "၁", False)   # stack onto a digit
        self.assertEqual(idx, 1)

    def test_virama_at_end_of_slice_is_not_judged(self):
        # The follower may still arrive in the next chunk — scan must not cut.
        idx, _ = scan("က်", False)
        self.assertEqual(idx, 2)

    def test_runaway_run(self):
        text = "က" + "ို့" * (MAX_MARKS // 2 + 4)
        idx, _ = scan(text, False)
        self.assertLess(idx, len(text))


class TestSanitize(unittest.TestCase):
    def test_valid_unchanged(self):
        for s in VALID:
            self.assertEqual(sanitize(s), s, msg=repr(s))

    def test_repeated_virama_cut(self):
        # Session 20's real corruption: 4x virama in a row.
        self.assertEqual(sanitize("ထိင်္္္္န်"), "ထိင်")

    def test_trailing_virama_cut(self):
        self.assertEqual(sanitize(TRAILING_VIRAMA), "မန္တလေး")

    def test_duplicate_vowel_repaired(self):
        # Repeated marks are dropped, not punished: the rest of the reply
        # must survive (a cut here used to end the whole message).
        self.assertEqual(sanitize("ကိို"), "ကို")

    def test_duplicate_medial_repaired(self):
        # Real probe output: model wrote ခင်ဗျ + ျ + ာ.
        self.assertEqual(sanitize("မလဲခင်ဗျျာ။"), "မလဲခင်ဗျာ။")

    def test_doubled_virama_still_ends_the_message(self):
        # A collapsed consonant stack has no single correct reading, so a
        # virama repeat keeps the old truncation policy.
        self.assertEqual(sanitize("ထိင်္္န်"), "ထိင်")

    def test_orphan_mark_cut(self):
        self.assertEqual(sanitize("ိက"), "")

    def test_kinzi_preserved(self):
        s = "အင်္ဂလိပ်"
        self.assertEqual(sanitize(s), s)

    def test_duplicate_asat_repaired_keeps_rest(self):
        # One duplicated asat: drop the repeat, keep the word after it.
        self.assertEqual(sanitize("ကျွန်တော််ပါ"), "ကျွန်တော်ပါ")

    def test_empty_and_non_myanmar(self):
        self.assertEqual(sanitize(""), "")
        self.assertEqual(sanitize("plain english 123"), "plain english 123")
        self.assertEqual(sanitize("中文"), "中文")   # not our filter's job


class TestStreaming(unittest.TestCase):
    def _stream(self, text, parts):
        v = BurmeseValidator()
        out = ""
        for p in parts:
            got, cut = v.feed(p)
            out += got
            if cut:
                break
        out += v.flush()
        return out

    def test_every_split_of_valid_text_is_identical(self):
        # A chunk boundary may land anywhere — including inside a syllable.
        for s in VALID:
            if not s:
                continue
            for i in range(len(s) + 1):
                got = self._stream(s, [s[:i], s[i:]])
                self.assertEqual(got, s, msg=f"{repr(s)} split at {i}")

    def test_every_split_matches_one_shot_on_broken_text(self):
        for s in ("ထိင်္္္္န်", "ကိို", TRAILING_VIRAMA, "ကျွန်တော််ပါ"):
            want = sanitize(s)
            for i in range(len(s) + 1):
                got = self._stream(s, [s[:i], s[i:]])
                self.assertEqual(got, want, msg=f"{repr(s)} split at {i}")

    def test_cut_is_sticky(self):
        v = BurmeseValidator()
        _, cut = v.feed("ိက")          # orphan mark — no safe repair
        self.assertTrue(cut)
        self.assertEqual(v.feed("ကမာ"), ("", True))
        self.assertEqual(v.feed("ကမာ"), ("", True))
        self.assertEqual(v.flush(), "")

    def test_held_run_is_emitted_on_flush(self):
        v = BurmeseValidator()
        out, cut = v.feed("က")
        self.assertFalse(cut)
        got, cut = v.feed("ိ")           # trailing mark — held back
        self.assertFalse(cut)
        out += got
        out += v.flush()
        self.assertEqual(out, "ကိ")

    def test_flush_cuts_lone_trailing_virama(self):
        v = BurmeseValidator()
        out, cut = v.feed(TRAILING_VIRAMA)
        self.assertFalse(cut)            # nothing decided yet: no follower known
        out += v.flush()
        self.assertEqual(out, "မန္တလေး")
        self.assertTrue(v.cut)

    def test_three_way_split_kinzi(self):
        s = "အင်္ဂလိပ်"
        for i in range(1, len(s)):
            for j in range(i + 1, len(s)):
                got = self._stream(s, [s[:i], s[i:j], s[j:]])
                self.assertEqual(got, s, msg=f"split {i},{j}")

    def test_accepted_tracks_emitted_text(self):
        v = BurmeseValidator()
        v.feed("ကျောင်း")
        v.flush()
        self.assertEqual(v.accepted, "ကျောင်း")

    def test_reset_clears_state(self):
        v = BurmeseValidator()
        v.feed("ိက")                    # unrecoverable — leaves a cut
        self.assertTrue(v.cut)
        v.reset()
        self.assertFalse(v.cut)
        out, cut = v.feed("ကျောင်း")
        self.assertFalse(cut)
        v.flush()
        self.assertEqual(v.accepted, "ကျောင်း")


if __name__ == "__main__":
    unittest.main()
