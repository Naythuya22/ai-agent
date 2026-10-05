"""Burmese text must reach the UI correct — Thai lookalikes + agent prompt.

Session 22 findings (real model, real output):

1. ``filter_scripts`` dropped Thai characters outright, so a reply the model
   wrote as ``ခင်ဗျးာ။`` (Thai SARA AA instead of Myanmar AA) came out of the
   UI as ``ခင်ဗျ။`` — the vowel silently vanished.
2. ``AGENT_SYSTEM`` taught the model the wrong reply to 'ကျေးဇူးတင်ပါတယ်' —
   it answered 'ကြိုဆိုပါတယ်ခင်ဗျာ။' (welcome) verbatim, because that was the
   example in the prompt.

Run:  sh tests/run.sh -p "test_textfilter.py"
"""
from __future__ import annotations

import unittest

from _harness import bootstrap

bootstrap()

from routes._helpers import (  # noqa: E402
    _THAI_TO_MYANMAR,
    filter_scripts,
    normalize_confusables,
    repair_honorific,
    sanitize_text,
)

# Thai code points, written as escapes on purpose: Thai and Myanmar glyphs
# look identical on screen, so a literal would hide which block it came from.
THAI_SARA_AA = "\u0e32"     # း  -> should be ာ (U+102C)
THAI_SARA_I = "\u0e34"      # ိ  -> should be ိ (U+102D)
THAI_KO = "\u0e01"          # ก  (Thai consonant — still dropped)
THAI_MAI_EK = "\u0e48"      # ่  (Thai tone mark — still dropped)

# The exact byte the probe model produced.
PROBE_BUG = "ခင်ဗျ" + THAI_SARA_AA + "။"
PROBE_WANT = "ခင်ဗျာ။"

# Real Burmese sentences — none may change.
CLEAN = [
    "မင်္ဂလာပါ။ ဘာကူညီပေးရမလဲခင်ဗျာ။",
    "ကျေးဇူးတင်ပါတယ်ခင်ဗျာ။",
    "Python 3.14 ကို Termux မှာ install လုပ်ပါ။",
    "ဈေးနှုန်း – ၁၀၀၀ ကျပ်။ ၅၀% လျှော့။",
    "config.py ဖတ်ပြပါ (read_file)",
]


class TestThaiLookalikes(unittest.TestCase):
    def test_probe_case_keeps_the_vowel(self):
        """The recorded failure: dropping Thai deleted the vowel."""
        self.assertEqual(filter_scripts(PROBE_BUG), PROBE_WANT)

    def test_every_mapped_pair(self):
        for thai, target in _THAI_TO_MYANMAR.items():
            self.assertEqual(
                normalize_confusables(chr(thai)), chr(target),
                msg="U+%04X should map to U+%04X" % (thai, target),
            )

    def test_mappings_stay_inside_the_myanmar_block(self):
        for thai, myanmar in _THAI_TO_MYANMAR.items():
            self.assertTrue(0x1000 <= myanmar <= 0x109F)
            self.assertTrue(0x0e00 <= thai <= 0x0e7f)

    def test_unmapped_thai_is_still_dropped(self):
        """Consonants and tone marks have no safe twin — drop, never invent."""
        self.assertEqual(filter_scripts(THAI_KO), "")
        self.assertEqual(filter_scripts("က" + THAI_MAI_EK), "က")

    def test_thai_inside_a_sentence(self):
        self.assertEqual(
            filter_scripts("အခု" + THAI_SARA_AA + " " + THAI_KO + "hello"),
            "အခုာ hello",
        )

    def test_no_change_without_thai(self):
        for s in CLEAN:
            self.assertEqual(filter_scripts(s), s, msg=repr(s))

    def test_empty_and_none_safe(self):
        self.assertEqual(filter_scripts(""), "")
        self.assertEqual(normalize_confusables(""), "")
        self.assertEqual(normalize_confusables(None), None)


class TestChunkSafety(unittest.TestCase):
    """filter_scripts runs per streaming chunk — the split must not matter."""

    def test_any_chunk_size_equals_whole_string(self):
        text = PROBE_BUG + " ကျေး" + THAI_SARA_I + "းတယ်။ ဒါ" + THAI_SARA_AA + "မှန်တယ်"
        whole = filter_scripts(text)
        for size in (1, 2, 3, 5, 7, 64):
            joined = "".join(
                filter_scripts(text[i:i + size])
                for i in range(0, len(text), size)
            )
            self.assertEqual(joined, whole, msg="chunk size %d" % size)


class TestSanitizePath(unittest.TestCase):
    def test_sanitize_text_keeps_the_vowel(self):
        self.assertEqual(sanitize_text(PROBE_BUG), PROBE_WANT)

    def test_validator_does_not_cut_after_normalising(self):
        from core.burmese import BurmeseValidator

        v = BurmeseValidator()
        out, cut = v.feed(filter_scripts(PROBE_BUG))
        out += v.flush()
        self.assertFalse(cut)
        self.assertEqual(out, PROBE_WANT)

    def test_clean_sentences_survive_round_trip(self):
        for s in CLEAN:
            self.assertEqual(sanitize_text(s), s, msg=repr(s))


class TestHonorificRepair(unittest.TestCase):
    """The model drops the final AA vowel from ခင်ဗျ even when told not to."""

    def test_missing_aa_at_end_is_added(self):
        self.assertEqual(sanitize_text("မလဲခင်ဗျ။"), "မလဲခင်ဗျာ။")

    def test_missing_aa_before_space(self):
        self.assertEqual(sanitize_text("ကျေးဇူးတင်ပါတယ် ခင်ဗျ"),
                         "ကျေးဇူးတင်ပါတယ် ခင်ဗျာ")

    def test_already_correct_is_untouched(self):
        for s in ("မင်္ဂလာပါ။ ဘာကူညီပေးရမလဲခင်ဗျာ။", "ကျေးဇူးတင်ပါတယ်ခင်ဗျာ"):
            self.assertEqual(sanitize_text(s), s, msg=repr(s))

    def test_real_words_keep_their_spelling(self):
        for s in ("ခင်ဗျို့တို့", "ခင်ဗျံးလိုက်", "ခင်ဗျာ့"):
            self.assertEqual(repair_honorific(s), s, msg=repr(s))

    def test_repair_is_idempotent(self):
        once = sanitize_text("မလဲခင်ဗျ။")
        self.assertEqual(sanitize_text(once), once)

    def test_streaming_path_does_not_insert(self):
        """filter_scripts runs per chunk — no insertion where the follower
        may still be in the next chunk (that would risk ခင်ဗျာာ)."""
        self.assertEqual(filter_scripts("မလဲခင်ဗျ။"), "မလဲခင်ဗျ။")

    def test_probe_honorific_with_thai_vowel(self):
        """Thai SARA AA normalises to Myanmar AA, so nothing is inserted."""
        self.assertEqual(sanitize_text(PROBE_BUG), PROBE_WANT)

    def test_wrong_variants_from_live_transcript(self):
        """One live agent reply used five spellings of the same honorific."""
        cases = (
            ("ဖန်တက်ပေးနိုင်တယ်ခင်ျ။", "ဖန်တက်ပေးနိုင်တယ်ခင်ဗျာ။"),
            ("ပြောပြပါခင်ဗွာ။", "ပြောပြပါခင်ဗျာ။"),
            ("ပါတယ်ခဗျ။", "ပါတယ်ခင်ဗျာ။"),
            ("ပါခင်ဗလျ၊", "ပါခင်ဗျာ၊"),
            ("ခင်ျာ", "ခင်ဗျာ"),
        )
        for src, want in cases:
            self.assertEqual(sanitize_text(src), want, msg=repr(src))

    def test_variant_never_stops_at_a_real_letter(self):
        """A Myanmar letter after the match means the token is not finished."""
        for s in ("ခင်ျာကြိုက်တယ်", "ခင်ျာဆိုးတယ်"):
            self.assertEqual(repair_honorific(s), s, msg=repr(s))

    def test_variant_with_trailing_aa_does_not_double_it(self):
        """'ခင်ျာ' is rewritten as a whole token, then the AA rule sees the
        'ာ' it just produced and must not append a second one."""
        self.assertEqual(sanitize_text("မမှားခင်ျာ"), "မမှားခင်ဗျာ")

    def test_dropped_kha_is_left_alone(self):
        """'င်ဗျ' could be the tail of 'ခင်ဗျ' — ambiguous, so never touched."""
        s = "ရေးသားပေးနိုင်ပါတယ် င်ဗျ။"
        self.assertEqual(repair_honorific(s), s)

    def test_variant_repair_is_idempotent(self):
        once = sanitize_text("ပြောပြပါခင်ဗွာ။")
        self.assertEqual(sanitize_text(once), once)

    def test_variant_repair_does_not_touch_correct_honorific(self):
        for s in ("မင်္ဂလာပါခင်ဗျာ။", "ကျေးဇူးတင်ပါတယ်ခင်ဗျာ"):
            self.assertEqual(repair_honorific(s), s, msg=repr(s))


class TestAgentPrompt(unittest.TestCase):
    """The prompt is the other half of 'Burmese must be correct'.

    AGENT_SYSTEM was replaced wholesale on 2026-10-02 with the shorter
    "ညီ"/"ကိုကြီး" persona prompt, which drops the Thai-confusable and
    honorific-variant rules those tests used to pin down. Prompt-level
    assertions therefore now check the invariants THIS prompt promises;
    the rules that moved out of the prompt are pinned in
    TestAgentPromptCodeDefences, so none of the old regressions lose a
    guard.
    """

    @classmethod
    def setUpClass(cls):
        from modes.agent import AGENT_SYSTEM
        cls.prompt = AGENT_SYSTEM

    def test_wrong_thankyou_example_is_gone(self):
        # This exact wrong example was what the model parroted back.
        self.assertNotIn("ကြိုဆိုပါတယ်ခင်ဗျာ", self.prompt)

    def test_language_rules_demand_burmese_and_no_mixing(self):
        """Replaces the thank-you-example check: the prompt still pins the
        language contract that makes the example unnecessary."""
        self.assertIn("Always reply in Burmese", self.prompt)
        self.assertIn("Never mix languages within a sentence", self.prompt)
        self.assertIn("English is ONLY for technical terms", self.prompt)

    def test_language_rules_do_not_contradict(self):
        """'Always Burmese' and 'user's language' used to fight each other."""
        self.assertNotIn("Respond in the user's language (Burmese, English, etc.)",
                         self.prompt)
        self.assertNotIn("respond in the user's language", self.prompt.lower())

    def test_sections_are_not_run_together(self):
        """A missing "\\n" glued 'read_file(...)' onto 'ANDROID DEVICE…'.
        Whatever sections exist, every '## ' heading must open a line."""
        i = -1
        found = 0
        while True:
            i = self.prompt.find("## ", i + 1)
            if i < 0:
                break
            found += 1
            if i > 0:
                self.assertEqual(self.prompt[i - 1], "\n",
                                 msg="heading glued at %d: %r" % (i, self.prompt[i:i + 40]))
        self.assertGreater(found, 0, "prompt has no ## sections at all")
        self.assertNotIn(")ANDROID", self.prompt)
        self.assertNotIn(")SMART", self.prompt)

    def test_response_style_rules_are_present(self):
        """Replaces the dropped BURMESE CORRECTNESS block: the honesty and
        concision rules this prompt does carry."""
        for frag in ("Be concise", "Never make things up", "Skip generic greetings"):
            self.assertIn(frag, self.prompt)

    def test_prompt_names_the_wrong_honorific_variants(self):
        """The model invents these four.

        The prompt no longer spells them out (see module note), so the guard
        is the repair that runs on every reply: if any variant stops being
        rewritten, this test fails.
        """
        from routes._helpers import sanitize_text
        for wrong in ("ခင်ဗျ'", "ခင်ဗွာ", "ခင်ျ", "ခဗျ"):
            reply = sanitize_text("ကျေးဇူးတင်ပါတယ်" + wrong + "။")
            self.assertNotIn(wrong, reply, msg="%r survived repair: %r" % (wrong, reply))
            self.assertIn("ခင်ဗျာ", reply, msg=repr(reply))
        # and the prompt must not be teaching the wrong forms either
        for wrong in ("ခင်ဗွာ", "ခင်ျ", "ခဗျ"):
            self.assertNotIn(wrong, self.prompt)


class TestAgentPromptCodeDefences(unittest.TestCase):
    """Rules that used to live in AGENT_SYSTEM and now live in code.

    Keeping these here means a prompt rewrite can never silently take a
    Burmese-correctness defence with it.
    """

    def test_thai_confusables_are_still_mapped_out(self):
        from routes._helpers import _THAI_TO_MYANMAR, normalize_confusables
        # SARA AA / SARA I / SARA U / NIKHAHIT — the four that show up inside
        # Burmese words when the model slips into Thai codepoints.
        for thai in ("า", "ิ", "ุ", "ํ"):
            self.assertIn(ord(thai), _THAI_TO_MYANMAR, msg=repr(thai))
        self.assertNotIn("า", normalize_confusables("ခင်ဗျာ"))  # already correct
        self.assertEqual(normalize_confusables("ကောင်းပါ" + "า"),
                         "ကောင်းပါ" + "ာ")

    def test_honorific_repair_still_runs_on_whole_reply(self):
        from routes._helpers import sanitize_text
        bad = "ကျေးဇူးတင်ပါတယ်ခင်ဗွာ။"
        good = sanitize_text(bad)
        self.assertNotIn("ခင်ဗွာ", good)
        self.assertIn("ခင်ဗျာ", good)

    def test_reply_is_still_script_filtered(self):
        """`filter_scripts` keeps Burmese/Latin/visible punctuation and drops
        invisible Unicode (the model sprays ZWSP/BOM between myanmar glyphs)."""
        from routes._helpers import filter_scripts
        out = filter_scripts("မ​င်း﻿ ပါ")
        self.assertNotIn("​", out)
        self.assertNotIn("﻿", out)
        self.assertIn("မင်း ပါ", out)


if __name__ == "__main__":
    unittest.main()
