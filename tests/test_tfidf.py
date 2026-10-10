"""Unit tests for core.tokens + core.tfidf (BM25-lite).

Run:  python3 test.py test_tfidf.py
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.tfidf import BM25Lite
from core.tokens import NGRAM_SIZES, tokenize


class TestTokenize(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize(None or ""), [])

    def test_latin_word_unigrams(self):
        self.assertEqual(tokenize("Hello World"), ["hello", "world"])

    def test_latin_keeps_digits(self):
        self.assertEqual(tokenize("port 8080"), ["port", "8080"])

    def test_punctuation_is_separator(self):
        self.assertEqual(tokenize("a,b;c"), ["a", "b", "c"])

    def test_burmese_char_ngrams(self):
        toks = tokenize("ငွေကြေး")
        # every n-gram size that fits must be present
        self.assertTrue(all(t in toks for t in ("ငွေ", "ကြေး")))
        # n-grams of each size 2..4 exist (len=5 -> all three sizes fit)
        self.assertTrue(any(len(t) == 2 for t in toks))
        self.assertTrue(any(len(t) == 3 for t in toks))
        self.assertTrue(any(len(t) == 4 for t in toks))

    def test_burmese_single_char_kept(self):
        self.assertEqual(tokenize("က"), ["က"])

    def test_mixed_burmese_latin(self):
        toks = tokenize("Termux မှာ")
        self.assertIn("termux", toks)
        self.assertTrue(any(len(t) >= 2 and t != "termux" for t in toks))

    def test_overlap_between_query_and_doc(self):
        # shared Burmese substring must produce a shared token
        a = set(tokenize("ငွေကြေးစာအုပ်"))
        b = set(tokenize("Rich Dad Poor Dad စာအုပ်"))
        self.assertTrue(a & b)


class TestBM25Lite(unittest.TestCase):
    def _idx(self):
        idx = BM25Lite()
        idx.build(
            [
                ("f1", "ငွေကြေးစာအုပ် Rich Dad Poor Dad"),
                ("f2", "ai-agent is a Termux project"),
                ("f3", "သိပ္ပံနဲ့နည်းပညာ"),
            ]
        )
        return idx

    def test_empty_index(self):
        idx = BM25Lite()
        self.assertEqual(idx.search("anything"), [])

    def test_empty_query(self):
        idx = self._idx()
        self.assertEqual(idx.search(""), [])
        self.assertEqual(idx.search("   "), [])

    def test_no_match_returns_empty(self):
        idx = self._idx()
        self.assertEqual(idx.search("zzzznotpresentzzz"), [])

    def test_basic_ranking(self):
        idx = self._idx()
        hits = idx.search("ငွေကြေးစာအုပ်")
        self.assertTrue(hits)
        self.assertEqual(hits[0][0], "f1")
        self.assertTrue(hits[0][1] > 0)

    def test_k_limits(self):
        idx = BM25Lite()
        idx.build([(f"f{i}", "common text common") for i in range(30)])
        self.assertLessEqual(len(idx.search("common", k=5)), 5)

    def test_scores_sorted_desc(self):
        idx = self._idx()
        hits = idx.search("Termux project")
        scores = [s for _, s in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_idf_formula(self):
        idx = BM25Lite()
        idx.build([("a", "x y"), ("b", "x z"), ("c", "w v")])
        # token "x": N=3, df=2 -> log((3-2+0.5)/(2+0.5)+1)
        expected = math.log((3 - 2 + 0.5) / (2 + 0.5) + 1)
        self.assertAlmostEqual(idx._idf("x"), expected)
        # unknown token -> 0
        self.assertEqual(idx._idf("nope"), 0.0)

    def test_update_replace(self):
        idx = BM25Lite()
        idx.build([("a", "alpha"), ("b", "beta")])
        idx.update("a", "gamma")
        self.assertEqual(idx.search("alpha"), [])
        self.assertEqual(idx.search("gamma")[0][0], "a")
        self.assertEqual(idx._N, 2)

    def test_update_append(self):
        idx = BM25Lite()
        idx.build([("a", "alpha")])
        idx.update("b", "beta")
        self.assertEqual(idx._N, 2)
        self.assertEqual(idx.search("beta")[0][0], "b")

    def test_rebuild_resets_df(self):
        idx = self._idx()
        idx.build([("z", "sole item")])
        self.assertEqual(idx._N, 1)
        self.assertEqual(idx.search("Termux"), [])


if __name__ == "__main__":
    unittest.main()
