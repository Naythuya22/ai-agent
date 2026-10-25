"""Prompt-content tests for FAILURE REPORTING vocabulary."""
import sys
import unittest
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from core import prompts as P


class FailureVocabTests(unittest.TestCase):
    def _read(self, name):
        return (Path(P.PROMPTS_DIR) / name).read_text(encoding="utf-8")

    def test_base_prompt_contains_failure_block(self):
        t = self._read("_base.txt")
        self.assertIn("FAILURE REPORTING", t)
        self.assertIn("BLOCKED:", t)

    def test_base_failure_uses_three_prefixes(self):
        t = self._read("_base.txt")
        for tok in ("BLOCKED:", "PARTIAL:", "UNSURE:"):
            self.assertIn(tok, t, tok + " missing")

    def test_each_tier_references_blocked(self):
        for tier in ("cloud_strong", "cloud_fast", "local"):
            t = self._read("agent_" + tier + ".txt")
            self.assertIn("BLOCKED", t, tier + " missing BLOCKED")

    def test_local_prompt_has_burmese_example(self):
        t = self._read("agent_local.txt")
        self.assertIn("BLOCKED:", t)
        has_burmese = any(0x1000 <= ord(c) <= 0x109F for c in t)
        self.assertTrue(has_burmese, "local prompt missing Burmese")

    def test_all_prompts_still_fit_budget(self):
        # 2026-10-10: same re-tune as test_prompts.py — Burmese-native rewrite
        # made the composed prompts denser; caps track that with ~10% headroom.
        # Release 28 (2026-10-23): caps moved with the two-axis matrix —
        # load_prompt() now assembles base + tier + the Balanced effort
        # overlay. Kept in step with test_prompts.py on purpose: two
        # pins of the same budget must never disagree.
        base = self._read("_base.txt")
        self.assertLessEqual(P.estimate_tokens(base), 760)
        # Release 31: local 1110 -> 1160, in step with test_prompts.py
        # (CODE ANSWERS section in the local tier file).
        caps = {"cloud_strong": 1220, "cloud_fast": 1300, "local": 1160}
        for tier, cap in caps.items():
            loaded = P.load_prompt(tier)
            n = P.estimate_tokens(loaded)
            self.assertLessEqual(n, cap, tier + "=" + str(n) + ">" + str(cap))


if __name__ == "__main__":
    unittest.main()
