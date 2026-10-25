"""Release 31: the local code gate (core.codecheck) + its two call sites.

Fixtures are the owner's own phone evidence (2026-10-25): Gemma 4
(Local) answered a strict code-only request with a ``for`` loop and the
final ``return`` dedented out of the function — a SyntaxError — and a
spoken correction round did not fix it. The gate exists so that exact
failure gets one machine-checked repair attempt, and only ever on the
local path.
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stubs"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import codecheck  # noqa: E402

GEMMA_BROKEN = (
    "def calculate_total(items):\n"
    "    if not items:\n"
    "        return 0\n"
    "    total = 0\n"
    "for item in items:\n"
    "    try:\n"
    "        price = float(item['price'])\n"
    "        quantity = int(item['quantity'])\n"
    "    except ValueError:\n"
    "        raise ValueError(\"Price or quantity must be a number.\")\n"
    "    total += price * quantity\n"
    "return total"
)

FIXED = (
    "def calculate_total(items):\n"
    "    total = 0\n"
    "    for item in items:\n"
    "        price = item['price']\n"
    "        quantity = item['quantity']\n"
    "        if price < 0 or quantity < 0:\n"
    "            raise ValueError(\"price and quantity must be non-negative\")\n"
    "        total += price * quantity\n"
    "    return total"
)

PROSE = ("ဒီနေ့ ရာသီဥတု ကောင်းပါတယ်။ def ဆိုတာ Python မှာ function "
         "စဖွင့်တဲ့ စကားလုံး ဖြစ်ပါတယ်။")


class CountingSend:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.reply


class ExtractTests(unittest.TestCase):
    def test_bare_code_reply_is_extracted(self):
        self.assertEqual(codecheck.extract_code(GEMMA_BROKEN), GEMMA_BROKEN)

    def test_prose_mentioning_def_is_not_code(self):
        self.assertIsNone(codecheck.extract_code(PROSE))

    def test_fenced_block_wins(self):
        text = "ဒီမှာ code ပါ:\n```python\nx = 1\n```"
        self.assertEqual(codecheck.extract_code(text), "x = 1")

    def test_empty_is_not_code(self):
        self.assertIsNone(codecheck.extract_code(""))
        self.assertIsNone(codecheck.extract_code("   \n  "))


class SyntaxTests(unittest.TestCase):
    def test_gemma_broken_fails_on_the_escaped_return(self):
        err = codecheck.syntax_error(GEMMA_BROKEN)
        self.assertIsNotNone(err)
        # The return sits on the last line — the count is the assertion
        # that the checker found THAT error, not just any error.
        self.assertEqual(err[0], GEMMA_BROKEN.count("\n") + 1)

    def test_fixed_code_parses(self):
        self.assertIsNone(codecheck.syntax_error(FIXED))


class RepairTests(unittest.TestCase):
    def test_broken_code_is_repaired_in_one_send(self):
        send = CountingSend(FIXED)
        self.assertEqual(codecheck.repair(GEMMA_BROKEN, send), FIXED)
        self.assertEqual(len(send.prompts), 1)
        self.assertIn("syntax error", send.prompts[0])

    def test_still_broken_repair_keeps_the_original(self):
        send = CountingSend(GEMMA_BROKEN)
        self.assertEqual(codecheck.repair(GEMMA_BROKEN, send), GEMMA_BROKEN)
        self.assertEqual(len(send.prompts), 1)

    def test_valid_code_never_sends(self):
        send = CountingSend(FIXED)
        self.assertEqual(codecheck.repair(FIXED, send), FIXED)
        self.assertEqual(len(send.prompts), 0)

    def test_prose_never_sends(self):
        send = CountingSend(FIXED)
        self.assertEqual(codecheck.repair(PROSE, send), PROSE)
        self.assertEqual(len(send.prompts), 0)

    def test_send_failure_keeps_the_original(self):
        def boom(prompt):
            raise RuntimeError("engine died")
        self.assertEqual(codecheck.repair(GEMMA_BROKEN, boom), GEMMA_BROKEN)

    def test_gate_off_never_sends(self):
        send = CountingSend(FIXED)
        with mock.patch.dict(os.environ, {"AI_AGENT_CODE_GATE": "off"}):
            self.assertEqual(codecheck.repair(GEMMA_BROKEN, send),
                             GEMMA_BROKEN)
        self.assertEqual(len(send.prompts), 0)


class ScriptedConv:
    """The local loop's conversation shape: queued text responses."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []

    def send_message(self, message):
        self.sent.append(message)
        return {"content": [{"text": self.replies.pop(0)}]}


def _local_mode(replies):
    from modes.agent import AgentMode

    class _LLM:
        pass

    m = AgentMode(_LLM(), None)
    m._mode = lambda: "auto"
    m._try_cloud = lambda *a, **k: None
    m._ensure_conv = lambda **k: None
    m._cap_history = lambda **k: None
    m._finish_turn = lambda *a, **k: None
    m._conv = ScriptedConv(replies)
    return m


class HandleIntegrationTests(unittest.TestCase):
    def test_handle_repairs_broken_local_code(self):
        m = _local_mode([GEMMA_BROKEN, FIXED])
        out = m.handle("Python code တစ်ခုတည်းပဲ ထုတ်ပါ။ "
                       "calculate_total(items) function ရေးပါ။")
        self.assertEqual(out, FIXED)
        # Exactly one extra conversation send: the repair attempt.
        self.assertEqual(len(m._conv.sent), 2)
        self.assertIn("syntax error", m._conv.sent[1])
        self.assertEqual(m._messages[-1]["text"], FIXED)

    def test_handle_leaves_valid_local_code_alone(self):
        m = _local_mode([FIXED])
        out = m.handle("Python code တစ်ခုတည်းပဲ ထုတ်ပါ။")
        self.assertEqual(out, FIXED)
        self.assertEqual(len(m._conv.sent), 1)


class LocalPromptTests(unittest.TestCase):
    def test_local_tier_prompt_carries_the_code_section(self):
        from core import prompts as prompt_tiers
        text = prompt_tiers.load_prompt("local", "fast")
        self.assertIn("CODE ANSWERS", text)
        self.assertIn("total_price", text)

    def test_cloud_tier_prompts_have_no_code_section(self):
        from core import prompts as prompt_tiers
        for tier in ("cloud_fast", "cloud_strong"):
            self.assertNotIn("CODE ANSWERS",
                             prompt_tiers.load_prompt(tier, "fast"))


if __name__ == "__main__":
    unittest.main()
