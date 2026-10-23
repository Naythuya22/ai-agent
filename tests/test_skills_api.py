"""Skills API — Release 22 pin for the create endpoint's LLM path.

Skill creation generates its template through the shared LLM handle.
It used to borrow the ChatMode object's ``llm`` inside a try/except
whose fallback is a default template, so a broken handle degraded
skill creation *silently* instead of failing. This pin proves the
generated body really comes from the LLM: patch the singleton's
``ask`` and the saved skill must be the generated text, reported
with ``source == "llm"``.

Run:  sh tests/run.sh -p "test_skills_api.py"
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from _harness import bootstrap, get_app

bootstrap()
app = get_app()
client = app.test_client()

TEMPLATE = (
    "You are an expert proofreader. Fix the text described below and "
    "return only the corrected version, keeping the user's language.\n\n"
    "Input:\n\n{{args}}\n"
)


class TestSkillCreateUsesLLM(unittest.TestCase):
    def test_generated_template_comes_from_the_llm(self):
        import core.skills as skills_mod
        from routes._state import get_llm

        llm = get_llm(load=False)
        real_ask = llm.ask
        saved_dir = skills_mod.SKILLS_DIR
        with tempfile.TemporaryDirectory() as td:
            skills_mod.SKILLS_DIR = Path(td) / "skills"
            llm.ask = lambda prompt: TEMPLATE
            try:
                r = client.post("/api/skills/create", json={
                    "name": "proofread-pin",
                    "description": "fix my text",
                })
                self.assertEqual(r.status_code, 200,
                                 r.get_data(as_text=True))
                body = r.get_json()
                self.assertEqual(body["source"], "llm", body)
                sk = skills_mod.load_skill("proofread-pin")
                self.assertIsNotNone(sk)
                self.assertIn("expert proofreader", sk["body"])
            finally:
                llm.ask = real_ask
                skills_mod.SKILLS_DIR = saved_dir


if __name__ == "__main__":
    unittest.main()
