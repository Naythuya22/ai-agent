"""Unit tests for eval harness (scorer, runner, report). Stub mode only.

Run:  python3 test.py test_eval.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.report import render, summarize
from eval.runner import load_cases, run, run_stub, write_results
from eval.scorer import score_case


def _case(**over):
    base = {
        "id": "t1", "category": "memory", "input": "q",
        "expect": {"must_include": ["hello"]},
        "stub": {"answer": "hello world", "tools": []},
    }
    base.update(over)
    return base


class TestScorer(unittest.TestCase):
    def test_must_include_pass(self):
        s = score_case(_case(), "hello there", [])
        self.assertTrue(s.passed)

    def test_must_include_fail(self):
        s = score_case(_case(), "goodbye", [])
        self.assertFalse(s.passed)
        self.assertIn("must_include:hello", s.checks[0].name)

    def test_must_not_include(self):
        c = _case(expect={"must_not_include": ["secret"]})
        self.assertTrue(score_case(c, "clean answer", []).passed)
        self.assertFalse(score_case(c, "the secret is x", []).passed)

    def test_tools_required(self):
        c = _case(expect={"tools": ["web_search"]})
        self.assertTrue(score_case(c, "a", ["web_search"]).passed)
        self.assertFalse(score_case(c, "a", ["memory_read"]).passed)

    def test_no_tools(self):
        c = _case(expect={"no_tools": True})
        self.assertTrue(score_case(c, "a", []).passed)
        self.assertFalse(score_case(c, "a", ["web_search"]).passed)

    def test_case_insensitive(self):
        c = _case(expect={"must_include": ["HELLO"]})
        self.assertTrue(score_case(c, "hello world", []).passed)

    def test_no_criteria_fails(self):
        c = _case(expect={})
        s = score_case(c, "anything", [])
        self.assertFalse(s.passed)

    def test_to_dict(self):
        s = score_case(_case(), "hello", [])
        d = s.to_dict()
        self.assertEqual(d["case_id"], "t1")
        self.assertTrue(d["passed"])


class TestRunner(unittest.TestCase):
    def test_run_stub_uses_scripted_answer(self):
        scores = run_stub([_case()])
        self.assertEqual(len(scores), 1)
        self.assertTrue(scores[0].passed)

    def test_run_stub_failing_case(self):
        c = _case(stub={"answer": "wrong", "tools": []})
        self.assertFalse(run_stub([c])[0].passed)

    def test_run_dispatch(self):
        self.assertEqual(len(run([_case()], mode="stub")), 1)
        with self.assertRaises(ValueError):
            run([], mode="bogus")

    def test_run_live_unreachable_server_scores_as_failure(self):
        # 127.0.0.1:1 refuses fast — wiring check, NO real model.
        from eval.runner import run_live
        c = _case(expect={"must_include": ["__never__"]})
        scores = run_live([c], base_url="http://127.0.0.1:1", timeout=1)
        self.assertEqual(len(scores), 1)
        self.assertFalse(scores[0].passed)
        self.assertTrue(all(not ch.ok for ch in scores[0].checks))

    def test_run_live_parses_reply_and_tools(self):
        import json as _json
        from unittest import mock
        from eval.runner import run_live

        payload = _json.dumps({
            "reply": "hello world",
            "tool_trace": [
                {"type": "call", "tool": "memory_search", "args": {}},
                {"type": "result", "tool": "memory_search", "result": "ok"},
            ],
            "status": "ready",
        }).encode("utf-8")
        fake_resp = mock.MagicMock()
        fake_resp.__enter__.return_value.read.return_value = payload
        fake_resp.__exit__.return_value = False
        c = _case(expect={"must_include": ["hello"], "tools": ["memory_search"]})
        with mock.patch("urllib.request.urlopen", return_value=fake_resp):
            scores = run_live([c], base_url="http://example.invalid")
        self.assertTrue(scores[0].passed)

    def test_run_live_agent_error_scores_as_failure(self):
        import json as _json
        from unittest import mock
        from eval.runner import run_live

        payload = _json.dumps({"error": "boom"}).encode("utf-8")
        fake_resp = mock.MagicMock()
        fake_resp.__enter__.return_value.read.return_value = payload
        fake_resp.__exit__.return_value = False
        c = _case(expect={"must_include": ["__never__"]})
        with mock.patch("urllib.request.urlopen", return_value=fake_resp):
            scores = run_live([c], base_url="http://example.invalid")
        self.assertFalse(scores[0].passed)
        self.assertTrue(all(not ch.ok for ch in scores[0].checks))

    def test_load_cases_filters_category(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "c.jsonl"
            p.write_text(
                json.dumps({"id": "a", "category": "memory", "input": "x"})
                + "\n"
                + json.dumps({"id": "b", "category": "tool", "input": "y"})
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(len(load_cases(p)), 2)
            self.assertEqual(len(load_cases(p, category="memory")), 1)

    def test_load_cases_rejects_bad_shape(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "c.jsonl"
            p.write_text(json.dumps({"category": "x"}) + "\n",
                         encoding="utf-8")
            with self.assertRaises(ValueError):
                load_cases(p)

    def test_write_results(self):
        scores = run_stub([_case()])
        with tempfile.TemporaryDirectory() as td:
            out = write_results(scores, results_dir=Path(td))
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(data["total"], 1)
            self.assertEqual(data["passed"], 1)


class TestReport(unittest.TestCase):
    def test_summarize(self):
        scores = run_stub([
            _case(id="a", category="memory"),
            _case(id="b", category="tool",
                  stub={"answer": "x", "tools": []}),
        ])
        sm = summarize(scores)
        self.assertEqual(sm["total"], 2)
        self.assertIn("memory", sm["categories"])
        self.assertIn("tool", sm["categories"])

    def test_render_contains_rate(self):
        scores = run_stub([_case()])
        text = render(scores)
        self.assertIn("Eval: 1/1", text)


class TestBundledCases(unittest.TestCase):
    """The 10 shipped cases must load and all pass in stub mode."""

    def test_all_bundled_cases_pass_stub(self):
        cases = load_cases()
        self.assertGreaterEqual(len(cases), 10)
        cats = {c["category"] for c in cases}
        self.assertEqual(
            cats, {"memory", "tool", "language", "honesty", "security"})
        scores = run_stub(cases)
        failed = [s.case_id for s in scores if not s.passed]
        self.assertEqual(failed, [], msg=f"stub failures: {failed}")


if __name__ == "__main__":
    unittest.main()
