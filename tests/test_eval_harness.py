"""Release 20: the deterministic eval harness (eval/run_eval.py).

These tests cover the harness itself, offline, in stub mode only:
the shipped cases.json loads + validates, the stub run passes every
case through the REAL AgentMode path, a deliberately
failing check is reported by name with the score and exit code to
match, and a malformed case file is rejected with a clear error.

Run:  sh tests/run.sh -p "test_eval_harness.py"
"""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from _harness import bootstrap

bootstrap()

from eval import run_eval  # noqa: E402


class CasesFileTests(unittest.TestCase):
    def test_shipped_cases_load_and_validate(self):
        cases = run_eval.load_cases()
        ids = [c["id"] for c in cases]
        self.assertEqual(len(ids), len(set(ids)), "case ids must be unique")
        # Release 22: the 5 chat cases went with Chat Mode — 7 agent
        # cases remain.
        self.assertEqual(len(cases), 7)
        modes = {c["mode"] for c in cases}
        self.assertEqual(modes, {"agent"})

    def test_validate_rejects_duplicate_ids(self):
        data = {"version": 1, "cases": [
            {"id": "x", "mode": "agent", "prompt": "q",
             "checks": {"must_contain": ["a"]}},
            {"id": "x", "mode": "agent", "prompt": "q",
             "checks": {"must_contain": ["a"]}},
        ]}
        with self.assertRaisesRegex(ValueError, "duplicate case id"):
            run_eval.validate_cases(data)

    def test_validate_rejects_unknown_check(self):
        data = {"version": 1, "cases": [
            {"id": "x", "mode": "agent", "prompt": "q",
             "checks": {"vibes": ["good"]}},
        ]}
        with self.assertRaisesRegex(ValueError, "unknown check"):
            run_eval.validate_cases(data)

    def test_validate_rejects_bad_regex_and_wrong_version(self):
        bad_re = {"version": 1, "cases": [
            {"id": "x", "mode": "agent", "prompt": "q",
             "checks": {"regex": "("}},
        ]}
        with self.assertRaisesRegex(ValueError, "bad regex"):
            run_eval.validate_cases(bad_re)
        with self.assertRaisesRegex(ValueError, "version"):
            run_eval.validate_cases({"version": 2, "cases": []})

    def test_validate_rejects_setup_path_escape(self):
        data = {"version": 1, "cases": [
            {"id": "x", "mode": "agent", "prompt": "q",
             "setup": [{"path": "../evil.txt", "content": "x"}],
             "checks": {"must_contain": ["a"]}},
        ]}
        with self.assertRaisesRegex(ValueError, "sandbox"):
            run_eval.validate_cases(data)


class StubRunTests(unittest.TestCase):
    def test_stub_run_passes_every_shipped_case(self):
        cases = run_eval.load_cases()
        results = run_eval.run_cases(cases, stub=True)
        failed = [r["id"] for r in results if not r["passed"]]
        self.assertEqual(failed, [], run_eval.render(results))
        passed, total = run_eval.summarize(results)
        self.assertEqual(passed, total)
        self.assertEqual(run_eval.exit_code(results), 0)

    def test_failing_check_is_named_and_scores_down(self):
        case = copy.deepcopy(
            [c for c in run_eval.load_cases() if c["id"] == "agent-no-tool-question"][0])
        case["checks"] = {"must_contain": ["IMPOSSIBLE_ANSWER_TEXT"]}
        results = run_eval.run_cases([case], stub=True)
        self.assertFalse(results[0]["passed"])
        names = [ch["name"] for ch in results[0]["checks"] if not ch["ok"]]
        self.assertTrue(
            any(n.startswith("must_contain:IMPOSSIBLE") for n in names), names)
        self.assertEqual(run_eval.summarize(results), (0, 1))
        self.assertEqual(run_eval.exit_code(results), 1)
        self.assertIn("FAIL agent-no-tool-question", run_eval.render(results))
        self.assertIn("score: 0/1", run_eval.render(results))


class MainTests(unittest.TestCase):
    def test_main_stub_run_exits_zero(self):
        rc = run_eval.main(["--stub", "--no-report"])
        self.assertEqual(rc, 0)

    def test_main_rejects_a_bad_case_file(self):
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "cases.json"
            bad.write_text("{not json", encoding="utf-8")
            rc = run_eval.main(["--stub", "--no-report",
                                "--cases", str(bad)])
            self.assertEqual(rc, 2)
            bad.write_text(json.dumps({"version": 9, "cases": []}),
                           encoding="utf-8")
            rc = run_eval.main(["--stub", "--no-report",
                                "--cases", str(bad)])
            self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
