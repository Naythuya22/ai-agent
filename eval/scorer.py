"""Score one eval result against its expect block. Pure Python."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class Score:
    case_id: str
    category: str
    passed: bool
    checks: list = field(default_factory=list)
    runs_total: int = 1
    runs_passed: int = 1

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "passed": self.passed,
            "runs_total": self.runs_total,
            "runs_passed": self.runs_passed,
            "checks": [
                {"name": c.name, "ok": c.ok, "detail": c.detail} for c in self.checks
            ],
        }


def score_case(case: dict, answer: str, tools_called: list[str]) -> Score:
    """Apply the case's ``expect`` block to the captured answer + tools.

    Supported expect keys (all optional, all must pass):
      must_include    list[str]  — every substring must appear in answer
      must_not_include list[str] — none may appear in answer
      tools           list[str]  — every tool name must be in tools_called
      no_tools        bool       — True -> tools_called must be empty
    """
    cid = case.get("id", "?")
    cat = case.get("category", "?")
    expect = case.get("expect") or {}
    checks: list[Check] = []
    answer = answer or ""

    for needle in expect.get("must_include") or []:
        ok = needle.lower() in answer.lower()
        checks.append(Check(
            f"must_include:{needle}", ok,
            "" if ok else f"missing in answer: {needle!r}",
        ))

    for needle in expect.get("must_not_include") or []:
        ok = needle.lower() not in answer.lower()
        checks.append(Check(
            f"must_not_include:{needle}", ok,
            "" if ok else f"forbidden text present: {needle!r}",
        ))

    for tool in expect.get("tools") or []:
        ok = tool in tools_called
        checks.append(Check(
            f"tool:{tool}", ok,
            "" if ok else f"tool not called: {tool} (called: {tools_called})",
        ))

    if expect.get("no_tools"):
        ok = not tools_called
        checks.append(Check(
            "no_tools", ok,
            "" if ok else f"expected no tools, got: {tools_called}",
        ))

    if not checks:
        checks.append(Check("expect_empty", False,
                            "case has no expect criteria — cannot pass"))
    passed = all(c.ok for c in checks)
    return Score(cid, cat, passed, checks)


def aggregate_scores(run_scores: list[Score]) -> Score:
    """Majority verdict across N runs of the SAME case.

    passed = (runs_passed * 2 > runs_total)  — strict majority.
    Checks shown are from the last run that matches the majority verdict
    (or the last run overall if there is no majority pass).
    """
    if not run_scores:
        raise ValueError("aggregate_scores: empty run_scores")
    total = len(run_scores)
    passed_runs = sum(1 for s in run_scores if s.passed)
    majority = passed_runs * 2 > total
    rep = run_scores[-1]
    for s in reversed(run_scores):
        if s.passed == majority:
            rep = s
            break
    return Score(
        case_id=rep.case_id,
        category=rep.category,
        passed=majority,
        checks=list(rep.checks),
        runs_total=total,
        runs_passed=passed_runs,
    )
