"""Per-category pass-rate report. Pure Python."""
from __future__ import annotations

from eval.scorer import Score


def summarize(scores: list[Score]) -> dict:
    """Aggregate: overall + per-category pass counts."""
    by_cat: dict[str, list[Score]] = {}
    for s in scores:
        by_cat.setdefault(s.category, []).append(s)
    cats = {}
    for cat, ss in sorted(by_cat.items()):
        passed = sum(1 for s in ss if s.passed)
        cats[cat] = {"total": len(ss), "passed": passed,
                     "failed": len(ss) - passed}
    total = len(scores)
    passed = sum(1 for s in scores if s.passed)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": (passed / total) if total else 0.0,
        "categories": cats,
    }


def render(scores: list[Score]) -> str:
    """Human-readable table for the terminal."""
    sm = summarize(scores)
    lines = []
    lines.append(f"Eval: {sm['passed']}/{sm['total']} pass "
                 f"({sm['pass_rate']*100:.0f}%)")
    lines.append("")
    lines.append(f"{'category':<12} {'pass':>6} {'total':>6} {'rate':>7}")
    lines.append("-" * 34)
    for cat, c in sm["categories"].items():
        rate = (c["passed"] / c["total"] * 100) if c["total"] else 0.0
        lines.append(f"{cat:<12} {c['passed']:>6} {c['total']:>6} {rate:>6.0f}%")
    failures = [s for s in scores if not s.passed]
    if failures:
        lines.append("")
        lines.append("Failures:")
        for s in failures:
            bad = [c for c in s.checks if not c.ok]
            why = "; ".join(f"{c.name} ({c.detail})" for c in bad)
            lines.append(f"  [{s.category}] {s.case_id}: {why}")
    return "\n".join(lines)
