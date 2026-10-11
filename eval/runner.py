"""Eval runner: load JSONL cases, run against a backend, score results.

stub mode (default): each case carries its own scripted answer + tool
calls under a ``stub`` key — verifies the harness itself, not model
quality. live mode (Phase 3): plugs the real agent backend.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from eval.scorer import Score, score_case

CASES_DIR = Path(__file__).resolve().parent / "cases"


def load_cases(path: Path | None = None,
               category: str | None = None) -> list[dict]:
    """Load JSONL cases; filter by category if given. Validates shape."""
    p = path or (CASES_DIR / "cases.jsonl")
    cases: list[dict] = []
    for ln, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            c = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{p.name}:{ln}: bad JSON: {e}") from None
        if not c.get("id") or "category" not in c or "input" not in c:
            raise ValueError(f"{p.name}:{ln}: case needs id, category, input")
        if category and c["category"] != category:
            continue
        cases.append(c)
    return cases


def run_stub(cases: list[dict]) -> list[Score]:
    """Run scripted stub answers through the scorer (no model)."""
    scores: list[Score] = []
    for case in cases:
        stub = case.get("stub") or {}
        answer = stub.get("answer", "")
        tools = list(stub.get("tools") or [])
        scores.append(score_case(case, answer, tools))
    return scores


def run_live(cases: list[dict]) -> list[Score]:
    """Placeholder — real-model eval lands in Phase 3."""
    raise NotImplementedError("live mode arrives in Phase 3")


def run(cases: list[dict], mode: str = "stub") -> list[Score]:
    if mode == "stub":
        return run_stub(cases)
    if mode == "live":
        return run_live(cases)
    raise ValueError(f"unknown mode: {mode!r} (use 'stub' or 'live')")


def write_results(scores: list[Score], results_dir: Path | None = None) -> Path:
    """Persist a timestamped JSON report under eval/results/."""
    d = results_dir or Path(__file__).resolve().parent / "results"
    d.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = d / f"eval-{ts}.json"
    payload = {
        "ts": ts,
        "total": len(scores),
        "passed": sum(1 for s in scores if s.passed),
        "results": [s.to_dict() for s in scores],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    return out
