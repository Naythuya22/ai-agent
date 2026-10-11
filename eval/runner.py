"""Eval runner: load JSONL cases, run against a backend, score results.

stub mode (default): each case carries its own scripted answer + tool
calls under a ``stub`` key — verifies the harness itself, not model
quality. live mode: POSTs each case to the real agent server
(``POST /api/agent``) and scores the reply + tool_trace.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from eval.scorer import Score, score_case

CASES_DIR = Path(__file__).resolve().parent / "cases"
DEFAULT_BASE_URL = "http://127.0.0.1:8080"
LIVE_TIMEOUT_S = 120


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


def run_live(cases: list[dict], base_url: str = DEFAULT_BASE_URL,
             token: str | None = None,
             timeout: float = LIVE_TIMEOUT_S) -> list[Score]:
    """POST each case to ``base_url/api/agent`` and score the reply.

    Expects the standard agent JSON: ``{reply, tool_trace, ...}`` where
    tool_trace entries look like ``{type: "call", tool: name, ...}``.
    Connection or HTTP errors score as a failed case (never raises).
    """
    scores: list[Score] = []
    for case in cases:
        payload = json.dumps({"message": case.get("input", "")}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if token:
            headers["X-Agent-Token"] = token
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/agent",
            data=payload, headers=headers, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            scores.append(score_case(
                case, f"[live-error] {type(e).__name__}: {e}", []))
            continue
        if not isinstance(data, dict):
            scores.append(score_case(
                case, "[live-error] bad response shape", []))
            continue
        if data.get("error"):
            scores.append(score_case(
                case, f"[agent-error] {data['error']}", []))
            continue
        answer = data.get("reply") or ""
        tools = [
            t.get("tool", "")
            for t in (data.get("tool_trace") or [])
            if isinstance(t, dict) and t.get("type") == "call"
        ]
        scores.append(score_case(case, answer, tools))
    return scores


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
