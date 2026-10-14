"""Eval runner: load JSONL cases, run against a backend, score results.

stub mode (default): each case carries its own scripted answer + tool
calls under a ``stub`` key — verifies the harness itself, not model
quality. live mode: POSTs each case to the real agent server
(``POST /api/agent``) and scores the reply + tool_trace.

Answer capture (2026-10-11): every run also stores the raw answer +
tools per case in ``_ANSWER_LOG`` (module-level). ``write_results``
merges these into the JSON so criteria can be rebuilt from real data
instead of guessing. See Priority 1 in the handoff.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from eval.scorer import Score, aggregate_scores, score_case

CASES_DIR = Path(__file__).resolve().parent / "cases"
DEFAULT_BASE_URL = "http://127.0.0.1:8080"
LIVE_TIMEOUT_S = 120

# case_id -> {"answer": str, "tools": list[str], "mode": "stub"|"live"}
_ANSWER_LOG: dict[str, dict] = {}


def reset_answer_log() -> None:
    """Clear the module-level answer log (used by tests)."""
    _ANSWER_LOG.clear()


def _log_answer(case_id: str, answer: str, tools: list[str],
                mode: str) -> None:
    _ANSWER_LOG[case_id] = {
        "answer": answer,
        "tools": list(tools),
        "mode": mode,
    }


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
        _log_answer(case["id"], answer, tools, "stub")
        scores.append(score_case(case, answer, tools))
    return scores


def run_live(cases: list[dict], base_url: str = DEFAULT_BASE_URL,
             token: str | None = None,
             timeout: float = LIVE_TIMEOUT_S) -> list[Score]:
    """POST each case to ``base_url/api/agent`` and score the reply.

    Expects the standard agent JSON: ``{reply, tool_trace, ...}`` where
    tool_trace entries look like ``{type: "call", tool: name, ...}``.
    Connection or HTTP errors score as a failed case (never raises).
    Every path logs the raw answer to ``_ANSWER_LOG`` for later review.
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
            answer = f"[live-error] {type(e).__name__}: {e}"
            _log_answer(case["id"], answer, [], "live")
            scores.append(score_case(case, answer, []))
            continue
        if not isinstance(data, dict):
            answer = "[live-error] bad response shape"
            _log_answer(case["id"], answer, [], "live")
            scores.append(score_case(case, answer, []))
            continue
        if data.get("error"):
            answer = f"[agent-error] {data['error']}"
            _log_answer(case["id"], answer, [], "live")
            scores.append(score_case(case, answer, []))
            continue
        answer = data.get("reply") or ""
        tools = [
            t.get("tool", "")
            for t in (data.get("tool_trace") or [])
            if isinstance(t, dict) and t.get("type") == "call"
        ]
        _log_answer(case["id"], answer, tools, "live")
        scores.append(score_case(case, answer, tools))
    return scores


def run(cases: list[dict], mode: str = "stub") -> list[Score]:
    if mode == "stub":
        return run_stub(cases)
    if mode == "live":
        return run_live(cases)
    raise ValueError(f"unknown mode: {mode!r} (use 'stub' or 'live')")


def run_n(cases: list[dict], mode: str = "stub", runs: int = 3,
          **live_kwargs) -> list[Score]:
    """Run each case ``runs`` times; majority verdict per case.

    Live kwargs (base_url, token, timeout) are forwarded to run_live.
    Stub mode is deterministic, so runs>1 just repeats the same result —
    kept for API symmetry. Raises ValueError for runs < 1.
    """
    if runs < 1:
        raise ValueError(f"runs must be >= 1, got {runs}")
    all_scores: list[list[Score]] = []
    for _ in range(runs):
        if mode == "stub":
            all_scores.append(run_stub(cases))
        elif mode == "live":
            all_scores.append(run_live(cases, **live_kwargs))
        else:
            raise ValueError(f"unknown mode: {mode!r} (use 'stub' or 'live')")
    out: list[Score] = []
    for i in range(len(cases)):
        out.append(aggregate_scores([rs[i] for rs in all_scores]))
    return out


def write_results(scores: list[Score],
                  results_dir: Path | None = None) -> Path:
    """Persist a timestamped JSON report under eval/results/.

    Merges raw answers captured in ``_ANSWER_LOG`` into each result so
    criteria can be rebuilt from real model output (see Priority 1).
    """
    d = results_dir or Path(__file__).resolve().parent / "results"
    d.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = d / f"eval-{ts}.json"
    results: list[dict] = []
    for s in scores:
        item = s.to_dict()
        log = _ANSWER_LOG.get(item.get("case_id", ""))
        if log:
            item["answer"] = log["answer"]
            item["tools_called"] = log["tools"]
            item["mode"] = log["mode"]
        results.append(item)
    payload = {
        "ts": ts,
        "total": len(scores),
        "passed": sum(1 for s in scores if s.passed),
        "results": results,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    return out
