#!/usr/bin/env python3
"""Eval CLI:  python3 eval/run.py [--category X] [--stub|--live]

stub (default) verifies the harness with scripted answers;
live POSTs to a running agent server (``POST /api/agent``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.report import render, summarize
from eval.runner import DEFAULT_BASE_URL, load_cases, run, write_results


def _live_token(cli_token: str | None) -> str | None:
    """CLI --token wins; else read the server's auth_token file if present."""
    if cli_token:
        return cli_token
    try:
        from core.auth import configured_token
        return configured_token()
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ai-agent eval harness")
    ap.add_argument("--category", default=None,
                    help="run only this category")
    ap.add_argument("--live", action="store_true",
                    help="real-model mode (needs a running agent server)")
    ap.add_argument("--url", default=DEFAULT_BASE_URL,
                    help=f"agent base URL (default {DEFAULT_BASE_URL})")
    ap.add_argument("--token", default=None,
                    help="X-Agent-Token (default: read auth_token file)")
    ap.add_argument("--timeout", type=float, default=120,
                    help="per-case timeout seconds (default 120)")
    ap.add_argument("--no-save", action="store_true",
                    help="skip writing results JSON")
    args = ap.parse_args(argv)

    mode = "live" if args.live else "stub"
    try:
        cases = load_cases(category=args.category)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if not cases:
        print("error: no cases matched", file=sys.stderr)
        return 2

    if mode == "live":
        from eval.runner import run_live
        scores = run_live(cases, base_url=args.url,
                          token=_live_token(args.token),
                          timeout=args.timeout)
    else:
        scores = run(cases, mode=mode)

    print(render(scores))
    if not args.no_save:
        out = write_results(scores)
        print(f"\nsaved: {out}")
    sm = summarize(scores)
    return 0 if sm["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
