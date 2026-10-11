#!/usr/bin/env python3
"""Eval CLI:  python3 eval/run.py [--category X] [--stub|--live]

stub (default) verifies the harness with scripted answers;
live calls the real agent backend (Phase 3).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.report import render, summarize
from eval.runner import load_cases, run, write_results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ai-agent eval harness")
    ap.add_argument("--category", default=None,
                    help="run only this category")
    ap.add_argument("--live", action="store_true",
                    help="real-model mode (Phase 3)")
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

    try:
        scores = run(cases, mode=mode)
    except NotImplementedError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3

    print(render(scores))
    if not args.no_save:
        out = write_results(scores)
        print(f"\nsaved: {out}")
    sm = summarize(scores)
    return 0 if sm["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
