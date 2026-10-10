#!/usr/bin/env python3
"""Run the project's unittest suite without touching the real ~/.ai-agent.

Same contract as ``tests/run.sh``, but pure Python so it behaves the same
on the phone (Termux) and on a desktop:

    python3 test.py                 # whole suite, compact output
    python3 test.py -v              # one dot per test, with tracebacks
    python3 test.py test_config.py  # one module
    python3 test.py test_memory.py test_limits.py

Preflight imports flask, because it is the app's only real dependency;
everything else is stdlib or stubbed by ``tests/stubs``.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TESTS = ROOT / "tests"


def _preflight() -> int:
    if not TESTS.is_dir():
        print(f"error: no tests directory at {TESTS}", file=sys.stderr)
        return 2
    try:
        import flask  # noqa: F401
    except ImportError:
        print("error: flask is not importable, the suite needs it.", file=sys.stderr)
        print("       Try:  pip install flask", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str]) -> int:
    bad = _preflight()
    if bad:
        return bad

    # Throwaway HOME + data dir: a failed run can never touch ~/.ai-agent.
    home = Path(tempfile.mkdtemp(prefix="ai-agent-test-home-"))
    os.environ["HOME"] = str(home)
    os.environ["AI_AGENT_DATA_DIR"] = str(home / ".ai-agent")

    # tests/stubs must win over any installed litert_lm, then the project root.
    for extra in (TESTS / "stubs", ROOT):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))

    patterns: list[str] = []
    flags: list[str] = []
    for arg in argv:
        if arg.startswith("-"):
            flags.append(arg)
        elif arg.endswith(".py"):
            patterns.append(arg)
        else:
            patterns.append(f"test_{arg}.py")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for pattern in patterns or ["test*.py"]:
        suite.addTests(
            loader.discover(str(TESTS), pattern=pattern, top_level_dir=str(TESTS))
        )

    # Compact by default: 500 tests of dots bury the tracebacks that matter.
    verbosity = 2 if "-v" in flags else 1
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)

    broken = result.failures + result.errors
    if broken:
        print("\nbroken:")
        for case, _ in broken:
            print(f"  {case.id()}")
    print(
        f"\nran={result.testsRun} failures={len(result.failures)} "
        f"errors={len(result.errors)} skipped={len(result.skipped)}"
    )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    code = 1
    try:
        code = main(sys.argv[1:])
    finally:
        home = os.environ.get("HOME", "")
        if "ai-agent-test-home-" in home:
            shutil.rmtree(home, ignore_errors=True)
    sys.exit(code)
