#!/bin/sh
# Run the whole test suite. No LiteRT-LM, no model weights, no network needed.
#
#   sh tests/run.sh            # everything (node UI suites too, when node exists)
#   sh tests/run.sh -k config  # one module: sh tests/run.sh -p "test_config.py"
set -eu

cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"

# Keep the run self-contained: a temp HOME so a failed run cannot touch the
# real ~/.ai-agent, and unbuffered output so a hang is visible immediately.
HOME="${TMPDIR:-/tmp}/ai-agent-test-home.$$"
export HOME
mkdir -p "$HOME"
trap 'rm -rf "$HOME"' EXIT INT TERM

# Preflight: everything except Flask is either stdlib or stubbed. Say so
# plainly instead of failing later inside the test loader.
if ! "$PY" -c "import flask" 2>/dev/null; then
    echo "error: '$PY' cannot import flask." >&2
    echo "       The suite needs the app's only real dependency." >&2
    echo "       Try:  PYTHON=/path/to/python3 sh tests/run.sh" >&2
    echo "       Or:   pip install flask   (already done on the phone)" >&2
    exit 1
fi

status=0

# UI suites are node scripts with their own tiny servers; they used to be
# a separate manual step, which made "run the tests" locally mean only
# half the tests. Run them here when node is available; skip cleanly
# (CI has a dedicated UI job) when it is not.
if command -v node >/dev/null 2>&1; then
    for f in tests/ui_*.js; do
        echo "== $f"
        node "$f" || status=1
    done
else
    echo "note: node not found — skipping UI suites (tests/ui_*.js)"
fi

"$PY" -u -m unittest discover -s tests -t tests -v "$@" || status=1

exit $status