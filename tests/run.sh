#!/bin/sh
# Run the whole test suite. No LiteRT-LM, no model weights, no network needed.
#
#   sh tests/run.sh            # everything
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

exec "$PY" -u -m unittest discover -s tests -t tests -v "$@"