#!/bin/sh
# One-command setup for ai-agent.
#
#   sh install.sh              # core install (server + terminal UI)
#   sh install.sh --with-code  # + Code mode deps (node + opencode)
#
# Safe to re-run: every step is idempotent. On Termux it installs the
# system packages itself; elsewhere it assumes python3 + pip exist and
# only installs the Python dependencies.
set -eu
cd "$(dirname "$0")"

WITH_CODE=0
for arg in "$@"; do
    case "$arg" in
        --with-code) WITH_CODE=1 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

say() { printf '\n== %s\n' "$*"; }

if [ -n "${TERMUX_VERSION:-}" ] || [ -n "${PREFIX:-}" ] && case "${PREFIX:-}" in *com.termux*) true ;; *) false ;; esac; then
    IN_TERMUX=1
else
    IN_TERMUX=0
fi

if [ "$IN_TERMUX" = 1 ]; then
    say "Termux detected — installing system packages"
    pkg install -y python git
    if ! python -m pip --version >/dev/null 2>&1; then
        pkg install -y python-pip
    fi
    # Lets the app find models under Download/ (storage permission).
    if command -v termux-setup-storage >/dev/null 2>&1 && [ ! -d "$HOME/storage" ]; then
        say "Requesting storage access (needed for models in Download/)"
        termux-setup-storage || true
    fi
else
    if ! command -v python3 >/dev/null 2>&1; then
        echo "error: python3 not found. Install Python 3.11+ first." >&2
        exit 1
    fi
fi

say "Installing Python dependencies"
python3 -m pip install -r requirements.txt

if [ "$WITH_CODE" = 1 ]; then
    say "Installing Code mode dependencies (node + opencode)"
    if [ "$IN_TERMUX" = 1 ]; then
        pkg install -y nodejs-lts
    fi
    if command -v npm >/dev/null 2>&1; then
        npm install -g opencode-ai
    else
        echo "note: npm not found — install Node.js, then: npm install -g opencode-ai" >&2
    fi
fi

say "Done. Next steps"
cat <<'EOF'
  1. (Optional) put a .litertlm model in ./models or Download/gemmamodel
     — without one the server still starts in cloud-only mode.
  2. Start the server:   python3 server.py
  3. Add provider keys in the web UI under Settings (optional).
  4. Back up your data now and then regularly:
       python3 backup.py export
EOF
