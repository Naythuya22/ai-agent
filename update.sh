#!/bin/sh
# Update ai-agent to the newest version from git (beta testers).
#
#   sh update.sh
#
# Fast-forward only: this never merges, never rebases and never
# touches uncommitted work. If the checkout has modified tracked
# files it stops and says so — an update must never eat a tester's
# own changes. Untracked files (models, caches) do not block it.
set -eu
cd "$(dirname "$0")"

if [ ! -d .git ]; then
    echo "error: this folder is not a git checkout, so update.sh cannot update it." >&2
    echo "       (Zip နဲ့ သွင်းထားတာဆို zip အသစ်ကို ထပ်ပြီး သွင်းပါ။)" >&2
    exit 1
fi

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "error: you have uncommitted changes — update stopped, nothing was changed." >&2
    echo "       Commit or stash them first (git status ကို ကြည့်ပါ), then run update.sh again." >&2
    exit 1
fi

old=$(git describe --tags 2>/dev/null || git rev-parse --short HEAD)

git pull --ff-only

new=$(git describe --tags 2>/dev/null || git rev-parse --short HEAD)

if [ "$old" = "$new" ]; then
    echo "Already up to date: $new"
else
    echo "Updated: $old -> $new"
fi
echo "Restart the server to use it:  python3 server.py   (server ကို ပိတ်ပြီး ပြန်ဖွင့်ပါ)"
