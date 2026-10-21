"""Generate paste-able PYEOF apply scripts for Releases 1-4.

Each script embeds the exact gate-tested patch (base64) plus the
sha256 fingerprints of every touched file before and after the release.
On the phone it: locates the repo, refuses to run unless the current
state matches the expected "before" state exactly (or reports the
release as already applied), runs git apply --check then git apply,
and verifies the "after" state file by file.
"""
import base64
import hashlib
import subprocess
from pathlib import Path

REPO = Path.home() / "workspace" / "ai-agent-upgrade"
OUT = Path.home() / "workspace" / "your_files"

RELEASES = [
    (1, "3771203", "bb869d1", "ai-agent-release-1.patch"),
    (2, "bb869d1", "v1.2.0", "ai-agent-release-2.patch"),
    (3, "v1.2.0", "v1.2.1", "ai-agent-release-3.patch"),
    (4, "v1.2.1", "v1.3.0", "ai-agent-release-4.patch"),
]


def git(*args) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, check=True).stdout


def fingerprints(rev: str, files: list[str], allow_missing: bool) -> dict:
    out = {}
    for f in files:
        r = subprocess.run(["git", "-C", str(REPO), "show", f"{rev}:{f}"],
                           capture_output=True)
        if r.returncode != 0:
            if not allow_missing:
                raise RuntimeError(f"{rev}:{f} missing")
            out[f] = None  # file is created by this release
        else:
            out[f] = hashlib.sha256(r.stdout).hexdigest()[:12]
    return out


TEMPLATE = '''python3 << 'PYEOF'
# ai-agent Release {n} — apply script. Paste everything into Termux.
# It checks first and changes NOTHING if the state does not match.
import base64, hashlib, subprocess, sys
from pathlib import Path

PATCH_B64 = (
{b64}
)
PRE = {pre}
POST = {post}

def h(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]

def mismatches(root, table):
    bad = []
    for f, want in table.items():
        p = root / f
        if want is None:
            if p.exists():
                bad.append(f)
        elif not p.exists() or h(p) != want:
            bad.append(f)
    return bad

root = None
for c in [Path.cwd(), Path.home() / "ai-agent"]:
    if (c / "server.py").exists() and (c / ".git").exists():
        root = c
        break
if root is None:
    print("STOP: repo not found (looked in the current folder and ~/ai-agent). Nothing changed.")
    sys.exit(1)
print("Repo:", root)

post_bad = mismatches(root, POST)
if not post_bad:
    print("Release {n} is ALREADY applied - nothing to do.")
    sys.exit(0)
pre_bad = mismatches(root, PRE)
if pre_bad:
    print("STOP: these files do not match the expected state before Release {n}:")
    for f in pre_bad:
        print("  -", f)
    print("Nothing was changed. Please send this output back.")
    sys.exit(1)

pf = root / ".release-{n}.patch.tmp"
pf.write_bytes(base64.b64decode(PATCH_B64))
try:
    chk = subprocess.run(["git", "apply", "--check", str(pf)],
                         cwd=root, capture_output=True, text=True)
    if chk.returncode != 0:
        print("STOP: the patch does not apply cleanly. Nothing was changed.")
        print(chk.stdout, chk.stderr)
        sys.exit(1)
    r = subprocess.run(["git", "apply", str(pf)],
                       cwd=root, capture_output=True, text=True)
    if r.returncode != 0:
        print("STOP: apply failed.")
        print(r.stdout, r.stderr)
        sys.exit(1)
finally:
    pf.unlink(missing_ok=True)

bad = mismatches(root, POST)
if bad:
    print("WARNING: applied, but these files differ from the expected result:")
    for f in bad:
        print("  -", f)
    print("Please send this output back.")
    sys.exit(1)
print("OK: Release {n} applied and verified file by file.")
print("Next step:  sh tests/run.sh 2>&1 | tail -3")
PYEOF
'''


def wrap_b64(data: bytes) -> str:
    s = base64.b64encode(data).decode()
    lines = [s[i:i + 76] for i in range(0, len(s), 76)]
    return "\n".join(f'    "{line}"' for line in lines)


for n, frm, to, patch_name in RELEASES:
    files = git("diff", "--name-only", frm, to).decode().split()
    pre = fingerprints(frm, files, allow_missing=True)
    post = fingerprints(to, files, allow_missing=False)
    patch = (OUT / patch_name).read_bytes()
    script = TEMPLATE.format(n=n, b64=wrap_b64(patch), pre=pre, post=post)
    target = OUT / f"apply-release-{n}.txt"
    target.write_text(script, encoding="utf-8")
    print(f"Release {n}: {len(files)} files, script {len(script)} chars -> {target}")
