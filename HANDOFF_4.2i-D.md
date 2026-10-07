# 📋 ai-agent — Session Handoff (Part 4.2i-D Series)

Date: 2026-10-07
Repo: /data/data/com.termux/files/home/storage/shared/Opencode agent/ai-agent (1)
HEAD at write: 1da670d (user-added "# test 4" comment)

---

## 🎯 Session Objective

Continue the Batch 4.2i series after the 4.2i-B / 4.2i-C handoff.
Fix the audit HIGH cluster (H4+H5), then the Code Mode findings
from an in-app audit: silent git failure, sig11 shell crash, lock
coverage gap, stream timeout, project mis-registration. Then move
Code Mode from Level 0 (fire-and-pray) toward Level 2 (Trust) with
a diff viewer and a rollback button.

---

## 📊 Current State — 15+ Commits This Session

| Commit  | Title                                              | Tests |
|---------|----------------------------------------------------|-------|
| a0465ee | 4.2i-C  — H4 /v1 inference_guard + H5 swap guard   | 769   |
| 5014751 | 4.2i-D1 — surface git commit failures with reasons | 772   |
| 40f4ddb | 4.2i-D2 — strip LD_PRELOAD from child env          | 775   |
| a42e3cc | 4.2i-D3 — serialise one-shot opencode runs         | 777   |
| e9372e1 | 4.2i-D4 — timeout the streamed opencode run        | 780   |
| 1a6baa4 | tune    — raise streamed timeout for MiMo          | 780   |
| 87d9ee8 | 4.2i-E1a — diff endpoint                           | 782   |
| b79bf56 | 4.2i-E1b — rollback endpoint                       | 784   |
| 5cdf731 | 4.2i-E1c — always-on Burmese summary               | 785   |
| 5450c3b | 4.2i-E2a — openDiffViewer() helper                 | 785   |
| 8428ffd | 4.2i-E2b — View diff button on commits             | 785   |
| 160edcf | 4.2i-E2c — Rollback button on commits              | 785   |
| 5f7fddb | 4.2i-E2d — keep HTTP status in rollback promise    | 785   |
| 7f642ce | 4.2i-E2e — View diff shows the commit              | 787   |
| b3351eb | 4.2i-E2f — openDiffViewer actually shows the diff  | 788   |
| abab905 | (user) — "# test 2" README test commit            | —     |
| be40434 | (user) — "## Test section" README test commit     | —     |
| f5f0cc7 | (user) — "# test 3" server.py test commit         | —     |
| 1da670d | (user) — "# test 4" server.py test commit         | —     |

Test suite: **788 / 788 OK**. Working tree clean.

Server: running after every change, PID changes each restart
(Lesson #3). Last known PID: 1542 (kill + nohup python server.py).

---

## ✅ Completed & Verified

### 4.2i-C — Audit HIGH cluster (a0465ee)
- H4: `/v1/chat/completions` now holds `inference_guard()` around
  `engine.create_conversation` + `conv.send_message`
  (routes/openai_compat.py L52-70). Previously opencode bridge and
  /api/chat could drive the native engine at once → process drop.
- H5: session swap sites (create / delete / bulk_delete / activate
  / /api/new) now take `_session_swap_guard()` — a new context
  manager in routes/sessions_api.py that takes `agent_turn_lock`
  then `inference_lock`, matching the documented order in
  routes/_state.py. Previously reload_session()/reset() could free
  a Conversation mid-generation → segfault.
- Tests: 5 new — 2 in tests/test_openai_guard.py, 3 in
  tests/test_session_guard.py.

### 4.2i-D1 — Silent git failure (5014751)
- commit_changes() in routes/code.py now returns
  `{ok: bool, hash, message}` on success and
  `{ok: False, reason}` on failure — reason strings for "not a git
  repository", "git add failed: ...", "nothing staged",
  "git commit failed: ...", "git timed out", "<Class>: <msg>".
- Three callers (api_code, api_code_web_run, stream generate())
  surface a new `commit_error` field / SSE event.
- Tests: 3 new — TestCommitChanges in tests/test_code.py.

### 4.2i-D2 — LD_PRELOAD strip (40f4ddb)
- Root cause of "selfchecker: checker 'remold.bash' detects sig11"
  was the opencode-termux wrapper exporting LD_PRELOAD globally
  (musl loader into a bionic child). Manual unset in the wrapper
  fixed it live, but the wrapper auto-updates.
- core/opencode.py env() now pops LD_PRELOAD from the returned dict
  (L150-157) so it never reaches a child, whatever the wrapper
  does. LD_LIBRARY_PATH is deliberately kept.
- Tests: 3 new — TestEnvChildEnvironment in tests/test_code.py.

### 4.2i-D3 — Lock coverage (a42e3cc)
- api_code() and api_code_web_run() now take `code_lock` around
  their run + commit + remember sequence; previously they bypassed
  the lock the streamed route already took, so a one-shot run and a
  streamed run could edit the same files at once.
- Tests: 2 new — TestOneShotRunsHoldCodeLock.

### 4.2i-D4 — Stream timeout (e9372e1)
- core/code_stream.py stream_opencode() now wakes on select() with
  a 1s poll and kills the child after STREAM_IDLE_TIMEOUT (silence)
  or STREAM_HARD_TIMEOUT (total). Yields an error frame with a
  `reason` (idle_timeout / hard_timeout) before the usual end
  frame. GeneratorExit, finally: _terminate, parse_event and the
  callbacks unchanged.
- Tuned values (1a6baa4): IDLE = 300, HARD = 1200. MiMo has silent
  thinking phases of 60-150s between tool calls, and multi-edit
  tasks run 5-15 minutes.
- Tests: 3 new — TestStreamTimeouts.

### 4.2i-E1a / E1b — Backend endpoints (87d9ee8, b79bf56)
- GET /api/code/diff?scope=workdir|staged|head — unified diff
  capped at DIFF_CAP (200 KB), stat parsed from git diff --numstat.
  Not a repo → {ok:false, reason: "not a git repository"} (200).
- POST /api/code/rollback {"confirm": true, "count": 1} — runs
  `git reset --hard HEAD~N` for N in 1..10. confirm must be the
  literal boolean True; count must be a real int (bool rejected).
  Only HEAD~N is ever passed to git.
- Shared helper: `_git(workdir, args, timeout=15)` returns
  (rc, stdout, stderr), never raises — 124 for timeout, 127 for
  git-not-found.
- Tests: 2 new for diff, 2 new for rollback.

### 4.2i-E1c — Always-on Burmese summary (5cdf731)
- build_code_prompt() now appends BURMESE_SUMMARY_DIRECTIVE after
  LANG_DIRECTIVE, so every Code Mode reply ends with a
  "## မြန်မာလို အကျဉ်း" section (4 bullets: what changed, why,
  warnings, test count). Code terms stay in English.
- Tests: 1 new — TestBuildCodePrompt.

### 4.2i-E2a-E2f — Frontend (5450c3b .. b3351eb)
- openDiffViewer(title, diffText) — reuses #viewer overlay, renders
  each line as a span with class diff-add / diff-del / diff-hunk.
- fetchDiffAndShow(label, commitHash) — fetches
  /api/code/diff?commit=<hash> and shows it.
- askAndRollback(label) — window.confirm warning, POSTs
  {confirm:true,count:1} to /api/code/rollback, renders system or
  error message.
- Commit bubble now carries View diff + Rollback buttons (danger
  color #a33 for Rollback, accent green for View diff).
- Bug fix E2e (7f642ce): the button used to pass ?scope=head,
  which is empty right after a commit. It now passes evt.hash, and
  the endpoint gained a commit=<hash> mode using git show.
- Bug fix E2f (b3351eb): openDiffViewer() unhid #viewer but not
  #viewer-text, so the overlay opened to an empty box. It now
  unhides the pane and hides the img pane for parity with
  openTextViewer.
- Tests: 1 new — TestWebDiffViewer (reads web/index.html).

### Manual fixes (outside commits)
- Wrapper: `/data/data/com.termux/files/usr/lib/node_modules/
  opencode-termux/bin/opencode` L39 changed from
  `export LD_PRELOAD=...` to `unset LD_PRELOAD  # sig11-fix: ...`
  — Termux's own termux-exec preload was also crashing the musl
  binary.
- ripgrep: `pkg install ripgrep` → 15.2.0.
- Project registry: ~/.ai-agent/code_projects.json — added a new
  entry `ai-agent -> .../Opencode agent/ai-agent (1)` and set it
  active. The old entry pointed at the parent folder, so
  `pwd` in Code Mode was `/storage/emulated/0/Opencode agent`.

---

## 🔴 Open Items (Priority Order)

### HIGH — Live bugs found this session, not yet fixed

| #  | Bug                                                            | Location                |
|----|----------------------------------------------------------------|-------------------------|
| B1 | Service worker serves old index.html after a change → user     | web/sw.js               |
|    | fixes on the server are invisible until site data is cleared   |                         |
| B2 | Code Mode reload leaves "Loading code..." stuck; chat history  | web/index.html:674      |
|    | never reappears because line 674 only calls loadProjects()     |                         |
|    | and not loadHistory()                                          |                         |
| B3 | _code_history is in-memory only — every server restart wipes   | routes/code.py          |
|    | the Code Mode chat list, so old commit bubbles (and their      |                         |
|    | View diff buttons) cannot be recovered                         |                         |

### MEDIUM — Audit findings from prior handoff, still open

| #  | Bug                                          | Location              |
|----|----------------------------------------------|-----------------------|
| M1 | Non-dict sessions.json → permanent 500       | sessions.py:55        |
| M2 | Non-dict memory.json → all mode 500s         | memory.py:61-73       |
| M3 | Shared .tmp on save (race)                   | sessions.py:108,      |
|    |                                              | memory.py:77,         |
|    |                                              | config.py:447         |
| M4 | SSE local tool loop feeds only last result   | routes/chat.py:604-609|
| M5 | (fixed — see 4.2i-D3)                        | —                     |
| M6 | (fixed — see 4.2i-D4)                        | —                     |
| M7 | CLI dispatch outside try/except              | app.py:782            |
| M8 | config.py _litertlm_files TOCTOU sort        | config.py:388         |
| M9 | Non-ASCII token → 500                        | core/auth.py:107-108  |
| M10| Idle watchdog exits mid-use                  | routes/safety.py      |

### MEDIUM — Code Mode findings still open

- `--auto` always on (build_command auto=True) — no preview/confirm
  before file edits.
- ttyd lifecycle: server exit leaves the ttyd child running; no
  wait()/reap on exit.
- MODEL_ID hardcoded to "mimo-v2.6-flash-free" (core/opencode.py:28);
  no env override.
- `/api/code` default workdir (`~/opencode-work`) differs from the
  web route's default (`_resolve_workdir` → shared storage).
- Rollback count is hardcoded to 1 in the UI; API accepts 1..10.
- Rollback success appends a system message but does not refresh
  the commit bubble list.
- window.confirm() is native; unspoofable but unstyled.

### DEFERRED — 4.2i-c (_KNOWN_GAPS)

tools/audit.py:14-15 is stale. Evidence in NOTES_4.2i-b.md.
Recommended: split multi_modal_input into cloud_vision_input +
audio_input; fix docstring L157-158 (stale: parallel_tool_calls).

### LIVE — Bug A (trailing duplicate fragment)

Seen in UI: a book-summary response ended cleanly, then re-emitted
a prior paragraph cut off mid-word. 3 hypotheses (H1 verify
corruption, H2 model re-emit, H3 frontend buffer). Diagnostic not
run yet. Still open from prior handoff.

### LOW (L1–L9)

Off-by-one in read_file; empty assistant on exhausted max_iter;
turns drift; menu hint wrong; exit code discarded; /v1 not
incremental; ttyd never reaped; client disconnect skips reflection.

---

## 🧠 Key Lessons — Cumulative

| #  | Lesson |
|----|--------|
| 1-9 | (see prior handoffs) |
| 10 | Handoff docs drift fast. Cross-check HEAD, git log, disk markers. |
| 11 | Reviewer pass can corrupt tool-grounded answers. Skip verify when nudge-fired. Structural > prompt rule. |
| 12 | Test helper sentinels. Use `_SENTINEL = object()`, not None. |
| 13 | PyEOF test count. A class with N methods = N tests. Count precisely. |
| 14 | Code Mode's shell tool can crash with sig11 (selfchecker "remold.bash"). Root cause: LD_PRELOAD leak (either direction: musl→bionic or bionic→musl). Manual wrapper fix + app-side env() strip both needed. |
| 15 | MiMo-V2.6-Flash Free has long silent thinking phases (60-150s) between tool calls. STREAM_IDLE_TIMEOUT must be >= 300; STREAM_HARD_TIMEOUT >= 1200 for multi-edit tasks. 90/300 was too tight. |
| 16 | Service worker caches web/index.html. After any frontend change the user must clear site data (chrome://settings/siteData → 127.0.0.1 → Delete) or use a fresh browser context, otherwise the server ships new code and the user sees old behaviour. P1 fix: cache-bust by content hash. |
| 17 | The `_run` contract is (returncode, lines), not (lines, returncode). Test fakes must return (0, []) — the reverse raises TypeError: 'int' object is not iterable. |
| 18 | Frontend closure bug pattern: `var` inside a for loop gives ONE binding; a click handler that closes over it reads the LAST frame's value. Use `let` when the handler needs this frame's value (sendStream's evt was exactly this). |
| 19 | `git diff HEAD` is empty right after a commit — the working tree equals HEAD. To show "what did this commit change", use `git show <hash> --format= --patch`. The View diff button needed the latter. |
| 20 | Frontend overlay visibility is per-element: unhiding the outer container is not enough if a nested pane (e.g. #viewer-text) is also hidden. openTextViewer and openDiffViewer must both set viewer-text.hidden=false. |

---

## ⚠️ Cautions for Next Session

- AGENT_SYSTEM is ~1083 tokens / 8 sections — near budget. Merge
  before adding.
- prompts/agent_cloud_fast.txt — 3 tokens headroom. Trim before add.
- Memory cleaned 4x in the prior session. Watch for new
  self-deprecating facts.
- Server has died 2+ times mid-session. `ps aux | grep "python
  server.py" | grep -v grep` after every change.
- PyEOF scripts must handle dict-shaped sessions (not list).
- _VERIFY_SYSTEM (modes/agent.py:173) is small — high leverage.
- The audit could NOT execute tests (bash broken for the auditor).
  Re-verify with `sh tests/run.sh` locally.
- The opencode-termux wrapper auto-updates on every run. The manual
  `unset LD_PRELOAD` fix will disappear on the next `npm install`;
  the app-side pop in core/opencode.py is the durable defense.
- opcode-termux wrapper is at
  `/data/data/com.termux/files/usr/lib/node_modules/opencode-termux/
  bin/opencode`; backup at `<wrapper>.bak.sig11-<date>`.
- CODE Mode runs inside proot-distro ubuntu. Its `bash -lc` inherits
  the parent env — LD_PRELOAD from Termux would still leak if the
  app-side pop were ever removed.
- Code Mode's project registry is
  `~/.ai-agent/code_projects.json`. There are now TWO projects: the
  old parent folder (`/storage/emulated/0/Opencode agent`, id
  8512b1ae) and the repo itself (id 22e46387). ACTIVE is 22e46387.

---

## ✅ Quick Resume Commands

    cd "/data/data/com.termux/files/home/storage/shared/Opencode agent/ai-agent (1)"

    # State check
    git log --oneline -8
    git status --short
    ps aux | grep "python server.py" | grep -v grep

    # Tests
    sh tests/run.sh 2>&1 | tail -3
    # Expected: Ran 788 tests ... OK

    # If server down:
    AI_AGENT_LOG_LEVEL=DEBUG nohup python server.py > ~/ai-agent.log 2>&1 &
    sleep 15
    ps aux | grep "python server.py" | grep -v grep

    # Active code project (should print 22e46387):
    python3 -c "import json,os; print(json.load(open(os.path.expanduser('~/.ai-agent/code_projects.json')))['active'])"

---

## 🎯 Next Concrete Action (Pick One)

Option 1 (recommended) — B1 + B2 (frontend P1 cluster)
- B1: Service worker cache invalidation for web/index.html.
  Either hash the file content into sw.js's cache key, or set a
  short max-age with a revalidate; the current sw.js serves stale
  HTML after a deploy.
- B2: web/index.html:674 — the Code mode switch calls loadProjects()
  but not loadHistory(); add the call so the commit bubbles come
  back after a reload.
- Read routes/pwa.py for _VERSION_FILES (which feeds the SW cache
  name) and web/sw.js for the cache strategy.

Option 2 — B3 (_code_history persistence)
- routes/code.py:66 in-memory list. Persist to a JSONL file under
  ~/.ai-agent/ (per-project), or rebuild from `git log` on load.
  Without this, every server restart wipes the commit list that
  the View diff buttons depend on.

Option 3 — Level 3a (project auto-detect)
- First step of the test-aware loop: detect pytest / npm / cargo
  from the project tree and expose the command to the run.
- Read routes/code.py's build_code_prompt and _resolve_workdir.

Option 4 — M-batch (atomicity cluster)
- M1 + M2 + M3: isinstance guard + per-store lock + unique tmp.
- Reference: core/key_manager.py.

Option 5 — Bug A (trailing fragment)
- Still open from prior handoff.
- Run the 5-command diagnostic.

Option 6 — 4.2i-c (_KNOWN_GAPS)
- Evidence ready in NOTES_4.2i-b.md.
- 1 tuple + 1 docstring change, plus tests/test_audit.py (0 coverage).

Option 7 — Code Mode --auto opt-in
- build_command(auto=True) — no confirm/preview before edits.
- Add a UI toggle and pass auto=False by default, or add a preview
  step.

Verify protocol (for any option):
1. Read source at exact offsets — no guesses.
2. PyEOF with idempotency marker + auto-backup + py_compile guard.
3. Post-Writing Audit x3 (syntax / logic / security).
4. Run tests: `sh tests/run.sh 2>&1 | tail -3`.
5. Restart server (Lesson #3).
6. UI verify (browser refresh; if frontend changed, clear site
   data first — Lesson #16).
7. Commit with the exact test count (Lesson #13).

---

## 📈 Session Metrics

- Commits: 15 in-repo + 1 tune + 4 user test commits = 20 total
- Tests added: 24 (764 → 788)
- Files touched: routes/openai_compat.py, routes/sessions_api.py,
  routes/code.py, core/opencode.py, core/code_stream.py,
  web/index.html, tests/test_code.py, tests/test_openai_guard.py,
  tests/test_session_guard.py
- Level: Code Mode moved Level 0 (fire-and-pray) → Level 2 (trust:
  diff + rollback + visible failures + timeouts + locks).

---

End of handoff.
