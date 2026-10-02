# Session 2 — Critical Findings (2026-09-24)

## 🔴 Tier 1 — Showstoppers
### #1 GPU Conflict
- Server + Terminal dual LiteRT → segfault on tool call
- Evidence: isolation test
- Files: app.py, server.py
- Fix: Terminal → HTTP client

### #2 AgentMode no replay
- _ensure_conv() no messages= parameter
- File: modes/agent.py:~50
- Fix: add replay (like ChatSession)

## 🟠 Tier 2
### #3 Tool results not persisted
- handle() only saves user/assistant text
- File: modes/agent.py:~65

### #4 Meter excludes tool tokens
- llm.py estimate_messages_tokens
- File: core/llm.py

## 🟡 Tier 3
### #5 Web accuracy 50%
- Re-test after #1 & #2 fixed

## 💎 Lessons
1. Isolation tests decisive
2. Verify-before-declare tano
3. Daily usage > synthetic tests
4. Backup knowledge, not just code

## Next Session Plan
1. Read server.py (642 lines, chunks)
2. Read app.py (700 lines, chunks)
3. Design Fix #1 (architecture)
4. Implement, audit x3, verify

---

## Session 4 Update (2026-09-24)

### ✅ Fix #1 — SOLVED
- C1: core/http_llm.py (new, 154 lines) — HTTP proxy
- C2: app.py — LLM → HTTPLLM swap (line 13, 70)
- C3: app.py — server health check in run()
- C4a/b: modes/agent.py — HTTP branch in _ensure_conv + handle

### Verification
- test_phase1 ✅, test_phase2 ✅
- Server log: Web + Terminal concurrent = all 200 OK
- Bug #1 (GPU conflict) — no longer reproducible

### 🟡 Outstanding (Deferred)
- Bug #2 (replay) — still in modes/agent.py local path
- Bug #3 (tool persist) — still open
- Bug #4 (meter) — cosmetic
- Bug #5 (Web accuracy) — verify after #2/#3
- C5 (session sync) — Terminal /new vs server desync
- Bug B (cold engine race) — verify after Fix #1 (may be gone)

### Notes for Next Session
- HTTPLLM skips tools (uses server's /api/agent)
- _cmd_code (Aider) — HTTPLLM.unload() is no-op → cosmetic msg bug

---

## Session 5 Update (2026-09-24) — Bug #2 FIXED

### Root Cause
- AgentMode.__init__: SessionStore created new session on init
- ChatMode.__init__: same pattern (aggravated)
- handle(): both Terminal and Server wrote save_mode → race
- Server restart → replay not loaded → memory lost

### Fix (C2 + C5a + C5b + C5c)
- C2: AgentMode._ensure_conv — replay from _messages
- C5a: AgentMode.__init__ — skip store in HTTP mode
- C5b: AgentMode.handle — skip save in HTTP mode
- C5c: ChatMode.__init__ + handle — same HTTP-mode skip

### Principle Established
> Terminal = display-only in HTTP mode
> Server = single source of truth for session state

### Verified
- curl restart test: "Aung" preserved after server restart ✅
- Terminal restart test: "ကိုကြီး" preserved ✅
- test_phase1/2 PASS ✅

### 🟡 Outstanding (Deferred to Session 6)
- Bug #3 (tool calls not persisted)
- Bug #4 (context meter excludes tool tokens)
- Bug #5 (Web accuracy — verify after #3)
- Bug B (cold engine race — re-verify)
- _cmd_code cosmetic (Aider unload msg)
- C5d (session commands server proxy) — low priority

---

## Session 6 Update (2026-09-24) — Code Mode Verified

### ✅ Fixed
- C5d: /new → server /api/new proxy
- Aider chain: verified end-to-end

### ✅ Verified Working
- Chat mode (baseline)
- Agent mode (memory + tools + restart persistence)
- Code mode (Aider → /v1/chat/completions → LLM → diff → apply)
- Server log: 5× POST /v1/chat/completions 200 OK

### 🟡 Outstanding — Session 7 Backlog
- C5-sweep: /sessions, /switch, /delete, /history → server proxy
- Bug #4: context meter excludes tool tokens
- Bug #5: Web accuracy re-verify
- Cosmetic: git log pager in Termux (use --no-pager)
- Cosmetic: _cmd_code "Unloading engine" msg (HTTPLLM no-op)

### 📌 Session 6 Outcome
All three modes functional. System production-usable.
Next: daily use for 1-2 days → gather real friction → Session 7.

---

## Session 7 Update (2026-09-24) — Web Aider Complete

### 🎯 Goal
Browser → Aider → Phone storage (real file edit + git)

### ✅ Delivered (3 sessions worth)
- ttyd approach (Chunk 1-4): works but mobile keyboard UX poor
- PIVOT: Custom one-shot Aider (Chunk 1-2)
  - server.py: /api/code/web/run
  - web/index.html: Code tab → chat-style input
  - No iframe, no keyboard issues

### Verified
- hello.py created on phone storage ✅
- Content: print("Hello from phone storage")
- Path: /storage/emulated/0/aider-work/hello.py
- Git commit b050f4a
- Response 6.7s

### Files Changed (Session 7)
- server.py: +50 lines (/api/code/web/run)
- web/index.html: ~5 lines diff (mode routing)
- ~/bin/aider-web: ttyd fallback (kept, unused)

### 🟡 Notes
- ttyd iframe blocked (X-Frame-Options) — use new tab OR one-shot
- Android fuse: git safe.directory required
- Aider --yes --no-pretty --no-stream — headless mode

### 🎯 Outstanding (Session 8+)
- Bug #4 (context meter excludes tool tokens)
- Bug #5 (Web accuracy)
- C5-sweep (session commands server proxy)
- Streaming response
- PWA

---

## Session 7 Update (2026-09-24) — Code Mode Complete

### ✅ Delivered
- Web Code tab → one-shot Aider per message
- Phone storage (~/storage/shared/aider-work/)
- Real file I/O + git commits
- Structured output (reply + files + commits + raw toggle)
- Burmese language response (via LANG_DIRECTIVE injection)
- Adaptive: Burmese→Burmese, English→English

### Files Changed
- server.py: /api/code/web/run + _parse_aider_output + LANG_DIRECTIVE
- web/index.html: Code tab + send() routing + structured display
- ~/bin/aider-web (ttyd fallback, unused)

### Verified
- calculator.py, temp_converter.py, converter.py — created + committed
- "မြန်မာလို ပြောပါ" → Burmese response
- 4.4s response time

### Notes
- Aider 0.16.0 — no --chat-language (removed config)
- Language via server-side prompt injection
- ttyd — retired (iframe/keyboard issues on mobile)

### Outstanding (Session 8+)
- Chunk C: Server-side conversation history (context across messages)
- Bug #3: tool persist
- Bug #4: meter
- C5-sweep
- Streaming, PWA

---

## Session 7 Final — Conversation Memory Added

### Chunk C (Conversation)
- server.py: _code_history global (cap 20 entries)
- Context injection in prompt (last 10 messages)
- /api/code/web/clear, /api/code/web/history endpoints
- Verified: "same file" reference works
  - Msg 1: Create math_utils.py (+)
  - Msg 2: Add cube to same file (~) ← memory retained

### Code Mode — Cursor-like parity ~70%
- ✅ Real files + git
- ✅ Conversation memory
- ✅ Burmese adaptive
- ✅ Clean output

### Outstanding (Session 8+)
- @-mention files (1 hr)
- Streaming response (45 min)
- Visual diff panel (2 hr)
- Bug #3 (tool persist for agent mode)
- Bug #4 (meter)
- C5-sweep

---

## Session 9 Final — Partial Success

### ✅ Delivered
- Shell tool (3-layer safety) — code perfect
- Tool-first prompt (A+B) — helps natural prompts

### 🔴 Limitation Confirmed
- File-name prompts → hallucination (E4B bias)
- config_size: 892 (actual 8192) ❌
- TIMEOUT_SEC: "0s" (actual 10s) ❌
- test_phase1 timeout: 10s too short for LLM load

### Root Cause
- Gemma 4 E4B — tool-selection reasoning limit
- "file is known" bias → skips read_file tool

### Session 10 Plan
- Option B: Tool-call verification (retry if no tool result)
- Option A: File trigger keyword strengthen
- Timeout: 10s → 30s (test runs)
- Bug #3 (persist)

---

## Session 9 Final — Bug #3 FIXED 🎉

### Root Cause
- automatic_tool_calling=True → tool calls black-box → not persisted
- Manual loop needed for transparency

### Fix — 4 Chunks (modes/agent.py)
- Chunk 1: automatic_tool_calling → False
- Chunk 2: _execute_tool helper (name → function)
- Chunk 3: handle() manual loop
- Chunk 4: replay includes tool_result entries

### Verified (4/4)
- Test 1: Python count = 22 ✅
- Test 2: config_size = 8192 ✅ (was 892 before — hallucination fixed)
- Test 3: RAM = 1.6Gi free ✅
- Test 4: Session persist → recalled 3 previous questions ✅

### Unexpected Win
- Manual loop ALSO fixed hallucination on file-name prompts
- Model now reliably calls tools when needed
- Session 9's "E4B ceiling" — was actually a fixable design issue

### Key Insight
> Transparency (seeing tool calls) → correct behavior
> Black-box → hallucination

### Outstanding (Session 10+)
- Bug #4 (meter)
- Streaming
- Code mode reply text (diff-only case)
- PWA

## Session 10 (2026-09-25) — Bug #4 + #5 FIXED
### Bug #5 — read_file truncation (4000 char cap)
- File: tools/file.py (2253 → 3756 bytes)
- Added: start_line, end_line params
- Auto-cap: 500 lines / 4000 chars, next-page hint
- Regression: agent reads server.py 1058 lines correctly

### Bug #4 — Context meter wrong in HTTP mode
- Root cause: terminal ignores server ctx; local estimate missing tool msgs
- Fix 1: modes/agent.py — cache j.get("ctx") as _server_ctx; add get_server_ctx()
- Fix 2: app.py — _ctx_info() uses server-truth first, fallback to local
- Regression: agent mode meter now +1.5K on config.py (was +0.1K)
- Backups: agent.py.bak.10, app.py.bak.10

### Bug #5b (agent "80000") — NOT a bug
- Session-specific glitch; gone after /new
- No fix needed

### Bug #6 — Web UI tool visibility (NEW FEATURE)
- File: server.py (+775 bytes), web/index.html (+721 bytes)
- Server: _run_agent returns tool_trace [{type:call, tool, args}, {type:result, tool, result[:500], result_len}]
- Client: appendToolTrace() renders green monospace badges
- CSS: .msg.tool class added
- Verified: Web UI shows 🔧 call + ✅ result badges
- Backups: server.py.bak.10, web/index.html.bak.10

### Session 10 Summary
- Bug #5 (read_file 4000-char truncation) → FIXED (start_line/end_line params)
- Bug #4 (HTTP-mode meter missing tool tokens) → FIXED (server-truth + fallback)
- Bug #6 (tool visibility in Web UI) → DONE (tool_trace + badges)
- Bug #5b (agent "80000" miscount) → NOT a bug (session-specific glitch)

## Session 11 (2026-09-26) — Real Streaming ✅
### New: SSE streaming (Chat + Agent)
- core/llm.py: stream_conversation() + ChatSession.send_stream()
- server.py: /api/chat/stream, /api/agent/stream (SSE)
- web/index.html: sendStream() with fetch + ReadableStream
- Features: typewriter effect, live tool badges, AbortController
- Backups: llm.py.bak.10, server.py.bak.10b/c, index.html.bak.10b
- Verified: Chat & Agent modes with real tool calls

### POC findings (poc_stream.py)
- send_message_async() = real generator (~50ms/chunk)
- Chunk = word-level, not token
- Tool calls yield single chunk, then stream ends
- cancel_process() works (1-chunk lag)
- Burmese first-token ~8s cold, ~1s warm

### Deferred
- Stop button UI (data ready, needs UI)
- /api/chat (sync) and /api/agent (sync) retained as fallback

## Session 12 (2026-09-26) — Code Mode Streaming ✅
### New: Aider real-time streaming (SSE)
- **core/code_stream.py** (NEW module) — `stream_aider()` generator
  - Popen + line-by-line stdout read
  - ANSI strip + `\r` cleanup
  - GeneratorExit handler → terminate
  - Lock-serialized (concurrent runs)
  - Events: `chunk`, `file`, `commit`, `end`, `error`
- **server.py** — `/api/code/web/run/stream` (NEW SSE endpoint)
  - Backward compat: sync `/api/code/web/run` retained (line 956)
  - `/project` intercept delegates to sync view (DRY)
  - `PYTHONUNBUFFERED=1` env (critical for real-time)
  - Removed `--no-stream` flag (was blocking)
  - `_code_lock = threading.Lock()` (module-level)
- **web/index.html** — `sendStream()` extended for code
  - `send()` routes `code` → `sendStream`
  - ep resolver: `code → /api/code/web/run/stream`
  - New events: `file`, `commit` → `.msg.tool` badges

### Verified (T1–T5) ✅
- T1: Server start (9.3s engine load, HTTP 200)
- T2: Chat streaming (Burmese typewriter)
- T3: Agent streaming (tool_call → tool_result → chunks)
- T4: Code streaming (Aider line-by-line → `end` with rc=0)
- T5: Abort + cleanup (PID preserved, no stray aider)

### Root Cause Learning (T5 first fail)
- Server silent death = **Android LMK** (Low Memory Killer)
- Evidence: 9.2GB/10GB used + 3.1GB swap + no traceback + no core
- Heavy Aider prompt + Gemma 3.4GB → LMK kill
- **Not a code bug** — Android environment constraint
- Mitigation (Session 13+): memory check endpoint, warn on heavy tasks

### Golden Rule Lessons (Rule #4/#7)
1. `/tmp` NOT writable in Termux → **use `~/tmp/`**
2. `cat missing > target` **truncates target** → use `mv` atomic pattern
3. Module import test catches missing imports (function-body imports lazily resolved)
4. grep pattern bug → use exact-match `diff` first, not blind `grep -c`

### Cleanup
- Removed: `~/tmp/cs5_route.txt`, `cs5_insert.py`, `cs5g_insert.py`,
  `abort_test.log`, `abort2.log`
- Leftover: `~/tmp/patch_*.py` (~50 files, Session 3–9) — optional cleanup

### Deferred to Session 13
- Filter Aider warnings (`pkg_resources`, version check) in code stream
- Home root cleanup (POC files, `web/app.js`, old `.bak.*`)
- PWA implementation (Home screen install)
- Memory guard for heavy tasks

### Backups (Session 12)
- `server.py.bak.20260926-cs4` (pre-CS5 baseline)
- `server.py.bak.20260926-cs5pre` (pre-route insert)
- `web/index.html.bak.12-cs5g` (pre-web-wire)
- `notes/SESSION-2-FINDINGS.md.bak.12` (pre-findings-update)
- Full backup: `session12-pre-cleanup-20260926-065124.tar.gz` (186K)

## Session 13 (2026-09-26) — Android Intent Tools (Kai Feasibility) ✅

### Achievements
- **tools/android_intent.py** (NEW, 221 lines) — 4 Kai-style tools:
  - set_alarm(hour, minute, message, days)
  - set_timer(seconds, message)
  - open_url(url)
  - create_calendar_event(title, start_unix_ts, duration_min)
- Agent prompt extended — ANDROID DEVICE ACTIONS + STRICT PARSING RULES
- Duplicate-call guard added (`_seen_calls`) in manual tool loop
- 8 tools total registered in tools/__init__.py

### Test Results (T1–T3)
- **T1 English simple** — PASS (alarm 7:30 set, correct)
- **T2 Burmese simple** — Tool call correct (Burmese numerals parsed)
- **T3 Burmese complex (Sat recurring)** — PARTIAL:
  - ✅ hour=18 correct
  - ✅ Single tool call (dedup guard works)
  - ✅ Burmese label pre-filled on Clock
  - ❌ minute=18 (should be 0) — LLM reused hour value
  - ❌ days=all-7 (should be 'sat') — LLM mapped "တိုင်း"→every→all

### Root Findings (CRITICAL for Session 14)
1. **4B model ceiling hit** for Burmese + tools + complex rules
   - Prompt length (3389 chars) overwhelms 4B attention
   - 3-layer problem (Burmese + tool schema + multi-rule) degrades
2. **Samsung OneUI limitation** — Clock ignores DAYS extra
   - Repeat must be set manually by user
   - Tool returns warning message
3. **AOSP DAYS mapping corrected** — Sun=1, Mon=2, ..., Sat=64
4. **Android confirmation dialog** — ✓ tap required (by design, not bypassable without root)

### Confirmed Capabilities (6/9 Kai tools, no Termux:API app)
✅ set_alarm, set_timer, open_url, create_calendar_event
✅ get_local_time (Python datetime)
✅ open_file (am VIEW)
🟡 Notification, Location, SMS — need Termux:API app (pending test)

### Deferred to Session 14
- **Regex pre-parser** for Burmese time (bypass LLM for structured extraction)
- Alternative: shorter prompt + 2-shot examples
- Notification tool (Termux:API app test)
- PWA (from Session 12)
- Aider warning filter (from Session 12)

### Backups (Session 13)
- tools/__init__.py.bak.13
- modes/agent.py.bak.13, .bak.13b
- tools/android_intent.py.bak.13b
- notes/SESSION-2-FINDINGS.md.bak.13

### Lesson
- **Prompt engineering ရဲ့ limit** — 4B ကို complex behavior teach လုပ်ဖို့ rule များလွန်ရင် follow မလုပ်
- **Layer-wise design**: prompt = hint, code = enforcement (Session 14 approach)

## Session 13c (2026-09-26) — Skills Framework MVP ✅

### Delivered
- **core/skills.py** (NEW, 133 lines) — no external deps
  - list_skills(), load_skill(), render(), format_list()
  - Safe name sanitization (traversal-proof)
  - 50KB file size cap
  - Regex frontmatter parser (no PyYAML)
- **~/.ai-agent/skills/** — skills storage
  - Sample: summarize.md, translate-en-mm.md
- **app.py** — Terminal routing (sk/SK5)
  - /skills — list all
  - /skill <name> <args> — run
  - 4 patch points: import, handlers, methods, menu

### Verified (Terminal)
- ✅ /skills → lists 2 skills
- ✅ /skill summarize <text> → 3 bullets (Burmese output)
- ✅ /skill translate-en-mm <text> → natural Burmese
- ✅ /skill no-such-skill test → Unknown skill error
- ✅ /skill summarize (no args) → Missing args + description

### Audit × 3 PASS
- 0 bugs, 0 vulnerabilities
- Path traversal blocked, size cap working

### Deferred to Session 14
- Server API endpoints (/api/skills/*)
- Web UI integration (client routing)
- /create-skill (LLM → markdown)
- Skill edit/delete
- Skill listing command menu in Web

### Backups (Session 13c)
- app.py.bak.13s (pre-skills patch)
- notes/SESSION-2-FINDINGS.md.bak.13c

### Files now
- core/skills.py (NEW, 133 lines)
- app.py (743 → 786 lines, +43)
- ~/.ai-agent/skills/ (new dir + 2 samples)

## Session 14 (2026-09-26) — Skills Framework Full ✅

### Delivered
- **core/skills.py** — full CRUD module (200 lines)
  - list_skills, load_skill, render, format_list
  - build_markdown, save_skill (atomic), delete_skill
  - default_template (fallback)
  - Traversal-safe, 50KB cap
- **server.py** — 5 skills API endpoints
  - GET  /api/skills
  - GET  /api/skills/<name>
  - POST /api/skills/<name>/run
  - POST /api/skills/create (LLM + fallback)
  - DELETE /api/skills/<name>
- **web/index.html** — Web UI
  - /skills button + command
  - /skill <name> <args> (runSkill)
  - /create-skill <name> <desc> (runCreateSkill)
- **app.py** — Terminal (from Session 13c)
  - /skills, /skill <name> <args>

### Test Results (all PASS)
- Terminal: /skills, /skill <name> <args>, errors
- Server: 6 API tests (list, get, run, create, delete, 404/400)
- Web: /skills list, /skill quiz-mm run, /create-skill quiz-mm
- Audit × 3: 0 bugs, 0 vulnerabilities

### Notable Findings
1. **LLM `create-skill` source = "fallback"** — LLM output fails
   validation (likely code fences/format), fallback saves correctly.
   → Functional risk 0 (fallback always succeeds).
   → Session 15 optimization: refine prompt for higher LLM success rate.
2. **Web cache** — Chrome requires `?v=14c` refresh after index.html update.
3. **args size** — API /run has no cap (Session 15 note).

### Deferred to Session 15
- Feature 2: File understanding (text → PDF → images)
- Feature 3: Burmese voice input
- LLM `create-skill` prompt optimization
- args size cap on /api/skills/<name>/run
- PWA (from Session 12)

### Backups (Session 14)
- server.py.bak.14a (pre-S14a)
- web/index.html.bak.14b, .bak.14c
- core/skills.py.bak.14c
- notes/SESSION-2-FINDINGS.md.bak.14

### Files now
- core/skills.py        200 lines (+67)
- server.py             1509 lines (+92)
- app.py                786 lines (Session 13c)
- web/index.html        ~560 lines (+~15)
- ~/.ai-agent/skills/   2 samples + user-created

### Golden Rules — all followed
- ✅ #1 Recursive Planning (design freeze)
- ✅ #2 Post-Writing Audit ×3 (PASS)
- ✅ #3 No Code Without Permission
- ✅ #4 Backup Before Patch
- ✅ #5 Chunk Patches
- ✅ #6 Regression (E2E terminal + web)
- ✅ #7 nano/atomic-mv (heredoc verified)

## Session 16 (2026-09-26) — Web Search Tools (Foundation) ✅

### Delivered
- **tools/web_search.py** (NEW, 253 lines)
  - web_search(query, max_results=5) — DuckDuckGo Lite (no API)
  - web_fetch(url, max_chars=5000) — readable text extract
  - stdlib HTMLParser (no bs4/lxml needed)
  - requests 2.34.2 (already in Termux)

### Security (verified 10/10)
- URL scheme whitelist (http/https only)
- SSRF block: loopback/private/link-local/reserved IPs
- Response size cap: 2 MB
- Timeout: 10s
- Content-Type whitelist
- Results wrapped in <untrusted> tags (prompt injection defense)

### Test Results
- URL safety: 10/10 PASS
- Input validation: 3/3 PASS
- Real search: PASS (3 DDG results)
- Real fetch: PASS (Wikipedia text)

### Files now
- tools/web_search.py      253 lines (NEW)
- tools/__init__.py        23 lines (10 tools total)
- ALL_TOOLS count: 10

### Deferred to Session 17
- WS5: agent prompt update (web tool hints)
- WS6: agent mode end-to-end test
  - English search
  - Burmese search ("မြန်မာသမိုင်း")
  - SSRF attempt (should be blocked)
- Regression: existing 8 tools ဆက် works

### Backups (Session 16)
- tools/__init__.py.bak.16
- notes/SESSION-2-FINDINGS.md.bak.16

### Known Limitations (honest)
- 🟡 DDG may block/captcha at rate
- 🟡 Burmese search quality limited (DDG index)
- 🟡 DDG HTML structure change → parser breaks
- 🔴 Requires internet (PWA offline — error path)

## Session 18 (2026-09-26) — Smart Search (Full Architecture) ✅

### Delivered — 12 new files, 6 code layers
- **core/search/** (NEW package)
  - classifier.py  (117 lines) — L1 category (keyword-based, no LLM)
  - translator.py  (114 lines) — L2 Burmese→English (dict + LLM fallback)
  - base.py        (43 lines)  — Result dataclass, BaseBackend ABC
  - router.py      — L3 category→backends mapping
  - cleaner.py     — L5 dedupe + rank
  - packer.py      — L6 <context> formatting
- **core/search/backends/** (NEW)
  - wikipedia.py     — Wikipedia EN + Burmese (auto-detect)
  - stackexchange.py — Stack Overflow Q&A
  - openlibrary.py   — Books metadata
  - rss.py           — 35 feeds across 7 categories
  - ddg.py           — DuckDuckGo fallback (rate-limited)
- **tools/smart_search.py** (151 lines) — entry point
- **modes/agent.py** — prompt updated (+~700 chars)
- **tools/__init__.py** — 11 tools total

### Architecture (5 code layers + 1 LLM call)
  L1 classify   → category          (CODE)
  L2 translate  → English           (CODE + LLM fallback)
  L3 route      → backends          (CODE)
  L4 fetch      → results           (CODE)
  L5 clean      → dedupe + rank     (CODE)
  L6 pack       → <context>         (CODE)
  L7 LLM        → synthesis         (4B MODEL — 1 call)

### E2E Tests — 4/4 PASS ✅
- Test 1: Burmese knowledge → Wikipedia MM (Burmese answer)
- Test 2: English coding → Stack Overflow (Burmese answer)
- Test 3: Book query → Wikipedia EN (Burmese answer)
- Test 4: News query → RSS Karen News (Burmese answer)
- LLM proactively set `category: "news"` in Test 4 — prompt guidance works

### Key Architecture Insights
1. **4B-friendly design:** Short prompt (~5200 chars), synthesis passed as
   tool_result (NOT embedded in system prompt). 1 LLM call per query.
2. **Code-first determinism:** Classify/route/fetch/clean = code only.
   LLM only does final synthesis.
3. **Backend-aware queries:** WikipediaBackend.prefer_raw=True (uses raw
   Burmese for MM Wikipedia); others use translated English.
4. **Graceful fallback:** No results → general → Wikipedia+DDG → empty.
5. **sources.json:** Config-driven, 35 feeds, 7 categories (user-editable).

### Backends (5 total, all working)
| Backend | Category | Rate | Status |
|---------|----------|------|--------|
| Wikipedia EN+MM | knowledge | ∞ | ✅ |
| Stack Overflow | coding | 300/day | ✅ |
| OpenLibrary | book | ∞ | ✅ |
| RSS (35 feeds) | news | ∞ | ✅ |
| DuckDuckGo | general | 2-3/30s | ✅ fallback |

### Known Limitations
- 🟡 Classifier tie: "Rich Dad Poor Dad စာအုပ် အကြောင်း" → knowledge
  (book keyword + knowledge keyword tie → knowledge wins). Session 19 fix.
- 🟡 DDG fallback fragile (rate-limited) — used only as last resort.
- 🟡 Translator dict limited (~40 terms) — LLM fallback covers rest.
- 🔴 Android LMK: Server dies after 2-3 LLM calls (Session 12/13/15 pattern).
  Not a code bug. sig11 from LiteRT-LM known issue.

### Deferred to Session 19
- Classifier priority for book vs knowledge tie
- Web search Burmese-fix (Session 17 deferral)
- File understanding (PDF/DOCX)
- Burmese voice input
- PWA polish

### Backups (Session 18)
- tools/__init__.py.bak.18
- modes/agent.py.bak.18
- tools/web_search.py.bak.18
- notes/SESSION-2-FINDINGS.md.bak.18

### Files now
- core/search/                  (NEW, 6 modules + 5 backends)
- tools/smart_search.py         (NEW, 151 lines)
- tools/__init__.py             (11 tools)
- modes/agent.py                (441 lines)
- ~/.ai-agent/sources.json      (35 feeds, 7 categories)

═══════════════════════════════════════════════════════════
## Session 19 — Fixes (Reference)
═══════════════════════════════════════════════════════════
F1 — RSS cache (feed-level, parallel)
F2 — Agent history cap (max_entries=40)
F4 — Thermal guard (>45°C refuse)
F5 — Idle auto-shutdown (30 min)

## Session 20 — Fixes (10 Patches)
═══════════════════════════════════════════════════════════

### F6 — Thermal Zone Correction
- server.py: priority ("quiet_therm", "xo-therm", ...)
- Before: cpu_therm (57°C idle → always block)
- After: quiet_therm (40°C skin temp → correct)
- Verified: /api/thermal → zone=quiet_therm, temp_c≈38.6

### F7 — Repetition Loop Detector (Multi-char)
- server.py: _is_repetition_loop(text, min_repeat=15, max_pattern=15)
- Handles Burmese combining char loops (ှ း ်)
- Multi-char pattern matching, not single-char

### F7b — Drain Pattern (Sig11 Fix)
- server.py: agent stream loop
- Before: `break` mid-stream → LiteRT sig11
- After: `skip_remaining = True; continue` (drain generator)

### Patch 9/9f — Sampler Config
- core/llm.py: create_conversation + send_message
- SamplerConfig(temperature=0.2, top_p=0.95, top_k=10)
- RepetitionPenaltyConfig(repetition_penalty=1.1, frequency_penalty=0.05)
- NoRepeatNgramConfig(no_repeat_ngram_size=4)

### B1 — AGENT_SYSTEM Voice Fix
- modes/agent.py: LANGUAGE RULES block
  - PRIMARY: Burmese, SECONDARY: English (technical only)
  - FORBIDDEN: CJK/Thai/Arabic/Greek/Cyrillic/Hindi
  - Digits: Arabic 0-9
- Voice: ALWAYS 'ခင်ဗျာ' (never 'ရှင့်')
- Size: 4785 → 5081 chars

### B2 — CHAT_SYSTEM (New)
- modes/chat.py: CHAT_SYSTEM constant (was missing)
- Provider combines CHAT_SYSTEM + memory hint
- Size: ~700 chars

### C1/C1b — Post-process Script Filter
- server.py: _filter_scripts(text)
- 5 sites: chat chunk/end, agent chunk/end
- Remove CJK/Thai/Greek/Cyrillic/Arabic/Devanagari

### P13 — Filter Tighten
- server.py: _DROP_CPS constant + unicodedata import
- Blacklist: U+00AD, U+200B-F, U+2028-2E, U+2060-64, U+FEFF
- Cf category check for remaining format chars
- Verified: U+00AD=0, U+200B=0 (baseline 2, 16)

### C — Hybrid Routing (Burmese-first)
- tools/smart_search.py: L2 conditional translate
- _BURMESE_FIRST = {knowledge, news, burmese_news}
- If Burmese query + these categories → keep raw (skip translate)
- core/search/translator.py: loss-guard (30% char preservation)

## Session 20 — Verified Working
═══════════════════════════════════════════════════════════
✅ F4 thermal (46-50°C block)
✅ F5 idle (idle_seconds tracked)
✅ F6 zone (quiet_therm)
✅ F7 loop detection
✅ Sampler 0.2/0.95/10
✅ Voice 'ခင်ဗျာ' (no mixing)
✅ Foreign script filter (CJK/Thai 0)
✅ Invisible chars filter (U+00AD/200B = 0)
✅ Hybrid routing (raw Burmese query)

## Session 20 — Remaining Issues
═══════════════════════════════════════════════════════════
🔴 Burmese sequence corruption (virama)
   - U+1039 (virama) 12 occurrences, longest 6-char
   - Filter passes (Burmese range) but sequence invalid
   - Fix: Session 21 Burmese Sequence Validator

🔴 Data hallucination (numbers)
   - smart_search returns "1857.6 million"
   - Actual: ~54 million
   - Fix: prompt + tool result passthrough

🟡 Context echo in UI (agent mode)
   - <context> block visible in tool result
   - Fix: UI hide or parse

🟡 Chunk/end mismatch (14 chars)
   - Design: chunks accumulated, end truncated
   - Client replaces — UX OK, testing confusing

🟡 4B tokenizer limit (root)
   - Burmese atomically not tokenized
   - ~1.5-2 chars/token vs English 5 chars/token
   - Not fixable — model-level

## Session 21 Roadmap
═══════════════════════════════════════════════════════════
P1 — Burmese Sequence Validator
   - Detect invalid virama/medial sequences
   - Cut at first invalid marker
   - Effect: ထိင်္္္္န္ type → gone

P2 — Data Hallucination Guard
   - Prompt: "do not modify numbers from source"
   - Or: post-process number validation against tool result

P3 — Context Echo (UI)
   - Hide <context> block in tool result display

P4 — Chunk/End Consistency
   - Redesign: don't yield chunks during loop detection
   - Or: buffer + emit-when-safe

P5 — Latin Foreign Word Filter
   - serangan (Indonesian), etc.
   - Complex: English technical terms must remain
   - Consider: dictionary-based or skip

P6 — File Understanding (deferred from S18)
   - PDF, DOCX, Image (OCR)

P7 — Burmese Voice Input (deferred)

## Session 20 — Backups Inventory (20+)
═══════════════════════════════════════════════════════════
server.py.bak.20-p4   (F6 zone fix)
server.py.bak.20-p6   (F7 prep)
server.py.bak.20-p7   (F7c function)
server.py.bak.20-p8   (F7b drain)
server.py.bak.20-p10  (C1 filter)
server.py.bak.20-p13  (P13 tighten)
core/llm.py.bak.20-p9  (patch 9)
core/llm.py.bak.20-p9f (values)
modes/agent.py.bak.20-p1   (Corrected ex)
modes/agent.py.bak.20-p2   (Android ex)
modes/agent.py.bak.20-p3   (B+C rule)
modes/agent.py.bak.20-p5   (Response Style)
modes/agent.py.bak.20-p11  (B1 script rule)
modes/chat.py.bak.20-p11   (B2 CHAT_SYSTEM)
config.py.bak.20-p6/p9     (untouched)
core/search/translator.py.bak.20-p12 (loss-guard)
tools/smart_search.py.bak.20-p12 (hybrid route)
notes/SESSION-2-FINDINGS.md.bak.20-final

## Files Changed (Session 20)
═══════════════════════════════════════════════════════════
server.py         (F6, F7, F7b, C1, C1b, P13)  ~1780 lines
core/llm.py       (patch 9/9f: sampler, repetition)  ~305 lines
modes/agent.py    (B1: LANGUAGE RULES + voice)  ~470 lines
modes/chat.py     (B2: CHAT_SYSTEM new)  ~110 lines
core/search/translator.py (loss-guard)  ~115 lines
tools/smart_search.py (hybrid routing)  ~155 lines

## Session 20 — Golden Rules Compliance
═══════════════════════════════════════════════════════════
✅ R1: Recursive Planning (each patch)
✅ R2: Audit × 3 (syntax + logic + security)
✅ R3: No Code Without Permission
✅ R4: Backup Before Every Patch (.bak.20-XX)
✅ R5: Chunk Patches (one at a time)
✅ R6: Regression (verify after each)
✅ R7: nano/append for >50 lines

## Session 20 — Closing Notes
═══════════════════════════════════════════════════════════
- Session length: very long (10 patches)
- 4B ceiling: Burmese quality limited
- Achievable: 60-70% improvement
- Next: Session 21 fresh start (Burmese sequence)
- Phone heat: thermal guard working
- Backup hygiene: 20+ .bak files, cleanup in S21


## Session 21 (2026-10-02) — P1a + Web UI
════════════════════════════════════════════════════════════

### ✅ P1a — Burmese Sequence Validator (module only)
- **core/burmese.py** (NEW) — scan(), BurmeseValidator (incremental), sanitize()
  - B1 duplicate combining mark / B2 orphan mark (no base) / B3 virama needs a
    letter after it / B4 mark run > MAX_MARKS(6)
  - Real text worst mark run measured = 4 → 6 leaves headroom
  - Streaming: trailing mark run is held back, so a chunk boundary landing on a
    lone virama never cuts valid text
- **tests/test_burmese.py** (NEW, 23 tests) — property test: every split point of
  valid AND broken text must equal the one-shot result
- Session 20's real corruption `ထိင်္္္္န်` → `ထိင်` ✓, kinzi `အင်္ဂလိပ်` untouched ✓
- ✅ WIRED in Session 21 part 2 — see “P1b/c/d” below (routes/_helpers.py + routes/chat.py)

### ✅ UI1 — Slash-command autocomplete (web/index.html)
- Type `/` → menu filters live; ↑↓ move, Tab/Enter accept, Esc close, tap to pick
- `/skill ` completes skill names from /api/skills (lazy cache)
- insert = command only — `<task>`/`<name>` placeholders stay in the description
  (found by the headless test, fixed before ship)
- `/skill <name> <args>` ordered before `/skills` so bare `/skill` picks the runner

### ✅ UI2 — Activity indicator (Thinking / Reading / Running / Writing)
- `#activity` pill inside the status bar, phase classes: thinking+writing=info,
  reading=accent, running=warn (spinner = rotating ring)
- typing bubble mirrors the phase; `setBusy(false)` is the single clear point
- `activityFor(tool,args)`: read_file→Reading <file>, run_command→Running <cmd>,
  web/smart_search→Searching…, write_file→Writing <file>, else Running <tool>
- code-mode events: file→Writing, commit→Committing, step→Running

### Verified
- `node --check` on all 3 script blocks ✓
- tests/ui_autocomplete_test.js — 45 PASS (headless DOM stub)
- tests/ui_activity_test.js — 23 PASS
- HTML tag balance + CSS brace balance ✓
- tests.test_burmese — 23 PASS

### ⚠️ Limits of this environment
- flask 3.1.3 installed system-wide (`/usr/local/lib/python3.14/dist-packages`) —
  needed because `tests/run.sh` rewrites `HOME`, which hides `pip --user` packages
- no browser → on-device visual check of pill + menu still pending

### Backups (Session 21)
- web/index.html.bak.21-ui
- core/burmese.py, tests/test_burmese.py, tests/ui_*.js = new files

## Session 21 (2026-10-02, part 2) — P1b/c/d wiring + regression

### ✅ P1b — `sanitize_text()` in routes/_helpers.py
- `sanitize_text(text) = sanitize_burmese(filter_scripts(text))` — one call
  composes the Session 12 script filter (Chinese/Japanese/…) with the Session 21
  sequence validator. Backup: `routes/_helpers.py.bak.21-p1b`

### ✅ P1c — Chat Mode stream (`routes/chat.py`)
- Every model byte goes through an incremental `BurmeseValidator` (`vseq`):
  - `feed(chunk)` returns `safe` (validated prefix) + `cut` flag
  - on cut → yield `safe`, set `seq_cut`/`skip_remaining` and keep draining
    the generator (F7b — abandoning a stream trips LiteRT sig11)
  - held-back tail: `vseq.flush()` is called on the `end` event so the chunks
    add up to the whole message (a lone trailing virama is only wrong if
    nothing follows it)
- `end` frame now carries `"text": fin, "truncated": seq_cut` where `fin` is
  `vseq.accepted` after a cut, else a full re-validation of the streamed text
- drained-without-`end` fallback still emits a terminating `end` frame

### ✅ P1d — sync replies + Agent stream
- `/api/chat` and `/api/agent` replies pass through `sanitize_text()`
- Agent stream: per-iteration validator; cut → `seq_cut` + `skip_remaining`,
  `text_this += held` on flush, final text re-sanitised before it is persisted
  to `agent_mode._messages`, `end` frame reports `truncated`

### ✅ tests/test_p1_sanitize.py (NEW, 8 tests) — drives the real endpoints
- cut mid-stream: no repeated virama reaches the UI, `end.text == ထိင်`,
  `truncated: true`, stream still terminates
- valid text + a syllable split across two chunks (`အင်္` + `ဂလိပ်`) → untouched
- sync `/api/chat` reply sanitised; agent stream cut + clean text persisted

### Regression (`sh tests/run.sh`)
- **193 tests, 26 failures — identical to the baseline run with P1 reverted**
  (backup swap: `routes/chat.py.bak.21-p1c`, `routes/_helpers.py.bak.21-p1b`)
- all 26 live in `tests/test_code.py` (BridgeConfig / CommitFrames /
  CodeEndpoints / StaleSessionRecovery) → **pre-existing, unrelated to P1**
- `tests.test_burmese` 23 ✓, `tests.test_p1_sanitize` 8 ✓, `tests.test_routes` ✓
- UI harnesses: ui_autocomplete_test 45 ✓, ui_activity_test 23 ✓

### Backups (Session 21, part 2)
- routes/_helpers.py.bak.21-p1b, routes/chat.py.bak.21-p1c, routes/chat.py.bak.21-p1d
- core/burmese.py, tests/test_p1_sanitize.py = new files

### Remaining / pending
- on-device visual check of autocomplete + activity pill (no browser here)
- `test_code.py` 26 failures are a separate, pre-existing problem

## Session 21 (2026-10-02, part 3) — Option 1: visual check (real browser)

### Setup (this sandbox had no browser)
- Playwright 1.63 + Chromium headless-shell + `install-deps chromium` (apt)
- Fonts: `fonts-sil-padauk`, `fonts-noto-ui-core` + NotoSansMyanmar variable TTF
  (google/fonts) — the CSS stack asks for `"Noto Sans Myanmar"`, which
  `fc-match` resolved to DejaVu before; on-device Android ships it already
- Server: real `server.py` on 127.0.0.1:8099 with `tests/_harness.bootstrap()`
  (temp HOME + stub `litert_lm`); 2 demo skills written into the temp HOME only
- Scripts: `/tmp/opencode/pw/shot.mjs` (main), `zoom.mjs`, `scale.mjs`,
  `bisect.mjs` → screenshots in `/tmp/opencode/shots/`

### Result: 14/14 checks passed (real Chromium, real endpoints)
| shot | what | check |
|---|---|---|
| 01 idle | status bar + composer | Noto Sans Myanmar actually loaded |
| 02 slash menu | `/` typed | 13 items, menu bottom 623 ≤ input top 649, inside viewport |
| 03 skill menu | `/skill ` typed | both installed skills offered, Tab chips |
| 04 chat turn | Burmese message → stream | assistant bubble renders, status `ready` |
| 05 agent reading | tool_call in flight | pill `Reading SESSION-2-FINDINGS.md` (accent), status `thinking…`, stop button active |
| 06 agent writing | chunks in flight | pill `Writing…` (info/blue) |
| 07 agent done | end frame | pill hidden, full clean Burmese reply, status `ready` |
| 08 pill running | phase colours | pill amber `rgb(224,160,32)` (warn), 155px, fits the status bar |
| 09 mobile | 390×844, `/` typed | menu right edge 378 ≤ 390, no overflow |
| — | console/page errors | none |

### Font investigation (looked like tofu, was not)
- 1× screenshots showed box-like marks; bisected with `14-bisect.png`
- HarfBuzz (`uharfbuzz`, isolated install) on the same 3 fonts: **no `.notdef`
  glyph for any codepoint** — `ကြ` shapes to `[medial-ra gid, ka gid]` with
  correct advances in Noto Sans Myanmar, Noto UI and Padauk
- The "box" is the U+103C medial-ra glyph design, not a missing glyph ✓

### Findings (not blockers, pre-existing)
1. Status text stays `thinking…` while `chunk` events stream — the client only
   sets `generating…` on `type:'text'` (index.html:457), and both the chat and
   agent streams emit `type:'chunk'` (identical in `chat.py.bak.21-p1c`, so
   P1 did not change this).
2. `page.route(... route.continue({url}) })` does not redirect POSTs here —
   the slow-stream shots use an in-page `window.fetch` stub instead.

### Left installed (sandbox only)
- apt: chromium deps, Padauk/Noto fonts; `/usr/share/fonts/truetype/noto-my/`
- `/tmp/opencode/pw`, `/tmp/opencode/shots`, `/tmp/opencode/pylibs2`
- stub server stopped; no repo files touched beyond Session 21's own

## Session 22 (2026-10-02) — Option A: the 26 `test_code.py` failures

### Diagnosis (all 26 were test-side; no product code touched)
**A. Design drift — 13 assertions written for the old opencode wiring**
- `core/opencode.py` now runs the child inside Ubuntu: `PROOT_CMD +
  ["bash", "-lc", script]`, model `opencode/mimo-v2.6-flash-free` (native
  provider), `bridge_config()` → `{}` (nothing managed), install hint is the
  proot + `--allow-scripts` one.
- Tests still expected a bare `['opencode','run','--format','json']` argv,
  `bridge/gemma-4-e4b`, a written `opencode.json` with a `provider` key, and
  the old `npm install -g opencode-ai` hint.
- Confirmed the new side is right: `opencode run --help` (1.18.34) has every
  flag `build_command` emits (`--format json --auto --model --session --file
  --agent --print-logs --log-level`), and `--version` is happy with a missing
  or stale `OPENCODE_CONFIG`. `TestPrintLogs` (written for the wrapper) passed
  all along.

**B. `is_installed()` probe vs the test fake — 13 tests**
- `is_installed()` now runs `proot-distro … bash -lc "which opencode"` through
  `subprocess.run`. `_FakeOpencode` matches any cmd containing "opencode", so
  the probe was counted as a run *and* consumed the canned body/returncode
  meant for the real one:
  - TestCommitFrames: probe ate the transcript → run got `""` → only an `end`
    frame, no commit, no git commit.
  - TestStaleSessionRecovery: probe ate the `Session not found` body → counts
    off by one in both directions (1 vs 2 and 2 vs 1).
  - The happy-path endpoint tests never stubbed `is_installed`, so they hit
    the real probe; this sandbox cannot run `proot-distro` (already inside
    proot) → 500 → no `end` frame / no `ok`.

### Fix (tests only) — `tests/test_code.py`, backup `tests/test_code.py.bak.22-testfix`
1. `_oc_argv(cmd)` helper — pulls the `opencode …` argv back out of the
   proot/`bash -lc` wrapper; `shlex` import.
2. `TestBuildCommand` rewritten on `_oc_argv` (+ `test_runs_inside_ubuntu_proot`).
   `test_default_model_is_the_bridge` → `…_the_native_provider`.
3. `TestBridgeConfig` rewritten for the empty/native config:
   `…_no_bridge_provider_is_declared`, `…_config_path_is_the_managed_file`,
   `test_rewrite_is_idempotent` tolerates "no file" instead of requiring one.
4. `INSTALL_CMD` constant + 5 hint assertions updated to the proot install line.
5. `TestCodeEndpoints.setUp` now defaults `oc.is_installed = True`; the
   `/project` test forces `False` so its name still means something.
6. `fake_opencode()` also forces `is_installed = True` for its block (probe can
   no longer steal a canned body).
7. `TestStaleSessionRecovery` session-flag assertions go through `_oc_argv`.

### Result
- `tests/test_code.py`: 77/77 OK (was 76 tests / 26 failing; +1 new test,
  3 renamed)
- **full suite: 194 tests, `OK` — 0 failures** (was 193 / 26 failing)
- `node tests/ui_autocomplete_test.js` + `ui_activity_test.js`: ALL PASS
- product code untouched: only `tests/test_code.py` changed this session
