# test
# ai-agent

Hybrid AI assistant for Termux (Android): the local Gemma model through
LiteRT-LM (GPU) plus five cloud providers, falling back to the phone whenever
the cloud cannot answer. What sets it apart: Burmese-native output, a context
window chosen from the model that answers, tier-aware prompts, and an
offline-first setup that upgrades to the cloud once you add keys.

## Features

- **Local + cloud** — 5 providers (OpenAI, Gemini, Anthropic, OpenRouter,
  DeepSeek): keys round-robin, 429s park a key, exhausted providers hand over,
  and every failure path ends on the local model unless you pinned Cloud Only.
- **Model-aware context** — window read from the model (8K on the phone, up to
  1M in the cloud): registry table + runtime learning from provider 400s. No
  fixed slider.
- **Tier-aware prompts** — `cloud_strong` / `cloud_fast` / `local`,
  auto-classified from backend + model name, overridable in Settings.
- **Robustness** — tool failures reported to the model, a verification pass
  after a tool turn (cloud only), tool results reused within one turn, scanned
  tool output marked untrusted.
- **Self-awareness** — `BLOCKED:` / `PARTIAL:` / `UNSURE:` replies plus a live
  `[BUDGET: ctx …% iter …/…]` line in the system prompt.
- Chat / Agent / Code modes with native tool calling (file, shell, search,
  Android intents); memory (facts + profile); Burmese sanitizer + validator;
  PWA (installable, dark / light); opencode integration for Code mode.

Code mode runs opencode with `--format json`, resuming each project folder's
conversation (`--session`); `GET /api/code/doctor` reports the binary, version,
model and config path if anything looks wrong.

## Providers

| Provider | Base URL | Default model |
| --- | --- | --- |
| openai | `https://api.openai.com/v1` | `gpt-4o-mini` |
| gemini | `https://generativelanguage.googleapis.com/v1beta` | `gemini-2.5-flash` |
| openrouter | `https://openrouter.ai/api/v1` | `google/gemini-2.5-flash` |
| anthropic | `https://api.anthropic.com/v1` | `claude-3-5-haiku-latest` |
| deepseek | `https://api.deepseek.com/v1` | `deepseek-chat` |

Keys are entered in Settings and stored in `<project>/data/agent_config.json`
(0600, git-ignored) — nothing comes from the environment; `base_url` and
`model` are editable per provider (presets from `MODELS`). Fallback order
defaults to openai → gemini → openrouter → anthropic → deepseek.

## Model-aware context

Budget per request comes from the model that will answer it: a limit learned at
runtime → exact registry hit → longest prefix match → endpoint guess (unknown
local = 8K, unknown cloud = 128K). `Config.context_size` is a different number
— only the local engine's KV cache.

    gemma:4b          ->  8K
    deepseek-flash    ->  64K
    gpt-4o            ->  128K
    gemini-2.5-flash  ->   1M

A provider 400 teaches the resolver the real limit: the number is parsed out
of the error, saved to `~/.ai-agent/model_limits.json`, and the request is
retried once at the corrected size before the turn may fail.

## Setup

    pkg install python python-pip git
    pip install -r requirements.txt
    python settings.py     # optional: local engine KV-cache size

# Local model: copy a *.litertlm into models/ — auto-discovered (legacy
# Download/gemmamodel dirs and nearby *model* folders are searched too).

# Code mode needs opencode (Termux):
pkg install nodejs-lts
npm install -g opencode-ai
opencode --version

# Code mode looks for that binary inside the Ubuntu proot — if it says it is
# missing: proot-distro login ubuntu && apt install nodejs npm -y && npm
# install -g --allow-scripts=opencode-ai opencode-ai

# Optional: browser terminal from the Code page (Code ▸ Terminal).
pkg install ttyd

## Run

Server — the web UI, `/v1`, Code mode and the terminal app all go through it:

    nohup python server.py > ~/server.log 2>&1 &

Terminal assistant (needs the server already running):

    python app.py

First run: the local model is auto-discovered, backend mode is `auto`, and no
cloud provider is called until its price has been acknowledged in Settings.

Env overrides — optional; defaults and revert paths under **Configuration flags**:

    AI_AGENT_DATA_DIR  AI_AGENT_MAX_TOKENS  AI_AGENT_MAX_ITER  AI_AGENT_PROMPT_TIER
    AI_AGENT_TOOL_ERRORS  AI_AGENT_VERIFY  AI_AGENT_CACHE  AI_AGENT_INJECTION
    AI_AGENT_ADAPTIVE_READ  AI_AGENT_BUDGET_HINT  AI_AGENT_READ_CEILING

## Configuration flags

Read from `config.py`, defaults verbatim:

| Variable | Default | What it does |
| --- | --- | --- |
| `AI_AGENT_DATA_DIR` | `<project>/data` | where `agent_config.json` lives |
| `AI_AGENT_MAX_TOKENS` | 8192 | output tokens per request (ceiling 65536) |
| `AI_AGENT_MAX_ITER` | 30 | tool-loop iterations per turn (max 100) |
| `AI_AGENT_TOOL_ERRORS` | 1 | show tool failures to the model — revert `=0` |
| `AI_AGENT_VERIFY` | 1 | reviewer pass after a tool turn — revert `=0` |
| `AI_AGENT_VERIFY_TOKENS` | 512 | budget for that pass (max 1024) |
| `AI_AGENT_CACHE` | 1 | reuse a tool result within a turn — revert `=0` |
| `AI_AGENT_INJECTION` | 1 | untrusted banner on tool output — revert `=0` |
| `AI_AGENT_ADAPTIVE_READ` | 1 | model-aware `read_file` cap — revert `=0` |
| `AI_AGENT_READ_CEILING` | 100000 | max chars in one read (max 500000) |
| `AI_AGENT_BUDGET_HINT` | 1 | `[BUDGET: …]` line — revert `=0` |
| `AI_AGENT_BUDGET_WARN_CTX` | 80 | ctx% that turns the hint into a warning |
| `AI_AGENT_PROMPT_TIER` | auto | pin a tier when Settings has none |
| `AI_AGENT_REFLECT` | 1 | after each turn, ask the model whether a durable lesson should be saved to memory — revert `=0` |

Booleans take `1/true/yes/on` and `0/false/no/off`; anything else warns and
keeps the default. Settings wins over `AI_AGENT_PROMPT_TIER`. Other vars live
in `core/auth.py`, `core/log.py` and `routes/safety.py`.

## Commands

    /help              Show help
    /chat  /agent      Switch mode
    /code [dir]        Launch opencode in a project dir
    /clear             Clear history
    /mode  /ctx        Mode + context / context usage detail
    /new  /sessions    Fresh chat (archive) / list sessions
    /switch <id>  /delete <id>   Jump to / remove a session
    /history  /summary Recent messages / compress session
    /skills  /skill <n> <args>   List / run a skill
    /remember <text>   Add fact        /forget <kw>   Remove matching
    /memory  /profile  Show memory / profile
    /set <k> <v>  /unset <k>     Set / remove profile field
    /edit              Open memory.json
    /logs [n]          Recent log lines (default 30)
    /exit              Exit

Tip: Type / and Enter for the interactive menu.

## Testing

    sh tests/run.sh                 # full backend suite (704 tests)
    node tests/ui_settings_test.js  # UI suite

Flask is the only third-party dependency; `tests/_harness.py` puts the
`litert_lm` stub on `sys.path`, so the suite needs no model weights or network.

## Data

    ~/.ai-agent/config.json
    ~/.ai-agent/memory.json
    ~/.ai-agent/sessions.json
    ~/.ai-agent/code_projects.json   ~/.ai-agent/skills/
    ~/.ai-agent/model_limits.json    context windows learned from 400s
    ~/.ai-agent/logs/agent.log       rotating, 512 KB x 3
    /sdcard/.ai-agent/{opencode,code_sessions}.json   shared with Ubuntu proot
    <project>/data/agent_config.json
                              Agent Mode API keys — fixed absolute path, never
                              derived from $HOME (the server runs with HOME=
                              /data/data/com.termux/files/home, the shell with
                              /root). 0600, git-ignored, migrated on first run.

## Logging

Everything goes to `~/.ai-agent/logs/agent.log`, including failures that would
otherwise be silent (a save that could not be written, a search backend that
timed out, an opencode session delete that failed). The REPL owns the terminal,
so nothing reaches stdout by default — read the file or use `/logs [n]`.
`AI_AGENT_LOG_CONSOLE=1` also logs to stderr, `AI_AGENT_LOG_LEVEL=DEBUG` adds
cleanup paths. A log directory that cannot be created falls back to console.

## Retry

Search backends and the web-fetch tool retry with jittered exponential
backoff — attempts, per-wait cap and wall-clock budget are set per caller (e.g.
DuckDuckGo: 3 attempts, 1 s base, 8 s per wait, 20 s budget) and `Retry-After`
is honoured as a delay floor. Transport failures and 408/429/5xx get another
attempt; anything else fails fast.

## Layout

    core/      — logic (agent_backend, model_registry, context_trim, prompts,
                 safety, key_manager, burmese, opencode)
    modes/     — chat + agent loops         routes/ — Flask blueprints
    tools/     — read/write/shell/web tools
    prompts/   — 3-tier prompt files (_base + agent_<tier>)
    web/       — PWA UI                     tests/  — unittest + node UI tests

## Notes

Design decisions live in `notes/` (`notes/SESSION-2-FINDINGS.md`); for the
current change history run `git log` in a clone of this repo.

## Test section
