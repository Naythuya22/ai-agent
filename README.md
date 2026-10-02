# ai-agent

Offline AI assistant for Termux (Android). Powered by Gemma 4 E4B via LiteRT-LM (GPU).

## Features

- Fully offline, Burmese-native
- Chat / Agent / Code modes
- Native tool calling (file ops)
- Memory (facts + relationship profile)
- Context slider (8K / 12K / 16K)
- Dark / light theme toggle (header ☀️/🌙, saved per device)
- opencode integration for code editing (replaces aider)

Code mode runs opencode against this app's own LiteRT bridge on
127.0.0.1:8080. The provider config is generated at `~/.ai-agent/opencode.json`
and passed to the child process, and each project folder keeps its own
conversation (`~/.ai-agent/code_sessions.json`) so follow-up turns continue
where the last one stopped. `GET /api/code/doctor` reports the binary, version,
model and config path if anything looks wrong.

## Setup

pkg install python
pip install flask questionary
python settings.py

# Code mode needs opencode (Termux):
pkg install nodejs-lts
npm install -g opencode-ai
opencode --version

# Optional: browser terminal from the Code page (Code ▸ Terminal).
pkg install ttyd

## Run

Server (for code mode):
nohup python server.py > ~/server.log 2>&1 &

Assistant:
python app.py

## Commands

/help      Show help
/chat      Chat mode
/agent     Agent mode (tools)
/code      Launch opencode in a project dir
/clear     Clear history
/mode      Show mode + context
/remember  Add fact
/forget    Remove matching
/memory    Show memory
/profile   Show profile
/set       Set profile field
/unset     Remove profile field
/edit      Open memory.json
/logs      Show the last 50 log lines (/logs 200 for more)
/exit      Exit

Tip: Type / and Enter for interactive menu.

## Data

~/.ai-agent/config.json
~/.ai-agent/memory.json
~/.ai-agent/sessions.json
~/.ai-agent/code_sessions.json
~/.ai-agent/code_projects.json
~/.ai-agent/skills/
~/.ai-agent/logs/agent.log    Rotating log (512 KB x 3)

## Logging

Everything is logged to `~/.ai-agent/logs/agent.log`, including failures that
would otherwise be silent — a session or memory save that could not be
written, a search backend that timed out, an opencode session delete that
failed. The chat REPL owns the terminal, so nothing is written to stdout by
default; read the file, or use `/logs [n]` inside the app.

    AI_AGENT_LOG_CONSOLE=1   Also log to stderr (for `python app.py` in a
                             terminal, or `nohup python server.py`).
    AI_AGENT_LOG_LEVEL=DEBUG Default is INFO; DEBUG adds the best-effort
                             cleanup paths (conv close, proc kill, stat).

Failures never take the app down: if the log directory cannot be created
(read-only storage, full disk), logging degrades to console-only and the
request continues.

## Retry

Network calls that are expected to be flaky (search backends, HTTP LLM
proxy) retry with exponential backoff up to ~10 s each and respect
`Retry-After` on 429/408/5xx. Other errors fail fast.
