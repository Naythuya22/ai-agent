# Quickstart Guide (English)

> This is the canonical English edition of the quickstart guide.
> Burmese original: `QUICKSTART.my.md`. If the two ever differ, this
> English edition is the one to follow.

This app is an AI assistant that runs on an Android phone (in Termux).
It works offline with a local model (Gemma) stored on the phone, and once
you add a key it can also use cloud AIs (OpenAI, Gemini, Anthropic,
OpenRouter, DeepSeek).

## 1. Install

In Termux — the easiest way is to run the installer:

```
pkg install git
git clone https://github.com/Naythuya22/ai-agent.git ai-agent
cd ai-agent
sh install.sh
```

If you also want Code mode, run `sh install.sh --with-code`.
To install everything manually, step by step:

```
pkg install python python-pip git
pip install -r requirements.txt
```

## 2. Place the local model (optional — not required)

Put the `.litertlm` file in one of these locations — the app finds it
automatically:

- the `models/` folder inside the project folder
- `Download/gemmamodel/`

The server starts even without a model — the UI, Settings and cloud turns
work; only local turns will answer "Model not found".

### ⚠️ Using local mode — set Prompt Effort to Fast

When you use the local model in Agent mode, keep
**Settings → Backend → Prompt Effort set to Fast**. With Balanced / Auto
the agent prompt is long, the prefill keeps the CPU at full load for a
long time, and the phone OS may kill the Termux app (verified on a
Xiaomi / HyperOS phone, 2026-11-18). With Fast the prompt is short and
everything runs normally.

Two more things to remember:

- Use only the **Gemma 4 E4B `.litertlm`** file as the local model. If
  another model file with a small context (for example a file with only
  4096 state entries) is left in `Download/`, the app may pick that file
  up automatically and every local turn will fail — do not keep unused
  model files in `Download/`.
- The server exits by itself after 30 idle minutes (a battery-saving
  rule). If the browser shows "offline / 503", the server has probably
  stopped — just run `python server.py` again in Termux.

## 3. Run

```
python server.py
```

The startup screen shows the server address (normally
`http://127.0.0.1:8080`) and a link that contains a token. Open that link
in the phone browser.

To chat from the terminal instead (while the server is running):

```
python app.py
```

## 4. Add cloud keys (optional)

In the Web UI's **Settings**, add a key for each provider you want to
use. Keys are stored only in Termux's private storage (app data) —
never in the project folder. Backup zips can contain your keys, so
never give a backup file to anyone.

## 5. Code mode (optional)

```
pkg install nodejs-lts
npm install -g opencode-ai
```

## 6. Back up (important)

Memory, sessions and keys are hard to recover if the phone breaks.
Back up regularly:

```
python backup.py export              # including keys (keep it secret)
python backup.py export --no-secrets # without keys
python backup.py list <backup.zip>   # see what is inside
python backup.py import <backup.zip> # restore (snapshots current data first)
```

## Help

- Detailed documentation: `README.md`
- If something goes wrong, the log is at `~/.ai-agent/logs/agent.log`
  (you can also open `/logs` inside the app).
