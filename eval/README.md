# Behavioural eval harness (Release 20)

A small, deterministic eval for answer quality — the companion to the
unit suite. The unit tests prove the code does what the code says;
this harness checks that real answers stay good when prompts, models
or code change, by driving the **real** `AgentMode`
path over a fixed set of cases and scoring the outcomes.

```bash
python3 eval/run_eval.py --stub          # offline, fake backend (default)
python3 eval/run_eval.py --live          # real providers / local engine
python3 eval/run_eval.py --stub --case agent-jail-refusal
python3 eval/run_eval.py --live --no-report
```

Exit code is `0` only when every case passes (`1` = a case failed,
`2` = usage / case-file error). A JSON report is written to
`eval/results/run-eval-latest.json` unless `--no-report` is given.

> မှတ်ချက် (မြန်မာ): ဒါက အဖြေ အရည်အသွေးကို တိုင်းတဲ့ စမ်းသပ်မှု
> အသေးပါ။ `--stub` ဆိုရင် model အတုနဲ့ offline စမ်းတာ (key မလို၊
> ပိုက်ဆံ မကုန်)၊ `--live` ဆိုရင် တကယ့် provider/model နဲ့
> စမ်းတာပါ။ Case အသစ် ထည့်ချင်ရင် `cases.json` ထဲမှာ ပုံစံတူ
> တစ်ခု ထပ်ထည့်ရုံပါပဲ။

## Stub vs live

- **Stub mode** uses a scripted fake backend (the same test-seam
  pattern as the unit tests: `KeyManager` + `AgentBackend` + a fake
  transport). The script only decides what the "model" asks for — tools
  really execute in the sandbox, the file jail and the secret
  redactor really run, and the checks judge the real outcome. It
  must pass 100%; the unit suite asserts exactly that
  (`tests/test_eval_harness.py`). Stub mode verifies the *harness
  and the machinery*; it says nothing about model quality.
- **Live mode** answers with the configured providers / local
  engine from the user's real config. It spends real tokens and is
  **manual only — never run in the unit suite**. The real config
  files are *copied* into the case's isolated home; originals are
  only read.

## Isolation

Every case runs in a fresh temp HOME with a fresh config dir and a
fresh sandbox directory (the sandbox is also the file-tools jail
root, and the working directory for the turn). Sessions,
checkpoints, the usage ledger and backups land in the temp tree,
which is deleted afterwards. An eval run cannot touch the user's
real data, keys, sessions or usage ledger.

## Cases

`cases.json` (version 1) holds the case set. One case:

```json
{
  "id": "agent-read-seeded",
  "mode": "agent",
  "prompt": "data/answer.txt ဖိုင်ထဲက လျှို့ဝှက် ကိန်းဂဏန်း ကို ဖတ်ပြီး ပြောပါ",
  "setup": [{"path": "data/answer.txt", "content": "လျှို့ဝှက် ကိန်းဂဏန်း: 42"}],
  "stub": [
    {"tool_call": {"name": "read_file", "args": {"path": "data/answer.txt"}}},
    {"answer": "လျှို့ဝှက် ကိန်းဂဏန်းက 42 ပါ ကိုကြီး။"}
  ],
  "checks": {"tool_called": ["read_file"], "must_contain": ["42"]}
}
```

- `mode`: `"agent"` (AgentMode) — the only conversation mode
  since Release 22.
- `setup`: files seeded into the sandbox before the turn
  (relative paths only).
- `stub`: the fake model's script, replayed in order — one step is
  consumed per tool result in the conversation. A step is
  `{"answer": "…"}`, `{"tool_call": {"name": …, "args": {…}}}`, or
  `{"answer_from_tool_result": true}` (the "model" quotes the tool
  result back — how the jail-refusal and redaction cases observe
  what a tool really returned).
- `checks` — deterministic only:
  - `must_contain` / `must_not_contain`: substrings of the answer
  - `regex`: a pattern that must match the answer
  - `tool_called`: tool names that must appear in the turn's real
    transcript
  - `no_tool_calls`: `true` when the turn must use no tools
  - `file_exists`: sandbox-relative paths that must exist after
  - `file_content`: `[{path, contains}]` content assertions

## Why deterministic checks only

No LLM judge in v1. A judge is neither repeatable (the same answer
can score differently twice) nor free (every judgement is another
paid call), and an eval you cannot re-run identically is a rumour,
not a measurement. Deterministic checks cover the behaviours that
regress silently — language, format, tool use, jail, redaction,
file side-effects. Nuance can come later, on top of a harness that
is trusted.

## Relationship to the older harness

The Phase-3 era harness (`eval/run.py` + `eval/runner.py`, which
scored *canned* stub answers and POSTed live cases to a *running
server*) was removed in Release 23 (v1.21.0); it is recoverable from
git tag `v1.20.0`. This harness superseded it: it runs in-process
through the real mode objects, so tool execution, the jail and the
redactor are part of what is measured, and it needs no running
server.
