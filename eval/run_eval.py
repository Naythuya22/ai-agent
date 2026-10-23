#!/usr/bin/env python3
"""Release 20 — a small, deterministic behavioural eval harness.

The unit suite (1156 tests) proves the code does what the code says.
This harness asks the other question: do real answers stay good when
prompts, models or code change? It drives the REAL AgentMode path
headlessly over a fixed case set (``cases.json``)
and scores each answer with deterministic checks only — substrings,
a regex, which tools really ran, and what really landed in the
sandbox. No LLM judge: a judge is neither repeatable nor free, and
an eval you cannot re-run identically is a rumour, not a measurement.

Two run modes:

  --stub  (default) a scripted fake backend stands in for the model,
          so the harness itself is verified offline with zero keys.
          The script only decides what the "model" asks for — tools
          really execute, the jail and the redactor really run, and
          the checks judge the real outcome. Must pass 100%.
  --live  the configured providers / local engine from the user's
          real config answer for real. Manual use only (never in the
          unit suite): it spends real tokens. The real config files
          are COPIED into the isolated home — originals are only read.

Isolation: every case runs with a fresh temp HOME, a fresh config
dir and a fresh sandbox directory (which is also the file-tools
jail root). Sessions, checkpoints, the usage ledger and backups all
land in the temp tree, which is deleted afterwards. An eval run can
never touch the user's real data.

Usage:
  python3 eval/run_eval.py [--stub] [--case ID] [--cases FILE]
                           [--report FILE | --no-report]
Exit code: 0 when every case passes, 1 when any case fails,
2 for usage / case-file errors.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CASES_FILE = Path(__file__).resolve().parent / "cases.json"
DEFAULT_REPORT = Path(__file__).resolve().parent / "results" / "run-eval-latest.json"
CHECK_KEYS = {"must_contain", "must_not_contain", "regex", "tool_called",
              "no_tool_calls", "file_exists", "file_content"}
STUB_KEY = "<redacted>"


def _ensure_engine_importable() -> None:
    """core.llm imports litert_lm; off-phone the test stub provides it.

    Mirrors tests/_harness.py: only add the stub when the real engine
    library is absent, so the phone keeps using the real thing.
    """
    try:
        import litert_lm  # noqa: F401
        return
    except Exception:
        stub = ROOT / "tests" / "stubs"
        if stub.is_dir() and str(stub) not in sys.path:
            sys.path.insert(0, str(stub))


# ---------------------------------------------------------------- cases

def load_cases(path=CASES_FILE) -> list[dict]:
    """Read + validate a cases file. ValueError with a clear message."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_cases(data)


def _check_rel(path: str, where: str) -> None:
    p = Path(path)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"{where}: path must stay inside the sandbox: {path!r}")


def validate_cases(data) -> list[dict]:
    if not isinstance(data, dict):
        raise ValueError("cases file must be a JSON object")
    if data.get("version") != 1:
        raise ValueError("cases file: 'version' must be 1")
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases file: 'cases' must be a non-empty list")
    seen: set = set()
    for i, c in enumerate(cases):
        where = f"case #{i + 1}"
        if not isinstance(c, dict):
            raise ValueError(f"{where}: each case must be an object")
        cid = c.get("id")
        if not isinstance(cid, str) or not cid:
            raise ValueError(f"{where}: 'id' must be a non-empty string")
        if cid in seen:
            raise ValueError(f"duplicate case id: {cid}")
        seen.add(cid)
        where = f"case {cid}"
        if c.get("mode") != "agent":
            raise ValueError(f"{where}: 'mode' must be 'agent'")
        if not isinstance(c.get("prompt"), str) or not c["prompt"]:
            raise ValueError(f"{where}: 'prompt' must be a non-empty string")
        checks = c.get("checks")
        if not isinstance(checks, dict) or not checks:
            raise ValueError(f"{where}: 'checks' must be a non-empty object")
        unknown = set(checks) - CHECK_KEYS
        if unknown:
            raise ValueError(f"{where}: unknown check(s): {sorted(unknown)}")
        for k in ("must_contain", "must_not_contain", "tool_called", "file_exists"):
            if k in checks:
                v = checks[k]
                if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                    raise ValueError(f"{where}: check '{k}' must be a list of strings")
        for rel in checks.get("file_exists", []):
            _check_rel(rel, where)
        if "regex" in checks:
            if not isinstance(checks["regex"], str):
                raise ValueError(f"{where}: check 'regex' must be a string")
            try:
                re.compile(checks["regex"])
            except re.error as e:
                raise ValueError(f"{where}: bad regex: {e}") from e
        if "no_tool_calls" in checks and not isinstance(checks["no_tool_calls"], bool):
            raise ValueError(f"{where}: check 'no_tool_calls' must be true/false")
        if "file_content" in checks:
            fc = checks["file_content"]
            if (not isinstance(fc, list) or not all(
                    isinstance(x, dict) and isinstance(x.get("path"), str)
                    and isinstance(x.get("contains"), str) for x in fc)):
                raise ValueError(
                    f"{where}: 'file_content' must be a list of {{path, contains}}")
            for x in fc:
                _check_rel(x["path"], where)
        for entry in c.get("setup", []):
            if (not isinstance(entry, dict)
                    or not isinstance(entry.get("path"), str)
                    or not isinstance(entry.get("content"), str)):
                raise ValueError(f"{where}: setup entries must be {{path, content}}")
            _check_rel(entry["path"], where)
        stub = c.get("stub")
        if stub is not None:
            if not isinstance(stub, list) or not stub:
                raise ValueError(f"{where}: 'stub' must be a non-empty list of steps")
            for step in stub:
                if not isinstance(step, dict) or len(step) != 1:
                    raise ValueError(f"{where}: each stub step needs exactly one key")
                if "answer" in step:
                    if not isinstance(step["answer"], str):
                        raise ValueError(f"{where}: stub 'answer' must be a string")
                elif "answer_from_tool_result" in step:
                    if step["answer_from_tool_result"] is not True:
                        raise ValueError(
                            f"{where}: 'answer_from_tool_result' must be true")
                elif "tool_call" in step:
                    tc = step["tool_call"]
                    if (not isinstance(tc, dict)
                            or not isinstance(tc.get("name"), str)
                            or not isinstance(tc.get("args", {}), dict)):
                        raise ValueError(
                            f"{where}: stub 'tool_call' must be {{name, args}}")
                else:
                    raise ValueError(f"{where}: unknown stub step: {sorted(step)}")
    return cases


# ---------------------------------------------------------------- checks

def evaluate(case: dict, answer: str, tools_called: list, sandbox: Path) -> list:
    """Score one finished turn. Returns [{name, ok, detail}]."""
    out = []
    ans = answer or ""
    spec = case["checks"]

    def add(name, ok, detail=""):
        out.append({"name": name, "ok": bool(ok), "detail": "" if ok else detail})

    for s in spec.get("must_contain", []):
        add(f"must_contain:{s}", s in ans, "not in the answer")
    for s in spec.get("must_not_contain", []):
        add(f"must_not_contain:{s}", s not in ans, "found in the answer")
    if "regex" in spec:
        add(f"regex:{spec['regex']}",
            re.search(spec["regex"], ans) is not None, "pattern not in the answer")
    for name in spec.get("tool_called", []):
        add(f"tool_called:{name}", name in tools_called,
            f"tools called: {tools_called or 'none'}")
    if spec.get("no_tool_calls"):
        add("no_tool_calls", not tools_called, f"tools called: {tools_called}")
    for rel in spec.get("file_exists", []):
        add(f"file_exists:{rel}", (sandbox / rel).is_file(), "missing in the sandbox")
    for fc in spec.get("file_content", []):
        p = sandbox / fc["path"]
        ok = p.is_file() and fc["contains"] in p.read_text(
            encoding="utf-8", errors="replace")
        add(f"file_content:{fc['path']}", ok, "content mismatch or file missing")
    return out


# ---------------------------------------------------------------- fakes

class _FnTool:
    """The phone's tool-wrapper shape (an object with .fn), as in the
    unit tests: the off-phone stub's tool_from_function returns dicts,
    which the cloud loop cannot serialise."""

    def __init__(self, fn):
        self.fn = fn


class _ScriptedTransport:
    """A fake provider for stub mode, content-driven like the unit
    tests' ToolCallTransport: it replays the case's script in order —
    a step is consumed per tool result already in the conversation —
    so extra auxiliary calls can never shift the script."""

    def __init__(self, script):
        self.script = script
        self.calls = []

    def __call__(self, url, body, headers, timeout=30):
        self.calls.append(body)
        body = body or {}
        msgs = body.get("messages") or []
        has_tools = bool(body.get("tools"))
        n_results = sum(1 for m in msgs if m.get("role") == "tool")
        last_tool_text = ""
        for m in reversed(msgs):
            if m.get("role") == "tool":
                c = m.get("content")
                last_tool_text = c if isinstance(c, str) else json.dumps(
                    c, ensure_ascii=False)
                break
        step = self._next_step(n_results, has_tools)
        if step and "tool_call" in step:
            tc = step["tool_call"]
            return {"choices": [{"message": {
                "content": "",
                "tool_calls": [{"id": "eval-call", "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": json.dumps(
                                        tc.get("args") or {},
                                        ensure_ascii=False)}}],
            }}], "usage": {"prompt_tokens": 4, "completion_tokens": 2}}
        text = ""
        if step and step.get("answer_from_tool_result"):
            text = last_tool_text
        elif step and "answer" in step:
            text = step["answer"]
        return {"choices": [{"message": {"content": text}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2}}

    def _next_step(self, n_results, has_tools):
        for step in self.script[n_results:]:
            if "tool_call" in step:
                if has_tools:
                    return step
                continue  # a tools-free caller (reviewer) gets the answer
            return step
        return None


class _FakeLLM:
    """core.llm.LLM stand-in for stub mode (the tests' FakeLocalLLM
    pattern). Agent stub turns are pinned to the cloud backend, so
    the local engine is never driven."""

    def __init__(self, answers):
        self._answers = list(answers)
        self.touched = []

    def ask(self, prompt):
        return self._answers[0] if self._answers else ""

    def load(self):
        self.touched.append("load")


# ---------------------------------------------------------------- isolation

@contextlib.contextmanager
def _isolated_env(seed_files=None):
    """Fresh temp HOME + config dir + sandbox for ONE case.

    config's path attributes are computed at import time, and
    sessions/projects bind theirs the same way, so all of them are
    re-pointed for the case and restored afterwards — the same trick
    tests/_harness.py uses, per case instead of per suite.
    """
    import config
    from core import projects as projects_mod
    import sessions as sessions_mod

    root = Path(tempfile.mkdtemp(prefix="ai-agent-eval-"))
    home = root / "home"
    cfg_dir = home / ".ai-agent"
    sandbox = root / "sandbox"
    cfg_dir.mkdir(parents=True)
    sandbox.mkdir(parents=True)

    copied = set()
    for src in seed_files or []:
        try:
            if src.is_file():
                shutil.copyfile(src, cfg_dir / src.name)
                copied.add(src.name)
        except OSError:
            pass
    if "config.json" not in copied:
        model = home / "stub-model.litertlm"
        model.write_bytes(b"eval stub model - never loaded")
        (cfg_dir / "config.json").write_text(json.dumps(
            {"model_path": str(model), "backend": "CPU",
             "context_size": 8192}))

    saved_env = {k: os.environ.get(k) for k in ("HOME", "AI_AGENT_DATA_DIR")}
    cfg_attrs = ("CONFIG_DIR", "CONFIG_FILE", "DATA_DIR",
                 "AGENT_CONFIG_FILE", "PROJECT_ROOT")
    saved_cfg = {k: getattr(config, k) for k in cfg_attrs}
    flag_attrs = ("AGENT_VERIFY_ENABLED", "AGENT_REFLECT_ENABLED")
    saved_flags = {k: getattr(config, k, None) for k in flag_attrs}
    saved_projects = projects_mod.PROJECTS_FILE
    saved_sessions = (sessions_mod.SESS_DIR, sessions_mod.SESS_FILE,
                      sessions_mod.ARCHIVE_DIR)

    os.environ["HOME"] = str(home)
    os.environ["AI_AGENT_DATA_DIR"] = str(cfg_dir)
    config.CONFIG_DIR = cfg_dir
    config.CONFIG_FILE = cfg_dir / "config.json"
    config.DATA_DIR = cfg_dir
    config.AGENT_CONFIG_FILE = cfg_dir / "agent_config.json"
    config.PROJECT_ROOT = sandbox
    # Deterministic scoring: the verify/reflect passes are auxiliary
    # model judgements (the unit tests pin them off the same way).
    for k in flag_attrs:
        if saved_flags[k] is not None:
            setattr(config, k, False)
    projects_mod.PROJECTS_FILE = cfg_dir / "code_projects.json"
    sessions_mod.SESS_DIR = cfg_dir
    sessions_mod.SESS_FILE = cfg_dir / "sessions.json"
    sessions_mod.ARCHIVE_DIR = cfg_dir / "archives"
    try:
        yield root, home, sandbox
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for k, v in saved_cfg.items():
            setattr(config, k, v)
        for k, v in saved_flags.items():
            if v is not None:
                setattr(config, k, v)
        projects_mod.PROJECTS_FILE = saved_projects
        (sessions_mod.SESS_DIR, sessions_mod.SESS_FILE,
         sessions_mod.ARCHIVE_DIR) = saved_sessions
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------- drivers

def _run_stub_agent(case):
    from core.agent_backend import AgentBackend
    from core.key_manager import KeyManager
    from modes.agent import AgentMode
    from tools import ALL_TOOLS

    km = KeyManager()
    if km.path.exists():
        km.path.unlink()
    km.add_key("openai", STUB_KEY, "eval-stub")
    cfg = km.load()
    cfg["prefer_cloud"] = True
    cfg["cost_ack"] = True
    km.save()
    km.set_mode("cloud")
    km.save()
    transport = _ScriptedTransport(case.get("stub") or [])
    be = AgentBackend(keys=km, net_state=lambda: "online", http=transport)
    mode = AgentMode(_FakeLLM([]), None)
    mode._backend = be
    mode._tools = [_FnTool(fn) for fn in ALL_TOOLS]
    answer = mode.handle(case["prompt"])
    tools = [m.get("tool") for m in mode._messages
             if m.get("role") == "tool_call"]
    return answer, tools


def _run_live(case):
    """Real engine + real configured backend. Manual use only."""
    from core.llm import LLM
    llm = LLM()
    from modes.agent import AgentMode
    mode = AgentMode(llm, None)
    answer = mode.handle(case["prompt"])
    tools = [m.get("tool") for m in mode._messages
             if m.get("role") == "tool_call"]
    return answer, tools


def run_case(case: dict, stub: bool = True, seed_files=None) -> dict:
    result = {"id": case["id"], "mode": case["mode"], "passed": False,
              "checks": [], "error": None, "answer": ""}
    try:
        _ensure_engine_importable()
        with _isolated_env(seed_files) as (_root, _home, sandbox):
            for entry in case.get("setup", []):
                p = sandbox / entry["path"]
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(entry["content"], encoding="utf-8")
            if stub and not case.get("stub"):
                raise ValueError(
                    f"case {case['id']}: no 'stub' script (stub mode needs one)")
            cwd = os.getcwd()
            os.chdir(sandbox)
            try:
                if stub:
                    answer, tools = _run_stub_agent(case)
                else:
                    answer, tools = _run_live(case)
            finally:
                os.chdir(cwd)
            result["answer"] = (answer or "")[:400]
            result["checks"] = evaluate(case, answer, tools, sandbox)
            result["passed"] = all(c["ok"] for c in result["checks"])
    except Exception as e:  # a broken case is a FAILED case, never a crash
        result["error"] = f"{type(e).__name__}: {e}"
        result["checks"] = [{"name": "run", "ok": False,
                              "detail": result["error"]}]
    return result


def run_cases(cases, stub: bool = True, seed_files=None) -> list:
    return [run_case(c, stub=stub, seed_files=seed_files) for c in cases]


# ---------------------------------------------------------------- output

def summarize(results) -> tuple:
    return sum(1 for r in results if r["passed"]), len(results)


def exit_code(results) -> int:
    passed, _total = summarize(results)
    return 0 if passed == len(results) else 1


def render(results) -> str:
    lines = []
    for r in results:
        if r["passed"]:
            lines.append(f"PASS {r['id']}")
        else:
            lines.append(f"FAIL {r['id']}")
            for ch in r["checks"]:
                if not ch["ok"]:
                    detail = f" ({ch['detail']})" if ch["detail"] else ""
                    lines.append(f"  - {ch['name']}{detail}")
    passed, total = summarize(results)
    lines.append(f"score: {passed}/{total}")
    return "\n".join(lines)


def write_report(results, path, stub: bool) -> Path:
    import time
    passed, total = summarize(results)
    payload = {"version": 1, "mode": "stub" if stub else "live",
               "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "score": f"{passed}/{total}", "results": results}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    return path


def _live_seed_files() -> list:
    """The user's real config files, to COPY into the isolated home.

    Read at startup, before any case re-points config — the paths are
    the import-time (real) ones. Originals are never written.
    """
    import config
    seeds = []
    for p in (Path(config.CONFIG_FILE), Path(config.AGENT_CONFIG_FILE),
              config.private_keys_file()):
        if p.is_file() and p not in seeds:
            seeds.append(p)
    return seeds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="ai-agent behavioural eval (deterministic checks)")
    ap.add_argument("--stub", action="store_true",
                    help="offline fake backend (default)")
    ap.add_argument("--live", action="store_true",
                    help="real providers/local engine from the real config "
                         "(spends real tokens; manual use only)")
    ap.add_argument("--cases", default=str(CASES_FILE),
                    help=f"cases file (default {CASES_FILE})")
    ap.add_argument("--case", default=None,
                    help="run only the case with this id")
    ap.add_argument("--report", default=None,
                    help=f"JSON report path (default {DEFAULT_REPORT})")
    ap.add_argument("--no-report", action="store_true",
                    help="do not write a JSON report")
    args = ap.parse_args(argv)
    if args.stub and args.live:
        print("error: --stub and --live are alternatives", file=sys.stderr)
        return 2
    stub = not args.live
    try:
        cases = load_cases(Path(args.cases))
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            print(f"error: no case with id {args.case!r}", file=sys.stderr)
            return 2
    seed = None if stub else _live_seed_files()
    results = run_cases(cases, stub=stub, seed_files=seed)
    print(render(results))
    if not args.no_report:
        out = write_report(results, Path(args.report) if args.report
                           else DEFAULT_REPORT, stub)
        print(f"\nreport: {out}")
    return exit_code(results)


if __name__ == "__main__":
    raise SystemExit(main())
