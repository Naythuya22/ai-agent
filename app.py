"""ai-agent — main entry point (REPL with mode switching)."""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import Config
from core.llm import LLMError, estimate_messages_tokens
from core.http_llm import HTTPLLM
from core.log import get_logger, tail as log_tail, log_path
from memory import Memory
from memory_extract import extract as extract_profile
from core.skills import load_skill, render, format_list  # SK5

log = get_logger("app")

try:
    from sessions import SessionStore
except ImportError:
    SessionStore = None
from modes.agent import AgentMode

# Command menu (for / trigger)
COMMAND_MENU = [
    ("/help",     "Show help"),
    ("/skills",   "List available skills"),
    ("/skill",    "Run a skill — /skill <name> <args>"),
    ("/agent",    "Switch to agent mode (tools)"),
    ("/clear",    "Clear current mode's history"),
    ("/mode",     "Show current mode + context"),
    ("/memory",   "Show current memory"),
    ("/profile",  "Show relationship profile"),
    ("/new",      "Start a fresh chat (archives current)"),
    ("/sessions", "List sessions (current mode)"),
    ("/switch",   "Switch session — then type id"),
    ("/delete",   "Delete session — then type id"),
    ("/history",  "Show recent messages"),
    ("/summary",  "Compress session (archive + summarize)"),
    ("/ctx",      "Show context usage detail"),
    ("/8k",       "Set context to 8K (8192)"),
    ("/12k",      "Set context to 12K (12288)"),
    ("/16k",      "Set context to 16K (16384)"),
    ("/remember", "Add a fact — then type text"),
    ("/forget",   "Remove matching items — then type keyword"),
    ("/set",      "Set profile field — then type: <field> <value>"),
    ("/unset",    "Remove profile field — then type: <field>"),
    ("/edit",     "Open memory.json in nano"),
    ("/logs",     "Show recent log lines (silent failures land here)"),
    ("/code",     "Launch opencode (code mode) — optional: dir"),
    ("/exit",     "Exit app"),
]

# Commands that need an argument after selection
ARG_REQUIRED = {"/remember", "/forget", "/set", "/unset", "/switch", "/delete"}

# ANSI colors
C_CYAN = "\033[0;36m"
C_GREEN = "\033[0;32m"
C_YELLOW = "\033[0;33m"
C_RED = "\033[0;31m"
C_DIM = "\033[2m"
C_RESET = "\033[0m"


class App:
    """REPL app with mode switching (agent, code)."""

    def __init__(self):
        self.config = Config.load()
        self.memory = Memory.load()
        self.llm = HTTPLLM(self.config)
        self.modes = {
            "agent": AgentMode(self.llm, self.memory),
        }
        self.mode_name = "agent"
        self.running = True

    @property
    def mode(self):
        return self.modes[self.mode_name]

    # ---------- Prompt ----------

    def _prompt(self) -> str:
        ctx_k = self.config.context_size // 1024
        return f"{C_CYAN}[{self.mode_name}:{ctx_k}K]{C_RESET} > "

    # ---------- Commands ----------

    def _cmd_help(self) -> None:
        print(f"\n{C_GREEN}Commands:{C_RESET}")
        print("  /agent            Switch to agent mode (tools)")
        print("  /new              Start fresh chat (archive current)")
        print("  /sessions         List sessions (current mode)")
        print("  /switch <id>      Switch to session id")
        print("  /delete <id>      Delete session id")
        print("  /history          Show recent messages")
        print("  /summary          Compress session (archive + summarize)")
        print("  /ctx              Show context usage")
        print("  /8k /12k /16k     Set context size (reload on next msg)")
        print("  /clear            Clear current mode's history")
        print("  /mode             Show current mode + context")
        print("  /remember <text>  Add a fact to memory")
        print("  /memory           Show current memory")
        print("  /forget <keyword> Remove matching items from memory")
        print("  /profile          Show relationship profile")
        print("  /set <k> <v>      Set a profile field")
        print("  /unset <k>        Remove a profile field")
        print("  /edit             Open memory.json in nano")
        print("  /logs [n]         Show last n log lines (default 30)")
        print("  /code [dir]       Launch opencode (optional workdir)")
        print("  /help             Show this help")
        print("  /exit             Exit app")
        print()

    def _cmd_switch(self, name: str) -> None:
        if name not in self.modes:
            print(f"{C_RED}Unknown mode:{C_RESET} {name}")
            return
        if name == self.mode_name:
            print(f"{C_DIM}Already in {name} mode.{C_RESET}")
            return
        self.mode_name = name
        print(f"{C_GREEN}Switched to {name} mode.{C_RESET}")

    def _cmd_clear(self) -> None:
        self.mode.reset()
        print(f"{C_DIM}Conversation cleared ({self.mode_name}).{C_RESET}")

    def _cmd_mode(self) -> None:
        print(f"  Mode:    {self.mode_name}")
        print(f"  Context: {self.config.context_size} "
              f"({self.config.context_size // 1024}K)")
        print(f"  Backend: {self.config.backend}")
        print(f"  Turns:   {self.mode.turns}")
        print(f"  Loaded:  {self.llm.is_loaded}")
        print(f"  Modes:   {', '.join(self.modes.keys())}")

    def _cmd_remember(self, arg: str) -> None:
        text = arg.strip()
        if not text:
            print(f"{C_RED}Usage:{C_RESET} /remember <text>")
            return
        ok, reason = self.memory.add_fact(text, source=self.mode_name)
        if ok:
            print(f"{C_GREEN}Remembered:{C_RESET} {text}")
        else:
            print(f"{C_DIM}({reason}){C_RESET}")

    def _cmd_memory(self) -> None:
        print(self.memory.show())

    def _cmd_forget(self, arg: str) -> None:
        kw = arg.strip()
        if not kw:
            print(f"{C_RED}Usage:{C_RESET} /forget <keyword>")
            return
        n = self.memory.forget(kw)
        if n == 0:
            print(f"{C_DIM}No items matched '{kw}'.{C_RESET}")
        else:
            print(f"{C_GREEN}Forgot {n} item(s).{C_RESET}")

    def _cmd_profile(self) -> None:
        prof = self.memory.get_profile()
        if not prof:
            print(f"{C_DIM}(profile empty){C_RESET}")
            return
        for k, v in prof.items():
            print(f"  {k}: {v}")

    def _cmd_set(self, arg: str) -> None:
        parts = arg.strip().split(None, 1)
        if len(parts) != 2:
            print(f"{C_RED}Usage:{C_RESET} /set <field> <value>")
            print(f"{C_DIM}Fields:{C_RESET} user_name ai_name relationship tone language")
            return
        key, value = parts
        ok, reason = self.memory.set_profile(key, value)
        if ok:
            print(f"{C_GREEN}Set:{C_RESET} {key} = {value}")
        else:
            print(f"{C_DIM}({reason}){C_RESET}")

    def _cmd_unset(self, arg: str) -> None:
        key = arg.strip()
        if not key:
            print(f"{C_RED}Usage:{C_RESET} /unset <field>")
            return
        if self.memory.unset_profile(key):
            print(f"{C_GREEN}Unset:{C_RESET} {key}")
        else:
            print(f"{C_DIM}Field not set: {key}{C_RESET}")

    def _cmd_edit(self) -> None:
        import subprocess
        path = str(self.memory.path)
        if not self.memory.path.exists():
            self.memory.save()
        print(f"{C_DIM}Opening {path} in nano (Ctrl+O, Enter, Ctrl+X to save+exit){C_RESET}")
        try:
            subprocess.call(["nano", path])
        except FileNotFoundError:
            print(f"{C_RED}nano not found.{C_RESET} Try: pkg install nano")
            return
        # Reload after edit
        self.memory = Memory.load()
        for m in self.modes.values():
            m.memory = self.memory
        print(f"{C_GREEN}Reloaded memory.{C_RESET}")

    @staticmethod
    def _parse_code_args(arg: str) -> tuple:
        """Split `--backend <name>` out of a /code argument line.

        Returns (workdir, backend_name_or_None). The flag may sit before or
        after the path, and a bare `--backend` with nothing after it is
        reported rather than quietly dropped.
        """
        tokens = (arg or "").split()
        parts: list = []
        backend = None
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "--backend":
                if i + 1 >= len(tokens):
                    raise ValueError("--backend needs a name")
                backend = tokens[i + 1]
                i += 2
                continue
            if tok.startswith("--backend="):
                backend = tok.split("=", 1)[1]
                i += 1
                continue
            parts.append(tok)
            i += 1
        return " ".join(parts), (backend or None)

    def _cmd_code(self, arg: str) -> None:
        """Launch the Code-mode TUI in the given (or default) directory."""
        import subprocess
        import urllib.request

        from core import code_backends

        # 1) Workdir and backend: /code [--backend <name>] [dir]
        try:
            rest, backend_arg = self._parse_code_args(arg)
        except ValueError as e:
            print(f"{C_RED}{e}{C_RESET}")
            print(f"{C_DIM}Usage: /code [--backend openclaude] [dir]{C_RESET}")
            return
        if backend_arg and backend_arg not in code_backends.names():
            print(f"{C_RED}Unknown backend:{C_RESET} {backend_arg}")
            print(f"{C_DIM}Available: {', '.join(code_backends.names())}"
                  f"{C_RESET}")
            return
        b = code_backends.get(backend_arg or code_backends.default_name())
        if b is None:
            print(f"{C_RED}Unknown backend:{C_RESET} {backend_arg}")
            return

        workdir = rest.strip() or str(Path.home() / "opencode-work")
        wd = Path(workdir).expanduser()
        if not wd.exists():
            print(f"{C_RED}Directory not found:{C_RESET} {wd}")
            print(f"{C_DIM}Create it or specify another: /code ~/myproject{C_RESET}")
            return
        if not wd.is_dir():
            print(f"{C_RED}Not a directory:{C_RESET} {wd}")
            return

        # 2) the selected backend must be installed
        if not b.is_installed():
            print(f"{C_RED}{b.BINARY} not found.{C_RESET}")
            print(f"{C_DIM}Install:{C_RESET}")
            print(f"  {b.missing_install_message()}")
            return

        # 3) Check Flask server
        server_ok = False
        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:8080/", timeout=2
            ) as resp:
                server_ok = (resp.status == 200)
        except Exception:
            server_ok = False

        if not server_ok:
            print(f"{C_RED}Server not running.{C_RESET}")
            print(f"{C_DIM}Start it in another session:{C_RESET}")
            print("  cd ~/ai-agent && nohup python server.py > ~/server.log 2>&1 &")
            return

        # 4) Unload ai-agent engine (free RAM for opencode + server)
        if self.llm.is_loaded:
            print(f"{C_DIM}Unloading engine (free RAM)...{C_RESET}")
            for m in self.modes.values():
                try:
                    m.close()
                except Exception as e:
                    log.debug("mode close failed before code launch: %s: %s",
                              type(e).__name__, e)
            try:
                self.llm.unload()
            except Exception as e:
                log.debug("engine unload failed before code launch: %s: %s",
                          type(e).__name__, e)

        # 5) Ensure git identity exists (opencode commits changes)
        try:
            check = subprocess.run(
                ["git", "config", "--global", "user.email"],
                capture_output=True, text=True, timeout=5,
            )
            if check.returncode != 0 or not check.stdout.strip():
                subprocess.run(
                    ["git", "config", "--global", "user.email", "ai-agent@localhost"],
                    capture_output=True, timeout=5,
                )
                subprocess.run(
                    ["git", "config", "--global", "user.name", "ai-agent"],
                    capture_output=True, timeout=5,
                )
                print(f"{C_DIM}Git identity auto-set (ai-agent@localhost){C_RESET}")
        except Exception as e:
            # Without an identity opencode's commits fail later, far away
            # from where they should have been reported.
            log.warning("could not configure git identity: %s: %s",
                        type(e).__name__, e)

        # 6) Launch the selected backend against its own environment
        try:
            env = b.env()
        except Exception as e:
            print(f"{C_RED}Could not prepare {b.BINARY}:{C_RESET} {e}")
            return

        print(f"{C_CYAN}Launching {b.BINARY} in {wd}{C_RESET}")
        print(f"{C_DIM}Ctrl+C twice, then 'exit', to return.{C_RESET}\n")

        # opencode takes the folder as a positional; openclaude's positional
        # slot is the prompt, so the cwd above is what scopes it.
        argv = ([str(a) for a in b.tui_command(wd)]
                if hasattr(b, "tui_command")
                else [str(b.BINARY), "--model", b.MODEL_REF])
        try:
            rc = subprocess.call(
                argv,
                cwd=str(wd),
                env=env,
            )
        except FileNotFoundError:
            print(f"{C_RED}{b.BINARY} not found.{C_RESET}")
            print(f"  {b.missing_install_message()}")
            return
        except KeyboardInterrupt:
            print(f"\n{C_DIM}({b.BINARY} interrupted){C_RESET}")
            rc = -1

        print(f"\n{C_GREEN}Returned from {b.BINARY}{C_RESET} (rc={rc})")
        print(f"{C_DIM}Engine will reload on next LLM use.{C_RESET}")

    def _ctx_info(self) -> tuple:
        """Return (used_tokens, total_tokens, pct)."""
        total = self.config.context_size
        # Bug #4: Server-truth first (HTTP mode)
        try:
            if hasattr(self.mode, "get_server_ctx"):
                srv = self.mode.get_server_ctx()
                if srv and isinstance(srv, dict) and "used" in srv:
                    used = int(srv["used"])
                    total = int(srv.get("total", total))
                    pct = (used / total * 100) if total > 0 else 0
                    return used, total, pct
        except Exception as e:
            log.debug("server context probe failed; using local estimate: %s: %s",
                      type(e).__name__, e)
        # Fallback: local estimate
        try:
            msgs = self.mode.get_messages() if hasattr(self.mode, "get_messages") else []
            used = estimate_messages_tokens(msgs)
        except Exception:
            used = 0
        pct = (used / total * 100) if total > 0 else 0
        return used, total, pct

    def _ctx_color(self, pct: float) -> str:
        if pct < 50:
            return C_GREEN
        if pct < 75:
            return C_YELLOW
        return C_RED

    def _ctx_meter(self) -> str:
        used, total, pct = self._ctx_info()
        color = self._ctx_color(pct)
        used_k = used / 1000
        total_k = total / 1000
        return f"{C_DIM}[ctx: {color}{used_k:.1f}K{C_DIM}/{total_k:.0f}K {color}{pct:.0f}%{C_DIM}]{C_RESET}"

    def _cmd_ctx(self) -> None:
        used, total, pct = self._ctx_info()
        color = self._ctx_color(pct)
        print(f"  Context: {color}{used} / {total} tokens ({pct:.1f}%){C_RESET}")
        try:
            msgs = self.mode.get_messages()
            print(f"  Messages: {len(msgs)}")
            print(f"  Mode: {self.mode_name}")
            print(f"  Session: {getattr(self.mode, 'session_id', '?')}")
        except Exception as e:
            log.debug("could not read session info for /ctx: %s: %s",
                      type(e).__name__, e)

    def _cmd_context_size(self, size: int) -> None:
        if size == self.config.context_size:
            print(f"{C_DIM}Already at {size // 1024}K{C_RESET}")
            return
        print(f"{C_DIM}Switching to {size // 1024}K — unloading engine...{C_RESET}")
        # Close modes + unload engine (session data is on disk)
        for m in self.modes.values():
            try:
                m.close()
            except Exception as e:
                log.debug("mode close failed during context resize: %s: %s",
                          type(e).__name__, e)
        try:
            self.llm.unload()
        except Exception as e:
            log.debug("engine unload failed during context resize: %s: %s",
                      type(e).__name__, e)
        # Update config
        self.config.context_size = size
        try:
            self.config.save()
        except Exception as e:
            print(f"{C_RED}Save failed:{C_RESET} {e}")
            return
        print(f"{C_GREEN}✓ Context set to {size // 1024}K{C_RESET}")
        print(f"{C_DIM}Next message will reload engine (~10-15s){C_RESET}")
        print(f"{C_DIM}Session history preserved.{C_RESET}")

    def _cmd_summary(self) -> None:
        if not hasattr(self.mode, "compress_session"):
            print(f"{C_RED}Mode does not support /summary{C_RESET}")
            return
        msgs = self.mode.get_messages() if hasattr(self.mode, "get_messages") else []
        n = len(msgs)
        print(f"{C_DIM}(Summarizing {n} messages...){C_RESET}")
        try:
            r = self.mode.compress_session()
        except Exception as e:
            print(f"{C_RED}Error:{C_RESET} {type(e).__name__}: {e}")
            return
        if "error" in r:
            print(f"{C_RED}Failed:{C_RESET} {r['error']}")
            return
        print()
        print(f"{C_GREEN}--- Summary ---{C_RESET}")
        print(r.get("summary", ""))
        print(f"{C_GREEN}---------------{C_RESET}")
        print()
        before = r.get("before", 0)
        after = r.get("after", 0)
        saved = r.get("saved_pct", 0)
        print(f"{C_GREEN}✓ Session compressed:{C_RESET} {before} → {after} tokens ({saved}% saved)")
        if r.get("archived"):
            import os as _os
            print(f"{C_GREEN}✓ Archived:{C_RESET} {_os.path.basename(r['archived'])}")

    def _cmd_sessions(self) -> None:
        if SessionStore is None:
            print(f"{C_RED}sessions module not available{C_RESET}")
            return
        store = SessionStore()
        items = store.list_sessions(self.mode_name)
        if not items:
            print(f"{C_DIM}(no sessions){C_RESET}")
            return
        active = getattr(self.mode, "session_id", None)
        print(f"{C_GREEN}Sessions [{self.mode_name}]:{C_RESET}")
        for s in items:
            mark = f"{C_GREEN}*{C_RESET}" if s["id"] == active else " "
            title = s["title"][:50]
            print(f"  {mark} {s['id']}")
            print(f"      {C_DIM}{title}  ({s['count']} msgs){C_RESET}")
        print()

    def _cmd_switch_session(self, arg: str) -> None:
        if SessionStore is None:
            print(f"{C_RED}sessions module not available{C_RESET}")
            return
        sid = arg.strip()
        if not sid:
            print(f"{C_RED}Usage:{C_RESET} /switch <session-id>")
            return
        # try exact, else prefix match
        store = SessionStore()
        items = store.list_sessions(self.mode_name)
        match = None
        for s in items:
            if s["id"] == sid:
                match = s["id"]; break
        if not match:
            for s in items:
                if s["id"].startswith(sid):
                    match = s["id"]; break
        if not match:
            print(f"{C_RED}No session found:{C_RESET} {sid}")
            return
        if hasattr(self.mode, "reload_session") and self.mode.reload_session(match):
            print(f"{C_GREEN}Switched to:{C_RESET} {match}")
        else:
            print(f"{C_RED}Switch failed{C_RESET}")

    def _cmd_delete(self, arg: str) -> None:
        if SessionStore is None:
            print(f"{C_RED}sessions module not available{C_RESET}")
            return
        sid = arg.strip()
        if not sid:
            print(f"{C_RED}Usage:{C_RESET} /delete <session-id>")
            return
        store = SessionStore()
        items = store.list_sessions(self.mode_name)
        match = None
        for s in items:
            if s["id"] == sid:
                match = s["id"]; break
        if not match:
            for s in items:
                if s["id"].startswith(sid):
                    match = s["id"]; break
        if not match:
            print(f"{C_RED}No session found:{C_RESET} {sid}")
            return
        try:
            ok = store.delete_session(match)
        except Exception as e:
            log.error("could not delete session %s: %s: %s",
                      match, type(e).__name__, e)
            ok = False
        if not ok:
            print(f"{C_RED}Delete failed{C_RESET}")
            return
        print(f"{C_GREEN}Deleted:{C_RESET} {match}")
        # If active was deleted, reload current mode
        active = getattr(self.mode, "session_id", None)
        if active == match:
            new_active = store.get_active_id(self.mode_name)
            if new_active and hasattr(self.mode, "reload_session"):
                self.mode.reload_session(new_active)
            else:
                self.mode.reset()

    def _cmd_new(self) -> None:
        # C5d: HTTP mode — server owns session state; proxy to /api/new
        try:
            from core.http_llm import HTTPLLM
            is_http = isinstance(self.llm, HTTPLLM)
        except Exception:
            is_http = False
        if is_http:
            try:
                from core.http_llm import _post, CHAT_TIMEOUT
                j = _post(
                    f"{self.llm.server_url}/api/new",
                    {"mode": self.mode_name},
                    timeout=CHAT_TIMEOUT,
                )
                archived = j.get("archived")
                if archived:
                    print(f"{C_GREEN}Archived:{C_RESET} {archived}")
            except Exception as e:
                print(f"{C_RED}New chat failed: {e}{C_RESET}")
                return
            if hasattr(self.mode, "reset"):
                self.mode.reset()
            print(f"{C_GREEN}New chat started ({self.mode_name}).{C_RESET}")
            return
        # Local (standalone) mode — original behavior
        if SessionStore is None:
            print(f"{C_RED}sessions module not available{C_RESET}")
            return
        store = SessionStore()
        archived = store.archive_mode(self.mode_name)
        self.mode.reset()
        if archived:
            print(f"{C_GREEN}Archived:{C_RESET} {archived}")
        print(f"{C_GREEN}New chat started ({self.mode_name}).{C_RESET}")

    def _cmd_history(self, n: int = 10) -> None:
        msgs = []
        try:
            sess = getattr(self.mode, "session", None)
            if sess is not None and hasattr(sess, "get_messages"):
                msgs = sess.get_messages()
            elif hasattr(self.mode, "_messages"):
                msgs = self.mode._messages
        except Exception:
            msgs = []
        if not msgs:
            print(f"{C_DIM}(no history){C_RESET}")
            return
        recent = msgs[-n * 2:]
        print(f"{C_GREEN}Last {len(recent) // 2} turn(s) [{self.mode_name}]:{C_RESET}")
        for m in recent:
            role = m.get("role", "?")
            text = (m.get("text", "") or "")[:200]
            marker = f"{C_CYAN}You:{C_RESET}" if role == "user" else f"{C_GREEN}AI:{C_RESET}"
            print(f"  {marker} {text}")
        print()

    def _cmd_exit(self) -> None:
        self.running = False
        print(f"{C_DIM}Bye.{C_RESET}")

    def _cmd_logs(self, n: int = 30) -> None:
        """Tail the log file — the only place silent failures show up."""
        lines = log_tail(n)
        if not lines:
            print(f"{C_DIM}No log yet: {log_path()}{C_RESET}")
            return
        print(f"{C_DIM}── {log_path()} ──{C_RESET}")
        for ln in lines:
            print(ln.rstrip("\n"))
        print()

    def _handle_command(self, line: str) -> bool:
        stripped = line.strip()
        cmd = stripped.lower()
        if cmd == "/help":
            self._cmd_help(); return True
        if cmd == "/agent":
            self._cmd_switch("agent"); return True
        if cmd == "/clear":
            self._cmd_clear(); return True
        if cmd == "/mode":
            self._cmd_mode(); return True
        if cmd == "/memory":
            self._cmd_memory(); return True
        if cmd == "/new":
            self._cmd_new(); return True
        if cmd == "/sessions":
            self._cmd_sessions(); return True
        if cmd.startswith("/switch"):
            self._cmd_switch_session(stripped[len("/switch"):]); return True
        if cmd.startswith("/delete"):
            self._cmd_delete(stripped[len("/delete"):]); return True
        if cmd == "/history":
            self._cmd_history(); return True
        if cmd == "/summary":
            self._cmd_summary(); return True
        if cmd == "/ctx":
            self._cmd_ctx(); return True
        if cmd in ("/8k", "/8192"):
            self._cmd_context_size(8192); return True
        if cmd in ("/12k", "/12288"):
            self._cmd_context_size(12288); return True
        if cmd in ("/16k", "/16384"):
            self._cmd_context_size(16384); return True
        if cmd == "/profile":
            self._cmd_profile(); return True
        if cmd == "/edit":
            self._cmd_edit(); return True
        if cmd == "/logs":
            self._cmd_logs(); return True
        if cmd.startswith("/logs"):
            arg = stripped[len("/logs"):].strip()
            try:
                n = max(1, min(500, int(arg)))
            except ValueError:
                n = 30
            self._cmd_logs(n); return True
        if cmd.startswith("/code"):
            self._cmd_code(stripped[len("/code"):]); return True
        if cmd.startswith("/remember"):
            self._cmd_remember(stripped[len("/remember"):]); return True
        if cmd.startswith("/forget"):
            self._cmd_forget(stripped[len("/forget"):]); return True
        if cmd.startswith("/set "):
            self._cmd_set(stripped[len("/set"):]); return True
        if cmd.startswith("/unset "):
            self._cmd_unset(stripped[len("/unset"):]); return True
        if cmd in ("/exit", "/quit"):
            self._cmd_exit(); return True
        if cmd == "/skills":
            self._cmd_skills_list(); return True
        if cmd == "/skill":
            self._cmd_skill_run(""); return True
        if cmd.startswith("/skill "):
            self._cmd_skill_run(stripped[len("/skill"):]); return True
        if cmd.startswith("/"):
            print(f"{C_RED}Unknown command:{C_RESET} {cmd}. Try /help")
            return True
        return False

    def _cmd_skills_list(self) -> None:
        """List all available skills."""
        print()
        print(format_list())
        print()

    def _cmd_skill_run(self, arg: str) -> None:
        """Run a skill: /skill <name> <args>."""
        parts = arg.strip().split(None, 1)
        if not parts:
            print(f"{C_RED}Usage:{C_RESET} /skill <name> <args>")
            print(f"{C_DIM}Try /skills to list available skills.{C_RESET}")
            return
        name = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        sk = load_skill(name)
        if not sk:
            print(f"{C_RED}Unknown skill:{C_RESET} /{name}")
            print(f"{C_DIM}Try /skills to list.{C_RESET}")
            return
        if not args.strip():
            print(f"{C_RED}Missing args:{C_RESET} /skill {name} <{sk['args']}>")
            print(f"{C_DIM}{sk['description']}{C_RESET}")
            return
        rendered = render(sk, args)
        print(f"{C_DIM}Running skill: {name}...{C_RESET}")
        try:
            reply = self.mode.handle(rendered)
            print()
            print(reply)
            print()
        except Exception as e:
            print(f"{C_RED}Skill error:{C_RESET} {type(e).__name__}: {e}")

    # ---------- Command Menu ----------

    def _show_command_menu(self) -> str | None:
        """Show interactive menu. Returns chosen line or None."""
        try:
            import questionary
        except ImportError:
            print(f"{C_DIM}(questionary not installed; type commands manually){C_RESET}")
            return None

        try:
            choices = [
                questionary.Choice(title=f"{cmd:12} {desc}", value=cmd)
                for cmd, desc in COMMAND_MENU
            ]
            selected = questionary.select(
                "Select command (↑↓ + Enter):",
                choices=choices,
                qmark="",
            ).ask()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{C_DIM}(cancelled){C_RESET}")
            return None

        if not selected:
            return None

        # Arg-required → prompt
        if selected in ARG_REQUIRED:
            try:
                if selected == "/set":
                    hint = "field value"
                elif selected == "/unset":
                    hint = "field"
                elif selected == "/remember":
                    hint = "text"
                else:  # /forget
                    hint = "keyword"
                arg = input(f"  {selected} [{hint}] > ").strip()
            except (KeyboardInterrupt, EOFError):
                print(f"\n{C_DIM}(cancelled){C_RESET}")
                return None
            if not arg:
                print(f"{C_DIM}(empty — cancelled){C_RESET}")
                return None
            return f"{selected} {arg}"

        return selected

    # ---------- Banner ----------

    def _banner(self) -> None:
        ctx_k = self.config.context_size // 1024
        print("=" * 60)
        print(f"  {C_CYAN}ai-agent{C_RESET} — Phase 2 (Chat + Agent)")
        print("=" * 60)
        print(f"  Model:   {Path(self.config.model_path).name}")
        print(f"  Context: {self.config.context_size} ({ctx_k}K)")
        print(f"  Backend: {self.config.backend} {C_DIM}(lazy load){C_RESET}")
        print(f"  Modes:   {', '.join(self.modes.keys())}")
        print(f"  {C_DIM}Type /help for commands, /exit to quit.{C_RESET}")
        print("=" * 60)

    # ---------- Main Loop ----------

    def run(self) -> int:
        if not self.llm.health_check():
            print(f"{C_RED}Server not running.{C_RESET}")
            print(f"{C_DIM}Start:  python3 server.py  (in the project folder){C_RESET}")
            return 1
        self._banner()
        while self.running:
            try:
                line = input(self._prompt())
            except (EOFError, KeyboardInterrupt):
                print(f"\n{C_DIM}Interrupted.{C_RESET}")
                break

            if not line.strip():
                continue

            # Slash-alone triggers command menu
            if line.strip() == "/":
                picked = self._show_command_menu()
                if not picked:
                    continue
                line = picked

            if self._handle_command(line):
                continue

            # Auto-extract profile from user message
            try:
                extracted = extract_profile(line)
                prof = extracted.get("profile", {})
                if prof:
                    applied = []
                    for k, v in prof.items():
                        ok, _reason = self.memory.set_profile(k, v)
                        if ok:
                            applied.append(f"{k}={v}")
                    if applied:
                        print(f"{C_DIM}(remembered: {', '.join(applied)}){C_RESET}")
            except Exception as e:
                # Never break the chat, but don't drop the fact silently either.
                log.warning("could not extract profile from message: %s: %s",
                            type(e).__name__, e)

            try:
                reply = self.mode.handle(line)
                print(f"\n{reply}\n")
                # Context meter
                used, total, pct = self._ctx_info()
                print(self._ctx_meter())
                if pct >= 80:
                    print(f"{C_YELLOW}⚠ Context {pct:.0f}% full — consider /summary{C_RESET}")
                print()
            except LLMError as e:
                print(f"{C_RED}LLM error:{C_RESET} {e}")
            except ValueError as e:
                print(f"{C_RED}Input error:{C_RESET} {e}")
            except Exception as e:
                print(f"{C_RED}Error:{C_RESET} {type(e).__name__}: {e}")

        # Cleanup
        for m in self.modes.values():
            try:
                m.close()
            except Exception as e:
                log.debug("mode close failed at shutdown: %s: %s",
                          type(e).__name__, e)
        try:
            self.llm.unload()
        except Exception as e:
            log.debug("engine unload failed at shutdown: %s: %s",
                      type(e).__name__, e)
        return 0


def main() -> int:
    return App().run()


if __name__ == "__main__":
    raise SystemExit(main())
