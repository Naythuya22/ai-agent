"""Shared test setup — import this FIRST, before anything from the project.

Does three things that must happen in a fixed order:

1. Redirects HOME to a throwaway dir, so a test run never touches the real
   ~/.ai-agent. ``config`` computes CONFIG_DIR at import time, so this has to
   happen before ``import config``.
2. Puts the litert_lm stub and the project root on sys.path, so the app imports
   on a machine with no LiteRT-LM (i.e. anywhere but the phone).
3. Points model_path at a file that exists, since core.llm refuses to load
   without one. Any *.litertlm will do — no weights are ever read by the stub.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent

_stub_dir = TESTS_DIR / "stubs"
if str(_stub_dir) not in sys.path:
    sys.path.insert(0, str(_stub_dir))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def use_temp_home() -> Path:
    """Isolate ~/.ai-agent. Call before importing config."""
    home = Path(tempfile.mkdtemp(prefix="ai-agent-tests-"))
    os.environ["HOME"] = str(home)
    # The key file no longer lives under $HOME — it has a fixed default path
    # (config.DEFAULT_DATA_DIR). Point the override at the throwaway HOME too,
    # so a config imported *after* this call still lands in the temp dir.
    os.environ["AI_AGENT_DATA_DIR"] = str(home / ".ai-agent")
    return home


def dummy_model(home: Path | None = None) -> Path:
    """Create a fake .litertlm and return its path (engine is stubbed)."""
    root = (home or Path(os.environ["HOME"])) / "fake-models"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "stub-model.litertlm"
    path.write_bytes(b"not a real model")
    return path


def _retarget_config_module(home: Path) -> None:
    """Re-point config's CONFIG_DIR/CONFIG_FILE at the new home.

    Those are computed at import time from os.environ["HOME"], so if config has
    already been imported (e.g. bootstrap() called twice) they would keep
    pointing at the first temp home. The same is true of DATA_DIR /
    AGENT_CONFIG_FILE, which are *not* HOME-derived — they need the explicit
    re-point here.

    core.projects has the same shape of bug: PROJECTS_FILE is bound to
    Path.home() at import, so the *previous* temp home's store — with
    whatever "active project" a suite left behind — would keep answering
    for every later suite. That leaks into the file tools' path jail
    (tools/file.py reads the active project on every call), so it is
    re-pointed here too.
    """
    proj = sys.modules.get("core.projects")
    if proj is not None:
        proj.PROJECTS_FILE = home / ".ai-agent" / "code_projects.json"

    cfg = sys.modules.get("config")
    if cfg is None:
        return
    cfg.CONFIG_DIR = home / ".ai-agent"
    cfg.CONFIG_FILE = cfg.CONFIG_DIR / "config.json"
    cfg.DATA_DIR = home / ".ai-agent"
    cfg.AGENT_CONFIG_FILE = cfg.DATA_DIR / "agent_config.json"


def bootstrap(tmp_home: bool = True) -> Path:
    # Batch 3.2b: reflection off — most suites pin API call counts
    # and the reflection step adds one call per turn. Dedicated
    # reflection tests (test_reflection.py) turn it back on locally.
    import config as _cfg
    _cfg.AGENT_REFLECT_ENABLED = False
    """Full setup: temp HOME + fake model config. Returns the model path.

    Safe to call more than once — each call gets a fresh HOME and config is
    re-targeted to match.
    """
    home = use_temp_home() if tmp_home else Path(os.environ["HOME"])
    _retarget_config_module(home)
    model = dummy_model(home)
    cfg_dir = home / ".ai-agent"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.json").write_text(
        '{"model_path": "%s", "backend": "CPU", "context_size": 8192}'
        % model
    )
    return model


def current_home() -> Path:
    """The HOME currently in effect (a temp dir unless bootstrap was told otherwise)."""
    return Path(os.environ["HOME"])


def get_app():
    """Import and return the Flask app (after bootstrap())."""
    import server

    server.app.config["TESTING"] = True
    return server.app