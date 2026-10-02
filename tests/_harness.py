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
    """Isolate ~/.ai-agent. Call before importing config/server."""
    home = Path(tempfile.mkdtemp(prefix="ai-agent-tests-"))
    os.environ["HOME"] = str(home)
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
    pointing at the first temp home.
    """
    cfg = sys.modules.get("config")
    if cfg is None:
        return
    cfg.CONFIG_DIR = home / ".ai-agent"
    cfg.CONFIG_FILE = cfg.CONFIG_DIR / "config.json"


def bootstrap(tmp_home: bool = True) -> Path:
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