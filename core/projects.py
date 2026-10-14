"""Project store for Cursor-like folder management (Code mode)."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

from core.log import get_logger

log = get_logger("projects")

PROJECTS_FILE = Path.home() / ".ai-agent" / "code_projects.json"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _empty() -> dict:
    return {"version": 1, "projects": [], "active": None}


def load() -> dict:
    if not PROJECTS_FILE.exists():
        return _empty()
    try:
        data = json.loads(PROJECTS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _empty()
    if not isinstance(data.get("projects"), list):
        data["projects"] = []
    data.setdefault("active", None)
    data.setdefault("version", 1)
    # `backend` arrived with the second Code-mode backend; a store written
    # before it existed must still answer with a complete project, so the
    # routes never have to guard for a missing key.
    for p in data["projects"]:
        if isinstance(p, dict):
            p.setdefault("backend", None)
    return data


def _save(data: dict) -> None:
    PROJECTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = PROJECTS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(PROJECTS_FILE)


def list_all() -> list:
    return load().get("projects", [])


def get(pid: str) -> dict | None:
    for p in list_all():
        if p.get("id") == pid:
            return p
    return None


def get_active() -> dict | None:
    data = load()
    aid = data.get("active")
    if not aid:
        return None
    for p in data.get("projects", []):
        if p.get("id") == aid:
            return p
    return None


def add(name: str, path: str) -> dict:
    """Add a project. Path must exist and be a directory."""
    p = Path(path).expanduser().resolve()
    if not p.exists() or not p.is_dir():
        raise ValueError(f"Not a directory: {p}")

    data = load()
    # Duplicate check by resolved path
    for existing in data["projects"]:
        try:
            if Path(existing["path"]).resolve() == p:
                return existing
        except Exception as e:
            log.debug("could not resolve a stored project path: %s: %s",
                      type(e).__name__, e)

    proj = {
        "id": uuid.uuid4().hex[:8],
        "name": (name or p.name or "project").strip()[:40],
        "path": str(p),
        # None = follow AI_AGENT_CODE_BACKEND (opencode unless told otherwise).
        "backend": None,
        "created": _now(),
        "last_used": _now(),
    }
    data["projects"].append(proj)
    if not data.get("active"):
        data["active"] = proj["id"]
    _save(data)
    return proj


def remove(pid: str) -> bool:
    data = load()
    before = len(data["projects"])
    data["projects"] = [p for p in data["projects"] if p.get("id") != pid]
    if len(data["projects"]) == before:
        return False
    if data.get("active") == pid:
        data["active"] = data["projects"][0]["id"] if data["projects"] else None
    _save(data)
    return True


def set_active(pid: str) -> bool:
    data = load()
    for p in data["projects"]:
        if p.get("id") == pid:
            data["active"] = pid
            p["last_used"] = _now()
            _save(data)
            return True
    return False


def set_backend(pid: str, backend: str | None) -> bool:
    """Set the Code-mode backend for one project. False when it is unknown.

    An empty name clears the field rather than storing "", so the project
    goes back to following the default (AI_AGENT_CODE_BACKEND). The name is
    not checked against the registry here — this store is about persistence;
    routes/code.py validates before it calls.
    """
    name = str(backend or "").strip() or None
    data = load()
    for p in data["projects"]:
        if p.get("id") == pid:
            p["backend"] = name
            _save(data)
            return True
    return False
