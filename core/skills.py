"""Skills — reusable prompt templates (Kai-style).

Each skill is a markdown file in ~/.ai-agent/skills/ with YAML frontmatter:

    ---
    name: summarize
    description: Summarize text concisely
    args: text
    ---
    Body with {{args}} placeholder.

No external dependencies — regex frontmatter parsing.
"""
from __future__ import annotations

import re
from pathlib import Path

from core.log import get_logger

log = get_logger("skills")

SKILLS_DIR = Path.home() / ".ai-agent" / "skills"
_MAX_FILE_SIZE = 50 * 1024  # 50 KB
_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_FRONTMATTER_RE = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL
)


def _ensure_dir() -> None:
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)


def _sanitize_name(name: str) -> str:
    """Return a safe skill name, or empty string if invalid."""
    if not name:
        return ""
    name = name.strip().lower()
    # Accept letters, digits, dash, underscore
    if not _NAME_RE.match(name):
        return ""
    return name


def _parse_frontmatter(text: str) -> tuple:
    """Parse YAML-ish frontmatter → (meta_dict, body). Returns ({}, text) if none."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    raw_meta, body = m.group(1), m.group(2)
    meta = {}
    for line in raw_meta.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip().lower()
        v = v.strip().strip('"').strip("'")
        if k:
            meta[k] = v
    return meta, body


def list_skills() -> list:
    """Return a list of skill metadata dicts, sorted by name."""
    _ensure_dir()
    out = []
    for f in sorted(SKILLS_DIR.glob("*.md")):
        if f.name.startswith("_"):
            continue
        try:
            if f.stat().st_size > _MAX_FILE_SIZE:
                continue
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        meta, _ = _parse_frontmatter(text)
        name = meta.get("name") or f.stem
        out.append({
            "name": name,
            "description": meta.get("description", ""),
            "args": meta.get("args", "text"),
            "file": f.name,
        })
    return out


def load_skill(name: str) -> dict | None:
    """Load a single skill by name. Returns None if not found or invalid."""
    safe = _sanitize_name(name)
    if not safe:
        return None
    _ensure_dir()
    f = SKILLS_DIR / f"{safe}.md"
    if not f.is_file():
        return None
    try:
        if f.stat().st_size > _MAX_FILE_SIZE:
            return None
        text = f.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    meta, body = _parse_frontmatter(text)
    return {
        "name": meta.get("name") or safe,
        "description": meta.get("description", ""),
        "args": meta.get("args", "text"),
        "body": body.strip(),
    }


def render(skill: dict, args: str = "") -> str:
    """Substitute {{args}} in the skill body. Returns the final prompt string."""
    if not skill:
        return ""
    body = skill.get("body", "")
    return body.replace("{{args}}", args or "")


def build_markdown(name: str, description: str, body: str, args: str = "text") -> str:
    """Build the full markdown file content for a skill."""
    safe = _sanitize_name(name)
    if not safe:
        raise ValueError(f"invalid skill name: {name!r}")
    description = (description or "").replace("\n", " ").strip()
    if not description:
        raise ValueError("description is required")
    body = (body or "").strip()
    if not body:
        raise ValueError("body is required")
    if "{{args}}" not in body:
        body = body + "\n\n{{args}}"
    args = (args or "text").strip() or "text"
    fm = f"---\nname: {safe}\ndescription: {description}\nargs: {args}\n---\n\n"
    return fm + body + "\n"


def save_skill(name: str, description: str, body: str, args: str = "text") -> dict:
    """Save (or overwrite) a skill file atomically.

    Returns the saved skill's metadata dict. Raises ValueError if invalid.
    """
    content = build_markdown(name, description, body, args)
    safe = _sanitize_name(name)
    _ensure_dir()
    target = SKILLS_DIR / f"{safe}.md"
    tmp = SKILLS_DIR / f".{safe}.md.tmp"
    try:
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(target)
    except Exception:
        try:
            tmp.unlink()
        except Exception as e:
            log.debug("could not remove the temporary skill file: %s: %s",
                      type(e).__name__, e)
        raise
    return {"name": safe, "description": description.strip(), "args": args}


def delete_skill(name: str) -> bool:
    """Delete a skill file. Returns True if deleted, False if not found/invalid."""
    safe = _sanitize_name(name)
    if not safe:
        return False
    f = SKILLS_DIR / f"{safe}.md"
    if not f.is_file():
        return False
    try:
        f.unlink()
        return True
    except Exception:
        return False


def default_template(name: str, description: str) -> str:
    """Fallback template when LLM generation fails or returns invalid output."""
    name = _sanitize_name(name) or "skill"
    description = (description or "").strip() or f"A {name} task"
    return (
        f"You are a helpful assistant. {description}\n\n"
        "Provide a clear, accurate response. Use the user's language.\n\n"
        "Input:\n\n{{args}}"
    )


def format_list() -> str:
    """Human-readable skill list (for /skills command)."""
    skills = list_skills()
    if not skills:
        return "No skills yet. Create one with /create-skill (coming soon)."
    lines = ["Available skills:"]
    for s in skills:
        desc = s["description"] or "(no description)"
        lines.append(f"  /{s['name']:<20} {desc}")
    return "\n".join(lines)


__all__ = [
    "SKILLS_DIR", "list_skills", "load_skill", "render", "format_list",
    "build_markdown", "save_skill", "delete_skill", "default_template",
]
