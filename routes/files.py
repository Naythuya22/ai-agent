"""Project file explorer — /api/files/tree and /api/files/content (P0).

The UI's "📁 Project Files" panel asks exactly two questions: what is in the
selected project, and what does one file say. Both answers are jailed to the
project root with tools._fs_guard.is_safe_path, so a crafted `path` query
parameter can never read outside the folder the user picked.

Endpoints answer with a JSON body on every path — a missing project or a
1 MB file is an ordinary state to ask about, not a 500 traceback.
"""
from __future__ import annotations

from pathlib import Path

from flask import Blueprint, jsonify, request

from core import projects as projects_store
from tools._fs_guard import is_safe_path

bp = Blueprint("files", __name__)

# The viewer renders one file into one pane; past this it is a download job.
MAX_CONTENT_BYTES = 1024 * 1024

# Tree caps: a phone-sized project must not turn one request into a walk of
# a node_modules forest, so depth, breadth and the whole response are bounded.
TREE_MAX_DEPTH = 6
TREE_MAX_ENTRIES = 200            # per directory
TREE_MAX_NODES = 3000             # whole response

SKIP_DIRS = {
    ".git", ".hg", ".svn", ".idea", ".vscode", ".gradle",
    "node_modules", "__pycache__", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
}

# Extension → the class name the viewer's <code> element wants (P0 keeps the
# highlighting step for a later PR; the attribute must exist for it to hook).
LANG_BY_EXT = {
    "py": "python", "js": "javascript", "mjs": "javascript",
    "jsx": "javascript", "ts": "typescript", "tsx": "typescript",
    "json": "json", "md": "markdown", "html": "html", "htm": "html",
    "css": "css", "scss": "css", "sh": "bash", "bash": "bash",
    "txt": "text", "csv": "text", "log": "text", "yml": "yaml",
    "yaml": "yaml", "toml": "toml", "ini": "ini", "cfg": "ini",
    "xml": "xml", "svg": "xml", "c": "c", "h": "c", "cpp": "cpp",
    "hpp": "cpp", "java": "java", "kt": "kotlin", "go": "go",
    "rs": "rust", "rb": "ruby", "php": "php", "sql": "sql",
}


def _language_for(name: str) -> str:
    """The `language-…` suffix for one filename ('' → 'text')."""
    ext = Path(name).suffix.lower().lstrip(".")
    return LANG_BY_EXT.get(ext) or ext or "text"


def _resolve_root(pid):
    """(project dict, resolved root Path, error response). error is None.

    `project` is a registry id; an empty one means "the active project",
    which is what the panel sends once a project is selected.
    """
    pid = (pid or "").strip()
    if pid:
        proj = projects_store.get(pid)
        if not isinstance(proj, dict):
            return None, None, (jsonify({"error": "project not found"}), 404)
    else:
        proj = projects_store.get_active()
        if not isinstance(proj, dict):
            return None, None, (jsonify({"error": "no active project"}), 400)
    try:
        root = Path(proj.get("path") or "").expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return None, None, (jsonify({"error": "invalid project path"}), 400)
    if not root.exists() or not root.is_dir():
        return None, None, (
            jsonify({"error": f"project path missing: {root}"}), 400)
    return proj, root, None


def _rel_join(rel: str, name: str) -> str:
    return f"{rel}/{name}" if rel else name


def _rel_of(target: Path, root: Path) -> str:
    """root-relative posix path; falls back to the name if the prefixes
    ever diverge (both sides are resolved, so this should not happen)."""
    try:
        return target.relative_to(root).as_posix()
    except ValueError:
        return target.name


def _file_node(entry: Path, rel: str) -> dict:
    st = entry.stat()
    return {
        "name": entry.name,
        "type": "file",
        "path": rel,
        "size": st.st_size,
        "mtime": round(st.st_mtime, 3),
    }


def _dir_node(entry: Path, rel: str, depth: int, budget: list) -> dict:
    """One directory, recursively. `budget` counts down across the whole
    walk so one call can never emit more than TREE_MAX_NODES nodes."""
    node = {"name": entry.name, "type": "dir", "path": rel, "children": []}
    if depth >= TREE_MAX_DEPTH:
        node["truncated"] = True
        return node
    try:
        children = sorted(entry.iterdir(),
                          key=lambda e: (0 if e.is_dir() else 1,
                                         e.name.lower()))
    except PermissionError:
        node["error"] = "permission denied"
        return node
    except OSError as e:
        # SD cards unmount mid-walk; answer instead of raising a 500.
        node["error"] = f"{type(e).__name__}: {e}"
        return node

    shown = 0
    for child in children:
        if shown >= TREE_MAX_ENTRIES or budget[0] <= 0:
            node["truncated"] = True
            break
        try:
            if child.is_dir():
                if child.name in SKIP_DIRS or child.name.startswith("."):
                    continue
                budget[0] -= 1
                shown += 1
                node["children"].append(
                    _dir_node(child, _rel_join(rel, child.name),
                              depth + 1, budget))
            elif child.is_file():
                budget[0] -= 1
                shown += 1
                node["children"].append(
                    _file_node(child, _rel_join(rel, child.name)))
        except OSError:
            continue
    return node


@bp.route("/api/files/tree", methods=["GET"])
def api_files_tree():
    """Recursive project tree: folders first, then files with size/mtime."""
    proj, root, err = _resolve_root(request.args.get("project"))
    if err:
        return err
    budget = [TREE_MAX_NODES]
    try:
        tree = _dir_node(root, "", 0, budget)
    except OSError as e:
        return jsonify({"error": f"cannot list {root}: {e}"}), 500
    return jsonify({
        "ok": True,
        "project": {"id": proj.get("id"), "name": proj.get("name"),
                    "path": str(root)},
        "root": str(root),
        "tree": tree,
    })


@bp.route("/api/files/content", methods=["GET"])
def api_files_content():
    """Text of one project file — 1 MB cap, binary refused, path jailed."""
    raw = (request.args.get("path") or "").strip()
    if not raw:
        return jsonify({"error": "path required"}), 400
    _, root, err = _resolve_root(request.args.get("project"))
    if err:
        return err

    rel = Path(raw)
    # `..` is refused before resolve() so the reason reads clearly; the
    # is_safe_path check below still catches a symlink or an absolute path.
    if ".." in rel.parts:
        return jsonify({"error": f"unsafe path: {raw}"}), 403
    try:
        target = (rel if rel.is_absolute() else (root / rel))
        target = target.expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return jsonify({"error": "invalid path"}), 400
    if not is_safe_path(str(target), str(root)):
        return jsonify({"error": f"unsafe path: {raw}"}), 403
    if not target.exists():
        return jsonify({"error": f"not found: {raw}"}), 404
    if not target.is_file():
        return jsonify({"error": f"not a file: {raw}"}), 400

    try:
        st = target.stat()
        if st.st_size > MAX_CONTENT_BYTES:
            return jsonify({
                "error": f"file too large: {st.st_size} bytes "
                         f"(limit {MAX_CONTENT_BYTES})",
                "size": st.st_size,
                "limit": MAX_CONTENT_BYTES,
            }), 413
        data = target.read_bytes()
    except PermissionError:
        return jsonify({"error": f"permission denied: {raw}"}), 403
    except OSError as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    if b"\x00" in data[:8192]:
        return jsonify({"error": f"binary file: {raw}"}), 400
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return jsonify({"error": f"not UTF-8 text: {raw}"}), 400

    return jsonify({
        "ok": True,
        "path": _rel_of(target, root),
        "name": target.name,
        "language": _language_for(target.name),
        "size": len(data),
        "mtime": round(st.st_mtime, 3),
        "content": text,
    })
