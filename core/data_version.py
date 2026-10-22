"""One versioning rule for the app's JSON data files.

Every store that keeps user data in JSON carries a top-level
``"version"`` integer: memory (1), sessions (2 — it has already
migrated once), code projects (1), model limits (1), the provider
keys file (1, from its default config — stamped but, before
v1.10.0, never checked), and since v1.10.0 the server config.

Files written before versioning have no key at all; by definition
those are version 1, the format everything understood then. A
loader that meets a HIGHER version than it knows is reading data
written by a newer build (a rollback, a copied file) — the
merge-based loaders would silently drop what they do not
understand, so the one hard rule is: notice it, and say so in
the log.
"""
from __future__ import annotations

VERSION_KEY = "version"
UNVERSIONED = 1


def file_version(data) -> int:
    """The schema version of a loaded JSON document (dict-shaped)."""
    if isinstance(data, dict):
        v = data.get(VERSION_KEY)
        if isinstance(v, int) and not isinstance(v, bool) and v >= 1:
            return v
    return UNVERSIONED


def warn_if_newer(name: str, data, current: int, log) -> int:
    """Log a warning when *data* was written by a newer build."""
    v = file_version(data)
    if v > current:
        log.warning(
            "%s has data version %d but this build understands %d — "
            "reading it anyway; fields it does not know will be "
            "ignored and dropped on the next save", name, v, current)
    return v
