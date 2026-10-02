"""OpenLibrary backend — book search (title, author, year, subjects)."""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
import urllib.error
from typing import List

from core.log import get_logger
from core.retry import RetryError, get_json
from core.search.base import BaseBackend, Result

log = get_logger("search.openlibrary")

_UA = "ai-agent/1.0 (Termux)"
_API = "https://openlibrary.org/search.json"


def _fetch_json(url: str, timeout: int = 10) -> dict:
    """Fetch a URL and parse JSON. Returns {} on failure (already retried)."""
    try:
        raw = get_json(url, headers={"User-Agent": _UA}, timeout=timeout,
                       max_bytes=80000, what="openlibrary fetch")
        return json.loads(raw.decode("utf-8", errors="replace"))
    except RetryError as e:
        log.warning("giving up on %s: %s", url, e)
        return {}
    except Exception as e:
        log.warning("bad response from %s: %s: %s", url, type(e).__name__, e)
        return {}


class OpenLibraryBackend(BaseBackend):
    name = "OpenLibrary"
    category = "book"
    lang = "en"

    def search(self, query: str, max_results: int = 3) -> List[Result]:
        q = (query or "").strip()
        if not q:
            return []
        try:
            n = max(1, min(10, int(max_results)))
        except Exception:
            n = 3

        url = (
            f"{_API}?q={urllib.parse.quote(q)}"
            f"&limit={n}"
            f"&fields=key,title,author_name,first_publish_year,subject,isbn"
        )
        data = _fetch_json(url)
        docs = data.get("docs", []) or []
        if not docs:
            return []

        out: List[Result] = []
        for d in docs[:n]:
            title = d.get("title", "").strip()
            if not title:
                continue
            authors = d.get("author_name", []) or []
            year = d.get("first_publish_year", "")
            subjects = (d.get("subject", []) or [])[:3]
            key = d.get("key", "") or ""
            page_url = f"https://openlibrary.org{key}" if key else ""

            snippet_parts = []
            if authors:
                snippet_parts.append(f"By {', '.join(authors[:2])}")
            if year:
                snippet_parts.append(f"First published: {year}")
            if subjects:
                snippet_parts.append(f"Subjects: {', '.join(subjects)}")

            out.append(Result(
                title=title,
                url=page_url,
                snippet=" | ".join(snippet_parts),
                source="OpenLibrary",
                category="book",
                lang="en",
            ))
        return out


__all__ = ["OpenLibraryBackend"]
