"""Wikipedia backend — search + clean summary (EN + Burmese).

Uses:
  - Search API:   /w/api.php?action=query&list=search
  - REST Summary: /api/rest_v1/page/summary/<title>

Free, no rate limit. Supports EN and Burmese (my).
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
import urllib.error
from typing import List

from core.log import get_logger
from core.retry import RetryError, get_json
from core.search.base import BaseBackend, Result

log = get_logger("search.wikipedia")

_UA = "ai-agent/1.0 (Termux; https://github.com/ai-agent)"
_BURMESE_RE = re.compile(r"[\u1000-\u109F]")


def _is_burmese(text: str) -> bool:
    return bool(_BURMESE_RE.search(text or ""))


def _fetch_json(url: str, timeout: int = 10) -> dict:
    """Fetch a URL and parse JSON. Returns {} on failure (already retried)."""
    try:
        raw = get_json(url, headers={"User-Agent": _UA}, timeout=timeout,
                       max_bytes=50000, what="wikipedia fetch")
        return json.loads(raw.decode("utf-8", errors="replace"))
    except RetryError as e:
        log.warning("giving up on %s: %s", url, e)
        return {}
    except Exception as e:
        log.warning("bad response from %s: %s: %s", url, type(e).__name__, e)
        return {}


class WikipediaBackend(BaseBackend):
    name = "Wikipedia"
    category = "knowledge"
    lang = "en"
    prefer_raw = True   # auto-detect MM/EN from raw query

    def search(self, query: str, max_results: int = 3) -> List[Result]:
        """Search Wikipedia (EN or MM depending on query)."""
        q = (query or "").strip()
        if not q:
            return []

        # Choose lang subdomain
        if _is_burmese(q):
            host = "my.wikipedia.org"
        else:
            host = "en.wikipedia.org"

        # Step 1: Search API
        search_url = (
            f"https://{host}/w/api.php"
            f"?action=query&list=search"
            f"&srsearch={urllib.parse.quote(q)}"
            f"&srlimit={min(max(1, max_results), 10)}"
            f"&format=json&utf8=1"
        )
        data = _fetch_json(search_url)
        hits = data.get("query", {}).get("search", [])
        if not hits:
            return []

        # Step 2: REST summary per hit (parallelism skipped for simplicity)
        out: List[Result] = []
        for h in hits[:max_results]:
            title = h.get("title", "")
            if not title:
                continue
            page_url = f"https://{host}/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
            summary_url = (
                f"https://{host}/api/rest_v1/page/summary/"
                + urllib.parse.quote(title.replace(" ", "_"))
            )
            sj = _fetch_json(summary_url)
            extract = sj.get("extract", "") or h.get("snippet", "")
            # Strip HTML if snippet came with tags
            extract = re.sub(r"<[^>]+>", "", extract).strip()

            out.append(Result(
                title=title,
                url=page_url,
                snippet=extract[:500],
                source=f"Wikipedia ({'MM' if host.startswith('my') else 'EN'})",
                category="knowledge",
                lang="my" if host.startswith("my") else "en",
            ))

        return out


__all__ = ["WikipediaBackend"]
