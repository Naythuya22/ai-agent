"""Stack Exchange backend — search Stack Overflow Q&A."""
from __future__ import annotations

import gzip
import json
import re
import urllib.parse
import urllib.request
import urllib.error
from typing import List

from core.log import get_logger
from core.retry import RetryError, call_with_retry
from core.search.base import BaseBackend, Result

log = get_logger("search.stackexchange")

_UA = "ai-agent/1.0 (Termux)"
_API = "https://api.stackexchange.com/2.3/search/advanced"


def _fetch_gzip(url: str, timeout: int = 10) -> dict:
    """Fetch URL, decompress gzip, parse JSON. Returns {} on failure (retried)."""
    def _once() -> bytes:
        req = urllib.request.Request(url, headers={
            "User-Agent": _UA,
            "Accept-Encoding": "gzip",
        })
        r = urllib.request.urlopen(req, timeout=timeout)
        raw = r.read(80000)
        if (r.headers.get("Content-Encoding") or "").lower() == "gzip":
            try:
                raw = gzip.decompress(raw)
            except Exception as e:
                # Truncated body (we capped the read at 80 KB) — the JSON
                # below will fail to parse and the caller returns {}.
                log.debug("gzip decompress failed for %s: %s", url, e)
        return raw

    try:
        return json.loads(call_with_retry(
            _once, attempts=3, what="stackexchange fetch"
        ).decode("utf-8", errors="replace"))
    except RetryError as e:
        log.warning("giving up on %s: %s", url, e)
        return {}
    except Exception as e:
        log.warning("bad response from %s: %s: %s", url, type(e).__name__, e)
        return {}


def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "").strip()


class StackExchangeBackend(BaseBackend):
    name = "Stack Overflow"
    category = "coding"
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
            f"{_API}?order=desc&sort=relevance"
            f"&q={urllib.parse.quote(q)}"
            f"&site=stackoverflow&pagesize={n}&filter=default"
        )
        data = _fetch_gzip(url)
        items = data.get("items", []) or []
        if not items:
            return []

        out: List[Result] = []
        for it in items[:n]:
            title = _strip_html(it.get("title", ""))
            link = it.get("link", "")
            score = it.get("score", 0)
            ans = it.get("answer_count", 0)
            answered = it.get("is_answered", False)
            tags = it.get("tags", [])[:5]
            snippet = (
                f"Score: {score}, Answers: {ans} "
                f"({'answered' if answered else 'unanswered'}) "
                f"| Tags: {', '.join(tags)}"
            )
            out.append(Result(
                title=title,
                url=link,
                snippet=snippet,
                source="Stack Overflow",
                category="coding",
                lang="en",
                score=float(score),
            ))
        return out


__all__ = ["StackExchangeBackend"]
