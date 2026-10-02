"""DuckDuckGo fallback backend — last resort for general queries.

WARNING: DDG lite rate-limits at ~2-3 requests/30s.
Throttled with _MIN_INTERVAL, and transient failures (429/5xx/DNS) are
retried with backoff by core.retry. A total failure still returns [] — this
backend is intentionally the last resort and must never raise into the router.
"""
from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
import urllib.error
from html.parser import HTMLParser
from typing import List

from core.log import get_logger
from core.retry import RetryError, call_with_retry
from core.search.base import BaseBackend, Result

log = get_logger("search.ddg")

_UA = "Mozilla/5.0 (Linux; Android 11; Termux; ai-agent)"
_ENDPOINT = "https://lite.duckduckgo.com/lite/"
_TIMEOUT = 8
_MIN_INTERVAL = 3.0  # sec between calls (rate-limit guard)

# Module-level: last call timestamp
_last_call = 0.0


class _DDGParser(HTMLParser):
    """Parse DDG lite: <a class="result-link" href> + <td class="result-snippet">."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results = []
        self._in_link = False
        self._in_snippet = False
        self._cur = None
        self._buf = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = (a.get("class") or "").lower()
        if tag == "a" and "result-link" in cls:
            href = a.get("href", "")
            if href.startswith("/l/?") or "uddg=" in href:
                try:
                    q = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                    href = q.get("uddg", [href])[0]
                except Exception as e:
                    log.debug("could not unwrap DDG redirect %r: %s", href, e)
            self._cur = {"title": "", "url": href, "snippet": ""}
            self._in_link = True
            self._buf = []
        elif tag in ("td", "div") and "result-snippet" in cls:
            self._in_snippet = True
            self._buf = []

    def handle_endtag(self, tag):
        if self._in_link and tag == "a":
            if self._cur is not None:
                self._cur["title"] = "".join(self._buf).strip()
            self._in_link = False
            self._buf = []
        elif self._in_snippet and tag in ("td", "div"):
            if self._cur is not None:
                self._cur["snippet"] = "".join(self._buf).strip()
                if self._cur["title"] and self._cur["url"]:
                    self.results.append(self._cur)
                self._cur = None
            self._in_snippet = False
            self._buf = []

    def handle_data(self, data):
        if self._in_link or self._in_snippet:
            self._buf.append(data)


class DDGBackend(BaseBackend):
    name = "DuckDuckGo"
    category = "general"
    lang = "en"

    def search(self, query: str, max_results: int = 3) -> List[Result]:
        global _last_call
        q = (query or "").strip()
        if not q:
            return []

        # Rate guard. Previously this silently returned [] whenever two calls
        # landed within 3s, which looked identical to "DDG is down" — now we
        # wait out the remainder (bounded) and say so in the log.
        wait = _MIN_INTERVAL - (time.time() - _last_call)
        if wait > 0:
            log.debug("rate guard: sleeping %.1fs before DDG call", wait)
            time.sleep(min(wait, _MIN_INTERVAL))
        _last_call = time.time()

        try:
            n = max(1, min(10, int(max_results)))
        except Exception:
            n = 3

        url = _ENDPOINT + "?q=" + urllib.parse.quote(q)

        def _once() -> bytes:
            req = urllib.request.Request(url, headers={
                "User-Agent": _UA,
                "Accept-Language": "en-US,en;q=0.9",
            })
            r = urllib.request.urlopen(req, timeout=_TIMEOUT)
            status = getattr(r, "status", 200)
            if status != 200:
                raise urllib.error.HTTPError(url, status, "unexpected status",
                                             r.headers, None)
            return r.read(120000)

        try:
            html = call_with_retry(
                _once, attempts=3, base_delay=1.0, max_delay=8.0,
                budget=20.0, what="duckduckgo search",
            ).decode("utf-8", errors="replace")
        except RetryError as e:
            log.warning("duckduckgo gave up: %s", e)
            return []
        except Exception as e:
            log.warning("duckduckgo request failed: %s: %s", type(e).__name__, e)
            return []

        parser = _DDGParser()
        try:
            parser.feed(html)
        except Exception as e:
            log.warning("could not parse duckduckgo response: %s: %s",
                        type(e).__name__, e)
            return []

        out: List[Result] = []
        for h in parser.results[:n]:
            snip = re.sub(r"<[^>]+>", "", h.get("snippet", "")).strip()
            out.append(Result(
                title=h["title"],
                url=h["url"],
                snippet=snip[:400],
                source="DuckDuckGo",
                category="general",
                lang="en",
            ))
        return out


__all__ = ["DDGBackend"]
