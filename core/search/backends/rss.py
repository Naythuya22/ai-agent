"""RSS backend — fetch and filter news from configured feeds.

Reads ~/.ai-agent/sources.json for category → feed mapping.
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List
from xml.etree import ElementTree as ET

from core.log import get_logger
from core.retry import RetryError, call_with_retry
from core.search.base import BaseBackend, Result

log = get_logger("search.rss")

_UA = "Mozilla/5.0 (Linux; Android 11; Termux; ai-agent)"
_SOURCES_FILE = Path.home() / ".ai-agent" / "sources.json"
_TIMEOUT = 8
_MAX_PER_FEED = 3

# ── F1: TTL cache for feed content ──
_FEED_CACHE: dict = {}          # {url: (timestamp, xml_bytes)}
_CACHE_TTL = 900                # 15 minutes
_CACHE_MAX = 50                 # cap on entries
_MAX_WORKERS = 4                # parallel fetch limit


def _load_sources() -> dict:
    try:
        if not _SOURCES_FILE.is_file():
            return {}
        return json.loads(_SOURCES_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("could not read %s (%s: %s) — no RSS feeds configured",
                    _SOURCES_FILE, type(e).__name__, e)
        return {}


def _strip_html(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _fetch(url: str, timeout: int = _TIMEOUT) -> bytes:
    def _once() -> bytes:
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        r = urllib.request.urlopen(req, timeout=timeout)
        return r.read(200000)

    try:
        return call_with_retry(_once, attempts=2, base_delay=0.3,
                               what=f"rss fetch {url}")
    except RetryError as e:
        log.warning("feed unreachable: %s", e)
        return b""
    except Exception as e:
        log.warning("feed fetch failed for %s: %s: %s", url, type(e).__name__, e)
        return b""


def _fetch_cached(url: str) -> bytes:
    """F1: Fetch with TTL cache. On fetch failure, return cached (stale) if any."""
    now = time.time()

    # Cache hit?
    if url in _FEED_CACHE:
        ts, data = _FEED_CACHE[url]
        if now - ts < _CACHE_TTL:
            return data

    # Fetch fresh
    xml = _fetch(url)

    if xml:
        _FEED_CACHE[url] = (now, xml)
        # Evict oldest if cap exceeded
        if len(_FEED_CACHE) > _CACHE_MAX:
            oldest = min(_FEED_CACHE.items(), key=lambda kv: kv[1][0])
            _FEED_CACHE.pop(oldest[0], None)
        return xml

    # Fetch failed → return stale if we have any
    if url in _FEED_CACHE:
        return _FEED_CACHE[url][1]
    return b""


def _fetch_parallel(urls: List[str], max_workers: int = _MAX_WORKERS) -> dict:
    """F1: Fetch multiple URLs in parallel. Returns {url: xml_bytes}."""
    out: dict = {}
    if not urls:
        return out

    # Cap workers to url count
    workers = max(1, min(max_workers, len(urls)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_fetch_cached, u): u for u in urls}
        for fut in as_completed(futures):
            u = futures[fut]
            try:
                out[u] = fut.result()
            except Exception as e:
                log.warning("parallel feed fetch failed for %s: %s: %s",
                            u, type(e).__name__, e)
                out[u] = b""
    return out


def _parse_rss(xml_bytes: bytes, max_items: int = 3) -> List[dict]:
    """Parse RSS/Atom. Returns list of {title, link, description, pub_date}."""
    if not xml_bytes:
        return []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        log.debug("unparseable feed XML (%d bytes): %s", len(xml_bytes), e)
        return []

    out: List[dict] = []

    # RSS 2.0: <rss><channel><item>...
    for item in root.iter("item"):
        t = item.findtext("title") or ""
        l = item.findtext("link") or ""
        d = item.findtext("description") or ""
        p = item.findtext("pubDate") or ""
        out.append({"title": t.strip(), "link": l.strip(),
                    "description": d.strip(), "pub_date": p.strip()})
        if len(out) >= max_items:
            return out

    # Atom: <feed><entry>
    if not out:
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for entry in root.iterfind(".//a:entry", ns):
            t = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
            links = entry.findall("a:link", ns)
            l = ""
            for lk in links:
                href = lk.get("href")
                if href:
                    l = href.strip()
                    break
            d = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
            if not d:
                d = (entry.findtext("a:content", default="", namespaces=ns) or "").strip()
            p = (entry.findtext("a:updated", default="", namespaces=ns) or "").strip()
            out.append({"title": t, "link": l, "description": d, "pub_date": p})
            if len(out) >= max_items:
                return out

    return out


def _matches(item: dict, query: str) -> bool:
    """Simple keyword match: any word from query in title/description."""
    if not query:
        return True
    q_lower = query.lower()
    words = [w for w in re.split(r"\W+", q_lower) if len(w) >= 3]
    if not words:
        return True
    haystack = (item.get("title", "") + " " + item.get("description", "")).lower()
    return any(w in haystack for w in words)


class RSSBackend(BaseBackend):
    name = "RSS News"
    category = "news"
    lang = "mixed"

    def __init__(self, category_key: str = "news"):
        self.category_key = category_key
        self._sources = _load_sources()

    def _get_feeds(self) -> List[dict]:
        cats = self._sources.get("categories", {})
        cat = cats.get(self.category_key, {})
        return cat.get("feeds", []) or []

    def search(self, query: str, max_results: int = 3) -> List[Result]:
        q = (query or "").strip()
        feeds = self._get_feeds()
        if not feeds:
            return []

        # Cap feeds to avoid heavy fetches (max 4)
        feeds = feeds[:4]

        # Build (url, name, lang) list, filter empty urls
        feed_info = []
        for feed in feeds:
            url = feed.get("url", "")
            if url:
                feed_info.append({
                    "url": url,
                    "name": feed.get("name", "RSS"),
                    "lang": feed.get("lang", "en"),
                })

        if not feed_info:
            return []

        # F1: Parallel fetch all feeds
        urls = [fi["url"] for fi in feed_info]
        fetched = _fetch_parallel(urls)

        # Process in stable order
        all_items: List[Result] = []
        for fi in feed_info:
            if len(all_items) >= max_results:
                break
            xml = fetched.get(fi["url"], b"")
            if not xml:
                continue
            items = _parse_rss(xml, max_items=_MAX_PER_FEED)
            for it in items:
                if not _matches(it, q):
                    continue
                all_items.append(Result(
                    title=it["title"] or "(no title)",
                    url=it["link"],
                    snippet=_strip_html(it["description"])[:400],
                    source=fi["name"],
                    category="news",
                    lang=fi["lang"],
                    extra={"pub_date": it.get("pub_date", "")},
                ))
                if len(all_items) >= max_results:
                    break

        return all_items[:max_results]


__all__ = ["RSSBackend"]
