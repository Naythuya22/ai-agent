"""Web search + fetch tools for agent mode.

Search is a CHAIN of independent no-key sources (Release 28):
DuckDuckGo Lite → Bing → Brave → Wikipedia. The first source that
answers wins; a source that bot-blocks, errors, times out or
returns nothing parseable is skipped and remembered as down for a
short cooldown, so one dead source cannot tax every later search.
No API key is needed for any of them. Candidates probed and
rejected as bot-walled from this network (2026-10-23): DDG HTML,
Mojeek, Ecosia, Yandex, Startpage, Marginalia, public SearXNG
instances (antibot walls / JSON disabled / rate-limited).

Requires internet. Results are wrapped in <untrusted> tags to
signal to the LLM that the content is not user instructions.
"""
from __future__ import annotations

import base64
import ipaddress
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

from core.log import get_logger
from core.net import offline_notice
from core.retry import RetryError, call_with_retry, should_retry

log = get_logger("tools.web_search")

_UA = "Mozilla/5.0 (Linux; Android 11; Termux; ai-agent/1.0)"
_TIMEOUT = 10
_MAX_FETCH_BYTES = 2 * 1024 * 1024  # 2 MB
_DDG_ENDPOINT = "https://lite.duckduckgo.com/lite/"


# ── HTTP (stdlib only — no third-party dependency) ──
class NetworkError(Exception):
    """The request could not be completed at all."""


class _Headers(dict):
    """dict whose lookup ignores key case (mirrors requests' CaseInsensitiveDict)."""

    def __init__(self, pairs):
        pairs = list(pairs)
        super().__init__(pairs)
        self._lower = {k.lower(): v for k, v in pairs}

    def get(self, key, default=None):
        return self._lower.get(key.lower(), default)

    def __getitem__(self, key):
        return self._lower[key.lower()]


class _Response:
    """Minimal stand-in for a requests.Response, backed by urllib."""

    def __init__(self, status, header_pairs, body, encoding=None):
        self.status_code = status
        self.headers = _Headers(header_pairs)
        self._body = body
        self.encoding = encoding

    @property
    def text(self) -> str:
        return self._body.decode(self.encoding or "utf-8", errors="replace")

    def iter_content(self, chunk_size=65536, decode_unicode=False):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def close(self):
        pass


class _RetryableHTTP(Exception):
    """A response status worth another attempt. Carries the response back."""

    def __init__(self, resp: "_Response"):
        super().__init__(f"HTTP {resp.status_code}")
        self.resp = resp


# Statuses that mean "try again", not "this URL is bad".
_RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})


def _http_get(url, params=None, headers=None, timeout=_TIMEOUT,
              max_bytes=_MAX_FETCH_BYTES):
    """GET url via urllib. HTTP error statuses come back as a _Response;
    transport failures raise NetworkError. Reads at most max_bytes.

    Transient failures (DNS blip, reset, 429/5xx) are retried with backoff
    before giving up. A terminal transport failure raises NetworkError and a
    terminal error status is returned as a _Response, so callers keep the
    handling they already had.
    """
    if params:
        sep = "&" if "?" in url else "?"
        url = url + sep + urllib.parse.urlencode(params)

    def _attempt() -> "_Response":
        req = urllib.request.Request(url, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = getattr(resp, "status", 200)
                if status >= 400:
                    out = _Response(status, resp.headers.items(), b"")
                else:
                    body = resp.read(max_bytes + 1)
                    enc = None
                    ctype = resp.headers.get("Content-Type") or ""
                    if "charset=" in ctype:
                        enc = ctype.split("charset=", 1)[1].split(";")[0].strip() or None
                    out = _Response(status, resp.headers.items(), body, enc)
        except urllib.error.HTTPError as e:
            pairs = e.headers.items() if e.headers else []
            out = _Response(e.code, pairs, b"")
        except Exception as e:
            raise NetworkError(f"{type(e).__name__}: {e}") from e
        if out.status_code in _RETRY_STATUS:
            raise _RetryableHTTP(out)
        return out

    def _retryable(exc: BaseException) -> bool:
        if isinstance(exc, _RetryableHTTP):
            return True
        if isinstance(exc, NetworkError):
            return should_retry(exc.__cause__ or exc)
        return False

    try:
        return call_with_retry(
            _attempt, attempts=3, base_delay=0.4, budget=12.0,
            what=f"GET {url}", retry_if=_retryable,
        )
    except RetryError as e:
        cause = e.__cause__
        if isinstance(cause, _RetryableHTTP):
            log.warning("%s still failing after retries: %s", url, cause)
            return cause.resp
        raise NetworkError(str(e)) from e


# ── Search chain plumbing (Release 28) ──
# Failover IS the retry for chain sources: one short attempt each,
# then the next source. A source that fails is remembered as down
# for _SOURCE_COOLDOWN seconds so a dead source cannot tax every
# later search in the session (module-level state, like the DDG
# backend's rate guard in core/search/backends/ddg.py).
_SOURCE_TIMEOUT = 6
_SOURCE_COOLDOWN = 120.0
_BING_ENDPOINT = "https://www.bing.com/search"
_BRAVE_ENDPOINT = "https://search.brave.com/search"
_down_until: dict = {}


def _source_cooling(name: str) -> bool:
    return time.monotonic() < _down_until.get(name, 0.0)


def _mark_source_down(name: str) -> None:
    _down_until[name] = time.monotonic() + _SOURCE_COOLDOWN


def _mark_source_up(name: str) -> None:
    _down_until.pop(name, None)


def _fetch_page(url, params=None, timeout=_SOURCE_TIMEOUT,
                max_bytes=512 * 1024):
    """Single-attempt GET for chain sources. Any HTTP status comes
    back as a _Response; transport failures raise NetworkError.
    Deliberately NOT the retrying _http_get: the chain would rather
    move to the next independent source than wait out backoff."""
    if params:
        sep = "&" if "?" in url else "?"
        url = url + sep + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "User-Agent": _UA,
        "Accept": "text/html",
        "Accept-Language": "en-US,en;q=0.9",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", 200)
            body = resp.read(max_bytes + 1) if status < 400 else b""
            pairs = resp.headers.items() if resp.headers else []
            enc = None
            ctype = resp.headers.get("Content-Type") or ""
            if "charset=" in ctype:
                enc = ctype.split("charset=", 1)[1].split(";")[0].strip() or None
            return _Response(status, pairs, body, enc)
    except urllib.error.HTTPError as e:
        pairs = e.headers.items() if e.headers else []
        return _Response(e.code, pairs, b"")
    except Exception as e:
        raise NetworkError(f"{type(e).__name__}: {e}") from e


# ── URL safety ──
def _url_is_safe(url: str) -> tuple:
    """Return (ok, reason). Blocks non-http(s) + private/loopback IPs."""
    try:
        p = urllib.parse.urlparse(url)
    except Exception:
        return False, "invalid URL"
    if p.scheme not in ("http", "https"):
        return False, f"scheme not allowed: {p.scheme}"
    host = (p.hostname or "").lower()
    if not host:
        return False, "missing hostname"
    if host in ("localhost", "localhost.localdomain"):
        return False, "localhost blocked"
    # IP literal?
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return False, f"private/reserved IP blocked: {host}"
    except ValueError:
        pass  # not an IP literal — domain name, OK
    return True, ""


# ── HTML → text (strip scripts/styles/tags) ──
class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "noscript", "iframe", "svg", "head"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self._SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag.lower() in self._SKIP and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
        raw = re.sub(r"\n\s*\n+", "\n\n", raw)
        return raw.strip()


# ── DDG Lite result parser ──
class _DDGParser(HTMLParser):
    """Parse DDG lite results: <a class=result-link href=...>title</a> + snippet cells."""
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
            # DDG lite may wrap: /l/?uddg=ENCODED
            if href.startswith("/l/?") or "uddg=" in href:
                try:
                    q = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                    href = q.get("uddg", [href])[0]
                except Exception as e:
                    log.debug("could not unwrap a DDG redirect: %s: %s",
                              type(e).__name__, e)
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


# ── Public tools ──
def _ddg_hits(q: str, n: int):
    """DDG Lite hits, or None when the service itself failed.

    Transport errors, non-200 answers and unparseable pages are all
    "the service failed" — the detail (including the raw HTTP status)
    is logged here and never returned: the old error string carried
    the status into the model's context and from there into the
    user's reply ("DDG ကနေ HTTP 202 ဆိုတဲ့ error ပြန်လာတယ်" —
    tester evaluation round 3). An empty list is a successful,
    genuinely empty answer, which is a different thing.
    """
    try:
        r = _http_get(
            _DDG_ENDPOINT,
            params={"q": q},
            headers={"User-Agent": _UA, "Accept-Language": "en-US,en;q=0.9"},
            timeout=_TIMEOUT,
        )
    except NetworkError as e:
        log.warning("DDG search failed for %r: %s", q, e)
        return None
    if r.status_code != 200:
        log.warning("DDG search for %r answered HTTP %s", q, r.status_code)
        return None
    parser = _DDGParser()
    try:
        parser.feed(r.text)
    except Exception as e:
        log.warning("DDG result parse failed for %r: %s: %s",
                    q, type(e).__name__, e)
        return None
    return parser.results[:n]


def _fallback_hits(q: str, n: int) -> list:
    """The chain's last source (Release 28): Wikipedia, through the
    smart_search backends — in-repo, needs no API key, and a
    different service from every HTML engine ahead of it, so a
    bot-block wall does not take it down with them. Encyclopedic
    rather than general-web, which is why it stands last. Hits are
    reshaped to the dicts the formatter below already renders."""
    try:
        from core.search.backends import WikipediaBackend
        results = WikipediaBackend().search(q, max_results=n) or []
    except Exception as e:
        log.warning("fallback search failed for %r: %s: %s",
                    q, type(e).__name__, e)
        return []
    out = []
    for r in results:
        title = getattr(r, "title", "") or ""
        url = getattr(r, "url", "") or ""
        if title and url:
            out.append({"title": title, "url": url,
                        "snippet": getattr(r, "snippet", "") or ""})
    return out


# ── Bing ──
def _bing_unwrap(url: str) -> str:
    """Bing sometimes routes result links through /ck/a?u=a1<base64>;
    unwrap to the real destination when that shape shows up."""
    if "bing.com/ck/a" not in url:
        return url
    try:
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        u = (q.get("u") or [""])[0]
        if u.startswith("a1"):
            raw = base64.urlsafe_b64decode(u[2:] + "=" * (-len(u[2:]) % 4))
            out = raw.decode("utf-8", errors="strict")
            if out.startswith(("http://", "https://")):
                return out
    except Exception as e:
        log.debug("could not unwrap a Bing redirect: %s: %s",
                  type(e).__name__, e)
    return url


class _BingParser(HTMLParser):
    """Parse Bing results: <li class=b_algo> blocks. The title link
    is the anchor wrapping an <h2>; the snippet is the block's first
    substantial <p>. Layout variants without an <h2> fall back to
    the first external anchor with real text."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results = []
        self._li_depth = 0
        self._anchors = []
        self._anchor = None
        self._in_h2 = False
        self._p_buf = None
        self._snippet = ""

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = (a.get("class") or "").lower()
        if tag == "li":
            if self._li_depth == 0 and "b_algo" in cls:
                self._li_depth = 1
                self._anchors = []
                self._snippet = ""
            elif self._li_depth > 0:
                self._li_depth += 1
            return
        if self._li_depth == 0:
            return
        if tag == "a":
            self._anchor = {"href": a.get("href", ""), "h2": "",
                            "text": [], "h2buf": []}
            self._in_h2 = False
        elif tag == "h2" and self._anchor is not None:
            self._in_h2 = True
        elif tag == "p" and self._p_buf is None and not self._snippet:
            self._p_buf = []

    def handle_endtag(self, tag):
        if tag == "li" and self._li_depth > 0:
            self._li_depth -= 1
            if self._li_depth == 0:
                self._finish()
            return
        if self._li_depth == 0:
            return
        if tag == "a" and self._anchor is not None:
            self._anchor["h2"] = "".join(self._anchor["h2buf"]).strip()
            self._anchor["text"] = "".join(self._anchor["text"]).strip()
            self._anchors.append(self._anchor)
            self._anchor = None
            self._in_h2 = False
        elif tag == "h2":
            self._in_h2 = False
        elif tag == "p" and self._p_buf is not None:
            text = "".join(self._p_buf).strip()
            if len(text) > 20:
                self._snippet = text
            self._p_buf = None

    def handle_data(self, data):
        if self._anchor is not None:
            self._anchor["text"].append(data)
            if self._in_h2:
                self._anchor["h2buf"].append(data)
        if self._p_buf is not None:
            self._p_buf.append(data)

    def _finish(self):
        pick = None
        for a in self._anchors:
            if a["h2"] and a["href"].startswith(("http://", "https://")):
                pick = (a["h2"], a["href"])
                break
        if pick is None:
            for a in self._anchors:
                if (a["href"].startswith(("http://", "https://"))
                        and "bing.com" not in a["href"]
                        and len(a["text"]) >= 8):
                    pick = (a["text"], a["href"])
                    break
        if pick:
            self.results.append({
                "title": " ".join(pick[0].split()),
                "url": _bing_unwrap(pick[1]),
                "snippet": " ".join(self._snippet.split()),
            })


def _bing_hits(q: str, n: int):
    """Bing hits, or None when Bing failed — including a 200 page
    with zero parseable results, which for a general engine means a
    consent/block page, not an honest empty answer."""
    try:
        r = _fetch_page(_BING_ENDPOINT, params={"q": q})
    except NetworkError as e:
        log.warning("Bing search failed for %r: %s", q, e)
        return None
    if r.status_code != 200:
        log.warning("Bing search for %r answered HTTP %s", q, r.status_code)
        return None
    parser = _BingParser()
    try:
        parser.feed(r.text)
    except Exception as e:
        log.warning("Bing result parse failed for %r: %s: %s",
                    q, type(e).__name__, e)
        return None
    return parser.results[:n] or None


# ── Brave ──
class _BraveParser(HTMLParser):
    """Parse Brave results: <div class="snippet ..." data-type="web">
    blocks (server-rendered). The title is the block's div.title
    (its title attribute is the cleanest copy); the snippet is the
    div.content inside div.generic-snippet. Captures remember the
    div depth they opened at, so nested markup cannot leak text
    across fields."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results = []
        self._depth = 0
        self._cur = None
        self._generic_depth = 0
        self._cap = None        # "title" | "snippet"
        self._cap_depth = 0
        self._buf = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = (a.get("class") or "").lower()
        if tag == "div":
            if self._depth == 0:
                if a.get("data-type") == "web" and "snippet" in cls:
                    self._depth = 1
                    self._cur = {"title": "", "url": "", "snippet": ""}
                    self._generic_depth = 0
                return
            self._depth += 1
            if "generic-snippet" in cls:
                self._generic_depth = self._depth
            elif "title" in cls and self._cur is not None \
                    and not self._cur["title"] and self._cap is None:
                self._cur["title"] = (a.get("title") or "").strip()
                self._cap, self._cap_depth, self._buf = "title", self._depth, []
            elif "content" in cls and self._generic_depth \
                    and self._cur is not None and not self._cur["snippet"] \
                    and self._cap is None:
                self._cap, self._cap_depth, self._buf = "snippet", self._depth, []
        elif tag == "a" and self._depth > 0 and self._cur is not None \
                and not self._cur["url"]:
            href = a.get("href", "")
            if href.startswith(("http://", "https://")):
                self._cur["url"] = href

    def handle_endtag(self, tag):
        if tag != "div" or self._depth == 0:
            return
        if self._cap is not None and self._cap_depth == self._depth:
            text = " ".join("".join(self._buf).split())
            if self._cap == "title":
                self._cur["title"] = self._cur["title"] or text
            else:
                self._cur["snippet"] = text
            self._cap, self._buf = None, []
        if self._generic_depth == self._depth:
            self._generic_depth = 0
        self._depth -= 1
        if self._depth == 0 and self._cur is not None:
            if self._cur["title"] and self._cur["url"]:
                self.results.append(self._cur)
            self._cur = None

    def handle_data(self, data):
        if self._cap is not None:
            self._buf.append(data)


def _brave_hits(q: str, n: int):
    """Brave hits, or None on failure / zero parseable results."""
    try:
        r = _fetch_page(_BRAVE_ENDPOINT, params={"q": q})
    except NetworkError as e:
        log.warning("Brave search failed for %r: %s", q, e)
        return None
    if r.status_code != 200:
        log.warning("Brave search for %r answered HTTP %s", q, r.status_code)
        return None
    parser = _BraveParser()
    try:
        parser.feed(r.text)
    except Exception as e:
        log.warning("Brave result parse failed for %r: %s: %s",
                    q, type(e).__name__, e)
        return None
    return parser.results[:n] or None


# ── The chain ──
def _chain_sources():
    """Ordered (name, fn) pairs. Each fn returns a hits list on
    success (possibly empty — a genuinely empty answer) or None on
    failure. DuckDuckGo keeps its legacy contract ([] = answered,
    nothing found); the HTML engines return None on zero parseable
    results because for them that shape means a block page."""
    return [
        ("DuckDuckGo", _ddg_hits),
        ("Bing", _bing_hits),
        ("Brave", _brave_hits),
        ("Wikipedia", _fallback_hits),
    ]


def _run_search_chain(q: str, n: int):
    """Walk the chain. Returns (hits, via, any_answered, primary_failed):

    - hits: the first non-empty result set, or [].
    - via: the answering source's name when it is not the primary.
    - any_answered: at least one source answered successfully.
    - primary_failed: the primary source was tried and failed —
      the verdict rule web_search has always used (a failed primary
      plus no hits anywhere reads as "unavailable", not "no results").
    A source in cooldown is skipped without being counted as tried.
    """
    any_answered = False
    primary_failed = False
    for i, (name, fn) in enumerate(_chain_sources()):
        if _source_cooling(name):
            continue
        try:
            res = fn(q, n)
        except Exception as e:
            log.warning("search source %s raised: %s: %s",
                        name, type(e).__name__, e)
            res = None
        if res:
            _mark_source_up(name)
            return res, (None if i == 0 else name), True, primary_failed
        if res is None:
            _mark_source_down(name)
            if i == 0:
                primary_failed = True
        else:
            any_answered = True
    return [], None, any_answered, primary_failed


def chain_results(query: str, max_results: int = 5) -> list:
    """Chain hits reshaped as core.search Result objects — the
    smart_search pipeline's last-resort general source. Never raises."""
    from core.search.base import Result
    hits, via, _any, _pf = _run_search_chain((query or "").strip(),
                                             max(1, int(max_results)))
    out = []
    for h in hits:
        out.append(Result(title=h["title"], url=h["url"],
                          snippet=(h.get("snippet") or "")[:400],
                          source=via or "DuckDuckGo",
                          category="general", lang="en"))
    return out


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web (DDG Lite → Bing → Brave → Wikipedia chain).
    Returns top results as plain text.

    Args:
        query: Search query (any language — Burmese, English, etc).
        max_results: Number of results to return (1-10, default 5).
    """
    q = (query or "").strip()
    if not q:
        return "Error: query is required"

    # No network: fail fast rather than sitting on the socket timeout.
    notice = offline_notice("web_search", q)
    if notice:
        return notice

    try:
        n = max(1, min(10, int(max_results)))
    except Exception:
        n = 5

    # Release 28: search used to be a single point of failure — when
    # DDG bot-blocked (HTTP 202) or timed out, the turn simply had no
    # answer. The chain (DDG Lite → Bing → Brave → Wikipedia) now
    # stands behind it; only when no source produces anything does
    # the tool report failure, and then in words, never a raw HTTP
    # status. Verdict rule, unchanged from the two-source era: the
    # primary failing (or nothing answering at all) reads as
    # "unavailable"; a source answering empty-handed reads as
    # "no results".
    hits, via, any_answered, primary_failed = _run_search_chain(q, n)
    if not hits:
        if primary_failed or not any_answered:
            return ("Error: search is unavailable right now — the search "
                    "service did not respond. Please try again shortly.")
        return f"No results found for: {q}"

    lines = [f"Web search results for: {q}"]
    if via:
        lines.append(f"(Results via {via}.)")
    lines += ["<untrusted>", ""]
    for i, h in enumerate(hits, 1):
        lines.append(f"{i}. {h['title']}")
        lines.append(f"   URL: {h['url']}")
        sn = h.get("snippet") or ""
        if sn:
            lines.append(f"   {sn[:300]}")
        lines.append("")
    lines.append("</untrusted>")
    return "\n".join(lines)


def web_fetch(url: str, max_chars: int = 5000) -> str:
    """Fetch a URL and return its readable text content (truncated).

    Args:
        url: Full URL (http/https). Blocked: private/loopback IPs.
        max_chars: Max characters returned (500-20000, default 5000).
    """
    u = (url or "").strip()
    if not u:
        return "Error: url is required"
    ok, reason = _url_is_safe(u)
    if not ok:
        return f"Error: URL blocked — {reason}"

    # No network: fail fast rather than sitting on the socket timeout.
    notice = offline_notice("web_fetch", u)
    if notice:
        return notice

    try:
        limit = max(500, min(20000, int(max_chars)))
    except Exception:
        limit = 5000

    try:
        r = _http_get(
            u,
            headers={"User-Agent": _UA},
            timeout=_TIMEOUT,
            max_bytes=_MAX_FETCH_BYTES,
        )
    except NetworkError as e:
        return f"Error: fetch failed: {e}"
    if r.status_code != 200:
        # Same rule as web_search (Release 28): the status is logged,
        # never handed to the model — raw codes leak into user replies.
        log.warning("fetch of %s answered HTTP %s", u, r.status_code)
        return ("Error: the page could not be fetched — the site did "
                "not answer successfully.")

    # Cap response size
    chunks = []
    total = 0
    try:
        for chunk in r.iter_content(chunk_size=65536, decode_unicode=False):
            if not chunk:
                continue
            total += len(chunk)
            if total > _MAX_FETCH_BYTES:
                break
            chunks.append(chunk)
    except Exception as e:
        return f"Error: read failed: {type(e).__name__}: {e}"
    finally:
        r.close()

    body = b"".join(chunks)
    ctype = (r.headers.get("Content-Type") or "").lower()
    if "html" not in ctype and "text" not in ctype and "xml" not in ctype:
        return f"Error: unsupported content type: {ctype[:60]}"

    try:
        html_text = body.decode(r.encoding or "utf-8", errors="replace")
    except Exception:
        html_text = body.decode("utf-8", errors="replace")

    ex = _TextExtractor()
    try:
        ex.feed(html_text)
    except Exception as e:
        log.debug("HTML extraction failed; partial text used: %s: %s",
                  type(e).__name__, e)
    text = ex.text()
    if not text:
        return "Error: no readable text extracted"
    if len(text) > limit:
        text = text[:limit] + f"\n\n...(truncated at {limit} chars)"

    return f"Fetched: {u}\n<untrusted>\n{text}\n</untrusted>"


ALL_TOOLS = [web_search, web_fetch]
__all__ = ["web_search", "web_fetch", "ALL_TOOLS"]
