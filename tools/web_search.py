"""Web search + fetch tools for agent mode.

Uses DuckDuckGo Lite (no API key) + stdlib HTML parser.
Requires internet. Results are wrapped in <untrusted> tags to
signal to the LLM that the content is not user instructions.
"""
from __future__ import annotations

import ipaddress
import re
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
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web via DuckDuckGo. Returns top results as plain text.

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

    try:
        r = _http_get(
            _DDG_ENDPOINT,
            params={"q": q},
            headers={"User-Agent": _UA, "Accept-Language": "en-US,en;q=0.9"},
            timeout=_TIMEOUT,
        )
    except NetworkError as e:
        return f"Error: network failed: {e}"
    if r.status_code != 200:
        return f"Error: DDG returned HTTP {r.status_code}"
    parser = _DDGParser()
    try:
        parser.feed(r.text)
    except Exception as e:
        return f"Error: parse failed: {type(e).__name__}: {e}"

    hits = parser.results[:n]
    if not hits:
        return f"No results found for: {q}"

    lines = [f"Web search results for: {q}", "<untrusted>", ""]
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
        return f"Error: HTTP {r.status_code}"

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
