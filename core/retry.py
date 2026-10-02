"""Retry with exponential backoff for transient network failures.

The search backends and the fetch tool talk to the open internet from a phone,
where DNS blips, connection resets and rate limits are routine. Previously a
single failure meant an empty result set for the whole turn, and DDG's own
rate limit (documented in ``core/search/backends/ddg.py``) was noted in a
comment but never acted on.

``call_with_retry`` wraps a callable so that:
  * transient transport errors are retried with jittered exponential backoff,
  * HTTP 429 honours ``Retry-After`` when the server sends one,
  * permanent errors (4xx other than 429/408) are raised immediately,
  * every attempt is logged, so a dead network is diagnosable afterwards.

Retries are capped by ``attempts`` *and* by a wall-clock budget, so a slow
backend cannot stall a streaming response indefinitely.
"""
from __future__ import annotations

import random
import time
import urllib.error
from typing import Callable, Iterable, TypeVar

from core.log import get_logger

T = TypeVar("T")

log = get_logger("retry")

# Retry these transport-level conditions.
_RETRYABLE_EXC = (
    ConnectionError,   # includes ConnectionResetError / BrokenPipeError
    TimeoutError,      # socket.timeout is an OSError subclass of this
    OSError,           # DNS and connect failures surface as OSError
)
# ...but never these: they are bugs or fatal conditions, not flakiness.
_NEVER_RETRY = (
    KeyboardInterrupt,
    SystemExit,
    MemoryError,
    SyntaxError,
    TypeError,
    ValueError,
    KeyError,
    AttributeError,
)

# Statuses worth another attempt. 408 = request timeout, 429 = rate limited.
_RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class RetryError(Exception):
    """All attempts were exhausted. __cause__ holds the last real error."""

    def __init__(self, message: str, attempts: int, elapsed: float):
        super().__init__(message)
        self.attempts = attempts
        self.elapsed = elapsed


def _retry_after(exc: BaseException) -> float | None:
    """Seconds the server asked us to wait, or None. Used as a delay floor."""
    if not isinstance(exc, urllib.error.HTTPError):
        return None
    try:
        raw = exc.headers.get("Retry-After") if exc.headers else None
    except Exception:
        return None
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None  # HTTP-date form — not worth parsing here


def should_retry(exc: BaseException) -> bool:
    """True when ``exc`` is a transient failure worth another attempt."""
    if isinstance(exc, _NEVER_RETRY):
        return False
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in _RETRY_STATUS
    # urllib wraps DNS/connect failures in URLError (an OSError subclass).
    return isinstance(exc, _RETRYABLE_EXC)


def call_with_retry(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 0.4,
    max_delay: float = 6.0,
    budget: float = 15.0,
    what: str = "request",
    sleep: Callable[[float], None] = time.sleep,
    retry_if: Callable[[BaseException], bool] | None = None,
) -> T:
    """Call ``fn`` until it succeeds, retrying transient failures.

    ``budget`` is a hard wall-clock ceiling across all sleeps; when exceeded
    the last error is raised rather than waiting out the remaining backoff.
    ``sleep`` is injectable so tests never actually wait.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    predicate = retry_if or should_retry
    started = time.monotonic()
    last: BaseException | None = None

    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except BaseException as exc:  # noqa: BLE001 — filtered below
            last = exc
            if not predicate(exc):
                log.debug("%s: permanent failure (%s: %s)", what,
                          type(exc).__name__, exc)
                raise
            if attempt >= attempts:
                break
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            server_wait = _retry_after(exc)
            if server_wait is not None:
                delay = max(delay, min(server_wait, max_delay))
            delay += random.uniform(0, delay * 0.25)  # jitter — avoid sync storms
            elapsed = time.monotonic() - started
            if elapsed + delay > budget:
                break
            log.warning("%s: attempt %d/%d failed (%s: %s); retrying in %.1fs",
                        what, attempt, attempts, type(exc).__name__, exc, delay)
            sleep(delay)

    elapsed = time.monotonic() - started
    log.warning("%s: giving up after %d attempt(s) in %.1fs (%s: %s)",
                what, attempts, elapsed, type(last).__name__, last)
    raise RetryError(
        f"{what} failed after {attempts} attempt(s): {type(last).__name__}: {last}",
        attempts=attempts,
        elapsed=elapsed,
    ) from last


def get_json(url: str, *, headers: dict | None = None, timeout: float = 10.0,
             attempts: int = 3, max_bytes: int = 200000,
             what: str = "json fetch") -> bytes:
    """GET ``url`` and return the body, retrying transient failures.

    Shared by the search backends so rate limits and flaky links behave the
    same way everywhere. Returns raw bytes; callers decode. At most
    ``max_bytes`` are read — the backends never needed more and an unbounded
    read is how a bad feed exhausts memory on a phone.
    """
    import urllib.request

    def _once() -> bytes:
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", 200)
            if status >= 400:
                raise urllib.error.HTTPError(
                    url, status, "client error", resp.headers, None)
            return resp.read(max_bytes)

    return call_with_retry(_once, attempts=attempts, what=what)


__all__ = [
    "call_with_retry", "get_json", "should_retry", "RetryError",
]
