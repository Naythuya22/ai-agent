"""Smart search tool — code pipeline + optional LLM synthesis.

Pipeline:
  L1 classify  → category
  L2 translate → English
  L3 route     → backends
  L4 fetch     → results (all backends)
  L5 clean     → dedupe + rank
  L6 pack      → <context>
  L7 LLM       → short Burmese/English answer (if llm provided)
"""
from __future__ import annotations

from typing import Optional

from core.log import get_logger
from core.log import get_logger
from core.net import offline_notice
from core.search import (
    classify, translate, route, clean, pack, sources_line,
    Result, is_burmese,
)

log = get_logger("tools.smart_search")

log = get_logger("tools.smart_search")


# ── Synthesis prompt (SHORT — 4B friendly) ──
_SYNTH_PROMPT = (
    "Answer the user's question using ONLY the context below.\n"
    "Rules:\n"
    "1. Reply in the user's language (Burmese if user is Burmese).\n"
    "2. If context has the answer → summarize in 3 to 6 sentences.\n"
    "3. If context is irrelevant → say 'ရှာမတွေ့ပါ' (Burmese) or 'No relevant results.'\n"
    "4. Do NOT invent facts. Do NOT add URLs unless asked.\n\n"
    "Context:\n{context}\n\n"
    "User question: {query}\n\n"
    "Answer:"
)


def _collect_results(backends, raw_query: str, translated_query: str,
                     per_backend: int, max_total: int) -> list:
    """Run backends in order, using per-backend query preference."""
    out: list = []
    for be in backends:
        try:
            if not be.available():
                continue
            # Backend decides: raw (Wikipedia) vs translated (others)
            q_for_backend = raw_query if getattr(be, 'prefer_raw', False) else translated_query
            hits = be.search(q_for_backend, max_results=per_backend) or []
            out.extend(hits)
            if len(out) >= max_total:
                break
        except Exception:
            continue
    return out


def smart_search(
    query: str,
    category: str = "auto",
    max_results: int = 5,
    llm=None,
) -> str:
    """Search the web smartly and answer the user's question.

    Chooses the best source automatically:
    - news → RSS feeds (BBC Burmese, RFA, VOA, DVB, Irrawaddy, ...)
    - coding → Stack Overflow
    - knowledge → Wikipedia (EN + Burmese)
    - book → OpenLibrary
    - general → DuckDuckGo

    Args:
        query: The user's question (any language — Burmese, English, etc).
        category: Leave as 'auto' to detect automatically. Can force one of:
                  'news', 'coding', 'knowledge', 'book', 'general'.
        max_results: Maximum number of sources to consider (1-10, default 5).
        llm: Internal — the LLM instance for synthesis (agent passes it).

    Returns:
        A short answer in the user's language, with a source list at the end.
        If no LLM is given, returns the raw <context> block instead.
    """
    q = (query or "").strip()
    if not q:
        return "Error: query is required"

    # No network: fail fast instead of waiting out every backend timeout
    # (news = 3 feeds x 8s) and then reporting a misleading "not found".
    notice = offline_notice("smart_search", q)
    if notice:
        return notice

    try:
        n = max(1, min(10, int(max_results)))
    except Exception:
        n = 5

    # L1 — classify (if auto)
    cat = (category or "auto").strip().lower()
    if cat == "auto" or not cat:
        c = classify(q)
        cat = c.category
        conf = c.confidence
    else:
        conf = 1.0

    # L2 — translate to English (skip if category uses Burmese sources)
    _BURMESE_FIRST = {"knowledge", "news", "burmese_news"}
    if cat in _BURMESE_FIRST and is_burmese(q):
        search_query = q   # keep raw Burmese — Wikipedia MM / RSS MM accept it
    else:
        search_query = translate(q, llm=llm)

    # L3 — route
    backends = route(cat)

    # L4 — fetch (Wikipedia uses raw, others use English)
    raw = _collect_results(backends, q, search_query, per_backend=n, max_total=n * 2)

    # If nothing found → try general fallback
    if not raw and cat != "general":
        try:
            from core.search.backends import DDGBackend
            raw = DDGBackend().search(search_query, max_results=n) or []
            cat = "general"
        except Exception as e:
            log.debug("general DDG fallback failed: %s: %s", type(e).__name__, e)

    # L5 — clean
    cleaned = clean(raw, max_keep=n)

    # L6 — pack
    context = pack(cleaned, q)
    src = sources_line(cleaned)

    # No results at all
    if not cleaned:
        msg = (
            f"ရှာမတွေ့ပါ။ Category: {cat} | Query: {search_query!r}"
            if any("\u1000" <= ch <= "\u109F" for ch in q)
            else f"No results found. Category: {cat} | Query: {search_query!r}"
        )
        return msg

    # No LLM → return raw context
    if llm is None:
        return context

    # L7 — synthesize (short prompt)
    prompt = _SYNTH_PROMPT.format(context=context, query=q)
    try:
        answer = llm.ask(prompt)
        answer = (answer or "").strip()
    except Exception as e:
        return f"{context}\n\n(LLM synthesis failed: {type(e).__name__})"

    if not answer:
        answer = "(no answer)"

    # Append source list at the end
    if src:
        answer = f"{answer}\n\n📚 Sources: {src}"
    return answer


__all__ = ["smart_search"]
