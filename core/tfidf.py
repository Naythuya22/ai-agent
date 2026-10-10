"""BM25-lite ranking for memory search. Pure Python — no deps.

Usage:
    idx = BM25Lite()
    idx.build([("id1", "text 1"), ("id2", "text 2")])
    results = idx.search("query", k=10)  # -> [(id, score), ...]
"""
from __future__ import annotations

import math

from core.tokens import tokenize


class BM25Lite:
    """Compute BM25 scores over a small item set (memory: ~25-10000 items)."""

    K1 = 1.5

    def __init__(self) -> None:
        self._ids: list = []
        self._tokens: list = []  # list[list[str]] — parallel to _ids
        self._df: dict = {}  # token -> doc count
        self._N: int = 0
        self._avgdl: float = 0.0

    def build(self, items) -> None:
        """items: list[(id, text)]. Replaces any existing index."""
        self._ids = []
        self._tokens = []
        self._df = {}
        lengths = []
        for id_, text in items:
            toks = tokenize(text)
            self._ids.append(id_)
            self._tokens.append(toks)
            lengths.append(len(toks))
            for t in set(toks):
                self._df[t] = self._df.get(t, 0) + 1
        self._N = len(self._ids)
        self._avgdl = (sum(lengths) / self._N) if self._N else 0.0

    def _idf(self, token: str) -> float:
        df = self._df.get(token, 0)
        if df == 0:
            return 0.0
        return math.log((self._N - df + 0.5) / (df + 0.5) + 1)

    def search(self, query: str, k: int = 10) -> list:
        """Return [(id, score), ...] sorted by score desc, score > 0 only."""
        if not self._ids or not query:
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = [0.0] * self._N
        for qt in q_tokens:
            idf = self._idf(qt)
            if idf == 0.0:
                continue
            for i, doc_toks in enumerate(self._tokens):
                tf = doc_toks.count(qt)
                if tf == 0:
                    continue
                denom = tf + self.K1  # simple length-normalized TF
                scores[i] += idf * (tf / denom)
        ranked = [(self._ids[i], s) for i, s in enumerate(scores) if s > 0]
        ranked.sort(key=lambda x: -x[1])
        return ranked[:k]

    def update(self, id_: str, text: str) -> None:
        """Replace or append one item. O(N) but fine for small N."""
        # remove if exists
        if id_ in self._ids:
            idx = self._ids.index(id_)
            old_toks = self._tokens[idx]
            for t in set(old_toks):
                self._df[t] = max(0, self._df.get(t, 0) - 1)
                if self._df[t] == 0:
                    del self._df[t]
            del self._ids[idx]
            del self._tokens[idx]
        # re-add
        toks = tokenize(text)
        self._ids.append(id_)
        self._tokens.append(toks)
        for t in set(toks):
            self._df[t] = self._df.get(t, 0) + 1
        self._N = len(self._ids)
        total = sum(len(t) for t in self._tokens)
        self._avgdl = (total / self._N) if self._N else 0.0
