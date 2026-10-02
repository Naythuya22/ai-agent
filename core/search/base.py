"""Base types and abstract backend for the search subsystem."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Result:
    """A single search result — backend-agnostic."""
    title: str
    url: str = ""
    snippet: str = ""
    source: str = ""           # e.g. "Wikipedia", "StackOverflow"
    category: str = ""         # e.g. "knowledge", "coding"
    lang: str = "en"           # "en" | "my" | ...
    score: float = 0.0         # optional relevance score
    extra: dict = field(default_factory=dict)

    def __str__(self) -> str:
        parts = [f"Title: {self.title}"]
        if self.source:
            parts.append(f"Source: {self.source}")
        if self.url:
            parts.append(f"URL: {self.url}")
        if self.snippet:
            parts.append(f"Content: {self.snippet}")
        return "\n".join(parts)


class BaseBackend:
    """Abstract search backend. Subclasses implement search()."""
    name: str = "base"
    category: str = "general"
    lang: str = "en"

    def search(self, query: str, max_results: int = 3) -> List[Result]:
        """Fetch up to max_results results. Must be overridden."""
        raise NotImplementedError

    def available(self) -> bool:
        """Return False if backend is temporarily disabled."""
        return True
