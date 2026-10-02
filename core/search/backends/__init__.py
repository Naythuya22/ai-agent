"""Search backends — each implements BaseBackend."""
from core.search.backends.wikipedia import WikipediaBackend
from core.search.backends.stackexchange import StackExchangeBackend
from core.search.backends.openlibrary import OpenLibraryBackend
from core.search.backends.rss import RSSBackend
from core.search.backends.ddg import DDGBackend

__all__ = [
    "WikipediaBackend", "StackExchangeBackend",
    "OpenLibraryBackend", "RSSBackend", "DDGBackend",
]
