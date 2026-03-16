"""LRU translation cache to avoid redundant translations.

Uses an ordered dict as an LRU cache with configurable max size
and TTL (time-to-live) for entries.
"""

import time
from collections import OrderedDict

from voxfusion.config.models import TranslationCacheConfig
from voxfusion.logging import get_logger

log = get_logger(__name__)


class TranslationCache:
    """In-memory LRU cache for translated text.

    Keys are ``(text, source_lang, target_lang)`` tuples.
    """

    def __init__(self, config: TranslationCacheConfig | None = None) -> None:
        cfg = config or TranslationCacheConfig()
        self._max_size = cfg.max_size
        self._ttl = cfg.ttl
        self._enabled = cfg.enabled
        self._cache: OrderedDict[tuple[str, str, str], tuple[str, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, text: str, source_lang: str, target_lang: str) -> str | None:
        """Look up a cached translation. Returns ``None`` on miss."""
        if not self._enabled:
            return None

        key = (text, source_lang, target_lang)
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None

        translated, ts = entry
        if self._ttl > 0 and (time.monotonic() - ts) > self._ttl:
            del self._cache[key]
            self._misses += 1
            return None

        self._cache.move_to_end(key)
        self._hits += 1
        return translated

    def put(self, text: str, source_lang: str, target_lang: str, translated: str) -> None:
        """Store a translation in the cache."""
        if not self._enabled:
            return

        key = (text, source_lang, target_lang)
        self._cache[key] = (translated, time.monotonic())
        self._cache.move_to_end(key)

        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        """Remove all entries."""
        self._cache.clear()

    @property
    def size(self) -> int:
        """Number of cached entries."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0–1.0)."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
