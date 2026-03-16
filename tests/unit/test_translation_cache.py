"""Unit tests for voxfusion.translation.cache."""

from voxfusion.config.models import TranslationCacheConfig
from voxfusion.translation.cache import TranslationCache


class TestTranslationCache:
    def test_put_and_get(self) -> None:
        cache = TranslationCache()
        cache.put("hello", "en", "fr", "bonjour")
        assert cache.get("hello", "en", "fr") == "bonjour"

    def test_miss_returns_none(self) -> None:
        cache = TranslationCache()
        assert cache.get("unknown", "en", "fr") is None

    def test_size(self) -> None:
        cache = TranslationCache()
        assert cache.size == 0
        cache.put("a", "en", "fr", "x")
        cache.put("b", "en", "fr", "y")
        assert cache.size == 2

    def test_max_size_eviction(self) -> None:
        config = TranslationCacheConfig(max_size=2)
        cache = TranslationCache(config)
        cache.put("a", "en", "fr", "x")
        cache.put("b", "en", "fr", "y")
        cache.put("c", "en", "fr", "z")
        assert cache.size == 2
        # Oldest entry "a" should be evicted
        assert cache.get("a", "en", "fr") is None
        assert cache.get("c", "en", "fr") == "z"

    def test_hit_rate(self) -> None:
        cache = TranslationCache()
        cache.put("hello", "en", "fr", "bonjour")
        cache.get("hello", "en", "fr")  # hit
        cache.get("missing", "en", "fr")  # miss
        assert cache.hit_rate == 0.5

    def test_clear(self) -> None:
        cache = TranslationCache()
        cache.put("a", "en", "fr", "x")
        cache.clear()
        assert cache.size == 0

    def test_disabled_cache(self) -> None:
        config = TranslationCacheConfig(enabled=False)
        cache = TranslationCache(config)
        cache.put("hello", "en", "fr", "bonjour")
        assert cache.get("hello", "en", "fr") is None
        assert cache.size == 0

    def test_different_language_pairs(self) -> None:
        cache = TranslationCache()
        cache.put("hello", "en", "fr", "bonjour")
        cache.put("hello", "en", "de", "hallo")
        assert cache.get("hello", "en", "fr") == "bonjour"
        assert cache.get("hello", "en", "de") == "hallo"
