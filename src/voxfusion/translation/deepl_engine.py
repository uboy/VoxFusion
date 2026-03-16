"""Cloud-based translation via DeepL API.

Requires a DeepL API key (free or pro tier).  The ``deepl`` Python
package (MIT license) is used as the client library.
"""

import asyncio

from voxfusion.config.models import TranslationConfig
from voxfusion.exceptions import (
    TranslationAPIError,
    TranslationError,
    UnsupportedLanguagePair,
)
from voxfusion.logging import get_logger
from voxfusion.translation.cache import TranslationCache

log = get_logger(__name__)

# DeepL uses uppercase ISO 639-1 codes, with regional variants
_DEEPL_SUPPORTED_SOURCES = {
    "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr",
    "hu", "id", "it", "ja", "ko", "lt", "lv", "nb", "nl", "pl",
    "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "zh",
}

_DEEPL_SUPPORTED_TARGETS = {
    "bg", "cs", "da", "de", "el", "en-gb", "en-us", "es", "et",
    "fi", "fr", "hu", "id", "it", "ja", "ko", "lt", "lv", "nb",
    "nl", "pl", "pt-br", "pt-pt", "ro", "ru", "sk", "sl", "sv",
    "tr", "uk", "zh-hans", "zh-hant",
}


class DeepLTranslationEngine:
    """Translation engine using the DeepL API.

    Requires the ``deepl`` package and a valid API key set via
    configuration or the ``VOXFUSION_TRANSLATION__DEEPL_API_KEY``
    environment variable.
    """

    def __init__(
        self,
        config: TranslationConfig | None = None,
        api_key: str | None = None,
    ) -> None:
        self._config = config or TranslationConfig()
        self._api_key = api_key
        self._cache = TranslationCache(self._config.cache)
        self._translator: object | None = None

    def _get_translator(self) -> object:
        """Lazily create the DeepL translator client."""
        if self._translator is not None:
            return self._translator

        try:
            import deepl
        except ImportError as exc:
            raise TranslationError(
                "deepl package is not installed. "
                "Install with: pip install deepl"
            ) from exc

        key = self._api_key
        if not key:
            raise TranslationError(
                "DeepL API key required. Set via config or "
                "VOXFUSION_TRANSLATION__DEEPL_API_KEY environment variable."
            )

        self._translator = deepl.Translator(key)
        log.info("deepl.translator_created")
        return self._translator

    def _translate_sync(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        """Synchronous translation via DeepL API."""
        import deepl

        src_upper = source_language.upper()
        tgt_upper = target_language.upper()

        # Validate language codes
        if source_language.lower() not in _DEEPL_SUPPORTED_SOURCES:
            raise UnsupportedLanguagePair(
                f"Source language {source_language!r} not supported by DeepL. "
                f"Supported: {sorted(_DEEPL_SUPPORTED_SOURCES)}"
            )

        translator = self._get_translator()

        try:
            result = translator.translate_text(  # type: ignore[union-attr]
                text,
                source_lang=src_upper,
                target_lang=tgt_upper,
            )
            return str(result)
        except deepl.DeepLException as exc:
            raise TranslationAPIError(
                f"DeepL API error: {exc}"
            ) from exc

    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        """Translate text using DeepL API.

        Results are cached to avoid redundant API calls.
        """
        cached = self._cache.get(text, source_language, target_language)
        if cached is not None:
            return cached

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, self._translate_sync, text, source_language, target_language
        )

        self._cache.put(text, source_language, target_language, result)
        return result

    async def translate_batch(
        self,
        texts: list[str],
        source_language: str,
        target_language: str,
    ) -> list[str]:
        """Translate multiple texts via DeepL.

        DeepL supports batch translation natively, but we use
        sequential calls with caching for consistency.
        """
        return [
            await self.translate(t, source_language, target_language)
            for t in texts
        ]
