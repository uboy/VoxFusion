"""Offline translation using Argos Translate.

Argos Translate is a libre, offline machine translation library
(MIT license). Models are downloaded on first use.
"""

import asyncio

from voxfusion.config.models import TranslationConfig
from voxfusion.exceptions import TranslationError, UnsupportedLanguagePair
from voxfusion.logging import get_logger
from voxfusion.translation.cache import TranslationCache

log = get_logger(__name__)


class ArgosTranslationEngine:
    """Translation engine backed by Argos Translate.

    Requires the ``argostranslate`` package to be installed.
    """

    def __init__(self, config: TranslationConfig | None = None) -> None:
        self._config = config or TranslationConfig()
        self._cache = TranslationCache(self._config.cache)
        self._installed_packages: list[object] = []

    @property
    def supported_language_pairs(self) -> list[tuple[str, str]]:
        """Return installed language pairs from Argos."""
        try:
            from argostranslate import package as argos_package

            return [
                (p.from_code, p.to_code)
                for p in argos_package.get_installed_packages()
            ]
        except ImportError:
            return []

    def _get_translation(self, text: str, source_lang: str, target_lang: str) -> str:
        """Synchronous translation via Argos."""
        try:
            from argostranslate import translate as argos_translate
        except ImportError as exc:
            raise TranslationError(
                "argostranslate is not installed. "
                "Install with: pip install argostranslate"
            ) from exc

        installed = argos_translate.get_installed_languages()
        src = next((l for l in installed if l.code == source_lang), None)
        tgt = next((l for l in installed if l.code == target_lang), None)

        if src is None or tgt is None:
            raise UnsupportedLanguagePair(
                f"Language pair {source_lang}->{target_lang} not installed. "
                f"Installed: {[l.code for l in installed]}"
            )

        translation = src.get_translation(tgt)
        if translation is None:
            raise UnsupportedLanguagePair(
                f"No translation model for {source_lang}->{target_lang}"
            )

        return translation.translate(text)

    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        """Translate text using Argos Translate.

        Results are cached to avoid redundant inference.
        """
        cached = self._cache.get(text, source_language, target_language)
        if cached is not None:
            return cached

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, self._get_translation, text, source_language, target_language
        )

        self._cache.put(text, source_language, target_language, result)
        return result

    async def translate_batch(
        self,
        texts: list[str],
        source_language: str,
        target_language: str,
    ) -> list[str]:
        """Translate multiple texts. Each is translated individually."""
        return [
            await self.translate(t, source_language, target_language)
            for t in texts
        ]
