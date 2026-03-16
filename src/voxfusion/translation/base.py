"""TranslationEngine protocol definition."""

from typing import Protocol


class TranslationEngine(Protocol):
    """Text translation engine interface."""

    @property
    def supported_language_pairs(self) -> list[tuple[str, str]]:
        """List of ``(source_lang, target_lang)`` pairs supported."""
        ...

    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        """Translate *text* from *source_language* to *target_language*."""
        ...

    async def translate_batch(
        self,
        texts: list[str],
        source_language: str,
        target_language: str,
    ) -> list[str]:
        """Translate multiple texts efficiently."""
        ...
