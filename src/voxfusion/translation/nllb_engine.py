"""Offline translation using Meta's NLLB-200 model via CTranslate2.

NLLB (No Language Left Behind) supports 200+ languages.  This engine
uses the CTranslate2-optimized version for efficient CPU/GPU inference.
The model is distributed under CC-BY-NC 4.0 (non-commercial research);
check licensing requirements for your use case.
"""

import asyncio

from voxfusion.config.models import TranslationConfig
from voxfusion.exceptions import TranslationError, UnsupportedLanguagePair
from voxfusion.logging import get_logger
from voxfusion.translation.cache import TranslationCache

log = get_logger(__name__)

# NLLB uses BCP-47 style language codes with script tags
_NLLB_LANG_MAP: dict[str, str] = {
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "nl": "nld_Latn",
    "ru": "rus_Cyrl",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "tr": "tur_Latn",
    "pl": "pol_Latn",
    "uk": "ukr_Cyrl",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "sv": "swe_Latn",
    "da": "dan_Latn",
    "fi": "fin_Latn",
    "no": "nob_Latn",
    "cs": "ces_Latn",
    "ro": "ron_Latn",
    "hu": "hun_Latn",
    "el": "ell_Grek",
    "he": "heb_Hebr",
    "id": "ind_Latn",
    "ms": "zsm_Latn",
    "bn": "ben_Beng",
    "ta": "tam_Taml",
}


def _to_nllb_code(lang: str) -> str:
    """Convert ISO 639-1 code to NLLB language code."""
    code = _NLLB_LANG_MAP.get(lang)
    if code is None:
        # If already an NLLB code, pass through
        if "_" in lang and len(lang) == 8:
            return lang
        raise UnsupportedLanguagePair(
            f"Language {lang!r} not in NLLB language map. "
            f"Available: {sorted(_NLLB_LANG_MAP)}"
        )
    return code


class NLLBTranslationEngine:
    """Translation engine using Meta's NLLB-200 via CTranslate2.

    The model is lazily loaded on first ``translate`` call.
    Inference is offloaded to a thread executor.
    """

    def __init__(
        self,
        config: TranslationConfig | None = None,
        model_path: str = "facebook/nllb-200-distilled-600M",
    ) -> None:
        self._config = config or TranslationConfig()
        self._model_path = model_path
        self._cache = TranslationCache(self._config.cache)
        self._translator: object | None = None
        self._tokenizer: object | None = None

    def _load_model(self) -> None:
        """Load the NLLB CTranslate2 model and tokenizer."""
        if self._translator is not None:
            return

        try:
            import ctranslate2
            import transformers
        except ImportError as exc:
            raise TranslationError(
                "ctranslate2 and transformers are required for NLLB. "
                "Install with: pip install ctranslate2 transformers"
            ) from exc

        log.info("nllb.loading_model", model=self._model_path)
        try:
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                self._model_path,
            )
            self._translator = ctranslate2.Translator(
                self._model_path,
                device="auto",
            )
        except Exception as exc:
            raise TranslationError(
                f"Failed to load NLLB model: {exc}"
            ) from exc

        log.info("nllb.model_loaded")

    def _translate_sync(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        """Synchronous translation using CTranslate2."""
        self._load_model()

        src_code = _to_nllb_code(source_language)
        tgt_code = _to_nllb_code(target_language)

        self._tokenizer.src_lang = src_code  # type: ignore[union-attr]
        tokens = self._tokenizer.convert_ids_to_tokens(  # type: ignore[union-attr]
            self._tokenizer.encode(text)  # type: ignore[union-attr]
        )

        results = self._translator.translate_batch(  # type: ignore[union-attr]
            [tokens],
            target_prefix=[[tgt_code]],
            beam_size=4,
        )

        output_tokens = results[0].hypotheses[0]
        # Remove the target language prefix token
        if output_tokens and output_tokens[0] == tgt_code:
            output_tokens = output_tokens[1:]

        translated = self._tokenizer.decode(  # type: ignore[union-attr]
            self._tokenizer.convert_tokens_to_ids(output_tokens)  # type: ignore[union-attr]
        )
        return translated.strip()

    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        """Translate text using NLLB.

        Results are cached to avoid redundant inference.
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
        """Translate multiple texts."""
        return [
            await self.translate(t, source_language, target_language)
            for t in texts
        ]
