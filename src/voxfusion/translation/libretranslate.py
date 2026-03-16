"""Translation via LibreTranslate (self-hosted or public instance).

LibreTranslate is a free, open-source machine translation API
(AGPLv3 for server, MIT for client).  This engine communicates
with an instance via its REST API.
"""

import asyncio
import json
import urllib.error
import urllib.request

from voxfusion.config.models import TranslationConfig
from voxfusion.exceptions import TranslationAPIError, TranslationError
from voxfusion.logging import get_logger
from voxfusion.translation.cache import TranslationCache

log = get_logger(__name__)

_DEFAULT_API_URL = "http://localhost:5000"


class LibreTranslateEngine:
    """Translation engine using a LibreTranslate REST API.

    Communicates with a LibreTranslate instance (self-hosted or
    public) via HTTP.  No heavy dependencies required -- uses
    ``urllib`` from the standard library.
    """

    def __init__(
        self,
        config: TranslationConfig | None = None,
        api_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._config = config or TranslationConfig()
        self._api_url = (api_url or _DEFAULT_API_URL).rstrip("/")
        self._api_key = api_key or ""
        self._cache = TranslationCache(self._config.cache)

    def _translate_sync(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        """Synchronous translation via LibreTranslate REST API."""
        url = f"{self._api_url}/translate"

        payload = {
            "q": text,
            "source": source_language,
            "target": target_language,
            "format": "text",
        }
        if self._api_key:
            payload["api_key"] = self._api_key

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                translated = body.get("translatedText", "")
                if not translated:
                    raise TranslationError(
                        f"LibreTranslate returned empty response: {body}"
                    )
                return translated
        except urllib.error.HTTPError as exc:
            error_body = ""
            try:
                error_body = exc.read().decode("utf-8")
            except Exception:
                pass
            raise TranslationAPIError(
                f"LibreTranslate HTTP {exc.code}: {error_body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise TranslationAPIError(
                f"LibreTranslate connection error: {exc.reason}"
            ) from exc

    def _detect_sync(self, text: str) -> str:
        """Detect language of text via LibreTranslate API."""
        url = f"{self._api_url}/detect"

        payload = {"q": text}
        if self._api_key:
            payload["api_key"] = self._api_key

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                if isinstance(body, list) and body:
                    return body[0].get("language", "en")
                return "en"
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            log.warning("libretranslate.detect_failed", error=str(exc))
            return "en"

    def _get_languages_sync(self) -> list[dict[str, str]]:
        """Fetch supported languages from the LibreTranslate instance."""
        url = f"{self._api_url}/languages"
        req = urllib.request.Request(url, method="GET")

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            log.warning("libretranslate.languages_failed", error=str(exc))
            return []

    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        """Translate text using LibreTranslate.

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
        """Translate multiple texts via LibreTranslate."""
        return [
            await self.translate(t, source_language, target_language)
            for t in texts
        ]

    async def get_languages(self) -> list[dict[str, str]]:
        """Fetch supported languages from the server."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get_languages_sync)
