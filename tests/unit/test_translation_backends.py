"""Tests for translation backend registry and engine instantiation."""

from voxfusion.config.models import TranslationConfig
from voxfusion.exceptions import ConfigurationError
from voxfusion.translation.registry import get_backend, list_backends

import pytest


class TestTranslationRegistry:
    """Tests for the translation backend registry."""

    def test_all_backends_registered(self):
        backends = list_backends()
        assert "argos" in backends
        assert "nllb" in backends
        assert "deepl" in backends
        assert "libretranslate" in backends

    def test_four_backends_total(self):
        assert len(list_backends()) == 4

    def test_get_argos_backend(self):
        config = TranslationConfig(backend="argos")
        engine = get_backend(config)
        assert engine.__class__.__name__ == "ArgosTranslationEngine"

    def test_get_nllb_backend(self):
        config = TranslationConfig(backend="nllb")
        engine = get_backend(config)
        assert engine.__class__.__name__ == "NLLBTranslationEngine"

    def test_get_deepl_backend(self):
        config = TranslationConfig(backend="deepl")
        engine = get_backend(config)
        assert engine.__class__.__name__ == "DeepLTranslationEngine"

    def test_get_libretranslate_backend(self):
        config = TranslationConfig(backend="libretranslate")
        engine = get_backend(config)
        assert engine.__class__.__name__ == "LibreTranslateEngine"

    def test_unknown_backend_raises(self):
        config = TranslationConfig(backend="nonexistent")
        with pytest.raises(ConfigurationError, match="Unknown translation backend"):
            get_backend(config)


class TestNLLBLangMap:
    """Tests for NLLB language code mapping."""

    def test_known_language(self):
        from voxfusion.translation.nllb_engine import _to_nllb_code
        assert _to_nllb_code("en") == "eng_Latn"
        assert _to_nllb_code("fr") == "fra_Latn"
        assert _to_nllb_code("zh") == "zho_Hans"

    def test_passthrough_nllb_code(self):
        from voxfusion.translation.nllb_engine import _to_nllb_code
        assert _to_nllb_code("eng_Latn") == "eng_Latn"

    def test_unknown_language_raises(self):
        from voxfusion.translation.nllb_engine import _to_nllb_code
        from voxfusion.exceptions import UnsupportedLanguagePair
        with pytest.raises(UnsupportedLanguagePair):
            _to_nllb_code("xx")


class TestLibreTranslateEngine:
    """Tests for LibreTranslateEngine construction."""

    def test_default_url(self):
        from voxfusion.translation.libretranslate import LibreTranslateEngine
        engine = LibreTranslateEngine()
        assert engine._api_url == "http://localhost:5000"

    def test_custom_url(self):
        from voxfusion.translation.libretranslate import LibreTranslateEngine
        engine = LibreTranslateEngine(api_url="http://myserver:8080/")
        assert engine._api_url == "http://myserver:8080"

    def test_api_key_stored(self):
        from voxfusion.translation.libretranslate import LibreTranslateEngine
        engine = LibreTranslateEngine(api_key="test-key")
        assert engine._api_key == "test-key"
