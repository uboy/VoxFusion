"""Translation backend registry for pluggable translation engines.

Provides a central registry of translation engine factory functions
so new backends can be added without modifying the orchestrator.
"""

from collections.abc import Callable
from typing import Any

from voxfusion.config.models import TranslationConfig
from voxfusion.exceptions import ConfigurationError
from voxfusion.logging import get_logger
from voxfusion.translation.base import TranslationEngine

log = get_logger(__name__)

EngineFactory = Callable[[TranslationConfig], Any]

_REGISTRY: dict[str, EngineFactory] = {}


def register_backend(name: str, factory: EngineFactory) -> None:
    """Register a translation backend factory.

    Args:
        name: Backend identifier (e.g. ``"argos"``, ``"deepl"``).
        factory: Callable that takes a ``TranslationConfig`` and returns
                 a ``TranslationEngine``-compatible object.
    """
    _REGISTRY[name] = factory


def get_backend(config: TranslationConfig) -> Any:
    """Instantiate the translation backend specified in *config*.

    Raises:
        ConfigurationError: If the backend is not registered.
    """
    factory = _REGISTRY.get(config.backend)
    if factory is None:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ConfigurationError(
            f"Unknown translation backend {config.backend!r}. "
            f"Available: {available}"
        )
    return factory(config)


def list_backends() -> list[str]:
    """Return names of all registered backends."""
    return sorted(_REGISTRY)


def get_translation_engine(backend_name: str, config: TranslationConfig) -> TranslationEngine:
    """Get a translation engine instance by name.

    Args:
        backend_name: Name of the backend (e.g., "argos", "deepl").
        config: Translation configuration.

    Returns:
        A TranslationEngine instance.

    Raises:
        ConfigurationError: If backend is not registered.
    """
    factory = _REGISTRY.get(backend_name)
    if factory is None:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ConfigurationError(
            f"Unknown translation backend {backend_name!r}. "
            f"Available: {available}"
        )
    return factory(config)


def _register_defaults() -> None:
    """Register built-in backends (lazy imports)."""
    def _argos_factory(config: TranslationConfig) -> Any:
        from voxfusion.translation.argos_engine import ArgosTranslationEngine
        return ArgosTranslationEngine(config)

    def _nllb_factory(config: TranslationConfig) -> Any:
        from voxfusion.translation.nllb_engine import NLLBTranslationEngine
        return NLLBTranslationEngine(config)

    def _deepl_factory(config: TranslationConfig) -> Any:
        from voxfusion.translation.deepl_engine import DeepLTranslationEngine
        return DeepLTranslationEngine(config)

    def _libretranslate_factory(config: TranslationConfig) -> Any:
        from voxfusion.translation.libretranslate import LibreTranslateEngine
        return LibreTranslateEngine(config)

    register_backend("argos", _argos_factory)
    register_backend("nllb", _nllb_factory)
    register_backend("deepl", _deepl_factory)
    register_backend("libretranslate", _libretranslate_factory)


_register_defaults()
