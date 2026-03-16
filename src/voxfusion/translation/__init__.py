"""Text translation subsystem with pluggable backends."""

from voxfusion.translation.base import TranslationEngine
from voxfusion.translation.cache import TranslationCache
from voxfusion.translation.registry import get_backend, list_backends, register_backend

__all__ = [
    "TranslationCache",
    "TranslationEngine",
    "get_backend",
    "list_backends",
    "register_backend",
]

# Note: ArgosTranslationEngine, NLLBTranslationEngine, DeepLTranslationEngine,
# and LibreTranslateEngine are available via lazy imports through the registry
# or direct import from their respective modules.
