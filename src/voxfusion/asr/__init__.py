"""Automatic Speech Recognition engine integrations."""

from voxfusion.asr.base import ASREngine
from voxfusion.asr_catalog import (
    ASRModelInfo,
    LanguageInfo,
    get_language_code,
    get_language_label,
    get_model_catalog,
    get_model_info,
    list_languages_for_model,
    list_model_ids,
    normalize_language_for_model,
)
from voxfusion.asr.dedup import OverlapDeduplicator
from voxfusion.asr.faster_whisper import FasterWhisperEngine
from voxfusion.asr.gigaam_engine import GigaAMCTCEngine
from voxfusion.asr.streaming import StreamingASR

__all__ = [
    "ASREngine",
    "ASRModelInfo",
    "FasterWhisperEngine",
    "GigaAMCTCEngine",
    "LanguageInfo",
    "OverlapDeduplicator",
    "StreamingASR",
    "get_language_code",
    "get_language_label",
    "get_model_catalog",
    "get_model_info",
    "list_languages_for_model",
    "list_model_ids",
    "normalize_language_for_model",
]
