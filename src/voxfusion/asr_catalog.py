"""Catalog metadata for ASR models and language compatibility."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageInfo:
    """User-facing language metadata."""

    code: str | None
    label: str


@dataclass(frozen=True)
class ASRModelInfo:
    """Metadata for a selectable ASR model."""

    id: str
    name: str
    engine: str
    description: str
    accuracy_score: float
    speed_score: float
    supported_languages: tuple[str, ...]
    supports_language_selection: bool = True
    supports_translation: bool = False
    recommended: bool = False
    supports_live_capture: bool = True


_WHISPER_LANGUAGE_ROWS: tuple[tuple[str, str], ...] = (
    ("af", "Afrikaans"),
    ("am", "Amharic"),
    ("ar", "Arabic"),
    ("as", "Assamese"),
    ("az", "Azerbaijani"),
    ("ba", "Bashkir"),
    ("be", "Belarusian"),
    ("bg", "Bulgarian"),
    ("bn", "Bengali"),
    ("bo", "Tibetan"),
    ("br", "Breton"),
    ("bs", "Bosnian"),
    ("ca", "Catalan"),
    ("cs", "Czech"),
    ("cy", "Welsh"),
    ("da", "Danish"),
    ("de", "German"),
    ("el", "Greek"),
    ("en", "English"),
    ("es", "Spanish"),
    ("et", "Estonian"),
    ("eu", "Basque"),
    ("fa", "Persian"),
    ("fi", "Finnish"),
    ("fo", "Faroese"),
    ("fr", "French"),
    ("gl", "Galician"),
    ("gu", "Gujarati"),
    ("ha", "Hausa"),
    ("haw", "Hawaiian"),
    ("he", "Hebrew"),
    ("hi", "Hindi"),
    ("hr", "Croatian"),
    ("ht", "Haitian Creole"),
    ("hu", "Hungarian"),
    ("hy", "Armenian"),
    ("id", "Indonesian"),
    ("is", "Icelandic"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("jw", "Javanese"),
    ("ka", "Georgian"),
    ("kk", "Kazakh"),
    ("km", "Khmer"),
    ("kn", "Kannada"),
    ("ko", "Korean"),
    ("la", "Latin"),
    ("lb", "Luxembourgish"),
    ("ln", "Lingala"),
    ("lo", "Lao"),
    ("lt", "Lithuanian"),
    ("lv", "Latvian"),
    ("mg", "Malagasy"),
    ("mi", "Maori"),
    ("mk", "Macedonian"),
    ("ml", "Malayalam"),
    ("mn", "Mongolian"),
    ("mr", "Marathi"),
    ("ms", "Malay"),
    ("mt", "Maltese"),
    ("my", "Myanmar"),
    ("ne", "Nepali"),
    ("nl", "Dutch"),
    ("nn", "Nynorsk"),
    ("no", "Norwegian"),
    ("oc", "Occitan"),
    ("pa", "Punjabi"),
    ("pl", "Polish"),
    ("ps", "Pashto"),
    ("pt", "Portuguese"),
    ("ro", "Romanian"),
    ("ru", "Russian"),
    ("sa", "Sanskrit"),
    ("sd", "Sindhi"),
    ("si", "Sinhala"),
    ("sk", "Slovak"),
    ("sl", "Slovenian"),
    ("sn", "Shona"),
    ("so", "Somali"),
    ("sq", "Albanian"),
    ("sr", "Serbian"),
    ("su", "Sundanese"),
    ("sv", "Swedish"),
    ("sw", "Swahili"),
    ("ta", "Tamil"),
    ("te", "Telugu"),
    ("tg", "Tajik"),
    ("th", "Thai"),
    ("tk", "Turkmen"),
    ("tl", "Tagalog"),
    ("tr", "Turkish"),
    ("tt", "Tatar"),
    ("uk", "Ukrainian"),
    ("ur", "Urdu"),
    ("uz", "Uzbek"),
    ("vi", "Vietnamese"),
    ("yi", "Yiddish"),
    ("yo", "Yoruba"),
    ("zh", "Chinese"),
)

AUTO_LANGUAGE = LanguageInfo(code=None, label="Auto Detect")
DEFAULT_LANGUAGE_CODE = "ru"

LANGUAGE_CATALOG: tuple[LanguageInfo, ...] = (
    AUTO_LANGUAGE,
    *(LanguageInfo(code=code, label=label) for code, label in _WHISPER_LANGUAGE_ROWS),
)

_LANGUAGE_BY_CODE = {language.code: language for language in LANGUAGE_CATALOG}
_LANGUAGE_BY_LABEL = {language.label: language for language in LANGUAGE_CATALOG}
_WHISPER_LANGUAGE_CODES = tuple(code for code, _label in _WHISPER_LANGUAGE_ROWS)

ASR_MODEL_CATALOG: tuple[ASRModelInfo, ...] = (
    ASRModelInfo(
        id="tiny",
        name="Whisper Tiny",
        engine="faster-whisper",
        description="Fastest model, lowest accuracy.",
        accuracy_score=0.35,
        speed_score=0.98,
        supported_languages=_WHISPER_LANGUAGE_CODES,
        supports_translation=True,
    ),
    ASRModelInfo(
        id="base",
        name="Whisper Base",
        engine="faster-whisper",
        description="Very fast with improved baseline quality.",
        accuracy_score=0.50,
        speed_score=0.90,
        supported_languages=_WHISPER_LANGUAGE_CODES,
        supports_translation=True,
    ),
    ASRModelInfo(
        id="small",
        name="Whisper Small",
        engine="faster-whisper",
        description="Balanced default for local transcription.",
        accuracy_score=0.68,
        speed_score=0.78,
        supported_languages=_WHISPER_LANGUAGE_CODES,
        supports_translation=True,
        recommended=True,
    ),
    ASRModelInfo(
        id="medium",
        name="Whisper Medium",
        engine="faster-whisper",
        description="Higher accuracy with moderate CPU cost.",
        accuracy_score=0.80,
        speed_score=0.58,
        supported_languages=_WHISPER_LANGUAGE_CODES,
        supports_translation=True,
    ),
    ASRModelInfo(
        id="large-v3",
        name="Whisper Large v3",
        engine="faster-whisper",
        description="Best quality among current built-in models, but slower.",
        accuracy_score=0.90,
        speed_score=0.34,
        supported_languages=_WHISPER_LANGUAGE_CODES,
        supports_translation=True,
    ),
    ASRModelInfo(
        id="gigaam-v3-e2e-ctc",
        name="GigaAM v3",
        engine="gigaam",
        description="Russian-focused ONNX/CTC model for file transcription.",
        accuracy_score=0.85,
        speed_score=0.75,
        supported_languages=("ru",),
        supports_translation=False,
        supports_live_capture=False,
    ),
    ASRModelInfo(
        id="parakeet-tdt-0.6b-v3",
        name="Parakeet V3",
        engine="parakeet",
        description="Fast and accurate. Supports 25 European languages incl. Russian/Ukrainian. File transcription only. Requires model download.",
        accuracy_score=0.80,
        speed_score=0.85,
        supported_languages=(
            "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de",
            "el", "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk",
            "sl", "es", "sv", "ru", "uk",
        ),
        supports_language_selection=False,
        supports_translation=False,
        supports_live_capture=False,
        recommended=True,
    ),
    ASRModelInfo(
        id="breeze-asr",
        name="Breeze ASR",
        engine="breeze",
        description="Whisper-based multilingual model. File transcription only. Requires model download.",
        accuracy_score=0.85,
        speed_score=0.35,
        supported_languages=_WHISPER_LANGUAGE_CODES,
        supports_translation=False,
        supports_live_capture=False,
    ),
)

# ---------------------------------------------------------------------------
# Quality presets
# These are merged into the ASR config overrides at transcription time.
# Keys match ASRConfig field names; vad_parameters is a nested dict.
# Only faster-whisper reads these fields; other engines ignore them silently.
# ---------------------------------------------------------------------------

#: Human-readable labels in the order shown in the UI dropdown.
QUALITY_PRESET_LABELS: tuple[str, ...] = ("Fast", "Balanced", "Quality")

#: Maps lowercase preset key → ASRConfig override dict.
QUALITY_PRESETS: dict[str, dict[str, object]] = {
    "fast": {
        # Speed-optimised: greedy decoding, aggressive VAD, int8 quantisation.
        "compute_type": "int8",
        "beam_size": 1,
        "best_of": 1,
        "vad_filter": True,
        "vad_parameters": {"threshold": 0.6, "min_silence_duration_ms": 1000},
    },
    "balanced": {
        # Default: good accuracy/speed tradeoff.
        "compute_type": "int8_float32",
        "beam_size": 5,
        "best_of": 5,
        "vad_filter": True,
        "vad_parameters": {"threshold": 0.5, "min_silence_duration_ms": 2000},
    },
    "quality": {
        # Best accuracy: full-precision, wider beam, permissive VAD.
        "compute_type": "float32",
        "beam_size": 10,
        "best_of": 5,
        "vad_filter": True,
        "vad_parameters": {"threshold": 0.3, "min_silence_duration_ms": 500},
    },
}

#: Map label → key (e.g. "Balanced" → "balanced").
_PRESET_LABEL_TO_KEY: dict[str, str] = {
    label: label.lower() for label in QUALITY_PRESET_LABELS
}


def get_quality_preset(label_or_key: str) -> dict[str, object]:
    """Return the ASRConfig overrides dict for *label_or_key*.

    Accepts either a display label (``"Balanced"``) or a lowercase key
    (``"balanced"``).  Falls back to ``"balanced"`` for unknown values.
    """
    key = _PRESET_LABEL_TO_KEY.get(label_or_key, label_or_key.lower())
    return dict(QUALITY_PRESETS.get(key, QUALITY_PRESETS["balanced"]))


_MODEL_BY_ID = {model.id: model for model in ASR_MODEL_CATALOG}


def get_model_catalog() -> tuple[ASRModelInfo, ...]:
    """Return the ordered ASR model catalog."""
    return ASR_MODEL_CATALOG


def get_model_info(model_id: str | None) -> ASRModelInfo:
    """Return catalog metadata for *model_id* or the default model."""
    if model_id and model_id in _MODEL_BY_ID:
        return _MODEL_BY_ID[model_id]
    return _MODEL_BY_ID["small"]


def list_model_ids() -> tuple[str, ...]:
    """Return selectable model ids."""
    return tuple(model.id for model in ASR_MODEL_CATALOG)


def list_languages_for_model(model_id: str | None) -> tuple[LanguageInfo, ...]:
    """Return the language options supported by *model_id*."""
    model = get_model_info(model_id)
    if not model.supports_language_selection:
        return (AUTO_LANGUAGE,)
    languages = [AUTO_LANGUAGE]
    for code in model.supported_languages:
        info = _LANGUAGE_BY_CODE.get(code)
        if info is not None:
            languages.append(info)
    return tuple(languages)


def normalize_language_for_model(model_id: str | None, language_code: str | None) -> str | None:
    """Return a model-compatible language code, falling back to auto."""
    if language_code in (None, "", "auto"):
        return None
    model = get_model_info(model_id)
    if not model.supports_language_selection:
        return None
    if language_code in model.supported_languages:
        return language_code
    return None


def get_language_label(language_code: str | None, model_id: str | None = None) -> str:
    """Return a user-facing label for *language_code* within *model_id*."""
    normalized = normalize_language_for_model(model_id, language_code)
    info = _LANGUAGE_BY_CODE.get(normalized)
    return info.label if info is not None else AUTO_LANGUAGE.label


def get_language_code(label: str, model_id: str | None = None) -> str | None:
    """Resolve a label back to a compatible language code."""
    info = _LANGUAGE_BY_LABEL.get(label)
    if info is None:
        return None
    return normalize_language_for_model(model_id, info.code)
