"""Unit tests for voxfusion.config models and loader."""

import pytest

from voxfusion.config.loader import (
    _deep_merge,
    load_config,
    load_defaults,
    merge_configs,
)
from voxfusion.config.models import (
    ASRConfig,
    CaptureConfig,
    DiarizationConfig,
    OutputConfig,
    PipelineConfig,
    SecurityConfig,
    TranslationConfig,
    VADParameters,
)


class TestPipelineConfig:
    def test_defaults(self, default_config: PipelineConfig) -> None:
        assert default_config.log_level == "INFO"
        assert default_config.asr.engine == "faster-whisper"
        assert default_config.asr.model_size == "small"

    def test_asr_config_defaults(self) -> None:
        cfg = ASRConfig()
        assert cfg.beam_size == 5
        assert cfg.compute_type == "int8_float32"
        assert cfg.vad_filter is True

    def test_asr_config_normalizes_unknown_model_to_default(self) -> None:
        cfg = ASRConfig(model_size="unknown-model")
        assert cfg.model_size == "small"

    def test_asr_config_drops_unsupported_language_to_auto(self) -> None:
        cfg = ASRConfig(model_size="small", language="xx")
        assert cfg.language is None

    def test_capture_config_defaults(self) -> None:
        cfg = CaptureConfig()
        assert cfg.sample_rate == 44100  # Changed for compatibility
        assert cfg.channels == 1
        assert cfg.chunk_duration_ms == 500

    def test_output_config_defaults(self) -> None:
        cfg = OutputConfig()
        assert cfg.format == "json"

    def test_security_config_defaults(self) -> None:
        cfg = SecurityConfig()
        assert cfg.encrypt_output is False
        assert cfg.auto_delete_temp_files is True

    def test_translation_disabled_by_default(self) -> None:
        cfg = TranslationConfig()
        assert cfg.enabled is False

    def test_vad_parameters_validation(self) -> None:
        with pytest.raises(Exception):
            VADParameters(threshold=2.0)  # must be <= 1.0


class TestConfigLoader:
    def test_load_defaults(self) -> None:
        defaults = load_defaults()
        assert isinstance(defaults, dict)
        assert "asr" in defaults

    def test_deep_merge_simple(self) -> None:
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_deep_merge_nested(self) -> None:
        base = {"a": {"x": 1, "y": 2}}
        override = {"a": {"y": 3, "z": 4}}
        result = _deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 3, "z": 4}}

    def test_merge_configs(self) -> None:
        c1 = {"a": 1}
        c2 = {"b": 2}
        c3 = {"a": 3}
        result = merge_configs(c1, c2, c3)
        assert result == {"a": 3, "b": 2}

    def test_load_config_defaults(self) -> None:
        config = load_config()
        assert isinstance(config, PipelineConfig)
        assert config.asr.engine == "faster-whisper"

    def test_load_config_with_overrides(self) -> None:
        config = load_config({"asr": {"model_size": "large-v3"}})
        assert config.asr.model_size == "large-v3"
