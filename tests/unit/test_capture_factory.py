"""Unit tests for voxfusion.capture.factory."""

import sys

import pytest

from voxfusion.capture.factory import create_file_source, detect_platform
from voxfusion.exceptions import UnsupportedPlatformError


class TestDetectPlatform:
    def test_returns_string(self) -> None:
        result = detect_platform()
        assert result in ("wasapi", "coreaudio", "pulseaudio", "generic")

    def test_windows(self) -> None:
        if sys.platform == "win32":
            assert detect_platform() == "wasapi"

    def test_linux(self) -> None:
        if sys.platform.startswith("linux"):
            assert detect_platform() == "pulseaudio"


class TestCreateFileSource:
    def test_creates_file_source(self, tmp_path) -> None:
        import soundfile as sf
        import numpy as np

        wav = tmp_path / "test.wav"
        sf.write(str(wav), np.zeros(16000, dtype="float32"), 16000)
        source = create_file_source(wav)
        assert source.device_name == "file:test.wav"

    def test_respects_config(self, tmp_path) -> None:
        import soundfile as sf
        import numpy as np
        from voxfusion.config.models import CaptureConfig

        wav = tmp_path / "test.wav"
        sf.write(str(wav), np.zeros(16000, dtype="float32"), 16000)
        config = CaptureConfig(chunk_duration_ms=250)
        source = create_file_source(wav, config)
        assert source._chunk_duration_ms == 250
