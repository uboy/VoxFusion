"""Unit tests for voxfusion.output formatters."""

import json
from pathlib import Path

import pytest

from voxfusion.models.result import TranscriptionResult
from voxfusion.output import get_formatter, FORMATTERS
from voxfusion.output.json_formatter import JSONFormatter
from voxfusion.output.srt_formatter import SRTFormatter
from voxfusion.output.txt_formatter import TXTFormatter
from voxfusion.output.vtt_formatter import VTTFormatter


class TestFormatterRegistry:
    def test_all_formats_registered(self) -> None:
        assert set(FORMATTERS.keys()) == {"json", "srt", "vtt", "txt"}

    def test_get_formatter_json(self) -> None:
        f = get_formatter("json")
        assert isinstance(f, JSONFormatter)

    def test_get_formatter_srt(self) -> None:
        f = get_formatter("srt")
        assert isinstance(f, SRTFormatter)

    def test_get_formatter_vtt(self) -> None:
        f = get_formatter("vtt")
        assert isinstance(f, VTTFormatter)

    def test_get_formatter_txt(self) -> None:
        f = get_formatter("txt")
        assert isinstance(f, TXTFormatter)

    def test_get_formatter_unknown(self) -> None:
        from voxfusion.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="Unknown output format"):
            get_formatter("xml")


class TestJSONFormatter:
    def test_format_returns_valid_json(
        self, transcription_result: TranscriptionResult
    ) -> None:
        f = JSONFormatter()
        output = f.format(transcription_result)
        data = json.loads(output)
        assert "segments" in data
        assert "voxfusion_version" in data

    def test_format_segment_count(
        self, multi_segment_result: TranscriptionResult
    ) -> None:
        f = JSONFormatter()
        output = f.format(multi_segment_result)
        data = json.loads(output)
        assert len(data["segments"]) == 2

    def test_segment_fields(self, transcription_result: TranscriptionResult) -> None:
        f = JSONFormatter()
        data = json.loads(f.format(transcription_result))
        seg = data["segments"][0]
        assert seg["original_text"] == "Hello world"
        assert seg["speaker_id"] == "SPEAKER_00"
        assert seg["original_language"] == "en"

    def test_format_name(self) -> None:
        assert JSONFormatter().format_name == "json"

    def test_file_extension(self) -> None:
        assert JSONFormatter().file_extension == ".json"

    def test_write(self, transcription_result: TranscriptionResult, tmp_path: Path) -> None:
        f = JSONFormatter()
        out = tmp_path / "output.json"
        f.write(transcription_result, out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert len(data["segments"]) == 1


class TestSRTFormatter:
    def test_format_structure(self, transcription_result: TranscriptionResult) -> None:
        f = SRTFormatter()
        output = f.format(transcription_result)
        lines = output.strip().split("\n")
        assert lines[0] == "1"  # sequence number
        assert "-->" in lines[1]  # timestamp line

    def test_multi_segment(self, multi_segment_result: TranscriptionResult) -> None:
        f = SRTFormatter()
        output = f.format(multi_segment_result)
        assert output.count("-->") == 2

    def test_format_name(self) -> None:
        assert SRTFormatter().format_name == "srt"


class TestVTTFormatter:
    def test_starts_with_webvtt(self, transcription_result: TranscriptionResult) -> None:
        f = VTTFormatter()
        output = f.format(transcription_result)
        assert output.startswith("WEBVTT")

    def test_contains_timestamps(self, transcription_result: TranscriptionResult) -> None:
        f = VTTFormatter()
        output = f.format(transcription_result)
        assert "-->" in output

    def test_format_name(self) -> None:
        assert VTTFormatter().format_name == "vtt"


class TestTXTFormatter:
    def test_format_contains_text(
        self, transcription_result: TranscriptionResult
    ) -> None:
        f = TXTFormatter()
        output = f.format(transcription_result)
        assert "Hello world" in output

    def test_format_contains_speaker(
        self, transcription_result: TranscriptionResult
    ) -> None:
        f = TXTFormatter()
        output = f.format(transcription_result)
        assert "SPEAKER_00" in output

    def test_format_name(self) -> None:
        assert TXTFormatter().format_name == "txt"
