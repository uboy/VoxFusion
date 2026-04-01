"""Tests for media metadata probing fallbacks used by the GUI."""

from __future__ import annotations

from pathlib import Path

from voxfusion.gui.helpers import probe_media_metadata
import voxfusion.gui.helpers as gui_helpers


def test_probe_media_metadata_uses_extracted_audio_when_direct_probes_fail(
    tmp_path: Path,
    monkeypatch,
) -> None:
    media_path = tmp_path / "meeting.webm"
    media_path.write_bytes(b"container")
    extracted_path = tmp_path / "meeting.wav"
    extracted_path.write_bytes(b"RIFF")

    monkeypatch.setattr(gui_helpers, "_probe_duration_with_soundfile", lambda path: 42.5 if path == extracted_path else None)
    monkeypatch.setattr(gui_helpers, "_probe_duration_with_ffprobe", lambda _path: None)
    monkeypatch.setattr(gui_helpers, "needs_extraction", lambda _path: True)
    monkeypatch.setattr(gui_helpers, "extract_audio", lambda _path: extracted_path)

    duration_s, size_bytes = probe_media_metadata(media_path)

    assert duration_s == 42.5
    assert size_bytes == len(b"container")
    assert not extracted_path.exists()
