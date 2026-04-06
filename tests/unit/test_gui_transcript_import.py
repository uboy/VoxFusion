"""Tests for importing existing transcript files into the GUI results table."""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import MagicMock

from voxfusion.gui.main import TranscriptionGUI

gui_main = importlib.import_module("voxfusion.gui.main")


class _FakeTable:
    def __init__(self) -> None:
        self._rows: dict[str, tuple[str, str, str]] = {}
        self._counter = 0
        self.last_scroll: float | None = None

    def get_children(self) -> tuple[str, ...]:
        return tuple(self._rows)

    def delete(self, item: str) -> None:
        self._rows.pop(item, None)

    def insert(self, _parent: str, _index: str, *, values: tuple[str, str, str]) -> str:
        item_id = f"row-{self._counter}"
        self._counter += 1
        self._rows[item_id] = values
        return item_id

    def item(self, item: str, _option: str) -> tuple[str, str, str]:
        return self._rows[item]

    def yview_moveto(self, value: float) -> None:
        self.last_scroll = value


def test_parse_transcript_rows_preserves_timestamped_format() -> None:
    text = "[00:00:05] [SPEAKER_00] Hello\n[00:00:09] [SPEAKER_01] World\n"

    rows = TranscriptionGUI._parse_transcript_rows(text)

    assert rows == [
        ("00:00:05", "SPEAKER_00", "Hello"),
        ("00:00:09", "SPEAKER_01", "World"),
    ]


def test_parse_transcript_rows_falls_back_to_plain_text_lines() -> None:
    text = "First line\n\nSecond line\n"

    rows = TranscriptionGUI._parse_transcript_rows(text)

    assert rows == [
        ("00:00:00", "IMPORTED", "First line"),
        ("00:00:01", "IMPORTED", "Second line"),
    ]


def test_parse_srt_rows_extracts_timestamps_and_speaker() -> None:
    text = (
        "1\n"
        "00:00:05,100 --> 00:00:08,400\n"
        "[SPEAKER_00] Hello there\n\n"
        "2\n"
        "00:00:09,000 --> 00:00:11,000\n"
        "General Kenobi\n"
    )

    rows = TranscriptionGUI._parse_srt_rows(text)

    assert rows == [
        ("00:00:05", "SPEAKER_00", "Hello there"),
        ("00:00:09", "IMPORTED", "General Kenobi"),
    ]


def test_load_transcript_file_populates_results_table(tmp_path: Path, monkeypatch) -> None:
    transcript_path = tmp_path / "meeting.transcript.txt"
    transcript_path.write_text(
        "[00:00:05] [SPEAKER_00] Hello\n[00:00:09] [SPEAKER_01] World\n",
        encoding="utf-8",
    )

    fake_log = MagicMock()
    fake_table = _FakeTable()

    gui = object.__new__(TranscriptionGUI)
    gui._file_table = fake_table
    gui._file_seg_count = 7
    gui._file_segments = [object()]
    gui._last_transcript_path = None
    gui._file_progress = {}
    gui._file_seg_counter_label = MagicMock()
    gui._file_status_label = MagicMock()
    gui._clear_llm_output = MagicMock()
    gui._refresh_file_workflow = MagicMock()
    gui._tr = lambda key, **kwargs: key if not kwargs else f"{key}:{kwargs}"

    monkeypatch.setattr(gui_main, "log", fake_log)
    monkeypatch.setattr(gui_main.filedialog, "askopenfilename", lambda **kwargs: str(transcript_path))

    TranscriptionGUI._load_transcript_file(gui)

    assert fake_table.get_children() == ("row-0", "row-1")
    assert fake_table.item("row-0", "values") == ("00:00:05", "SPEAKER_00", "Hello")
    assert fake_table.item("row-1", "values") == ("00:00:09", "SPEAKER_01", "World")
    assert gui._file_seg_count == 2
    assert gui._file_segments == []
    assert gui._last_transcript_path == transcript_path
    assert gui._file_progress["value"] == 0
    assert fake_table.last_scroll == 1.0
    fake_log.info.assert_called_once_with(
        "gui.transcript_loaded",
        file=str(transcript_path),
        rows=2,
    )


def test_load_markdown_transcript_file_uses_plain_text_fallback(tmp_path: Path, monkeypatch) -> None:
    transcript_path = tmp_path / "meeting.md"
    transcript_path.write_text(
        "First line\n\nSecond line\n",
        encoding="utf-8",
    )

    fake_log = MagicMock()
    fake_table = _FakeTable()

    gui = object.__new__(TranscriptionGUI)
    gui._file_table = fake_table
    gui._file_seg_count = 0
    gui._file_segments = []
    gui._last_transcript_path = None
    gui._file_progress = {}
    gui._file_seg_counter_label = MagicMock()
    gui._file_status_label = MagicMock()
    gui._clear_llm_output = MagicMock()
    gui._refresh_file_workflow = MagicMock()
    gui._tr = lambda key, **kwargs: key if not kwargs else f"{key}:{kwargs}"

    monkeypatch.setattr(gui_main, "log", fake_log)
    monkeypatch.setattr(gui_main.filedialog, "askopenfilename", lambda **kwargs: str(transcript_path))

    TranscriptionGUI._load_transcript_file(gui)

    assert fake_table.get_children() == ("row-0", "row-1")
    assert fake_table.item("row-0", "values") == ("00:00:00", "IMPORTED", "First line")
    assert fake_table.item("row-1", "values") == ("00:00:01", "IMPORTED", "Second line")


def test_load_srt_transcript_file_populates_results_table(tmp_path: Path, monkeypatch) -> None:
    transcript_path = tmp_path / "meeting.srt"
    transcript_path.write_text(
        "1\n"
        "00:00:05,100 --> 00:00:08,400\n"
        "[SPEAKER_00] Hello there\n\n"
        "2\n"
        "00:00:09,000 --> 00:00:11,000\n"
        "General Kenobi\n",
        encoding="utf-8",
    )

    fake_log = MagicMock()
    fake_table = _FakeTable()

    gui = object.__new__(TranscriptionGUI)
    gui._file_table = fake_table
    gui._file_seg_count = 0
    gui._file_segments = []
    gui._last_transcript_path = None
    gui._file_progress = {}
    gui._file_seg_counter_label = MagicMock()
    gui._file_status_label = MagicMock()
    gui._clear_llm_output = MagicMock()
    gui._refresh_file_workflow = MagicMock()
    gui._tr = lambda key, **kwargs: key if not kwargs else f"{key}:{kwargs}"

    monkeypatch.setattr(gui_main, "log", fake_log)
    monkeypatch.setattr(gui_main.filedialog, "askopenfilename", lambda **kwargs: str(transcript_path))

    TranscriptionGUI._load_transcript_file(gui)

    assert fake_table.get_children() == ("row-0", "row-1")
    assert fake_table.item("row-0", "values") == ("00:00:05", "SPEAKER_00", "Hello there")
    assert fake_table.item("row-1", "values") == ("00:00:09", "IMPORTED", "General Kenobi")
