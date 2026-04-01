"""Tests for GUI file-queue batch transcription helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from voxfusion.gui.main import _FileQueueItem
from voxfusion.gui.main import TranscriptionGUI


class _FakeVar:
    def __init__(self, value: str = "") -> None:
        self._value = value

    def get(self) -> str:
        return self._value

    def set(self, value: str) -> None:
        self._value = value


class _FakeTreeview:
    def __init__(self) -> None:
        self._items: dict[str, tuple[object, ...]] = {}
        self._order: list[str] = []
        self._selection: tuple[str, ...] = ()

    def insert(self, _parent: str, _index: object, iid: str, values: tuple[object, ...]) -> None:
        self._items[iid] = values
        self._order.append(iid)

    def item(self, iid: str, values: tuple[object, ...] | None = None) -> dict[str, tuple[object, ...]]:
        if values is not None:
            self._items[iid] = values
        return {"values": self._items[iid]}

    def delete(self, iid: str) -> None:
        self._items.pop(iid, None)
        self._order = [item_id for item_id in self._order if item_id != iid]
        self._selection = tuple(item_id for item_id in self._selection if item_id != iid)

    def selection(self) -> tuple[str, ...]:
        return self._selection

    def selection_set(self, item_id: str) -> None:
        self._selection = (item_id,)

    def selection_remove(self, _selection: tuple[str, ...]) -> None:
        self._selection = ()

    def focus(self, _item_id: str) -> None:
        pass

    def see(self, _item_id: str) -> None:
        pass

    def get_children(self) -> tuple[str, ...]:
        return tuple(self._order)


def test_add_files_to_queue_deduplicates_and_selects_first(tmp_path: Path) -> None:
    audio_a = tmp_path / "a.wav"
    audio_b = tmp_path / "b.wav"
    audio_a.write_bytes(b"RIFF")
    audio_b.write_bytes(b"RIFF")

    gui = object.__new__(TranscriptionGUI)
    gui._file_queue_items = {}
    gui._file_queue_lookup = {}
    gui._file_queue_serial = 0
    gui._file_active_queue_id = None
    gui._file_queue_table = _FakeTreeview()
    gui._file_path_var = _FakeVar()
    gui._refresh_file_workflow = MagicMock()

    added = TranscriptionGUI._add_files_to_queue(gui, [audio_a, audio_a, audio_b])

    assert added == 2
    assert gui._file_path_var.get() == str(audio_a)
    assert gui._file_queue_table.get_children() == ("file-1", "file-2")


def test_on_file_worker_finished_advances_to_next_queued_file(tmp_path: Path) -> None:
    transcript_path = tmp_path / "a.transcript.txt"

    gui = object.__new__(TranscriptionGUI)
    gui._file_queue_items = {
        "file-1": _FileQueueItem(file_path=tmp_path / "a.wav", status="In progress"),
        "file-2": _FileQueueItem(file_path=tmp_path / "b.wav", status="Queued"),
    }
    gui._file_active_queue_id = "file-1"
    gui._file_active_error_message = None
    gui._file_batch_cancel_requested = False
    gui._file_worker = MagicMock(_cancelled=False)
    gui._file_start_time = 1.0
    gui._file_time_label = MagicMock()
    gui._file_seg_count = 3
    gui._last_transcript_path = None
    gui._auto_save_transcript = MagicMock(return_value=transcript_path)
    gui._persist_gui_settings = MagicMock()
    gui._update_file_queue_row = MagicMock()
    gui._next_pending_file_queue_id = MagicMock(return_value="file-2")
    gui._start_next_file_in_queue = MagicMock()
    gui._finish_file_batch_run = MagicMock()

    TranscriptionGUI._on_file_worker_finished(gui)

    item = gui._file_queue_items["file-1"]
    assert item.status == "Done"
    assert item.output_path == transcript_path
    assert item.result == transcript_path.name
    assert gui._file_active_queue_id is None
    gui._start_next_file_in_queue.assert_called_once()
    gui._finish_file_batch_run.assert_not_called()
