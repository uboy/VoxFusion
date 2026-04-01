"""Tests for GUI file-queue metadata columns."""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import MagicMock

from voxfusion.gui.main import TranscriptionGUI

gui_main = importlib.import_module("voxfusion.gui.main")


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


def test_add_files_to_queue_populates_duration_and_size_columns(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audio_path = tmp_path / "meeting.webm"
    audio_path.write_bytes(b"container")

    gui = object.__new__(TranscriptionGUI)
    gui._file_queue_items = {}
    gui._file_queue_lookup = {}
    gui._file_queue_serial = 0
    gui._file_active_queue_id = None
    gui._file_queue_table = _FakeTreeview()
    gui._file_path_var = _FakeVar()
    gui._refresh_file_workflow = MagicMock()

    monkeypatch.setattr(gui_main, "probe_media_metadata", lambda _path: (125.4, 3 * 1024 * 1024))

    added = TranscriptionGUI._add_files_to_queue(gui, [audio_path])

    assert added == 1
    row = gui._file_queue_table.item("file-1")["values"]
    assert row == (
        str(audio_path),
        "02:05",
        "3.0 MB",
        "Queued",
        "—",
        "",
    )
