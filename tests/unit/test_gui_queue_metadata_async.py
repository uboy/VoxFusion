"""Tests for asynchronous GUI queue metadata updates."""

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


class _FakeRoot:
    def after(self, _delay: int, callback, *args) -> None:  # noqa: ANN001
        callback(*args)


class _FakeFuture:
    def __init__(self, fn) -> None:  # noqa: ANN001
        self._fn = fn
        self._callback = None
        self._result = None

    def add_done_callback(self, callback) -> None:  # noqa: ANN001
        self._callback = callback

    def result(self):  # noqa: ANN201
        return self._result

    def resolve(self) -> None:
        self._result = self._fn()
        assert self._callback is not None
        self._callback(self)


class _FakeExecutor:
    def __init__(self) -> None:
        self.futures: list[_FakeFuture] = []

    def submit(self, fn):  # noqa: ANN001, ANN201
        future = _FakeFuture(fn)
        self.futures.append(future)
        return future


def _build_async_gui() -> TranscriptionGUI:
    gui = object.__new__(TranscriptionGUI)
    gui.root = _FakeRoot()
    gui._file_queue_items = {}
    gui._file_queue_lookup = {}
    gui._file_queue_serial = 0
    gui._file_active_queue_id = None
    gui._file_queue_generation = 0
    gui._file_queue_table = _FakeTreeview()
    gui._file_path_var = _FakeVar()
    gui._refresh_file_workflow = MagicMock()
    gui._queue_metadata_async_enabled = True
    gui._queue_metadata_executor = _FakeExecutor()
    return gui


def test_add_files_to_queue_inserts_immediately_and_updates_metadata_later(
    tmp_path: Path,
    monkeypatch,
) -> None:
    media_path = tmp_path / "meeting.webm"
    media_path.write_bytes(b"container")

    gui = _build_async_gui()

    monkeypatch.setattr(gui_main, "probe_media_size", lambda _path: 2048)
    monkeypatch.setattr(gui_main, "probe_media_metadata", lambda _path: (125.4, 2048))

    added = TranscriptionGUI._add_files_to_queue(gui, [media_path])

    assert added == 1
    assert gui._file_queue_table.item("file-1")["values"] == (
        str(media_path),
        "—",
        "2.0 KB",
        "Queued",
        "—",
        "",
    )

    gui._queue_metadata_executor.futures[0].resolve()

    assert gui._file_queue_table.item("file-1")["values"] == (
        str(media_path),
        "02:05",
        "2.0 KB",
        "Queued",
        "—",
        "",
    )


def test_apply_file_queue_metadata_ignores_stale_generation(tmp_path: Path) -> None:
    media_path = tmp_path / "meeting.webm"
    media_path.write_bytes(b"container")

    gui = _build_async_gui()
    TranscriptionGUI._add_files_to_queue(gui, [media_path])
    gui._file_queue_items["file-1"].metadata_generation = 9

    TranscriptionGUI._apply_file_queue_metadata(gui, "file-1", 8, 90.0, 1024)

    assert gui._file_queue_table.item("file-1")["values"] == (
        str(media_path),
        "—",
        "9 B",
        "Queued",
        "—",
        "",
    )

