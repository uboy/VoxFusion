"""Tests for GUI tooltip lifecycle behavior."""

from __future__ import annotations

import voxfusion.gui.tooltip as tooltip_mod
from voxfusion.gui.tooltip import ToolTip


class _FakeTopLevel:
    def __init__(self) -> None:
        self.bindings: dict[str, object] = {}

    def bind(self, event: str, callback, add: str | None = None) -> None:  # noqa: ANN001
        self.bindings[event] = callback


class _FakeWidget:
    def __init__(self) -> None:
        self.bindings: dict[str, object] = {}
        self.after_calls: list[tuple[int, object]] = []
        self.cancelled: list[str] = []
        self._top = _FakeTopLevel()
        self._counter = 0

    def bind(self, event: str, callback, add: str | None = None) -> None:  # noqa: ANN001
        self.bindings[event] = callback

    def winfo_toplevel(self) -> _FakeTopLevel:
        return self._top

    def after(self, delay: int, callback):  # noqa: ANN001, ANN201
        self._counter += 1
        handle = f"after-{self._counter}"
        self.after_calls.append((delay, callback))
        return handle

    def after_cancel(self, handle: str) -> None:
        self.cancelled.append(handle)

    def winfo_rootx(self) -> int:
        return 100

    def winfo_rooty(self) -> int:
        return 200

    def winfo_height(self) -> int:
        return 20


class _FakePopup:
    def __init__(self, _widget) -> None:  # noqa: ANN001
        self.destroyed = False

    def wm_overrideredirect(self, _value: bool) -> None:
        pass

    def wm_geometry(self, _value: str) -> None:
        pass

    def wm_attributes(self, _name: str, _value: bool) -> None:
        pass

    def destroy(self) -> None:
        self.destroyed = True


class _FakeLabel:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def pack(self) -> None:
        pass


def test_tooltip_binds_focus_loss_and_unmap_events() -> None:
    widget = _FakeWidget()
    ToolTip(widget, "hello")

    assert "<FocusOut>" in widget.bindings
    assert "<Unmap>" in widget.bindings
    assert "<Destroy>" in widget.bindings
    assert "<FocusOut>" in widget.winfo_toplevel().bindings
    assert "<Unmap>" in widget.winfo_toplevel().bindings


def test_tooltip_show_replaces_previous_active_tip(monkeypatch) -> None:
    monkeypatch.setattr(tooltip_mod.tk, "Toplevel", _FakePopup)
    monkeypatch.setattr(tooltip_mod.tk, "Label", _FakeLabel)

    first_widget = _FakeWidget()
    second_widget = _FakeWidget()
    first = ToolTip(first_widget, "first")
    second = ToolTip(second_widget, "second")

    first._show()  # noqa: SLF001
    assert ToolTip._active_tip is first
    assert first_widget.after_calls[-1][0] == ToolTip.AUTO_HIDE_MS

    second._show()  # noqa: SLF001

    assert ToolTip._active_tip is second
    assert first._tip_window is None  # noqa: SLF001

