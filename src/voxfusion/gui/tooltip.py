"""Reusable hover tooltip widget for Tkinter.

Usage::

    from voxfusion.gui.tooltip import ToolTip, create_help_icon

    # Attach a tooltip to any existing widget
    ToolTip(widget, text="Explanation of what this does.")

    # Create a small '?' icon that shows a tooltip on hover
    create_help_icon(parent_frame, text="More details here.")
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class ToolTip:
    """Shows a floating tooltip when the pointer hovers over a widget.

    The tooltip appears after ``SHOW_DELAY_MS`` milliseconds and is dismissed
    immediately when the pointer leaves the widget or a mouse button is pressed.

    Args:
        widget: The Tkinter widget to attach the tooltip to.
        text: Tooltip text.  Use ``\\n`` for explicit line breaks.
    """

    SHOW_DELAY_MS: int = 400
    AUTO_HIDE_MS: int = 4500
    WRAP_LENGTH_PX: int = 320
    _active_tip: ToolTip | None = None

    def __init__(self, widget: tk.Widget, text: str) -> None:
        self._widget = widget
        self._text = text
        self._tip_window: tk.Toplevel | None = None
        self._show_id: str | None = None
        self._hide_id: str | None = None

        widget.bind("<Enter>", self._schedule_show, add="+")
        widget.bind("<Leave>", self._cancel_and_hide, add="+")
        widget.bind("<ButtonPress>", self._cancel_and_hide, add="+")
        widget.bind("<FocusOut>", self._cancel_and_hide, add="+")
        widget.bind("<Unmap>", self._cancel_and_hide, add="+")
        widget.bind("<Destroy>", self._cancel_and_hide, add="+")
        try:
            widget.winfo_toplevel().bind("<FocusOut>", self._cancel_and_hide, add="+")
            widget.winfo_toplevel().bind("<Unmap>", self._cancel_and_hide, add="+")
        except tk.TclError:
            pass

    def set_text(self, text: str) -> None:
        """Replace tooltip text for already-bound widgets."""
        self._text = text
        if self._tip_window is not None:
            self._cancel_and_hide()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _schedule_show(self, _event: object) -> None:
        if self._show_id is not None:
            try:
                self._widget.after_cancel(self._show_id)
            except tk.TclError:
                pass
        self._show_id = self._widget.after(self.SHOW_DELAY_MS, self._show)

    def _show(self) -> None:
        self._show_id = None
        if self._tip_window or not self._text:
            return
        if ToolTip._active_tip is not None and ToolTip._active_tip is not self:
            ToolTip._active_tip._cancel_and_hide()
        try:
            x = self._widget.winfo_rootx() + 20
            y = self._widget.winfo_rooty() + self._widget.winfo_height() + 4
        except tk.TclError:
            return

        self._tip_window = tw = tk.Toplevel(self._widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        try:
            tw.wm_attributes("-topmost", True)
        except tk.TclError:
            pass

        tk.Label(
            tw,
            text=self._text,
            justify=tk.LEFT,
            background="#FFFFCC",
            foreground="#333333",
            relief=tk.SOLID,
            borderwidth=1,
            wraplength=self.WRAP_LENGTH_PX,
            padx=6,
            pady=4,
        ).pack()
        ToolTip._active_tip = self
        try:
            self._hide_id = self._widget.after(self.AUTO_HIDE_MS, self._cancel_and_hide)
        except tk.TclError:
            self._hide_id = None

    def _cancel_and_hide(self, _event: object = None) -> None:
        if self._show_id is not None:
            try:
                self._widget.after_cancel(self._show_id)
            except tk.TclError:
                pass
            self._show_id = None
        if self._hide_id is not None:
            try:
                self._widget.after_cancel(self._hide_id)
            except tk.TclError:
                pass
            self._hide_id = None
        if self._tip_window is not None:
            try:
                self._tip_window.destroy()
            except tk.TclError:
                pass
            self._tip_window = None
        if ToolTip._active_tip is self:
            ToolTip._active_tip = None


def create_help_icon(
    parent: tk.Widget,
    text: str,
    *,
    side: str = tk.LEFT,
    padx: int | tuple[int, int] = (2, 4),
) -> ttk.Label:
    """Pack a small circled-? label into *parent* with a tooltip attached.

    Args:
        parent: The frame to pack the icon into.
        text: Tooltip text shown on hover.
        side: ``pack()`` side argument (default ``tk.LEFT``).
        padx: ``pack()`` padx argument.

    Returns:
        The label widget (can be ignored).
    """
    icon = ttk.Label(
        parent,
        text="(?)",
        foreground="#4488CC",
        cursor="question_arrow",
    )
    icon.pack(side=side, padx=padx)
    ToolTip(icon, text)
    return icon
