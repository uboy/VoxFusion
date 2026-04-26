"""Visual styling helpers for the Tkinter GUI."""

from __future__ import annotations

import platform
import tkinter as tk
from tkinter import ttk


def _apply_dpi_awareness(root: tk.Tk) -> None:
    """Enable HiDPI / Retina awareness so the window is crisp on high-DPI screens.

    * Windows — calls the Win32 ``SetProcessDpiAwareness(1)`` shim so the OS
      stops bitmap-scaling the window and lets Tk manage its own coordinates.
    * macOS — Tk on macOS already handles Retina natively; nothing to do.
    * Linux — reads the screen DPI reported by the X server and scales the Tk
      ``scaling`` factor proportionally so text and widgets are the right size.
    """
    os_name = platform.system()
    if os_name == "Windows":
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)  # type: ignore[attr-defined]
        except Exception:  # — ctypes call may not exist on all Windows versions
            pass
    elif os_name == "Linux":
        try:
            # X server reports DPI in millimetres; 25.4 mm/inch → DPI.
            # Tk's default scaling assumes 72 DPI; adjust the ratio accordingly.
            screen_mm = root.winfo_screenmwidth()
            screen_px = root.winfo_screenwidth()
            if screen_mm > 0:
                dpi = screen_px / screen_mm * 25.4
                scale = dpi / 96.0  # 96 DPI is the canonical "1×" baseline
                if scale > 1.1:  # only upscale, never downscale legacy 72-DPI displays
                    root.tk.call("tk", "scaling", scale)
        except Exception:  # — graceful degradation
            pass


def configure_gui_theme(root: tk.Tk) -> None:
    """Apply HiDPI awareness and a more intentional ttk theme and sizing defaults."""
    _apply_dpi_awareness(root)
    style = ttk.Style(root)
    available = set(style.theme_names())
    if "clam" in available:
        style.theme_use("clam")

    root.configure(background="#f3f0e8")

    style.configure("TFrame", background="#f3f0e8")
    style.configure("TLabelframe", background="#f3f0e8", borderwidth=1, relief="solid")
    style.configure("TLabelframe.Label", background="#f3f0e8", foreground="#2f2618")
    style.configure("TLabel", background="#f3f0e8", foreground="#2f2618")
    style.configure(
        "Header.TLabel", background="#f3f0e8", foreground="#1f170c", font=("", 9, "bold")
    )
    style.configure("Muted.TLabel", background="#f3f0e8", foreground="#6a6254")
    style.configure("Primary.TButton", padding=(8, 4))
    style.configure("Accent.TButton", padding=(8, 4))
    style.map(
        "Accent.TButton",
        background=[("active", "#d5b980"), ("!disabled", "#c7a861")],
        foreground=[("!disabled", "#20170a")],
    )
    style.configure("Treeview", rowheight=22, fieldbackground="#fffdfa", background="#fffdfa")
    style.configure("Treeview.Heading", padding=(6, 4))
    style.configure("TNotebook", background="#f3f0e8", tabmargins=(3, 3, 3, 0))
    style.configure("TNotebook.Tab", padding=(8, 4), background="#dfd7c8")
    style.map(
        "TNotebook.Tab",
        background=[("selected", "#fffdfa"), ("active", "#ece4d6")],
        foreground=[("selected", "#20170a"), ("!selected", "#5c5448")],
    )
    style.configure(
        "Horizontal.TProgressbar",
        troughcolor="#ddd5c8",
        background="#b6893d",
        bordercolor="#ddd5c8",
        lightcolor="#b6893d",
        darkcolor="#b6893d",
    )
