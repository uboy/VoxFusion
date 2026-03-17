"""Visual styling helpers for the Tkinter GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


def configure_gui_theme(root: tk.Tk) -> None:
    """Apply a more intentional ttk theme and sizing defaults."""
    style = ttk.Style(root)
    available = set(style.theme_names())
    if "clam" in available:
        style.theme_use("clam")

    root.configure(background="#f3f0e8")

    style.configure("TFrame", background="#f3f0e8")
    style.configure("TLabelframe", background="#f3f0e8", borderwidth=1, relief="solid")
    style.configure("TLabelframe.Label", background="#f3f0e8", foreground="#2f2618")
    style.configure("TLabel", background="#f3f0e8", foreground="#2f2618")
    style.configure("Header.TLabel", background="#f3f0e8", foreground="#1f170c", font=("", 10, "bold"))
    style.configure("Muted.TLabel", background="#f3f0e8", foreground="#6a6254")
    style.configure("Primary.TButton", padding=(10, 6))
    style.configure("Accent.TButton", padding=(10, 6))
    style.map(
        "Accent.TButton",
        background=[("active", "#d5b980"), ("!disabled", "#c7a861")],
        foreground=[("!disabled", "#20170a")],
    )
    style.configure("Treeview", rowheight=24, fieldbackground="#fffdfa", background="#fffdfa")
    style.configure("Treeview.Heading", padding=(8, 6))
    style.configure("TNotebook", background="#f3f0e8", tabmargins=(6, 6, 6, 0))
    style.configure("TNotebook.Tab", padding=(14, 8), background="#dfd7c8")
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
