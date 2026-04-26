"""Live Capture tab widget builder.

``LiveCaptureTab`` is responsible for constructing all widgets inside the
Live Capture notebook tab.  It holds a reference to the parent
``TranscriptionGUI`` instance (``self._gui``) and delegates every callback,
state-variable access, and helper call to it.

This is Phase 1 of the ARCH-3 God-Object reduction.  All logic (methods,
state vars) still lives in ``TranscriptionGUI``; Phase 2 will move them here
incrementally.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

from voxfusion.asr_catalog import get_available_model_catalog
from voxfusion.gui.model_summary import ModelSummaryCard

if TYPE_CHECKING:
    from voxfusion.gui.main import TranscriptionGUI

_ASR_MODEL_CHOICES: tuple[str, ...] = tuple(m.id for m in get_available_model_catalog())


class LiveCaptureTab:
    """Builds the Live Capture tab widgets onto a given parent frame.

    All widget references are stored on the ``TranscriptionGUI`` instance
    (``gui``) so that the rest of the application code is unchanged.
    """

    def __init__(self, gui: TranscriptionGUI) -> None:
        self._gui = gui

    def build(self, parent: ttk.Frame) -> None:
        """Create all live-capture widgets inside *parent*."""
        _gui = self._gui
        live_paned = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        live_paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        top_frame = ttk.Frame(live_paned)
        live_paned.add(top_frame, weight=1)

        settings_box = ttk.LabelFrame(top_frame, text="", padding=(6, 5))
        settings_box.pack(fill=tk.X, padx=0, pady=(0, 4))
        _gui._bind_labelframe_text(settings_box, "live.section.capture_setup")
        settings_box.columnconfigure(1, weight=1)
        settings_box.columnconfigure(3, weight=2)

        # Row 0: Multi-select device picker
        _gui._live_devices_label = ttk.Label(settings_box, text="")
        _gui._live_devices_label.grid(row=0, column=0, sticky="w", padx=(0, 4))
        _gui._bind_text(_gui._live_devices_label, "live.label.devices")
        _gui.device_picker = ttk.Menubutton(
            settings_box,
            textvariable=_gui._device_picker_var,
            direction="below",
        )
        _gui.device_picker.grid(row=0, column=1, columnspan=3, sticky="ew", padx=(0, 12))
        _gui._device_menu = tk.Menu(_gui.device_picker, tearoff=0)
        _gui.device_picker.configure(menu=_gui._device_menu)
        _gui._bind_tooltip(_gui.device_picker, "tooltip.live.devices")

        # Row 1: Model | Language | Translate
        _gui._live_model_label = ttk.Label(settings_box, text="")
        _gui._live_model_label.grid(row=1, column=0, sticky="w", padx=(0, 4), pady=(4, 0))
        _gui._bind_text(_gui._live_model_label, "live.label.model")
        _gui.model_combo = ttk.Combobox(
            settings_box,
            textvariable=_gui._model_var,
            state="readonly",
            width=20,
            values=_ASR_MODEL_CHOICES,
        )
        _gui.model_combo.grid(row=1, column=1, sticky="w", padx=(0, 12), pady=(4, 0))
        _gui.model_combo.bind("<<ComboboxSelected>>", _gui._on_model_changed)
        _gui._bind_tooltip(_gui.model_combo, "tooltip.live.model")

        lang_row = ttk.Frame(settings_box)
        lang_row.grid(row=1, column=2, columnspan=2, sticky="ew", pady=(4, 0))
        _gui._live_language_label = ttk.Label(lang_row, text="")
        _gui._live_language_label.pack(side=tk.LEFT, padx=(0, 4))
        _gui._bind_text(_gui._live_language_label, "live.label.language")
        _gui.language_combo = ttk.Combobox(
            lang_row,
            textvariable=_gui._language_var,
            state="readonly",
            width=18,
        )
        _gui.language_combo.pack(side=tk.LEFT, padx=(0, 12))
        _gui.language_combo.bind("<<ComboboxSelected>>", _gui._on_live_language_changed)
        _gui._bind_tooltip(_gui.language_combo, "tooltip.live.language")
        _gui._live_translate_label = ttk.Label(lang_row, text="")
        _gui._live_translate_label.pack(side=tk.LEFT, padx=(0, 4))
        _gui._bind_text(_gui._live_translate_label, "live.label.translate")
        _gui.translate_entry = ttk.Entry(
            lang_row,
            textvariable=_gui._translate_var,
            width=8,
        )
        _gui.translate_entry.pack(side=tk.LEFT)
        _gui._bind_tooltip(_gui.translate_entry, "tooltip.live.translate")

        # Row 2: Action buttons + stats
        btn_row = ttk.Frame(settings_box)
        btn_row.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(6, 2))

        _gui.start_button = ttk.Button(
            btn_row, text="", command=_gui._start_capture, style="Primary.TButton"
        )
        _gui.start_button.pack(side=tk.LEFT, padx=(0, 4))
        _gui._bind_text(_gui.start_button, "live.button.start")
        _gui._bind_tooltip(_gui.start_button, "tooltip.live.start")
        _gui.stop_button = ttk.Button(btn_row, text="", command=_gui._stop_capture)
        _gui.stop_button.configure(state=tk.DISABLED)
        _gui.stop_button.pack(side=tk.LEFT, padx=(0, 4))
        _gui._bind_text(_gui.stop_button, "live.button.stop")
        _gui._bind_tooltip(_gui.stop_button, "tooltip.live.stop")
        _gui.pause_button = ttk.Button(btn_row, text="", command=_gui._toggle_recording_pause)
        _gui.pause_button.configure(state=tk.DISABLED)
        _gui.pause_button.pack(side=tk.LEFT, padx=(0, 4))
        _gui._bind_text(_gui.pause_button, "live.button.pause")
        _gui._bind_tooltip(_gui.pause_button, "tooltip.live.pause")
        _gui.record_button = ttk.Button(
            btn_row,
            text="",
            command=_gui._start_recording,
            style="Accent.TButton",
        )
        _gui.record_button.pack(side=tk.LEFT, padx=(0, 4))
        _gui._bind_text(_gui.record_button, "live.button.record_audio")
        _gui._bind_tooltip(_gui.record_button, "tooltip.live.record_audio")
        _gui._rec_format_combo = ttk.Combobox(
            btn_row,
            textvariable=_gui._rec_format_var,
            values=["wav", "ogg", "opus", "mp3"],
            state="readonly",
            width=5,
        )
        _gui._rec_format_combo.pack(side=tk.LEFT, padx=(0, 12))
        _gui._bind_tooltip(_gui._rec_format_combo, "tooltip.live.record_format")

        ttk.Separator(btn_row, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))

        _gui.clear_button = ttk.Button(btn_row, text="", command=_gui._clear_table)
        _gui.clear_button.pack(side=tk.LEFT, padx=(0, 4))
        _gui._bind_text(_gui.clear_button, "live.button.clear")
        _gui._bind_tooltip(_gui.clear_button, "tooltip.live.clear")
        _gui.save_button = ttk.Button(btn_row, text="", command=_gui._save_to_file)
        _gui.save_button.pack(side=tk.LEFT)
        _gui._bind_text(_gui.save_button, "live.button.save")
        _gui._bind_tooltip(_gui.save_button, "tooltip.live.save")

        _gui.queue_label = ttk.Label(btn_row, text="")
        _gui.queue_label.pack(side=tk.RIGHT, padx=(8, 0))
        _gui._bind_tooltip(_gui.queue_label, "tooltip.live.summary")
        _gui.counter_label = ttk.Label(btn_row, text="")
        _gui.counter_label.pack(side=tk.RIGHT)
        _gui._bind_tooltip(_gui.counter_label, "tooltip.live.segment_counter")

        # Hidden model summary (kept for API compatibility — not displayed)
        _gui._live_model_summary = ModelSummaryCard(
            settings_box,
            title=_gui._tr("model_summary.title.live"),
            translate=_gui._tr,
        )

        _gui.status_label = ttk.Label(top_frame, text="", anchor="w")
        _gui.status_label.pack(fill=tk.X, padx=0, pady=(2, 2))
        _gui._bind_tooltip(_gui.status_label, "tooltip.live.status")

        table_frame = ttk.Frame(live_paned)
        live_paned.add(table_frame, weight=6)

        _style = ttk.Style()
        _style.configure("Treeview", rowheight=22)

        columns = ("time", "source", "text", "translation")
        _gui.table = ttk.Treeview(table_frame, columns=columns, show="headings")
        _gui._bind_tree_heading(_gui.table, "time", "live.table.time")
        _gui._bind_tree_heading(_gui.table, "source", "live.table.source")
        _gui._bind_tree_heading(_gui.table, "text", "live.table.text")
        _gui._bind_tree_heading(_gui.table, "translation", "live.table.translation")
        _gui.table.column("time", width=80, minwidth=70, stretch=False)
        _gui.table.column("source", width=80, minwidth=70, stretch=False)
        _gui.table.column("text", width=500, minwidth=220)
        _gui.table.column("translation", width=400, minwidth=220)
        _gui.table.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        _gui._bind_tooltip(_gui.table, "tooltip.live.table")
        _gui.table.bind("<Control-c>", _gui._copy_selected_rows)
        _gui.table.bind("<Button-3>", _gui._show_context_menu)
        _gui.table.bind("<Control-a>", lambda e: _gui._select_all_rows())
        _gui._context_menu = tk.Menu(_gui.root, tearoff=0)
        _gui._context_menu.add_command(label="", command=_gui._copy_selected_rows)
        _gui._context_menu.add_command(label="", command=_gui._copy_text_only)
        _gui._context_menu.add_separator()
        _gui._context_menu.add_command(label="", command=_gui._select_all_rows)
        _gui._register_ui_refresher(
            lambda: _gui._context_menu.entryconfigure(
                0, label=_gui._tr("live.menu.copy_selected")
            )
        )
        _gui._register_ui_refresher(
            lambda: _gui._context_menu.entryconfigure(
                1, label=_gui._tr("live.menu.copy_text_only")
            )
        )
        _gui._register_ui_refresher(
            lambda: _gui._context_menu.entryconfigure(
                3, label=_gui._tr("live.menu.select_all")
            )
        )

        scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=_gui.table.yview)
        scroll.pack(fill=tk.Y, side=tk.RIGHT)
        _gui.table.configure(yscrollcommand=scroll.set)
        _gui.table.tag_configure("dropped", foreground="red")
        _gui.table.tag_configure("continuation", foreground="#666666")

        _gui.root.after(500, _gui._poll_stats)
