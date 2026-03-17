"""Tkinter GUI entry point for VoxFusion — live capture and file transcription."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import textwrap
import threading
import tkinter as tk
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, scrolledtext, ttk

from voxfusion.asr_catalog import (
    DEFAULT_LANGUAGE_CODE,
    get_language_code,
    get_language_label,
    get_model_info,
    list_languages_for_model,
    list_model_ids,
    normalize_language_for_model,
)
from voxfusion.gui.helpers import (
    build_file_workflow_status,
    configure_gui_logging,
    default_transcript_path,
    load_gui_settings,
    save_gui_settings,
)
from voxfusion.gui.model_summary import ModelSummaryCard
from voxfusion.gui.runtime import (
    CaptureOptions,
    CaptureWorker,
    DeviceOption,
    FileTranscribeWorker,
    LLMWorker,
    RecordingOptions,
    RecordingWorker,
    TextRedirector,
    resolve_preferred_device_index,
)
from voxfusion.gui.theme import configure_gui_theme
from voxfusion.llm.client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    fetch_models,
)
from voxfusion.llm.prompts import BUILTIN_PROMPTS
from voxfusion.media.extractor import NEEDS_EXTRACTION_EXTENSIONS
from voxfusion.models.translation import TranslatedSegment
from voxfusion.recording import RecordingStats

ASR_MODEL_CHOICES: tuple[str, ...] = list_model_ids()
GUI_DEFAULT_LANGUAGE = DEFAULT_LANGUAGE_CODE

# File dialog filter for supported media files
_AUDIO_EXTENSIONS = " ".join(
    f"*{ext}" for ext in sorted(
        {".wav", ".flac", ".ogg", ".aiff", ".au", ".w64"} | NEEDS_EXTRACTION_EXTENSIONS
    )
)
MEDIA_FILETYPES = [
    ("All supported media", _AUDIO_EXTENSIONS),
    ("Video files", " ".join(f"*{e}" for e in sorted({
        ".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".ts",
    }))),
    ("Audio files", " ".join(f"*{e}" for e in sorted({
        ".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".opus",
    }))),
    ("All files", "*.*"),
]

_build_file_workflow_status = build_file_workflow_status
_default_transcript_path = default_transcript_path
_load_gui_settings = load_gui_settings
_save_gui_settings = save_gui_settings
_resolve_preferred_device_index = resolve_preferred_device_index


# ---------------------------------------------------------------------------
# Main GUI application
# ---------------------------------------------------------------------------

class TranscriptionGUI:
    """Main GUI application with two tabs: Live Capture and File Transcription."""

    def __init__(self, root: tk.Tk, options: CaptureOptions) -> None:
        self.root = root
        self.options = options
        self.root.title("VoxFusion")
        self.root.geometry("1200x780")
        configure_gui_theme(self.root)

        # Live tab state
        self._worker: CaptureWorker | None = None
        self._record_worker: RecordingWorker | None = None
        self._segment_count = 0
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        initial_model = get_model_info(options.model).id
        self._source_var = tk.StringVar(value=options.source)
        self._model_var = tk.StringVar(value=initial_model)
        self._language_var = tk.StringVar(
            value=self._language_label_for_code(options.language, initial_model)
        )
        self._translate_var = tk.StringVar(value=options.translate or "")
        self._device_var = tk.StringVar(value="Auto (System default)")
        self._device_options: list[DeviceOption] = []
        self._requested_device_index = options.device_index
        self._last_recorded_file: Path | None = None

        # File tab state
        self._file_worker: FileTranscribeWorker | None = None
        self._file_path_var = tk.StringVar()
        self._file_model_var = tk.StringVar(value=initial_model)
        self._file_lang_var = tk.StringVar(
            value=self._language_label_for_code(options.language, initial_model)
        )
        self._file_seg_count = 0
        self._file_segments: list[TranslatedSegment] = []
        self._last_transcript_path: Path | None = None

        # LLM summarize state
        self._llm_worker: LLMWorker | None = None
        self._llm_url_var = tk.StringVar(value=DEFAULT_BASE_URL)
        self._llm_model_var = tk.StringVar(value=DEFAULT_MODEL)
        self._llm_key_var = tk.StringVar(value="")
        self._llm_prompt_var = tk.StringVar(value="summarize")
        self._llm_custom_user_prompt = ""
        self._available_llm_models: list[str] = []
        self._llm_model_refreshing = False
        self._apply_saved_gui_settings()

        self._build_layout()
        self._install_redirection()
        configure_gui_logging()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._refresh_device_options()
        self._refresh_language_choices()
        self._refresh_file_workflow()
        self._set_live_status("Select source/model and click Start or Record Audio.")
        self.root.after(250, self._refresh_llm_models)

    # ------------------------------------------------------------------
    # Layout builders
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        """Build the top-level layout with a two-tab Notebook and resizable log pane."""
        paned = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=(4, 4))

        notebook_frame = ttk.Frame(paned)
        self._notebook = ttk.Notebook(notebook_frame)
        self._notebook.pack(fill=tk.BOTH, expand=True)
        paned.add(notebook_frame, weight=3)

        live_tab = ttk.Frame(self._notebook)
        self._notebook.add(live_tab, text="  Live Capture  ")
        self._build_live_tab(live_tab)

        file_tab = ttk.Frame(self._notebook)
        self._notebook.add(file_tab, text="  File Transcription  ")
        self._build_file_tab(file_tab)

        log_frame = ttk.Frame(paned)
        ttk.Label(log_frame, text="Logs", anchor="w").pack(fill=tk.X, padx=4)
        self.log_widget = scrolledtext.ScrolledText(
            log_frame,
            wrap=tk.WORD,
            state=tk.DISABLED,
        )
        self.log_widget.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
        paned.add(log_frame, weight=1)

    def _build_live_tab(self, parent: ttk.Frame) -> None:
        """Build the Live Capture tab contents."""
        settings_box = ttk.LabelFrame(parent, text="Capture Setup", padding=(8, 6))
        settings_box.pack(fill=tk.X, padx=8, pady=(8, 4))
        settings_box.columnconfigure(1, weight=1)
        settings_box.columnconfigure(3, weight=2)

        # Row 0: Source | Device (stretches)
        ttk.Label(settings_box, text="Source:").grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.source_combo = ttk.Combobox(
            settings_box,
            textvariable=self._source_var,
            state="readonly",
            width=14,
            values=("microphone", "system", "both"),
        )
        self.source_combo.grid(row=0, column=1, sticky="w", padx=(0, 12))
        self.source_combo.bind("<<ComboboxSelected>>", self._on_source_changed)

        ttk.Label(settings_box, text="Device:").grid(row=0, column=2, sticky="w", padx=(0, 4))
        self.device_combo = ttk.Combobox(
            settings_box,
            textvariable=self._device_var,
            state="readonly",
        )
        self.device_combo.grid(row=0, column=3, sticky="ew")

        # Row 1: Model | Language | Translate
        ttk.Label(settings_box, text="Model:").grid(
            row=1, column=0, sticky="w", padx=(0, 4), pady=(4, 0)
        )
        self.model_combo = ttk.Combobox(
            settings_box,
            textvariable=self._model_var,
            state="readonly",
            width=20,
            values=ASR_MODEL_CHOICES,
        )
        self.model_combo.grid(row=1, column=1, sticky="w", padx=(0, 12), pady=(4, 0))
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_changed)

        lang_row = ttk.Frame(settings_box)
        lang_row.grid(row=1, column=2, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Label(lang_row, text="Language:").pack(side=tk.LEFT, padx=(0, 4))
        self.language_combo = ttk.Combobox(
            lang_row,
            textvariable=self._language_var,
            state="readonly",
            width=18,
        )
        self.language_combo.pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(lang_row, text="Translate:").pack(side=tk.LEFT, padx=(0, 4))
        self.translate_entry = ttk.Entry(
            lang_row,
            textvariable=self._translate_var,
            width=8,
        )
        self.translate_entry.pack(side=tk.LEFT)

        # Row 2: Action buttons + stats (all in one row)
        btn_row = ttk.Frame(settings_box)
        btn_row.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(6, 2))

        self.start_button = ttk.Button(
            btn_row, text="Start", command=self._start_capture, style="Primary.TButton"
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 4))
        self.stop_button = ttk.Button(btn_row, text="Stop", command=self._stop_capture)
        self.stop_button.configure(state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 4))
        self.pause_button = ttk.Button(
            btn_row, text="Pause", command=self._toggle_recording_pause
        )
        self.pause_button.configure(state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=(0, 4))
        self.record_button = ttk.Button(
            btn_row,
            text="Record Audio",
            command=self._start_recording,
            style="Accent.TButton",
        )
        self.record_button.pack(side=tk.LEFT, padx=(0, 12))

        ttk.Separator(btn_row, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))

        self.clear_button = ttk.Button(btn_row, text="Clear", command=self._clear_table)
        self.clear_button.pack(side=tk.LEFT, padx=(0, 4))
        self.save_button = ttk.Button(btn_row, text="Save...", command=self._save_to_file)
        self.save_button.pack(side=tk.LEFT)

        self.queue_label = ttk.Label(btn_row, text="Queue: —  |  ASR: —  |  Dropped: 0")
        self.queue_label.pack(side=tk.RIGHT, padx=(8, 0))
        self.counter_label = ttk.Label(btn_row, text="Segments: 0")
        self.counter_label.pack(side=tk.RIGHT)

        # Hidden model summary (kept for API compatibility — not displayed)
        self._live_model_summary = ModelSummaryCard(settings_box, title="Model Overview")

        self.status_label = ttk.Label(parent, text="Ready", anchor="w")
        self.status_label.pack(fill=tk.X, padx=8, pady=(4, 2))

        table_frame = ttk.Frame(parent)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))

        _style = ttk.Style()
        _style.configure("Treeview", rowheight=22)

        columns = ("time", "source", "text", "translation")
        self.table = ttk.Treeview(table_frame, columns=columns, show="headings")
        self.table.heading("time", text="Time")
        self.table.heading("source", text="Source")
        self.table.heading("text", text="Text")
        self.table.heading("translation", text="Translation")
        self.table.column("time", width=80, minwidth=70, stretch=False)
        self.table.column("source", width=80, minwidth=70, stretch=False)
        self.table.column("text", width=500, minwidth=220)
        self.table.column("translation", width=400, minwidth=220)
        self.table.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        self.table.bind("<Control-c>", self._copy_selected_rows)
        self.table.bind("<Button-3>", self._show_context_menu)
        self.table.bind("<Control-a>", lambda e: self._select_all_rows())
        self._context_menu = tk.Menu(self.root, tearoff=0)
        self._context_menu.add_command(label="Copy selected (Ctrl+C)", command=self._copy_selected_rows)
        self._context_menu.add_command(label="Copy text only", command=self._copy_text_only)
        self._context_menu.add_separator()
        self._context_menu.add_command(label="Select all", command=self._select_all_rows)

        scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.table.yview)
        scroll.pack(fill=tk.Y, side=tk.RIGHT)
        self.table.configure(yscrollcommand=scroll.set)
        self.table.tag_configure("dropped", foreground="red")
        self.table.tag_configure("continuation", foreground="#666666")

        self.root.after(500, self._poll_stats)

    def _build_file_tab(self, parent: ttk.Frame) -> None:
        """Build the File Transcription tab contents."""
        workflow_hdr = ttk.Frame(parent)
        workflow_hdr.pack(fill=tk.X, padx=8, pady=(10, 4))
        ttk.Label(
            workflow_hdr,
            text="Workflow: Record audio -> Transcribe -> Send transcript to Open WebUI",
            style="Header.TLabel",
        ).pack(side=tk.LEFT)

        self._file_workflow_label = ttk.Label(
            parent,
            text=_build_file_workflow_status(last_recorded_file=None, transcript_ready=False),
            anchor="w",
            foreground="#555555",
        )
        self._file_workflow_label.pack(fill=tk.X, padx=8, pady=(0, 4))

        # -- File picker row --
        top = ttk.Frame(parent)
        top.pack(fill=tk.X, padx=8, pady=(0, 6))
        top.columnconfigure(0, weight=3)
        top.columnconfigure(1, weight=2)

        transcribe_box = ttk.LabelFrame(top, text="Transcription Setup", padding=12)
        transcribe_box.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        picker = ttk.Frame(transcribe_box)
        picker.pack(fill=tk.X, pady=(0, 4))

        ttk.Label(picker, text="File:").pack(side=tk.LEFT, padx=(0, 6))
        self._file_path_entry = ttk.Entry(picker, textvariable=self._file_path_var, width=70)
        self._file_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        ttk.Button(picker, text="Browse...", command=self._browse_file).pack(side=tk.LEFT)

        # -- Options row --
        opts = ttk.Frame(transcribe_box)
        opts.pack(fill=tk.X, pady=(0, 4))

        ttk.Label(opts, text="Model:").pack(side=tk.LEFT, padx=(0, 6))
        self._file_model_combo = ttk.Combobox(
            opts,
            textvariable=self._file_model_var,
            state="readonly",
            width=10,
            values=ASR_MODEL_CHOICES,
        )
        self._file_model_combo.pack(side=tk.LEFT, padx=(0, 12))
        self._file_model_combo.bind("<<ComboboxSelected>>", self._on_file_model_changed)

        ttk.Label(opts, text="Language:").pack(side=tk.LEFT, padx=(0, 6))
        self._file_lang_combo = ttk.Combobox(
            opts,
            textvariable=self._file_lang_var,
            state="readonly",
            width=18,
        )
        self._file_lang_combo.pack(side=tk.LEFT, padx=(0, 16))

        self._file_transcribe_btn = ttk.Button(
            opts, text="Transcribe", command=self._start_file_transcribe, style="Accent.TButton"
        )
        self._file_transcribe_btn.pack(side=tk.LEFT, padx=(0, 4))

        # -- Status + progress row --
        prog_row = ttk.Frame(transcribe_box)
        prog_row.pack(fill=tk.X, pady=(0, 4))

        self._file_status_label = ttk.Label(
            prog_row, text="Select a file and click Transcribe.", anchor="w"
        )
        self._file_status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._file_progress = ttk.Progressbar(
            prog_row, orient="horizontal", length=220, mode="determinate", maximum=100
        )
        self._file_progress.pack(side=tk.RIGHT)

        self._file_artifact_label = ttk.Label(
            transcribe_box,
            text="Transcript file: not created yet.",
            anchor="w",
            foreground="#555555",
        )
        self._file_artifact_label.pack(fill=tk.X, pady=(0, 4))

        self._file_model_summary = ModelSummaryCard(top, title="Selected Model")
        self._file_model_summary.grid(row=0, column=1, sticky="nsew")

        # -- Results table --
        file_table_frame = ttk.Frame(parent)
        file_table_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))

        file_cols = ("time", "speaker", "text")
        self._file_table = ttk.Treeview(
            file_table_frame, columns=file_cols, show="headings"
        )
        self._file_table.heading("time", text="Timestamp")
        self._file_table.heading("speaker", text="Speaker")
        self._file_table.heading("text", text="Text")
        self._file_table.column("time", width=90, minwidth=70, stretch=False)
        self._file_table.column("speaker", width=110, minwidth=80, stretch=False)
        self._file_table.column("text", width=800, minwidth=300)
        self._file_table.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        self._file_table.bind("<Control-c>", self._file_copy_selected)
        self._file_table.bind("<Control-a>", lambda _e: self._file_table.selection_set(
            self._file_table.get_children()
        ))

        file_scroll = ttk.Scrollbar(
            file_table_frame, orient=tk.VERTICAL, command=self._file_table.yview
        )
        file_scroll.pack(fill=tk.Y, side=tk.RIGHT)
        self._file_table.configure(yscrollcommand=file_scroll.set)

        # -- File controls --
        file_ctrl = ttk.Frame(parent)
        file_ctrl.pack(fill=tk.X, padx=8, pady=(0, 4))

        ttk.Button(file_ctrl, text="Clear", command=self._clear_file_table).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(file_ctrl, text="Save...", command=self._save_file_result).pack(
            side=tk.LEFT
        )
        self._file_seg_counter_label = ttk.Label(file_ctrl, text="Segments: 0")
        self._file_seg_counter_label.pack(side=tk.RIGHT)

        # -- LLM processing panel --
        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, padx=8, pady=(4, 0))

        llm_box = ttk.LabelFrame(parent, text="Transcript Processing", padding=12)
        llm_box.pack(fill=tk.X, padx=8, pady=(4, 6))

        llm_hdr = ttk.Frame(llm_box)
        llm_hdr.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(llm_hdr, text="Process Transcript with Open WebUI", font=("", 9, "bold")).pack(
            side=tk.LEFT
        )

        llm_cfg = ttk.Frame(llm_box)
        llm_cfg.pack(fill=tk.X, pady=(0, 2))

        ttk.Label(llm_cfg, text="URL:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(llm_cfg, textvariable=self._llm_url_var, width=26).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        ttk.Button(llm_cfg, text="Refresh Models", command=self._refresh_llm_models).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        ttk.Label(llm_cfg, text="Model:").pack(side=tk.LEFT, padx=(0, 4))
        self._llm_model_combo = ttk.Combobox(
            llm_cfg,
            textvariable=self._llm_model_var,
            width=24,
        )
        self._llm_model_combo.pack(
            side=tk.LEFT, padx=(0, 10)
        )
        ttk.Label(llm_cfg, text="API Key:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(llm_cfg, textvariable=self._llm_key_var, width=14, show="*").pack(
            side=tk.LEFT, padx=(0, 10)
        )
        ttk.Label(llm_cfg, text="Prompt:").pack(side=tk.LEFT, padx=(0, 4))
        self._llm_prompt_combo = ttk.Combobox(
            llm_cfg,
            textvariable=self._llm_prompt_var,
            state="readonly",
            width=14,
            values=tuple(BUILTIN_PROMPTS.keys()),
        )
        self._llm_prompt_combo.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(llm_cfg, text="Prompt...", command=self._open_prompt_editor).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        self._llm_summarize_btn = ttk.Button(
            llm_cfg, text="Send to LLM", command=self._start_llm_summarize, style="Accent.TButton"
        )
        self._llm_summarize_btn.pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(llm_cfg, text="Copy", command=self._copy_llm_output).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(llm_cfg, text="Clear", command=self._clear_llm_output).pack(side=tk.LEFT)

        self._llm_status_label = ttk.Label(llm_cfg, text="", anchor="w", foreground="#555555")
        self._llm_status_label.pack(side=tk.LEFT, padx=(12, 0))

        llm_out_frame = ttk.Frame(llm_box)
        llm_out_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 0))

        self._llm_output = scrolledtext.ScrolledText(
            llm_out_frame,
            height=10,
            wrap=tk.WORD,
            state=tk.DISABLED,
        )
        self._llm_output.pack(fill=tk.BOTH, expand=True)
        self._refresh_file_workflow()

    # ------------------------------------------------------------------
    # Live capture methods
    # ------------------------------------------------------------------

    def _install_redirection(self) -> None:
        redirector = TextRedirector(self.log_widget)
        sys.stdout = redirector
        sys.stderr = redirector

    def _restore_redirection(self) -> None:
        sys.stdout = self._stdout
        sys.stderr = self._stderr

    def _start_capture(self) -> None:
        if self._worker is not None or self._record_worker is not None:
            return
        model_info = get_model_info(self._model_var.get() or "small")
        if not model_info.supports_live_capture:
            self._set_live_status(
                f"{model_info.name} currently supports file transcription only. "
                "Use Record Audio + File Transcription."
            )
            return

        options = CaptureOptions(
            source=self._source_var.get() or "microphone",
            model=model_info.id,
            language=self._language_code_for_label(
                self._language_var.get(),
                self._model_var.get() or "small",
            ),
            translate=(self._translate_var.get().strip() or None),
            device_index=None,
        )
        selected_label = self._device_var.get().strip()
        options = CaptureOptions(
            source=options.source,
            model=options.model,
            language=options.language,
            translate=options.translate,
            device_index=_resolve_preferred_device_index(
                self._device_options,
                selected_label,
                options.source,
            ),
        )

        self._set_live_controls_enabled(False)
        self.stop_button.configure(state=tk.NORMAL)
        self.pause_button.configure(state=tk.DISABLED)
        self._set_live_status("Starting...")
        self._worker = CaptureWorker(
            options=options,
            on_status=self._schedule_live_status,
            on_segment=self._schedule_segment,
            on_error=self._schedule_error,
            on_finished=self._schedule_finished,
            on_drop=self._schedule_drop,
        )
        self._worker.start()

    def _start_recording(self) -> None:
        if self._worker is not None or self._record_worker is not None:
            return

        default_name = (
            f"recording_{self._source_var.get() or 'microphone'}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        )
        path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            initialfile=default_name,
            filetypes=[
                ("WAV audio", "*.wav"),
                ("All files", "*.*"),
            ],
            title="Save recorded audio",
        )
        if not path:
            return

        selected_label = self._device_var.get().strip()
        options = RecordingOptions(
            source=self._source_var.get() or "microphone",
            device_index=_resolve_preferred_device_index(
                self._device_options,
                selected_label,
                self._source_var.get() or "microphone",
            ),
            output_path=Path(path),
        )

        self._set_live_controls_enabled(False)
        self.stop_button.configure(state=tk.NORMAL)
        self.pause_button.configure(state=tk.NORMAL)
        self.pause_button.configure(text="Pause")
        self.queue_label.configure(text="Queue: raw recording in progress")
        self._set_live_status(f"Recording audio to {options.output_path.name}...")
        self._record_worker = RecordingWorker(
            options=options,
            on_status=self._schedule_live_status,
            on_error=self._schedule_error,
            on_finished=self._schedule_recording_finished,
        )
        self._record_worker.start()

    def _stop_capture(self) -> None:
        if self._worker is not None:
            self._set_live_status("Stopping...")
            self._worker.stop()
            self.stop_button.configure(state=tk.DISABLED)
        elif self._record_worker is not None:
            self._set_live_status("Stopping recording...")
            self._record_worker.stop()
            self.stop_button.configure(state=tk.DISABLED)
            self.pause_button.configure(state=tk.DISABLED)

    def _toggle_recording_pause(self) -> None:
        if self._record_worker is None:
            return
        paused = self._record_worker.toggle_pause()
        self.pause_button.configure(text=("Resume" if paused else "Pause"))
        self.queue_label.configure(
            text=("Queue: recording paused" if paused else "Queue: raw recording in progress")
        )

    def _on_close(self) -> None:
        self._stop_capture()
        self._persist_gui_settings()
        self._restore_redirection()
        self.root.destroy()

    def _schedule_live_status(self, status: str) -> None:
        with suppress(tk.TclError, RuntimeError):
            self.root.after(0, self._set_live_status, status)

    def _schedule_segment(self, time_str: str, speaker: str, text: str, translation: str | None) -> None:
        with suppress(tk.TclError, RuntimeError):
            self.root.after(0, self._add_segment, time_str, speaker, text, translation)

    def _schedule_error(self, message: str) -> None:
        with suppress(tk.TclError, RuntimeError):
            self.root.after(0, self._show_error, message)

    def _schedule_finished(self) -> None:
        with suppress(tk.TclError, RuntimeError):
            self.root.after(0, self._on_worker_finished)

    def _schedule_recording_finished(self, stats: RecordingStats | None) -> None:
        with suppress(tk.TclError, RuntimeError):
            self.root.after(0, self._on_recording_finished, stats)

    def _set_live_status(self, status: str) -> None:
        self.status_label.configure(text=status)

    def _show_error(self, message: str) -> None:
        self._set_live_status(f"Error: {message}")
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._append_log_line(f"{timestamp} | ERROR | {message}\n")

    def _add_segment(self, time_str: str, speaker: str, text: str, translation: str | None) -> None:
        self._segment_count += 1
        self.counter_label.configure(text=f"Segments: {self._segment_count}")
        source_label = "mic" if "LOCAL" in speaker else "system" if "REMOTE" in speaker else speaker
        text_lines = textwrap.wrap(text, width=70) if text else [""]
        trans_lines = textwrap.wrap(translation or "", width=57) if translation else [""]
        n = max(len(text_lines), len(trans_lines))
        while len(text_lines) < n:
            text_lines.append("")
        while len(trans_lines) < n:
            trans_lines.append("")
        for i, (tl, tr) in enumerate(zip(text_lines, trans_lines)):
            tv = time_str if i == 0 else ""
            sv = source_label if i == 0 else ""
            tags: tuple[str, ...] = () if i == 0 else ("continuation",)
            self.table.insert("", tk.END, values=(tv, sv, tl, tr), tags=tags)
        self.table.yview_moveto(1.0)

    def _add_dropped_row(self, time_str: str, source: str) -> None:
        src_label = "mic" if source == "microphone" else "sys" if source == "system" else source
        self.table.insert(
            "", tk.END,
            values=(time_str, src_label, "speech may have been lost", ""),
            tags=("dropped",),
        )
        self.table.yview_moveto(1.0)

    def _schedule_drop(self, time_str: str, source: str) -> None:
        with suppress(tk.TclError, RuntimeError):
            self.root.after(0, self._add_dropped_row, time_str, source)

    def _poll_stats(self) -> None:
        if self._worker is not None:
            stats = self._worker.get_stats()
            if stats is not None:
                q = stats["preprocess_q"] + stats["asr_q"]
                in_asr = stats["in_asr"]
                dropped = stats["dropped"]
                self.queue_label.configure(
                    text=f"Queue: {q}  |  ASR: {in_asr}  |  Dropped: {dropped}"
                )
        with suppress(tk.TclError, RuntimeError):
            self.root.after(500, self._poll_stats)

    def _show_context_menu(self, event: object) -> None:
        row = self.table.identify_row(getattr(event, "y", 0))
        if row:
            if row not in self.table.selection():
                self.table.selection_set(row)
        try:
            self._context_menu.tk_popup(
                getattr(event, "x_root", 0),
                getattr(event, "y_root", 0),
            )
        finally:
            self._context_menu.grab_release()

    def _copy_selected_rows(self, _event: object | None = None) -> str:
        selected = set(self.table.selection())
        if not selected:
            return "break"
        lines = self._format_items(
            item for item in self.table.get_children() if item in selected
        )
        self.root.clipboard_clear()
        self.root.clipboard_append("\n".join(lines))
        return "break"

    def _copy_text_only(self, _event: object | None = None) -> None:
        selected = set(self.table.selection())
        if not selected:
            return
        parts: list[str] = []
        cur: list[str] = []
        for item in self.table.get_children():
            if item not in selected:
                if cur:
                    parts.append(" ".join(cur))
                    cur = []
                continue
            tags = self.table.item(item, "tags")
            text_val = self.table.item(item, "values")[2]
            if "continuation" in tags:
                if text_val:
                    cur.append(text_val)
            else:
                if cur:
                    parts.append(" ".join(cur))
                cur = [text_val] if text_val else []
        if cur:
            parts.append(" ".join(cur))
        self.root.clipboard_clear()
        self.root.clipboard_append("\n".join(parts))

    def _format_items(self, items: object) -> list[str]:
        lines: list[str] = []
        cur_time = cur_src = ""
        cur_text: list[str] = []
        cur_trans: list[str] = []

        def _flush() -> None:
            if not cur_time and not cur_text:
                return
            line = f"[{cur_time}] [{cur_src}] {' '.join(cur_text)}"
            t = " ".join(p for p in cur_trans if p)
            if t:
                line += f" | {t}"
            lines.append(line)
            cur_text.clear()
            cur_trans.clear()

        for item in items:  # type: ignore[union-attr]
            values = self.table.item(item, "values")
            tags = self.table.item(item, "tags")
            time_val = values[0] if values else ""
            src_val = values[1] if len(values) > 1 else ""
            text_val = values[2] if len(values) > 2 else ""
            trans_val = values[3] if len(values) > 3 else ""
            if "dropped" in tags:
                _flush()
                lines.append(f"[{time_val}] {text_val}")
            elif "continuation" in tags:
                if text_val:
                    cur_text.append(text_val)
                if trans_val:
                    cur_trans.append(trans_val)
            else:
                _flush()
                cur_time, cur_src = time_val, src_val
                if text_val:
                    cur_text.append(text_val)
                if trans_val:
                    cur_trans.append(trans_val)
        _flush()
        return lines

    def _select_all_rows(self) -> None:
        self.table.selection_set(self.table.get_children())

    def _clear_table(self) -> None:
        for item in self.table.get_children():
            self.table.delete(item)
        self._segment_count = 0
        self.counter_label.configure(text="Segments: 0")

    def _save_to_file(self) -> None:
        children = self.table.get_children()
        if not children:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[
                ("Text file", "*.txt"),
                ("CSV file", "*.csv"),
                ("All files", "*.*"),
            ],
            title="Save live transcription",
        )
        if not path:
            return
        lines = self._format_items(children)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
            self._set_live_status(f"Saved {len(lines)} segments -> {path}")
        except OSError as exc:
            self._set_live_status(f"Save failed: {exc}")

    def _on_worker_finished(self) -> None:
        self._worker = None
        self.queue_label.configure(text="Queue: —  |  ASR: —  |  Dropped: 0")
        self._set_live_controls_enabled(True)
        self.stop_button.configure(state=tk.DISABLED)
        self.pause_button.configure(state=tk.DISABLED)
        self.pause_button.configure(text="Pause")

    def _on_recording_finished(self, stats: RecordingStats | None) -> None:
        self._record_worker = None
        self._set_live_controls_enabled(True)
        self.stop_button.configure(state=tk.DISABLED)
        self.pause_button.configure(state=tk.DISABLED)
        self.pause_button.configure(text="Pause")
        self.queue_label.configure(text="Queue: —  |  ASR: —  |  Dropped: 0")
        if stats is None:
            self._refresh_file_workflow()
            return
        self._last_recorded_file = stats.output_path
        self._persist_gui_settings()
        self._set_live_status(f"Recorded {stats.duration_s:.1f}s -> {stats.output_path.name}")
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._append_log_line(
            f"{timestamp} | RECORDED | {stats.output_path} | "
            f"{stats.duration_s:.1f}s | {stats.sample_rate} Hz\n"
        )
        self._load_recorded_file_into_flow(stats.output_path, switch_tab=True)

    def _set_live_controls_enabled(self, enabled: bool) -> None:
        combo_state = "readonly" if enabled else "disabled"
        self.start_button.configure(state=(tk.NORMAL if enabled else tk.DISABLED))
        self.record_button.configure(state=(tk.NORMAL if enabled else tk.DISABLED))
        self.source_combo.configure(state=combo_state)
        self.model_combo.configure(state=combo_state)
        self.language_combo.configure(state=combo_state)
        self.translate_entry.configure(state=(tk.NORMAL if enabled else tk.DISABLED))
        self.device_combo.configure(state=combo_state)

    def _refresh_language_choices(self) -> None:
        live_model = get_model_info(self._model_var.get()).id
        live_model_info = get_model_info(live_model)
        live_values = [language.label for language in list_languages_for_model(live_model)]
        current_live = self._language_code_for_label(self._language_var.get(), live_model)
        self.language_combo.configure(values=live_values)
        self._language_var.set(self._language_label_for_code(current_live, live_model))
        self._live_model_summary.set_model(live_model)
        if live_model_info.supports_live_capture:
            if self._worker is None and self._record_worker is None:
                self.start_button.configure(state=tk.NORMAL)
        elif self._worker is None and self._record_worker is None:
            self.start_button.configure(state=tk.DISABLED)
            self._set_live_status(
                f"{live_model_info.name} is available for file transcription only."
            )

        file_model = get_model_info(self._file_model_var.get()).id
        file_values = [language.label for language in list_languages_for_model(file_model)]
        current_file = self._language_code_for_label(self._file_lang_var.get(), file_model)
        self._file_lang_combo.configure(values=file_values)
        self._file_lang_var.set(self._language_label_for_code(current_file, file_model))
        self._file_model_summary.set_model(file_model)

    def _on_model_changed(self, _event: object | None = None) -> None:
        self._model_var.set(get_model_info(self._model_var.get()).id)
        self._refresh_language_choices()

    def _on_file_model_changed(self, _event: object | None = None) -> None:
        self._file_model_var.set(get_model_info(self._file_model_var.get()).id)
        self._refresh_language_choices()

    def _append_log_line(self, text: str) -> None:
        self.log_widget.configure(state=tk.NORMAL)
        self.log_widget.insert(tk.END, text)
        self.log_widget.see(tk.END)
        self.log_widget.configure(state=tk.DISABLED)

    def _apply_saved_gui_settings(self) -> None:
        settings = _load_gui_settings()
        self._llm_url_var.set(settings.get("llm_url", DEFAULT_BASE_URL))
        self._llm_model_var.set(settings.get("llm_model", DEFAULT_MODEL))
        self._llm_key_var.set(settings.get("llm_api_key", ""))
        self._llm_prompt_var.set(settings.get("llm_prompt", "summarize"))
        self._llm_custom_user_prompt = settings.get("llm_custom_user_prompt", "")

        last_rec = settings.get("last_recorded_file", "")
        if last_rec:
            p = Path(last_rec)
            if p.exists():
                self._last_recorded_file = p
                self._file_path_var.set(str(p))

        last_tx = settings.get("last_transcript_path", "")
        if last_tx:
            p = Path(last_tx)
            if p.exists():
                self._last_transcript_path = p

    def _persist_gui_settings(self) -> None:
        _save_gui_settings(
            {
                "llm_url": self._llm_url_var.get().strip(),
                "llm_model": self._llm_model_var.get().strip(),
                "llm_api_key": self._llm_key_var.get(),
                "llm_prompt": self._llm_prompt_var.get().strip() or "summarize",
                "llm_custom_user_prompt": self._llm_custom_user_prompt,
                "last_recorded_file": str(self._last_recorded_file) if self._last_recorded_file else "",
                "last_transcript_path": str(self._last_transcript_path) if self._last_transcript_path else "",
            }
        )

    def _load_recorded_file_into_flow(self, path: Path, *, switch_tab: bool) -> None:
        self._last_recorded_file = path
        self._file_path_var.set(str(path))
        self._clear_file_table()
        self._last_transcript_path = None
        self._file_status_label.configure(
            text=f"Step 2: Ready to transcribe recorded audio: {path.name}"
        )
        if switch_tab:
            self._notebook.select(1)
        self._refresh_file_workflow()

    def _refresh_file_workflow(self) -> None:
        transcript_ready = self._file_seg_count > 0
        self._file_workflow_label.configure(
            text=_build_file_workflow_status(
                last_recorded_file=self._last_recorded_file,
                transcript_ready=transcript_ready,
            )
        )
        llm_enabled = transcript_ready and self._llm_worker is None and self._file_worker is None
        self._llm_summarize_btn.configure(state=(tk.NORMAL if llm_enabled else tk.DISABLED))
        if self._last_transcript_path is not None and self._last_transcript_path.exists():
            self._file_artifact_label.configure(text=f"Transcript file: {self._last_transcript_path}")
        else:
            self._file_artifact_label.configure(text="Transcript file: not created yet.")

    def _refresh_llm_models(self) -> None:
        if self._llm_model_refreshing:
            return
        self._llm_model_refreshing = True
        self._llm_status_label.configure(text="Loading models from Open WebUI...")
        self._persist_gui_settings()

        def _run() -> None:
            try:
                models = asyncio.run(
                    fetch_models(
                        base_url=self._llm_url_var.get().strip() or DEFAULT_BASE_URL,
                        api_key=self._llm_key_var.get().strip(),
                    )
                )
                self.root.after(0, self._on_llm_models_loaded, models, None)
            except Exception as exc:  # pragma: no cover
                self.root.after(0, self._on_llm_models_loaded, [], str(exc))

        threading.Thread(target=_run, daemon=True).start()

    def _on_llm_models_loaded(self, models: list[str], error: str | None) -> None:
        self._llm_model_refreshing = False
        if error:
            self._llm_status_label.configure(text=f"Model load failed: {error}")
            return
        self._available_llm_models = models
        self._llm_model_combo.configure(values=models)
        if models and self._llm_model_var.get().strip() not in models:
            self._llm_model_var.set(models[0])
        self._persist_gui_settings()
        self._llm_status_label.configure(text=f"Loaded {len(models)} model(s) from Open WebUI.")

    def _open_prompt_editor(self) -> None:
        prompt_name = self._llm_prompt_var.get().strip() or "summarize"
        prompt_def = BUILTIN_PROMPTS[prompt_name]
        dialog = tk.Toplevel(self.root)
        dialog.title("Prompt Template")
        dialog.geometry("900x700")

        ttk.Label(dialog, text=f"Prompt template: {prompt_name}", font=("", 10, "bold")).pack(
            anchor="w", padx=10, pady=(10, 4)
        )
        ttk.Label(dialog, text="System prompt", anchor="w").pack(fill=tk.X, padx=10)
        system_text = scrolledtext.ScrolledText(dialog, height=8, wrap=tk.WORD)
        system_text.pack(fill=tk.BOTH, expand=False, padx=10, pady=(0, 8))
        system_text.insert("1.0", prompt_def["system"])
        system_text.configure(state=tk.DISABLED)

        ttk.Label(
            dialog,
            text="User prompt template (must include {transcript})",
            anchor="w",
        ).pack(fill=tk.X, padx=10)
        user_text = scrolledtext.ScrolledText(dialog, height=18, wrap=tk.WORD)
        user_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))
        user_text.insert("1.0", self._llm_custom_user_prompt or prompt_def["user"])

        button_row = ttk.Frame(dialog)
        button_row.pack(fill=tk.X, padx=10, pady=(0, 10))

        def _reset() -> None:
            user_text.delete("1.0", tk.END)
            user_text.insert("1.0", prompt_def["user"])

        def _save() -> None:
            prompt_text = user_text.get("1.0", tk.END).strip()
            if "{transcript}" not in prompt_text:
                self._llm_status_label.configure(
                    text="Prompt save failed: template must include {transcript}."
                )
                return
            self._llm_custom_user_prompt = "" if prompt_text == prompt_def["user"] else prompt_text
            self._persist_gui_settings()
            self._llm_status_label.configure(text=f"Prompt saved for template '{prompt_name}'.")
            dialog.destroy()

        ttk.Button(button_row, text="Reset to Default", command=_reset).pack(side=tk.LEFT)
        ttk.Button(button_row, text="Save Prompt", command=_save).pack(side=tk.RIGHT)
        ttk.Button(button_row, text="Close", command=dialog.destroy).pack(side=tk.RIGHT, padx=(0, 6))

    def _on_source_changed(self, _event: object | None = None) -> None:
        self._refresh_device_options()

    def _refresh_device_options(self) -> None:
        source = self._source_var.get() or "microphone"
        is_loopback = source == "system"

        options: list[DeviceOption] = [DeviceOption("Auto (System default)", None)]

        try:
            import sounddevice as sd

            hostapis = list(sd.query_hostapis())
            hostapi_names = {
                i: str(api.get("name", f"HostAPI {i}"))
                for i, api in enumerate(hostapis)
            }
            wasapi_host_ids = {
                i for i, name in hostapi_names.items()
                if "wasapi" in name.lower()
            }
            devices = list(sd.query_devices())
            ranked: list[tuple[int, int, str, str]] = []
            for idx, dev in enumerate(devices):
                hostapi_idx = int(dev.get("hostapi", -1))
                hostapi_name = hostapi_names.get(hostapi_idx, f"HostAPI {hostapi_idx}")
                hostapi_key = hostapi_name.lower()
                if is_loopback:
                    # System audio: only WASAPI output devices
                    if hostapi_idx not in wasapi_host_ids:
                        continue
                    channels = int(dev.get("max_output_channels", 0))
                    if channels <= 0:
                        continue
                else:
                    # Microphone: only WASAPI input devices (matches Windows Sound Settings)
                    if "wasapi" not in hostapi_key:
                        continue
                    channels = int(dev.get("max_input_channels", 0))
                    if channels <= 0:
                        continue
                if is_loopback:
                    priority = 0
                else:
                    priority = 0
                name = str(dev.get("name", f"Device {idx}"))
                label = f"{name} [{hostapi_name} #{idx}]"
                ranked.append((priority, idx, label, hostapi_name))

            ranked.sort(key=lambda item: (item[0], item[1]))
            for _priority, idx, label, _hostapi_name in ranked:
                options.append(DeviceOption(label=label, index=idx))
        except Exception:
            pass

        self._device_options = options
        labels = [option.label for option in options]
        self.device_combo.configure(values=labels)

        selected_label = options[0].label
        if self._requested_device_index is not None:
            requested = next(
                (option for option in options if option.index == self._requested_device_index),
                None,
            )
            if requested is not None:
                selected_label = requested.label
        elif self._device_var.get() in labels:
            selected_label = self._device_var.get()
        self._device_var.set(selected_label)

    # ------------------------------------------------------------------
    # File transcription methods
    # ------------------------------------------------------------------

    def _browse_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select audio or video file",
            filetypes=MEDIA_FILETYPES,
        )
        if path:
            self._file_path_var.set(path)
            self._last_transcript_path = None
            self._file_status_label.configure(text=f"Ready: {Path(path).name}")
            self._file_progress["value"] = 0
            self._clear_llm_output()
            self._refresh_file_workflow()

    def _start_file_transcribe(self) -> None:
        raw_path = self._file_path_var.get().strip()
        if not raw_path:
            self._file_status_label.configure(text="Error: no file selected.")
            self._refresh_file_workflow()
            return

        file_path = Path(raw_path)
        if not file_path.exists():
            self._file_status_label.configure(text=f"Error: file not found: {file_path.name}")
            self._refresh_file_workflow()
            return

        if self._file_worker is not None:
            return  # already running

        model = get_model_info(self._file_model_var.get() or "small").id
        language = self._language_code_for_label(self._file_lang_var.get(), model)

        self._file_transcribe_btn.configure(state=tk.DISABLED)
        self._file_model_combo.configure(state="disabled")
        self._file_lang_combo.configure(state="disabled")
        self._file_progress["value"] = 0
        self._last_transcript_path = None
        self._file_status_label.configure(text=f"Step 2: Transcribing {file_path.name}...")
        self._refresh_file_workflow()

        self._file_worker = FileTranscribeWorker(
            file_path=file_path,
            model=model,
            language=language,
            on_status=self._schedule_file_status,
            on_segments=self._schedule_file_segments,
            on_error=self._schedule_file_error,
            on_finished=self._schedule_file_finished,
        )
        self._file_worker.start()

    def _schedule_file_status(self, msg: str, progress: float) -> None:
        with suppress(tk.TclError, RuntimeError):
            self.root.after(0, self._update_file_status, msg, progress)

    def _schedule_file_segments(self, segments: list[TranslatedSegment]) -> None:
        with suppress(tk.TclError, RuntimeError):
            self.root.after(0, self._add_file_segments, segments)

    def _schedule_file_error(self, message: str) -> None:
        with suppress(tk.TclError, RuntimeError):
            self.root.after(0, self._show_file_error, message)

    def _schedule_file_finished(self) -> None:
        with suppress(tk.TclError, RuntimeError):
            self.root.after(0, self._on_file_worker_finished)

    def _update_file_status(self, msg: str, progress: float) -> None:
        self._file_status_label.configure(text=msg)
        self._file_progress["value"] = int(progress * 100)
        self._refresh_file_workflow()

    def _show_file_error(self, message: str) -> None:
        self._file_status_label.configure(text=f"Error: {message}")
        self._file_progress["value"] = 0
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._append_log_line(f"{timestamp} | FILE ERROR | {message}\n")
        self._refresh_file_workflow()

    def _add_file_segments(self, segments: list[TranslatedSegment]) -> None:
        for seg in segments:
            ts = seg.diarized.segment
            speaker = seg.diarized.speaker_id
            # Format as HH:MM:SS
            total_secs = int(ts.start_time)
            h, remainder = divmod(total_secs, 3600)
            m, s = divmod(remainder, 60)
            time_str = f"{h:02d}:{m:02d}:{s:02d}"
            self._file_table.insert(
                "", tk.END, values=(time_str, speaker, ts.text.strip())
            )
            self._file_seg_count += 1

        self._file_segments.extend(segments)
        self._file_seg_counter_label.configure(text=f"Segments: {self._file_seg_count}")
        self._file_table.yview_moveto(1.0)
        self._refresh_file_workflow()

    def _on_file_worker_finished(self) -> None:
        self._file_worker = None
        self._file_transcribe_btn.configure(state=tk.NORMAL)
        self._file_model_combo.configure(state="readonly")
        self._file_lang_combo.configure(state="readonly")
        if self._file_seg_count > 0:
            self._last_transcript_path = self._auto_save_transcript()
            self._persist_gui_settings()
            self._file_status_label.configure(
                text=(
                    f"Step 3: Transcript ready. Saved to {self._last_transcript_path.name}. "
                    "Send it to Open WebUI or use Save... for another format."
                )
            )
        else:
            self._file_status_label.configure(
                text=(
                    "Transcription finished, but no speech segments were found. "
                    "Transcript file was not created."
                )
            )
        self._refresh_file_workflow()

    def _file_copy_selected(self, _event: object | None = None) -> str:
        selected = self._file_table.selection()
        if not selected:
            return "break"
        lines: list[str] = []
        for item in selected:
            vals = self._file_table.item(item, "values")
            if vals:
                lines.append(f"[{vals[0]}] [{vals[1]}] {vals[2]}")
        self.root.clipboard_clear()
        self.root.clipboard_append("\n".join(lines))
        return "break"

    def _clear_file_table(self) -> None:
        for item in self._file_table.get_children():
            self._file_table.delete(item)
        self._file_seg_count = 0
        self._file_segments = []
        self._last_transcript_path = None
        self._file_seg_counter_label.configure(text="Segments: 0")
        self._file_progress["value"] = 0
        self._file_status_label.configure(text="Cleared. Select a file and click Transcribe.")
        self._clear_llm_output()
        self._refresh_file_workflow()

    def _save_file_result(self) -> None:
        children = self._file_table.get_children()
        if not children:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[
                ("Text file", "*.txt"),
                ("SRT subtitles", "*.srt"),
                ("All files", "*.*"),
            ],
            title="Save file transcription",
        )
        if not path:
            return
        try:
            out_path = Path(path)
            if out_path.suffix.lower() == ".srt":
                self._save_as_srt(out_path)
            else:
                self._save_as_txt(out_path, children)
            self._file_status_label.configure(
                text=f"Saved {self._file_seg_count} segments -> {out_path.name}"
            )
        except OSError as exc:
            self._file_status_label.configure(text=f"Save failed: {exc}")
        self._refresh_file_workflow()

    def _save_as_txt(self, path: Path, children: tuple[str, ...]) -> None:
        lines: list[str] = []
        for item in children:
            vals = self._file_table.item(item, "values")
            if vals:
                lines.append(f"[{vals[0]}] [{vals[1]}] {vals[2]}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _auto_save_transcript(self) -> Path:
        source_path = Path(self._file_path_var.get().strip())
        transcript_path = _default_transcript_path(source_path)
        children = self._file_table.get_children()
        self._save_as_txt(transcript_path, children)
        return transcript_path

    def _save_as_srt(self, path: Path) -> None:
        lines: list[str] = []
        for i, seg in enumerate(self._file_segments, start=1):
            ts = seg.diarized.segment
            start = _secs_to_srt(ts.start_time)
            end = _secs_to_srt(ts.end_time)
            text = ts.text.strip()
            lines.append(f"{i}\n{start} --> {end}\n{text}\n")
        path.write_text("\n".join(lines), encoding="utf-8")

    # ------------------------------------------------------------------
    # LLM summarize methods
    # ------------------------------------------------------------------

    def _get_file_transcript_text(self) -> str:
        """Build a plain-text transcript from the current file table rows."""
        lines: list[str] = []
        for item in self._file_table.get_children():
            vals = self._file_table.item(item, "values")
            if vals and len(vals) >= 3 and str(vals[2]).strip():
                lines.append(f"[{vals[0]}] [{vals[1]}] {vals[2]}")
        return "\n".join(lines)

    def _start_llm_summarize(self) -> None:
        if self._llm_worker is not None:
            return
        transcript = self._get_file_transcript_text()
        if not transcript:
            self._llm_status_label.configure(
                text="No transcript available. Complete Step 2 first."
            )
            self._refresh_file_workflow()
            return

        url = self._llm_url_var.get().strip() or DEFAULT_BASE_URL
        model = self._llm_model_var.get().strip() or DEFAULT_MODEL
        api_key = self._llm_key_var.get().strip()
        prompt_name = self._llm_prompt_var.get().strip() or "summarize"
        self._persist_gui_settings()

        self._clear_llm_output()
        self._llm_summarize_btn.configure(state=tk.DISABLED)
        self._llm_status_label.configure(text=f"Sending to {model}...")

        self._llm_worker = LLMWorker(
            text=transcript,
            model=model,
            base_url=url,
            api_key=api_key,
            prompt_name=prompt_name,
            custom_user_prompt=(self._llm_custom_user_prompt or None),
            on_token=self._schedule_llm_token,
            on_error=self._schedule_llm_error,
            on_finished=self._schedule_llm_finished,
        )
        self._llm_worker.start()

    def _schedule_llm_token(self, token: str) -> None:
        with suppress(tk.TclError, RuntimeError):
            self.root.after(0, self._append_llm_token, token)

    def _schedule_llm_error(self, message: str) -> None:
        with suppress(tk.TclError, RuntimeError):
            self.root.after(0, self._show_llm_error, message)

    def _schedule_llm_finished(self) -> None:
        with suppress(tk.TclError, RuntimeError):
            self.root.after(0, self._on_llm_finished)

    def _append_llm_token(self, token: str) -> None:
        self._llm_output.configure(state=tk.NORMAL)
        self._llm_output.insert(tk.END, token)
        self._llm_output.see(tk.END)
        self._llm_output.configure(state=tk.DISABLED)

    def _show_llm_error(self, message: str) -> None:
        self._llm_status_label.configure(text=f"Error: {message[:80]}")
        self._append_llm_token(f"\n\n[ERROR] {message}\n")
        self._refresh_file_workflow()

    def _on_llm_finished(self) -> None:
        self._llm_worker = None
        self._llm_summarize_btn.configure(state=tk.NORMAL)
        current = self._llm_status_label.cget("text")
        if not current.startswith("Error"):
            self._llm_status_label.configure(text="Done. Open WebUI response received.")
        self._refresh_file_workflow()

    def _clear_llm_output(self) -> None:
        self._llm_output.configure(state=tk.NORMAL)
        self._llm_output.delete("1.0", tk.END)
        self._llm_output.configure(state=tk.DISABLED)
        self._llm_status_label.configure(text="")
        self._refresh_file_workflow()

    def _copy_llm_output(self) -> None:
        text = self._llm_output.get("1.0", tk.END).strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self._llm_status_label.configure(text="Copied to clipboard.")

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _language_label_for_code(language_code: str | None, model_id: str | None = None) -> str:
        normalized = normalize_language_for_model(model_id, language_code)
        if normalized is None and language_code is None and model_id is None:
            normalized = GUI_DEFAULT_LANGUAGE
        return get_language_label(normalized, model_id)

    @staticmethod
    def _language_code_for_label(label: str, model_id: str | None = None) -> str | None:
        return get_language_code(label, model_id)


def _secs_to_srt(secs: float) -> str:
    """Convert float seconds to SRT timestamp (HH:MM:SS,mmm)."""
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    ms = int((secs - int(secs)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run VoxFusion GUI mode.")
    parser.add_argument(
        "--source",
        choices=["microphone", "system", "both"],
        default="both",
        help="Default audio source for live capture.",
    )
    parser.add_argument(
        "--translate",
        default=None,
        help="Target translation language code (optional).",
    )
    parser.add_argument(
        "--model",
        default="small",
        help="ASR model size.",
    )
    parser.add_argument(
        "--language",
        default=GUI_DEFAULT_LANGUAGE,
        help="ASR language code (e.g. ru, en).",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Optional sounddevice device ID.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for GUI mode."""
    args = _build_parser().parse_args(argv)
    if sys.platform != "win32":
        print("Live capture requires Windows WASAPI. File transcription works on all platforms.")

    options = CaptureOptions(
        source=args.source or "both",
        model=args.model,
        language=args.language,
        translate=args.translate,
        device_index=args.device,
    )

    root = tk.Tk()
    TranscriptionGUI(root, options)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
