"""Tkinter GUI entry point for VoxFusion — live capture and file transcription."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import queue
import re
import sys
import textwrap
import threading
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import monotonic
from tkinter import filedialog, messagebox, scrolledtext, ttk

from voxfusion.asr_catalog import (
    DEFAULT_LANGUAGE_CODE,
    QUALITY_PRESET_LABELS,
    get_available_model_catalog,
    get_default_model_id,
    get_language_code,
    get_language_label,
    get_model_info,
    list_languages_for_model,
    normalize_language_for_model,
)
from voxfusion.capture.windows_audio import (
    list_windows_capture_devices,
)
from voxfusion.gui.helpers import (
    apply_proxy_settings,
    build_file_workflow_status,
    configure_gui_logging,
    default_transcript_path,
    find_ffmpeg,
    get_system_proxies,
    install_ffmpeg_local,
    load_detection_audio_chunk,
    load_gui_settings,
    models_dir,
    probe_media_metadata,
    probe_media_size,
    save_gui_settings,
)
from voxfusion.gui.i18n import (
    DEFAULT_GUI_LANGUAGE,
    SUPPORTED_GUI_LANGUAGES,
    detect_system_gui_language,
    load_gui_locale,
    normalize_gui_language,
    resolve_initial_gui_language,
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
    derive_capture_source,
)
from voxfusion.gui.theme import configure_gui_theme
from voxfusion.gui.tooltip import ToolTip, create_help_icon
from voxfusion.llm.client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    LLMModelDescriptor,
    complete,
    fetch_model_catalog,
    verify_model_ready,
)
from voxfusion.llm.prompts import BUILTIN_PROMPTS
from voxfusion.logging import get_logger
from voxfusion.media.extractor import NEEDS_EXTRACTION_EXTENSIONS
from voxfusion.models.translation import TranslatedSegment
from voxfusion.recording import RecordingStats
from voxfusion.runtime_subprocess import patch_subprocess_popen_no_window

ASR_MODEL_CHOICES: tuple[str, ...] = tuple(m.id for m in get_available_model_catalog())
GUI_DEFAULT_LANGUAGE = DEFAULT_LANGUAGE_CODE
FILE_DIARIZATION_CHOICES: tuple[str, ...] = ("auto", "none", "channel", "ml", "hybrid")
FILE_SPEAKER_PRESET_CODES: tuple[str, ...] = ("auto", "1", "2", "3", "4plus", "custom")
_LLM_PROBE_TIMEOUT_READ = 30.0
_LLM_PROBE_MESSAGES = [{"role": "user", "content": "Reply with OK and nothing else."}]
_IMPORTED_TRANSCRIPT_SPEAKER = "IMPORTED"
_LLM_MODELS_CACHE_KEY = "llm_models_cache_json"
_LLM_MODEL_CONTEXT_CACHE_KEY = "llm_model_context_cache_json"
_LLM_CONTEXT_TOKEN_ENV = "VOXFUSION_LLM_CONTEXT_TOKENS"
_LLM_DEFAULT_CONTEXT_TOKENS = 2048
_LLM_MIN_CONTEXT_TOKENS = 512
_IMPORTED_TRANSCRIPT_LINE_RE = re.compile(
    r"^\[(?P<time>\d{2}:\d{2}:\d{2})\]\s+\[(?P<speaker>[^\]]+)\]\s*(?P<text>.+?)\s*$"
)
_IMPORTED_SRT_TIME_RANGE_RE = re.compile(
    r"^(?P<start>\d{2}:\d{2}:\d{2})(?:[,.]\d{3})?\s+-->\s+(?P<end>\d{2}:\d{2}:\d{2})(?:[,.]\d{3})?$"
)
_IMPORTED_SRT_SPEAKER_RE = re.compile(r"^\[(?P<speaker>[^\]]+)\]\s*(?P<text>.+?)\s*$")
log = get_logger(__name__)

# File dialog filter for supported media files
_AUDIO_EXTENSIONS = " ".join(
    f"*{ext}"
    for ext in sorted(
        {".wav", ".flac", ".ogg", ".aiff", ".au", ".w64"} | NEEDS_EXTRACTION_EXTENSIONS
    )
)
_build_file_workflow_status = build_file_workflow_status
_default_transcript_path = default_transcript_path
_load_gui_settings = load_gui_settings
_save_gui_settings = save_gui_settings
_derive_capture_source = derive_capture_source


@dataclass
class _FileQueueItem:
    file_path: Path
    duration_s: float | None = None
    size_bytes: int | None = None
    status: str = "Queued"
    progress: float = 0.0
    result: str = ""
    output_path: Path | None = None
    metadata_generation: int = 0


# ---------------------------------------------------------------------------
# Main GUI application
# ---------------------------------------------------------------------------


class TranscriptionGUI:
    """Main GUI application with two tabs: Live Capture and File Transcription."""

    def __init__(self, root: tk.Tk, options: CaptureOptions) -> None:
        self.root = root
        self.options = options
        self._ui_language_code = detect_system_gui_language()
        self._ui_language_explicit = False
        self._ui_language_var = tk.StringVar(value="")
        self._locale = load_gui_locale(self._ui_language_code)
        self._ui_refreshers: list[object] = []
        self._tooltips: list[tuple[ToolTip, str]] = []
        self._log_mode_code = "normal"
        self._queue_metadata_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="voxfusion-gui-meta",
        )
        self._queue_metadata_async_enabled = True
        self._file_queue_generation = 0

        self.root.title(self._tr("app.title"))
        self.root.geometry("1200x780")
        configure_gui_theme(self.root)

        # Live tab state
        self._worker: CaptureWorker | None = None
        self._record_worker: RecordingWorker | None = None
        self._segment_count = 0
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        _live_default = options.model or get_default_model_id(for_live_capture=True)
        initial_model = get_model_info(_live_default).id
        self._model_var = tk.StringVar(value=initial_model)
        self._language_var = tk.StringVar(
            value=self._language_label_for_code(options.language, initial_model)
        )
        self._translate_var = tk.StringVar(value=options.translate or "")
        self._device_picker_var = tk.StringVar(value=self._tr("device.loading"))
        self._device_options: list[DeviceOption] = []
        self._requested_device_index = options.microphone_device_id or options.system_device_id
        self._device_check_vars: dict[str, tk.BooleanVar] = {}
        self._selected_microphone_id: str | int | None = options.microphone_device_id
        self._selected_system_id: str | int | None = options.system_device_id
        self._device_list_fingerprint: frozenset = frozenset()
        self._last_recorded_file: Path | None = None
        self._ffmpeg_path: Path | None = find_ffmpeg()
        self._rec_format_var = tk.StringVar(value="mp3" if self._ffmpeg_path is not None else "wav")

        # File tab state
        self._file_worker: FileTranscribeWorker | None = None
        self._file_path_var = tk.StringVar()
        _file_default = get_default_model_id(for_live_capture=False)
        self._file_model_var = tk.StringVar(value=_file_default)
        self._file_lang_var = tk.StringVar(
            value=self._language_label_for_code(options.language, _file_default)
        )
        self._file_quality_var = tk.StringVar(value="Balanced")
        self._file_quality_display_var = tk.StringVar(value="")
        self._file_diarization_var = tk.StringVar(value="auto")
        self._file_speaker_preset_var = tk.StringVar(value="auto")
        self._file_speaker_preset_display_var = tk.StringVar(value="")
        self._file_min_speakers_var = tk.StringVar(value="")
        self._file_max_speakers_var = tk.StringVar(value="")
        self._file_queue_items: dict[str, _FileQueueItem] = {}
        self._file_queue_lookup: dict[str, str] = {}
        self._file_queue_serial = 0
        self._file_active_queue_id: str | None = None
        self._file_batch_cancel_requested = False
        self._file_active_error_message: str | None = None
        self._file_detect_worker_thread: threading.Thread | None = None
        self._file_seg_count = 0
        self._file_segments: list[TranslatedSegment] = []
        self._last_transcript_path: Path | None = None
        self._file_start_time: float | None = None
        self._file_current_progress: float = 0.0
        # (timestamp, progress) samples for velocity-based ETA
        self._file_progress_samples: list[tuple[float, float]] = []

        # Proxy / network settings state
        self._proxy_use_system_var = tk.BooleanVar(value=True)
        self._proxy_http_var = tk.StringVar(value="")
        self._proxy_https_var = tk.StringVar(value="")
        self._proxy_no_var = tk.StringVar(value="")
        self._proxy_ca_var = tk.StringVar(value="")
        self._hf_token_var = tk.StringVar(value="")
        self._log_mode_var = tk.StringVar(value="")

        # LLM summarize state
        self._llm_worker: LLMWorker | None = None
        self._llm_url_var = tk.StringVar(value=DEFAULT_BASE_URL)
        self._llm_model_var = tk.StringVar(value=DEFAULT_MODEL)
        self._llm_key_var = tk.StringVar(value="")
        self._llm_prompt_var = tk.StringVar(value="summarize")
        self._llm_context_var = tk.StringVar(value="")
        self._llm_custom_user_prompt = ""
        self._available_llm_models: list[str] = []
        self._cached_llm_models: list[str] = []
        self._llm_model_contexts: dict[str, int] = {}
        self._cached_llm_model_contexts: dict[str, int] = {}
        self._llm_model_refreshing = False
        self._llm_probe_running = False
        self._llm_preflight_running = False
        self._llm_last_error_message: str | None = None
        self._llm_model_var.trace_add("write", lambda *_: self._refresh_llm_context_hint())
        self._llm_context_var.trace_add("write", lambda *_: self._refresh_llm_context_hint())
        self._apply_saved_gui_settings()

        self._build_layout()
        self._apply_localized_ui()
        self._install_redirection()
        self._apply_gui_log_mode()

        # Refresh detect-button state when file selection changes
        self._file_path_var.trace_add(
            "write",
            lambda *_: self._refresh_file_diarization_controls(),
        )

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._refresh_device_options()
        self._refresh_language_choices()
        self._refresh_file_workflow()
        self._set_live_status(self._tr("live.status.default"))
        self.root.after(250, self._refresh_llm_models)
        self.root.after(5000, self._poll_device_changes)

    # ------------------------------------------------------------------
    # Localization helpers
    # ------------------------------------------------------------------

    def _tr(self, key: str, **kwargs: object) -> str:
        locale = getattr(self, "_locale", None) or load_gui_locale(DEFAULT_GUI_LANGUAGE)
        template = locale.get(key, key)
        if kwargs:
            return template.format(**kwargs)
        return template

    def _register_ui_refresher(self, refresher: object) -> None:
        def _safe_refresh() -> None:
            with suppress(tk.TclError, RuntimeError):
                refresher()  # type: ignore[misc]

        self._ui_refreshers.append(_safe_refresh)
        _safe_refresh()

    def _bind_text(self, widget: ttk.Widget | tk.Widget, key: str, **kwargs: object) -> None:
        self._register_ui_refresher(lambda: widget.configure(text=self._tr(key, **kwargs)))

    def _bind_labelframe_text(
        self,
        widget: ttk.LabelFrame,
        key: str,
        **kwargs: object,
    ) -> None:
        self._register_ui_refresher(lambda: widget.configure(text=self._tr(key, **kwargs)))

    def _bind_tree_heading(
        self,
        tree: ttk.Treeview,
        column: str,
        key: str,
        **kwargs: object,
    ) -> None:
        self._register_ui_refresher(lambda: tree.heading(column, text=self._tr(key, **kwargs)))

    def _bind_notebook_tab(self, tab_index: int, key: str, **kwargs: object) -> None:
        self._register_ui_refresher(
            lambda: self._notebook.tab(tab_index, text=f"  {self._tr(key, **kwargs)}  ")
        )

    def _bind_tooltip(self, widget: tk.Widget, key: str, **kwargs: object) -> ToolTip:
        tip = ToolTip(widget, self._tr(key, **kwargs))
        self._register_ui_refresher(lambda: tip.set_text(self._tr(key, **kwargs)))
        self._tooltips.append((tip, key))
        return tip

    def _language_label(self, code: str) -> str:
        return self._tr(f"language.name.{normalize_gui_language(code)}")

    def _language_code_from_label(self, label: str) -> str:
        normalized = str(label).strip()
        for code in SUPPORTED_GUI_LANGUAGES:
            if normalized == self._language_label(code):
                return code
        return normalize_gui_language(normalized)

    def _log_mode_label(self, code: str) -> str:
        normalized = "debug" if str(code).strip().lower() == "debug" else "normal"
        return self._tr(f"log_mode.{normalized}")

    def _log_mode_code_from_label(self, label: str) -> str:
        normalized = str(label).strip()
        if normalized == self._log_mode_label("debug") or normalized.lower() == "debug":
            return "debug"
        return "normal"

    def _quality_label(self, value: str) -> str:
        normalized = self._normalize_quality_label(value)
        return self._tr(f"quality.{normalized.lower()}")

    def _normalize_quality_label(self, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            return "Balanced"
        for label in QUALITY_PRESET_LABELS:
            if normalized == label or normalized.lower() == label.lower():
                return label
            if normalized == self._tr(f"quality.{label.lower()}"):
                return label
        lowered = normalized.lower()
        if lowered in {"fast", "balanced", "quality"}:
            return lowered.capitalize() if lowered != "quality" else "Quality"
        return "Balanced"

    def _refresh_quality_selector(self) -> None:
        if not hasattr(self, "_file_quality_combo"):
            return
        values = [self._quality_label(label) for label in QUALITY_PRESET_LABELS]
        self._file_quality_combo.configure(values=values)
        self._file_quality_var.set(self._normalize_quality_label(self._file_quality_var.get()))
        self._file_quality_display_var.set(self._quality_label(self._file_quality_var.get()))

    def _on_file_quality_changed(self, _event: object | None = None) -> None:
        self._file_quality_var.set(
            self._normalize_quality_label(self._file_quality_display_var.get())
        )
        self._file_quality_display_var.set(self._quality_label(self._file_quality_var.get()))

    def _speaker_preset_label(self, value: str) -> str:
        normalized = self._normalize_speaker_preset(value)
        return self._tr(f"speaker_preset.{normalized}")

    def _normalize_speaker_preset(self, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            return "auto"
        legacy_map = {
            "auto": "auto",
            "1": "1",
            "1 speaker": "1",
            "2": "2",
            "2 speakers": "2",
            "3": "3",
            "3 speakers": "3",
            "4+": "4plus",
            "4plus": "4plus",
            "4+ speakers": "4plus",
            "custom": "custom",
        }
        lowered = normalized.casefold()
        if lowered in legacy_map:
            return legacy_map[lowered]
        for code in FILE_SPEAKER_PRESET_CODES:
            if normalized == self._speaker_preset_label(code):
                return code
        return "auto"

    def _refresh_speaker_preset_selector(self) -> None:
        if not hasattr(self, "_file_speaker_preset_combo"):
            return
        values = [self._speaker_preset_label(code) for code in FILE_SPEAKER_PRESET_CODES]
        self._file_speaker_preset_combo.configure(values=values)
        self._file_speaker_preset_var.set(
            self._normalize_speaker_preset(self._file_speaker_preset_var.get())
        )
        self._file_speaker_preset_display_var.set(
            self._speaker_preset_label(self._file_speaker_preset_var.get())
        )

    def _device_option_label(self, kind: str, device_label: str) -> str:
        key = "device.option.microphone" if kind == "microphone" else "device.option.system"
        return self._tr(key, device=device_label)

    def _device_short_label(self, kind: str, device_label: str) -> str:
        key = "device.selected.microphone" if kind == "microphone" else "device.selected.system"
        return self._tr(key, device=device_label)

    def _media_filetypes(self) -> list[tuple[str, str]]:
        return [
            (self._tr("dialog.filetype.supported_media"), _AUDIO_EXTENSIONS),
            (
                self._tr("dialog.filetype.video_files"),
                " ".join(
                    f"*{ext}"
                    for ext in sorted(
                        {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".ts"}
                    )
                ),
            ),
            (
                self._tr("dialog.filetype.audio_files"),
                " ".join(
                    f"*{ext}"
                    for ext in sorted({".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".opus"})
                ),
            ),
            (self._tr("dialog.filetype.all_files"), "*.*"),
        ]

    def _recording_save_filetypes(self, fmt: str) -> tuple[str, list[tuple[str, str]]]:
        fmt_filetypes = {
            "wav": ("wav", [(self._tr("dialog.filetype.wav_audio"), "*.wav")]),
            "ogg": ("ogg", [(self._tr("dialog.filetype.ogg_audio"), "*.ogg")]),
            "opus": ("opus", [(self._tr("dialog.filetype.opus_audio"), "*.opus")]),
            "mp3": ("mp3", [(self._tr("dialog.filetype.mp3_audio"), "*.mp3")]),
        }
        return fmt_filetypes.get(
            fmt,
            (fmt, [(self._tr("dialog.filetype.audio_files"), f"*.{fmt}")]),
        )

    def _apply_localized_ui(self) -> None:
        self.root.title(self._tr("app.title"))
        for refresher in self._ui_refreshers:
            refresher()
        self._refresh_language_selector()
        self._refresh_log_mode_selector()
        self._refresh_quality_selector()
        self._refresh_speaker_preset_selector()
        if hasattr(self, "_live_model_summary"):
            self._live_model_summary.set_translator(self._tr)
        if hasattr(self, "_file_model_summary"):
            self._file_model_summary.set_translator(self._tr)
        if hasattr(self, "_device_menu"):
            self._refresh_device_options()
        self._refresh_file_queue_rows()
        self._refresh_live_summary_labels()
        self._refresh_file_workflow()

    def _refresh_language_selector(self) -> None:
        if not hasattr(self, "_ui_language_combo"):
            return
        values = [self._language_label(code) for code in SUPPORTED_GUI_LANGUAGES]
        self._ui_language_combo.configure(values=values)
        self._ui_language_var.set(self._language_label(self._ui_language_code))

    def _refresh_log_mode_selector(self) -> None:
        if not hasattr(self, "_log_mode_combo"):
            return
        values = [self._log_mode_label("normal"), self._log_mode_label("debug")]
        self._log_mode_combo.configure(values=values)
        self._log_mode_var.set(self._log_mode_label(self._log_mode_code))

    def _on_ui_language_changed(self, _event: object | None = None) -> None:
        selected_code = self._language_code_from_label(self._ui_language_var.get())
        self._ui_language_code = normalize_gui_language(selected_code)
        self._ui_language_explicit = True
        self._locale = load_gui_locale(self._ui_language_code)
        self._apply_localized_ui()
        self._persist_gui_settings()

    # ------------------------------------------------------------------
    # Layout builders
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        """Build the top-level layout with a two-tab Notebook and resizable log pane."""
        header = ttk.Frame(self.root)
        header.pack(fill=tk.X, padx=6, pady=(6, 2))

        header_title = ttk.Label(header, text=self._tr("app.title"), anchor="w")
        header_title.pack(side=tk.LEFT)

        header_controls = ttk.Frame(header)
        header_controls.pack(side=tk.RIGHT)

        self._settings_button = ttk.Button(header_controls, text="", command=self._open_settings)
        self._settings_button.pack(side=tk.RIGHT, padx=(8, 0))
        self._bind_text(self._settings_button, "header.settings")
        self._bind_tooltip(self._settings_button, "tooltip.header.settings")

        self._ui_language_combo = ttk.Combobox(
            header_controls,
            textvariable=self._ui_language_var,
            state="readonly",
            width=10,
        )
        self._ui_language_combo.pack(side=tk.RIGHT, padx=(0, 8))
        self._ui_language_combo.bind("<<ComboboxSelected>>", self._on_ui_language_changed)
        self._bind_tooltip(self._ui_language_combo, "tooltip.header.language")

        self._ui_language_label = ttk.Label(header_controls, text="")
        self._ui_language_label.pack(side=tk.RIGHT, padx=(0, 4))
        self._bind_text(self._ui_language_label, "header.language")

        paned = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=2, pady=(2, 2))

        notebook_frame = ttk.Frame(paned)
        self._notebook = ttk.Notebook(notebook_frame)
        self._notebook.pack(fill=tk.BOTH, expand=True)
        paned.add(notebook_frame, weight=3)

        live_tab = ttk.Frame(self._notebook)
        self._notebook.add(live_tab, text="")
        self._bind_notebook_tab(0, "live.tab")
        self._build_live_tab(live_tab)

        file_tab = ttk.Frame(self._notebook)
        self._notebook.add(file_tab, text="")
        self._bind_notebook_tab(1, "file.tab")
        self._build_file_tab(file_tab)

        log_frame = ttk.Frame(paned)
        log_header = ttk.Frame(log_frame)
        log_header.pack(fill=tk.X, padx=4, pady=(2, 0))
        self._logs_label = ttk.Label(log_header, text="", anchor="w")
        self._logs_label.pack(side=tk.LEFT)
        self._bind_text(self._logs_label, "header.logs")

        log_controls = ttk.Frame(log_header)
        log_controls.pack(side=tk.RIGHT)

        self._log_mode_combo = ttk.Combobox(
            log_controls,
            textvariable=self._log_mode_var,
            state="readonly",
            width=10,
        )
        self._log_mode_combo.pack(side=tk.RIGHT, padx=(0, 8))
        self._log_mode_combo.bind("<<ComboboxSelected>>", self._on_log_mode_changed)
        self._bind_tooltip(self._log_mode_combo, "tooltip.header.log_mode")

        self._log_mode_caption = ttk.Label(log_controls, text="")
        self._log_mode_caption.pack(side=tk.RIGHT, padx=(0, 4))
        self._bind_text(self._log_mode_caption, "header.log_mode")

        self.log_widget = scrolledtext.ScrolledText(
            log_frame,
            wrap=tk.WORD,
            state=tk.DISABLED,
        )
        self.log_widget.pack(fill=tk.BOTH, expand=True, padx=4, pady=(2, 4))
        self._bind_tooltip(self.log_widget, "tooltip.logs.panel")
        paned.add(log_frame, weight=1)

    def _build_live_tab(self, parent: ttk.Frame) -> None:
        """Build the Live Capture tab contents."""
        live_paned = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        live_paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        top_frame = ttk.Frame(live_paned)
        live_paned.add(top_frame, weight=1)

        settings_box = ttk.LabelFrame(top_frame, text="", padding=(6, 5))
        settings_box.pack(fill=tk.X, padx=0, pady=(0, 4))
        self._bind_labelframe_text(settings_box, "live.section.capture_setup")
        settings_box.columnconfigure(1, weight=1)
        settings_box.columnconfigure(3, weight=2)

        # Row 0: Multi-select device picker
        self._live_devices_label = ttk.Label(settings_box, text="")
        self._live_devices_label.grid(row=0, column=0, sticky="w", padx=(0, 4))
        self._bind_text(self._live_devices_label, "live.label.devices")
        self.device_picker = ttk.Menubutton(
            settings_box,
            textvariable=self._device_picker_var,
            direction="below",
        )
        self.device_picker.grid(row=0, column=1, columnspan=3, sticky="ew", padx=(0, 12))
        self._device_menu = tk.Menu(self.device_picker, tearoff=0)
        self.device_picker.configure(menu=self._device_menu)
        self._bind_tooltip(self.device_picker, "tooltip.live.devices")

        # Row 1: Model | Language | Translate
        self._live_model_label = ttk.Label(settings_box, text="")
        self._live_model_label.grid(row=1, column=0, sticky="w", padx=(0, 4), pady=(4, 0))
        self._bind_text(self._live_model_label, "live.label.model")
        self.model_combo = ttk.Combobox(
            settings_box,
            textvariable=self._model_var,
            state="readonly",
            width=20,
            values=ASR_MODEL_CHOICES,
        )
        self.model_combo.grid(row=1, column=1, sticky="w", padx=(0, 12), pady=(4, 0))
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_changed)
        self._bind_tooltip(self.model_combo, "tooltip.live.model")

        lang_row = ttk.Frame(settings_box)
        lang_row.grid(row=1, column=2, columnspan=2, sticky="ew", pady=(4, 0))
        self._live_language_label = ttk.Label(lang_row, text="")
        self._live_language_label.pack(side=tk.LEFT, padx=(0, 4))
        self._bind_text(self._live_language_label, "live.label.language")
        self.language_combo = ttk.Combobox(
            lang_row,
            textvariable=self._language_var,
            state="readonly",
            width=18,
        )
        self.language_combo.pack(side=tk.LEFT, padx=(0, 12))
        self.language_combo.bind("<<ComboboxSelected>>", self._on_live_language_changed)
        self._bind_tooltip(self.language_combo, "tooltip.live.language")
        self._live_translate_label = ttk.Label(lang_row, text="")
        self._live_translate_label.pack(side=tk.LEFT, padx=(0, 4))
        self._bind_text(self._live_translate_label, "live.label.translate")
        self.translate_entry = ttk.Entry(
            lang_row,
            textvariable=self._translate_var,
            width=8,
        )
        self.translate_entry.pack(side=tk.LEFT)
        self._bind_tooltip(self.translate_entry, "tooltip.live.translate")

        # Row 2: Action buttons + stats (all in one row)
        btn_row = ttk.Frame(settings_box)
        btn_row.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(6, 2))

        self.start_button = ttk.Button(
            btn_row, text="", command=self._start_capture, style="Primary.TButton"
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 4))
        self._bind_text(self.start_button, "live.button.start")
        self._bind_tooltip(self.start_button, "tooltip.live.start")
        self.stop_button = ttk.Button(btn_row, text="", command=self._stop_capture)
        self.stop_button.configure(state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 4))
        self._bind_text(self.stop_button, "live.button.stop")
        self._bind_tooltip(self.stop_button, "tooltip.live.stop")
        self.pause_button = ttk.Button(btn_row, text="", command=self._toggle_recording_pause)
        self.pause_button.configure(state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=(0, 4))
        self._bind_text(self.pause_button, "live.button.pause")
        self._bind_tooltip(self.pause_button, "tooltip.live.pause")
        self.record_button = ttk.Button(
            btn_row,
            text="",
            command=self._start_recording,
            style="Accent.TButton",
        )
        self.record_button.pack(side=tk.LEFT, padx=(0, 4))
        self._bind_text(self.record_button, "live.button.record_audio")
        self._bind_tooltip(self.record_button, "tooltip.live.record_audio")
        self._rec_format_combo = ttk.Combobox(
            btn_row,
            textvariable=self._rec_format_var,
            values=["wav", "ogg", "opus", "mp3"],
            state="readonly",
            width=5,
        )
        self._rec_format_combo.pack(side=tk.LEFT, padx=(0, 12))
        self._bind_tooltip(self._rec_format_combo, "tooltip.live.record_format")

        ttk.Separator(btn_row, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))

        self.clear_button = ttk.Button(btn_row, text="", command=self._clear_table)
        self.clear_button.pack(side=tk.LEFT, padx=(0, 4))
        self._bind_text(self.clear_button, "live.button.clear")
        self._bind_tooltip(self.clear_button, "tooltip.live.clear")
        self.save_button = ttk.Button(btn_row, text="", command=self._save_to_file)
        self.save_button.pack(side=tk.LEFT)
        self._bind_text(self.save_button, "live.button.save")
        self._bind_tooltip(self.save_button, "tooltip.live.save")

        self.queue_label = ttk.Label(btn_row, text="")
        self.queue_label.pack(side=tk.RIGHT, padx=(8, 0))
        self._bind_tooltip(self.queue_label, "tooltip.live.summary")
        self.counter_label = ttk.Label(btn_row, text="")
        self.counter_label.pack(side=tk.RIGHT)
        self._bind_tooltip(self.counter_label, "tooltip.live.segment_counter")

        # Hidden model summary (kept for API compatibility — not displayed)
        self._live_model_summary = ModelSummaryCard(
            settings_box,
            title=self._tr("model_summary.title.live"),
            translate=self._tr,
        )

        self.status_label = ttk.Label(top_frame, text="", anchor="w")
        self.status_label.pack(fill=tk.X, padx=0, pady=(2, 2))
        self._bind_tooltip(self.status_label, "tooltip.live.status")

        table_frame = ttk.Frame(live_paned)
        live_paned.add(table_frame, weight=6)

        _style = ttk.Style()
        _style.configure("Treeview", rowheight=22)

        columns = ("time", "source", "text", "translation")
        self.table = ttk.Treeview(table_frame, columns=columns, show="headings")
        self._bind_tree_heading(self.table, "time", "live.table.time")
        self._bind_tree_heading(self.table, "source", "live.table.source")
        self._bind_tree_heading(self.table, "text", "live.table.text")
        self._bind_tree_heading(self.table, "translation", "live.table.translation")
        self.table.column("time", width=80, minwidth=70, stretch=False)
        self.table.column("source", width=80, minwidth=70, stretch=False)
        self.table.column("text", width=500, minwidth=220)
        self.table.column("translation", width=400, minwidth=220)
        self.table.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        self._bind_tooltip(self.table, "tooltip.live.table")
        self.table.bind("<Control-c>", self._copy_selected_rows)
        self.table.bind("<Button-3>", self._show_context_menu)
        self.table.bind("<Control-a>", lambda e: self._select_all_rows())
        self._context_menu = tk.Menu(self.root, tearoff=0)
        self._context_menu.add_command(label="", command=self._copy_selected_rows)
        self._context_menu.add_command(label="", command=self._copy_text_only)
        self._context_menu.add_separator()
        self._context_menu.add_command(label="", command=self._select_all_rows)
        self._register_ui_refresher(
            lambda: self._context_menu.entryconfigure(0, label=self._tr("live.menu.copy_selected"))
        )
        self._register_ui_refresher(
            lambda: self._context_menu.entryconfigure(1, label=self._tr("live.menu.copy_text_only"))
        )
        self._register_ui_refresher(
            lambda: self._context_menu.entryconfigure(3, label=self._tr("live.menu.select_all"))
        )

        scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.table.yview)
        scroll.pack(fill=tk.Y, side=tk.RIGHT)
        self.table.configure(yscrollcommand=scroll.set)
        self.table.tag_configure("dropped", foreground="red")
        self.table.tag_configure("continuation", foreground="#666666")

        self.root.after(500, self._poll_stats)

    def _build_file_tab(self, parent: ttk.Frame) -> None:
        """Build the File Transcription tab contents."""
        file_paned = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        file_paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # -- FFmpeg warning banner (hidden when FFmpeg is present) --
        self._ffmpeg_banner = tk.Frame(parent, bg="#fff3cd")
        if self._ffmpeg_path is None:
            self._ffmpeg_banner.pack(fill=tk.X, padx=6, pady=(6, 0))
        self._ffmpeg_banner_label = tk.Label(
            self._ffmpeg_banner,
            text="",
            bg="#fff3cd",
            fg="#856404",
            anchor="w",
        )
        self._ffmpeg_banner_label.pack(side=tk.LEFT, padx=(8, 12), pady=4)
        self._bind_text(self._ffmpeg_banner_label, "file.banner.ffmpeg_missing")
        self._ffmpeg_install_btn = tk.Button(
            self._ffmpeg_banner,
            text="",
            command=self._install_ffmpeg,
            bg="#e0a800",
            fg="white",
            relief="flat",
            padx=8,
            pady=2,
        )
        self._ffmpeg_install_btn.pack(side=tk.LEFT, pady=4)
        self._bind_text(self._ffmpeg_install_btn, "file.button.install_ffmpeg")
        self._ffmpeg_install_status = tk.Label(
            self._ffmpeg_banner,
            text="",
            bg="#fff3cd",
            fg="#333333",
            anchor="w",
        )
        self._ffmpeg_install_status.pack(side=tk.LEFT, padx=(8, 0), pady=4)

        top_area = ttk.Frame(file_paned)
        file_paned.add(top_area, weight=3)

        workflow_hdr = ttk.Frame(top_area)
        workflow_hdr.pack(fill=tk.X, padx=0, pady=(0, 2))
        self._file_workflow_title = ttk.Label(
            workflow_hdr,
            text="",
            style="Header.TLabel",
        )
        self._file_workflow_title.pack(side=tk.LEFT)
        self._bind_text(self._file_workflow_title, "file.workflow.title")

        self._file_workflow_label = ttk.Label(
            top_area,
            text="",
            anchor="w",
            foreground="#555555",
        )
        self._file_workflow_label.pack(fill=tk.X, padx=0, pady=(0, 4))

        # -- File picker row --
        # Use PanedWindow so column widths are driven by the sash position,
        # not by widget content — switching models won't cause the layout to jump.
        top = ttk.PanedWindow(top_area, orient=tk.HORIZONTAL)
        top.pack(fill=tk.BOTH, expand=True, padx=0, pady=(0, 4))

        transcribe_box = ttk.LabelFrame(top, text="", padding=8)
        top.add(transcribe_box, weight=3)
        self._bind_labelframe_text(transcribe_box, "file.section.transcription_setup")

        picker = ttk.Frame(transcribe_box)
        picker.pack(fill=tk.X, pady=(0, 4))

        self._file_current_label = ttk.Label(picker, text="")
        self._file_current_label.pack(side=tk.LEFT, padx=(0, 6))
        self._bind_text(self._file_current_label, "file.label.current")
        self._file_path_entry = ttk.Entry(picker, textvariable=self._file_path_var, width=70)
        self._file_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self._bind_tooltip(self._file_path_entry, "tooltip.file.current")
        self._file_add_btn = ttk.Button(
            picker,
            text="",
            command=self._browse_file,
        )
        self._file_add_btn.pack(side=tk.LEFT, padx=(0, 4))
        self._bind_text(self._file_add_btn, "file.button.add_files")
        self._bind_tooltip(self._file_add_btn, "tooltip.file.add_files")
        self._file_remove_btn = ttk.Button(
            picker,
            text="",
            command=self._remove_selected_file_queue_items,
        )
        self._file_remove_btn.pack(side=tk.LEFT, padx=(0, 4))
        self._bind_text(self._file_remove_btn, "file.button.remove")
        self._bind_tooltip(self._file_remove_btn, "tooltip.file.remove")
        self._file_clear_queue_btn = ttk.Button(
            picker,
            text="",
            command=self._clear_file_queue,
        )
        self._file_clear_queue_btn.pack(side=tk.LEFT)
        self._bind_text(self._file_clear_queue_btn, "file.button.clear_list")
        self._bind_tooltip(self._file_clear_queue_btn, "tooltip.file.clear_list")

        queue_frame = ttk.Frame(transcribe_box)
        queue_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 6))

        queue_cols = ("file", "duration", "size", "status", "progress", "result")
        self._file_queue_table = ttk.Treeview(
            queue_frame,
            columns=queue_cols,
            show="headings",
            height=4,
        )
        self._bind_tree_heading(self._file_queue_table, "file", "file.table.file")
        self._bind_tree_heading(self._file_queue_table, "duration", "file.table.duration")
        self._bind_tree_heading(self._file_queue_table, "size", "file.table.size")
        self._bind_tree_heading(self._file_queue_table, "status", "file.table.status")
        self._bind_tree_heading(self._file_queue_table, "progress", "file.table.progress")
        self._bind_tree_heading(self._file_queue_table, "result", "file.table.result")
        self._file_queue_table.column("file", width=360, minwidth=220)
        self._file_queue_table.column("duration", width=88, minwidth=76, stretch=False)
        self._file_queue_table.column("size", width=88, minwidth=76, stretch=False)
        self._file_queue_table.column("status", width=120, minwidth=90, stretch=False)
        self._file_queue_table.column("progress", width=90, minwidth=70, stretch=False)
        self._file_queue_table.column("result", width=220, minwidth=150)
        self._file_queue_table.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        self._bind_tooltip(self._file_queue_table, "tooltip.file.queue_table")
        self._file_queue_table.bind(
            "<<TreeviewSelect>>",
            self._on_file_queue_selection,
        )

        queue_scroll = ttk.Scrollbar(
            queue_frame,
            orient=tk.VERTICAL,
            command=self._file_queue_table.yview,
        )
        queue_scroll.pack(fill=tk.Y, side=tk.RIGHT)
        self._file_queue_table.configure(yscrollcommand=queue_scroll.set)

        # -- Options row --
        opts = ttk.Frame(transcribe_box)
        opts.pack(fill=tk.X, pady=(0, 4))

        self._file_model_label = ttk.Label(opts, text="")
        self._file_model_label.pack(side=tk.LEFT, padx=(0, 6))
        self._bind_text(self._file_model_label, "file.label.model")
        self._file_model_combo = ttk.Combobox(
            opts,
            textvariable=self._file_model_var,
            state="readonly",
            width=10,
            values=ASR_MODEL_CHOICES,
        )
        self._file_model_combo.pack(side=tk.LEFT, padx=(0, 12))
        self._file_model_combo.bind("<<ComboboxSelected>>", self._on_file_model_changed)
        self._bind_tooltip(self._file_model_combo, "tooltip.file.model")

        self._file_language_label = ttk.Label(opts, text="")
        self._file_language_label.pack(side=tk.LEFT, padx=(0, 6))
        self._bind_text(self._file_language_label, "file.label.language")
        self._file_lang_combo = ttk.Combobox(
            opts,
            textvariable=self._file_lang_var,
            state="readonly",
            width=18,
        )
        self._file_lang_combo.pack(side=tk.LEFT, padx=(0, 12))
        self._bind_tooltip(self._file_lang_combo, "tooltip.file.language")

        self._file_quality_label = ttk.Label(opts, text="")
        self._file_quality_label.pack(side=tk.LEFT, padx=(0, 6))
        self._bind_text(self._file_quality_label, "file.label.quality")
        self._file_quality_combo = ttk.Combobox(
            opts,
            textvariable=self._file_quality_display_var,
            state="readonly",
            width=11,
            values=(),
        )
        self._file_quality_combo.pack(side=tk.LEFT, padx=(0, 16))
        self._file_quality_combo.bind("<<ComboboxSelected>>", self._on_file_quality_changed)
        self._bind_tooltip(self._file_quality_combo, "tooltip.file.quality")

        diar_opts = ttk.Frame(transcribe_box)
        diar_opts.pack(fill=tk.X, pady=(0, 4))

        self._file_diarization_label = ttk.Label(diar_opts, text="")
        self._file_diarization_label.pack(side=tk.LEFT, padx=(0, 4))
        self._bind_text(self._file_diarization_label, "file.label.speaker_separation")
        self._file_diarization_combo = ttk.Combobox(
            diar_opts,
            textvariable=self._file_diarization_var,
            state="readonly",
            width=9,
            values=FILE_DIARIZATION_CHOICES,
        )
        self._file_diarization_combo.pack(side=tk.LEFT, padx=(0, 0))
        self._file_diarization_combo.bind(
            "<<ComboboxSelected>>",
            self._on_file_diarization_changed,
        )
        self._bind_tooltip(self._file_diarization_combo, "tooltip.file.diarization")
        self._file_diarization_help = create_help_icon(diar_opts, "")
        self._bind_tooltip(self._file_diarization_help, "tooltip.file.diarization_help")

        ttk.Separator(diar_opts, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=(8, 8))

        self._file_speakers_label = ttk.Label(diar_opts, text="")
        self._file_speakers_label.pack(side=tk.LEFT, padx=(0, 4))
        self._bind_text(self._file_speakers_label, "file.label.speakers")
        self._file_speaker_preset_combo = ttk.Combobox(
            diar_opts,
            textvariable=self._file_speaker_preset_display_var,
            state="readonly",
            width=12,
            values=(),
        )
        self._file_speaker_preset_combo.pack(side=tk.LEFT, padx=(0, 0))
        self._file_speaker_preset_combo.bind(
            "<<ComboboxSelected>>",
            self._on_speaker_preset_changed,
        )
        self._bind_tooltip(self._file_speaker_preset_combo, "tooltip.file.speaker_preset")
        self._file_speaker_help = create_help_icon(diar_opts, "")
        self._bind_tooltip(self._file_speaker_help, "tooltip.file.speaker_help")

        ttk.Separator(diar_opts, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=(8, 8))

        self._file_min_lbl = ttk.Label(diar_opts, text="")
        self._file_min_lbl.pack(side=tk.LEFT, padx=(0, 4))
        self._bind_text(self._file_min_lbl, "file.label.min")
        self._file_min_speakers_entry = ttk.Entry(
            diar_opts,
            textvariable=self._file_min_speakers_var,
            width=4,
        )
        self._file_min_speakers_entry.pack(side=tk.LEFT, padx=(0, 6))
        self._bind_tooltip(self._file_min_speakers_entry, "tooltip.file.min_speakers")

        self._file_max_lbl = ttk.Label(diar_opts, text="")
        self._file_max_lbl.pack(side=tk.LEFT, padx=(0, 4))
        self._bind_text(self._file_max_lbl, "file.label.max")
        self._file_max_speakers_entry = ttk.Entry(
            diar_opts,
            textvariable=self._file_max_speakers_var,
            width=4,
        )
        self._file_max_speakers_entry.pack(side=tk.LEFT, padx=(0, 6))
        self._bind_tooltip(self._file_max_speakers_entry, "tooltip.file.max_speakers")

        self._file_detect_btn = ttk.Button(
            diar_opts,
            text="",
            command=self._detect_speakers,
            width=7,
        )
        self._file_detect_btn.pack(side=tk.LEFT, padx=(0, 4))
        self._bind_text(self._file_detect_btn, "file.button.detect")
        self._bind_tooltip(self._file_detect_btn, "tooltip.file.detect")

        self._file_download_btn = ttk.Button(opts, text="", command=self._download_file_model)
        self._file_download_btn.pack(side=tk.LEFT, padx=(0, 12))
        self._bind_text(self._file_download_btn, "file.button.download_model")
        self._bind_tooltip(self._file_download_btn, "tooltip.file.download_model")

        self._file_transcribe_btn = ttk.Button(
            opts,
            text="",
            command=self._start_file_transcribe,
            style="Accent.TButton",
        )
        self._file_transcribe_btn.pack(side=tk.LEFT, padx=(0, 4))
        self._bind_text(self._file_transcribe_btn, "file.button.transcribe_queue")
        self._bind_tooltip(self._file_transcribe_btn, "tooltip.file.transcribe_queue")
        self._file_cancel_btn = ttk.Button(
            opts, text="", command=self._cancel_file_transcribe, state=tk.DISABLED
        )
        self._file_cancel_btn.pack(side=tk.LEFT, padx=(0, 4))
        self._bind_text(self._file_cancel_btn, "file.button.cancel")
        self._bind_tooltip(self._file_cancel_btn, "tooltip.file.cancel")

        # -- Status + progress row --
        # Pack right-anchored widgets first so the expanding status label fills the rest.
        prog_row = ttk.Frame(transcribe_box)
        prog_row.pack(fill=tk.X, pady=(0, 4))

        self._file_progress = ttk.Progressbar(
            prog_row, orient="horizontal", length=180, mode="determinate", maximum=100
        )
        self._file_progress.pack(side=tk.RIGHT)
        self._bind_tooltip(self._file_progress, "tooltip.file.progress")

        self._file_time_label = ttk.Label(prog_row, text="", anchor="e", width=18)
        self._file_time_label.pack(side=tk.RIGHT, padx=(0, 6))

        self._file_status_label = ttk.Label(prog_row, text="", anchor="w")
        self._file_status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._bind_tooltip(self._file_status_label, "tooltip.file.status")

        self._file_artifact_label = ttk.Label(
            transcribe_box,
            text="",
            anchor="w",
            foreground="#555555",
        )
        self._file_artifact_label.pack(fill=tk.X, pady=(0, 4))
        self._bind_tooltip(self._file_artifact_label, "tooltip.file.artifact")

        self._file_model_summary = ModelSummaryCard(
            top,
            title=self._tr("model_summary.title.file"),
            translate=self._tr,
        )
        top.add(self._file_model_summary, weight=2)

        # -- Results table --
        results_frame = ttk.Frame(file_paned)
        file_paned.add(results_frame, weight=6)

        file_table_frame = ttk.Frame(results_frame)
        file_table_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=(0, 4))

        file_cols = ("time", "speaker", "text")
        self._file_table = ttk.Treeview(file_table_frame, columns=file_cols, show="headings")
        self._bind_tree_heading(self._file_table, "time", "file.table.timestamp")
        self._bind_tree_heading(self._file_table, "speaker", "file.table.speaker")
        self._bind_tree_heading(self._file_table, "text", "file.table.text")
        self._file_table.column("time", width=90, minwidth=70, stretch=False)
        self._file_table.column("speaker", width=110, minwidth=80, stretch=False)
        self._file_table.column("text", width=800, minwidth=300)
        self._file_table.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        self._bind_tooltip(self._file_table, "tooltip.file.results_table")
        self._file_table.bind("<Control-c>", self._file_copy_selected)
        self._file_table.bind(
            "<Control-a>",
            lambda _e: self._file_table.selection_set(self._file_table.get_children()),
        )

        file_scroll = ttk.Scrollbar(
            file_table_frame, orient=tk.VERTICAL, command=self._file_table.yview
        )
        file_scroll.pack(fill=tk.Y, side=tk.RIGHT)
        self._file_table.configure(yscrollcommand=file_scroll.set)

        # -- File controls --
        file_ctrl = ttk.Frame(results_frame)
        file_ctrl.pack(fill=tk.X, padx=0, pady=(0, 4))

        self._file_load_transcript_btn = ttk.Button(
            file_ctrl, text="", command=self._load_transcript_file
        )
        self._file_load_transcript_btn.pack(side=tk.LEFT, padx=(0, 4))
        self._bind_text(self._file_load_transcript_btn, "file.button.load_transcript")
        self._bind_tooltip(self._file_load_transcript_btn, "tooltip.file.load_transcript")
        self._file_clear_btn = ttk.Button(file_ctrl, text="", command=self._clear_file_table)
        self._file_clear_btn.pack(side=tk.LEFT, padx=(0, 4))
        self._bind_text(self._file_clear_btn, "file.button.clear")
        self._bind_tooltip(self._file_clear_btn, "tooltip.file.clear")
        self._file_save_btn = ttk.Button(file_ctrl, text="", command=self._save_file_result)
        self._file_save_btn.pack(side=tk.LEFT)
        self._bind_text(self._file_save_btn, "file.button.save")
        self._bind_tooltip(self._file_save_btn, "tooltip.file.save")
        self._file_seg_counter_label = ttk.Label(file_ctrl, text="")
        self._file_seg_counter_label.pack(side=tk.RIGHT)
        self._bind_tooltip(self._file_seg_counter_label, "tooltip.file.segment_counter")

        # -- LLM processing panel --
        llm_box = ttk.LabelFrame(file_paned, text="", padding=8)
        file_paned.add(llm_box, weight=2)
        self._bind_labelframe_text(llm_box, "file.section.transcript_processing")

        llm_hdr = ttk.Frame(llm_box)
        llm_hdr.pack(fill=tk.X, pady=(0, 2))
        self._llm_header_label = ttk.Label(llm_hdr, text="", font=("", 9, "bold"))
        self._llm_header_label.pack(side=tk.LEFT)
        self._bind_text(self._llm_header_label, "llm.header")

        self._llm_export_btn = ttk.Button(llm_hdr, text="", command=self._save_file_result)
        self._llm_export_btn.pack(side=tk.RIGHT, padx=(4, 0))
        self._bind_text(self._llm_export_btn, "file.button.save")
        self._bind_tooltip(self._llm_export_btn, "tooltip.file.save")

        self._llm_load_transcript_btn = ttk.Button(
            llm_hdr,
            text="",
            command=self._load_transcript_file,
        )
        self._llm_load_transcript_btn.pack(side=tk.RIGHT, padx=(8, 0))
        self._bind_text(self._llm_load_transcript_btn, "file.button.load_transcript")
        self._bind_tooltip(self._llm_load_transcript_btn, "tooltip.file.load_transcript")

        llm_cfg = ttk.Frame(llm_box)
        llm_cfg.pack(fill=tk.X, pady=(0, 2))

        self._llm_url_label = ttk.Label(llm_cfg, text="")
        self._llm_url_label.pack(side=tk.LEFT, padx=(0, 4))
        self._bind_text(self._llm_url_label, "llm.label.url")
        self._llm_url_entry = ttk.Entry(llm_cfg, textvariable=self._llm_url_var, width=26)
        self._llm_url_entry.pack(side=tk.LEFT, padx=(0, 10))
        self._bind_tooltip(self._llm_url_entry, "tooltip.llm.url")
        self._llm_refresh_btn = ttk.Button(llm_cfg, text="", command=self._refresh_llm_models)
        self._llm_refresh_btn.pack(side=tk.LEFT, padx=(0, 4))
        self._bind_text(self._llm_refresh_btn, "file.button.refresh_models")
        self._bind_tooltip(self._llm_refresh_btn, "tooltip.llm.refresh_models")
        self._llm_probe_btn = ttk.Button(llm_cfg, text="", command=self._probe_llm_model)
        self._llm_probe_btn.pack(side=tk.LEFT, padx=(0, 10))
        self._bind_text(self._llm_probe_btn, "llm.button.test_model")
        self._bind_tooltip(self._llm_probe_btn, "tooltip.llm.test_model")
        self._llm_model_label = ttk.Label(llm_cfg, text="")
        self._llm_model_label.pack(side=tk.LEFT, padx=(0, 4))
        self._bind_text(self._llm_model_label, "llm.label.model")
        self._llm_model_combo = ttk.Combobox(
            llm_cfg,
            textvariable=self._llm_model_var,
            width=24,
        )
        self._llm_model_combo.pack(side=tk.LEFT, padx=(0, 10))
        self._bind_tooltip(self._llm_model_combo, "tooltip.llm.model")
        self._llm_api_key_label = ttk.Label(llm_cfg, text="")
        self._llm_api_key_label.pack(side=tk.LEFT, padx=(0, 4))
        self._bind_text(self._llm_api_key_label, "llm.label.api_key")
        self._llm_key_entry = ttk.Entry(llm_cfg, textvariable=self._llm_key_var, width=14, show="*")
        self._llm_key_entry.pack(side=tk.LEFT, padx=(0, 10))
        self._bind_tooltip(self._llm_key_entry, "tooltip.llm.api_key")
        self._llm_prompt_label = ttk.Label(llm_cfg, text="")
        self._llm_prompt_label.pack(side=tk.LEFT, padx=(0, 4))
        self._bind_text(self._llm_prompt_label, "llm.label.prompt")
        self._llm_prompt_combo = ttk.Combobox(
            llm_cfg,
            textvariable=self._llm_prompt_var,
            state="readonly",
            width=14,
            values=tuple(BUILTIN_PROMPTS.keys()),
        )
        self._llm_prompt_combo.pack(side=tk.LEFT, padx=(0, 6))
        self._bind_tooltip(self._llm_prompt_combo, "tooltip.llm.prompt")
        self._llm_prompt_btn = ttk.Button(llm_cfg, text="", command=self._open_prompt_editor)
        self._llm_prompt_btn.pack(side=tk.LEFT, padx=(0, 10))
        self._bind_text(self._llm_prompt_btn, "file.button.prompt")
        self._bind_tooltip(self._llm_prompt_btn, "tooltip.llm.prompt_edit")
        self._llm_summarize_btn = ttk.Button(
            llm_cfg, text="", command=self._start_llm_summarize, style="Accent.TButton"
        )
        self._llm_summarize_btn.pack(side=tk.LEFT, padx=(0, 4))
        self._bind_text(self._llm_summarize_btn, "file.button.send_to_llm")
        self._bind_tooltip(self._llm_summarize_btn, "tooltip.llm.send")
        self._llm_copy_btn = ttk.Button(llm_cfg, text="", command=self._copy_llm_output)
        self._llm_copy_btn.pack(side=tk.LEFT, padx=(0, 4))
        self._bind_text(self._llm_copy_btn, "file.button.copy")
        self._bind_tooltip(self._llm_copy_btn, "tooltip.llm.copy")
        self._llm_clear_btn = ttk.Button(llm_cfg, text="", command=self._clear_llm_output)
        self._llm_clear_btn.pack(side=tk.LEFT)
        self._bind_text(self._llm_clear_btn, "file.button.clear")
        self._bind_tooltip(self._llm_clear_btn, "tooltip.llm.clear")

        self._llm_status_label = ttk.Label(llm_cfg, text="", anchor="w", foreground="#555555")
        self._llm_status_label.pack(side=tk.LEFT, padx=(12, 0))
        self._bind_tooltip(self._llm_status_label, "tooltip.llm.status")

        llm_ctx_cfg = ttk.Frame(llm_box)
        llm_ctx_cfg.pack(fill=tk.X, pady=(0, 2))
        self._llm_context_label = ttk.Label(llm_ctx_cfg, text="")
        self._llm_context_label.pack(side=tk.LEFT, padx=(0, 4))
        self._bind_text(self._llm_context_label, "llm.label.context")
        self._llm_context_entry = ttk.Entry(
            llm_ctx_cfg, textvariable=self._llm_context_var, width=8
        )
        self._llm_context_entry.pack(side=tk.LEFT, padx=(0, 8))
        self._bind_tooltip(self._llm_context_entry, "tooltip.llm.context")
        self._llm_context_hint_label = ttk.Label(
            llm_ctx_cfg, text="", anchor="w", foreground="#666666"
        )
        self._llm_context_hint_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        llm_out_frame = ttk.Frame(llm_box)
        llm_out_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 0))

        self._llm_output = scrolledtext.ScrolledText(
            llm_out_frame,
            height=10,
            wrap=tk.WORD,
            state=tk.DISABLED,
        )
        self._llm_output.pack(fill=tk.BOTH, expand=True)
        self._bind_tooltip(self._llm_output, "tooltip.llm.output")
        self._refresh_file_diarization_controls()
        self._refresh_file_workflow()
        self._refresh_llm_context_hint()

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
                self._tr(
                    "live.status.file_only_model_action",
                    model_name=model_info.name,
                )
            )
            return

        options = CaptureOptions(
            model=model_info.id,
            language=self._language_code_for_label(
                self._language_var.get(),
                self._model_var.get() or "small",
            ),
            translate=(self._translate_var.get().strip() or None),
            microphone_device_id=self._selected_microphone_id,
            system_device_id=self._selected_system_id,
        )
        if options.translate and not model_info.supports_translation:
            self._set_live_status(
                self._tr(
                    "live.status.translate_unsupported",
                    model_name=model_info.name,
                )
            )
            return
        capture_source = _derive_capture_source(
            self._selected_microphone_id,
            self._selected_system_id,
        )
        if capture_source == "none":
            self._set_live_status(self._tr("live.status.select_device_capture"))
            return

        log.info(
            "gui.live_capture_requested",
            model=model_info.id,
            language=options.language,
            translate=options.translate,
            source=capture_source,
            microphone_device_id=self._selected_microphone_id,
            system_device_id=self._selected_system_id,
        )

        self._set_live_controls_enabled(False)
        self.stop_button.configure(state=tk.NORMAL)
        self.pause_button.configure(state=tk.DISABLED)
        self._set_live_status(self._tr("live.status.starting"))
        self._worker = CaptureWorker(
            options=options,
            on_status=self._schedule_live_status,
            on_segment=self._schedule_segment,
            on_replace_segments=self._schedule_replace_segments,
            on_error=self._schedule_error,
            on_finished=self._schedule_finished,
            on_drop=self._schedule_drop,
        )
        self._worker.start()

    def _start_recording(self) -> None:
        if self._worker is not None or self._record_worker is not None:
            return

        source = _derive_capture_source(
            self._selected_microphone_id,
            self._selected_system_id,
        )
        if source == "none":
            self._set_live_status(self._tr("live.status.select_device_record"))
            return
        fmt = self._rec_format_var.get()
        ext, filetypes = self._recording_save_filetypes(fmt)
        default_name = f"recording_{source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}"
        path = filedialog.asksaveasfilename(
            defaultextension=f".{ext}",
            initialfile=default_name,
            filetypes=[*filetypes, (self._tr("dialog.filetype.all_files"), "*.*")],
            title=self._tr("dialog.title.save_recorded_audio"),
        )
        if not path:
            return

        options = RecordingOptions(
            microphone_device_id=self._selected_microphone_id,
            system_device_id=self._selected_system_id,
            output_path=Path(path),
            output_format=fmt,
        )

        self._set_live_controls_enabled(False)
        self.stop_button.configure(state=tk.NORMAL)
        self.pause_button.configure(state=tk.NORMAL)
        self.pause_button.configure(text=self._tr("live.button.pause"))
        self.record_button.configure(state=tk.NORMAL, text="00:00:00")
        self._set_live_status(
            self._tr(
                "live.status.recording_to",
                file_name=options.output_path.name,
            )
        )
        self._record_worker = RecordingWorker(
            options=options,
            on_status=self._schedule_live_status,
            on_error=self._schedule_error,
            on_finished=self._schedule_recording_finished,
        )
        self._record_worker.start()
        self.root.after(500, self._tick_recording_timer)

    def _stop_capture(self) -> None:
        if self._worker is not None:
            self._set_live_status(self._tr("live.status.stopping"))
            self._worker.stop()
            self.stop_button.configure(state=tk.DISABLED)
        elif self._record_worker is not None:
            self._set_live_status(self._tr("live.status.stopping_recording"))
            self._record_worker.stop()
            self.stop_button.configure(state=tk.DISABLED)
            self.pause_button.configure(state=tk.DISABLED)

    def _toggle_recording_pause(self) -> None:
        if self._record_worker is None:
            return
        paused = self._record_worker.toggle_pause()
        self.pause_button.configure(
            text=self._tr("live.button.resume" if paused else "live.button.pause")
        )

    def _tick_recording_timer(self) -> None:
        if self._record_worker is None or not self._record_worker.is_running:
            return
        elapsed = self._record_worker.elapsed_s
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        time_str = f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
        if self._record_worker._recorder.is_paused:
            time_str += " ⏸"
        self.record_button.configure(text=time_str)
        self.root.after(500, self._tick_recording_timer)

    def _on_close(self) -> None:
        # Signal all background workers to stop before destroying the window.
        self._stop_capture()
        if self._file_worker is not None:
            self._file_worker.cancel()
        self._persist_gui_settings()
        self._restore_redirection()
        self.root.destroy()
        # Python's ThreadPoolExecutor registers an atexit handler that calls
        # shutdown(wait=True), blocking until in-flight model loading or
        # inference tasks finish — which can take minutes.  Force-exit the
        # process immediately after saving settings to avoid this hang.
        os._exit(0)

    def _schedule_live_status(self, status: str) -> None:
        with suppress(tk.TclError, RuntimeError):
            self.root.after(0, self._set_live_status, status)

    def _schedule_segment(
        self, time_str: str, speaker: str, text: str, translation: str | None
    ) -> None:
        with suppress(tk.TclError, RuntimeError):
            self.root.after(0, self._add_segment, time_str, speaker, text, translation)

    def _schedule_replace_segments(self, rows: list[tuple[str, str, str, str | None]]) -> None:
        with suppress(tk.TclError, RuntimeError):
            self.root.after(0, self._replace_segments, rows)

    def _schedule_error(self, message: str) -> None:
        with suppress(tk.TclError, RuntimeError):
            self.root.after(0, self._show_error, message)

    def _schedule_finished(self) -> None:
        with suppress(tk.TclError, RuntimeError):
            self.root.after(0, self._on_worker_finished)

    def _schedule_recording_finished(self, stats: RecordingStats | None) -> None:
        with suppress(tk.TclError, RuntimeError):
            self.root.after(0, self._on_recording_finished, stats)

    def _refresh_live_summary_labels(self) -> None:
        if hasattr(self, "counter_label"):
            self.counter_label.configure(
                text=self._tr("live.summary.segments", count=self._segment_count)
            )
        if hasattr(self, "queue_label") and self._worker is None:
            self.queue_label.configure(text=self._tr("live.summary.queue_empty"))

    def _set_live_status(self, status: str) -> None:
        self.status_label.configure(text=status)

    def _show_error(self, message: str) -> None:
        self._set_live_status(self._tr("live.status.error", message=message))
        log.error("gui.live_error", error=message)

    def _add_segment(self, time_str: str, speaker: str, text: str, translation: str | None) -> None:
        self._segment_count += 1
        self.counter_label.configure(
            text=self._tr("live.summary.segments", count=self._segment_count)
        )
        source_label = (
            self._tr("live.source.mic")
            if "LOCAL" in speaker
            else self._tr("live.source.system")
            if "REMOTE" in speaker
            else speaker
        )
        text_lines = textwrap.wrap(text, width=70) if text else [""]
        trans_lines = textwrap.wrap(translation or "", width=57) if translation else [""]
        n = max(len(text_lines), len(trans_lines))
        while len(text_lines) < n:
            text_lines.append("")
        while len(trans_lines) < n:
            trans_lines.append("")
        for i, (tl, tr) in enumerate(zip(text_lines, trans_lines, strict=False)):
            tv = time_str if i == 0 else ""
            sv = source_label if i == 0 else ""
            tags: tuple[str, ...] = () if i == 0 else ("continuation",)
            self.table.insert("", tk.END, values=(tv, sv, tl, tr), tags=tags)
        self.table.yview_moveto(1.0)

    def _add_dropped_row(self, time_str: str, source: str) -> None:
        src_label = (
            self._tr("live.source.mic")
            if source == "microphone"
            else self._tr("live.source.sys_short")
            if source == "system"
            else source
        )
        self.table.insert(
            "",
            tk.END,
            values=(time_str, src_label, self._tr("live.dropped.text"), ""),
            tags=("dropped",),
        )
        self.table.yview_moveto(1.0)

    def _schedule_drop(self, time_str: str, source: str) -> None:
        with suppress(tk.TclError, RuntimeError):
            self.root.after(0, self._add_dropped_row, time_str, source)

    def _replace_segments(self, rows: list[tuple[str, str, str, str | None]]) -> None:
        self._clear_table()
        for row in rows:
            self._add_segment(*row)

    def _poll_stats(self) -> None:
        if self._worker is not None:
            stats = self._worker.get_stats()
            if stats is not None:
                q = stats["preprocess_q"] + stats["asr_q"]
                in_asr = stats["in_asr"]
                dropped = stats["dropped"]
                self.queue_label.configure(
                    text=self._tr(
                        "live.summary.queue",
                        queue=q,
                        in_asr=in_asr,
                        dropped=dropped,
                    )
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
        lines = self._format_items(item for item in self.table.get_children() if item in selected)
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
        self.counter_label.configure(text=self._tr("live.summary.segments", count=0))

    def _save_to_file(self) -> None:
        children = self.table.get_children()
        if not children:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[
                (self._tr("dialog.filetype.text_file"), "*.txt"),
                (self._tr("dialog.filetype.csv_file"), "*.csv"),
                (self._tr("dialog.filetype.all_files"), "*.*"),
            ],
            title=self._tr("dialog.title.save_live_transcription"),
        )
        if not path:
            return
        lines = self._format_items(children)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
            self._set_live_status(self._tr("live.status.saved", count=len(lines), path=path))
        except OSError as exc:
            self._set_live_status(self._tr("live.status.save_failed", error=exc))

    def _on_worker_finished(self) -> None:
        self._worker = None
        self.queue_label.configure(text=self._tr("live.summary.queue_empty"))
        self._set_live_controls_enabled(True)
        self.stop_button.configure(state=tk.DISABLED)
        self.pause_button.configure(state=tk.DISABLED)
        self.pause_button.configure(text=self._tr("live.button.pause"))

    def _on_recording_finished(self, stats: RecordingStats | None) -> None:
        self._record_worker = None
        self._set_live_controls_enabled(True)
        self.stop_button.configure(state=tk.DISABLED)
        self.pause_button.configure(state=tk.DISABLED)
        self.pause_button.configure(text=self._tr("live.button.pause"))
        self.record_button.configure(text=self._tr("live.button.record_audio"))
        if stats is None:
            self._refresh_file_workflow()
            return
        self._last_recorded_file = stats.output_path
        self._persist_gui_settings()
        self._set_live_status(
            self._tr(
                "live.status.recorded",
                duration=stats.duration_s,
                file_name=stats.output_path.name,
            )
        )
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
        self.model_combo.configure(state=combo_state)
        self.language_combo.configure(state=combo_state)
        self.translate_entry.configure(state=(tk.NORMAL if enabled else tk.DISABLED))
        self.device_picker.configure(state=("normal" if enabled else "disabled"))

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
                self._tr(
                    "live.status.file_only_model",
                    model_name=live_model_info.name,
                )
            )

        file_model = get_model_info(self._file_model_var.get()).id
        file_values = [language.label for language in list_languages_for_model(file_model)]
        current_file = self._language_code_for_label(self._file_lang_var.get(), file_model)
        self._file_lang_combo.configure(values=file_values)
        self._file_lang_var.set(self._language_label_for_code(current_file, file_model))
        self._file_model_summary.set_model(file_model)
        self._refresh_file_diarization_controls()

    def _on_model_changed(self, _event: object | None = None) -> None:
        self._model_var.set(get_model_info(self._model_var.get()).id)
        self._refresh_language_choices()
        self._persist_gui_settings()

    def _on_live_language_changed(self, _event: object | None = None) -> None:
        live_model = get_model_info(self._model_var.get()).id
        current_live = self._language_code_for_label(self._language_var.get(), live_model)
        self._language_var.set(self._language_label_for_code(current_live, live_model))
        self._persist_gui_settings()

    def _on_file_model_changed(self, _event: object | None = None) -> None:
        self._file_model_var.set(get_model_info(self._file_model_var.get()).id)
        self._refresh_language_choices()

    def _on_file_diarization_changed(self, _event: object | None = None) -> None:
        strategy = (self._file_diarization_var.get() or "auto").strip().lower()
        if strategy not in FILE_DIARIZATION_CHOICES:
            strategy = "auto"
        self._file_diarization_var.set(strategy)
        self._refresh_file_diarization_controls()

    def _on_speaker_preset_changed(self, _event: object | None = None) -> None:
        display_value = (
            self._file_speaker_preset_display_var.get()
            if hasattr(self, "_file_speaker_preset_display_var")
            else self._file_speaker_preset_var.get()
        )
        preset = self._normalize_speaker_preset(display_value)
        self._file_speaker_preset_var.set(preset)
        if hasattr(self, "_file_speaker_preset_display_var"):
            self._file_speaker_preset_display_var.set(self._speaker_preset_label(preset))
        preset_map = {
            "1": ("1", "1"),
            "2": ("2", "2"),
            "3": ("3", "3"),
            "4plus": ("4", ""),
        }
        if preset in preset_map:
            lo, hi = preset_map[preset]
            self._file_min_speakers_var.set(lo)
            self._file_max_speakers_var.set(hi)
        elif preset == "auto":
            self._file_min_speakers_var.set("")
            self._file_max_speakers_var.set("")
        self._refresh_file_diarization_controls()

    def _refresh_file_diarization_controls(self) -> None:
        strategy = (self._file_diarization_var.get() or "auto").strip().lower()
        if strategy not in FILE_DIARIZATION_CHOICES:
            strategy = "auto"
            self._file_diarization_var.set(strategy)

        running = self._file_worker is not None
        combo_state = "disabled" if running else "readonly"
        ml_relevant = strategy in {"auto", "ml", "hybrid"}
        preset_state = combo_state if ml_relevant else "disabled"

        # Min/Max entries stay editable for ML-capable strategies.
        # Presets are shortcuts that fill values, not hard locks.
        entry_state = "normal" if (not running and ml_relevant) else "disabled"

        self._file_diarization_combo.configure(state=combo_state)
        self._file_speaker_preset_combo.configure(state=preset_state)
        self._file_min_speakers_entry.configure(state=entry_state)
        self._file_max_speakers_entry.configure(state=entry_state)

        # Detect button: enabled only when ML is possible and file is selected
        has_file = bool(self._file_path_var.get().strip())
        detect_state = "normal" if (not running and ml_relevant and has_file) else "disabled"
        self._file_detect_btn.configure(state=detect_state)

    def _detect_speakers(self) -> None:
        """Run fast speaker count estimation on a sample of the selected file."""
        raw_path = self._file_path_var.get().strip()
        if not raw_path:
            self._file_status_label.configure(text=self._tr("file.status.select_file"))
            return

        file_path = Path(raw_path)
        if not file_path.exists():
            self._file_status_label.configure(
                text=self._tr("file.status.file_not_found", file_name=file_path.name),
                foreground="red",
            )
            return

        hf_token = self._hf_token_var.get().strip() or (
            __import__("os").environ.get("HF_TOKEN")
            or __import__("os").environ.get("HUGGING_FACE_HUB_TOKEN")
        )
        if not hf_token:
            self._file_status_label.configure(
                text=self._tr("file.status.hf_token_required"),
                foreground="#BB6600",
            )
            return

        self._file_detect_btn.configure(state=tk.DISABLED, text=self._tr("file.button.detecting"))
        self._file_status_label.configure(
            text=self._tr("file.status.detecting_speakers"), foreground=""
        )

        def _run() -> None:
            import asyncio

            try:
                from voxfusion.diarization.speaker_counter import estimate_speaker_count

                log.info(
                    "gui.speaker_detect_started",
                    file=str(file_path),
                    max_sample_duration_s=120.0,
                )

                async def _async_detect() -> int:
                    audio = load_detection_audio_chunk(file_path, max_duration_s=120.0)
                    if audio.samples.size == 0:
                        return 0
                    return await estimate_speaker_count(
                        audio,
                        hf_token=hf_token,
                    )

                count = asyncio.run(_async_detect())
            except Exception as exc:
                log.error(
                    "gui.speaker_detect_failed",
                    file=str(file_path),
                    error=str(exc),
                )
                self.root.after(0, self._on_detect_done, None, str(exc))
                return
            log.info(
                "gui.speaker_detect_completed",
                file=str(file_path),
                detected_speakers=count,
            )
            self.root.after(0, self._on_detect_done, count, None)

        self._file_detect_worker_thread = threading.Thread(target=_run, daemon=True)
        self._file_detect_worker_thread.start()

    def _on_detect_done(self, count: int | None, error: str | None) -> None:
        """Called on the Tk thread when speaker detection finishes."""
        self._file_detect_btn.configure(text=self._tr("file.button.detect"))
        self._refresh_file_diarization_controls()
        if error:
            self._file_status_label.configure(
                text=self._tr("file.status.detect_failed", error=error), foreground="red"
            )
            return
        if count is None or count == 0:
            self._file_status_label.configure(
                text=self._tr("file.status.detect_empty"), foreground="#BB6600"
            )
            return

        # Preserve detected exact counts for 4+ speakers instead of degrading them
        # into the open-ended "4+" preset.
        preset_map = {1: "1", 2: "2", 3: "3"}
        preset = preset_map.get(count)
        if preset is not None:
            self._file_speaker_preset_var.set(preset)
            if hasattr(self, "_file_speaker_preset_display_var"):
                self._file_speaker_preset_display_var.set(self._speaker_preset_label(preset))
            self._on_speaker_preset_changed()
        else:
            exact_count = str(count)
            self._file_speaker_preset_var.set("custom")
            if hasattr(self, "_file_speaker_preset_display_var"):
                self._file_speaker_preset_display_var.set(self._speaker_preset_label("custom"))
            self._file_min_speakers_var.set(exact_count)
            self._file_max_speakers_var.set(exact_count)
        self._file_status_label.configure(
            text=self._tr(
                "file.status.detected_speakers",
                count=count,
                s_suffix=("s" if count != 1 else ""),
            ),
            foreground="",
        )

    @staticmethod
    def _parse_optional_positive_int(raw: str, field_name: str) -> int | None:
        value = raw.strip()
        if not value:
            return None
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a whole number.") from exc
        if parsed < 1:
            raise ValueError(f"{field_name} must be at least 1.")
        return parsed

    def _parse_optional_positive_int_localized(self, raw: str, field_name: str) -> int | None:
        value = raw.strip()
        if not value:
            return None
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(
                self._tr("validation.error.whole_number", field_name=field_name)
            ) from exc
        if parsed < 1:
            raise ValueError(self._tr("validation.error.at_least_one", field_name=field_name))
        return parsed

    def _append_log_line(self, text: str) -> None:
        self.log_widget.configure(state=tk.NORMAL)
        self.log_widget.insert(tk.END, text)
        self.log_widget.see(tk.END)
        self.log_widget.configure(state=tk.DISABLED)

    def _current_log_mode(self) -> str:
        return self._log_mode_code

    def _apply_gui_log_mode(self) -> None:
        self._refresh_log_mode_selector()
        level = logging.DEBUG if self._current_log_mode() == "debug" else logging.INFO
        configure_gui_logging(level=level, log_mode=self._current_log_mode())

    def _on_log_mode_changed(self, _event: object | None = None) -> None:
        self._log_mode_code = self._log_mode_code_from_label(self._log_mode_var.get())
        self._apply_gui_log_mode()
        self._persist_gui_settings()

    def _apply_saved_gui_settings(self) -> None:
        settings = _load_gui_settings()
        self._ui_language_code, self._ui_language_explicit = resolve_initial_gui_language(
            settings.get("gui_language", DEFAULT_GUI_LANGUAGE),
            settings.get("gui_language_explicit", ""),
        )
        self._locale = load_gui_locale(self._ui_language_code)
        self._log_mode_code = (
            "debug"
            if str(settings.get("gui_log_mode", "normal")).strip().lower() == "debug"
            else "normal"
        )
        self._llm_url_var.set(settings.get("llm_url", DEFAULT_BASE_URL))
        self._llm_model_var.set(settings.get("llm_model", DEFAULT_MODEL))
        self._llm_key_var.set(settings.get("llm_api_key", ""))
        self._llm_prompt_var.set(settings.get("llm_prompt", "summarize"))
        self._llm_context_var.set(settings.get("llm_context_tokens_override", "").strip())
        self._llm_custom_user_prompt = settings.get("llm_custom_user_prompt", "")
        self._cached_llm_models = self._load_cached_llm_models(settings)
        self._cached_llm_model_contexts = self._load_cached_llm_model_contexts(settings)
        if self._cached_llm_models:
            self._available_llm_models = list(self._cached_llm_models)
            self._llm_model_contexts = dict(self._cached_llm_model_contexts)

        # Proxy settings
        self._proxy_use_system_var.set(settings.get("proxy_use_system", "true").lower() != "false")
        self._proxy_http_var.set(settings.get("proxy_http", ""))
        self._proxy_https_var.set(settings.get("proxy_https", ""))
        self._proxy_no_var.set(settings.get("proxy_no", ""))
        self._proxy_ca_var.set(settings.get("proxy_ca_bundle", ""))
        apply_proxy_settings(settings)

        # HuggingFace token
        token = settings.get("hf_token", "")
        self._hf_token_var.set(token)
        if token:
            import os

            os.environ["HF_TOKEN"] = token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token

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

        self._file_quality_var.set(
            self._normalize_quality_label(settings.get("file_quality", "Balanced"))
        )

        saved_diarization = settings.get("file_diarization_strategy", "auto").strip().lower()
        if saved_diarization in FILE_DIARIZATION_CHOICES:
            self._file_diarization_var.set(saved_diarization)
        self._file_min_speakers_var.set(settings.get("file_min_speakers", "").strip())
        self._file_max_speakers_var.set(settings.get("file_max_speakers", "").strip())

        self._file_speaker_preset_var.set(
            self._normalize_speaker_preset(settings.get("file_speaker_preset", "auto"))
        )

        saved_live_model = settings.get("live_model", "")
        _avail = {m.id for m in get_available_model_catalog()}
        if saved_live_model and saved_live_model in _avail:
            self._model_var.set(saved_live_model)
            saved_live_lang = settings.get("live_language", "")
            if saved_live_lang:
                self._language_var.set(saved_live_lang)

        saved_file_model = settings.get("file_model", "")
        if saved_file_model and saved_file_model in _avail:
            self._file_model_var.set(saved_file_model)
            saved_file_lang = settings.get("file_language", "")
            if saved_file_lang:
                self._file_lang_var.set(saved_file_lang)

    def _persist_gui_settings(self) -> None:
        _save_gui_settings(
            {
                "llm_url": self._llm_url_var.get().strip(),
                "llm_model": self._llm_model_var.get().strip(),
                "llm_api_key": self._llm_key_var.get(),
                "llm_prompt": self._llm_prompt_var.get().strip() or "summarize",
                "llm_context_tokens_override": self._llm_context_var.get().strip(),
                "llm_custom_user_prompt": self._llm_custom_user_prompt,
                _LLM_MODELS_CACHE_KEY: json.dumps(self._cached_llm_models, ensure_ascii=False),
                _LLM_MODEL_CONTEXT_CACHE_KEY: json.dumps(
                    self._cached_llm_model_contexts, ensure_ascii=False
                ),
                "gui_language": self._ui_language_code,
                "gui_language_explicit": "true" if self._ui_language_explicit else "false",
                "gui_log_mode": self._current_log_mode(),
                "last_recorded_file": str(self._last_recorded_file)
                if self._last_recorded_file
                else "",
                "last_transcript_path": str(self._last_transcript_path)
                if self._last_transcript_path
                else "",
                # Proxy
                "proxy_use_system": "true" if self._proxy_use_system_var.get() else "false",
                "proxy_http": self._proxy_http_var.get().strip(),
                "proxy_https": self._proxy_https_var.get().strip(),
                "proxy_no": self._proxy_no_var.get().strip(),
                "proxy_ca_bundle": self._proxy_ca_var.get().strip(),
                # HuggingFace
                "hf_token": self._hf_token_var.get().strip(),
                # Live transcription model/language
                "live_model": self._model_var.get().strip(),
                "live_language": self._language_var.get().strip(),
                # Transcription quality
                "file_quality": self._file_quality_var.get(),
                "file_diarization_strategy": self._file_diarization_var.get().strip(),
                "file_min_speakers": self._file_min_speakers_var.get().strip(),
                "file_max_speakers": self._file_max_speakers_var.get().strip(),
                "file_speaker_preset": self._file_speaker_preset_var.get(),
                # File transcription model/language
                "file_model": self._file_model_var.get().strip(),
                "file_language": self._file_lang_var.get().strip(),
            }
        )

    def _load_recorded_file_into_flow(self, path: Path, *, switch_tab: bool) -> None:
        self._last_recorded_file = path
        self._replace_file_queue([path])
        self._clear_file_table()
        self._last_transcript_path = None
        self._file_status_label.configure(
            text=self._tr("file.workflow.step2_ready_recorded", file_name=path.name)
        )
        if switch_tab:
            self._notebook.select(1)
        self._refresh_file_workflow()

    @staticmethod
    def _file_queue_key(path: Path) -> str:
        with suppress(OSError):
            return str(path.resolve()).casefold()
        return str(path).casefold()

    @staticmethod
    def _format_queue_result(text: str, *, limit: int = 56) -> str:
        clean = " ".join(str(text).split())
        if len(clean) <= limit:
            return clean
        return clean[: limit - 1].rstrip() + "…"

    @staticmethod
    def _format_queue_duration(duration_s: float | None) -> str:
        if duration_s is None or duration_s < 0:
            return "—"
        total_seconds = max(0, int(round(duration_s)))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    @staticmethod
    def _format_queue_size(size_bytes: int | None) -> str:
        if size_bytes is None or size_bytes < 0:
            return "—"
        value = float(size_bytes)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if value < 1024.0 or unit == "TB":
                if unit == "B":
                    return f"{int(value)} {unit}"
                return f"{value:.1f} {unit}"
            value /= 1024.0
        return "—"

    def _file_queue_values(self, item: _FileQueueItem) -> tuple[str, str, str, str, str, str]:
        progress = "—"
        if item.status != "Queued":
            progress = f"{int(max(0.0, min(item.progress, 1.0)) * 100):d}%"
        return (
            str(item.file_path),
            self._format_queue_duration(item.duration_s),
            self._format_queue_size(item.size_bytes),
            self._tr(f"file.queue.status.{item.status}"),
            progress,
            self._format_queue_result(item.result) if item.result else "",
        )

    def _update_file_queue_row(self, item_id: str) -> None:
        item = self._file_queue_items[item_id]
        self._file_queue_table.item(item_id, values=self._file_queue_values(item))

    def _refresh_file_queue_rows(self) -> None:
        if not hasattr(self, "_file_queue_table"):
            return
        for item_id in list(self._file_queue_items):
            if item_id in self._file_queue_items:
                self._update_file_queue_row(item_id)

    def _schedule_queue_metadata_probe(self, item_id: str, item: _FileQueueItem) -> None:
        if not getattr(self, "_queue_metadata_async_enabled", False):
            return

        generation = item.metadata_generation
        path = item.file_path

        def _run() -> tuple[float | None, int | None]:
            try:
                return probe_media_metadata(path)
            except Exception:
                return None, item.size_bytes

        future = self._queue_metadata_executor.submit(_run)

        def _on_done(_future: object) -> None:
            try:
                duration_s, size_bytes = future.result()
            except Exception:
                duration_s, size_bytes = None, item.size_bytes
            with suppress(tk.TclError, RuntimeError):
                self.root.after(
                    0,
                    self._apply_file_queue_metadata,
                    item_id,
                    generation,
                    duration_s,
                    size_bytes,
                )

        future.add_done_callback(_on_done)

    def _apply_file_queue_metadata(
        self,
        item_id: str,
        generation: int,
        duration_s: float | None,
        size_bytes: int | None,
    ) -> None:
        item = self._file_queue_items.get(item_id)
        if item is None or item.metadata_generation != generation:
            return
        item.duration_s = duration_s
        if size_bytes is not None:
            item.size_bytes = size_bytes
        self._update_file_queue_row(item_id)

    def _select_file_queue_item(self, item_id: str | None) -> None:
        if item_id is None:
            self._file_queue_table.selection_remove(self._file_queue_table.selection())
            return
        self._file_queue_table.selection_set(item_id)
        self._file_queue_table.focus(item_id)
        self._file_queue_table.see(item_id)
        self._file_path_var.set(str(self._file_queue_items[item_id].file_path))

    def _replace_file_queue(self, paths: list[Path]) -> None:
        self._file_queue_generation += 1
        for item_id in list(self._file_queue_items):
            self._file_queue_table.delete(item_id)
        self._file_queue_items.clear()
        self._file_queue_lookup.clear()
        self._file_active_queue_id = None
        self._file_queue_serial = 0
        self._file_path_var.set("")
        self._add_files_to_queue(paths)

    def _add_files_to_queue(self, paths: list[Path]) -> int:
        added = 0
        first_added_id: str | None = None
        for raw_path in paths:
            path = Path(raw_path).expanduser()
            key = self._file_queue_key(path)
            if key in self._file_queue_lookup:
                continue
            self._file_queue_serial += 1
            item_id = f"file-{self._file_queue_serial}"
            if getattr(self, "_queue_metadata_async_enabled", False):
                duration_s = None
                size_bytes = probe_media_size(path)
            else:
                duration_s, size_bytes = probe_media_metadata(path)
            item = _FileQueueItem(
                file_path=path,
                duration_s=duration_s,
                size_bytes=size_bytes,
                metadata_generation=getattr(self, "_file_queue_generation", 0),
            )
            self._file_queue_items[item_id] = item
            self._file_queue_lookup[key] = item_id
            self._file_queue_table.insert(
                "", tk.END, iid=item_id, values=self._file_queue_values(item)
            )
            if getattr(self, "_queue_metadata_async_enabled", False):
                self._schedule_queue_metadata_probe(item_id, item)
            if first_added_id is None:
                first_added_id = item_id
            added += 1
        if first_added_id is not None and not self._file_path_var.get().strip():
            self._select_file_queue_item(first_added_id)
        self._refresh_file_workflow()
        return added

    def _next_pending_file_queue_id(self) -> str | None:
        for item_id in self._file_queue_table.get_children():
            item = self._file_queue_items[item_id]
            if item.status == "Queued":
                return item_id
        return None

    def _on_file_queue_selection(self, _event: object | None = None) -> None:
        selected = self._file_queue_table.selection()
        if not selected:
            return
        item_id = selected[0]
        self._file_path_var.set(str(self._file_queue_items[item_id].file_path))
        self._refresh_file_workflow()

    def _remove_selected_file_queue_items(self) -> None:
        if self._file_worker is not None:
            return
        selected = list(self._file_queue_table.selection())
        if not selected:
            return
        current_selection = None
        for item_id in selected:
            item = self._file_queue_items.pop(item_id, None)
            if item is None:
                continue
            self._file_queue_lookup.pop(self._file_queue_key(item.file_path), None)
            self._file_queue_table.delete(item_id)
        children = self._file_queue_table.get_children()
        if children:
            current_selection = children[0]
        self._select_file_queue_item(current_selection)
        if current_selection is None:
            self._file_path_var.set("")
        self._refresh_file_workflow()

    def _clear_file_queue(self) -> None:
        if self._file_worker is not None:
            return
        self._replace_file_queue([])
        self._file_status_label.configure(text=self._tr("file.status.cleared_list"))
        self._refresh_file_workflow()

    def _refresh_file_workflow(self) -> None:
        transcript_ready = self._file_seg_count > 0
        has_queued_items = any(item.status == "Queued" for item in self._file_queue_items.values())
        if has_queued_items and not transcript_ready and self._file_worker is None:
            workflow_text = self._tr(
                "file.workflow.step2_queue",
                count=len(self._file_queue_items),
            )
        elif transcript_ready:
            workflow_text = self._tr("file.workflow.step3")
        elif self._last_recorded_file is not None:
            workflow_text = self._tr(
                "file.workflow.step2_recorded",
                file_name=self._last_recorded_file.name,
            )
        else:
            workflow_text = self._tr("file.workflow.step1")
        self._file_workflow_label.configure(text=workflow_text)
        llm_enabled = (
            transcript_ready
            and self._llm_worker is None
            and self._file_worker is None
            and not self._llm_preflight_running
        )
        self._llm_summarize_btn.configure(state=(tk.NORMAL if llm_enabled else tk.DISABLED))
        llm_probe_enabled = (
            self._llm_worker is None
            and self._file_worker is None
            and not self._llm_model_refreshing
            and not self._llm_probe_running
            and not self._llm_preflight_running
        )
        if hasattr(self, "_llm_probe_btn"):
            self._llm_probe_btn.configure(state=(tk.NORMAL if llm_probe_enabled else tk.DISABLED))
        if self._last_transcript_path is not None and self._last_transcript_path.exists():
            self._file_artifact_label.configure(
                text=self._tr("file.label.transcript_file", path=self._last_transcript_path)
            )
        else:
            self._file_artifact_label.configure(text=self._tr("file.label.transcript_missing"))

    @staticmethod
    def _parse_llm_context_tokens(value: object) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            parsed = value
        elif isinstance(value, float) and value.is_integer():
            parsed = int(value)
        else:
            text = str(value).strip()
            if not text.isdigit():
                return None
            parsed = int(text)
        return parsed if parsed >= _LLM_MIN_CONTEXT_TOKENS else None

    @staticmethod
    def _load_cached_llm_models(settings: dict[str, str]) -> list[str]:
        raw = settings.get(_LLM_MODELS_CACHE_KEY, "").strip()
        if not raw:
            return []
        try:
            payload = json.loads(raw)
        except ValueError:
            return []
        if not isinstance(payload, list):
            return []
        models = [str(item).strip() for item in payload if str(item).strip()]
        return list(dict.fromkeys(models))

    @classmethod
    def _load_cached_llm_model_contexts(cls, settings: dict[str, str]) -> dict[str, int]:
        raw = settings.get(_LLM_MODEL_CONTEXT_CACHE_KEY, "").strip()
        if not raw:
            return {}
        try:
            payload = json.loads(raw)
        except ValueError:
            return {}
        if not isinstance(payload, dict):
            return {}
        contexts: dict[str, int] = {}
        for key, value in payload.items():
            model_id = str(key).strip()
            parsed = cls._parse_llm_context_tokens(value)
            if model_id and parsed is not None:
                contexts[model_id] = parsed
        return contexts

    @staticmethod
    def _model_context_map(model_descriptors: list[LLMModelDescriptor]) -> dict[str, int]:
        contexts: dict[str, int] = {}
        for descriptor in model_descriptors:
            if descriptor.context_tokens is not None:
                contexts[descriptor.id] = descriptor.context_tokens
        return contexts

    def _fallback_llm_context_limit(self) -> tuple[int, str]:
        raw = os.environ.get(_LLM_CONTEXT_TOKEN_ENV, "").strip()
        if raw.isdigit():
            return max(_LLM_MIN_CONTEXT_TOKENS, int(raw)), "env"
        return _LLM_DEFAULT_CONTEXT_TOKENS, "default"

    def _resolve_llm_context_limit(self) -> tuple[int, str]:
        manual_raw = self._llm_context_var.get().strip()
        if manual_raw:
            manual = self._parse_llm_context_tokens(manual_raw)
            if manual is None:
                raise ValueError(self._tr("llm.status.invalid_context"))
            return manual, "manual"
        model = self._llm_model_var.get().strip() or DEFAULT_MODEL
        detected = self._llm_model_contexts.get(model)
        if detected is not None:
            return detected, "model_metadata"
        return self._fallback_llm_context_limit()

    def _refresh_llm_context_hint(self) -> None:
        if not hasattr(self, "_llm_context_hint_label"):
            return
        manual_raw = self._llm_context_var.get().strip()
        if manual_raw:
            manual = self._parse_llm_context_tokens(manual_raw)
            if manual is None:
                text = self._tr("llm.context.invalid")
            else:
                text = self._tr("llm.context.manual", tokens=manual)
        else:
            model = self._llm_model_var.get().strip() or DEFAULT_MODEL
            detected = self._llm_model_contexts.get(model)
            if detected is not None:
                text = self._tr("llm.context.model_metadata", tokens=detected)
            else:
                fallback, source = self._fallback_llm_context_limit()
                key = "llm.context.env" if source == "env" else "llm.context.default"
                text = self._tr(key, tokens=fallback)
        self._llm_context_hint_label.configure(text=text)

    def _apply_llm_model_choices(
        self,
        models: list[str],
        *,
        contexts: dict[str, int] | None = None,
        base_url: str,
        source: str,
    ) -> str:
        self._available_llm_models = list(models)
        self._llm_model_contexts = {
            model_id: tokens for model_id, tokens in (contexts or {}).items() if model_id in models
        }
        self._llm_model_combo.configure(values=models)
        requested_model = self._llm_model_var.get().strip()
        selected_model = requested_model
        if models and requested_model not in models:
            selected_model = models[0]
            self._llm_model_var.set(selected_model)
            log.warning(
                "gui.llm_model_auto_selected",
                base_url=base_url,
                requested_model=requested_model or DEFAULT_MODEL,
                selected_model=selected_model,
                model_count=len(models),
                source=source,
            )
        self._refresh_llm_context_hint()
        return selected_model

    def _refresh_llm_models(self) -> None:
        if self._llm_model_refreshing:
            return
        self._llm_model_refreshing = True
        self._llm_status_label.configure(text=self._tr("file.status.loading_models"))
        self._persist_gui_settings()
        base_url = self._llm_url_var.get().strip() or DEFAULT_BASE_URL
        api_key = self._llm_key_var.get().strip()
        log.info(
            "gui.llm_models_refresh_requested",
            base_url=base_url,
            api_key_present=bool(api_key),
        )
        result_q: queue.Queue[tuple[list[LLMModelDescriptor], str | None]] = queue.Queue()

        def _poll() -> None:
            try:
                models, error = result_q.get_nowait()
            except queue.Empty:
                with suppress(tk.TclError, RuntimeError):
                    self.root.after(100, _poll)
                return
            self._on_llm_models_loaded(models, error)

        def _run() -> None:
            try:
                models = asyncio.run(fetch_model_catalog(base_url=base_url, api_key=api_key))
                result_q.put((models, None))
            except Exception as exc:  # pragma: no cover
                result_q.put(([], str(exc)))

        threading.Thread(target=_run, daemon=True).start()
        self.root.after(100, _poll)

    def _on_llm_models_loaded(self, models: list[LLMModelDescriptor], error: str | None) -> None:
        self._llm_model_refreshing = False
        base_url = self._llm_url_var.get().strip() or DEFAULT_BASE_URL
        if error:
            log.error(
                "gui.llm_models_load_failed",
                base_url=base_url,
                error=error,
            )
            cached_models = list(getattr(self, "_cached_llm_models", []))
            if cached_models:
                selected_model = self._apply_llm_model_choices(
                    cached_models,
                    contexts=getattr(self, "_cached_llm_model_contexts", {}),
                    base_url=base_url,
                    source="cache",
                )
                self._persist_gui_settings()
                log.warning(
                    "gui.llm_models_loaded_from_cache",
                    base_url=base_url,
                    model_count=len(cached_models),
                    selected_model=selected_model,
                    error=error,
                )
                self._llm_status_label.configure(
                    text=self._tr(
                        "file.status.loaded_models_cached",
                        count=len(cached_models),
                        error=error,
                    )
                )
                return
            self._llm_status_label.configure(
                text=self._tr("file.status.model_load_failed", error=error)
            )
            return
        model_ids = [descriptor.id for descriptor in models]
        model_contexts = self._model_context_map(models)
        self._cached_llm_models = list(model_ids)
        self._cached_llm_model_contexts = dict(model_contexts)
        selected_model = self._apply_llm_model_choices(
            model_ids,
            contexts=model_contexts,
            base_url=base_url,
            source="remote",
        )
        self._persist_gui_settings()
        log.info(
            "gui.llm_models_loaded",
            base_url=base_url,
            model_count=len(models),
            selected_model=selected_model,
        )
        self._llm_status_label.configure(
            text=self._tr("file.status.loaded_models", count=len(model_ids))
        )

    def _probe_llm_model(self) -> None:
        if self._llm_probe_running or self._llm_worker is not None or self._llm_preflight_running:
            log.warning("gui.llm_probe_skipped", reason="busy")
            return

        url = self._llm_url_var.get().strip() or DEFAULT_BASE_URL
        model = self._llm_model_var.get().strip() or DEFAULT_MODEL
        api_key = self._llm_key_var.get().strip()
        self._persist_gui_settings()
        self._llm_probe_running = True
        self._llm_status_label.configure(text=self._tr("llm.status.testing_model", model=model))
        self._refresh_file_workflow()
        log.info(
            "gui.llm_probe_requested",
            base_url=url,
            model=model,
            timeout_read=_LLM_PROBE_TIMEOUT_READ,
            api_key_present=bool(api_key),
        )
        result_q: queue.Queue[tuple[bool, str]] = queue.Queue()

        def _poll() -> None:
            try:
                success, detail = result_q.get_nowait()
            except queue.Empty:
                with suppress(tk.TclError, RuntimeError):
                    self.root.after(100, _poll)
                return
            self._on_llm_probe_finished(success, detail, model=model, base_url=url)

        def _run() -> None:
            try:
                reply = asyncio.run(
                    complete(
                        _LLM_PROBE_MESSAGES,
                        base_url=url,
                        model=model,
                        api_key=api_key,
                        timeout_read=_LLM_PROBE_TIMEOUT_READ,
                    )
                )
                result_q.put((True, (reply or "").strip()))
            except Exception as exc:  # pragma: no cover
                result_q.put((False, str(exc)))

        threading.Thread(target=_run, daemon=True).start()
        self.root.after(100, _poll)

    def _on_llm_probe_finished(
        self,
        success: bool,
        detail: str,
        *,
        model: str,
        base_url: str,
    ) -> None:
        self._llm_probe_running = False
        if success:
            preview = " ".join(detail.split())[:80] or "OK"
            log.info(
                "gui.llm_probe_succeeded",
                base_url=base_url,
                model=model,
                response_preview=preview,
            )
            self._llm_status_label.configure(text=self._tr("llm.status.model_ok", model=model))
        else:
            log.error(
                "gui.llm_probe_failed",
                base_url=base_url,
                model=model,
                error=detail,
            )
            self._llm_status_label.configure(
                text=self._tr("llm.status.model_failed", error=detail[:80])
            )
        self._refresh_file_workflow()

    def _open_settings(self) -> None:
        """Open the application settings dialog (proxy / network)."""
        dlg = tk.Toplevel(self.root)
        dlg.title(self._tr("settings.title"))
        dlg.geometry("580x480")
        dlg.resizable(False, False)
        dlg.grab_set()

        # Local copies so changes are only applied on Save
        use_sys = tk.BooleanVar(value=self._proxy_use_system_var.get())
        http_v = tk.StringVar(value=self._proxy_http_var.get())
        https_v = tk.StringVar(value=self._proxy_https_var.get())
        no_v = tk.StringVar(value=self._proxy_no_var.get())
        ca_v = tk.StringVar(value=self._proxy_ca_var.get())
        hf_token_v = tk.StringVar(value=self._hf_token_var.get())

        pad = {"padx": 8, "pady": 3}

        proxy_frame = ttk.LabelFrame(
            dlg, text=self._tr("settings.section.network"), padding=(10, 8)
        )
        proxy_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 4))
        proxy_frame.columnconfigure(1, weight=1)

        ttk.Checkbutton(
            proxy_frame,
            text=self._tr("settings.label.use_system_proxy"),
            variable=use_sys,
        ).grid(row=0, column=0, columnspan=3, sticky="w", **pad)

        ttk.Separator(proxy_frame, orient="horizontal").grid(
            row=1, column=0, columnspan=3, sticky="ew", pady=4
        )

        def _field_state(*_: object) -> None:
            state = tk.DISABLED if use_sys.get() else tk.NORMAL
            for w in manual_widgets:
                w.configure(state=state)

        use_sys.trace_add("write", _field_state)

        manual_widgets: list[ttk.Widget] = []

        def _lbl_entry(row: int, label: str, var: tk.StringVar, hint: str = "") -> None:
            ttk.Label(proxy_frame, text=label).grid(row=row, column=0, sticky="w", **pad)
            e = ttk.Entry(proxy_frame, textvariable=var, width=44)
            e.grid(row=row, column=1, columnspan=2, sticky="ew", **pad)
            manual_widgets.append(e)
            if hint:
                ttk.Label(proxy_frame, text=hint, foreground="#777777").grid(
                    row=row + 1, column=1, columnspan=2, sticky="w", padx=8
                )

        _lbl_entry(
            2, self._tr("settings.label.http_proxy"), http_v, self._tr("settings.hint.http_proxy")
        )
        _lbl_entry(
            4,
            self._tr("settings.label.https_proxy"),
            https_v,
            self._tr("settings.hint.https_proxy"),
        )
        _lbl_entry(6, self._tr("settings.label.no_proxy"), no_v, self._tr("settings.hint.no_proxy"))

        ttk.Label(proxy_frame, text=self._tr("settings.label.ca_bundle")).grid(
            row=8, column=0, sticky="w", **pad
        )
        ca_entry = ttk.Entry(proxy_frame, textvariable=ca_v, width=36)
        ca_entry.grid(row=8, column=1, sticky="ew", **pad)
        manual_widgets.append(ca_entry)

        def _browse_ca() -> None:
            path = filedialog.askopenfilename(
                title=self._tr("dialog.title.select_ca_certificate"),
                filetypes=[
                    (self._tr("dialog.filetype.pem_crt"), "*.pem *.crt *.cer"),
                    (self._tr("dialog.filetype.all_files"), "*.*"),
                ],
            )
            if path:
                ca_v.set(path)

        browse_ca_btn = ttk.Button(
            proxy_frame, text=self._tr("settings.button.browse"), command=_browse_ca
        )
        browse_ca_btn.grid(row=8, column=2, padx=(0, 4), pady=3)
        manual_widgets.append(browse_ca_btn)

        ttk.Label(proxy_frame, text=self._tr("settings.hint.ca_bundle"), foreground="#777777").grid(
            row=9, column=1, columnspan=2, sticky="w", padx=8
        )

        _field_state()  # set initial enabled/disabled state

        # -- HuggingFace Token --
        hf_frame = ttk.LabelFrame(dlg, text=self._tr("settings.section.hf"), padding=(10, 8))
        hf_frame.pack(fill=tk.X, padx=10, pady=(0, 4))
        hf_frame.columnconfigure(1, weight=1)

        ttk.Label(hf_frame, text=self._tr("settings.label.hf_token")).grid(
            row=0, column=0, sticky="w", **pad
        )
        hf_entry = ttk.Entry(hf_frame, textvariable=hf_token_v, width=44, show="*")
        hf_entry.grid(row=0, column=1, sticky="ew", **pad)

        def _toggle_token_visibility() -> None:
            hf_entry.configure(show="" if hf_entry.cget("show") == "*" else "*")

        ttk.Button(
            hf_frame, text=self._tr("settings.button.show"), command=_toggle_token_visibility
        ).grid(row=0, column=2, padx=(0, 4), pady=3)
        ttk.Label(
            hf_frame,
            text=self._tr("settings.hint.hf_token"),
            foreground="#777777",
        ).grid(row=1, column=0, columnspan=3, sticky="w", padx=8)

        # -- Bottom buttons --
        btn_row = ttk.Frame(dlg)
        btn_row.pack(fill=tk.X, padx=10, pady=(4, 10))

        def _detect() -> None:
            sys_p = get_system_proxies()
            http_v.set(sys_p["http"])
            https_v.set(sys_p["https"])
            no_v.set(sys_p["no"])

        def _save() -> None:
            import os

            self._proxy_use_system_var.set(use_sys.get())
            self._proxy_http_var.set(http_v.get().strip())
            self._proxy_https_var.set(https_v.get().strip())
            self._proxy_no_var.set(no_v.get().strip())
            self._proxy_ca_var.set(ca_v.get().strip())
            self._hf_token_var.set(hf_token_v.get().strip())
            self._persist_gui_settings()
            proxy_settings = {
                "proxy_use_system": "true" if use_sys.get() else "false",
                "proxy_http": http_v.get().strip(),
                "proxy_https": https_v.get().strip(),
                "proxy_no": no_v.get().strip(),
                "proxy_ca_bundle": ca_v.get().strip(),
            }
            apply_proxy_settings(proxy_settings)
            token = hf_token_v.get().strip()
            if token:
                os.environ["HF_TOKEN"] = token
                os.environ["HUGGING_FACE_HUB_TOKEN"] = token
            else:
                os.environ.pop("HF_TOKEN", None)
                os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
            dlg.destroy()

        ttk.Button(btn_row, text=self._tr("settings.button.detect_proxy"), command=_detect).pack(
            side=tk.LEFT
        )
        ttk.Button(btn_row, text=self._tr("settings.button.cancel"), command=dlg.destroy).pack(
            side=tk.RIGHT, padx=(4, 0)
        )
        ttk.Button(
            btn_row, text=self._tr("settings.button.save"), command=_save, style="Accent.TButton"
        ).pack(side=tk.RIGHT)

    def _open_prompt_editor(self) -> None:
        prompt_name = self._llm_prompt_var.get().strip() or "summarize"
        prompt_def = BUILTIN_PROMPTS[prompt_name]
        dialog = tk.Toplevel(self.root)
        dialog.title(self._tr("prompt.title"))
        dialog.geometry("900x700")

        ttk.Label(
            dialog,
            text=self._tr("prompt.header", prompt_name=prompt_name),
            font=("", 10, "bold"),
        ).pack(anchor="w", padx=10, pady=(10, 4))
        ttk.Label(dialog, text=self._tr("prompt.label.system"), anchor="w").pack(fill=tk.X, padx=10)
        system_text = scrolledtext.ScrolledText(dialog, height=8, wrap=tk.WORD)
        system_text.pack(fill=tk.BOTH, expand=False, padx=10, pady=(0, 8))
        system_text.insert("1.0", prompt_def["system"])
        system_text.configure(state=tk.DISABLED)

        ttk.Label(
            dialog,
            text=self._tr("prompt.label.user"),
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
                self._llm_status_label.configure(text=self._tr("prompt.status.invalid"))
                return
            self._llm_custom_user_prompt = "" if prompt_text == prompt_def["user"] else prompt_text
            self._persist_gui_settings()
            self._llm_status_label.configure(
                text=self._tr("prompt.status.saved", prompt_name=prompt_name)
            )
            dialog.destroy()

        ttk.Button(button_row, text=self._tr("prompt.button.reset"), command=_reset).pack(
            side=tk.LEFT
        )
        ttk.Button(button_row, text=self._tr("prompt.button.save"), command=_save).pack(
            side=tk.RIGHT
        )
        ttk.Button(button_row, text=self._tr("prompt.button.close"), command=dialog.destroy).pack(
            side=tk.RIGHT, padx=(0, 6)
        )

    def _poll_device_changes(self) -> None:
        """Check every 5 s if the audio device list has changed and refresh the menu."""
        try:
            devices = list_windows_capture_devices()
            fingerprint: frozenset = frozenset(
                (d.id, d.label, d.kind, d.is_default) for d in devices
            )
        except Exception:
            fingerprint = frozenset()

        if fingerprint != self._device_list_fingerprint:
            self._refresh_device_options()

        self.root.after(5000, self._poll_device_changes)

    def _refresh_device_options(self) -> None:
        options: list[DeviceOption] = []

        try:
            devices = list_windows_capture_devices()
            options.extend(
                DeviceOption(
                    label=self._device_option_label(device.kind, device.label),
                    index=device.id,
                    kind=device.kind,
                    is_default=device.is_default,
                )
                for device in devices
            )
            self._device_list_fingerprint = frozenset(
                (d.id, d.label, d.kind, d.is_default) for d in devices
            )
        except Exception:
            pass

        self._device_options = options
        self._rebuild_device_menu()
        self._apply_default_device_selection()
        self._update_device_picker_label()

    def _rebuild_device_menu(self) -> None:
        self._device_menu.delete(0, tk.END)
        self._device_check_vars = {}
        if not self._device_options:
            self._device_menu.add_command(
                label=self._tr("device.none_found"),
                state=tk.DISABLED,
            )
            return
        current_kind: str | None = None
        for option in self._device_options:
            if current_kind is not None and option.kind != current_kind:
                self._device_menu.add_separator()
            current_kind = option.kind
            variable = tk.BooleanVar(value=False)
            if option.index is not None:
                self._device_check_vars[str(option.index)] = variable
            self._device_menu.add_checkbutton(
                label=option.label,
                variable=variable,
                command=lambda opt=option: self._toggle_device_option(opt),
            )

    def _apply_default_device_selection(self) -> None:
        valid_ids = {option.index for option in self._device_options if option.index is not None}
        if self._selected_microphone_id not in valid_ids:
            self._selected_microphone_id = None
        if self._selected_system_id not in valid_ids:
            self._selected_system_id = None

        requested = self._requested_device_index
        if requested is not None and requested in valid_ids:
            requested_option = next(
                (option for option in self._device_options if option.index == requested),
                None,
            )
            if requested_option is not None:
                if requested_option.kind == "microphone":
                    self._selected_microphone_id = requested_option.index
                elif requested_option.kind == "system":
                    self._selected_system_id = requested_option.index

        if self._selected_microphone_id is None:
            default_mic = next(
                (
                    option.index
                    for option in self._device_options
                    if option.kind == "microphone" and option.is_default
                ),
                None,
            )
            if default_mic is None:
                default_mic = next(
                    (
                        option.index
                        for option in self._device_options
                        if option.kind == "microphone"
                    ),
                    None,
                )
            self._selected_microphone_id = default_mic
        if self._selected_system_id is None:
            default_system = next(
                (
                    option.index
                    for option in self._device_options
                    if option.kind == "system" and option.is_default
                ),
                None,
            )
            if default_system is None:
                default_system = next(
                    (option.index for option in self._device_options if option.kind == "system"),
                    None,
                )
            self._selected_system_id = default_system

        for option in self._device_options:
            if option.index is None:
                continue
            variable = self._device_check_vars.get(str(option.index))
            if variable is None:
                continue
            variable.set(option.index in (self._selected_microphone_id, self._selected_system_id))

    def _toggle_device_option(self, option: DeviceOption) -> None:
        if option.index is None:
            return
        variable = self._device_check_vars.get(str(option.index))
        if variable is None:
            return
        checked = bool(variable.get())
        if option.kind == "microphone":
            self._selected_microphone_id = option.index if checked else None
            if checked:
                self._clear_other_device_checks("microphone", keep_id=option.index)
        elif option.kind == "system":
            self._selected_system_id = option.index if checked else None
            if checked:
                self._clear_other_device_checks("system", keep_id=option.index)
        self._update_device_picker_label()

    def _clear_other_device_checks(self, kind: str, *, keep_id: str | int | None) -> None:
        for option in self._device_options:
            if option.kind != kind or option.index is None or option.index == keep_id:
                continue
            variable = self._device_check_vars.get(str(option.index))
            if variable is not None:
                variable.set(False)

    def _update_device_picker_label(self) -> None:
        labels: list[str] = []
        mic_option = next(
            (
                option
                for option in self._device_options
                if option.index == self._selected_microphone_id
            ),
            None,
        )
        system_option = next(
            (option for option in self._device_options if option.index == self._selected_system_id),
            None,
        )
        if mic_option is not None:
            labels.append(
                self._device_short_label("microphone", mic_option.label.split(": ", 1)[-1])
            )
        if system_option is not None:
            labels.append(
                self._device_short_label("system", system_option.label.split(": ", 1)[-1])
            )
        self._device_picker_var.set(" | ".join(labels) if labels else self._tr("device.select"))

    # ------------------------------------------------------------------
    # File transcription methods
    # ------------------------------------------------------------------

    def _install_ffmpeg(self) -> None:
        """Start local FFmpeg installation in a background thread."""
        self._ffmpeg_install_btn.configure(
            state="disabled", text=self._tr("ffmpeg.status.installing")
        )
        self._ffmpeg_install_status.configure(text=self._tr("ffmpeg.status.preparing"))

        def _run() -> None:
            ok = install_ffmpeg_local(
                on_output=lambda line: self.root.after(
                    0, lambda ln=line: self._ffmpeg_install_status.configure(text=ln[:80])
                )
            )

            def _finish() -> None:
                self._ffmpeg_path = find_ffmpeg()
                if self._ffmpeg_path is not None:
                    self._ffmpeg_install_status.configure(text=self._tr("ffmpeg.status.installed"))
                    self._ffmpeg_banner.pack_forget()
                elif ok:
                    self._ffmpeg_install_status.configure(
                        text=self._tr("ffmpeg.status.restart_required")
                    )
                else:
                    self._ffmpeg_install_status.configure(text=self._tr("ffmpeg.status.failed"))
                    self._ffmpeg_install_btn.configure(
                        state="normal", text=self._tr("ffmpeg.button.retry")
                    )

            self.root.after(0, _finish)

        threading.Thread(target=_run, daemon=True).start()

    def _browse_file(self) -> None:
        paths = filedialog.askopenfilenames(
            title=self._tr("file.button.add_files"),
            filetypes=self._media_filetypes(),
        )
        if paths:
            added = self._add_files_to_queue([Path(path) for path in paths])
            self._last_transcript_path = None
            if added:
                self._file_status_label.configure(
                    text=self._tr("file.status.queued_files", count=added)
                )
            else:
                self._file_status_label.configure(text=self._tr("file.status.queue_duplicate"))
            self._file_progress["value"] = 0
            self._clear_llm_output()
            self._refresh_file_workflow()

    def _prepare_file_results_for_next_run(self) -> None:
        for item in self._file_table.get_children():
            self._file_table.delete(item)
        self._file_seg_count = 0
        self._file_segments = []
        self._last_transcript_path = None
        self._file_seg_counter_label.configure(text=self._tr("file.summary.segments", count=0))
        self._file_progress["value"] = 0
        self._file_current_progress = 0.0
        self._file_progress_samples = []
        self._clear_llm_output()

    def _set_file_batch_ui_running(self, running: bool) -> None:
        control_state = tk.DISABLED if running else tk.NORMAL
        combo_state = "disabled" if running else "readonly"
        self._file_transcribe_btn.configure(state=control_state)
        self._file_cancel_btn.configure(state=(tk.NORMAL if running else tk.DISABLED))
        self._file_model_combo.configure(state=combo_state)
        self._file_lang_combo.configure(state=combo_state)
        self._file_quality_combo.configure(state=combo_state)
        self._file_download_btn.configure(state=control_state)
        self._file_path_entry.configure(state=("disabled" if running else "normal"))
        self._file_add_btn.configure(state=control_state)
        self._file_remove_btn.configure(state=control_state)
        self._file_clear_queue_btn.configure(state=control_state)
        self._refresh_file_diarization_controls()

    def _queue_position_label(self, item_id: str | None) -> str:
        if item_id is None:
            return ""
        children = list(self._file_queue_table.get_children())
        if item_id not in children:
            return ""
        return f"[{children.index(item_id) + 1}/{len(children)}] "

    def _finish_file_batch_run(self) -> None:
        self._file_active_queue_id = None
        self._file_batch_cancel_requested = False
        self._file_active_error_message = None
        self._set_file_batch_ui_running(False)
        self._file_start_time = None
        self._file_time_label.configure(text="")

        statuses = [item.status for item in self._file_queue_items.values()]
        done = sum(status == "Done" for status in statuses)
        errors = sum(status == "Error" for status in statuses)
        cancelled = sum(status == "Cancelled" for status in statuses)
        queued = sum(status == "Queued" for status in statuses)
        total = len(statuses)

        if (
            total == 1
            and done == 1
            and self._file_seg_count > 0
            and self._last_transcript_path is not None
        ):
            self._file_status_label.configure(
                text=self._tr(
                    "file.status.ready_transcript", file_name=self._last_transcript_path.name
                ),
                foreground="",
            )
        elif total == 1 and done == 1:
            self._file_status_label.configure(
                text=self._tr("file.status.finished_no_speech"),
                foreground="",
            )
        elif cancelled and done == 0 and errors == 0:
            self._file_status_label.configure(
                text=self._tr("file.status.queue_cancelled", queued=queued),
                foreground="",
            )
        else:
            self._file_status_label.configure(
                text=self._tr(
                    "file.status.queue_finished",
                    done=done,
                    errors=errors,
                    cancelled=cancelled,
                    queued=queued,
                ),
                foreground="",
            )
        self._refresh_file_workflow()

    def _start_next_file_in_queue(self) -> None:
        next_id = self._next_pending_file_queue_id()
        if next_id is None:
            self._finish_file_batch_run()
            return

        item = self._file_queue_items[next_id]
        file_path = item.file_path
        model = get_model_info(self._file_model_var.get() or "small").id
        language = self._language_code_for_label(self._file_lang_var.get(), model)
        diarization_strategy = (self._file_diarization_var.get() or "auto").strip().lower()
        min_speakers = self._parse_optional_positive_int_localized(
            self._file_min_speakers_var.get(),
            self._tr("validation.field.min_speakers"),
        )
        max_speakers = self._parse_optional_positive_int_localized(
            self._file_max_speakers_var.get(),
            self._tr("validation.field.max_speakers"),
        )

        self._file_active_queue_id = next_id
        self._file_active_error_message = None
        item.status = "In progress"
        item.progress = 0.0
        item.result = ""
        item.output_path = None
        self._update_file_queue_row(next_id)
        self._select_file_queue_item(next_id)

        self._prepare_file_results_for_next_run()
        self._file_progress["value"] = 0
        self._file_start_time = monotonic()
        self._file_current_progress = 0.0
        self._file_progress_samples = []
        self._file_time_label.configure(text="")
        self._file_status_label.configure(
            text=self._tr(
                "file.status.transcribing_queue",
                position=self._queue_position_label(next_id),
                file_name=file_path.name,
            ),
            foreground="",
        )
        self._refresh_file_workflow()
        self.root.after(500, self._tick_file_timer)

        self._file_worker = FileTranscribeWorker(
            file_path=file_path,
            model=model,
            language=language,
            diarization_strategy=diarization_strategy,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            quality=self._file_quality_var.get(),
            on_status=self._schedule_file_status,
            on_segments=self._schedule_file_segments,
            on_error=self._schedule_file_error,
            on_finished=self._schedule_file_finished,
        )
        self._file_worker.start()

    def _start_file_transcribe(self) -> None:
        raw_path = self._file_path_var.get().strip()

        if self._file_worker is not None:
            return  # already running

        try:
            min_speakers = self._parse_optional_positive_int_localized(
                self._file_min_speakers_var.get(),
                self._tr("validation.field.min_speakers"),
            )
            max_speakers = self._parse_optional_positive_int_localized(
                self._file_max_speakers_var.get(),
                self._tr("validation.field.max_speakers"),
            )
        except ValueError as exc:
            self._file_status_label.configure(
                text=self._tr("file.status.error", message=exc),
                foreground="red",
            )
            self._refresh_file_workflow()
            return
        if min_speakers is not None and max_speakers is not None and min_speakers > max_speakers:
            self._file_status_label.configure(
                text=self._tr("file.status.min_max_invalid"),
                foreground="red",
            )
            self._refresh_file_workflow()
            return

        if hasattr(self, "_file_queue_table"):
            if not self._file_queue_items:
                if not raw_path:
                    self._file_status_label.configure(text=self._tr("file.status.no_files_queued"))
                    self._refresh_file_workflow()
                    return
                file_path = Path(raw_path)
                if not file_path.exists():
                    self._file_status_label.configure(
                        text=self._tr("file.status.file_missing_short", file_name=file_path.name)
                    )
                    self._refresh_file_workflow()
                    return
                self._add_files_to_queue([file_path])
            self._clear_file_table()
            self._file_batch_cancel_requested = False
            self._set_file_batch_ui_running(True)
            self._start_next_file_in_queue()
            return

        if not raw_path:
            self._file_status_label.configure(text=self._tr("file.status.no_file_selected"))
            self._refresh_file_workflow()
            return

        file_path = Path(raw_path)
        if not file_path.exists():
            self._file_status_label.configure(
                text=self._tr("file.status.file_missing_short", file_name=file_path.name)
            )
            self._refresh_file_workflow()
            return

        self._clear_file_table()

        model = get_model_info(self._file_model_var.get() or "small").id
        language = self._language_code_for_label(self._file_lang_var.get(), model)
        diarization_strategy = (self._file_diarization_var.get() or "auto").strip().lower()

        self._file_transcribe_btn.configure(state=tk.DISABLED)
        self._file_cancel_btn.configure(state=tk.NORMAL)
        self._file_model_combo.configure(state="disabled")
        self._file_lang_combo.configure(state="disabled")
        self._file_quality_combo.configure(state="disabled")
        self._refresh_file_diarization_controls()
        self._file_progress["value"] = 0
        self._last_transcript_path = None
        self._file_start_time = monotonic()
        self._file_current_progress = 0.0
        self._file_progress_samples = []
        self._file_time_label.configure(text="")
        self._file_status_label.configure(
            text=self._tr("file.status.transcribing", file_name=file_path.name)
        )
        self._refresh_file_workflow()
        self.root.after(500, self._tick_file_timer)

        self._file_worker = FileTranscribeWorker(
            file_path=file_path,
            model=model,
            language=language,
            diarization_strategy=diarization_strategy,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            quality=self._file_quality_var.get(),
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
        self._file_status_label.configure(text=msg, foreground="")
        self._file_progress["value"] = int(progress * 100)
        self._file_current_progress = progress
        if self._file_active_queue_id is not None:
            item = self._file_queue_items[self._file_active_queue_id]
            item.status = "In progress"
            item.progress = progress
            item.result = msg
            self._update_file_queue_row(self._file_active_queue_id)
        self._refresh_file_workflow()

    def _show_file_error(self, message: str) -> None:
        self._file_status_label.configure(
            text=self._tr("file.status.error", message=message),
            foreground="red",
        )
        self._file_progress["value"] = 0
        self._file_current_progress = 0.0
        self._file_active_error_message = message
        if self._file_active_queue_id is not None:
            item = self._file_queue_items[self._file_active_queue_id]
            item.status = "Error"
            item.progress = 0.0
            item.result = message
            self._update_file_queue_row(self._file_active_queue_id)
        active_file = None
        if self._file_active_queue_id is not None:
            active_file = str(self._file_queue_items[self._file_active_queue_id].file_path)
        log.error("gui.file_error", file=active_file, error=message)
        self._refresh_file_workflow()

    def _cancel_file_transcribe(self) -> None:
        if self._file_worker is not None:
            self._file_batch_cancel_requested = True
            self._file_cancel_btn.configure(state=tk.DISABLED)
            self._file_status_label.configure(
                text=self._tr("file.status.cancelling_queue"), foreground=""
            )
            self._file_worker.cancel()

    def _download_file_model(self) -> None:
        """Download the currently selected file-transcription model in a background thread."""
        model_id = self._file_model_var.get() or "small"
        model_info = get_model_info(model_id)
        if self._is_model_cached_locally(model_info):
            should_redownload = messagebox.askyesno(
                self._tr("dialog.title.redownload_model"),
                self._tr("dialog.message.redownload_model", model_name=model_info.name),
                parent=self.root,
            )
            if not should_redownload:
                self._file_status_label.configure(
                    text=self._tr(
                        "file.status.download_cancelled_existing",
                        model_name=model_info.name,
                    ),
                    foreground="",
                )
                self._refresh_file_workflow()
                return

        self._file_download_btn.configure(state=tk.DISABLED)
        self._file_status_label.configure(
            text=self._tr("file.status.download_start", model_name=model_info.name),
            foreground="",
        )
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._append_log_line(f"{timestamp} | DOWNLOAD | Starting download of {model_info.name}…\n")

        result_q: queue.Queue[Exception | None] = queue.Queue()

        def _on_done(error: Exception | None) -> None:
            self._file_download_btn.configure(state=tk.NORMAL)
            ts = datetime.now().strftime("%H:%M:%S")
            if error:
                msg = self._tr("file.status.download_failed", error=error)
                self._file_status_label.configure(text=msg, foreground="red")
                self._append_log_line(f"{ts} | DOWNLOAD ERROR | {error}\n")
            else:
                msg = self._tr("file.status.download_ready", model_name=model_info.name)
                self._file_status_label.configure(text=msg, foreground="")
                self._append_log_line(f"{ts} | DOWNLOAD | {model_info.name} ready.\n")

        def _poll() -> None:
            try:
                error = result_q.get_nowait()
            except queue.Empty:
                self.root.after(100, _poll)
                return
            _on_done(error)

        def _run() -> None:
            try:
                if model_info.engine == "gigaam":
                    from transformers import AutoModel

                    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                    AutoModel.from_pretrained(
                        "ai-sage/GigaAM-v3",
                        trust_remote_code=True,
                        token=token,
                        force_download=True,
                    )
                elif model_info.engine == "breeze":
                    from huggingface_hub import snapshot_download

                    snapshot_download(
                        "MediaTek-Research/Breeze-ASR-25",
                        force_download=True,
                        ignore_patterns=[
                            "optimizer.bin",
                            "scheduler.bin",
                            "random_states_*.pkl",
                            "*.png",
                            "*.pt",
                        ],
                    )
                elif model_info.engine == "parakeet":
                    from nemo.collections.asr.models import ASRModel

                    ASRModel.from_pretrained(
                        model_name="nvidia/parakeet-tdt-0.6b-v3",
                        refresh_cache=True,
                    )
                else:
                    from huggingface_hub import snapshot_download

                    snapshot_download(
                        f"Systran/faster-whisper-{model_info.id}",
                        force_download=True,
                    )
                result_q.put(None)
            except Exception as exc:
                result_q.put(exc)

        threading.Thread(target=_run, daemon=True).start()
        self.root.after(100, _poll)

    def _tick_file_timer(self) -> None:
        if self._file_worker is None or self._file_start_time is None:
            return
        now = monotonic()
        elapsed = now - self._file_start_time
        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)
        elapsed_str = f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

        # Record sample and prune to a 10-second sliding window
        progress = self._file_current_progress
        self._file_progress_samples.append((now, progress))
        cutoff = now - 10.0
        self._file_progress_samples = [
            (t, p) for t, p in self._file_progress_samples if t >= cutoff
        ]

        # Show ETA only when progress is advancing at a measurable rate
        label = elapsed_str
        if len(self._file_progress_samples) >= 2:
            t0, p0 = self._file_progress_samples[0]
            t1, p1 = self._file_progress_samples[-1]
            dt = t1 - t0
            dp = p1 - p0
            # Require at least 2 s of window and 0.5% progress gained to compute ETA
            if dt >= 2.0 and dp >= 0.005 and progress < 1.0:
                velocity = dp / dt  # progress per second
                remaining = (1.0 - progress) / velocity
                rm, rs = divmod(int(remaining), 60)
                rh, rm = divmod(rm, 60)
                eta_str = f"{rh:02d}:{rm:02d}:{rs:02d}" if rh else f"{rm:02d}:{rs:02d}"
                label = f"{elapsed_str} | ~{eta_str} left"

        self._file_time_label.configure(text=label)
        self.root.after(500, self._tick_file_timer)

    def _add_file_segments(self, segments: list[TranslatedSegment]) -> None:
        for seg in segments:
            ts = seg.diarized.segment
            speaker = seg.diarized.speaker_id
            # Format as HH:MM:SS
            total_secs = int(ts.start_time)
            h, remainder = divmod(total_secs, 3600)
            m, s = divmod(remainder, 60)
            time_str = f"{h:02d}:{m:02d}:{s:02d}"
            self._file_table.insert("", tk.END, values=(time_str, speaker, ts.text.strip()))
            self._file_seg_count += 1

        self._file_segments.extend(segments)
        self._file_seg_counter_label.configure(
            text=self._tr("file.summary.segments", count=self._file_seg_count)
        )
        self._file_table.yview_moveto(1.0)
        self._refresh_file_workflow()

    def _on_file_worker_finished(self) -> None:
        was_cancelled = self._file_worker is not None and self._file_worker._cancelled
        active_id = self._file_active_queue_id
        self._file_worker = None
        self._file_start_time = None
        self._file_time_label.configure(text="")
        if active_id is None:
            self._set_file_batch_ui_running(False)
            self._refresh_file_workflow()
            return

        item = self._file_queue_items[active_id]
        if was_cancelled or self._file_batch_cancel_requested:
            item.status = "Cancelled"
            item.progress = 0.0
            item.result = "Cancelled"
            self._update_file_queue_row(active_id)
            self._finish_file_batch_run()
            return

        if self._file_active_error_message is None:
            item.status = "Done"
            item.progress = 1.0
            if self._file_seg_count > 0:
                self._last_transcript_path = self._auto_save_transcript()
                self._persist_gui_settings()
                item.output_path = self._last_transcript_path
                item.result = self._last_transcript_path.name
            else:
                item.result = self._tr("file.status.no_speech")
            self._update_file_queue_row(active_id)

        self._file_active_queue_id = None
        self._file_active_error_message = None

        if self._next_pending_file_queue_id() is not None:
            self._start_next_file_in_queue()
            return

        self._finish_file_batch_run()

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
        self._file_seg_counter_label.configure(text=self._tr("file.summary.segments", count=0))
        self._file_progress["value"] = 0
        self._file_status_label.configure(text=self._tr("file.status.cleared"))
        self._clear_llm_output()
        self._refresh_file_workflow()

    @staticmethod
    def _read_transcript_text(path: Path) -> str:
        data = path.read_bytes()
        last_error: UnicodeDecodeError | None = None
        for encoding in ("utf-8-sig", "utf-16", "cp1251"):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        return data.decode("utf-8-sig")

    @staticmethod
    def _transcript_time_label(seconds: int) -> str:
        hours, remainder = divmod(max(0, seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @staticmethod
    def _parse_transcript_rows(text: str) -> list[tuple[str, str, str]]:
        rows: list[tuple[str, str, str]] = []
        fallback_seconds = 0
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            match = _IMPORTED_TRANSCRIPT_LINE_RE.match(line)
            if match is not None:
                rows.append(
                    (
                        match.group("time"),
                        match.group("speaker").strip() or _IMPORTED_TRANSCRIPT_SPEAKER,
                        match.group("text").strip(),
                    )
                )
                continue
            rows.append(
                (
                    TranscriptionGUI._transcript_time_label(fallback_seconds),
                    _IMPORTED_TRANSCRIPT_SPEAKER,
                    line,
                )
            )
            fallback_seconds += 1
        return rows

    @staticmethod
    def _parse_srt_rows(text: str) -> list[tuple[str, str, str]]:
        rows: list[tuple[str, str, str]] = []
        current_block: list[str] = []

        def _flush_block(lines: list[str]) -> None:
            block = [line.strip() for line in lines if line.strip()]
            if not block:
                return
            if block[0].isdigit():
                block = block[1:]
            if len(block) < 2:
                return
            time_match = _IMPORTED_SRT_TIME_RANGE_RE.match(block[0])
            if time_match is None:
                return
            text_value = " ".join(part.strip() for part in block[1:] if part.strip())
            if not text_value:
                return
            speaker = _IMPORTED_TRANSCRIPT_SPEAKER
            speaker_match = _IMPORTED_SRT_SPEAKER_RE.match(text_value)
            if speaker_match is not None:
                speaker = speaker_match.group("speaker").strip() or _IMPORTED_TRANSCRIPT_SPEAKER
                text_value = speaker_match.group("text").strip()
            rows.append((time_match.group("start"), speaker, text_value))

        for raw_line in text.splitlines():
            if raw_line.strip():
                current_block.append(raw_line)
                continue
            _flush_block(current_block)
            current_block = []
        _flush_block(current_block)
        return rows

    def _load_transcript_file(self) -> None:
        path = filedialog.askopenfilename(
            title=self._tr("dialog.title.load_transcript"),
            filetypes=[
                (self._tr("dialog.filetype.text_file"), "*.txt"),
                (self._tr("dialog.filetype.vtt_file"), "*.vtt"),
                (self._tr("dialog.filetype.srt_file"), "*.srt"),
                (self._tr("dialog.filetype.markdown_file"), "*.md *.markdown"),
                (self._tr("dialog.filetype.all_files"), "*.*"),
            ],
        )
        if not path:
            return

        transcript_path = Path(path)
        try:
            transcript_text = self._read_transcript_text(transcript_path)
            if transcript_path.suffix.lower() in {".srt", ".vtt"}:
                rows = self._parse_srt_rows(transcript_text)
            else:
                rows = self._parse_transcript_rows(transcript_text)
        except (OSError, UnicodeError) as exc:
            log.error("gui.transcript_load_failed", file=str(transcript_path), error=str(exc))
            self._file_status_label.configure(
                text=self._tr("file.status.transcript_load_failed", error=exc)
            )
            self._refresh_file_workflow()
            return

        if not rows:
            log.warning("gui.transcript_load_empty", file=str(transcript_path))
            self._file_status_label.configure(text=self._tr("file.status.transcript_empty"))
            self._refresh_file_workflow()
            return

        for item in self._file_table.get_children():
            self._file_table.delete(item)
        self._file_seg_count = 0
        self._file_segments = []
        self._last_transcript_path = transcript_path
        self._file_progress["value"] = 0
        self._clear_llm_output()
        for time_str, speaker, segment_text in rows:
            self._file_table.insert("", tk.END, values=(time_str, speaker, segment_text))
        self._file_seg_count = len(rows)
        self._file_seg_counter_label.configure(
            text=self._tr("file.summary.segments", count=self._file_seg_count)
        )
        self._file_table.yview_moveto(1.0)
        self._file_status_label.configure(
            text=self._tr(
                "file.status.transcript_loaded",
                count=self._file_seg_count,
                file_name=transcript_path.name,
            )
        )
        log.info(
            "gui.transcript_loaded",
            file=str(transcript_path),
            rows=self._file_seg_count,
        )
        self._refresh_file_workflow()

    def _save_file_result(self) -> None:
        children = self._file_table.get_children()
        if not children:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[
                (self._tr("dialog.filetype.text_file"), "*.txt"),
                (self._tr("dialog.filetype.vtt_file"), "*.vtt"),
                (self._tr("dialog.filetype.srt_file"), "*.srt"),
                (self._tr("dialog.filetype.all_files"), "*.*"),
            ],
            title=self._tr("dialog.title.save_file_transcription"),
        )
        if not path:
            return
        try:
            out_path = Path(path)
            suffix = out_path.suffix.lower()
            if suffix == ".srt":
                self._save_as_srt(out_path)
            elif suffix == ".vtt":
                self._save_as_vtt(out_path)
            else:
                self._save_as_txt(out_path, children)
            self._file_status_label.configure(
                text=self._tr("file.status.saved", count=self._file_seg_count, path=out_path.name)
            )
        except (OSError, ValueError) as exc:
            self._file_status_label.configure(text=self._tr("file.status.save_failed", error=exc))
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

    def _save_as_vtt(self, path: Path) -> None:
        if not self._file_segments:
            raise ValueError(self._tr("file.status.vtt_requires_timestamps"))
        lines = ["WEBVTT", ""]
        for seg in self._file_segments:
            ts = seg.diarized.segment
            start = _secs_to_vtt(ts.start_time)
            end = _secs_to_vtt(ts.end_time)
            speaker = seg.diarized.speaker.strip()
            text = ts.text.strip()
            if speaker:
                text = f"[{speaker}] {text}"
            lines.append(f"{start} --> {end}")
            lines.append(text)
            lines.append("")
        path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

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
        if self._llm_worker is not None or self._llm_preflight_running:
            log.warning("gui.llm_summarize_skipped", reason="already_running")
            return
        transcript = self._get_file_transcript_text()
        if not transcript:
            log.warning("gui.llm_summarize_skipped", reason="no_transcript")
            self._llm_status_label.configure(text=self._tr("llm.status.no_transcript"))
            self._refresh_file_workflow()
            return

        url = self._llm_url_var.get().strip() or DEFAULT_BASE_URL
        model = self._llm_model_var.get().strip() or DEFAULT_MODEL
        api_key = self._llm_key_var.get().strip()
        prompt_name = self._llm_prompt_var.get().strip() or "summarize"
        try:
            context_tokens, context_source = self._resolve_llm_context_limit()
        except ValueError as exc:
            log.warning(
                "gui.llm_summarize_skipped",
                reason="invalid_context",
                raw_context=self._llm_context_var.get().strip(),
            )
            self._llm_status_label.configure(text=str(exc))
            self._refresh_file_workflow()
            return
        self._persist_gui_settings()
        self._llm_last_error_message = None

        self._clear_llm_output()
        self._llm_summarize_btn.configure(state=tk.DISABLED)
        self._llm_status_label.configure(text=self._tr("llm.status.checking_api", model=model))
        log.info(
            "gui.llm_summarize_requested",
            base_url=url,
            model=model,
            prompt_name=prompt_name,
            transcript_chars=len(transcript),
            transcript_lines=transcript.count("\n") + 1,
            api_key_present=bool(api_key),
            custom_user_prompt=bool(self._llm_custom_user_prompt),
            context_tokens_resolved=context_tokens,
            context_source=context_source,
        )
        self._llm_preflight_running = True
        self._refresh_file_workflow()
        result_q: queue.Queue[tuple[bool, str | None]] = queue.Queue()

        def _poll() -> None:
            try:
                success, error = result_q.get_nowait()
            except queue.Empty:
                with suppress(tk.TclError, RuntimeError):
                    self.root.after(100, _poll)
                return
            self._llm_preflight_running = False
            if not success:
                self._llm_summarize_btn.configure(state=tk.NORMAL)
                self._show_llm_error(error or self._tr("llm.status.api_unavailable"))
                return
            self._llm_status_label.configure(text=self._tr("llm.status.sending", model=model))
            self._llm_worker = LLMWorker(
                text=transcript,
                model=model,
                base_url=url,
                api_key=api_key,
                prompt_name=prompt_name,
                custom_user_prompt=(self._llm_custom_user_prompt or None),
                context_limit_tokens=context_tokens,
                on_token=self._schedule_llm_token,
                on_error=self._schedule_llm_error,
                on_finished=self._schedule_llm_finished,
            )
            self._llm_worker.start()

        def _run() -> None:
            try:
                asyncio.run(
                    verify_model_ready(
                        base_url=url,
                        model=model,
                        api_key=api_key,
                    )
                )
                result_q.put((True, None))
            except Exception as exc:  # pragma: no cover
                result_q.put((False, str(exc)))

        threading.Thread(target=_run, daemon=True).start()
        self.root.after(100, _poll)

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
        self._llm_last_error_message = message
        log.error(
            "gui.llm_error",
            base_url=self._llm_url_var.get().strip() or DEFAULT_BASE_URL,
            model=self._llm_model_var.get().strip() or DEFAULT_MODEL,
            prompt_name=self._llm_prompt_var.get().strip() or "summarize",
            error=message,
        )
        self._llm_status_label.configure(text=self._tr("llm.status.error", message=message[:80]))
        self._append_llm_token(f"\n\n[ERROR] {message}\n")
        self._refresh_file_workflow()

    def _on_llm_finished(self) -> None:
        log.info(
            "gui.llm_finished",
            base_url=self._llm_url_var.get().strip() or DEFAULT_BASE_URL,
            model=self._llm_model_var.get().strip() or DEFAULT_MODEL,
            prompt_name=self._llm_prompt_var.get().strip() or "summarize",
            success=self._llm_last_error_message is None,
        )
        self._llm_worker = None
        self._llm_summarize_btn.configure(state=tk.NORMAL)
        current = self._llm_status_label.cget("text")
        if not str(current).startswith(self._tr("llm.status.error", message="").split("{", 1)[0]):
            self._llm_status_label.configure(text=self._tr("llm.status.done"))
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
            self._llm_status_label.configure(text=self._tr("llm.status.copied"))

    @staticmethod
    def _hf_cache_roots() -> tuple[Path, ...]:
        candidates: list[Path] = []
        for raw in (
            os.environ.get("HUGGINGFACE_HUB_CACHE", "").strip(),
            os.environ.get("HF_HUB_CACHE", "").strip(),
        ):
            if raw:
                candidates.append(Path(raw).expanduser())
        hf_home = os.environ.get("HF_HOME", "").strip()
        if hf_home:
            candidates.append(Path(hf_home).expanduser() / "hub")
        candidates.extend(
            (
                models_dir() / "hub",
                Path.home() / ".cache" / "huggingface" / "hub",
            )
        )
        deduped: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate).lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return tuple(deduped)

    @classmethod
    def _snapshot_exists(cls, repo_id: str) -> bool:
        cache_name = f"models--{repo_id.replace('/', '--')}"
        for root in cls._hf_cache_roots():
            snapshot_dir = root / cache_name / "snapshots"
            if snapshot_dir.exists() and any(snapshot_dir.iterdir()):
                return True
        return False

    @classmethod
    def _is_model_cached_locally(cls, model_info: object) -> bool:
        engine = getattr(model_info, "engine", "")
        model_id = getattr(model_info, "id", "")
        if engine == "gigaam":
            return cls._snapshot_exists("ai-sage/GigaAM-v3")
        if engine == "breeze":
            return cls._snapshot_exists("MediaTek-Research/Breeze-ASR-25")
        if engine == "parakeet":
            return cls._snapshot_exists("nvidia/parakeet-tdt-0.6b-v3")
        return cls._snapshot_exists(f"Systran/faster-whisper-{model_id}")

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


def _secs_to_vtt(secs: float) -> str:
    """Convert float seconds to WebVTT timestamp (HH:MM:SS.mmm)."""
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    ms = int((secs - int(secs)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run VoxFusion GUI mode.")
    parser.add_argument(
        "--translate",
        default=None,
        help="Target translation language code (optional).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="ASR model size (default: auto-selected by quality from installed backends).",
    )
    parser.add_argument(
        "--language",
        default=GUI_DEFAULT_LANGUAGE,
        help="ASR language code (e.g. ru, en).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional Windows audio device id from the GUI/CLI device list.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for GUI mode."""
    args = _build_parser().parse_args(argv)
    if sys.platform != "win32":
        print("Live capture requires Windows WASAPI. File transcription works on all platforms.")
    else:
        patch_subprocess_popen_no_window(force=True)

    # Redirect HuggingFace model cache next to the binary (or project root in dev mode).
    # Must happen before any model library is imported.
    _hf_home = str(models_dir())
    os.environ.setdefault("HF_HOME", _hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(models_dir() / "hub"))
    os.environ.pop("TRANSFORMERS_CACHE", None)

    options = CaptureOptions(
        model=args.model,
        language=args.language,
        translate=args.translate,
        microphone_device_id=args.device
        if args.device and str(args.device).startswith("sd:")
        else None,
        system_device_id=args.device
        if args.device and not str(args.device).startswith("sd:")
        else None,
    )

    root = tk.Tk()
    TranscriptionGUI(root, options)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
