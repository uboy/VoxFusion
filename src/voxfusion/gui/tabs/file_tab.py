"""File Transcription tab widget builder.

``FileTranscriptionTab`` is responsible for constructing all widgets inside
the File Transcription notebook tab (file queue, options, result table, and
the LLM post-processing panel).

It holds a reference to the parent ``TranscriptionGUI`` instance
(``self._gui``) and delegates every callback, state-variable access, and
helper call to it.

This is Phase 1 of the ARCH-3 God-Object reduction.  All logic (methods,
state vars) still lives in ``TranscriptionGUI``; Phase 2 will move them here
incrementally.
"""

from __future__ import annotations

import tkinter as tk
from contextlib import suppress
from tkinter import scrolledtext, ttk
from typing import TYPE_CHECKING

from voxfusion.asr_catalog import get_available_model_catalog
from voxfusion.gui.model_summary import ModelSummaryCard
from voxfusion.gui.tooltip import create_help_icon
from voxfusion.llm.prompts import BUILTIN_PROMPTS

if TYPE_CHECKING:
    from voxfusion.gui.main import TranscriptionGUI

_ASR_MODEL_CHOICES: tuple[str, ...] = tuple(m.id for m in get_available_model_catalog())
_FILE_DIARIZATION_CHOICES: tuple[str, ...] = ("auto", "none", "channel", "ml", "hybrid")


class FileTranscriptionTab:
    """Builds the File Transcription tab widgets onto a given parent frame.

    All widget references are stored on the ``TranscriptionGUI`` instance
    (``gui``) so that the rest of the application code is unchanged.
    """

    def __init__(self, gui: TranscriptionGUI) -> None:
        self._gui = gui

    def build(self, parent: ttk.Frame) -> None:
        """Create all file-transcription widgets inside *parent*."""
        _gui = self._gui
        file_paned = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        file_paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # -- FFmpeg warning banner (hidden when FFmpeg is present) --
        _gui._ffmpeg_banner = tk.Frame(parent, bg="#fff3cd")
        if _gui._ffmpeg_path is None:
            _gui._ffmpeg_banner.pack(fill=tk.X, padx=6, pady=(6, 0))
        _gui._ffmpeg_banner_label = tk.Label(
            _gui._ffmpeg_banner,
            text="",
            bg="#fff3cd",
            fg="#856404",
            anchor="w",
        )
        _gui._ffmpeg_banner_label.pack(side=tk.LEFT, padx=(8, 12), pady=4)
        _gui._bind_text(_gui._ffmpeg_banner_label, "file.banner.ffmpeg_missing")
        _gui._ffmpeg_install_btn = tk.Button(
            _gui._ffmpeg_banner,
            text="",
            command=_gui._install_ffmpeg,
            bg="#e0a800",
            fg="white",
            relief="flat",
            padx=8,
            pady=2,
        )
        _gui._ffmpeg_install_btn.pack(side=tk.LEFT, pady=4)
        _gui._bind_text(_gui._ffmpeg_install_btn, "file.button.install_ffmpeg")
        _gui._ffmpeg_install_status = tk.Label(
            _gui._ffmpeg_banner,
            text="",
            bg="#fff3cd",
            fg="#333333",
            anchor="w",
        )
        _gui._ffmpeg_install_status.pack(side=tk.LEFT, padx=(8, 0), pady=4)

        top_area = ttk.Frame(file_paned)
        file_paned.add(top_area, weight=3)

        _gui._file_workflow_label = ttk.Label(top_area, text="", anchor="w", foreground="#555555")

        # -- File picker + queue --
        top = ttk.PanedWindow(top_area, orient=tk.HORIZONTAL)
        top.pack(fill=tk.BOTH, expand=True, padx=0, pady=(0, 4))

        transcribe_box = ttk.LabelFrame(top, text="", padding=8)
        top.add(transcribe_box, weight=3)
        _gui._bind_labelframe_text(transcribe_box, "file.section.transcription_setup")

        picker = ttk.Frame(transcribe_box)
        picker.pack(fill=tk.X, pady=(0, 4))

        _gui._file_current_label = ttk.Label(picker, text="")
        _gui._file_current_label.pack(side=tk.LEFT, padx=(0, 6))
        _gui._bind_text(_gui._file_current_label, "file.label.current")
        _gui._file_path_entry = ttk.Entry(picker, textvariable=_gui._file_path_var, width=70)
        _gui._file_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        _gui._bind_tooltip(_gui._file_path_entry, "tooltip.file.current")
        _gui._file_add_btn = ttk.Button(
            picker,
            text="",
            command=_gui._browse_file,
        )
        _gui._file_add_btn.pack(side=tk.LEFT, padx=(0, 4))
        _gui._bind_text(_gui._file_add_btn, "file.button.add_files")
        _gui._bind_tooltip(_gui._file_add_btn, "tooltip.file.add_files")
        _gui._file_remove_btn = ttk.Button(
            picker,
            text="",
            command=_gui._remove_selected_file_queue_items,
        )
        _gui._file_remove_btn.pack(side=tk.LEFT, padx=(0, 4))
        _gui._bind_text(_gui._file_remove_btn, "file.button.remove")
        _gui._bind_tooltip(_gui._file_remove_btn, "tooltip.file.remove")
        _gui._file_clear_queue_btn = ttk.Button(
            picker,
            text="",
            command=_gui._clear_file_queue,
        )
        _gui._file_clear_queue_btn.pack(side=tk.LEFT)
        _gui._bind_text(_gui._file_clear_queue_btn, "file.button.clear_list")
        _gui._bind_tooltip(_gui._file_clear_queue_btn, "tooltip.file.clear_list")

        queue_frame = ttk.Frame(transcribe_box)
        queue_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 6))

        queue_cols = ("file", "duration", "size", "status", "progress", "result")
        _gui._file_queue_table = ttk.Treeview(
            queue_frame,
            columns=queue_cols,
            show="headings",
            height=4,
        )
        _gui._bind_tree_heading(_gui._file_queue_table, "file", "file.table.file")
        _gui._bind_tree_heading(_gui._file_queue_table, "duration", "file.table.duration")
        _gui._bind_tree_heading(_gui._file_queue_table, "size", "file.table.size")
        _gui._bind_tree_heading(_gui._file_queue_table, "status", "file.table.status")
        _gui._bind_tree_heading(_gui._file_queue_table, "progress", "file.table.progress")
        _gui._bind_tree_heading(_gui._file_queue_table, "result", "file.table.result")
        _gui._file_queue_table.column("file", width=360, minwidth=220)
        _gui._file_queue_table.column("duration", width=88, minwidth=76, stretch=False)
        _gui._file_queue_table.column("size", width=88, minwidth=76, stretch=False)
        _gui._file_queue_table.column("status", width=120, minwidth=90, stretch=False)
        _gui._file_queue_table.column("progress", width=90, minwidth=70, stretch=False)
        _gui._file_queue_table.column("result", width=220, minwidth=150)
        _gui._file_queue_table.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        _gui._bind_tooltip(_gui._file_queue_table, "tooltip.file.queue_table")
        _gui._file_queue_table.bind(
            "<<TreeviewSelect>>",
            _gui._on_file_queue_selection,
        )

        queue_scroll = ttk.Scrollbar(
            queue_frame,
            orient=tk.VERTICAL,
            command=_gui._file_queue_table.yview,
        )
        queue_scroll.pack(fill=tk.Y, side=tk.RIGHT)
        _gui._file_queue_table.configure(yscrollcommand=queue_scroll.set)

        # -- Options row --
        opts = ttk.Frame(transcribe_box)
        opts.pack(fill=tk.X, pady=(0, 4))

        _gui._file_model_label = ttk.Label(opts, text="")
        _gui._file_model_label.pack(side=tk.LEFT, padx=(0, 6))
        _gui._bind_text(_gui._file_model_label, "file.label.model")
        _gui._file_model_combo = ttk.Combobox(
            opts,
            textvariable=_gui._file_model_var,
            state="readonly",
            width=28,
            values=_ASR_MODEL_CHOICES,
        )
        _gui._file_model_combo.pack(side=tk.LEFT, padx=(0, 12))
        _gui._file_model_combo.bind("<<ComboboxSelected>>", _gui._on_file_model_changed)
        _gui._bind_tooltip(_gui._file_model_combo, "tooltip.file.model")

        _gui._file_language_label = ttk.Label(opts, text="")
        _gui._file_language_label.pack(side=tk.LEFT, padx=(0, 6))
        _gui._bind_text(_gui._file_language_label, "file.label.language")
        _gui._file_lang_combo = ttk.Combobox(
            opts,
            textvariable=_gui._file_lang_var,
            state="readonly",
            width=18,
        )
        _gui._file_lang_combo.pack(side=tk.LEFT, padx=(0, 12))
        _gui._bind_tooltip(_gui._file_lang_combo, "tooltip.file.language")

        _gui._file_quality_label = ttk.Label(opts, text="")
        _gui._file_quality_label.pack(side=tk.LEFT, padx=(0, 6))
        _gui._bind_text(_gui._file_quality_label, "file.label.quality")
        _gui._file_quality_combo = ttk.Combobox(
            opts,
            textvariable=_gui._file_quality_display_var,
            state="readonly",
            width=11,
            values=(),
        )
        _gui._file_quality_combo.pack(side=tk.LEFT, padx=(0, 16))
        _gui._file_quality_combo.bind("<<ComboboxSelected>>", _gui._on_file_quality_changed)
        _gui._bind_tooltip(_gui._file_quality_combo, "tooltip.file.quality")

        diar_opts = ttk.Frame(transcribe_box)
        diar_opts.pack(fill=tk.X, pady=(0, 4))

        _gui._file_diarization_label = ttk.Label(diar_opts, text="")
        _gui._file_diarization_label.pack(side=tk.LEFT, padx=(0, 4))
        _gui._bind_text(_gui._file_diarization_label, "file.label.speaker_separation")
        _gui._file_diarization_combo = ttk.Combobox(
            diar_opts,
            textvariable=_gui._file_diarization_var,
            state="readonly",
            width=9,
            values=_FILE_DIARIZATION_CHOICES,
        )
        _gui._file_diarization_combo.pack(side=tk.LEFT, padx=(0, 0))
        _gui._file_diarization_combo.bind(
            "<<ComboboxSelected>>",
            _gui._on_file_diarization_changed,
        )
        _gui._bind_tooltip(_gui._file_diarization_combo, "tooltip.file.diarization")
        _gui._file_diarization_help = create_help_icon(diar_opts, "")
        _gui._bind_tooltip(_gui._file_diarization_help, "tooltip.file.diarization_help")

        ttk.Separator(diar_opts, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=(8, 8))

        _gui._file_speakers_label = ttk.Label(diar_opts, text="")
        _gui._file_speakers_label.pack(side=tk.LEFT, padx=(0, 4))
        _gui._bind_text(_gui._file_speakers_label, "file.label.speakers")
        _gui._file_speaker_preset_combo = ttk.Combobox(
            diar_opts,
            textvariable=_gui._file_speaker_preset_display_var,
            state="readonly",
            width=12,
            values=(),
        )
        _gui._file_speaker_preset_combo.pack(side=tk.LEFT, padx=(0, 0))
        _gui._file_speaker_preset_combo.bind(
            "<<ComboboxSelected>>",
            _gui._on_speaker_preset_changed,
        )
        _gui._bind_tooltip(_gui._file_speaker_preset_combo, "tooltip.file.speaker_preset")
        _gui._file_speaker_help = create_help_icon(diar_opts, "")
        _gui._bind_tooltip(_gui._file_speaker_help, "tooltip.file.speaker_help")

        ttk.Separator(diar_opts, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=(8, 8))

        _gui._file_min_lbl = ttk.Label(diar_opts, text="")
        _gui._file_min_lbl.pack(side=tk.LEFT, padx=(0, 4))
        _gui._bind_text(_gui._file_min_lbl, "file.label.min")
        _gui._file_min_speakers_entry = ttk.Entry(
            diar_opts,
            textvariable=_gui._file_min_speakers_var,
            width=4,
        )
        _gui._file_min_speakers_entry.pack(side=tk.LEFT, padx=(0, 6))
        _gui._bind_tooltip(_gui._file_min_speakers_entry, "tooltip.file.min_speakers")

        _gui._file_max_lbl = ttk.Label(diar_opts, text="")
        _gui._file_max_lbl.pack(side=tk.LEFT, padx=(0, 4))
        _gui._bind_text(_gui._file_max_lbl, "file.label.max")
        _gui._file_max_speakers_entry = ttk.Entry(
            diar_opts,
            textvariable=_gui._file_max_speakers_var,
            width=4,
        )
        _gui._file_max_speakers_entry.pack(side=tk.LEFT, padx=(0, 6))
        _gui._bind_tooltip(_gui._file_max_speakers_entry, "tooltip.file.max_speakers")

        _gui._file_detect_btn = ttk.Button(
            diar_opts,
            text="",
            command=_gui._detect_speakers,
            width=7,
        )
        _gui._file_detect_btn.pack(side=tk.LEFT, padx=(0, 4))
        _gui._bind_text(_gui._file_detect_btn, "file.button.detect")
        _gui._bind_tooltip(_gui._file_detect_btn, "tooltip.file.detect")

        _gui._file_download_btn = ttk.Button(opts, text="", command=_gui._download_file_model)
        _gui._file_download_btn.pack(side=tk.LEFT, padx=(0, 12))
        _gui._bind_text(_gui._file_download_btn, "file.button.download_model")
        _gui._bind_tooltip(_gui._file_download_btn, "tooltip.file.download_model")

        _gui._file_transcribe_btn = ttk.Button(
            opts,
            text="",
            command=_gui._start_file_transcribe,
            style="Accent.TButton",
        )
        _gui._file_transcribe_btn.pack(side=tk.LEFT, padx=(0, 4))
        _gui._bind_text(_gui._file_transcribe_btn, "file.button.transcribe_queue")
        _gui._bind_tooltip(_gui._file_transcribe_btn, "tooltip.file.transcribe_queue")
        _gui._file_cancel_btn = ttk.Button(
            opts, text="", command=_gui._cancel_file_transcribe, state=tk.DISABLED
        )
        _gui._file_cancel_btn.pack(side=tk.LEFT, padx=(0, 4))
        _gui._bind_text(_gui._file_cancel_btn, "file.button.cancel")
        _gui._bind_tooltip(_gui._file_cancel_btn, "tooltip.file.cancel")

        # -- Status + progress row --
        prog_row = ttk.Frame(transcribe_box)
        prog_row.pack(fill=tk.X, pady=(0, 4))

        _gui._file_progress = ttk.Progressbar(
            prog_row, orient="horizontal", length=180, mode="determinate", maximum=100
        )
        _gui._file_progress.pack(side=tk.RIGHT)
        _gui._bind_tooltip(_gui._file_progress, "tooltip.file.progress")

        _gui._file_time_label = ttk.Label(prog_row, text="", anchor="e", width=18)
        _gui._file_time_label.pack(side=tk.RIGHT, padx=(0, 6))

        _gui._file_status_label = ttk.Label(prog_row, text="", anchor="w")
        _gui._bind_tooltip(_gui._file_status_label, "tooltip.file.status")

        _gui._file_artifact_label = ttk.Label(
            transcribe_box,
            text="",
            anchor="w",
            foreground="#555555",
        )
        _gui._bind_tooltip(_gui._file_artifact_label, "tooltip.file.artifact")

        _gui._file_model_summary = ModelSummaryCard(
            top,
            title=_gui._tr("model_summary.title.file"),
            translate=_gui._tr,
        )
        top.add(_gui._file_model_summary, weight=2)

        # -- Results table (collapsible body, header always visible) --
        results_section = ttk.Frame(file_paned)
        file_paned.add(results_section, weight=6)
        results_hdr = ttk.Frame(results_section)
        results_hdr.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(results_hdr, text="Result", font=("", 9, "bold")).pack(side=tk.LEFT)
        _gui._file_results_toggle_btn = ttk.Button(results_hdr, text="▸", width=3)
        _gui._file_results_toggle_btn.pack(side=tk.RIGHT)
        results_frame = ttk.Frame(results_section)
        _gui._file_results_collapsed = True
        _gui._llm_section_collapsed = True

        def _apply_collapsed_layout() -> None:
            with suppress(Exception):
                file_paned.update_idletasks()
                total_h = file_paned.winfo_height()
                if total_h <= 0:
                    return
                top_target = max(220, int(total_h * 0.42))
                if _gui._file_results_collapsed and _gui._llm_section_collapsed:
                    file_paned.sashpos(0, total_h - 64)
                    file_paned.sashpos(1, total_h - 32)
                elif _gui._file_results_collapsed:
                    file_paned.sashpos(0, total_h - max(220, int(total_h * 0.35)))
                    file_paned.sashpos(1, total_h - 32)
                elif _gui._llm_section_collapsed:
                    file_paned.sashpos(0, top_target)
                    file_paned.sashpos(1, total_h - 32)
                else:
                    file_paned.sashpos(0, top_target)
                    file_paned.sashpos(1, max(top_target + 160, int(total_h * 0.78)))

        def _toggle_results() -> None:
            collapsed = getattr(_gui, "_file_results_collapsed", False)
            if collapsed:
                results_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=(0, 0))
                _gui._file_results_collapsed = False
                _gui._file_results_toggle_btn.configure(text="▾")
            else:
                results_frame.pack_forget()
                _gui._file_results_collapsed = True
                _gui._file_results_toggle_btn.configure(text="▸")
            _gui.root.after(0, _apply_collapsed_layout)

        _gui._file_results_toggle_btn.configure(command=_toggle_results)

        file_table_frame = ttk.Frame(results_frame)
        file_table_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=(0, 4))

        file_cols = ("time", "speaker", "text")
        _gui._file_table = ttk.Treeview(file_table_frame, columns=file_cols, show="headings")
        _gui._bind_tree_heading(_gui._file_table, "time", "file.table.timestamp")
        _gui._bind_tree_heading(_gui._file_table, "speaker", "file.table.speaker")
        _gui._bind_tree_heading(_gui._file_table, "text", "file.table.text")
        _gui._file_table.column("time", width=90, minwidth=70, stretch=False)
        _gui._file_table.column("speaker", width=110, minwidth=80, stretch=False)
        _gui._file_table.column("text", width=800, minwidth=300)
        _gui._file_table.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        _gui._bind_tooltip(_gui._file_table, "tooltip.file.results_table")
        _gui._file_table.bind("<Control-c>", _gui._file_copy_selected)
        _gui._file_table.bind(
            "<Control-a>",
            lambda _e: _gui._file_table.selection_set(_gui._file_table.get_children()),
        )

        file_scroll = ttk.Scrollbar(
            file_table_frame, orient=tk.VERTICAL, command=_gui._file_table.yview
        )
        file_scroll.pack(fill=tk.Y, side=tk.RIGHT)
        _gui._file_table.configure(yscrollcommand=file_scroll.set)

        # -- File controls --
        file_ctrl = ttk.Frame(results_frame)
        file_ctrl.pack(fill=tk.X, padx=0, pady=(0, 4))

        _gui._file_load_transcript_btn = ttk.Button(
            file_ctrl, text="", command=_gui._load_transcript_file
        )
        _gui._file_load_transcript_btn.pack(side=tk.LEFT, padx=(0, 4))
        _gui._bind_text(_gui._file_load_transcript_btn, "file.button.load_transcript")
        _gui._bind_tooltip(_gui._file_load_transcript_btn, "tooltip.file.load_transcript")
        _gui._file_clear_btn = ttk.Button(file_ctrl, text="", command=_gui._clear_file_table)
        _gui._file_clear_btn.pack(side=tk.LEFT, padx=(0, 4))
        _gui._bind_text(_gui._file_clear_btn, "file.button.clear")
        _gui._bind_tooltip(_gui._file_clear_btn, "tooltip.file.clear")
        _gui._file_save_btn = ttk.Button(file_ctrl, text="", command=_gui._save_file_result)
        _gui._file_save_btn.pack(side=tk.LEFT)
        _gui._bind_text(_gui._file_save_btn, "file.button.save")
        _gui._bind_tooltip(_gui._file_save_btn, "tooltip.file.save")
        _gui._file_seg_counter_label = ttk.Label(file_ctrl, text="")
        _gui._file_seg_counter_label.pack(side=tk.RIGHT)
        _gui._bind_tooltip(_gui._file_seg_counter_label, "tooltip.file.segment_counter")

        # -- LLM post-processing panel (collapsible) --
        llm_section = ttk.Frame(file_paned)
        file_paned.add(llm_section, weight=2)
        llm_section_hdr = ttk.Frame(llm_section)
        llm_section_hdr.pack(fill=tk.X, pady=(0, 2))
        _gui._llm_section_label = ttk.Label(llm_section_hdr, text="", font=("", 9, "bold"))
        _gui._llm_section_label.pack(side=tk.LEFT)
        _gui._bind_text(_gui._llm_section_label, "file.section.transcript_processing")
        _gui._llm_section_toggle_btn = ttk.Button(llm_section_hdr, text="▸", width=3)
        _gui._llm_section_toggle_btn.pack(side=tk.RIGHT)
        llm_box = ttk.Frame(llm_section, padding=8)
        def _toggle_llm() -> None:
            collapsed = getattr(_gui, "_llm_section_collapsed", False)
            if collapsed:
                llm_box.pack(fill=tk.BOTH, expand=True, pady=(0, 0))
                _gui._llm_section_collapsed = False
                _gui._llm_section_toggle_btn.configure(text="▾")
            else:
                llm_box.pack_forget()
                _gui._llm_section_collapsed = True
                _gui._llm_section_toggle_btn.configure(text="▸")
            _gui.root.after(0, _apply_collapsed_layout)

        _gui._llm_section_toggle_btn.configure(command=_toggle_llm)

        llm_hdr = ttk.Frame(llm_box)
        llm_hdr.pack(fill=tk.X, pady=(0, 2))
        _gui._llm_header_label = ttk.Label(llm_hdr, text="", font=("", 9, "bold"))
        _gui._llm_header_label.pack(side=tk.LEFT)
        _gui._bind_text(_gui._llm_header_label, "llm.header")

        _gui._llm_export_btn = ttk.Button(llm_hdr, text="", command=_gui._save_file_result)
        _gui._llm_export_btn.pack(side=tk.RIGHT, padx=(4, 0))
        _gui._bind_text(_gui._llm_export_btn, "file.button.save")
        _gui._bind_tooltip(_gui._llm_export_btn, "tooltip.file.save")

        _gui._llm_load_transcript_btn = ttk.Button(
            llm_hdr,
            text="",
            command=_gui._load_transcript_file,
        )
        _gui._llm_load_transcript_btn.pack(side=tk.RIGHT, padx=(8, 0))
        _gui._bind_text(_gui._llm_load_transcript_btn, "file.button.load_transcript")
        _gui._bind_tooltip(_gui._llm_load_transcript_btn, "tooltip.file.load_transcript")

        llm_cfg = ttk.Frame(llm_box)
        llm_cfg.pack(fill=tk.X, pady=(0, 2))

        _gui._llm_url_label = ttk.Label(llm_cfg, text="")
        _gui._llm_url_label.pack(side=tk.LEFT, padx=(0, 4))
        _gui._bind_text(_gui._llm_url_label, "llm.label.url")
        _gui._llm_url_entry = ttk.Entry(llm_cfg, textvariable=_gui._llm_url_var, width=26)
        _gui._llm_url_entry.pack(side=tk.LEFT, padx=(0, 10))
        _gui._bind_tooltip(_gui._llm_url_entry, "tooltip.llm.url")
        _gui._llm_refresh_btn = ttk.Button(llm_cfg, text="", command=_gui._refresh_llm_models)
        _gui._llm_refresh_btn.pack(side=tk.LEFT, padx=(0, 4))
        _gui._bind_text(_gui._llm_refresh_btn, "file.button.refresh_models")
        _gui._bind_tooltip(_gui._llm_refresh_btn, "tooltip.llm.refresh_models")
        _gui._llm_probe_btn = ttk.Button(llm_cfg, text="", command=_gui._probe_llm_model)
        _gui._llm_probe_btn.pack(side=tk.LEFT, padx=(0, 10))
        _gui._bind_text(_gui._llm_probe_btn, "llm.button.test_model")
        _gui._bind_tooltip(_gui._llm_probe_btn, "tooltip.llm.test_model")
        _gui._llm_model_label = ttk.Label(llm_cfg, text="")
        _gui._llm_model_label.pack(side=tk.LEFT, padx=(0, 4))
        _gui._bind_text(_gui._llm_model_label, "llm.label.model")
        _gui._llm_model_combo = ttk.Combobox(
            llm_cfg,
            textvariable=_gui._llm_model_var,
            width=24,
        )
        _gui._llm_model_combo.pack(side=tk.LEFT, padx=(0, 10))
        _gui._bind_tooltip(_gui._llm_model_combo, "tooltip.llm.model")
        _gui._llm_api_key_label = ttk.Label(llm_cfg, text="")
        _gui._llm_api_key_label.pack(side=tk.LEFT, padx=(0, 4))
        _gui._bind_text(_gui._llm_api_key_label, "llm.label.api_key")
        _gui._llm_key_entry = ttk.Entry(llm_cfg, textvariable=_gui._llm_key_var, width=14, show="*")
        _gui._llm_key_entry.pack(side=tk.LEFT, padx=(0, 10))
        _gui._bind_tooltip(_gui._llm_key_entry, "tooltip.llm.api_key")
        _gui._llm_prompt_label = ttk.Label(llm_cfg, text="")
        _gui._llm_prompt_label.pack(side=tk.LEFT, padx=(0, 4))
        _gui._bind_text(_gui._llm_prompt_label, "llm.label.prompt")
        _gui._llm_prompt_combo = ttk.Combobox(
            llm_cfg,
            textvariable=_gui._llm_prompt_var,
            state="readonly",
            width=14,
            values=tuple(BUILTIN_PROMPTS.keys()),
        )
        _gui._llm_prompt_combo.pack(side=tk.LEFT, padx=(0, 6))
        _gui._bind_tooltip(_gui._llm_prompt_combo, "tooltip.llm.prompt")
        _gui._llm_prompt_btn = ttk.Button(llm_cfg, text="", command=_gui._open_prompt_editor)
        _gui._llm_prompt_btn.pack(side=tk.LEFT, padx=(0, 10))
        _gui._bind_text(_gui._llm_prompt_btn, "file.button.prompt")
        _gui._bind_tooltip(_gui._llm_prompt_btn, "tooltip.llm.prompt_edit")
        _gui._llm_summarize_btn = ttk.Button(
            llm_cfg, text="", command=_gui._start_llm_summarize, style="Accent.TButton"
        )
        _gui._llm_summarize_btn.pack(side=tk.LEFT, padx=(0, 4))
        _gui._bind_text(_gui._llm_summarize_btn, "file.button.send_to_llm")
        _gui._bind_tooltip(_gui._llm_summarize_btn, "tooltip.llm.send")
        _gui._llm_cancel_btn = ttk.Button(
            llm_cfg, text="", command=_gui._cancel_llm_summarize, state=tk.DISABLED
        )
        _gui._llm_cancel_btn.pack(side=tk.LEFT, padx=(0, 4))
        _gui._bind_text(_gui._llm_cancel_btn, "llm.button.cancel")
        _gui._bind_tooltip(_gui._llm_cancel_btn, "tooltip.llm.cancel")
        _gui._llm_copy_btn = ttk.Button(llm_cfg, text="", command=_gui._copy_llm_output)
        _gui._llm_copy_btn.pack(side=tk.LEFT, padx=(0, 4))
        _gui._bind_text(_gui._llm_copy_btn, "file.button.copy")
        _gui._bind_tooltip(_gui._llm_copy_btn, "tooltip.llm.copy")
        _gui._llm_clear_btn = ttk.Button(llm_cfg, text="", command=_gui._clear_llm_output)
        _gui._llm_clear_btn.pack(side=tk.LEFT)
        _gui._bind_text(_gui._llm_clear_btn, "file.button.clear")
        _gui._bind_tooltip(_gui._llm_clear_btn, "tooltip.llm.clear")

        _gui._llm_status_label = ttk.Label(llm_cfg, text="", anchor="w", foreground="#555555")
        _gui._llm_status_label.pack(side=tk.LEFT, padx=(12, 0))
        _gui._bind_tooltip(_gui._llm_status_label, "tooltip.llm.status")

        llm_ctx_cfg = ttk.Frame(llm_box)
        llm_ctx_cfg.pack(fill=tk.X, pady=(0, 2))
        _gui._llm_context_label = ttk.Label(llm_ctx_cfg, text="")
        _gui._llm_context_label.pack(side=tk.LEFT, padx=(0, 4))
        _gui._bind_text(_gui._llm_context_label, "llm.label.context")
        _gui._llm_context_entry = ttk.Entry(
            llm_ctx_cfg, textvariable=_gui._llm_context_var, width=8
        )
        _gui._llm_context_entry.pack(side=tk.LEFT, padx=(0, 8))
        _gui._bind_tooltip(_gui._llm_context_entry, "tooltip.llm.context")
        _gui._llm_context_hint_label = ttk.Label(
            llm_ctx_cfg, text="", anchor="w", foreground="#666666"
        )
        _gui._llm_context_hint_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        llm_out_frame = ttk.Frame(llm_box)
        llm_out_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 0))

        _gui._llm_output = scrolledtext.ScrolledText(
            llm_out_frame,
            height=10,
            wrap=tk.WORD,
            state=tk.DISABLED,
        )
        _gui._llm_output.pack(fill=tk.BOTH, expand=True)
        _gui._bind_tooltip(_gui._llm_output, "tooltip.llm.output")

        _gui._refresh_file_diarization_controls()
        _gui._refresh_file_workflow()
        _gui._refresh_llm_context_hint()
        _gui._register_dnd_drop_targets()
        def _collapse_panels() -> None:
            with suppress(Exception):
                results_frame.pack_forget()
                llm_box.pack_forget()
                _gui._file_results_collapsed = True
                _gui._llm_section_collapsed = True
                _gui._file_results_toggle_btn.configure(text="▸")
                _gui._llm_section_toggle_btn.configure(text="▸")
                _apply_collapsed_layout()
        file_paned.bind("<Configure>", lambda _e: _apply_collapsed_layout())
        _gui.root.after(0, _collapse_panels)
