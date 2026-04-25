"""Model summary widget for catalog-driven GUI views."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import ttk

from voxfusion.asr_catalog import ASRModelInfo, get_model_info


class ModelSummaryCard(ttk.LabelFrame):
    """Compact summary card for the selected ASR model."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        title: str = "Model Overview",
        translate: Callable[[str], str] | None = None,
    ) -> None:
        super().__init__(parent, text=title, padding=12)
        self._translate = translate or (lambda text: text)
        self._current_model_id = "small"
        self.columnconfigure(0, weight=1)

        self._name_label = ttk.Label(self, style="Header.TLabel")
        self._name_label.grid(row=0, column=0, sticky="w")

        self._desc_label = ttk.Label(self, wraplength=360, justify=tk.LEFT)
        self._desc_label.grid(row=1, column=0, sticky="we", pady=(4, 8))

        self._meta_label = ttk.Label(self, style="Muted.TLabel", wraplength=360, justify=tk.LEFT)
        self._meta_label.grid(row=2, column=0, sticky="we", pady=(0, 8))

        self._recommended_label = ttk.Label(self, foreground="#7a5600")
        self._recommended_label.grid(row=3, column=0, sticky="w", pady=(0, 8))

        self._accuracy = ttk.Progressbar(self, orient="horizontal", mode="determinate", maximum=100)
        self._speed = ttk.Progressbar(self, orient="horizontal", mode="determinate", maximum=100)
        self._accuracy.grid(row=4, column=0, sticky="we", pady=(0, 2))
        self._speed.grid(row=6, column=0, sticky="we", pady=(0, 2))

        self._accuracy_label = ttk.Label(self, style="Muted.TLabel")
        self._speed_label = ttk.Label(self, style="Muted.TLabel")
        self._accuracy_label.grid(row=5, column=0, sticky="w", pady=(0, 6))
        self._speed_label.grid(row=7, column=0, sticky="w")

        self.set_model(self._current_model_id)

    def set_model(self, model_id: str) -> None:
        """Refresh the card from catalog metadata."""
        self._current_model_id = model_id
        model = get_model_info(model_id)
        self._render(model)

    def set_translator(self, translate: Callable[[str], str] | None) -> None:
        """Update the translator and rerender the current model."""
        self._translate = translate or (lambda text: text)
        self.set_model(self._current_model_id)

    def _render(self, model: ASRModelInfo) -> None:
        language_count = len(model.supported_languages)
        translation_text = self._translate(
            "model_summary.mode.translation"
            if model.supports_translation
            else "model_summary.mode.transcription_only"
        )

        self._name_label.configure(text=f"{model.name}  [{model.engine}]")
        self._desc_label.configure(text=model.description)
        self._meta_label.configure(
            text=self._translate(
                "model_summary.meta",
            ).format(
                language_count=language_count,
                mode=translation_text,
                model_id=model.id,
            )
        )
        self._recommended_label.configure(
            text=(self._translate("model_summary.recommended") if model.recommended else "")
        )
        self._accuracy["value"] = int(model.accuracy_score * 100)
        self._speed["value"] = int(model.speed_score * 100)
        self._accuracy_label.configure(
            text=self._translate("model_summary.accuracy").format(
                score=int(model.accuracy_score * 100)
            )
        )
        self._speed_label.configure(
            text=self._translate("model_summary.speed").format(score=int(model.speed_score * 100))
        )
