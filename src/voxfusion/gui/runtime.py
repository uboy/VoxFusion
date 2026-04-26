"""Runtime workers and Tk bridge helpers for the GUI."""

from __future__ import annotations

import asyncio
import os
import re
import sys
import threading
import tkinter as tk
import warnings
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from time import monotonic
from tkinter import scrolledtext
from typing import Any

from voxfusion.config.loader import load_config
from voxfusion.config.models import PipelineConfig
from voxfusion.diarization.channel import ChannelDiarizer
from voxfusion.gui.progress import close_all_progress, get_stage_progress
from voxfusion.llm.client import LLMError, complete, stream_completion
from voxfusion.llm.prompts import build_chunk_messages, build_merge_messages, build_messages
from voxfusion.logging import _should_suppress_log_message, get_logger
from voxfusion.models.translation import TranslatedSegment
from voxfusion.pipeline.streaming import StreamingPipeline
from voxfusion.preprocessing.normalize import Normalizer
from voxfusion.preprocessing.pipeline import PreProcessingPipeline
from voxfusion.preprocessing.resample import Resampler
from voxfusion.recording import AudioRecorder, RecordingStats, create_recording_source

log = get_logger(__name__)


def _describe_active_live_sources(audio_source: object) -> tuple[int, list[str], str | None]:
    from voxfusion.capture.mixer import AudioMixer

    if isinstance(audio_source, AudioMixer):
        names = [getattr(source, "device_name", "unknown") for source in audio_source._sources]
    else:
        names = [getattr(audio_source, "device_name", "unknown")]
    lowered = [str(name).lower() for name in names]
    if lowered and all(("loopback" in name or "system" in name) for name in lowered):
        mode = "system_only"
    elif lowered and all("microphone" in name for name in lowered):
        mode = "microphone_only"
    else:
        mode = None
    return len(names), names, mode


_LLM_CONTEXT_TOKEN_ENV = "VOXFUSION_LLM_CONTEXT_TOKENS"
_LLM_DEFAULT_CONTEXT_TOKENS = 2048
# Conservative chars-per-token used when computing a *character budget* from a token limit.
# Value 2 is intentionally low: 1 Russian char ≈ 2 UTF-8 bytes → ≈ 0.5 tokens (4 bytes/token),
# so 1 token ≈ 2 Russian chars.  For ASCII text this wastes a little space but never overflows.
_LLM_CHARS_PER_TOKEN_BUDGET = 2
# UTF-8 bytes per BPE token (average across typical multilingual BPE vocabularies).
_LLM_UTF8_BYTES_PER_TOKEN = 4
_LLM_RESERVED_COMPLETION_TOKENS = 384
_LLM_MIN_CHUNK_INPUT_TOKENS = 256
_LLM_MAX_MERGE_ROUNDS = 6
_LLM_CONTEXT_ERROR_MARKERS = (
    "maximum context length",
    "reduce the length of the input prompt",
    "prompt contains at least",
    "context length",
)


def _configure_gui_noise_controls() -> None:
    """Suppress safe third-party noise and set runtime env defaults for the GUI."""
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")
    thread_count = min(os.cpu_count() or 4, 16)
    os.environ.setdefault("NUMEXPR_MAX_THREADS", str(thread_count))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(thread_count))
    warnings.filterwarnings(
        "ignore",
        message="`huggingface_hub` cache-system uses symlinks by default.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*torchcodec is not installed correctly so built-in audio decoding will fail.*",
        category=UserWarning,
    )


class TextRedirector:
    """Thread-safe redirector from stdout/stderr into a Tk text widget."""

    def __init__(self, widget: scrolledtext.ScrolledText) -> None:
        self._widget = widget
        self._ansi_re = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
        self._orphan_ansi_token_re = re.compile(r"\[(?:\d{1,3}(?:;\d{1,3})*)m|\[A")
        self._buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        clean = self._sanitize(text)
        if not clean:
            return len(text)
        try:
            self._widget.after(0, self._append, clean)
        except RuntimeError:
            pass
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            buffered = self._buffer
            self._buffer = ""
            try:
                self._widget.after(0, self._append, buffered)
            except RuntimeError:
                pass

    def readable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def _append(self, text: str) -> None:
        self._widget.configure(state=tk.NORMAL)
        self._widget.insert(tk.END, text)
        self._widget.see(tk.END)
        self._widget.configure(state=tk.DISABLED)

    def _sanitize(self, text: str) -> str:
        text = self._ansi_re.sub("", text)
        text = self._orphan_ansi_token_re.sub("", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        if "\n" not in text:
            self._buffer += text
            return ""
        text = self._buffer + text
        self._buffer = ""
        kept_lines = [
            line
            for line in text.splitlines(keepends=True)
            if not _should_suppress_log_message(line)
        ]
        return "".join(kept_lines)


@dataclass(frozen=True)
class CaptureOptions:
    """GUI runtime options for live capture."""

    model: str
    language: str | None
    translate: str | None
    microphone_device_id: str | int | None
    system_device_id: str | int | None


@dataclass(frozen=True)
class RecordingOptions:
    """GUI runtime options for raw audio recording."""

    microphone_device_id: str | int | None
    system_device_id: str | int | None
    output_path: Path
    output_format: str = "mp3"


@dataclass(frozen=True)
class DeviceOption:
    """User-facing device selection option."""

    label: str
    index: str | int | None
    kind: str
    is_default: bool = False


def derive_capture_source(
    microphone_device_id: str | int | None,
    system_device_id: str | int | None,
) -> str:
    """Derive capture mode from explicit mic/system selections."""
    if microphone_device_id is not None and system_device_id is not None:
        return "both"
    if system_device_id is not None:
        return "system"
    if microphone_device_id is not None:
        return "microphone"
    return "none"


class FileTranscribeWorker:
    """Runs batch file transcription in a background thread."""

    def __init__(
        self,
        file_path: Path,
        model: str,
        language: str | None,
        on_status: Callable[[str, float], None],
        on_segments: Callable[[list[TranslatedSegment]], None],
        on_error: Callable[[str], None],
        on_finished: Callable[[], None],
        *,
        diarization_strategy: str = "auto",
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        quality: str = "balanced",
    ) -> None:
        self._file_path = file_path
        self._model = model
        self._language = language
        self._diarization_strategy = diarization_strategy
        self._min_speakers = min_speakers
        self._max_speakers = max_speakers
        self._quality = quality
        self._on_status = on_status
        self._on_segments = on_segments
        self._on_error = on_error
        self._on_finished = on_finished
        self._thread: threading.Thread | None = None
        self._cancelled = False

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        """Request cancellation of the running transcription."""
        self._cancelled = True

    def _run(self) -> None:
        try:
            asyncio.run(self._run_async())
        except Exception as exc:
            self._on_error(str(exc))
        finally:
            self._on_finished()

    async def _run_async(self) -> None:
        from voxfusion.asr_catalog import get_quality_preset
        from voxfusion.pipeline.events import EventType, PipelineStage
        from voxfusion.pipeline.orchestrator import PipelineOrchestrator

        _configure_gui_noise_controls()

        # Start with quality preset (compute_type, beam_size, best_of, vad_*),
        # then override model-level settings that always take priority.
        asr_overrides: dict[str, Any] = get_quality_preset(self._quality)
        asr_overrides.update(
            {
                "model_size": self._model,
                "cpu_threads": os.cpu_count() or 4,
            }
        )
        if self._language:
            asr_overrides["language"] = self._language
        overrides: dict[str, Any] = {
            "asr": asr_overrides,
            "diarization": {"strategy": self._diarization_strategy},
        }
        if self._min_speakers is not None or self._max_speakers is not None:
            overrides["diarization"]["ml"] = {}
            if self._min_speakers is not None:
                overrides["diarization"]["ml"]["min_speakers"] = self._min_speakers
            if self._max_speakers is not None:
                overrides["diarization"]["ml"]["max_speakers"] = self._max_speakers

        config = load_config(overrides)
        token_source = None
        if config.diarization.ml.hf_auth_token:
            token_source = "config-or-env:VOXFUSION"
        elif os.environ.get("HF_TOKEN"):
            token_source = "env:HF_TOKEN"
        elif os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            token_source = "env:HUGGING_FACE_HUB_TOKEN"
        log.info(
            "gui.file_transcribe_requested",
            file=str(self._file_path),
            model=self._model,
            asr_engine=config.asr.engine,
            language=self._language,
            quality=self._quality,
            diarization_strategy_requested=self._diarization_strategy,
            diarization_strategy_config=config.diarization.strategy,
            min_speakers=config.diarization.ml.min_speakers,
            max_speakers=config.diarization.ml.max_speakers,
            hf_token_present=token_source is not None,
            hf_token_source=token_source,
        )
        stage_started_pct: dict[PipelineStage, float] = {
            PipelineStage.CAPTURE: 0.05,
            PipelineStage.PREPROCESSING: 0.30,
            PipelineStage.ASR: 0.45,
            PipelineStage.DIARIZATION: 0.80,
        }
        stage_done_pct: dict[PipelineStage, float] = {
            PipelineStage.CAPTURE: 0.28,
            PipelineStage.PREPROCESSING: 0.43,
            PipelineStage.ASR: 0.78,
            PipelineStage.DIARIZATION: 0.95,
        }

        from voxfusion.pipeline.events import PipelineEvent

        last_progress = 0.0

        def on_event(event: PipelineEvent) -> None:
            nonlocal last_progress
            match event.event_type:
                case EventType.PIPELINE_STARTED:
                    last_progress = 0.02
                    self._on_status(event.message, last_progress)
                case EventType.STAGE_STARTED:
                    last_progress = stage_started_pct.get(event.stage, last_progress)
                    self._on_status(event.message, last_progress)
                case EventType.STAGE_COMPLETED:
                    last_progress = stage_done_pct.get(event.stage, last_progress)
                    self._on_status(event.message, last_progress)
                case EventType.PIPELINE_COMPLETED:
                    last_progress = 1.0
                    self._on_status(event.message, last_progress)
                case EventType.PROGRESS:
                    if event.progress > 0:
                        last_progress = max(last_progress, event.progress)
                    self._on_status(event.message, last_progress)
                case EventType.PIPELINE_FAILED:
                    self._on_status(f"Failed: {event.message}", 0.0)
                case EventType.WARNING:
                    self._on_status(f"Warning: {event.message}", last_progress)

        orchestrator = PipelineOrchestrator(config, on_event=on_event)
        self._on_status("Loading model...", 0.01)
        try:
            task = asyncio.create_task(orchestrator.transcribe_file(self._file_path))
            while not task.done():
                if self._cancelled:
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
                    self._on_status("Transcription cancelled.", 0.0)
                    return
                await asyncio.sleep(0.2)
            result = task.result()
            log.info(
                "gui.file_transcribe_completed",
                file=str(self._file_path),
                segments=len(result.segments),
                processing_info=result.processing_info,
            )
            self._on_segments(result.segments)
        finally:
            orchestrator.close()


class LLMWorker:
    """Streams or hierarchically summarizes an Open WebUI response in a background thread."""

    def __init__(
        self,
        text: str,
        model: str,
        base_url: str,
        api_key: str,
        prompt_name: str,
        custom_user_prompt: str | None,
        context_limit_tokens: int | None,
        on_token: Callable[[str], None],
        on_error: Callable[[str], None],
        on_finished: Callable[[], None],
    ) -> None:
        self._text = text
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        self._prompt_name = prompt_name
        self._custom_user_prompt = custom_user_prompt
        self._context_limit_tokens_explicit = context_limit_tokens
        self._on_token = on_token
        self._on_error = on_error
        self._on_finished = on_finished
        self._thread: threading.Thread | None = None
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation of the ongoing LLM request."""
        self._cancelled = True

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            asyncio.run(self._run_async())
        except Exception as exc:
            if not self._cancelled:
                self._on_error(str(exc))
        finally:
            self._on_finished()

    @staticmethod
    def _estimate_text_tokens(text: str) -> int:
        clean = str(text or "")
        if not clean:
            return 0
        utf8_bytes = len(clean.encode("utf-8"))
        return max(1, (utf8_bytes + _LLM_UTF8_BYTES_PER_TOKEN - 1) // _LLM_UTF8_BYTES_PER_TOKEN)

    @classmethod
    def _estimate_messages_tokens(cls, messages: list[dict[str, str]]) -> int:
        total = 0
        for message in messages:
            if not isinstance(message, dict):
                continue
            total += 8 + cls._estimate_text_tokens(str(message.get("content", "")))
        return total

    @staticmethod
    def _looks_like_context_length_error(error_text: str) -> bool:
        lowered = str(error_text or "").lower()
        return any(marker in lowered for marker in _LLM_CONTEXT_ERROR_MARKERS)

    def _context_limit_tokens(self) -> int:
        if self._context_limit_tokens_explicit is not None:
            return max(512, int(self._context_limit_tokens_explicit))
        raw = os.environ.get(_LLM_CONTEXT_TOKEN_ENV, "").strip()
        if raw.isdigit():
            return max(512, int(raw))
        return _LLM_DEFAULT_CONTEXT_TOKENS

    def _input_token_budget(self, messages: list[dict[str, str]]) -> int:
        available = (
            self._context_limit_tokens()
            - self._estimate_messages_tokens(messages)
            - _LLM_RESERVED_COMPLETION_TOKENS
        )
        return max(_LLM_MIN_CHUNK_INPUT_TOKENS, available)

    def _split_block_to_fit(self, block: str, max_tokens: int) -> list[str]:
        max_chars = max_tokens * _LLM_CHARS_PER_TOKEN_BUDGET
        clean = block.strip()
        if not clean:
            return []
        if len(clean) <= max_chars:
            return [clean]

        pieces: list[str] = []
        current = ""
        for word in clean.split():
            if len(word) > max_chars:
                if current:
                    pieces.append(current)
                    current = ""
                for start in range(0, len(word), max_chars):
                    pieces.append(word[start : start + max_chars])
                continue
            candidate = f"{current} {word}".strip()
            if current and len(candidate) > max_chars:
                pieces.append(current)
                current = word
            else:
                current = candidate
        if current:
            pieces.append(current)
        return pieces

    def _pack_text_blocks(self, blocks: list[str], max_tokens: int, *, separator: str) -> list[str]:
        expanded: list[str] = []
        for block in blocks:
            expanded.extend(self._split_block_to_fit(block, max_tokens))

        packed: list[str] = []
        current: list[str] = []
        current_tokens = 0
        separator_tokens = self._estimate_text_tokens(separator)
        for block in expanded:
            block_tokens = self._estimate_text_tokens(block)
            projected = (
                block_tokens if not current else current_tokens + separator_tokens + block_tokens
            )
            if current and projected > max_tokens:
                packed.append(separator.join(current))
                current = [block]
                current_tokens = block_tokens
                continue
            current.append(block)
            current_tokens = projected
        if current:
            packed.append(separator.join(current))
        return packed

    def _split_transcript_into_chunks(self) -> list[str]:
        blocks = [line.strip() for line in self._text.splitlines() if line.strip()]
        if not blocks and self._text.strip():
            blocks = [self._text.strip()]
        if not blocks:
            return []
        max_tokens = self._input_token_budget(
            build_chunk_messages(
                self._prompt_name,
                "",
                chunk_index=1,
                chunk_count=1,
                custom_user=self._custom_user_prompt,
            )
        )
        return self._pack_text_blocks(blocks, max_tokens, separator="\n")

    def _pack_partial_outputs(self, partial_outputs: list[str]) -> list[str]:
        numbered_blocks = [
            f"### Partial {index}\n{output.strip()}"
            for index, output in enumerate(partial_outputs, start=1)
            if output.strip()
        ]
        if not numbered_blocks:
            return []
        max_tokens = self._input_token_budget(
            build_merge_messages(
                self._prompt_name,
                "",
                custom_user=self._custom_user_prompt,
            )
        )
        return self._pack_text_blocks(numbered_blocks, max_tokens, separator="\n\n")

    def _needs_chunked_summary(self, messages: list[dict[str, str]]) -> bool:
        return self._estimate_messages_tokens(messages) > (
            self._context_limit_tokens() - _LLM_RESERVED_COMPLETION_TOKENS
        )

    async def _summarize_chunks(self, reason: str, initial_messages: list[dict[str, str]]) -> str:
        chunks = self._split_transcript_into_chunks()
        if not chunks:
            raise LLMError("No transcript text available for chunked summarization.")
        log.info(
            "llm.chunking.plan",
            model=self._model,
            reason=reason,
            transcript_chars=len(self._text),
            estimated_input_tokens=self._estimate_messages_tokens(initial_messages),
            context_tokens=self._context_limit_tokens(),
            chunk_count=len(chunks),
        )

        partial_outputs: list[str] = []
        for chunk_index, chunk_text in enumerate(chunks, start=1):
            if self._cancelled:
                return ""
            log.info(
                "llm.chunking.chunk_start",
                model=self._model,
                chunk_index=chunk_index,
                chunk_count=len(chunks),
                input_chars=len(chunk_text),
            )
            chunk_messages = build_chunk_messages(
                self._prompt_name,
                chunk_text,
                chunk_index=chunk_index,
                chunk_count=len(chunks),
                custom_user=self._custom_user_prompt,
            )
            chunk_output = (
                await complete(
                    chunk_messages,
                    base_url=self._base_url,
                    model=self._model,
                    api_key=self._api_key,
                )
            ).strip()
            log.info(
                "llm.chunking.chunk_done",
                model=self._model,
                chunk_index=chunk_index,
                chunk_count=len(chunks),
                output_chars=len(chunk_output),
            )
            if chunk_output:
                partial_outputs.append(chunk_output)

        if not partial_outputs:
            raise LLMError("LLM returned empty chunk summaries.")

        current_outputs = partial_outputs
        for round_index in range(1, _LLM_MAX_MERGE_ROUNDS + 1):
            if len(current_outputs) <= 1:
                return current_outputs[0]
            batches = self._pack_partial_outputs(current_outputs)
            if not batches:
                raise LLMError("Failed to prepare partial summaries for final merge.")
            log.info(
                "llm.chunking.merge_round",
                model=self._model,
                round_index=round_index,
                input_count=len(current_outputs),
                batch_count=len(batches),
            )
            next_outputs: list[str] = []
            for batch_index, batch_text in enumerate(batches, start=1):
                if self._cancelled:
                    return ""
                merge_messages = build_merge_messages(
                    self._prompt_name,
                    batch_text,
                    custom_user=self._custom_user_prompt,
                )
                merged_output = (
                    await complete(
                        merge_messages,
                        base_url=self._base_url,
                        model=self._model,
                        api_key=self._api_key,
                    )
                ).strip()
                log.info(
                    "llm.chunking.merge_batch_done",
                    model=self._model,
                    round_index=round_index,
                    batch_index=batch_index,
                    batch_count=len(batches),
                    output_chars=len(merged_output),
                )
                if merged_output:
                    next_outputs.append(merged_output)
            if not next_outputs:
                raise LLMError("LLM returned an empty merged summary.")
            current_outputs = next_outputs

        raise LLMError("Failed to merge chunk summaries within the allowed number of rounds.")

    async def _run_async(self) -> None:
        messages = build_messages(
            self._prompt_name,
            self._text,
            custom_user=self._custom_user_prompt,
        )
        if self._needs_chunked_summary(messages):
            try:
                final_output = await self._summarize_chunks("estimated_context", messages)
                if not self._cancelled and final_output:
                    self._on_token(final_output)
            except LLMError as exc:
                if not self._cancelled:
                    self._on_error(str(exc))
            return

        try:
            async for token in stream_completion(
                messages,
                base_url=self._base_url,
                model=self._model,
                api_key=self._api_key,
            ):
                if self._cancelled:
                    return
                self._on_token(token)
        except LLMError as exc:
            if self._looks_like_context_length_error(str(exc)):
                log.warning(
                    "llm.chunking.context_retry",
                    model=self._model,
                    reason="context_error",
                    error=str(exc),
                )
                try:
                    final_output = await self._summarize_chunks("context_error_retry", messages)
                    self._on_token(final_output)
                except LLMError as retry_exc:
                    self._on_error(str(retry_exc))
                return
            self._on_error(str(exc))


class RecordingWorker:
    """Runs audio-only recording in a daemon thread."""

    def __init__(
        self,
        options: RecordingOptions,
        on_status: Callable[[str], None],
        on_error: Callable[[str], None],
        on_finished: Callable[[RecordingStats | None], None],
    ) -> None:
        self._options = options
        self._on_status = on_status
        self._on_error = on_error
        self._on_finished = on_finished
        self._thread: threading.Thread | None = None
        self._recorder = AudioRecorder(on_status=on_status)
        self._start_time: float | None = None

    @property
    def elapsed_s(self) -> float:
        """Seconds elapsed since recording started."""
        return monotonic() - self._start_time if self._start_time is not None else 0.0

    @property
    def is_running(self) -> bool:
        """True while the recording thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._recorder.request_stop()

    def toggle_pause(self) -> bool:
        if self._recorder.is_paused:
            self._recorder.request_resume()
            self._on_status("Recording resumed.")
            return False
        self._recorder.request_pause()
        self._on_status("Recording paused.")
        return True

    def _run(self) -> None:
        result: RecordingStats | None = None
        try:
            result = asyncio.run(self._run_async())
        except KeyboardInterrupt:
            self._recorder.request_stop()
        except Exception as exc:  # pragma: no cover
            self._on_error(str(exc))
        finally:
            self._on_finished(result)

    async def _run_async(self) -> RecordingStats:
        overrides: dict[str, dict[str, object]] = {
            "capture": {
                "sources": (
                    ["microphone", "system"]
                    if self._options.microphone_device_id and self._options.system_device_id
                    else ["system"]
                    if self._options.system_device_id
                    else ["microphone"]
                ),
            }
        }
        config = load_config(overrides)
        audio_source = create_recording_source(
            derive_capture_source(
                self._options.microphone_device_id,
                self._options.system_device_id,
            ),
            config.capture,
            device_index=(
                self._options.system_device_id
                if self._options.system_device_id and not self._options.microphone_device_id
                else self._options.microphone_device_id
            ),
            microphone_device_id=self._options.microphone_device_id,
            system_device_id=self._options.system_device_id,
        )
        try:
            self._start_time = monotonic()
            return await self._recorder.record(
                audio_source,
                self._options.output_path,
                format=self._options.output_format,
            )
        finally:
            # Cancel any residual tasks left by WASAPI background readers so that
            # asyncio.run() does not print "unhandled exception during shutdown".
            current = asyncio.current_task()
            for task in asyncio.all_tasks():
                if task is not current:
                    task.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await task


class CaptureWorker:
    """Runs async capture pipeline in a daemon thread."""

    def __init__(
        self,
        options: CaptureOptions,
        on_status: Callable[[str], None],
        on_segment: Callable[[str, str, str, str | None], None],
        on_error: Callable[[str], None],
        on_finished: Callable[[], None],
        on_replace_segments: Callable[[list[tuple[str, str, str, str | None]]], None] | None = None,
        on_drop: Callable[[str, str], None] | None = None,
    ) -> None:
        self._options = options
        self._on_status = on_status
        self._on_segment = on_segment
        self._on_replace_segments = on_replace_segments or (lambda _rows: None)
        self._on_error = on_error
        self._on_finished = on_finished
        self._on_drop = on_drop or (lambda _t, _s: None)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._pipeline: object | None = None

    def get_stats(self) -> dict[str, int] | None:
        if self._pipeline is None:
            return None
        return self._pipeline.get_stats()  # type: ignore[union-attr]

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _run(self) -> None:
        had_error = False
        try:
            asyncio.run(self._run_async())
        except KeyboardInterrupt:
            self._stop_event.set()
        except Exception as exc:  # pragma: no cover
            had_error = True
            self._on_error(f"{exc}")
        finally:
            close_all_progress()
            if not had_error:
                self._on_status("Stopped")
            self._on_finished()

    async def _run_async(self) -> None:
        if sys.platform != "win32":
            raise RuntimeError("GUI live capture currently requires Windows WASAPI.")

        _configure_gui_noise_controls()

        self._on_status("Loading configuration...")
        cpu_threads = os.cpu_count() or 4
        overrides: dict[str, dict[str, object]] = {
            "capture": {"buffer_size": 20, "lossy_mode": True, "chunk_duration_ms": 5000},
            "asr": {
                "model_size": self._options.model,
                "vad_filter": False,
                "no_speech_threshold": 0.6,
                "beam_size": 1,
                "best_of": 1,
                "cpu_threads": cpu_threads,
            },
        }
        if self._options.language:
            overrides["asr"]["language"] = self._options.language
        if self._options.translate:
            overrides["translation"] = {
                "enabled": True,
                "target_language": self._options.translate,
            }

        config = load_config(overrides)
        if config.asr.engine == "gigaam":
            if self._options.translate:
                raise RuntimeError("Live GigaAM translation is not supported.")
            await self._run_live_gigaam_async(config)
            return

        preprocessor = PreProcessingPipeline([Resampler(16_000), Normalizer()])
        from voxfusion.asr.factory import create_asr_engine

        asr_engine, asr_backend = create_asr_engine(config.asr)
        diarizer = ChannelDiarizer(config.diarization)
        translator = None

        if self._options.translate:
            from voxfusion.translation.registry import get_translation_engine

            translator = get_translation_engine("argos", config.translation)

        backend_info = {
            "cuda": "GPU NVIDIA CUDA",
            "openvino": "Intel OpenVINO",
            "cpu": f"CPU ({os.cpu_count()} threads)",
        }.get(asr_backend, asr_backend)
        model_label = config.asr.model_size
        if asr_backend == "cpu" and model_label in ("large-v2", "large-v3"):
            self._on_status(
                f"Loading: {model_label}  [{backend_info}]  "
                "Real-time not achievable on CPU — use 'small'"
            )
        elif asr_backend == "cpu" and model_label == "medium":
            self._on_status(
                f"Loading: {model_label}  [{backend_info}]  Delays possible — 'small' is faster"
            )
        elif asr_backend == "openvino":
            self._on_status(
                f"Loading: {model_label}  [{backend_info}]  (first run: model conversion ~5 min)"
            )
        else:
            self._on_status(f"Loading: {model_label}  [{backend_info}]")

        loading_progress = get_stage_progress("model-load", total=1)
        asr_engine.load_model()
        loading_progress.update(1)
        self._on_status("Model loaded. Initializing audio capture...")

        def on_drop_chunk(chunk: object) -> None:
            from voxfusion.models.audio import AudioChunk

            source = chunk.source if isinstance(chunk, AudioChunk) else "unknown"  # type: ignore[union-attr]
            self._on_drop(datetime.now().strftime("%H:%M:%S"), source)

        pipeline = StreamingPipeline(
            asr_engine=asr_engine,
            diarizer=diarizer,
            preprocessor=preprocessor,
            translator=translator,
            config=config,
            on_drop=on_drop_chunk,
            queue_size=50,
        )
        self._pipeline = pipeline

        from voxfusion.capture.vad_chunker import VadChunker
        from voxfusion.capture.windows_factory import create_windows_capture_source

        source = derive_capture_source(
            self._options.microphone_device_id,
            self._options.system_device_id,
        )
        if source == "both":
            from voxfusion.capture.mixer import AudioMixer

            base_source = create_windows_capture_source(
                source,
                config.capture,
                microphone_device_id=self._options.microphone_device_id,
                system_device_id=self._options.system_device_id,
            )
            if not isinstance(base_source, AudioMixer):
                raise RuntimeError("Expected AudioMixer for Windows 'both' capture source.")
            mic_vad = VadChunker(base_source._sources[0], max_duration_ms=5000)
            sys_vad = VadChunker(base_source._sources[1], max_duration_ms=5000)
            audio_source: object = AudioMixer(sources=[mic_vad, sys_vad])
        else:
            base_source = create_windows_capture_source(
                source,
                config.capture,
                microphone_device_id=self._options.microphone_device_id,
                system_device_id=self._options.system_device_id,
            )
            audio_source = VadChunker(base_source, max_duration_ms=5000)

        segment_progress = get_stage_progress("segments")

        def on_segments(segments: list[TranslatedSegment]) -> None:
            nonlocal last_segment_ts, next_wait_log_ts
            last_segment_ts = monotonic()
            next_wait_log_ts = last_segment_ts + 10
            segment_progress.update(len(segments))
            for segment in segments:
                transcription = segment.diarized.segment
                speaker = segment.diarized.speaker_id
                spoken_at = capture_start_time + timedelta(seconds=transcription.start_time)
                self._on_segment(
                    spoken_at.strftime("%H:%M:%S"),
                    speaker,
                    transcription.text,
                    segment.translated_text,
                )

        self._on_status("Starting capture...")
        pipeline_task: asyncio.Task[None] | None = None
        last_segment_ts = monotonic()
        capture_loop_started_at = last_segment_ts
        next_wait_log_ts = capture_loop_started_at + 10
        await audio_source.start()
        capture_start_time = datetime.now()

        active_source_count, active_sources, surviving_mode = _describe_active_live_sources(
            audio_source
        )
        log.info(
            "gui.live_capture_started",
            requested_source=source,
            active_source_count=active_source_count,
            active_sources=active_sources,
        )

        if active_source_count < 2 and surviving_mode == "microphone_only":
            self._on_status("Capture started (microphone only). Waiting for speech...")
        elif active_source_count < 2 and surviving_mode == "system_only":
            self._on_status("Capture started (system audio only). Waiting for speech...")
        else:
            self._on_status("Capture started. Waiting for speech...")
        try:
            pipeline_task = asyncio.create_task(pipeline.run(audio_source, on_segments=on_segments))
            while not self._stop_event.is_set() and not pipeline_task.done():
                now = monotonic()
                if now >= next_wait_log_ts:
                    stats = pipeline.get_stats()
                    self._on_status(
                        "Capture started. No speech segments yet — check microphone level/device."
                    )
                    log.info(
                        "gui.live_waiting_for_segments",
                        elapsed_s=round(now - capture_loop_started_at, 1),
                        since_last_segment_s=round(now - last_segment_ts, 1),
                        pipeline_stats=stats or {},
                        active_sources=active_sources,
                    )
                    next_wait_log_ts = now + 10
                await asyncio.sleep(0.1)
            if self._stop_event.is_set() and pipeline_task is not None:
                pipeline_task.cancel()
                with suppress(asyncio.CancelledError):
                    await pipeline_task
            elif pipeline_task is not None:
                await pipeline_task
        finally:
            if pipeline_task is not None and not pipeline_task.done():
                pipeline_task.cancel()
                with suppress(asyncio.CancelledError):
                    await pipeline_task
            await pipeline.stop()
            await audio_source.stop()
            asr_engine.unload_model()
            asr_engine.close()

    async def _run_live_gigaam_async(self, config: PipelineConfig) -> None:
        from voxfusion.live_gigaam.session import LiveGigaAMSessionController

        segment_progress = get_stage_progress("segments")
        capture_start_time: datetime | None = None

        def _rows_for_segments(
            segments: list[TranslatedSegment],
        ) -> list[tuple[str, str, str, str | None]]:
            rows: list[tuple[str, str, str, str | None]] = []
            anchor = capture_start_time or datetime.now()
            for segment in segments:
                transcription = segment.diarized.segment
                speaker = segment.diarized.speaker_id
                spoken_at = anchor + timedelta(seconds=transcription.start_time)
                rows.append(
                    (
                        spoken_at.strftime("%H:%M:%S"),
                        speaker,
                        transcription.text,
                        segment.translated_text,
                    )
                )
            return rows

        def on_segments(segments: list[TranslatedSegment]) -> None:
            segment_progress.update(len(segments))
            for row in _rows_for_segments(segments):
                self._on_segment(*row)

        def on_finalized_segments(segments: list[TranslatedSegment]) -> None:
            rows = _rows_for_segments(segments)
            self._on_replace_segments(rows)

        def on_capture_started(started_at: datetime) -> None:
            nonlocal capture_start_time
            capture_start_time = started_at

        controller = LiveGigaAMSessionController(
            config=config,
            microphone_device_id=self._options.microphone_device_id,
            system_device_id=self._options.system_device_id,
            on_status=self._on_status,
            on_segments=on_segments,
            on_finalized_segments=on_finalized_segments,
            on_capture_started=on_capture_started,
            requested_source=derive_capture_source(
                self._options.microphone_device_id,
                self._options.system_device_id,
            ),
        )
        self._pipeline = controller
        await controller.run(self._stop_event)
