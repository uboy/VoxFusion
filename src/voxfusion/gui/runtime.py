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
from voxfusion.diarization.channel import ChannelDiarizer
from voxfusion.gui.progress import close_all_progress, get_stage_progress
from voxfusion.llm.client import LLMError, stream_completion
from voxfusion.llm.prompts import build_messages
from voxfusion.models.translation import TranslatedSegment
from voxfusion.pipeline.streaming import StreamingPipeline
from voxfusion.preprocessing.normalize import Normalizer
from voxfusion.preprocessing.pipeline import PreProcessingPipeline
from voxfusion.preprocessing.resample import Resampler
from voxfusion.recording import AudioRecorder, RecordingStats, create_recording_source


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
        return text


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
    output_format: str = "wav"


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
    if microphone_device_id and system_device_id:
        return "both"
    if system_device_id:
        return "system"
    if microphone_device_id:
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
    ) -> None:
        self._file_path = file_path
        self._model = model
        self._language = language
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
        from voxfusion.pipeline.events import EventType, PipelineStage
        from voxfusion.pipeline.orchestrator import PipelineOrchestrator

        overrides: dict[str, Any] = {
            "asr": {
                "model_size": self._model,
                "cpu_threads": os.cpu_count() or 4,
                "beam_size": 5,
            },
        }
        if self._language:
            overrides["asr"]["language"] = self._language

        config = load_config(overrides)
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

        def on_event(event: PipelineEvent) -> None:
            match event.event_type:
                case EventType.PIPELINE_STARTED:
                    self._on_status(event.message, 0.02)
                case EventType.STAGE_STARTED:
                    self._on_status(event.message, stage_started_pct.get(event.stage, 0.0))
                case EventType.STAGE_COMPLETED:
                    self._on_status(event.message, stage_done_pct.get(event.stage, 0.0))
                case EventType.PIPELINE_COMPLETED:
                    self._on_status(event.message, 1.0)
                case EventType.PIPELINE_FAILED:
                    self._on_status(f"Failed: {event.message}", 0.0)

        # Emit model download / cache hints to the log before loading
        _cache = Path.home() / ".cache" / "huggingface" / "hub"
        _engine = config.asr.engine
        if _engine == "faster-whisper":
            _repo = f"Systran/faster-whisper-{self._model}"
        elif _engine == "gigaam":
            _repo = "salute-developers/GigaAM-CTC-v3"
        else:
            _repo = None
        print(f"[VoxFusion] Model: '{self._model}'  |  HF cache: {_cache}")
        if _repo:
            print(f"[VoxFusion] Pre-download (avoids waiting at startup):")
            print(f"[VoxFusion]   huggingface-cli download {_repo}")
            print(f"[VoxFusion]   -- or place the folder directly in: {_cache}")
        print(f"[VoxFusion] Faster HF downloads: pip install hf-xet")

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
            self._on_segments(result.segments)
        finally:
            orchestrator.close()


class LLMWorker:
    """Streams an LLM response from Open WebUI in a background thread."""

    def __init__(
        self,
        text: str,
        model: str,
        base_url: str,
        api_key: str,
        prompt_name: str,
        custom_user_prompt: str | None,
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
        self._on_token = on_token
        self._on_error = on_error
        self._on_finished = on_finished
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            asyncio.run(self._run_async())
        except Exception as exc:
            self._on_error(str(exc))
        finally:
            self._on_finished()

    async def _run_async(self) -> None:
        messages = build_messages(
            self._prompt_name,
            self._text,
            custom_user=self._custom_user_prompt,
        )
        try:
            async for token in stream_completion(
                messages,
                base_url=self._base_url,
                model=self._model,
                api_key=self._api_key,
            ):
                self._on_token(token)
        except LLMError as exc:
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
                    else ["system"] if self._options.system_device_id else ["microphone"]
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
        on_drop: Callable[[str, str], None] | None = None,
    ) -> None:
        self._options = options
        self._on_status = on_status
        self._on_segment = on_segment
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

        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        warnings.filterwarnings(
            "ignore",
            message="`huggingface_hub` cache-system uses symlinks by default.*",
            category=UserWarning,
        )

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
                f"Loading: {model_label}  [{backend_info}]  "
                "Delays possible — 'small' is faster"
            )
        elif asr_backend == "openvino":
            self._on_status(
                f"Loading: {model_label}  [{backend_info}]  "
                "(first run: model conversion ~5 min)"
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
            nonlocal last_segment_ts, waiting_hint_shown
            last_segment_ts = monotonic()
            waiting_hint_shown = False
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
        waiting_hint_shown = False
        await audio_source.start()
        capture_start_time = datetime.now()

        from voxfusion.capture.mixer import AudioMixer

        if isinstance(audio_source, AudioMixer) and len(audio_source._sources) < 2:
            self._on_status("Capture started (loopback unavailable — mic only). Waiting for speech...")
        else:
            self._on_status("Capture started. Waiting for speech...")
        try:
            pipeline_task = asyncio.create_task(pipeline.run(audio_source, on_segments=on_segments))
            while not self._stop_event.is_set() and not pipeline_task.done():
                if not waiting_hint_shown and monotonic() - last_segment_ts >= 10:
                    self._on_status(
                        "Capture started. No speech segments yet — check microphone level/device."
                    )
                    waiting_hint_shown = True
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
