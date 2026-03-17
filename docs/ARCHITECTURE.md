# VoxFusion Architecture Document

> **Version**: 0.1.0
> **Status**: Implemented
> **Last Updated**: 2026-03-17

---

## Table of Contents

1. [System Overview and Goals](#1-system-overview-and-goals)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Module Decomposition and Interfaces](#3-module-decomposition-and-interfaces)
4. [Data Flow](#4-data-flow)
5. [Platform-Specific Audio Capture Strategy](#5-platform-specific-audio-capture-strategy)
6. [ASR Integration Strategy](#6-asr-integration-strategy)
7. [Speaker Diarization Approach](#7-speaker-diarization-approach)
8. [Translation Subsystem Design](#8-translation-subsystem-design)
9. [Output Formatting Subsystem](#9-output-formatting-subsystem)
10. [CLI and API Design](#10-cli-and-api-design)
11. [Configuration Management](#11-configuration-management)
12. [Error Handling Strategy](#12-error-handling-strategy)
13. [Security and Privacy](#13-security-and-privacy)
14. [Project Structure](#14-project-structure)
15. [MVP Plan vs Full Roadmap](#15-mvp-plan-vs-full-roadmap)
16. [Acceptance Criteria and Metrics](#16-acceptance-criteria-and-metrics)
17. [Architecture Decision Records](#17-architecture-decision-records)

---

## 1. System Overview and Goals

### What VoxFusion Is

VoxFusion is a cross-platform system that captures all audio played on a user's device (system audio loopback and microphone input), applies speech-to-text transcription with speaker diarization, and produces translated text in a user-selected language. It supports both real-time (streaming) and batch (file-based) workflows.

### Primary Goals

1. **Universal Audio Capture** -- Capture audio from any source on Windows, Linux, and macOS with a unified API that abstracts platform differences.
2. **High-Quality Speech Recognition** -- Deliver accurate ASR using faster-whisper (CTranslate2-backed Whisper large-v3) with automatic language detection.
3. **Speaker Diarization** -- Identify "who spoke when" using both channel-based heuristics and ML-based diarization (pyannote.audio).
4. **Real-Time Translation** -- Translate recognized text into target languages using pluggable backends (local NMT models and external APIs).
5. **Flexible Output** -- Export results in JSON, SRT, VTT, and plain text with full metadata.
6. **Cross-Platform Parity** -- Consistent behavior across Windows (WASAPI), macOS (CoreAudio), and Linux (PulseAudio/PipeWire).

### Design Principles

- **Modularity**: Each pipeline stage is an independent, replaceable component behind a well-defined interface.
- **Streaming-First**: The architecture is designed around streaming data flow with backpressure support; batch mode is a degenerate case of the streaming pipeline.
- **Configuration over Code**: Behavior is controlled through configuration files and CLI flags, not code changes.
- **Graceful Degradation**: If a component is unavailable (e.g., no GPU for diarization), the system falls back to simpler alternatives and logs a warning rather than failing.
- **Testability**: Every module can be tested in isolation with mock inputs and outputs.

---

## 2. High-Level Architecture

### Pipeline Diagram

```
+-------------------+     +---------------------+     +------------------+
|   Audio Capture   |---->|   Pre-Processing    |---->|   ASR Engine     |
|   Layer           |     |   Engine            |     |   (faster-whisper)|
| (WASAPI/Core/Pulse)|     | (resample, norm,   |     |                  |
+-------------------+     |  VAD, chunking)     |     +--------+---------+
                          +---------------------+              |
                                                               v
+-------------------+     +---------------------+     +------------------+
|   Output          |<----|   Translation       |<----|   Diarization    |
|   Formatter       |     |   Module            |     |   Module         |
| (JSON/SRT/VTT/TXT)|     | (local NMT / API)  |     | (channel + ML)   |
+-------------------+     +---------------------+     +------------------+
        |
        v
+-------------------+
|   Delivery Layer  |
|  (CLI / API /     |
|   WebSocket / UI) |
+-------------------+
```

### Data Model (Core Types)

The following data types flow through the pipeline. All are immutable dataclasses.

```python
@dataclass(frozen=True)
class AudioChunk:
    """A chunk of raw or processed audio data."""
    samples: numpy.ndarray          # shape: (num_samples,) or (num_samples, channels)
    sample_rate: int                # e.g. 16000
    channels: int                   # 1 (mono) or 2 (stereo)
    timestamp_start: float          # seconds from capture start
    timestamp_end: float            # seconds from capture start
    source: str                     # "system" | "microphone" | "file" | "mixed"
    dtype: str                      # "float32" | "int16"

@dataclass(frozen=True)
class TranscriptionSegment:
    """A single segment of transcribed speech."""
    text: str
    language: str                   # ISO 639-1 code, e.g. "en"
    start_time: float               # seconds
    end_time: float                 # seconds
    confidence: float               # 0.0 - 1.0
    words: list[WordTiming] | None  # optional word-level timings
    no_speech_prob: float           # probability of no speech

@dataclass(frozen=True)
class WordTiming:
    """Word-level timing information."""
    word: str
    start_time: float
    end_time: float
    probability: float

@dataclass(frozen=True)
class DiarizedSegment:
    """A transcription segment annotated with speaker identity."""
    segment: TranscriptionSegment
    speaker_id: str                 # e.g. "SPEAKER_00", "SPEAKER_01"
    speaker_source: str             # "channel" | "ml" | "manual"

@dataclass(frozen=True)
class TranslatedSegment:
    """A diarized segment with optional translation."""
    diarized: DiarizedSegment
    translated_text: str | None     # None if no translation requested
    target_language: str | None     # ISO 639-1 code

@dataclass(frozen=True)
class TranscriptionResult:
    """Complete result of processing an audio source."""
    segments: list[TranslatedSegment]
    source_info: dict               # metadata about the audio source
    processing_info: dict           # timing, model info, etc.
    created_at: str                 # ISO 8601 timestamp
```

---

## 3. Module Decomposition and Interfaces

Each pipeline stage is defined by a Python Protocol (PEP 544 structural subtyping). This allows implementations to be swapped without changing calling code.

### 3.1 Audio Capture Interface

```python
from typing import Protocol, AsyncIterator, runtime_checkable

@runtime_checkable
class AudioCaptureSource(Protocol):
    """Interface for a single audio capture source (mic or system loopback)."""

    @property
    def device_name(self) -> str: ...

    @property
    def sample_rate(self) -> int: ...

    @property
    def channels(self) -> int: ...

    @property
    def is_active(self) -> bool: ...

    async def start(self) -> None:
        """Begin capturing audio. Raises AudioCaptureError on failure."""
        ...

    async def stop(self) -> None:
        """Stop capturing audio and release resources."""
        ...

    async def read_chunk(self, duration_ms: int = 500) -> AudioChunk:
        """Read the next chunk of audio data.

        Args:
            duration_ms: Target chunk duration in milliseconds.

        Returns:
            AudioChunk with captured audio samples.

        Raises:
            AudioCaptureError: If capture fails or device is disconnected.
            AudioCaptureTimeout: If no audio is available within timeout.
        """
        ...

    def stream(self, chunk_duration_ms: int = 500) -> AsyncIterator[AudioChunk]:
        """Yield audio chunks as an async iterator."""
        ...


class AudioDeviceEnumerator(Protocol):
    """Enumerates available audio capture devices on the current platform."""

    def list_input_devices(self) -> list[AudioDeviceInfo]: ...

    def list_loopback_devices(self) -> list[AudioDeviceInfo]: ...

    def get_default_input_device(self) -> AudioDeviceInfo | None: ...

    def get_default_loopback_device(self) -> AudioDeviceInfo | None: ...


@dataclass(frozen=True)
class AudioDeviceInfo:
    """Metadata about an audio device."""
    id: str
    name: str
    sample_rate: int
    channels: int
    device_type: str            # "input" | "loopback" | "virtual"
    is_default: bool
    platform_id: str            # OS-specific device identifier
```

### 3.2 Pre-Processing Interface

```python
class AudioPreProcessor(Protocol):
    """Transforms raw audio into a format suitable for ASR."""

    def process(self, chunk: AudioChunk) -> AudioChunk:
        """Apply pre-processing to an audio chunk.

        Typical operations: resample to 16kHz mono, normalize amplitude,
        apply noise reduction, trim silence.

        Returns:
            New AudioChunk with processed samples.
        """
        ...

    def reset(self) -> None:
        """Reset any internal state (e.g., running normalization stats)."""
        ...


class VADFilter(Protocol):
    """Voice Activity Detection filter."""

    def contains_speech(self, chunk: AudioChunk) -> bool:
        """Return True if the chunk likely contains speech."""
        ...

    def get_speech_segments(self, chunk: AudioChunk) -> list[tuple[float, float]]:
        """Return list of (start, end) time pairs within the chunk that contain speech."""
        ...
```

### 3.3 ASR Engine Interface

```python
class ASREngine(Protocol):
    """Speech-to-text engine interface."""

    @property
    def model_name(self) -> str: ...

    @property
    def supported_languages(self) -> list[str]: ...

    async def transcribe(
        self,
        audio: AudioChunk,
        *,
        language: str | None = None,
        initial_prompt: str | None = None,
        word_timestamps: bool = False,
    ) -> list[TranscriptionSegment]:
        """Transcribe an audio chunk to text segments.

        Args:
            audio: Pre-processed audio chunk (16kHz mono float32).
            language: ISO 639-1 language code, or None for auto-detection.
            initial_prompt: Optional prompt to guide transcription.
            word_timestamps: Whether to compute word-level timestamps.

        Returns:
            List of transcription segments with timing information.
        """
        ...

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[AudioChunk],
        *,
        language: str | None = None,
    ) -> AsyncIterator[TranscriptionSegment]:
        """Streaming transcription: yields segments as audio arrives."""
        ...

    def load_model(self) -> None:
        """Pre-load the ASR model into memory (CPU or GPU)."""
        ...

    def unload_model(self) -> None:
        """Release the ASR model from memory."""
        ...
```

### 3.4 Diarization Interface

```python
class DiarizationEngine(Protocol):
    """Speaker diarization engine interface."""

    async def diarize(
        self,
        segments: list[TranscriptionSegment],
        audio: AudioChunk | None = None,
    ) -> list[DiarizedSegment]:
        """Assign speaker identities to transcription segments.

        Args:
            segments: Transcribed segments with timing info.
            audio: Original audio (needed for ML-based diarization).

        Returns:
            Segments annotated with speaker IDs.
        """
        ...

    async def diarize_stream(
        self,
        segment_stream: AsyncIterator[tuple[TranscriptionSegment, AudioChunk]],
    ) -> AsyncIterator[DiarizedSegment]:
        """Streaming diarization: yields diarized segments as input arrives."""
        ...
```

### 3.5 Translation Interface

```python
class TranslationEngine(Protocol):
    """Text translation engine interface."""

    @property
    def supported_language_pairs(self) -> list[tuple[str, str]]:
        """List of (source_lang, target_lang) pairs supported."""
        ...

    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        """Translate text from source to target language.

        Raises:
            TranslationError: If translation fails.
            UnsupportedLanguagePair: If the language pair is not supported.
        """
        ...

    async def translate_batch(
        self,
        texts: list[str],
        source_language: str,
        target_language: str,
    ) -> list[str]:
        """Translate multiple texts efficiently."""
        ...
```

### 3.6 Output Formatter Interface

```python
class OutputFormatter(Protocol):
    """Formats transcription results into a specific output format."""

    @property
    def format_name(self) -> str:
        """e.g., 'json', 'srt', 'vtt', 'txt'"""
        ...

    @property
    def file_extension(self) -> str:
        """e.g., '.json', '.srt', '.vtt', '.txt'"""
        ...

    def format(self, result: TranscriptionResult) -> str:
        """Format the complete result as a string."""
        ...

    def format_segment(self, segment: TranslatedSegment, index: int) -> str:
        """Format a single segment (for streaming output)."""
        ...

    def write(self, result: TranscriptionResult, path: Path) -> None:
        """Write formatted result to a file."""
        ...
```

### 3.7 Pipeline Orchestrator

```python
class PipelineOrchestrator:
    """Coordinates the full processing pipeline.

    This is the central controller that wires together capture, preprocessing,
    ASR, diarization, translation, and output formatting. It manages the
    lifecycle of a processing session and handles backpressure between stages.
    """

    def __init__(
        self,
        capture_sources: list[AudioCaptureSource],
        preprocessor: AudioPreProcessor,
        asr_engine: ASREngine,
        diarizer: DiarizationEngine,
        translator: TranslationEngine | None,
        formatters: list[OutputFormatter],
        config: PipelineConfig,
    ) -> None: ...

    async def run_streaming(self) -> AsyncIterator[TranslatedSegment]:
        """Run the pipeline in streaming mode. Yields segments as produced."""
        ...

    async def run_batch(self, audio_path: Path) -> TranscriptionResult:
        """Run the pipeline on a file. Returns the complete result."""
        ...

    async def stop(self) -> None:
        """Gracefully stop a running pipeline."""
        ...
```

---

## 4. Data Flow

### 4.1 Streaming Pipeline

In streaming mode, audio is captured continuously and processed in overlapping chunks. Each stage operates concurrently using `asyncio` tasks connected by bounded `asyncio.Queue` instances for backpressure control.

```
                  asyncio.Queue         asyncio.Queue         asyncio.Queue
AudioCapture -----> [bounded] -----> PreProcessor -----> [bounded] -----> ASREngine
    (task)            buffer             (task)            buffer            (task)
                                                                              |
                                                                              v
                                                                        asyncio.Queue
Output  <------- [bounded] <------- Translation <------- [bounded] <------- Diarizer
(task)             buffer              (task)              buffer            (task)
```

**Backpressure**: If any downstream stage cannot keep up (e.g., ASR is slower than capture), the bounded queues apply backpressure. The capture stage will either drop old frames (lossy mode) or block until the queue has capacity (lossless mode), controlled by configuration.

**Chunk Overlap**: Adjacent audio chunks overlap by a configurable amount (default: 200ms) to avoid cutting words at chunk boundaries. The ASR engine deduplicates output from overlapping regions.

**Concurrency Model**: All pipeline stages run as `asyncio.Task` instances within a single event loop. CPU-bound work (ASR inference, diarization) is offloaded to a `concurrent.futures.ProcessPoolExecutor` or `ThreadPoolExecutor` via `asyncio.loop.run_in_executor()`.

### 4.2 Batch Pipeline

In batch mode, the entire audio file is loaded, optionally chunked for memory management, and processed sequentially through all pipeline stages. There is no backpressure concern since the data is finite and processing is sequential.

```
File Reader --> full AudioChunk(s) --> PreProcessor --> ASR --> Diarization --> Translation --> Formatter --> File Output
```

For large files, the batch pipeline uses a windowed approach: the file is read in segments (e.g., 30-second windows with 2-second overlap), each segment is processed through ASR, and diarization is applied to the full timeline after all segments are transcribed.

### 4.3 Multi-Source Audio Mixing

When capturing from both system loopback and microphone simultaneously, the pipeline handles them as follows:

1. **Separate Capture Tasks**: Each source has its own `AudioCaptureSource` producing independent `AudioChunk` streams.
2. **Channel Tagging**: Each chunk is tagged with its source ("system" or "microphone").
3. **Mixer Stage**: An `AudioMixer` component aligns chunks by timestamp and produces a single mixed stream for ASR, while preserving channel identity for channel-based diarization.
4. **Dual-Path Processing**: The mixed audio goes to ASR; the per-channel audio goes to the channel-based diarizer. Results are merged.

```python
class AudioMixer(Protocol):
    """Mixes multiple audio capture streams into a single timeline."""

    async def add_source(self, source_id: str, stream: AsyncIterator[AudioChunk]) -> None: ...

    def stream_mixed(self) -> AsyncIterator[AudioChunk]:
        """Yield mixed audio chunks aligned by timestamp."""
        ...

    def stream_per_channel(self) -> AsyncIterator[dict[str, AudioChunk]]:
        """Yield per-source audio chunks aligned by timestamp."""
        ...
```

---

## 5. Platform-Specific Audio Capture Strategy

### 5.1 Architecture

The `voxfusion.capture` package uses a **Strategy pattern** to provide platform-specific implementations behind the common `AudioCaptureSource` protocol. A factory function detects the current OS and returns the appropriate implementation.

```python
def create_capture_source(
    device: AudioDeviceInfo,
    config: CaptureConfig,
) -> AudioCaptureSource:
    """Factory: returns platform-appropriate capture source."""
    platform = sys.platform
    if platform == "win32":
        return WasapiCaptureSource(device, config)
    elif platform == "darwin":
        return CoreAudioCaptureSource(device, config)
    elif platform == "linux":
        return PulseAudioCaptureSource(device, config)
    else:
        raise UnsupportedPlatformError(f"Platform {platform} is not supported")
```

### 5.2 Windows -- WASAPI

**Library**: `pycaw` (Python bindings for Windows Core Audio / WASAPI) and/or `sounddevice` (PortAudio wrapper).

**System Audio (Loopback)**:
- Use WASAPI loopback mode via `sounddevice` with `wasapi_exclusive=False`.
- WASAPI loopback captures the audio mix being rendered to a specific output endpoint.
- No virtual audio driver is required; this is a native Windows capability.
- Requires specifying the output device to capture from (default render endpoint).

**Microphone**:
- Standard `sounddevice.InputStream` with the desired input device.

**Key Considerations**:
- WASAPI shared mode is preferred for compatibility; exclusive mode offers lower latency but locks the device.
- Sample rate must match the endpoint's configured rate (typically 44100 or 48000 Hz); resampling happens in the pre-processing stage.
- The `comtypes` package is needed for COM initialization on Windows.

### 5.3 macOS -- CoreAudio

**Library**: `sounddevice` (PortAudio) for microphone input; CoreAudio via `pyobjc-framework-CoreAudio` for system audio.

**System Audio (Loopback)**:
- macOS does not natively support loopback capture. Two strategies:
  1. **Virtual Audio Device (Preferred)**: Require the user to install a virtual audio device such as **BlackHole** (open-source, GPLv2 compatible) or **Loopback by Rogue Amoeba**. VoxFusion routes through the virtual device to capture system audio.
  2. **Screen Capture API (Fallback)**: On macOS 13+, `ScreenCaptureKit` can capture audio. Access via `pyobjc-framework-ScreenCaptureKit`. Requires user permission.
- VoxFusion will detect the presence of BlackHole or other virtual devices automatically and guide the user through setup if none are found.

**Microphone**:
- Standard `sounddevice.InputStream`.

**Key Considerations**:
- macOS requires explicit user consent for microphone and screen recording access.
- The application must handle `AVCaptureDevice` authorization requests gracefully.
- Audio Unit HAL (Hardware Abstraction Layer) provides device enumeration.

### 5.4 Linux -- PulseAudio / PipeWire

**Library**: `pulsectl` (PulseAudio client) and/or `sounddevice` (PortAudio).

**System Audio (Loopback)**:
- PulseAudio: Use a **monitor source** of the default sink. Every PulseAudio sink has an associated `.monitor` source that captures the mixed output.
  - Enumerate monitor sources via `pulsectl` and select the appropriate one.
- PipeWire: PipeWire is PulseAudio-compatible; monitor sources work identically. Additionally, PipeWire's native API allows more fine-grained capture.
- Fallback: `ffmpeg` with `pulse` input device targeting the monitor source.

**Microphone**:
- Standard `sounddevice.InputStream` or `pulsectl` source.

**Key Considerations**:
- PipeWire is the default audio server on modern Linux (Fedora 34+, Ubuntu 22.10+). It exposes a PulseAudio-compatible interface, so `pulsectl` works on both.
- Must handle the case where neither PulseAudio nor PipeWire is available (e.g., bare ALSA). In that case, fall back to `sounddevice` with ALSA backend.
- No special permissions beyond standard audio group membership.

### 5.5 Platform Abstraction Summary

| Feature | Windows (WASAPI) | macOS (CoreAudio) | Linux (PulseAudio/PipeWire) |
|---|---|---|---|
| Loopback capture | Native (WASAPI loopback) | Virtual device (BlackHole) or ScreenCaptureKit | Monitor source (native) |
| Microphone capture | Native (sounddevice) | Native (sounddevice) | Native (sounddevice) |
| Device enumeration | sounddevice + pycaw | sounddevice + pyobjc | sounddevice + pulsectl |
| Extra setup required | None | BlackHole installation (for loopback) | None (usually) |
| Permission model | No special permissions | Microphone + Screen Recording consent | Audio group membership |

---

## 6. ASR Integration Strategy

### 6.1 Primary Engine: faster-whisper

**Why faster-whisper**:
- CTranslate2-backed implementation of OpenAI Whisper -- 4x faster inference than the original with lower memory usage.
- Supports `large-v3` model for highest accuracy.
- Supports CPU (int8 quantization) and GPU (float16/int8) inference.
- Provides word-level timestamps.
- Apache 2.0 license (compatible with GPLv2).

**Integration Approach**:

```python
class FasterWhisperASR:
    """ASR engine backed by faster-whisper."""

    def __init__(self, config: ASRConfig) -> None:
        self._model: WhisperModel | None = None
        self._config = config
        # Config includes: model_size, device, compute_type, beam_size, etc.

    def load_model(self) -> None:
        from faster_whisper import WhisperModel
        self._model = WhisperModel(
            self._config.model_size,       # "large-v3", "medium", "small", "base", "tiny"
            device=self._config.device,     # "cpu", "cuda", "auto"
            compute_type=self._config.compute_type,  # "int8", "float16", "int8_float16"
            cpu_threads=self._config.cpu_threads,
            num_workers=self._config.num_workers,
        )

    async def transcribe(self, audio: AudioChunk, **kwargs) -> list[TranscriptionSegment]:
        # Offload to executor since faster-whisper is synchronous
        loop = asyncio.get_running_loop()
        segments, info = await loop.run_in_executor(
            self._executor,
            functools.partial(self._model.transcribe, audio.samples, **kwargs),
        )
        return self._convert_segments(segments, info)
```

### 6.2 Streaming ASR Strategy

Whisper is fundamentally a batch model (processes up to 30-second windows). For real-time streaming, we use the following approach:

1. **Chunked Inference**: Buffer incoming audio into chunks of configurable length (default: 5 seconds). Process each chunk through Whisper.
2. **Overlap and Merge**: Adjacent chunks overlap by 1-2 seconds. Use timestamp comparison and text deduplication to merge results from overlapping regions.
3. **Partial Results**: After each chunk is processed, emit the segments as partial results. The client can display them with a "provisional" flag.
4. **Finalization**: When the user stops capture or a significant pause is detected, re-process the final buffer window to produce clean final segments.

This approach achieves near-real-time performance (5-10 second latency) on modern hardware with the `large-v3` model on GPU, or with `small`/`medium` models on CPU.

### 6.3 Model Configuration

```yaml
asr:
  engine: "faster-whisper"
  model_size: "large-v3"         # large-v3 | medium | small | base | tiny
  device: "auto"                 # auto | cpu | cuda
  compute_type: "int8_float16"   # int8 | float16 | int8_float16 | float32
  beam_size: 5
  best_of: 5
  patience: 1.0
  language: null                 # null = auto-detect, or ISO 639-1 code
  initial_prompt: null
  word_timestamps: true
  vad_filter: true               # use Silero VAD to filter non-speech
  vad_parameters:
    threshold: 0.5
    min_speech_duration_ms: 250
    min_silence_duration_ms: 2000
  chunk_duration_s: 5            # streaming chunk duration
  chunk_overlap_s: 1             # overlap between chunks
```

### 6.4 Future ASR Backends

The ASR interface is designed to accommodate alternative backends:

- **whisper.cpp**: Via `pywhispercpp` bindings. Useful for CPU-only environments.
- **Cloud ASR APIs**: Google Cloud Speech-to-Text, Azure Speech Services, AWS Transcribe. Implement the same `ASREngine` protocol, sending audio to the cloud.
- **Distil-Whisper**: Smaller distilled models for lower-resource environments.

---

## 7. Speaker Diarization Approach

### 7.1 Dual Strategy

VoxFusion uses two complementary diarization strategies:

1. **Channel-Based Diarization (Simple, Fast)**: When audio is captured from distinct sources (e.g., microphone = local user, system loopback = remote participants), assign speaker identity based on the audio channel/source. This is deterministic and zero-latency.

2. **ML-Based Diarization (Advanced, Accurate)**: Use `pyannote.audio` to perform embedding-based diarization on the mixed audio. This identifies individual speakers within a single audio stream (e.g., multiple remote participants on a call).

The two strategies can be combined: channel-based diarization provides the "local vs. remote" split, and ML-based diarization further segments the remote audio into individual speakers.

### 7.2 Channel-Based Diarizer

```python
class ChannelBasedDiarizer:
    """Assigns speaker IDs based on audio source/channel."""

    def __init__(self, channel_map: dict[str, str]) -> None:
        """
        Args:
            channel_map: Maps source ID to speaker label.
                         e.g. {"microphone": "SPEAKER_LOCAL", "system": "SPEAKER_REMOTE"}
        """
        self._channel_map = channel_map

    async def diarize(
        self,
        segments: list[TranscriptionSegment],
        audio: AudioChunk | None = None,
    ) -> list[DiarizedSegment]:
        # Each segment inherits the speaker label from its source channel
        ...
```

### 7.3 ML-Based Diarizer (pyannote.audio)

**Library**: `pyannote.audio` >= 3.1

**Approach**:
- Use `pyannote.audio`'s pre-trained speaker diarization pipeline (`pyannote/speaker-diarization-3.1`).
- For batch processing: run the full diarization pipeline on the complete audio file.
- For streaming: use a windowed approach where diarization is applied to rolling windows of audio and speaker embeddings are tracked across windows.

```python
class PyAnnoteDiarizer:
    """ML-based speaker diarization using pyannote.audio."""

    def __init__(self, config: DiarizationConfig) -> None:
        self._config = config
        self._pipeline = None

    def load_model(self) -> None:
        from pyannote.audio import Pipeline
        self._pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self._config.hf_auth_token,
        )
        if self._config.device == "cuda":
            import torch
            self._pipeline.to(torch.device("cuda"))

    async def diarize(
        self,
        segments: list[TranscriptionSegment],
        audio: AudioChunk | None = None,
    ) -> list[DiarizedSegment]:
        # 1. Run pyannote pipeline on the audio to get speaker segments
        # 2. Align pyannote's speaker turns with ASR segments by timestamp overlap
        # 3. Assign speaker IDs to each TranscriptionSegment
        ...
```

### 7.4 Diarization Configuration

```yaml
diarization:
  strategy: "hybrid"             # channel | ml | hybrid
  channel_map:
    microphone: "SPEAKER_LOCAL"
    system: "SPEAKER_REMOTE"
  ml:
    engine: "pyannote"
    model: "pyannote/speaker-diarization-3.1"
    hf_auth_token: null          # set via env var VOXFUSION_HF_TOKEN
    device: "auto"
    min_speakers: null            # null = auto-detect
    max_speakers: null
    min_segment_duration: 0.5    # seconds
  embedding_model: "pyannote/wespeaker-voxceleb-resnet34-LM"
```

### 7.5 Alignment Algorithm

When combining ASR segments with diarization output, the alignment works as follows:

1. ASR produces segments with `(start_time, end_time, text)`.
2. Diarization produces speaker turns with `(start_time, end_time, speaker_id)`.
3. For each ASR segment, find the diarization turn with the maximum temporal overlap.
4. If an ASR segment spans multiple speaker turns, split it at the speaker turn boundary (using word-level timestamps if available).
5. Assign the speaker ID of the dominant turn to each resulting segment.

---

## 8. Translation Subsystem Design

### 8.1 Pluggable Architecture

The translation subsystem uses a **plugin/registry** pattern. Multiple translation backends can be registered, and the active backend is selected by configuration.

```python
class TranslationRegistry:
    """Registry of available translation backends."""

    _backends: dict[str, type[TranslationEngine]] = {}

    @classmethod
    def register(cls, name: str, backend_class: type[TranslationEngine]) -> None: ...

    @classmethod
    def create(cls, name: str, config: TranslationConfig) -> TranslationEngine: ...

    @classmethod
    def available_backends(cls) -> list[str]: ...
```

### 8.2 Local NMT Backend (Offline)

**Library**: `argos-translate` (open-source, MIT license, compatible with GPLv2) or `ctranslate2` with Opus-MT / NLLB models.

```python
class ArgosTranslationEngine:
    """Offline translation using Argos Translate."""

    def __init__(self, config: TranslationConfig) -> None:
        self._config = config
        self._installed_packages: list = []

    def install_language_pair(self, source: str, target: str) -> None:
        """Download and install a translation model for a language pair."""
        ...

    async def translate(self, text: str, source_language: str, target_language: str) -> str:
        import argostranslate.translate
        # Offload to executor
        ...
```

**Alternative -- CTranslate2 + NLLB-200**:

```python
class NLLBTranslationEngine:
    """Offline translation using Meta's NLLB-200 model via CTranslate2."""

    def __init__(self, config: TranslationConfig) -> None:
        self._model = None
        self._tokenizer = None

    def load_model(self) -> None:
        import ctranslate2
        import sentencepiece
        self._model = ctranslate2.Translator(
            self._config.model_path,
            device=self._config.device,
        )
        self._tokenizer = sentencepiece.SentencePieceProcessor()
        self._tokenizer.load(self._config.tokenizer_path)
```

### 8.3 API-Based Backend (Online)

For higher quality or broader language coverage, VoxFusion supports external translation APIs.

```python
class DeepLTranslationEngine:
    """Translation via DeepL API."""

    def __init__(self, config: TranslationConfig) -> None:
        self._api_key = config.api_key  # from env var or config
        self._base_url = "https://api-free.deepl.com/v2"

    async def translate(self, text: str, source_language: str, target_language: str) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{self._base_url}/translate", ...)
            ...


class LibreTranslateEngine:
    """Translation via LibreTranslate (self-hosted or public instance)."""
    ...
```

### 8.4 Translation Configuration

```yaml
translation:
  enabled: true
  target_language: "en"          # ISO 639-1 code
  backend: "argos"               # argos | nllb | deepl | libretranslate | none
  argos:
    model_dir: "~/.voxfusion/models/argos"
    auto_install: true           # automatically download missing language packs
  nllb:
    model_path: "~/.voxfusion/models/nllb-200-distilled-600M"
    device: "auto"
  deepl:
    api_key: null                # set via env var VOXFUSION_DEEPL_API_KEY
    formality: "default"         # default | more | less
  libretranslate:
    url: "http://localhost:5000"
    api_key: null
  cache:
    enabled: true
    max_size: 10000              # max cached translations
    ttl: 3600                    # cache TTL in seconds
```

### 8.5 Translation Caching

To avoid redundant translations (especially in streaming mode where partial results may be re-translated), the translation subsystem includes an LRU cache keyed by `(text, source_lang, target_lang)`.

---

## 9. Output Formatting Subsystem

### 9.1 Supported Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| JSON | `.json` | Machine-readable, complete metadata |
| SRT | `.srt` | Video subtitle format (SubRip) |
| VTT | `.vtt` | Web subtitle format (WebVTT) |
| TXT | `.txt` | Plain text transcript |

### 9.2 JSON Format

```json
{
  "voxfusion_version": "0.1.0",
  "created_at": "2026-02-11T14:30:00Z",
  "source": {
    "type": "live_capture",
    "devices": ["Microphone (Realtek)", "Speakers (Loopback)"],
    "duration_seconds": 3600.5
  },
  "processing": {
    "asr_model": "faster-whisper large-v3",
    "diarization_model": "pyannote/speaker-diarization-3.1",
    "translation_backend": "argos",
    "target_language": "en"
  },
  "segments": [
    {
      "index": 0,
      "start_time": 0.0,
      "end_time": 3.52,
      "speaker_id": "SPEAKER_00",
      "speaker_source": "ml",
      "original_text": "Bonjour, comment allez-vous?",
      "original_language": "fr",
      "translated_text": "Hello, how are you?",
      "target_language": "en",
      "confidence": 0.94,
      "words": [
        {"word": "Bonjour", "start": 0.0, "end": 0.8, "probability": 0.97},
        {"word": ",", "start": 0.8, "end": 0.9, "probability": 0.99},
        {"word": "comment", "start": 1.0, "end": 1.5, "probability": 0.92},
        {"word": "allez-vous", "start": 1.6, "end": 2.4, "probability": 0.91},
        {"word": "?", "start": 2.4, "end": 2.5, "probability": 0.99}
      ]
    }
  ]
}
```

### 9.3 SRT Format

```
1
00:00:00,000 --> 00:00:03,520
[SPEAKER_00] Bonjour, comment allez-vous?
(Hello, how are you?)

2
00:00:04,100 --> 00:00:07,800
[SPEAKER_01] Je vais bien, merci.
(I'm fine, thank you.)
```

### 9.4 VTT Format

```
WEBVTT

00:00:00.000 --> 00:00:03.520
<v SPEAKER_00>Bonjour, comment allez-vous?
(Hello, how are you?)

00:00:04.100 --> 00:00:07.800
<v SPEAKER_01>Je vais bien, merci.
(I'm fine, thank you.)
```

### 9.5 Streaming Output

In streaming mode, partial results are delivered via:
- **stdout** (CLI): Segments printed as they are produced, one per line.
- **WebSocket** (API): JSON-serialized segments pushed to connected clients.
- **Callback** (Library): A user-provided callback function invoked per segment.

---

## 10. CLI and API Design

### 10.1 CLI Design

The CLI is built with `click` (BSD license, GPLv2-compatible) and provides a hierarchical command structure.

```
voxfusion
  |-- capture           # Live audio capture and transcription
  |-- transcribe        # Transcribe an audio/video file (batch)
  |-- devices           # List available audio devices
  |-- config            # Manage configuration
  |   |-- show          # Display current config
  |   |-- set           # Set a config value
  |   |-- reset         # Reset to defaults
  |-- models            # Manage ASR/diarization/translation models
  |   |-- list          # List installed models
  |   |-- download      # Download a model
  |   |-- remove        # Remove a model
  |-- version           # Show version info
```

**Key CLI Commands**:

```bash
# Live capture with defaults (system + mic, auto language, no translation)
voxfusion capture

# Live capture with specific devices and translation
voxfusion capture \
  --mic "Microphone (Realtek)" \
  --loopback "Speakers (Realtek)" \
  --translate-to en \
  --output-format json \
  --output-file meeting.json

# Transcribe a file
voxfusion transcribe recording.wav \
  --language fr \
  --translate-to en \
  --diarize \
  --output-format srt \
  --output-file recording.srt

# List devices
voxfusion devices --json

# Download models
voxfusion models download --asr large-v3
voxfusion models download --diarization pyannote
voxfusion models download --translation argos fr-en
```

**CLI Output Modes**:
- `--quiet`: Suppress all output except errors.
- `--verbose`: Show debug-level logging.
- `--json`: Output structured JSON (for scripting).
- Default: Human-readable progress and results.

### 10.2 Python API (Library Usage)

VoxFusion can be used as a Python library for integration into other applications.

```python
import asyncio
from voxfusion import VoxFusion, CaptureConfig, ASRConfig

async def main():
    vf = VoxFusion(
        capture=CaptureConfig(sources=["microphone", "loopback"]),
        asr=ASRConfig(model_size="large-v3"),
        translate_to="en",
    )

    # Streaming mode
    async for segment in vf.stream():
        print(f"[{segment.diarized.speaker_id}] {segment.diarized.segment.text}")
        if segment.translated_text:
            print(f"  -> {segment.translated_text}")

asyncio.run(main())
```

```python
# Batch mode
from voxfusion import VoxFusion

async def main():
    vf = VoxFusion()
    result = await vf.transcribe_file(
        "recording.wav",
        diarize=True,
        translate_to="en",
        output_format="srt",
    )
    result.save("recording.srt")

asyncio.run(main())
```

### 10.3 WebSocket API (Future)

For real-time integrations, a WebSocket server can be started:

```bash
voxfusion serve --host 0.0.0.0 --port 8765
```

Protocol:
- Client connects to `ws://host:port/stream`.
- Server pushes JSON-serialized `TranslatedSegment` objects as they are produced.
- Client can send control messages: `{"action": "start"}`, `{"action": "stop"}`, `{"action": "configure", "config": {...}}`.

---

## 11. Configuration Management

### 11.1 Configuration Hierarchy

Configuration is resolved in the following order (later overrides earlier):

1. **Built-in defaults** (`voxfusion/config/defaults.yaml`)
2. **System-wide config** (`/etc/voxfusion/config.yaml` or `%PROGRAMDATA%\voxfusion\config.yaml`)
3. **User config** (`~/.voxfusion/config.yaml`)
4. **Project config** (`.voxfusion.yaml` in CWD)
5. **Environment variables** (`VOXFUSION_*` prefix)
6. **CLI flags** (highest priority)

### 11.2 Configuration Schema

Configuration is defined using Pydantic models for validation and serialization.

```python
from pydantic import BaseModel, Field

class CaptureConfig(BaseModel):
    sources: list[str] = Field(default=["microphone", "loopback"])
    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    channels: int = Field(default=1, ge=1, le=2)
    chunk_duration_ms: int = Field(default=500, ge=100, le=5000)
    buffer_size: int = Field(default=10)  # max chunks in queue
    lossy_mode: bool = Field(default=False)

class ASRConfig(BaseModel):
    engine: str = Field(default="faster-whisper")
    model_size: str = Field(default="large-v3")
    device: str = Field(default="auto")
    compute_type: str = Field(default="int8_float16")
    beam_size: int = Field(default=5)
    language: str | None = Field(default=None)
    word_timestamps: bool = Field(default=True)
    vad_filter: bool = Field(default=True)

class DiarizationConfig(BaseModel):
    strategy: str = Field(default="hybrid")
    ml_engine: str = Field(default="pyannote")
    ml_model: str = Field(default="pyannote/speaker-diarization-3.1")
    hf_auth_token: str | None = Field(default=None)
    min_speakers: int | None = Field(default=None)
    max_speakers: int | None = Field(default=None)

class TranslationConfig(BaseModel):
    enabled: bool = Field(default=False)
    target_language: str = Field(default="en")
    backend: str = Field(default="argos")
    cache_enabled: bool = Field(default=True)

class OutputConfig(BaseModel):
    format: str = Field(default="json")
    include_word_timestamps: bool = Field(default=False)
    include_translation: bool = Field(default=True)
    include_confidence: bool = Field(default=True)

class PipelineConfig(BaseModel):
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    asr: ASRConfig = Field(default_factory=ASRConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    log_level: str = Field(default="INFO")
    data_dir: str = Field(default="~/.voxfusion")
```

### 11.3 Environment Variables

All configuration fields can be overridden via environment variables using the `VOXFUSION_` prefix with double-underscore nesting:

```bash
VOXFUSION_ASR__MODEL_SIZE=medium
VOXFUSION_ASR__DEVICE=cpu
VOXFUSION_TRANSLATION__BACKEND=deepl
VOXFUSION_TRANSLATION__DEEPL__API_KEY=sk-...
VOXFUSION_DIARIZATION__HF_AUTH_TOKEN=hf_...
```

---

## 12. Error Handling Strategy

### 12.1 Exception Hierarchy

```python
class VoxFusionError(Exception):
    """Base exception for all VoxFusion errors."""
    pass

# -- Audio Capture Errors --
class AudioCaptureError(VoxFusionError):
    """Base for audio capture failures."""
    pass

class DeviceNotFoundError(AudioCaptureError):
    """Requested audio device does not exist."""
    pass

class DeviceAccessDeniedError(AudioCaptureError):
    """OS denied access to the audio device (permissions)."""
    pass

class DeviceDisconnectedError(AudioCaptureError):
    """Audio device was disconnected during capture."""
    pass

class AudioCaptureTimeout(AudioCaptureError):
    """No audio data received within the timeout period."""
    pass

class UnsupportedPlatformError(AudioCaptureError):
    """Current OS platform is not supported."""
    pass

# -- ASR Errors --
class ASRError(VoxFusionError):
    """Base for ASR failures."""
    pass

class ModelNotFoundError(ASRError):
    """Requested ASR model is not downloaded."""
    pass

class ModelLoadError(ASRError):
    """Failed to load ASR model (OOM, corrupted, etc.)."""
    pass

class TranscriptionError(ASRError):
    """Transcription failed for a given audio chunk."""
    pass

# -- Diarization Errors --
class DiarizationError(VoxFusionError):
    """Base for diarization failures."""
    pass

# -- Translation Errors --
class TranslationError(VoxFusionError):
    """Base for translation failures."""
    pass

class UnsupportedLanguagePair(TranslationError):
    """The requested language pair is not supported by the backend."""
    pass

class TranslationAPIError(TranslationError):
    """External translation API returned an error."""
    pass

# -- Configuration Errors --
class ConfigurationError(VoxFusionError):
    """Invalid or missing configuration."""
    pass

# -- Pipeline Errors --
class PipelineError(VoxFusionError):
    """Pipeline orchestration failure."""
    pass
```

### 12.2 Error Handling Policies

| Component | Failure Mode | Policy |
|-----------|-------------|--------|
| Audio Capture | Device disconnected | Retry 3 times with backoff, then stop pipeline with error |
| Audio Capture | Permission denied | Immediate error with clear user message |
| Pre-Processing | Invalid audio data | Log warning, skip chunk, continue |
| ASR | Model OOM | Attempt with smaller model, fail if all exhausted |
| ASR | Transcription failure | Log error, skip chunk, continue (streaming) or retry (batch) |
| Diarization | ML model unavailable | Fall back to channel-based diarization |
| Diarization | Processing failure | Assign "UNKNOWN" speaker, continue |
| Translation | API timeout/error | Retry with exponential backoff (3 attempts), then skip translation |
| Translation | Unsupported language | Return original text with warning |
| Output | Write failure | Retry once, then raise error to user |

### 12.3 Logging Strategy

- **Library**: Python standard `logging` module with `structlog` for structured JSON logging.
- **Log Levels**:
  - `DEBUG`: Per-chunk processing details, model inference timing.
  - `INFO`: Pipeline lifecycle events, segment results.
  - `WARNING`: Degraded operation (fallbacks, retries).
  - `ERROR`: Component failures that do not stop the pipeline.
  - `CRITICAL`: Pipeline-halting failures.
- **Log Destinations**: stderr (CLI), file (`~/.voxfusion/logs/`), and optionally syslog.
- **Sensitive Data**: Audio content and transcription text are NEVER logged at INFO level or below. DEBUG-level text logging is opt-in.

---

## 13. Security and Privacy

### 13.1 OS Permission Handling

- **macOS**: Request microphone and screen recording permissions via `AVCaptureDevice.requestAccessForMediaType`. Display clear instructions if denied.
- **Windows**: No special permissions for microphone or WASAPI loopback in user context.
- **Linux**: Check audio group membership; provide instructions for adding user to `audio` group if needed.

### 13.2 Data at Rest

- **Transcripts**: When saving output files, optionally encrypt using `cryptography.fernet` (symmetric encryption). The encryption key is derived from a user-provided passphrase via PBKDF2.
- **Model Cache**: Downloaded models are stored in `~/.voxfusion/models/` with file permission `0o700` (user-only access on Unix).
- **Logs**: Log files have restricted permissions. No transcription content in logs by default.

### 13.3 Data in Transit

- **API Translation Backends**: All HTTPS with TLS 1.2+.
- **WebSocket Server**: Supports WSS (TLS) when configured with certificate paths.
- **Local Processing**: All audio and text data stays on the local machine when using offline backends.

### 13.4 Privacy Controls

```yaml
privacy:
  encrypt_output: false
  encryption_passphrase: null    # set via VOXFUSION_ENCRYPTION_PASSPHRASE
  log_transcription_content: false
  telemetry: false               # no telemetry by default
  auto_delete_temp_files: true
  temp_file_ttl_hours: 24
```

---

## 14. Project Structure

```
VoxFusion/
|-- pyproject.toml                    # Project metadata, dependencies, build config
|-- poetry.lock                       # Locked dependency versions
|-- README.md                         # Project README
|-- LICENSE                           # GPLv2
|-- ARCHITECTURE.md                   # This document
|-- CLAUDE.md                         # Claude Code guidance
|-- CHANGELOG.md                      # Release changelog
|
|-- src/
|   |-- voxfusion/
|   |   |-- __init__.py               # Package root, public API exports
|   |   |-- __main__.py               # Entry point: python -m voxfusion
|   |   |-- version.py                # Single source of version truth
|   |   |
|   |   |-- capture/                  # Audio capture subsystem
|   |   |   |-- __init__.py
|   |   |   |-- base.py               # AudioCaptureSource protocol, AudioDeviceInfo
|   |   |   |-- factory.py            # Platform-detection factory
|   |   |   |-- wasapi.py             # Windows WASAPI implementation
|   |   |   |-- coreaudio.py          # macOS CoreAudio implementation
|   |   |   |-- pulseaudio.py         # Linux PulseAudio/PipeWire implementation
|   |   |   |-- file_source.py        # AudioCaptureSource from file (batch mode)
|   |   |   |-- mixer.py              # Multi-source audio mixer
|   |   |   |-- enumerator.py         # Device enumeration per platform
|   |   |
|   |   |-- preprocessing/            # Audio pre-processing
|   |   |   |-- __init__.py
|   |   |   |-- base.py               # AudioPreProcessor protocol
|   |   |   |-- pipeline.py           # Composable pre-processing pipeline
|   |   |   |-- resample.py           # Sample rate conversion (via librosa or soxr)
|   |   |   |-- normalize.py          # Amplitude normalization
|   |   |   |-- vad.py                # Voice Activity Detection (Silero VAD)
|   |   |   |-- noise.py              # Noise reduction (optional, via noisereduce)
|   |   |
|   |   |-- asr/                      # Automatic Speech Recognition
|   |   |   |-- __init__.py
|   |   |   |-- base.py               # ASREngine protocol
|   |   |   |-- faster_whisper.py     # faster-whisper implementation
|   |   |   |-- streaming.py          # Streaming ASR wrapper (chunked inference)
|   |   |   |-- dedup.py              # Overlap deduplication logic
|   |   |
|   |   |-- diarization/              # Speaker diarization
|   |   |   |-- __init__.py
|   |   |   |-- base.py               # DiarizationEngine protocol
|   |   |   |-- channel.py            # Channel-based diarizer
|   |   |   |-- pyannote_engine.py    # pyannote.audio ML-based diarizer
|   |   |   |-- hybrid.py             # Combined channel + ML diarizer
|   |   |   |-- alignment.py          # ASR-diarization segment alignment
|   |   |
|   |   |-- translation/              # Text translation
|   |   |   |-- __init__.py
|   |   |   |-- base.py               # TranslationEngine protocol
|   |   |   |-- registry.py           # Backend registry
|   |   |   |-- argos_engine.py       # Argos Translate backend
|   |   |   |-- nllb_engine.py        # NLLB-200 via CTranslate2
|   |   |   |-- deepl_engine.py       # DeepL API backend
|   |   |   |-- libretranslate.py     # LibreTranslate API backend
|   |   |   |-- cache.py              # Translation cache (LRU)
|   |   |
|   |   |-- output/                   # Output formatting
|   |   |   |-- __init__.py
|   |   |   |-- base.py               # OutputFormatter protocol
|   |   |   |-- json_formatter.py     # JSON output
|   |   |   |-- srt_formatter.py      # SRT subtitle output
|   |   |   |-- vtt_formatter.py      # WebVTT subtitle output
|   |   |   |-- txt_formatter.py      # Plain text output
|   |   |
|   |   |-- pipeline/                 # Pipeline orchestration
|   |   |   |-- __init__.py
|   |   |   |-- orchestrator.py       # PipelineOrchestrator
|   |   |   |-- streaming.py          # Streaming pipeline implementation
|   |   |   |-- batch.py              # Batch pipeline implementation
|   |   |   |-- events.py             # Pipeline event types and bus
|   |   |
|   |   |-- config/                   # Configuration management
|   |   |   |-- __init__.py
|   |   |   |-- models.py             # Pydantic config models
|   |   |   |-- loader.py             # Config file loading and merging
|   |   |   |-- defaults.yaml         # Built-in default configuration
|   |   |
|   |   |-- cli/                      # Command-line interface
|   |   |   |-- __init__.py
|   |   |   |-- main.py               # Click CLI entry point
|   |   |   |-- capture_cmd.py        # `voxfusion capture` command
|   |   |   |-- transcribe_cmd.py     # `voxfusion transcribe` command
|   |   |   |-- devices_cmd.py        # `voxfusion devices` command
|   |   |   |-- config_cmd.py         # `voxfusion config` commands
|   |   |   |-- models_cmd.py         # `voxfusion models` commands
|   |   |   |-- formatting.py         # CLI output formatting helpers
|   |   |
|   |   |-- models/                   # Data models (shared types)
|   |   |   |-- __init__.py
|   |   |   |-- audio.py              # AudioChunk, AudioDeviceInfo
|   |   |   |-- transcription.py      # TranscriptionSegment, WordTiming
|   |   |   |-- diarization.py        # DiarizedSegment
|   |   |   |-- translation.py        # TranslatedSegment
|   |   |   |-- result.py             # TranscriptionResult
|   |   |
|   |   |-- security/                 # Security and encryption
|   |   |   |-- __init__.py
|   |   |   |-- encryption.py         # File encryption/decryption
|   |   |   |-- permissions.py        # OS permission checks
|   |   |
|   |   |-- exceptions.py             # Exception hierarchy
|   |   |-- logging.py                # Logging configuration (structlog)
|
|-- tests/
|   |-- __init__.py
|   |-- conftest.py                   # Shared fixtures
|   |-- fixtures/                     # Test audio files, mock data
|   |   |-- audio/
|   |   |   |-- silence_1s.wav
|   |   |   |-- speech_en_5s.wav
|   |   |   |-- speech_fr_5s.wav
|   |   |   |-- multi_speaker_10s.wav
|   |
|   |-- unit/                         # Unit tests (fast, no GPU/models)
|   |   |-- test_preprocessing.py
|   |   |-- test_audio_models.py
|   |   |-- test_config_loader.py
|   |   |-- test_output_formatters.py
|   |   |-- test_translation_cache.py
|   |   |-- test_alignment.py
|   |   |-- test_dedup.py
|   |   |-- test_encryption.py
|   |
|   |-- integration/                  # Integration tests (may need models)
|   |   |-- test_faster_whisper.py
|   |   |-- test_pyannote.py
|   |   |-- test_argos_translate.py
|   |   |-- test_pipeline_batch.py
|   |   |-- test_pipeline_streaming.py
|   |
|   |-- platform/                     # Platform-specific tests
|   |   |-- test_wasapi.py
|   |   |-- test_coreaudio.py
|   |   |-- test_pulseaudio.py
|
|-- docs/                             # Additional documentation
|   |-- setup-macos.md                # macOS setup guide (BlackHole, etc.)
|   |-- setup-linux.md                # Linux audio setup
|   |-- api-reference.md              # Generated API docs
|
|-- scripts/                          # Development and CI scripts
|   |-- download_models.py            # Convenience script to download models
|   |-- benchmark.py                  # Benchmark ASR/diarization performance
```

### 14.1 Package Layout Rationale

- **`src/` layout**: Using the `src/` layout (PEP 517) prevents accidental imports of the package from the project root. This is the recommended layout for installable Python packages.
- **Flat module hierarchy**: Each subsystem is a top-level subpackage under `voxfusion/`. This keeps imports short and predictable (e.g., `from voxfusion.asr import FasterWhisperASR`).
- **Protocol definitions in `base.py`**: Each subpackage has a `base.py` defining the protocol/interface. Implementations import from `base.py`, never the reverse.
- **Models in a dedicated package**: Shared data types live in `voxfusion/models/` to avoid circular imports between subsystems.

---

## 15. MVP Plan vs Full Roadmap

### 15.1 MVP (v0.1.0) -- "It Works"

**Goal**: End-to-end transcription of a single audio source (microphone or file) with basic output.

**Scope**:
- Single-platform support (developer's primary OS first, likely Windows).
- Microphone capture only (no loopback in MVP).
- faster-whisper with `small` or `medium` model (lower resource requirements).
- Batch mode only (file input, complete output).
- No diarization.
- No translation.
- JSON and TXT output.
- Basic CLI: `voxfusion transcribe <file>`.
- Configuration via CLI flags only.

**Estimated Effort**: 2-3 weeks.

**Deliverables**:
- Working `voxfusion transcribe` command.
- Unit tests for pre-processing and output formatting.
- Integration test with a sample audio file.

### 15.2 v0.2.0 -- "Real-Time"

**Goal**: Streaming transcription from live audio.

**Scope**:
- Live microphone capture with streaming pipeline.
- Streaming ASR (chunked inference).
- System audio loopback capture (Windows WASAPI first).
- CLI: `voxfusion capture`.
- Real-time text output to stdout.

**Estimated Effort**: 2-3 weeks.

### 15.3 v0.3.0 -- "Who Said What"

**Goal**: Speaker diarization.

**Scope**:
- Channel-based diarization (mic vs. system).
- ML-based diarization via pyannote.audio.
- Hybrid diarization mode.
- SRT and VTT output formats.

**Estimated Effort**: 2-3 weeks.

### 15.4 v0.4.0 -- "Polyglot"

**Goal**: Translation support.

**Scope**:
- Argos Translate (offline) integration.
- DeepL API integration.
- Translation caching.
- `--translate-to` flag in CLI.

**Estimated Effort**: 1-2 weeks.

### 15.5 v0.5.0 -- "Everywhere"

**Goal**: Cross-platform parity.

**Scope**:
- macOS CoreAudio + BlackHole integration.
- Linux PulseAudio/PipeWire integration.
- Platform-specific tests and CI.
- Setup guides for each platform.

**Estimated Effort**: 2-3 weeks.

### 15.6 v1.0.0 -- "Production Ready"

**Goal**: Stable, documented, tested release.

**Scope**:
- Configuration file support (YAML).
- WebSocket server for real-time integrations.
- Output encryption.
- Comprehensive test suite.
- Documentation and guides.
- Performance benchmarks.
- CI/CD pipeline.

**Estimated Effort**: 3-4 weeks.

### 15.7 Post-1.0 Roadmap

- **GUI**: Desktop application (Tauri or Electron-based, or native with Qt/PySide6).
- **Cloud ASR backends**: Google, Azure, AWS as alternative ASR engines.
- **Speaker identification**: Match speakers to known voice profiles.
- **Custom vocabulary**: Domain-specific vocabulary boosting for ASR.
- **Plugin system**: Allow third-party pipeline stages.
- **NLLB-200 translation**: Advanced offline translation model.
- **Multi-file batch processing**: Process a directory of files.
- **REST API**: In addition to WebSocket, a REST endpoint for batch jobs.

---

## 16. Acceptance Criteria and Metrics

### 16.1 Functional Acceptance Criteria

| ID | Criterion | Verification Method |
|----|-----------|-------------------|
| AC-01 | System captures microphone audio and produces a transcript | Integration test with known audio file |
| AC-02 | System captures system audio via loopback and produces a transcript | Manual test on each platform |
| AC-03 | Transcription is produced in real-time (within 10s latency) on GPU | Benchmark test |
| AC-04 | Speaker diarization correctly separates 2+ speakers | Evaluation against annotated test set |
| AC-05 | Translation produces intelligible output for supported pairs | Manual review + BLEU score |
| AC-06 | Output formats (JSON, SRT, VTT, TXT) are valid and parseable | Unit tests with format validators |
| AC-07 | CLI commands work as documented | End-to-end CLI tests |
| AC-08 | Configuration hierarchy resolves correctly | Unit tests for config loader |
| AC-09 | System works on Windows, macOS, and Linux | CI tests on all platforms |
| AC-10 | Encrypted output can be decrypted with correct passphrase | Unit test for encryption round-trip |

### 16.2 Performance Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **ASR Word Error Rate (WER)** | < 10% on English broadcast news (large-v3) | Evaluate against LibriSpeech test-clean |
| **Diarization Error Rate (DER)** | < 15% on 2-speaker conversations | Evaluate against AMI meeting corpus subset |
| **End-to-End Latency (Streaming)** | < 10 seconds (GPU), < 30 seconds (CPU) | Measure time from audio capture to text output |
| **Throughput (Batch)** | > 10x real-time on GPU (large-v3) | Measure processing time vs. audio duration |
| **Memory Usage** | < 4 GB RAM (large-v3 on CPU, int8) | Monitor peak RSS during batch processing |
| **Startup Time** | < 5 seconds (model pre-loaded) | Measure from command invocation to first output |
| **Translation Latency** | < 500ms per segment (offline), < 1s per segment (API) | Measure per-segment translation time |

### 16.3 Quality Gates

Before any release:

1. All unit tests pass (`pytest tests/unit/`).
2. Integration tests pass on at least one platform (`pytest tests/integration/`).
3. No `CRITICAL` or `ERROR`-level linting issues (`ruff check`).
4. Type checking passes (`mypy src/`).
5. Test coverage >= 80% for core modules (capture, preprocessing, ASR, output).
6. No known security vulnerabilities in dependencies (`pip-audit`).

---

## 17. Architecture Decision Records

### ADR-001: Use `src/` Layout

**Context**: Python packages can use a flat layout (`voxfusion/` at repo root) or `src/` layout (`src/voxfusion/`).

**Decision**: Use `src/` layout.

**Rationale**: Prevents accidental imports from the repo root (which would bypass installation). This is the recommended layout per PyPA and avoids subtle test contamination issues.

**Consequences**: `import voxfusion` only works after `pip install -e .` (editable install). This is the desired behavior.

---

### ADR-002: Async-First with `asyncio`

**Context**: The streaming pipeline needs concurrent I/O (audio capture, network translation APIs) and CPU-bound work (ASR inference).

**Decision**: Use `asyncio` as the concurrency model. Offload CPU-bound work to executors.

**Rationale**: `asyncio` provides efficient I/O multiplexing for streaming and network operations. `ProcessPoolExecutor` handles CPU-bound inference without blocking the event loop. This is more composable and lighter-weight than threading for I/O-heavy workloads.

**Consequences**: All pipeline interfaces use `async` methods. Library users must use `asyncio.run()` or integrate with an existing event loop.

---

### ADR-003: Protocol-Based Interfaces (Structural Subtyping)

**Context**: Pipeline components need well-defined interfaces, but we want flexibility in implementation.

**Decision**: Use `typing.Protocol` (PEP 544) for all component interfaces instead of abstract base classes.

**Rationale**: Protocols enable structural subtyping -- any class that implements the required methods satisfies the protocol, without needing to inherit from a base class. This reduces coupling and makes testing easier (plain classes or mocks satisfy protocols without special setup).

**Consequences**: No inheritance-based coupling between interface definitions and implementations. `runtime_checkable` is used sparingly for explicit runtime validation.

---

### ADR-004: faster-whisper as Primary ASR Engine

**Context**: Multiple Whisper implementations exist: OpenAI's original, faster-whisper (CTranslate2), whisper.cpp, and cloud APIs.

**Decision**: Use `faster-whisper` as the primary and default ASR engine.

**Rationale**: faster-whisper provides the best balance of accuracy (same models as OpenAI Whisper), performance (4x speedup via CTranslate2 optimizations), memory efficiency (int8 quantization), and Python integration (native Python API). Apache 2.0 license is compatible with GPLv2.

**Consequences**: CTranslate2 is a required dependency. GPU support requires CUDA toolkit. The `ASREngine` protocol allows future backends.

---

### ADR-005: Pydantic for Configuration

**Context**: Configuration validation and serialization needs a robust solution.

**Decision**: Use Pydantic v2 for configuration models.

**Rationale**: Pydantic provides runtime type validation, serialization to/from YAML/JSON/dict, environment variable support via `pydantic-settings`, and excellent IDE support. It is the de facto standard for data validation in Python.

**Consequences**: Pydantic v2 is a required dependency. Configuration changes are validated at load time, catching errors early.

---

### ADR-006: Click for CLI

**Context**: The CLI needs subcommands, option parsing, help generation, and shell completion.

**Decision**: Use `click` for the CLI framework.

**Rationale**: Click is mature, well-documented, and provides composable command groups. It supports shell completion generation, type validation, and integrates naturally with Python. BSD license is compatible with GPLv2.

**Consequences**: Click is a required dependency. CLI commands are Python functions decorated with `@click.command()`.

---

### ADR-007: Poetry for Dependency Management

**Context**: The project needs a dependency manager that handles virtual environments, lock files, and publishing.

**Decision**: Use Poetry for dependency and build management.

**Rationale**: Poetry provides deterministic dependency resolution (`poetry.lock`), virtual environment management, a clean `pyproject.toml`-based configuration, and supports PEP 517 builds. It handles both development and production dependencies cleanly.

**Consequences**: Contributors must install Poetry. `pyproject.toml` is the single source of project metadata.

---

### ADR-008: Dual Diarization Strategy

**Context**: Speaker diarization requirements range from simple (local user vs. remote) to complex (multiple remote speakers).

**Decision**: Support both channel-based (heuristic) and ML-based (pyannote) diarization, combinable in a hybrid mode.

**Rationale**: Channel-based diarization is instant and reliable when audio sources are distinct (mic vs. loopback). ML-based diarization handles single-stream multi-speaker scenarios. Hybrid mode combines both: channel splits local/remote, then ML sub-segments remote audio.

**Consequences**: pyannote.audio is an optional dependency (only needed for ML diarization). A HuggingFace auth token is required for pyannote models.

---

### ADR-009: Pluggable Translation with Registry Pattern

**Context**: Users need different translation backends depending on their requirements (offline privacy, online quality, cost).

**Decision**: Use a registry pattern where translation backends are registered by name and selected via configuration.

**Rationale**: This allows adding new translation backends without modifying existing code. Users can choose based on their constraints (privacy = argos/NLLB, quality = DeepL, cost = LibreTranslate self-hosted).

**Consequences**: Each backend is an independent module. The registry provides a single factory point. Default backend (Argos) works offline with no API keys.

---

### ADR-010: GPLv2 License Compatibility

**Context**: The project is licensed under GPLv2. All dependencies must be compatible.

**Decision**: Only use dependencies with GPLv2-compatible licenses (MIT, BSD, Apache 2.0, LGPL, MPL 2.0).

**Rationale**: GPLv2 requires that the combined work is distributable under GPLv2. Permissive licenses (MIT, BSD, Apache 2.0) are compatible. LGPL and MPL 2.0 are compatible when used as libraries.

**Consequences**: Every new dependency must have its license checked before inclusion. A license audit should be part of the CI pipeline.

---

## Appendix A: Dependency Matrix

### Core Dependencies

| Package | Version | License | Purpose |
|---------|---------|---------|---------|
| `faster-whisper` | >= 1.0 | Apache 2.0 | ASR engine |
| `ctranslate2` | >= 4.0 | MIT | ASR model runtime |
| `numpy` | >= 1.24 | BSD | Audio array operations |
| `pydantic` | >= 2.0 | MIT | Configuration validation |
| `pydantic-settings` | >= 2.0 | MIT | Environment variable config |
| `click` | >= 8.0 | BSD | CLI framework |
| `sounddevice` | >= 0.4 | MIT | Cross-platform audio I/O |
| `structlog` | >= 23.0 | Apache 2.0 / MIT | Structured logging |
| `pyyaml` | >= 6.0 | MIT | YAML config parsing |
| `httpx` | >= 0.24 | BSD | Async HTTP client (for API backends) |

### Optional Dependencies

| Package | Version | License | Purpose | Install Extra |
|---------|---------|---------|---------|---------------|
| `pyannote.audio` | >= 3.1 | MIT | ML-based diarization | `[diarization]` |
| `torch` | >= 2.0 | BSD | ML model runtime | `[diarization]` or `[gpu]` |
| `argostranslate` | >= 1.9 | MIT | Offline translation | `[translation-offline]` |
| `noisereduce` | >= 3.0 | MIT | Audio noise reduction | `[noise-reduction]` |
| `soxr` | >= 0.3 | LGPL 2.1 | High-quality resampling | `[audio-quality]` |
| `librosa` | >= 0.10 | ISC | Audio analysis utilities | `[audio-quality]` |
| `cryptography` | >= 41.0 | Apache 2.0 / BSD | Output encryption | `[security]` |
| `pycaw` | >= 20230407 | MIT | Windows WASAPI enumeration | (Windows auto) |
| `pulsectl` | >= 23.5 | MIT | Linux PulseAudio control | (Linux auto) |
| `pyobjc-framework-CoreAudio` | >= 10.0 | MIT | macOS audio | (macOS auto) |
| `pyobjc-framework-ScreenCaptureKit` | >= 10.0 | MIT | macOS screen audio | (macOS auto) |

### Development Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | >= 7.0 | Test framework |
| `pytest-asyncio` | >= 0.21 | Async test support |
| `pytest-cov` | >= 4.0 | Coverage reporting |
| `ruff` | >= 0.1 | Linting and formatting |
| `mypy` | >= 1.5 | Static type checking |
| `pip-audit` | >= 2.6 | Dependency vulnerability scanning |
| `pre-commit` | >= 3.0 | Git hook management |

---

## Appendix B: Glossary

| Term | Definition |
|------|-----------|
| **ASR** | Automatic Speech Recognition -- converting audio to text |
| **CTranslate2** | Optimized inference engine for Transformer models |
| **DER** | Diarization Error Rate -- metric for diarization quality |
| **Diarization** | Segmenting audio by speaker identity ("who spoke when") |
| **Loopback** | Capturing audio output from the system's speakers/headphones |
| **NMT** | Neural Machine Translation |
| **VAD** | Voice Activity Detection -- detecting speech vs. silence |
| **WASAPI** | Windows Audio Session API |
| **WER** | Word Error Rate -- metric for transcription accuracy |
| **Whisper** | OpenAI's speech recognition model family |
