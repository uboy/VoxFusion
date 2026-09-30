# Changelog

All notable changes to VoxFusion are documented here.

## [Unreleased]

### Changed
- **API**: uploaded file is deleted immediately when a job completes successfully (owner request 2026-09-29: the upload is a transient artifact, "transcribe → return result → clean up"). Files of **failed** jobs are kept on disk so transcription can be re-run after the failure is fixed; the retention sweep (`VOXFUSION_API_RETENTION_HOURS`, 24 h) remains the safety net for them.
- **API**: response fields renamed/added - `duration_s` (was ambiguous, actually processing time) is now `processing_time_s`; new `audio_duration_s` carries the real audio duration from the pipeline (`source_info[duration_s]`); README documents both.
- **API**: partial upload file is now removed on client disconnect/timeout mid-upload (`asyncio.CancelledError` is a BaseException that slipped past `except Exception`, leaving an orphan in uploads until the retention sweep).
- **Diarization**: a locally cached pyannote model now satisfies the ML prerequisites without a HuggingFace token (a cached snapshot loads without auth, so fully offline installs keep ML diarization instead of silently falling back to channel); the token gate in `pyannote_engine` allows the same cache fallback. Without cache and token the behaviour is unchanged.

### Added
- **API**: `VOXFUSION_API_DIARIZATION_MODEL` env var (default `pyannote/speaker-diarization-3.1`) selects the diarization model, e.g. `pyannote/speaker-diarization-community-1` (CC-BY-4.0, pyannote.audio 4+); the local-cache token gate checks whichever model is configured.
- **API**: `min_speakers` / `max_speakers` form fields on `POST /v1/transcribe` - optional speaker-count hints passed to the diarization engine per job (blank = engine default; `min > max` is rejected with 400).
- **API**: job ETA - `POST /v1/transcribe` response and every job record now carry `eta_seconds` (audio duration via ffprobe × `VOXFUSION_API_ETA_MULTIPLIER`, default `1.8` = measured RTF ~1.5 on 2 CPU cores plus a ~20% buffer; `null` when ffprobe cannot read the duration, upload keeps working).
- **API**: `POST /v1/jobs/{id}/cancel` - a queued job is cancelled immediately (upload removed), a running job flips to the new terminal status `cancelled` within one pipeline progress interval (~seconds) via cooperative cancellation; cancelling a finished job is a no-op; `cancelled` jobs delete their upload like `done` ones. Note: cancelling gives up the result - the model call already running in its executor thread is not preempted and keeps the CPU busy until it finishes.
- **API**: `VOXFUSION_API_DIARIZATION_STRATEGY` env var (default `channel`) plugs the diarization strategy into `_CONFIG_OVERRIDES`; `auto` makes uploaded files go through ML diarization when pyannote + token/cache are available, an invalid value fails at service startup.
- **CI**: GitHub Actions workflow — lint (ruff, mypy) + test matrix Python 3.11/3.12 (P0 TEST-1)
- **Security**: `trust_remote_code=True` risk documented in README § Security and surfaced as a runtime warning on every GigaAM model load (P1 SEC-1)
- **GUI**: `[ERROR]` / `[WARNING]` text prefixes on all three status label areas (live, file, LLM) with colour coding (P1 UX-2)
- **GUI**: Cancel button for LLM summarization with cooperative cancellation in `LLMWorker` (P1 UX-3)
- **GUI**: VTT transcript import — existing `.txt`, `.srt`, `.vtt`, `.md` files can be loaded into the file-results table without re-running transcription
- **GUI**: Export button supporting `TXT`, `VTT`, and `SRT` from the file-results table
- **GUI**: Model redownload guard — warns before downloading an already-cached model
- **GUI**: LLM preflight check — lightweight API/model readiness check before sending the full transcript
- **GUI**: Chunked/hierarchical summarization fallback when transcript exceeds model context window
- **GUI**: UTF-8 byte-based LLM token estimation (`ceil(bytes / 4)`) — fixes undercount for Cyrillic/emoji text (P1 ASR-3)
- **GUI**: Manual context-window override field alongside Open WebUI controls
- **GUI**: Open WebUI model-list cache — last successful list retained across 503 transient errors
- **GUI**: `Test Model` button for real completion smoke-check against the selected model
- **ASR**: GigaAM v3 multi-variant support: `e2e_ctc`, `e2e_rnnt`, `ctc`, `rnnt`
- **ASR**: Directory input for batch transcription (`voxfusion transcribe /path/to/dir/`)
- **ASR**: Parakeet v3 and Breeze ASR in model catalog; download via `voxfusion models download --asr`
- **ASR**: OpenVINO Whisper auto-selection when Intel Iris Xe / Arc GPU is detected
- **CLI**: `--input-list` flag for `voxfusion transcribe` to process a text playlist of files
- **Config**: `GIGAAM_REVISIONS` centralised in `asr_catalog.py` — single source of truth (P1 ARCH-2)

### Changed
- **Code**: Replaced `object` with concrete `GigaAMModelProtocol` type in `gigaam_engine.py`; `EventCallback` type alias in `transcribe_cmd.py` (P1 CODE-2)
- **GUI**: Resizable panes for setup, transcript, LLM output, and log areas
- **GUI**: `Normal` / `Debug` log mode toggle in toolbar
- **GUI**: Language switcher (English / Russian / Chinese) persisted across launches
- **Diarization**: `auto` strategy prefers ML when pyannote + HF token are available; falls back gracefully
- **Diarization**: `hybrid` strategy combining channel and ML approaches

### Fixed
- Diarization speaker alignment and segment boundary issues
- GigaAM model download and loading reliability
- Timer ETA display
- Audio recording format handling (WAV, FLAC, MP3)
- Dependency resolution for optional extras (gigaam, parakeet, diarization)
- Proxy support for model downloads

## [0.1.0] — initial release

### Added
- Live audio capture: microphone, system loopback (WASAPI), and simultaneous `both` via `AudioMixer`
- Raw audio recording to WAV (`voxfusion record`, GUI Record Audio with Pause/Resume)
- Batch file transcription via faster-whisper (CPU / CUDA / OpenVINO auto-selection)
- GigaAM v3 ONNX/CTC backend for Russian batch and live transcription
- Speaker diarization: pyannote.audio ML, channel-based, and hybrid strategies
- Offline translation via Argos Translate (no API keys)
- Output formats: JSON, SRT, VTT, plain text
- GUI multi-step workflow: record → transcribe → send to Open WebUI LLM
- CLI subcommands: `capture`, `record`, `transcribe`, `devices`, `models`, `config`
- Binary packaging via PyInstaller (`scripts/build_binaries.py`) for Windows, macOS, Linux
- Project architecture documentation (`docs/ARCHITECTURE.md`)
