# Changelog

All notable changes to VoxFusion are documented here.

## [Unreleased]

### Added
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
