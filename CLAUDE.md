# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VoxFusion is a cross-platform (Windows, macOS, Linux) audio capture and transcription system. It captures any system audio -- mic input, calls, music, browser/app playback -- and produces speech transcriptions with speaker diarization and automatic translation. It supports both real-time streaming and batch processing workflows.

**License**: GPLv2 -- all contributions and derivative works must remain open-source under the same license. Every dependency must have a GPLv2-compatible license (MIT, BSD, Apache 2.0, LGPL, MPL 2.0).

## Architecture Summary

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full architecture document with detailed module interfaces, data flow diagrams, and ADRs.

### Pipeline

```
Audio Capture -> Pre-Processing -> ASR (faster-whisper) -> Diarization -> Translation -> Output Formatter -> CLI/API
```

Each stage is a modular component defined by a `typing.Protocol` interface. Stages communicate via `asyncio.Queue` in streaming mode or direct calls in batch mode.

### Key Technology Choices

| Component | Technology | Why |
|-----------|-----------|-----|
| ASR Engine | faster-whisper (CTranslate2) | 4x faster than original Whisper, int8 quantization, Apache 2.0 |
| Diarization | pyannote.audio + channel-based | ML for multi-speaker, channel-based for mic vs. system split |
| Translation | Argos Translate (offline default) | MIT license, no API keys needed, pluggable registry for alternatives |
| CLI | Click | Mature, composable subcommands, BSD license |
| Configuration | Pydantic v2 + YAML | Runtime validation, env var support, type safety |
| Concurrency | asyncio + ProcessPoolExecutor | Async I/O for streaming, executor for CPU-bound inference |
| Audio I/O | sounddevice + platform-specific libs | Cross-platform with native fallbacks (WASAPI, CoreAudio, PulseAudio) |

### Core Design Decisions

- **Async-first**: All pipeline interfaces use `async` methods. CPU-bound work is offloaded via `asyncio.loop.run_in_executor()`.
- **Protocol-based interfaces**: Use `typing.Protocol` (PEP 544) for all component boundaries. No ABC inheritance.
- **Streaming-first**: Architecture is designed around streaming data flow with backpressure. Batch mode is a degenerate case of streaming.
- **Graceful degradation**: If a component is unavailable (e.g., no GPU, no pyannote), the system falls back to simpler alternatives and logs a warning.
- **`src/` layout**: Package lives under `src/voxfusion/` to prevent accidental imports from the repo root.

## Package Structure

```
src/voxfusion/
    __init__.py               # Public API exports
    __main__.py               # python -m voxfusion entry point
    version.py                # Single source of version
    exceptions.py             # Exception hierarchy
    logging.py                # structlog configuration

    capture/                  # Audio capture (WASAPI, CoreAudio, PulseAudio)
        base.py               # AudioCaptureSource protocol
        factory.py            # Platform-detection factory
        wasapi.py / coreaudio.py / pulseaudio.py
        file_source.py        # File-based source for batch mode
        mixer.py              # Multi-source audio mixer
        enumerator.py         # Device enumeration

    preprocessing/            # Audio pre-processing
        base.py               # AudioPreProcessor protocol
        pipeline.py           # Composable pre-processing chain
        resample.py / normalize.py / vad.py / noise.py

    asr/                      # Automatic Speech Recognition
        base.py               # ASREngine protocol
        faster_whisper.py     # Primary ASR implementation
        streaming.py          # Chunked streaming wrapper
        dedup.py              # Overlap deduplication

    diarization/              # Speaker diarization
        base.py               # DiarizationEngine protocol
        channel.py            # Channel-based (deterministic)
        pyannote_engine.py    # pyannote.audio ML-based
        hybrid.py             # Combined strategy
        alignment.py          # ASR-diarization segment alignment

    translation/              # Text translation
        base.py               # TranslationEngine protocol
        registry.py           # Backend registry pattern
        argos_engine.py / nllb_engine.py / deepl_engine.py / libretranslate.py
        cache.py              # LRU translation cache

    output/                   # Output formatting
        base.py               # OutputFormatter protocol
        json_formatter.py / srt_formatter.py / vtt_formatter.py / txt_formatter.py

    pipeline/                 # Pipeline orchestration
        orchestrator.py       # PipelineOrchestrator
        streaming.py / batch.py
        events.py             # Event types

    config/                   # Configuration
        models.py             # Pydantic config models
        loader.py             # File loading and merging
        defaults.yaml         # Built-in defaults

    cli/                      # CLI (Click-based)
        main.py               # Entry point
        capture_cmd.py / transcribe_cmd.py / devices_cmd.py
        config_cmd.py / models_cmd.py
        formatting.py         # Output helpers

    models/                   # Shared data types
        audio.py / transcription.py / diarization.py / translation.py / result.py

    security/                 # Encryption and permissions
        encryption.py / permissions.py
```

Test structure mirrors `src/` under `tests/unit/`, `tests/integration/`, and `tests/platform/`.

## Development Setup

### Prerequisites

- Python 3.11 or later
- Poetry (dependency management)
- FFmpeg (optional, for some audio operations)
- CUDA toolkit (optional, for GPU-accelerated inference)

### Initial Setup

```bash
# Clone the repository
git clone <repo-url> VoxFusion
cd VoxFusion

# Install Poetry (if not already installed)
pip install poetry

# Install all dependencies (including dev)
poetry install --all-extras

# Or install with specific extras only
poetry install --extras "diarization"
poetry install --extras "translation-offline"

# Activate the virtual environment
poetry shell
```

### Running the Application

```bash
# Via Poetry
poetry run voxfusion --help
poetry run voxfusion capture
poetry run voxfusion transcribe recording.wav --output-format json

# Via python -m
python -m voxfusion --help
python -m voxfusion transcribe recording.wav

# Direct (after poetry shell or pip install -e .)
voxfusion --help
```

### Running Tests

```bash
# All tests
poetry run pytest

# Unit tests only (fast, no models needed)
poetry run pytest tests/unit/

# Integration tests (may need models downloaded)
poetry run pytest tests/integration/

# Platform-specific tests
poetry run pytest tests/platform/

# With coverage
poetry run pytest --cov=voxfusion --cov-report=html tests/

# Single test file
poetry run pytest tests/unit/test_output_formatters.py -v
```

### Linting and Type Checking

```bash
# Lint with ruff
poetry run ruff check src/ tests/

# Auto-fix lint issues
poetry run ruff check --fix src/ tests/

# Format with ruff
poetry run ruff format src/ tests/

# Check formatting without modifying
poetry run ruff format --check src/ tests/

# Type check with mypy
poetry run mypy src/

# Dependency vulnerability scan
poetry run pip-audit
```

### Pre-Commit Hooks

```bash
# Install pre-commit hooks
poetry run pre-commit install

# Run hooks on all files manually
poetry run pre-commit run --all-files
```

### Model Management

```bash
# Download ASR model
voxfusion models download --asr large-v3

# Download diarization model (requires HF token)
export VOXFUSION_DIARIZATION__HF_AUTH_TOKEN=hf_your_token
voxfusion models download --diarization pyannote

# Download translation models
voxfusion models download --translation argos fr-en

# List installed models
voxfusion models list
```

## Coding Conventions

### General

- Python 3.11+ syntax: use `X | Y` union types, `match` statements, `StrEnum`, etc.
- All modules must have module-level docstrings.
- All public functions and classes must have docstrings (Google style).
- Use `from __future__ import annotations` only if needed for forward references.
- Imports ordered: stdlib, third-party, local (enforced by ruff isort rules).

### Type Annotations

- All function signatures must have full type annotations.
- Use `typing.Protocol` for interfaces, not ABCs.
- Use `@runtime_checkable` sparingly -- only when runtime isinstance checks are needed.
- Prefer `X | None` over `Optional[X]`.
- Use `list[X]` and `dict[K, V]` (lowercase) not `List[X]` / `Dict[K, V]`.

### Async

- All pipeline component methods that perform I/O or inference are `async`.
- CPU-bound work must be offloaded to an executor via `asyncio.loop.run_in_executor()`.
- Use `asyncio.Queue` for inter-stage communication in streaming mode.
- Never use `asyncio.sleep(0)` as a yield point -- use proper awaitable operations.

### Error Handling

- Raise specific exceptions from the `voxfusion.exceptions` hierarchy.
- Never catch bare `Exception` -- always catch the most specific type.
- Use `logging.exception()` for unexpected errors to capture tracebacks.
- Pipeline stages should log and skip on non-fatal errors, not crash.

### Testing

- Test files mirror source files: `src/voxfusion/asr/faster_whisper.py` -> `tests/unit/test_faster_whisper.py`.
- Use `pytest` fixtures (defined in `conftest.py`) for shared setup.
- Use `pytest-asyncio` for async test functions (`@pytest.mark.asyncio`).
- Unit tests must not require GPU, internet, or downloaded models.
- Integration tests are marked with `@pytest.mark.integration` and may require models.
- Platform tests are marked with `@pytest.mark.platform` and skipped on wrong OS.

### Configuration

- All config values have sensible defaults in Pydantic models.
- Environment variables use `VOXFUSION_` prefix with `__` for nesting.
- Config files are YAML format.
- Never hardcode paths, API keys, or model names -- always use config.

### Git

- Conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
- One logical change per commit.
- Branch naming: `feat/short-description`, `fix/short-description`.

## Key Files Reference

| File | Purpose |
|------|---------|
| `pyproject.toml` | Project metadata, dependencies, tool configuration |
| `src/voxfusion/__init__.py` | Public API surface |
| `src/voxfusion/exceptions.py` | Complete exception hierarchy |
| `src/voxfusion/config/models.py` | All Pydantic configuration models |
| `src/voxfusion/pipeline/orchestrator.py` | Central pipeline coordination |
| `src/voxfusion/cli/main.py` | CLI entry point |
| `src/voxfusion/models/` | All shared data types (AudioChunk, TranscriptionSegment, etc.) |
| `tests/conftest.py` | Shared test fixtures |
| `docs/ARCHITECTURE.md` | Full architecture document |

## Current Status

v0.1.0 is implemented and working. Key capabilities shipped:
- Live audio capture (WASAPI mic + system loopback, `AudioMixer` for `both`)
- Raw audio recording to WAV (`voxfusion record`, GUI Record Audio button with Pause/Resume)
- Batch file transcription via faster-whisper (CPU/CUDA/OpenVINO auto-selection)
- GigaAM v3 ONNX/CTC backend for Russian batch transcription
- Parakeet V3 and Breeze ASR in the model catalog (backends pending download/implementation)
- GUI: multi-step workflow (record → transcribe → send to Open WebUI LLM)
- GUI settings persistence, WASAPI-only device list, resizable log pane
- Binary packaging via PyInstaller (`scripts/build_binaries.py`)
