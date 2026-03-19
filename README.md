# VoxFusion

VoxFusion captures any system audio — mic input, calls, music, browser or app playback — and turns it into high-quality speech transcriptions with speaker diarization and automatic translation. Built for Windows, macOS and Linux.

## Installation

**Requires Python 3.11+**

```bash
# With Poetry (recommended — installs all deps including dev tools)
pip install poetry
poetry install

# Or with pip directly
pip install -e .
```

The default install now includes runtime dependencies for the built-in Whisper path plus file-only backends such as GigaAM, Breeze, and Parakeet. The first install is heavier because it pulls model runtimes like `torch`, `transformers`, `onnxruntime`, and NeMo ASR support.

Optional backends are exposed when their Python packages are present, but actual transcription still depends on model artifacts being available locally or downloadable from Hugging Face / NeMo. `GigaAM`, `Breeze`, and `Parakeet` remain file-transcription-only backends.

To also install PyInstaller for binary builds:
```bash
# Poetry (included automatically in dev group)
poetry install

# pip
pip install pyinstaller
```

## Run

```bash
# GUI
voxfusion-gui
# or: python -m voxfusion.gui.main

# CLI help
voxfusion --help
# or: python -m voxfusion --help
```

## Common commands

```bash
# Live transcription (mic)
voxfusion capture

# Live transcription (mic + system audio)
voxfusion capture --source both

# Raw audio recording to WAV (no transcription)
voxfusion record --source microphone --output recording.wav

# Batch transcription from file
voxfusion transcribe recording.wav --output-format json

# Download ASR model
voxfusion models download --asr small
voxfusion models download --asr gigaam-v3-e2e-ctc

# List audio devices
voxfusion devices

# Record Windows system audio from an explicit loopback device
voxfusion devices --type loopback
voxfusion record --source system --device pa:3 --output system.wav
```

On Windows, `voxfusion devices` now prints explicit device ids such as `pa:3` (PyAudioWPatch loopback) and `sd:7` (sounddevice/WASAPI). For system-audio capture, `pa:*` devices are the preferred path.

## Build binaries

```bash
# Install PyInstaller first if using pip (not Poetry)
pip install pyinstaller

# Build GUI + CLI + ZIP archives
python scripts/build_binaries.py --target all

# GUI only
python scripts/build_binaries.py --target gui
```

See `docs/BINARY_BUILD.md` for packaging details and Windows notes.

## Environment variables

| Variable | Description |
|---|---|
| `VOXFUSION_ASR__MODEL_SIZE` | ASR model (tiny, small, medium, large-v3) |
| `VOXFUSION_ASR__MODEL_PATH` | Path to local model directory (e.g. GigaAM) |
| `VOXFUSION_ASR__LANGUAGE` | Force language code (e.g. `ru`, `en`) |
| `VOXFUSION_DIARIZATION__HF_AUTH_TOKEN` | HuggingFace token for pyannote diarization |
| `VOXFUSION_GUI_SETTINGS_PATH` | Override the GUI settings file location |

GUI settings are persisted by default in `~/.voxfusion/gui_settings.json`.

## Docs

- `docs/ARCHITECTURE.md` — full architecture document
- `docs/BINARY_BUILD.md` — packaging and Windows notes
- `docs/QUICK_START_RU.md` — quick start guide (Russian)
