# VoxFusion

Cross-platform audio capture and transcription. Records any audio — microphone, system playback, calls, browser tabs — and produces text transcriptions with speaker diarization and optional translation.

Runs on Windows, macOS, and Linux. Comes with a GUI and a CLI.

## Features

- **Live transcription** — real-time speech-to-text from mic, system audio, or both simultaneously
- **File transcription** — WAV, FLAC, and (with FFmpeg) MP3, MP4, MKV, AAC, and more
- **Multiple ASR backends**
  - Whisper via faster-whisper — live and file, 99 languages
  - GigaAM v3 — best accuracy for Russian, file only
  - Breeze ASR — Whisper-based multilingual, file only
  - Parakeet v3 — 25 European languages incl. Russian/Ukrainian, file only
  - OpenVINO Whisper — automatic when Intel Iris Xe / Arc GPU is detected
- **Speaker diarization** — identifies who said what (pyannote.audio or channel-based)
- **Offline translation** — no API keys required (Argos Translate)
- **Output formats** — JSON, SRT, VTT, plain text
- **GUI** — multi-step workflow: record → transcribe → send to LLM (Open WebUI compatible)
- **CLI** — scriptable, composable subcommands
- **Binary builds** — self-contained executables via PyInstaller

## Requirements

- Python 3.11+
- FFmpeg (optional — needed only for video and compressed audio files)

## Quick start

**Run (GUI):**

```bash
# bash / macOS / Linux
pip install poetry && poetry install && poetry run voxfusion-gui
```

```powershell
# PowerShell (Windows)
pip install poetry; poetry install; poetry run voxfusion-gui
```

**Build standalone binary (GUI):**

```bash
# bash
pip install poetry && poetry install && poetry run python scripts/build_binaries.py --target gui --skip-install
```

```powershell
# PowerShell (Windows)
pip install poetry; poetry install; poetry run python scripts/build_binaries.py --target gui --skip-install
```

## Installation

### Poetry

```bash
pip install poetry
poetry install
```

To add optional ASR backends:

```bash
poetry install --extras gigaam      # GigaAM v3 (Russian)
poetry install --extras parakeet    # Parakeet v3 (25 languages, ~2 GB)
```

### pip / venv

```bash
pip install -e .                          # Windows
pip install -e .[linux]                   # Linux
pip install -e .[macos]                   # macOS

pip install -e .[gigaam]                  # + GigaAM backend
pip install -e .[linux,gigaam,parakeet]   # multiple extras
```

Available extras: `gigaam`, `parakeet`, `diarization`, `translation-offline`, `audio-quality`, `noise-reduction`, `security`, `linux`, `macos`.

### FFmpeg

Required for transcribing video files and compressed audio (MP3, MP4, MKV, AAC…). WAV and FLAC work without it.

- **Windows:** download from [gyan.dev/ffmpeg/builds](https://www.gyan.dev/ffmpeg/builds/) and add to PATH
- **Linux:** `sudo apt install ffmpeg`
- **macOS:** `brew install ffmpeg`

## Running

```bash
# GUI
voxfusion-gui
python -m voxfusion.gui.main

# CLI
voxfusion --help
```

## CLI usage

```bash
# Live transcription from microphone
voxfusion capture

# Live transcription from mic + system audio simultaneously
voxfusion capture --source both

# Record to WAV without transcription
voxfusion record --source microphone --output recording.wav

# Transcribe a file
voxfusion transcribe recording.wav
voxfusion transcribe recording.wav --output-format srt
voxfusion transcribe interview.mp4 --output-format json   # requires FFmpeg

# List audio devices
voxfusion devices
voxfusion devices --type loopback

# Record Windows system audio via explicit loopback device
voxfusion record --source system --device pa:3 --output system.wav

# Download ASR model
voxfusion models download --asr large-v3
voxfusion models download --asr gigaam-v3-e2e-ctc
```

On Windows, `voxfusion devices` prints device IDs like `pa:3` (PyAudioWPatch loopback) and `sd:7` (WASAPI). Use `pa:*` IDs for system-audio capture.

## Build binaries

Produces self-contained `--onedir` bundles and ZIP archives under `dist/binaries/`.

```bash
python scripts/build_binaries.py --target gui   # GUI only
python scripts/build_binaries.py --target cli   # CLI only
python scripts/build_binaries.py --target all   # both + ZIPs
```

PyInstaller is included in Poetry dev dependencies. For pip installs: `pip install pyinstaller` first.

See `docs/BINARY_BUILD.md` for platform-specific packaging notes.

## Configuration

All settings can be set via environment variables (prefix `VOXFUSION_`, double underscore for nesting) or a YAML config file.

| Variable | Description |
|---|---|
| `VOXFUSION_ASR__MODEL_SIZE` | ASR model: `tiny`, `small`, `medium`, `large-v3`, `gigaam-v3-e2e-ctc` |
| `VOXFUSION_ASR__MODEL_PATH` | Path to a local model directory |
| `VOXFUSION_ASR__LANGUAGE` | Force language code, e.g. `ru`, `en` |
| `VOXFUSION_DIARIZATION__HF_AUTH_TOKEN` | HuggingFace token for pyannote diarization models |
| `VOXFUSION_GUI_SETTINGS_PATH` | Override GUI settings file location |

GUI settings persist to `~/.voxfusion/gui_settings.json`.

## Docs

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — pipeline design, module interfaces, ADRs
- [`docs/BINARY_BUILD.md`](docs/BINARY_BUILD.md) — binary packaging and Windows notes
- [`docs/QUICK_START_RU.md`](docs/QUICK_START_RU.md) — quick start guide in Russian

## License

GPLv2. All contributions and derivative works must remain open-source under the same license.
