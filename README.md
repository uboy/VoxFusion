# VoxFusion

Cross-platform audio capture and transcription. Records any audio — microphone, system playback, calls, browser tabs — and produces text transcriptions with speaker diarization and optional translation.

Runs on Windows, macOS, and Linux. Comes with a GUI and a CLI.

## Features

- **Live transcription** — real-time speech-to-text from mic, system audio, or both simultaneously
- **File transcription** — WAV, FLAC, and (with FFmpeg) MP3, MP4, MKV, AAC, and more
- **Batch file transcription** — queue multiple files in the GUI or process a text playlist in the CLI
- **Two log modes** — `Normal` keeps only key stages and errors, `Debug` shows full stage-by-stage logs
- **Multiple ASR backends**
  - Whisper via faster-whisper — live and file, 99 languages
  - GigaAM v3 — best accuracy for Russian, file transcription plus quality-first live draft + finalize mode
  - Breeze ASR — Whisper-based multilingual, file only
  - Parakeet v3 — 25 European languages incl. Russian/Ukrainian, file only
  - OpenVINO Whisper — automatic when Intel Iris Xe / Arc GPU is detected
- **Speaker diarization** — identifies who said what (pyannote.audio or channel-based)
  - file transcription can use `auto`, `ml`, `hybrid`, or `channel`
  - `auto` prefers ML diarization when pyannote + Hugging Face token are available
  - pyannote diarization requires gated Hugging Face access not only to `pyannote/speaker-diarization-3.1`, but also to its dependent gated models such as `pyannote/segmentation-3.0`
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

- **Windows, binary build:** if FFmpeg is missing from `PATH` during `scripts/build_binaries.py`, VoxFusion automatically downloads a portable FFmpeg build and bundles it into the app.
- **Windows, bundle layout:** in PyInstaller `--onedir` builds, bundled FFmpeg lives under the app `_internal/` directory and is discovered automatically at runtime.
- **Windows, runtime:** the GUI can install a local portable FFmpeg copy under `~/.voxfusion/ffmpeg/` if neither the bundle nor `PATH` provides it.
- **GUI speaker detect:** the file-tab `Detect` button uses the same FFmpeg extraction path as full transcription, so container formats such as `webm`, `mp4`, and compressed audio work there too.
- **Windows, manual:** download from [gyan.dev/ffmpeg/builds](https://www.gyan.dev/ffmpeg/builds/) and add to PATH
- **Linux:** `sudo apt install ffmpeg`
- **macOS:** `brew install ffmpeg`

### pyannote gated models

ML speaker diarization uses `pyannote.audio` and downloads gated models from Hugging Face on first use.
Frozen GUI/CLI bundles also need the packaged `pyannote/audio/telemetry/config.yaml` file at runtime; `scripts/build_binaries.py` now includes it automatically when `pyannote.audio` is installed.

Before running diarization, make sure you:

- created a Hugging Face token
- accepted access for `pyannote/speaker-diarization-3.1`
- accepted access for `pyannote/segmentation-3.0`
- entered the token in GUI `Settings -> HuggingFace Token` or set `HF_TOKEN` / `VOXFUSION_DIARIZATION__ML__HF_AUTH_TOKEN`

After the first successful download, the models are cached locally and can be reused offline.

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

# Live GigaAM draft transcription with stop-time finalization
voxfusion capture --model gigaam-v3-e2e-ctc
voxfusion capture --model gigaam-v3-e2e-ctc --source system --device pa:3

# Record to WAV without transcription
voxfusion record --source microphone --output recording.wav

# Transcribe a file
voxfusion transcribe recording.wav
voxfusion transcribe recording.wav --output-format srt
voxfusion transcribe interview.mp4 --output-format json   # requires FFmpeg
voxfusion transcribe meeting.wav --diarization-strategy auto
voxfusion transcribe meeting.wav --diarization-strategy ml --min-speakers 2 --max-speakers 6

# Transcribe multiple files sequentially
voxfusion transcribe part1.wav part2.wav part3.wav
voxfusion transcribe --input-list batch.txt --output-dir transcripts

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
Live GigaAM uses a draft-plus-finalize flow: while capture is active it emits provisional transcript lines, and after stop it reuses successful draft utterances by default while reprocessing only deferred or failed utterances from the spooled session audio before saving or replacing the GUI draft. If you need the older full second-pass behavior, set `VOXFUSION_LIVE_GIGAAM__STOP_FINALIZE_MODE=always`. Live GigaAM currently does not support translation.
In the GUI File Transcription tab, `Add Files...` creates a queue with per-file status, progress, and result columns, and VoxFusion processes those files one after another with the current settings.
Queue metadata (duration and size) is loaded asynchronously, so adding large `webm`/video files does not block the GUI; transcription can start immediately and the metadata fills in when ready.
The GUI layout uses resizable panes for setup, transcript, LLM output, and logs, and the interface language can be switched between English, Russian, and Chinese from the top-right header with the choice persisted across launches.
The GUI toolbar includes `Logs: Normal/Debug`; `Normal` keeps the log pane readable by showing key milestones and errors only, while `Debug` exposes the full structured pipeline log.
Open WebUI model loading and transcript-processing requests now emit dedicated `llm.*` / `gui.llm_*` lifecycle events in the log pane, including URL, model, and HTTP status diagnostics while keeping the API key out of the logs.
Use the GUI `Test Model` button to send a tiny real completion request to the selected model when model-list refresh alone is not enough to diagnose backend failures.
The file-results area also supports loading an existing transcript `.txt`, `.srt`, `.vtt`, or `.md` file directly into the table so it can be reviewed or sent to Open WebUI without re-running transcription.
The file-results area `Export...` button supports `TXT`, `VTT`, and `SRT`; the auto-saved sidecar transcript remains `.transcript.txt`.
If the selected ASR model is already cached locally, the GUI now warns before downloading again and lets you cancel or force a redownload in case the cache may be corrupted.
Before `Send to LLM` starts the real transcript request, VoxFusion now runs a lightweight API/model readiness check so unreachable-server failures are reported immediately instead of looking like a long model-load timeout.
If Open WebUI intermittently returns `503` while loading models, the GUI now keeps the last successful model list in a local cache and can continue using it until the server recovers.
Long transcript post-processing now automatically falls back to hierarchical chunked summarization when the selected LLM model or backend cannot accept the full transcript in one context window.
When model metadata exposes a context window, the GUI uses it before sending the request; otherwise you can set a manual context override next to the Open WebUI controls.
Opt-in hardware-in-the-loop tests for Windows live capture and live GigaAM live mode live under `tests/hardware/` and do not run in the default suite; use `pytest tests/hardware -m hardware --run-hardware -v` when you have real devices and local model assets prepared.
Hardware test environment variables:
- `VOXFUSION_HW_MIC_DEVICE` — optional microphone device id like `sd:17` for the mic smoke test
- `VOXFUSION_HW_SYSTEM_DEVICE` — optional loopback device id like `pa:21` or `sd:5`
- `VOXFUSION_HW_PLAYBACK_DEVICE` — optional sounddevice output id `sd:<index>` for controlled WAV playback; leave unset to use the default output device
- `VOXFUSION_HW_SPEECH_WAV` — local speech WAV file used for controlled loopback playback
- `VOXFUSION_HW_GIGAAM_MODEL_PATH` — local GigaAM model directory for the live GigaAM hardware e2e; hardware tests intentionally skip instead of downloading models from the internet
For reliable results, prefer a deterministic loopback setup such as VB-CABLE or another virtual audio cable and point both the playback and loopback env vars at the same Windows output path.
For longer manual validation, use the separate replay-based soak/load harness instead of pytest. A typical real 10-minute run is:

```powershell
.\venv\Scripts\python.exe scripts\live_gigaam_soak.py --duration-minutes 10 --json-out build\live_gigaam_soak\run-10m.json
```

The harness replays a local WAV through the real `LiveGigaAMSessionController`, emits periodic progress logs, and writes a final JSON summary with backlog and latency metrics. By default it auto-discovers repo-local `tmp_sample_10s.wav` / `tmp_sample_120s.wav` and a cached local GigaAM snapshot when available. For a cheap script-level smoke run without the real model, use `--mode fake`.
In the CLI, the default mode is the same compact `Normal` output. Use `voxfusion --debug ...` (or `-v`) when you need the full low-level logs.

## Build binaries

Produces self-contained `--onedir` bundles and ZIP archives under `dist/binaries/`.

```bash
python scripts/build_binaries.py --target gui   # GUI only
python scripts/build_binaries.py --target cli   # CLI only
python scripts/build_binaries.py --target all   # both + ZIPs
```

PyInstaller is included in Poetry dev dependencies. For pip installs: `pip install pyinstaller` first.

See `docs/BINARY_BUILD.md` for platform-specific packaging notes.

## Test validation

When ASR models are already cached locally, the most stable full-suite validation command is:

```powershell
$env:PYTHONPATH='src'
$env:HF_HUB_OFFLINE='1'
$env:VOXFUSION_FFMPEG_DIR='build\vendor\ffmpeg-runtime'
.\venv\Scripts\python.exe -m pytest -q
```

This avoids flaky proxy/network refreshes and reuses the local Hugging Face cache.

## Configuration

All settings can be set via environment variables (prefix `VOXFUSION_`, double underscore for nesting) or a YAML config file.

| Variable | Description |
|---|---|
| `VOXFUSION_ASR__MODEL_SIZE` | ASR model: `tiny`, `small`, `medium`, `large-v3`, `gigaam-v3-e2e-ctc` |
| `VOXFUSION_ASR__MODEL_PATH` | Path to a local model directory |
| `VOXFUSION_ASR__LANGUAGE` | Force language code, e.g. `ru`, `en` |
| `VOXFUSION_LIVE_GIGAAM__WORKER_COUNT` | Number of warm live GigaAM worker processes |
| `VOXFUSION_LIVE_GIGAAM__STOP_FINALIZE_MODE` | `if_needed` to reprocess only deferred/failed utterances on stop, `always` for a full second pass |
| `VOXFUSION_LIVE_GIGAAM__FINALIZE_LEFT_CONTEXT_MS` | Left context kept for stop-time live GigaAM finalization |
| `VOXFUSION_LIVE_GIGAAM__QUEUE_HARD_LIMIT_JOBS` | Maximum live GigaAM draft backlog before new utterances defer to finalization-only |
| `VOXFUSION_DIARIZATION__STRATEGY` | Diarization mode: `auto`, `channel`, `ml`, `hybrid` |
| `VOXFUSION_DIARIZATION__ML__HF_AUTH_TOKEN` | HuggingFace token for pyannote diarization models |
| `VOXFUSION_GUI_SETTINGS_PATH` | Override GUI settings file location |

GUI settings persist to `~/.voxfusion/gui_settings.json`.
The GUI file-transcription tab also persists speaker-separation settings and reuses the same Hugging Face token for gated models and pyannote diarization.

## Docs

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — pipeline design, module interfaces, ADRs
- [`docs/BINARY_BUILD.md`](docs/BINARY_BUILD.md) — binary packaging and Windows notes
- [`docs/QUICK_START_RU.md`](docs/QUICK_START_RU.md) — quick start guide in Russian

## License

GPLv2. All contributions and derivative works must remain open-source under the same license.
