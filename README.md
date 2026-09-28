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

- Python 3.10+
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

Available extras: `gigaam`, `parakeet`, `translation-offline`, `audio-quality`, `noise-reduction`, `security`, `linux`, `macos`.

> **Note:** `pyannote-audio` (ML diarization) is included in the base install — no extra flag needed. You only need to provide a Hugging Face token (see below).

### FFmpeg

Required for transcribing video files and compressed audio (MP3, MP4, MKV, AAC…). WAV and FLAC work without it.

- **Windows, binary build:** if FFmpeg is missing from `PATH` during `scripts/build_binaries.py`, VoxFusion automatically downloads a portable FFmpeg build and bundles it into the app.
- **Windows, bundle layout:** in PyInstaller `--onedir` builds, bundled FFmpeg lives under the app `_internal/` directory and is discovered automatically at runtime.
- **Windows, runtime:** the GUI can install a local portable FFmpeg copy under `~/.voxfusion/ffmpeg/` if neither the bundle nor `PATH` provides it.
- **GUI speaker detect:** the file-tab `Detect` button uses the same FFmpeg extraction path as full transcription, so container formats such as `webm`, `mp4`, and compressed audio work there too.
- **Windows, manual:** download from [gyan.dev/ffmpeg/builds](https://www.gyan.dev/ffmpeg/builds/) and add to PATH
- **Linux:** `sudo apt install ffmpeg`
- **macOS:** `brew install ffmpeg`

### Older GPUs (V100, GTX 1080, CC 7.0)

Recent PyTorch versions (2.8+) ship with CUDA 12.6+ which **drops support** for compute capability 7.0 (Volta / Tesla V100). If you see a *"Found GPU Tesla V100 which is of compute capability (CC) 7.0"* warning, install PyTorch with an older CUDA toolkit:

```bash
# CUDA 12.4 — supports CC 7.0 and later
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Check your GPU's compute capability: `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`. CC 7.0 needs CUDA ≤ 12.4; CC 7.5+ works with the latest PyTorch.

### Setting up ML diarization

ML diarization (`--diarization-strategy ml` or `auto`) uses `pyannote.audio` (included in the base install), which requires a free Hugging Face account and accepting the license for two gated model repositories.

**One-time setup (three steps):**

1. **Create a Hugging Face account** at <https://huggingface.co> and generate a read token at <https://huggingface.co/settings/tokens>.

2. **Accept the model licenses** (you must be logged in):
   - <https://huggingface.co/pyannote/speaker-diarization-3.1>
   - <https://huggingface.co/pyannote/segmentation-3.0>

3. **Provide the token to VoxFusion**:
   - **CLI**: run any transcribe command — VoxFusion will prompt you for the token interactively if none is found
   - GUI: `Settings → HuggingFace Token`
   - Environment: `export HF_TOKEN=hf_your_token`
   - Config: set `VOXFUSION_DIARIZATION__ML__HF_AUTH_TOKEN=hf_your_token`

Then **run diarization** — VoxFusion downloads the models automatically on first use (~300 MB) and caches them locally for offline reuse:

   ```bash
   voxfusion transcribe meeting.wav --diarization-strategy ml
   voxfusion transcribe meeting.wav --diarization-strategy auto  # tries ML first
   ```

Frozen GUI/CLI bundles also need the packaged `pyannote/audio/telemetry/config.yaml` file at runtime; `scripts/build_binaries.py` includes it automatically when `pyannote.audio` is installed.

## Offline model download (air-gapped environments)

VoxFusion runs fully offline after models are downloaded. Models are cached in:
- **Linux/macOS**: `~/.cache/huggingface/hub/`
- **Windows**: `%USERPROFILE%\.cache\huggingface\hub\`

### Method 1: Automatic download (requires internet)

```bash
# GigaAM v3 (Russian, best accuracy)
voxfusion models download --asr gigaam-v3-e2e-ctc

# Whisper large-v3 (99 languages)
voxfusion models download --asr large-v3

# Pyannote diarization (requires HF token)
export HF_TOKEN=hf_your_token
voxfusion models download --diarization pyannote
```

### Method 2: Manual download (for air-gapped systems)

1. **Download model files from Hugging Face**:
   - GigaAM v3: https://huggingface.co/ai-sage/GigaAM-v3/tree/main
   - Pyannote segmentation-3.0: https://huggingface.co/pyannote/segmentation-3.0 (gated, accept license first)
   - Pyannote speaker-diarization-3.1: https://huggingface.co/pyannote/speaker-diarization-3.1 (gated)

2. **Create the cache directory structure** on the target machine:
   ```bash
   # Linux/macOS
   mkdir -p ~/.cache/huggingface/hub/models--ai-sage--GigaAM-v3/snapshots/<commit-hash>/

   # Windows
   mkdir "%USERPROFILE%\.cache\huggingface\hub\models--ai-sage--GigaAM-v3\snapshots\<commit-hash>\"
   ```

3. **Copy model files** to the snapshot directory (all files from the HF repo: `config.json`, `model.onnx`, `preprocessor_config.json`, `tokenizer_config.json`, `vocab.txt`, etc.)

4. **Create `.no_exist` files** (HuggingFace cache requirement):
   ```bash
   touch ~/.cache/huggingface/hub/models--ai-sage--GigaAM-v3/snapshots/<commit-hash>/.no_exist
   ```

5. **Create refs** pointing to the snapshot:
   ```bash
   echo "<commit-hash>" > ~/.cache/huggingface/hub/models--ai-sage--GigaAM-v3/refs/main
   ```

### Verification

Run VoxFusion in offline mode to verify models are available:
```bash
# Force offline mode (default after initial setup)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Test transcription (should use cached models)
voxfusion transcribe test.wav --model gigaam-v3-e2e-ctc
```

### Supported ASR models

Pass any of these ids to `--model` (CLI) or `VOXFUSION_ASR__MODEL_SIZE`. Run
`voxfusion models list` to see which are available in your environment. An
unknown id or a model whose packages are missing now fails with a clear error
instead of silently falling back to Whisper small.

| `--model` id | Engine | Languages | Live capture | Install extra |
|--------------|--------|-----------|:------------:|---------------|
| `tiny` | faster-whisper | 99 languages | ✅ | built-in |
| `base` | faster-whisper | 99 languages | ✅ | built-in |
| `small` *(default)* | faster-whisper | 99 languages | ✅ | built-in |
| `medium` | faster-whisper | 99 languages | ✅ | built-in |
| `large-v3` | faster-whisper | 99 languages | ✅ | built-in |
| `gigaam-v3-e2e-rnnt` | gigaam | Russian | ❌ | `gigaam` |
| `gigaam-v3-e2e-ctc` | gigaam | Russian | ✅ | `gigaam` |
| `gigaam-v3-rnnt` | gigaam | Russian | ❌ | `gigaam` |
| `gigaam-v3-ctc` | gigaam | Russian | ❌ | `gigaam` |
| `parakeet-tdt-0.6b-v3` | parakeet | 25 European langs | ❌ | `parakeet` |
| `breeze-asr` | breeze | 99 languages | ❌ | (transformers + torch) |
| `funasr-paraformer-zh` | funasr | Chinese | ❌ | `chinese` |

### Model download sizes

| Model | Size | Languages | Quality |
|-------|------|-----------|---------|
| GigaAM v3 | ~1.5 GB | Russian (primary), English | Best for RU, excellent for EN |
| Whisper large-v3 | ~3 GB | 99 languages | Good multilingual, slower |
| Pyannote segmentation-3.0 | ~200 MB | - | Diarization support |
| Pyannote speaker-diarization-3.1 | ~100 MB | - | Diarization support |

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

**Build from the project `venv`, not the system Python.** A system-Python build can succeed
while silently omitting optional ML dependencies, producing a bundle that fails only at
runtime when diarization or speaker detection is used:

```powershell
.\venv\Scripts\python.exe scripts/build_binaries.py --target gui
```

Two more packaging constraints, both found the hard way:

- **`pyannote.audio` needs its telemetry config bundled explicitly.** Frozen builds crash on
  a missing `pyannote/audio/telemetry/config.yaml` unless the build adds it as data.
- **FFmpeg paths must be absolute before they become `--add-data` entries.** A relative
  `PATH` hit resolves during the build but breaks inside the bundle.
- **A running bundled `.exe` blocks rebuilding into the same output directory.** Close the
  previous build before rebuilding.

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

## Troubleshooting

Symptoms that come up repeatedly, with the cause rather than a workaround.

### `PortAudioError: MME error 11` on start, or capture opens but stays silent

The MME host API is the default in many PortAudio builds and is the least reliable one on
Windows. VoxFusion prefers WASAPI and falls back across host APIs and channel layouts. If
you pinned a device manually, drop the pin and let it choose.

### `Selected PyAudio loopback device pa:N failed ... Errno -9998 Invalid number of channels`

Some loopback endpoints reject the first channel count they advertise. VoxFusion retries an
ordered set of channel counts and falls back to mono. If you still see this on a specific
device, capture the device list (`voxfusion devices`) when reporting it.

### `sounddevice` sees no usable device

Three causes, in order of how often they turn out to be it:

1. the endpoint is held by another program (conferencing apps keep it open);
2. exclusive mode is enabled for the device in Windows sound settings;
3. the audio driver predates the current Windows audio stack.

### `Speaker detect failed` on `.webm`, `.mp4`, or other container formats

Speaker detection must read audio through the same FFmpeg extraction path as full
transcription. Reading a container directly through `soundfile` fails by design. If a
build of yours shows this, it is a bug in the preflight path, not in your file.

### Open WebUI returns `503` with a valid API key

Test the two endpoints separately: if `/api/models` answers and only
`/api/chat/completions` fails, the key is fine and the backend or the selected model is
overloaded. The GUI `Test Model` button sends a minimal real completion for exactly this.

Related: a saved or default model id is not guaranteed to exist in a remote catalog. When
the configured model is absent, VoxFusion picks a valid one and logs the mismatch instead
of failing silently.

### Expected log lines are missing

Check the log mode first. `Normal` deliberately shows only stage transitions and errors;
low-level lines are absent by design. Switch the GUI selector to `Debug`, or pass
`--debug` / `-v` in the CLI, before treating it as a defect.

## Design notes

Decisions that are easy to mistake for accidents.

- **Batch transcription is sequential by default.** One shared transcription worker is
  reused rather than starting concurrent model workers, because model memory, not CPU, is
  the limit on a single machine.
- **The GUI playlist is an explicit queue with per-file status**, not extra rows in the
  transcript table: the two have different lifetimes.
- **Queue metadata is probed asynchronously.** Duration and size appear when available, and
  transcription can start before probing finishes.
- **Live GigaAM emits drafts and finalizes on stop** rather than transcribing once at the
  end, so a long session is usable while it runs. See `Running` above for the mode switch.

## Docs

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — pipeline design, module interfaces, ADRs
- [`docs/BINARY_BUILD.md`](docs/BINARY_BUILD.md) — binary packaging and Windows notes
- [`docs/QUICK_START_RU.md`](docs/QUICK_START_RU.md) — quick start guide in Russian

## Security

### GigaAM — `trust_remote_code=True`

Loading GigaAM models calls `AutoModel.from_pretrained(..., trust_remote_code=True)`.
This flag allows the HuggingFace repository to execute arbitrary Python code on your
machine during model loading. It is **required by GigaAM** because the model
architecture is defined in custom Python files hosted in the `ai-sage/GigaAM-v3`
repository.

**What this means in practice:**

- Code runs from the HuggingFace repo at download/load time.
- If the repo is compromised (supply-chain attack), malicious code could execute with
  your user privileges.
- Mitigation: VoxFusion pins specific branch revisions (`e2e_ctc`, `ctc`, etc.) so you
  only load what was available at that revision, not arbitrary future commits. For full
  supply-chain protection, pin a specific commit hash via `model_path` pointing to a
  local copy.

**Who this affects:** Only users who download or use GigaAM models. All other backends
(faster-whisper, Breeze, Parakeet) do **not** use `trust_remote_code=True`.

VoxFusion logs a warning every time a model is loaded with `trust_remote_code=True` so
the action is always visible in the application log.

## License

GPLv2. All contributions and derivative works must remain open-source under the same license.

### Third-party dependency licenses

VoxFusion uses the following third-party packages. All are GPLv2-compatible.

**Core dependencies:**

| Package | Version | License | Purpose |
|---------|---------|---------|---------|
| faster-whisper | >= 1.0 | Apache 2.0 | ASR engine (CTranslate2-backed Whisper) |
| ctranslate2 | >= 4.0 | MIT | ASR model runtime |
| numpy | >= 1.24 | BSD | Audio array operations |
| scipy | >= 1.7 | BSD | Signal processing |
| pydantic | >= 2.0 | MIT | Configuration validation |
| pydantic-settings | >= 2.0 | MIT | Environment variable config |
| click | >= 8.0 | BSD | CLI framework |
| sounddevice | >= 0.4 | MIT | Cross-platform audio I/O |
| soundfile | >= 0.12 | BSD | Audio file reading/writing |
| structlog | >= 23.0 | Apache 2.0 / MIT | Structured logging |
| pyyaml | >= 6.0 | MIT | YAML config parsing |
| httpx | >= 0.24 | BSD | Async HTTP client |
| tqdm | >= 4.66 | MIT / MPL 2.0 | Progress bars |
| typing-extensions | >= 4.13 | PSF | Backported typing features |
| torch | >= 2.4 | BSD | ML model runtime |
| transformers | >= 4.57 | Apache 2.0 | HuggingFace model loading |
| pyannote-audio | >= 4.0 | MIT | ML-based speaker diarization |
| torchcodec | >= 0.7 | BSD | Video/audio decoding for pyannote |

**Optional dependencies:**

| Package | Version | License | Purpose | Install extra |
|---------|---------|---------|---------|---------------|
| argostranslate | >= 1.9 | MIT | Offline NMT translation | `translation-offline` |
| noisereduce | >= 0.4 | MIT | Audio noise reduction | `noise-reduction` |
| librosa | >= 0.10 | ISC | Audio analysis utilities | `audio-quality` |
| soxr | >= 0.3 | LGPL 2.1 | High-quality resampling | `audio-quality` |
| cryptography | >= 41.0 | Apache 2.0 / BSD | Output encryption | `security` |
| tkinterdnd2 | >= 0.3 | MIT | GUI drag-and-drop | `dnd` |
| nemo_toolkit | >= 2.0 | Apache 2.0 | Parakeet ASR backend | `parakeet` |
| funasr | >= 1.0 | MIT | Chinese ASR (Paraformer) | `chinese` |
| torchaudio | >= 2.4 | BSD | Audio transforms for GigaAM | `gigaam` |
| sentencepiece | >= 0.1.99 | Apache 2.0 | Tokenization for GigaAM | `gigaam` |
| omegaconf | >= 2.3 | MIT | Config framework for GigaAM | `gigaam` |
| hydra-core | >= 1.3 | MIT | Config framework for GigaAM | `gigaam` |

**Platform-specific dependencies:**

| Package | Version | License | Platform |
|---------|---------|---------|----------|
| pyaudiowpatch | >= 0.2.12 | MIT | Windows (WASAPI loopback) |
| pulsectl | >= 23.5 | MIT | Linux (PulseAudio/PipeWire) |
| pyobjc-framework-CoreAudio | >= 10.0 | MIT | macOS (audio devices) |
| pyobjc-framework-ScreenCaptureKit | >= 10.0 | MIT | macOS (screen audio capture) |

**Development dependencies:**

| Package | Version | License |
|---------|---------|---------|
| pytest | >= 7.0 | MIT |
| pytest-asyncio | >= 0.21 | Apache 2.0 |
| pytest-cov | >= 4.0 | MIT |
| ruff | >= 0.1 | MIT |
| mypy | >= 1.5 | MIT |
| pip-audit | >= 2.6 | Apache 2.0 |
| pre-commit | >= 3.0 | MIT |
| pyinstaller | >= 6.15 | GPL 2.0+ |
