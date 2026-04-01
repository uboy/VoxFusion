# Binary Build and Packaging

VoxFusion produces two binaries:

- `voxfusion-gui` for graphical mode.
- `voxfusion-cli` for command-line mode.

Build script: `scripts/build_binaries.py`

## Why `--onedir`

For all platforms we use PyInstaller `--onedir`.
For Windows this is mandatory to reduce Smart App Control blocking risk and keep DLLs next to `.exe`.

## Build commands

```bash
python scripts/build_binaries.py --target all
python scripts/build_binaries.py --target gui
python scripts/build_binaries.py --target cli
python scripts/build_binaries.py --target all --backends all
```

Output:

- Bundle directories: `dist/binaries/voxfusion-gui`, `dist/binaries/voxfusion-cli`
- ZIP archives: `dist/binaries/voxfusion-gui-<platform>.zip`, `dist/binaries/voxfusion-cli-<platform>.zip`

## Windows specifics

1. Build stays in `--onedir` mode (no single-file self-extracting EXE).
2. A PyInstaller `version-file` is generated with current classes:
   - `FixedFileInfo`
   - `StringStruct`
   - `VarStruct`
3. Users should unblock the downloaded ZIP in file properties before extraction.

## Included data

Build script explicitly adds:

- `src/voxfusion/config/defaults.yaml`
- `customtkinter/assets/themes` (if `customtkinter` is installed)
- `pyannote/audio/telemetry/config.yaml` when `pyannote.audio` is installed
- `ffmpeg(.exe)` and `ffprobe(.exe)` inside the Windows bundle `_internal/` directory
  - from `PATH` when available
  - on Windows, automatically downloaded as a portable build when `PATH` does not provide FFmpeg

## FFmpeg behavior

- Built Windows bundles are expected to work without a system-wide FFmpeg install.
- If the build machine does not have FFmpeg in `PATH`, `scripts/build_binaries.py` downloads a portable Windows FFmpeg build, extracts `ffmpeg.exe` and `ffprobe.exe`, and bundles them into `_internal/`.
- GUI speaker detection uses the same ffmpeg-backed extraction path for container media such as `webm`, `mp4`, and compressed-audio inputs, so "Detect" and full transcription support the same file classes.
- Breeze ASR no longer depends on `torchcodec` for normal file transcription, so bundled FFmpeg is primarily required for media extraction and GigaAM's external decode path.
- At runtime VoxFusion looks for FFmpeg in this order:
  - next to the executable
  - under the PyInstaller `_internal/` directory
  - in the app-managed local directory `~/.voxfusion/ffmpeg/`
  - in `PATH`

## Validated command

With models already cached locally, the full verification command used for a stable Windows validation pass was:

```powershell
$env:PYTHONPATH='src'
$env:HF_HUB_OFFLINE='1'
$env:VOXFUSION_FFMPEG_DIR='build\vendor\ffmpeg-runtime'
.\venv\Scripts\python.exe -m pytest -q
```

## Launchers

- GUI launcher: `gui_start.py`
- CLI launcher: `cli_start.py`

Both launchers use absolute package imports only.
