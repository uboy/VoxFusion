# FFmpeg Bundling v1

## Goal

Make VoxFusion usable with video/compressed-audio inputs even when `ffmpeg` is not present in `PATH`.

## Design

- Build-time:
  - `scripts/build_binaries.py` first tries `PATH`.
  - On Windows, if `PATH` does not provide FFmpeg, the build downloads a portable essentials ZIP from `gyan.dev`, extracts `ffmpeg.exe` and `ffprobe.exe`, and bundles them into the app root.
- Runtime:
  - VoxFusion resolves FFmpeg from:
    - explicit `VOXFUSION_FFMPEG_PATH` / `VOXFUSION_FFPROBE_PATH`
    - next to the executable
    - `~/.voxfusion/ffmpeg/`
    - `PATH`
  - The GUI installer writes a local portable FFmpeg copy into `~/.voxfusion/ffmpeg/` instead of delegating to `winget`.
- Extraction/encoding:
  - file extraction and MP3 encoding use the shared runtime FFmpeg resolver so local managed copies work for source runs too.

## Risks

- Windows auto-download depends on the external archive layout from `gyan.dev`.
- Linux/macOS still rely on system package managers for now.
- Network-restricted environments still need either PATH-provided FFmpeg or a pre-bundled binary.
