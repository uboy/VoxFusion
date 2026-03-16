# VoxFusion
VoxFusion captures any system audio — including mic input, calls, music, browser or app playback — and turns it into high-quality speech transcriptions with speaker diarization and automatic translation. Built for Windows, macOS and Linux, VoxFusion simplifies capturing, understanding and translating conversations in real time.

## Run modes

- CLI: `python cli_start.py --help` or `python -m voxfusion --help`
- GUI: `python gui_start.py`

## Common commands

- Live transcription: `python -m voxfusion capture`
- Raw audio recording to WAV: `python -m voxfusion record --source microphone --output recording.wav`
- Batch transcription from file: `python -m voxfusion transcribe recording.wav`

The GUI live tab now also includes a separate `Record Audio` button for plain WAV capture without real-time transcription.

## Build binaries

- `python scripts/build_binaries.py --target all`

See `docs/BINARY_BUILD.md` for packaging details and Windows notes.

## Docs

Project documentation is in `docs/` (`docs/README.md`).
