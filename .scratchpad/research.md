2026-03-16

# Research: audio-only recording mode

## User request

Add a mode that records audio without real-time transcription, and add a dedicated GUI button for plain audio recording.

## Current state

- CLI live capture is tightly coupled to ASR.
  - `src/voxfusion/cli/capture_cmd.py` documents `voxfusion capture` as "Capture live audio and transcribe in real-time."
  - The command builds `StreamingPipeline`, preloads ASR, prints live segments, and only saves transcription output.
- Streaming pipeline is ASR-centric.
  - `src/voxfusion/pipeline/streaming.py` feeds `audio_source.stream(...)` directly into preprocessing, ASR, diarization, translation, and output callbacks.
  - There is no branch for "capture raw audio only".
- GUI live capture is also tightly coupled to ASR.
  - `src/voxfusion/gui/main.py` uses `CaptureWorker`, which configures ASR, diarization, translation, `StreamingPipeline`, and WASAPI capture.
  - Existing GUI controls expose only Start/Stop for live transcription and separate file transcription.
- There is already a standalone recording script for microphone-only WAV capture.
  - `record_audio.py` records microphone audio to a WAV file and then suggests running `voxfusion transcribe <file>`.
  - This is not integrated into the package CLI or GUI.

## Architecture implications

- Reusing `StreamingPipeline` for raw recording is the wrong abstraction because it assumes ASR in the middle.
- A new recording path should stay separate from the transcription pipeline:
  - shared capture source creation can be reused,
  - output should be WAV-oriented,
  - no ASR/diarization/translation dependencies should load in recording mode.

## Likely implementation shape

- Introduce a reusable recorder component that:
  - consumes `AudioCaptureSource.stream(...)`,
  - accumulates PCM chunks,
  - writes WAV via `soundfile`,
  - supports `microphone`, `system`, and likely `both`.
- Add a CLI command or command mode for audio-only recording.
  - Extending `voxfusion capture` with a `--record-only` flag is possible, but mixes two different workflows in one command.
  - A dedicated `voxfusion record` command is cleaner and closer to existing `transcribe` separation.
- Add a GUI worker and a dedicated button for recording audio.
  - Reuse current source/device selection UI where possible.
  - Status should show recording state and saved path.

## Open design decisions

- For source `both`:
  - simplest option: write a mixed mono/stereo WAV using `AudioMixer`;
  - alternative: save separate files per source, which is more accurate but heavier UX.
- For non-Windows platforms:
  - current GUI live capture already warns that live capture requires Windows WASAPI;
  - recorder should match actual capture implementation availability, not promise unsupported platforms.
- File format:
  - WAV is the simplest safe default because the repo already uses WAV for ad hoc recording.

## Recommended approach

- Add a dedicated `voxfusion record` command for raw audio capture.
- Add a small recorder service/module independent from `StreamingPipeline`.
- Add a GUI `Record Audio` button plus `Stop Recording`, reusing existing source/device selectors and saving to WAV.
- Keep v1 scope to WAV output, one output file, with `both` recorded through the existing mixer path.

## Additional UX request: guided GUI flow

The user now wants a clear GUI flow for:

1. record audio,
2. transcribe the recorded audio,
3. send the resulting transcript to an LLM via Open WebUI API.

### Current GUI gap

- Step 1 is on the live tab.
- Step 2 and step 3 are on the file tab.
- After recording finishes, there is no explicit transition to:
  - populate the recorded file into the file transcription area,
  - switch the user to the next step,
  - expose the transcript-to-LLM action as the natural next action.

### Practical direction

- Do not invent a third separate workflow engine.
- Reuse the existing file transcription tab and LLM panel.
- Add lightweight orchestration state in GUI:
  - remember the last recorded file path,
  - after recording completes, offer/use “Transcribe recorded audio” directly,
  - after transcription completes, make “Send to LLM” the obvious next action.

### UX shape that best fits the current app

- Keep two tabs, but make the file tab the hub for the post-recording flow.
- In the live tab:
  - after recording, show a clear status with the saved path,
  - add a “Transcribe Last Recording” action or automatically move to file tab and preload the path.
- In the file tab:
  - add a simple step header/status:
    - Step 1: choose or use recorded file
    - Step 2: transcribe
    - Step 3: send transcript to LLM
  - rename/adjust the LLM panel text from “Summarize” to more generic transcript processing wording if needed.

### Constraint

- This is a UX refactor/extension, not an architecture rewrite.
- Existing Open WebUI API integration should remain the transport layer.

## User-reported issues after trial run

### Confirmed problems

- GUI transcription completion is not explicit enough.
  - The file transcription worker completes and fills the table, but there is no strong completion affordance.
  - No transcript file is auto-created; saving remains a separate manual action.
- The current flow has redundant controls.
  - `Use Last Recording` is redundant if the path is already auto-populated.
  - `Transcribe Last Recording` is redundant if `Transcribe` always operates on the currently selected/preloaded file.
- Open WebUI UX is incomplete.
  - URL/model/API key are not persisted.
  - GUI does not fetch the available model list from the configured Open WebUI instance.
  - There is no prompt editor/viewer, even though prompt templates exist in `src/voxfusion/llm/prompts.py`.
- Runtime diagnostics from the user log show an additional clarity issue.
  - Recording with `source=both` can degrade to a single active source when loopback startup fails.
  - GUI currently does not surface that fallback clearly enough in the post-recording flow.

### Important product decision surfaced by the bug report

- The GUI should treat transcription output as a first-class artifact, not only as rows in a table.
- At minimum, GUI should:
  - show a strong completion state,
  - make it obvious where the transcript lives,
  - allow immediate next-step processing by LLM.
