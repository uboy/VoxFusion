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

---

# Research: borrow plan from Handy

## User request

Build a plan for what VoxFusion can borrow from `C:\Users\devl\proj\Handy`, with specific interest in:

- `GigaAM v3`,
- language selection with model-aware mapping,
- visualization of model speed/accuracy,
- a cleaner, more polished settings UI.

## What Handy does relevant to this request

- `Handy` has a typed model registry in Rust (`src-tauri/src/managers/model.rs`).
  - Each model declares:
    - `engine_type`,
    - `accuracy_score`,
    - `speed_score`,
    - `supported_languages`,
    - `supports_language_selection`,
    - `supports_translation`,
    - `is_recommended`.
- `Handy` already includes `GigaAM v3`.
  - Model id: `gigaam-v3-e2e-ctc`
  - Engine: `EngineType::GigaAM`
  - Filename: `giga-am-v3.int8.onnx`
  - Declared support: Russian only (`["ru"]`)
  - UI score hints: `accuracy_score=0.85`, `speed_score=0.75`
- `Handy` enforces model-aware language behavior.
  - The global language list lives in `src/lib/constants/languages.ts`.
  - The active model advertises `supported_languages`.
  - `LanguageSelector.tsx` filters available languages against the current model.
  - `commands/models.rs` resets `selected_language` to `auto` if the newly selected model does not support the previously chosen language.
- `Handy` surfaces model tradeoffs in the UI.
  - `ModelCard.tsx` renders compact bar visualizations for `accuracy_score` and `speed_score`.
  - The same card also shows capability badges for multilingual support and translation.
- `Handy` settings UX is modular and more polished than current VoxFusion GUI.
  - General settings are grouped with reusable `SettingsGroup` / `SettingContainer`.
  - Model-specific settings are conditional (`ModelSettingsCard.tsx`).
  - Post-processing settings include provider selection, persisted values, model refresh, and prompt management.

## What is realistically portable to VoxFusion

### Good candidates to borrow conceptually

- Model registry metadata shape.
  - VoxFusion can benefit from first-class metadata per ASR backend/model:
    - display name,
    - engine/backend family,
    - supported languages,
    - translation support,
    - recommended/default flag,
    - estimated speed/accuracy score.
- Language-to-model compatibility rules.
  - This is directly useful in VoxFusion and not tied to Tauri/Rust.
  - The important behavior is not just filtering the dropdown; it is preventing invalid model/language combinations.
- Speed/accuracy visualization.
  - Lightweight bars or badges are easy to reproduce in Tkinter.
  - They give users a fast heuristic for choosing a model without reading long descriptions.
- Settings information architecture.
  - Separate sections for:
    - audio/capture,
    - transcription/model,
    - transcript processing / LLM,
    - advanced/debug.
  - Within the transcription section, model-aware sub-settings should appear only when relevant.
- Post-processing UX patterns.
  - Persist provider settings.
  - Fetch model list from provider.
  - Make prompt selection and editing explicit.

### Borrowable model/backend direction

- `GigaAM v3` is a strong candidate for VoxFusion.
  - It fits the user's explicit preference.
  - It also gives VoxFusion a Russian-specialized option beyond generic Whisper.
- The `Handy` integration path itself is not directly portable.
  - Handy uses `transcribe-rs` from Rust.
  - VoxFusion is currently Python-based.
  - So the transport/inference layer must be re-integrated in Python rather than copied from Handy.

## What should not be copied directly

- Tauri/React component architecture as implementation.
  - VoxFusion currently uses Python/Tkinter; a direct UI code port is not realistic.
- Rust audio/transcription runtime.
  - Handy's `cpal` / Rust manager layer is a different concurrency and packaging model.
- Hardcoded score values without provenance.
  - Handy's speed/accuracy scores are UX heuristics, not benchmark guarantees.
  - VoxFusion can adopt the pattern, but should label them as estimated or curated.

## Implications for VoxFusion

### GigaAM v3

- Recommended as a new ASR backend option if a Python integration path is feasible.
- Product positioning inside VoxFusion:
  - "Best for Russian"
  - batch transcription first,
  - only later consider live/streaming if the backend supports it well enough.

### Language mapping

- VoxFusion should move from a flat language selector to:
  - canonical language list,
  - per-model `supported_languages`,
  - per-model `supports_language_selection`,
  - automatic fallback to `auto` when the model changes incompatibly.
- This also improves Open WebUI prompting because downstream processing can know what language the transcript is expected to be in.

### Visual model selection

- VoxFusion should expose model cards or at least a richer selector with:
  - model name,
  - short description,
  - languages summary,
  - speed bar,
  - accuracy bar,
  - recommended badge.
- This is most valuable in GUI settings and onboarding-like first-run flows.

### Settings UI direction

- Current GUI can be made much clearer by adopting Handy-like grouping, but in Tkinter terms:
  - left navigation or tab/subtab grouping,
  - consistent labeled rows,
  - model-specific expandable section,
  - persistent LLM/provider settings,
  - clearer visual hierarchy.

## Recommended borrowing order

1. Model metadata and language compatibility rules.
   - Highest leverage, low UI risk.
2. Settings UI cleanup around transcription and LLM.
   - Improves usability without forcing a full redesign.
3. Speed/accuracy visualization.
   - Good UX win once model metadata exists.
4. `GigaAM v3` backend integration.
   - High product value, but technically the riskiest because it requires a new inference integration path in Python.
