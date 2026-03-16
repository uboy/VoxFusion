2026-03-16

# Plan: audio-only recording mode

## Scope

Implement a raw audio recording workflow without live transcription in CLI and GUI, using the existing capture stack where possible.

## Proposed changes

1. Add a reusable recording module.
   - Create a recorder abstraction under `src/voxfusion/recording/` or similar.
   - Input: started `AudioCaptureSource`.
   - Output: WAV file written from streamed chunks.
   - Support duration limit and graceful stop.

2. Add CLI entry point for recording.
   - Preferred: new `voxfusion record` command.
   - Options: `--source`, `--device`, `--duration`, `--output`.
   - Use the recorder module and skip all ASR initialization.

3. Add GUI recording worker and controls.
   - New background worker separate from `CaptureWorker`.
   - Dedicated button for audio-only recording.
   - Reuse current source/device selectors.
   - Save to a timestamped WAV path unless user picks another path.

4. Add/extend tests.
   - Unit test recorder behavior with a fake capture source.
   - CLI-level test coverage for record command option handling if practical.
   - Keep tests focused on orchestration and file output, not hardware access.

5. Update user docs.
   - README and/or Russian quick-start/streaming docs with the new recording workflow.

6. Verification.
   - Run targeted pytest for new/affected tests.
   - Run any lightweight command-level verification that does not require real hardware.

## Constraints

- No architecture docs will be changed without explicit approval.
- Recording mode must not instantiate ASR/translation objects.
- Do not regress existing `capture` live transcription behavior.

## Risks

- `both` source behavior depends on current mixer semantics; mixed output may not preserve per-source separation.
- GUI state management needs careful separation so recording and live transcription cannot run simultaneously.
- Hardware-dependent behavior cannot be fully validated in CI-like unit tests.

## CC requested

Please confirm this plan, or tell me if you want one of these changes before implementation:

- `voxfusion record` as a separate command, or a `--record-only` flag inside `voxfusion capture`
- for source `both`: one mixed WAV file, or separate files per source

---

# Revised Plan: guided GUI flow for record -> transcribe -> LLM

## Scope

Refine GUI so audio recording, transcription of the recorded file, and Open WebUI processing feel like one continuous user flow.

## Proposed GUI changes

1. Track recording outputs in GUI state.
   - Store `last_recorded_file`.
   - On recording completion, preload that path into the file transcription controls.

2. Add explicit next-step actions.
   - In the live tab, after recording completes:
     - show a clear success status,
     - add/use a “Transcribe Last Recording” action.
   - Optionally switch to the file tab automatically after recording.

3. Make the file tab read as a workflow.
   - Add short step labels/status text:
     - choose file / use last recording,
     - transcribe,
     - process transcript with LLM.
   - Add a dedicated button for LLM processing that is phrased as the next action after transcription, not a separate advanced feature.

4. Tighten the handoff between transcription and LLM.
   - Enable/populate LLM action based on transcript presence.
   - Improve empty-state and completion messaging.

5. Verification.
   - Add focused tests for GUI state transitions where practical.
   - Keep hardware-dependent behavior out of automated tests.

## Recommended UX decisions

- After recording finishes:
  - automatically populate the file path,
  - automatically switch to the file tab,
  - do not auto-start transcription without user action.
- In the file tab:
  - keep manual “Browse...” for arbitrary files,
  - add a clear CTA for “Transcribe recorded audio” when a recent recording exists.
- For the Open WebUI action:
  - keep existing API behavior,
  - rename the button/section to something more generic than “Summarize” if the goal is broader transcript processing.

## CC requested

Please confirm this revised GUI UX direction with `CC`, and I’ll implement it.

---

# Revised Plan v2: fix guided flow issues from real user run

## Scope

Fix the concrete UX/behavior issues reported after running the current flow.

## Proposed changes

1. Remove redundant file-flow buttons.
   - Remove `Use Last Recording`.
   - Remove `Transcribe Last Recording`.
   - Keep a single `Transcribe` button that always works on the currently populated file path.

2. Make transcription completion explicit.
   - Add a clear success state after file transcription completes.
   - Show whether transcript exists only in-memory or has been saved.
   - Strongly guide the user to either `Save Transcript` or `Send to LLM`.

3. Treat transcript output as an artifact.
   - Auto-generate/store transcript text after transcription in GUI state.
   - Optionally auto-save to a default `.txt` file next to the recorded audio, or at least expose the path if auto-save is enabled.
   - At minimum, remove ambiguity about “where the transcript is”.

4. Persist Open WebUI settings.
   - Save URL, API key, selected model, and prompt choice/custom prompt in local GUI settings.
   - Restore them on next launch.

5. Load available Open WebUI models.
   - Add a `Refresh Models` action.
   - Load models from the configured Open WebUI instance and populate a combobox.
   - Fall back cleanly to manual model entry if model listing fails.

6. Add prompt UX.
   - Show the current prompt choice.
   - Allow viewing/editing the user prompt template used for transcript processing.
   - Make it clear what prompt is being sent.

7. Surface capture fallback more clearly.
   - If `both` degrades to one source, reflect that in GUI status/logs in a user-facing way.

## Recommended defaults

- Auto-populate recorded file path, but keep `Transcribe` as the only transcription CTA.
- Persist Open WebUI settings to a lightweight local file under the project/app data directory.
- Provide at least:
  - one built-in prompt selector,
  - one editable custom prompt text box,
  - a read-only default prompt preview or reset button.

## CC requested

If this matches what you want, send `CC` and I will implement this revised GUI flow.
