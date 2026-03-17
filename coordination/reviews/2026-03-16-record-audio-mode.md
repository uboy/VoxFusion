2026-03-16

# Review Report: audio-only recording mode

## Scope

- Added raw audio recording path independent of ASR.
- Added CLI command `voxfusion record`.
- Added GUI `Record Audio` action in the live tab.
- Added GUI `Pause` / `Resume` for raw recording.
- Added guided GUI flow from recording to file transcription to Open WebUI processing.
- Added transcript auto-save, Open WebUI settings persistence, model loading, and prompt editing UX.
- Added unit tests and README usage notes.

## Findings

- No blocking issues found during local review of the changed files.
- Residual risk: raw recording for `source=both` is mixed into a single WAV by timestamp alignment and averaging; this is intentional for v1, but it does not preserve separate channels/sources in the saved file.
- Residual risk: hardware-backed capture flows were not exercised end-to-end in this environment; validation is limited to unit tests and wiring-level checks.
- Residual risk: the guided flow is stateful inside the GUI process; it improves UX for the latest recording, but it is not yet a full persisted session workflow.

## Verification

- `.\venv\Scripts\python.exe -m pytest tests\unit\test_recording.py tests\unit\test_capture_factory.py`
  - Result: pass (`8 passed`)
- `.\venv\Scripts\python.exe -m pytest tests\unit\test_config.py -q`
  - Result: pass (`13 passed`)
- `.\venv\Scripts\python.exe -m pytest tests\unit\test_recording.py -q`
  - Result: pass (`4 passed`)
- `.\venv\Scripts\python.exe -m pytest tests\unit\test_gui_flow.py tests\unit\test_recording.py -q`
  - Result: pass (`7 passed`)
- `.\venv\Scripts\python.exe -m pytest tests\unit\test_gui_flow.py tests\unit\test_llm_client.py tests\unit\test_recording.py tests\unit\test_capture_factory.py -q`
  - Result: pass (`17 passed`)
- `git diff --check -- src/voxfusion/cli/main.py src/voxfusion/cli/record_cmd.py src/voxfusion/gui/main.py src/voxfusion/recording README.md tests/unit/test_recording.py coordination/tasks.jsonl coordination/state/codex.md .scratchpad/research.md .scratchpad/plan.md`
  - Result: pass

## Notes

- Review report validation script was not present under `scripts/validate-review-report.*`, so no script-based validation could be run.
- Follow-up UX/refactor batch completed:
  - GUI helpers/workers/theme/model summary were extracted into dedicated modules.
  - `src/voxfusion/gui/main.py` was reduced substantially and now focuses on composition rather than worker/runtime implementation.
  - catalog-driven model overview cards and grouped section layout were added to improve usability.
  - broad regression verification was run:
    - `.\venv\Scripts\python.exe -m pytest tests\unit tests\integration\test_gui_smoke.py -q`
      - Result: pass (`165 passed`)
    - `.\venv\Scripts\python.exe -m voxfusion.gui.main --help`
      - Result: pass (help output shown; non-blocking `runpy` warning observed)
- GigaAM backend follow-up:
  - added a batch-oriented ONNX/CTC backend and factory routing for `gigaam-v3-e2e-ctc`
  - live capture now rejects the model explicitly because only file transcription is supported for this backend
  - verification:
    - `.\venv\Scripts\python.exe -m pytest tests\unit tests\integration\test_gui_smoke.py -q`
      - Result: pass (`171 passed`)
  - residual risk:
    - no real local GigaAM model/tokenizer artifacts were available for a true end-to-end transcription run in this environment
