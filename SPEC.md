# VoxFusion Runtime Spec

## Live GigaAM v1

### Summary

VoxFusion supports a quality-first live mode for `gigaam-v3-e2e-ctc`.

This mode is intentionally different from the existing faster-whisper live path:
- live output is draft text, not token-stable final text;
- every captured frame is durably spooled to the session directory;
- utterances are transcribed in warm worker processes;
- draft backlog is bounded and may defer new draft utterances under overload;
- when capture stops, VoxFusion reuses successful draft utterances by default and finalizes deferred or failed utterances from the spooled session audio before replacing the draft transcript with the final one.

### User-facing behavior

#### GUI

- The `Live` tab allows selecting `GigaAM v3`.
- Starting live GigaAM shows draft transcript rows while capture is active.
- When capture stops, the GUI replaces the draft table with the finalized transcript; by default only deferred or failed utterances are reprocessed on stop.
- Live GigaAM does not support translation.

#### CLI

- `voxfusion capture --model gigaam-v3-e2e-ctc` is supported on Windows live capture.
- The CLI prints draft lines during capture.
- If saving is enabled, the saved artifact contains the finalized transcript, not the draft transcript.
- `--translate` is rejected for live GigaAM.

### Functional requirements

- `LG-1`: Live GigaAM must preserve raw session audio durably for the entire session.
- `LG-2`: Draft utterances may complete out of order internally, but committed transcript output must be deterministic.
- `LG-3`: The live runtime must use warm worker processes rather than loading the model per utterance.
- `LG-4`: Stop-time finalization must recover deferred or failed utterances from the spooled session audio, and may be configured to rebuild the whole transcript.
- `LG-5`: If stop-time finalization fails for an utterance, VoxFusion must fall back to the corresponding draft text instead of silently dropping it.
- `LG-6`: GUI live timestamps must be anchored to the actual capture start, not to pre-warmup/model-load time.
- `LG-7`: The CLI and GUI must agree that live GigaAM is supported and translation is not.
- `LG-8`: If draft ASR falls behind capture throughput, VoxFusion must defer new draft utterances in a controlled way and recover them during stop-time finalization instead of letting the draft backlog grow without bound.

### Non-goals for v1

- True token streaming from GigaAM.
- Live ML diarization in the hot path.
- Background mid-session refinement beyond stop-time finalization.
