# Live GigaAM v1 Design

## Status

Implemented on feature branch `feature/live-gigaam-design-and-live-fix`.

## Problem

GigaAM gives the best current Russian quality in VoxFusion, but its engine is batch-oriented and does not expose a true streaming decoder state. A naive `N`-second round-robin chunk farm would maximize throughput on paper while damaging transcript quality at utterance boundaries and making ordering/backlog behavior unstable.

## Decision

Live GigaAM v1 uses:
- VAD-bounded utterances instead of rigid time slicing;
- a small warm process pool with least-busy dispatch;
- deterministic ordered transcript commit;
- bounded draft backlog with overload deferral;
- durable per-source session spooling;
- selective stop-time finalization from the spooled audio;
- GUI draft transcript replacement once finalization finishes.

## Pipeline

### 1. Capture and spool

- Live capture uses the existing Windows capture stack.
- Each source is wrapped by `SpoolingCaptureSource`.
- `SessionAudioSpool` normalizes to mono `16 kHz`, preserves the timeline, zero-fills gaps, and stores per-source WAV files under `data_dir/live_gigaam/session_*`.

### 2. Utterance boundaries

- `VadChunker` produces pause-bounded utterances with a hard duration cap.
- Each utterance becomes one `LiveGigaAMJob` with `seq_id`, `source`, time bounds, audio samples, and finalize flag.

### 3. Draft ASR dispatch

- `LiveASRDispatcher` owns a warm `ProcessPoolExecutor` slot per worker.
- Every worker loads one persistent `GigaAMCTCEngine`.
- Dispatch uses least-loaded worker selection, not strict round-robin.
- Draft jobs are allowed to finish out of order.
- Draft backlog is bounded.
- If the draft backlog reaches the hard limit, new utterances skip draft ASR and are deferred directly to stop-time finalization.
- Deferred draft placeholders keep transcript ordering deterministic without pretending the missing draft text exists.

### 4. Ordered commit

- `OrderedTranscriptCommitter` buffers results by `seq_id`.
- Transcript output is committed only when all prior sequence ids are resolved.
- Text overlap trimming is applied per source so microphone/system tails do not bleed into each other.

### 5. Stop-time finalization

- When capture stops, VoxFusion waits for all draft jobs and drains the draft collector.
- By default it reuses successful draft utterances and rereads only deferred or failed utterances from the session spool for stop-time recovery.
- Full second-pass re-finalization remains available through config when maximum stop-time quality matters more than stop latency.
- Utterances that were deferred during overload are recovered only at this stage.
- Finalization context is bounded inside the same-source silence gaps:
  - left context does not cross the previous utterance end;
  - right context does not cross the next utterance start.
- If finalization fails for an utterance, the corresponding draft text is preserved as fallback.

### 6. GUI/CLI semantics

- GUI receives draft segments via `on_segment`.
- After finalization, GUI receives a complete replacement transcript via `on_replace_segments`.
- CLI prints draft lines during capture and saves finalized segments at the end.

## Why this is better than strict round-robin fixed chunks

- It avoids splitting words at arbitrary `N`-second boundaries.
- It handles uneven chunk cost better than modulo scheduling.
- It preserves every captured frame for offline-quality recovery.
- It makes “draft now, recover what still needs stop-time work” explicit instead of pretending GigaAM is a token-streaming model.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Worker memory duplication | Default to a small worker count and bound threads per worker |
| Out-of-order worker completion | Ordered commit by `seq_id` |
| Unbounded draft backlog under ASR slowdown | Defer new draft utterances once backlog reaches the hard limit; recover them during finalization |
| Finalization dropping utterances | Draft fallback on finalize errors |
| Timestamp drift | GUI timestamps anchor on actual capture start callback |
| Context bleed across adjacent speech | Finalize context bounded inside same-source silence gaps |

## Verification focus

- spool durability and zero-padding
- ordered commit and overlap trimming
- least-loaded dispatch and retry behavior
- deferred-draft overload behavior and backlog telemetry
- stop-time finalization and fallback
- GUI draft replacement
- CLI finalized save behavior
