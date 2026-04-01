# Diarization Optimization — Feature Design

**Date**: 2026-03-30
**Status**: Draft
**Scope**: Performance optimization of diarization pipeline + GUI UX improvements

---

## Problem Statement

The pyannote.audio diarization pipeline is the dominant bottleneck in batch file transcription.
Even with the fast GigaAM ASR engine (~4x realtime on CPU), the overall pipeline is slow
because pyannote processes the **entire** audio through 3 neural networks sequentially:

1. **Segmentation** (PyanNet) — sliding-window NN across entire audio
2. **Speaker Embedding** (WeSpeaker/ResNet) — embedding extraction per segment
3. **Agglomerative Clustering** — speaker grouping

On CPU, a 10-minute file can take 3-8 minutes just for diarization.
On GPU, the same file takes 15-30 seconds.

---

## Feature 1: "None" Diarization Strategy

### Motivation

There is currently no way to skip diarization. The available strategies are:
`auto`, `channel`, `ml`, `hybrid`. Even `channel` still runs alignment code.
For single-speaker recordings or when diarization isn't needed, this is wasted work.

### Design

#### 1.1 New `NoneDiarizer` class

**File**: `src/voxfusion/diarization/none.py`

```python
class NoneDiarizer:
    """Assigns all segments to SPEAKER_00 without any analysis."""

    async def diarize(
        self,
        segments: list[TranscriptionSegment],
        audio: AudioChunk | None = None,
    ) -> list[DiarizedSegment]:
        return [
            DiarizedSegment(
                segment=seg,
                speaker_id="SPEAKER_00",
                speaker_source="none",
            )
            for seg in segments
        ]

    async def diarize_stream(
        self,
        segment_stream: AsyncIterator[tuple[TranscriptionSegment, AudioChunk]],
    ) -> AsyncIterator[DiarizedSegment]:
        async for seg, _audio in segment_stream:
            yield DiarizedSegment(
                segment=seg,
                speaker_id="SPEAKER_00",
                speaker_source="none",
            )
```

Key properties:
- No neural networks loaded
- Zero additional latency
- No `diarize_turns` method — batch pipeline takes the standard ASR-first path

#### 1.2 Factory changes

**File**: `src/voxfusion/diarization/factory.py`

Add `"none"` to the allowed strategies set in `create_diarizer()`:

```python
if requested == "none":
    return _log_selection(
        ...,
        selection=DiarizerSelection(NoneDiarizer(), requested, "none"),
        ...
    )
```

#### 1.3 GUI changes

**File**: `src/voxfusion/gui/main.py`

Update `FILE_DIARIZATION_CHOICES`:
```python
FILE_DIARIZATION_CHOICES: tuple[str, ...] = ("auto", "none", "channel", "ml", "hybrid")
```

When `"none"` is selected, disable Min/Max speakers fields (already handled by
`_refresh_file_diarization_controls` — just add `"none"` to the disabled set).

#### 1.4 Batch pipeline changes

**File**: `src/voxfusion/pipeline/batch.py`

In `_diarization_path_decision()`, add early return:
```python
if self._resolved_diarization_strategy == "none":
    return False, "Diarization disabled (strategy=none)."
```

In `process_file()`, when `diarized is None` and resolved strategy is `"none"`,
skip the stage 4 diarization entirely — call `NoneDiarizer.diarize()` directly
(which is effectively a no-op wrapper).

#### 1.5 Progress reporting

- Stage 4 (DIARIZATION) emits `STAGE_STARTED` + `STAGE_COMPLETED` immediately
  with message: `"Speaker separation disabled — skipping diarization."`
- Total pipeline time improvement: eliminates 30-70% of wall time for
  pyannote-heavy workflows.

### Tests

| Test | File | What it verifies |
|------|------|-----------------|
| `test_none_diarizer_assigns_speaker_00` | `tests/unit/test_diarization.py` | All segments get `SPEAKER_00`, `speaker_source="none"` |
| `test_none_diarizer_stream` | `tests/unit/test_diarization.py` | Streaming yields correct segments |
| `test_factory_none_strategy` | `tests/unit/test_diarization_factory.py` | `create_diarizer(strategy="none")` returns `NoneDiarizer` |
| `test_batch_pipeline_none_skips_diarization` | `tests/unit/test_batch_pipeline.py` | No `diarize_turns` call, pipeline completes, progress events emitted |
| `test_gui_none_disables_speaker_fields` | `tests/unit/test_gui_flow.py` | Min/Max entries disabled when "none" selected |

---

## Feature 2: Speaker Count Selection + Auto-Detection + Tooltips

### Motivation

- If user specifies `num_speakers` (or `min_speakers == max_speakers`), pyannote
  **skips** agglomerative clustering, saving ~20-40% of diarization time.
- "Auto" speaker count is the current default — pyannote estimates it internally.
- Users don't understand what each option does — need tooltips.

### Design

#### 2.1 Speaker count presets in GUI

Replace raw Min/Max entry fields with a combined dropdown + optional entries:

```
Speaker Count: [Auto ▼]  Min: [__]  Max: [__]
```

Dropdown values:
- `"Auto"` — no min/max hints (pyannote estimates internally)
- `"1 speaker"` — sets min=1, max=1, **skips** ML diarization clustering entirely
- `"2 speakers"` — sets min=2, max=2
- `"3 speakers"` — sets min=3, max=3
- `"Custom"` — enables Min/Max entry fields for manual input

When a preset is selected (not "Custom"), the Min/Max fields display the
values but are disabled (greyed out). Only "Custom" enables them.

When "1 speaker" is selected AND strategy is "auto" or "ml", show a hint:
`"Tip: Use 'none' strategy to skip diarization entirely for single speaker."`

#### 2.2 Quick speaker count estimation (pre-diarization)

**File**: `src/voxfusion/diarization/speaker_counter.py` (new)

A lightweight approach to estimate speaker count BEFORE full diarization.
Uses only the **embedding** model from pyannote (not the full pipeline):

```python
async def estimate_speaker_count(
    audio: AudioChunk,
    *,
    max_sample_duration_s: float = 120.0,
    segment_duration_s: float = 3.0,
    hf_token: str | None = None,
) -> int:
    """Fast speaker count estimation using VAD + embedding clustering.

    Algorithm:
    1. Run Silero VAD to find speech segments (fast, CPU-only)
    2. Sample up to max_sample_duration_s of speech segments
    3. Extract speaker embeddings per segment (WeSpeaker, ~50ms/segment)
    4. Cluster embeddings with simple agglomerative clustering
    5. Return estimated cluster count

    Returns 0 if no speech detected.
    """
```

This is ~5-10x faster than full pyannote because:
- Skips the heavy PyanNet segmentation model
- Uses only sampled audio, not the entire file
- Silero VAD is extremely fast (~0.1x realtime on CPU)

However, this is an **optional optimization** — the "Auto" dropdown value
simply passes no hints to pyannote (current behavior). The estimation
can be surfaced as a "Detect Speakers" button next to the dropdown.

#### 2.3 "Detect Speakers" button

Adds a small button `[Detect]` next to the Speaker Count dropdown.
When clicked:
1. Reads the selected audio file
2. Runs `estimate_speaker_count()` in a background thread
3. Updates the dropdown to show detected count
4. Shows progress: `"Detecting speakers... (est. ~5s)"`

This is optional — user can always pick a preset manually.

#### 2.4 Tooltip system

**File**: `src/voxfusion/gui/tooltip.py` (new)

Tkinter has no built-in tooltips. We create a reusable `ToolTip` class:

```python
class ToolTip:
    """Hover tooltip for any Tkinter widget.

    Usage:
        ToolTip(widget, text="This option controls...")
        # or with a help icon:
        create_help_icon(parent, text="Explanation...")
    """

    SHOW_DELAY_MS = 400    # delay before showing
    HIDE_DELAY_MS = 100    # delay before hiding
    WRAP_LENGTH_PX = 300   # text wrap width

    def __init__(self, widget: tk.Widget, text: str) -> None:
        self._widget = widget
        self._text = text
        self._tip_window: tk.Toplevel | None = None
        self._show_id: str | None = None

        widget.bind("<Enter>", self._schedule_show)
        widget.bind("<Leave>", self._hide)
        widget.bind("<ButtonPress>", self._hide)

    def _schedule_show(self, event: tk.Event) -> None:
        self._show_id = self._widget.after(self.SHOW_DELAY_MS, self._show)

    def _show(self) -> None:
        if self._tip_window:
            return
        x = self._widget.winfo_rootx() + 20
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 4
        self._tip_window = tw = tk.Toplevel(self._widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=self._text,
            justify=tk.LEFT,
            background="#FFFFDD",
            foreground="#333333",
            relief=tk.SOLID,
            borderwidth=1,
            wraplength=self.WRAP_LENGTH_PX,
            font=("Segoe UI", 9),
            padx=6,
            pady=4,
        )
        label.pack()

    def _hide(self, event: tk.Event | None = None) -> None:
        if self._show_id:
            self._widget.after_cancel(self._show_id)
            self._show_id = None
        if self._tip_window:
            self._tip_window.destroy()
            self._tip_window = None


def create_help_icon(
    parent: tk.Widget,
    text: str,
    *,
    side: str = tk.LEFT,
    padx: int | tuple[int, int] = (2, 6),
) -> ttk.Label:
    """Create a small '?' label with a tooltip attached."""
    icon = ttk.Label(parent, text="?", foreground="#5599CC", cursor="question_arrow",
                     font=("Segoe UI", 8, "bold"))
    icon.pack(side=side, padx=padx)
    ToolTip(icon, text)
    return icon
```

#### 2.5 Tooltip texts for all diarization controls

| Widget | Tooltip text |
|--------|-------------|
| Speaker Separation combo | `"Controls how speakers are identified in the transcript.\n\nnone — Skip speaker detection (fastest, single speaker).\nchannel — Assign by audio source (mic vs system).\nml — Use AI model (pyannote.audio). Requires HF token.\nhybrid — Channel first, ML fallback.\nauto — Best available strategy."` |
| Speaker Count combo | `"How many speakers are in the recording?\n\nAuto — Let the AI estimate (slower).\n1-3 speakers — Fixed count (faster, skips estimation).\nCustom — Set Min/Max manually."` |
| Min speakers entry | `"Minimum expected speakers. Helps the AI avoid merging different speakers into one."` |
| Max speakers entry | `"Maximum expected speakers. Helps the AI avoid splitting one speaker into many."` |
| Detect button | `"Quickly estimate the number of speakers without running full diarization (~5-10 seconds)."` |
| Quality combo | `"Transcription quality preset.\n\nFast — Greedy decoding, aggressive VAD. Best for real-time.\nBalanced — Good accuracy/speed tradeoff.\nQuality — Wider beam search, best accuracy."` |
| Model combo | `"ASR model for transcription.\n\nGigaAM v3 — Best for Russian (fast).\nWhisper — Multilingual (small→large-v3).\nParakeet — Fast, 25 European languages."` |
| Language combo | `"Audio language. 'Auto Detect' works but specifying the language is faster and more accurate."` |
| Transcribe button | `"Start transcription of the selected file with current settings."` |
| Download button | `"Pre-download the selected model to avoid waiting during transcription."` |

Tooltips will also be added to **Live Capture tab** controls:
| Widget | Tooltip text |
|--------|-------------|
| Microphone combo | `"Select the input microphone device for recording your voice."` |
| System audio combo | `"Select the loopback device to capture system/app audio (calls, playback)."` |
| Start/Stop button | `"Start or stop live audio capture and real-time transcription."` |
| Translation combo | `"Translate transcription to selected language in real-time (requires Argos Translate)."` |

### Tests

| Test | File | What it verifies |
|------|------|-----------------|
| `test_tooltip_creation` | `tests/unit/test_tooltip.py` | ToolTip creates/destroys toplevel on enter/leave |
| `test_help_icon_creation` | `tests/unit/test_tooltip.py` | `create_help_icon()` returns label with tooltip |
| `test_speaker_count_presets` | `tests/unit/test_gui_flow.py` | Preset selection fills min/max correctly |
| `test_speaker_count_custom_enables_fields` | `tests/unit/test_gui_flow.py` | Only "Custom" enables entry fields |
| `test_estimate_speaker_count_single` | `tests/unit/test_speaker_counter.py` | Returns 1 for single-speaker synthetic audio |
| `test_estimate_speaker_count_empty` | `tests/unit/test_speaker_counter.py` | Returns 0 for silence |
| `test_detect_button_updates_dropdown` | `tests/unit/test_gui_flow.py` | GUI updates after detection completes |

---

## Feature 3: Parallel Transcription of Diarization Windows

### Motivation

In the diarization-first path (`_transcribe_diarized_windows`), GigaAM transcribes
each speaker window **sequentially**. For a 10-minute recording with 30 windows,
this means 30 serial ASR calls. GigaAM inference is CPU-bound and benefits from
parallelism when multiple cores are available.

### Design

#### 3.1 Parallel window transcription

**File**: `src/voxfusion/pipeline/batch.py`

Replace the sequential `for index, turn in enumerate(normalized_turns)` loop with
`asyncio.gather` over batches, where each ASR call runs in an executor:

```python
async def _transcribe_diarized_windows(
    self,
    full_audio: AudioChunk,
) -> list[DiarizedSegment] | None:
    # ... (diarize_turns call stays the same) ...

    normalized_turns = _normalize_turns(turns)
    total_windows = len(normalized_turns)

    # Determine parallelism level
    max_workers = min(
        self._config.asr.cpu_threads or (os.cpu_count() or 4),
        total_windows,
        4,  # cap to avoid memory explosion with large models
    )

    # Process windows in parallel batches
    diarized: list[DiarizedSegment] = []
    completed = 0

    for batch_start in range(0, total_windows, max_workers):
        batch = normalized_turns[batch_start:batch_start + max_workers]
        tasks = []
        for turn in batch:
            window = _slice_audio_chunk(full_audio, turn.start_time, turn.end_time)
            if window.num_samples < _MIN_GIGAAM_WINDOW_SAMPLES:
                continue
            tasks.append(self._transcribe_single_window(window, turn))

        results = await asyncio.gather(*tasks)

        for window_segments in results:
            diarized.extend(window_segments)

        completed += len(batch)
        # Emit progress
        self._progress(
            stage=PipelineStage.ASR,
            message=f"Transcribing speaker windows {completed}/{total_windows} ...",
            progress=_ASR_PROGRESS_WINDOW_START + (
                (_ASR_PROGRESS_WINDOW_END - _ASR_PROGRESS_WINDOW_START)
                * completed / total_windows
            ),
            phase="speaker_window_transcription",
            completed_windows=completed,
            total_windows=total_windows,
        )

    diarized.sort(key=lambda item: (item.segment.start_time, item.segment.end_time))
    return diarized


async def _transcribe_single_window(
    self,
    window: AudioChunk,
    turn: SpeakerTurn,
) -> list[DiarizedSegment]:
    """Transcribe a single speaker window and rebase to absolute time."""
    segments = await self._asr.transcribe(
        window,
        language=self._config.asr.language,
        word_timestamps=self._config.asr.word_timestamps,
    )
    result = []
    for segment in segments:
        rebased = _rebase_segment(segment, window.timestamp_start)
        result.append(
            DiarizedSegment(
                segment=rebased,
                speaker_id=turn.speaker_id,
                speaker_source="ml",
            )
        )
    return result
```

#### 3.2 ASR engine thread safety

GigaAM uses `transformers` AutoModel which is **not** thread-safe for concurrent
inference on the same model instance. Two approaches:

**Option A (recommended)**: Keep `max_workers=1` for GigaAM by default, but allow
`max_workers > 1` for faster-whisper which supports concurrent calls via
CTranslate2's thread pool. Expose this as `PipelineConfig.asr.parallel_windows`.

**Option B**: Use a semaphore inside the ASR engine to serialize access.
This still allows asyncio-level parallelism for I/O but serializes inference.

We go with **Option A**: the parallel loop structure is in place, but the effective
parallelism depends on the engine. GigaAM defaults to 1 (serial with async structure),
faster-whisper can go up to 4.

#### 3.3 Progress reporting

Progress is emitted per **batch completion**, not per window. This means progress
updates are less granular but still frequent. Each progress event includes:

```python
{
    "completed_windows": int,
    "total_windows": int,
    "phase": "speaker_window_transcription",
    "eta_s": float | None,  # based on avg time per batch
}
```

ETA calculation:
```python
elapsed = time.monotonic() - window_phase_started_at
avg_per_window = elapsed / completed
remaining = avg_per_window * (total_windows - completed)
```

### Tests

| Test | File | What it verifies |
|------|------|-----------------|
| `test_parallel_windows_produces_same_result` | `tests/unit/test_batch_pipeline.py` | Output matches sequential version for same input |
| `test_parallel_windows_respects_max_workers` | `tests/unit/test_batch_pipeline.py` | With max_workers=2, no more than 2 concurrent calls |
| `test_parallel_progress_events` | `tests/unit/test_batch_pipeline.py` | Progress events emitted per batch with ETA |
| `test_parallel_windows_sorts_output` | `tests/unit/test_batch_pipeline.py` | Output sorted by start_time regardless of completion order |

---

## Feature 4: Chunked Diarization with Parallelization

### Motivation

Pyannote processes the entire audio in one pass. For a 30-minute file, this means
the segmentation NN processes ~30 minutes of audio through sliding windows.
By splitting the audio into chunks (e.g., 5 minutes each) and running pyannote
on each chunk in parallel, we can achieve near-linear speedup by CPU core count.

### Design

#### 4.1 Chunked diarization wrapper

**File**: `src/voxfusion/diarization/chunked.py` (new)

```python
_DEFAULT_CHUNK_DURATION_S = 300.0  # 5 minutes
_CHUNK_OVERLAP_S = 10.0            # 10s overlap for speaker re-identification


class ChunkedDiarizer:
    """Splits audio into chunks and runs diarization in parallel.

    Speaker labels from different chunks are reconciled by computing
    embedding similarity in the overlap regions.
    """

    def __init__(
        self,
        inner_factory: Callable[[], PyAnnoteDiarizer],
        *,
        chunk_duration_s: float = _DEFAULT_CHUNK_DURATION_S,
        chunk_overlap_s: float = _CHUNK_OVERLAP_S,
        max_workers: int | None = None,  # None = cpu_count // 2
    ) -> None:
        ...

    async def diarize_turns(self, audio: AudioChunk) -> list[SpeakerTurn]:
        """Split audio, diarize chunks in parallel, merge turns."""
        if audio.duration <= self._chunk_duration_s * 1.5:
            # Short audio — don't bother chunking
            return await self._inner.diarize_turns(audio)

        chunks = self._split_audio(audio)
        # Run in ProcessPoolExecutor for true parallelism (GIL bypass)
        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor(max_workers=self._max_workers) as pool:
            futures = [
                loop.run_in_executor(pool, self._diarize_chunk_sync, chunk)
                for chunk in chunks
            ]
            chunk_turns = await asyncio.gather(*futures)

        return self._merge_chunk_turns(chunk_turns, chunks)
```

#### 4.2 Chunk splitting strategy

```
Audio:   |-------- 30 min --------|
Chunk 1: |--- 5:00 + 0:10 overlap ---|
Chunk 2:      |--- 0:10 + 5:00 + 0:10 ---|
Chunk 3:           |--- 0:10 + 5:00 + 0:10 ---|
...
```

- Each chunk is `chunk_duration_s + chunk_overlap_s` long
- Overlap regions are used for speaker identity reconciliation
- Minimum useful chunk: if remaining audio < 30s, merge with previous chunk

#### 4.3 Speaker reconciliation across chunks

The overlap region contains speaker turns from both the current and next chunk.
We match speakers across chunks by:

1. In the overlap window, collect speaker turns from both chunks
2. For each speaker pair (chunk_N, chunk_N+1), compute temporal overlap
3. The pair with highest overlap = same speaker → rename labels
4. Unmatched speakers get new unique IDs

This is simple and fast since we're only comparing a small overlap region.
More sophisticated approaches (embedding comparison) can be added later.

#### 4.4 When to use chunked diarization

Chunked diarization is activated automatically when:
- Audio duration > 7.5 minutes (1.5x chunk size)
- Strategy is "ml" or "hybrid"
- More than 1 CPU core available

Configuration:
```python
class DiarizationMLConfig(BaseModel):
    # ... existing fields ...
    chunked: bool = True                    # enable chunked processing
    chunk_duration_s: float = 300.0         # 5 minutes
    chunk_overlap_s: float = 10.0           # overlap for reconciliation
    chunk_max_workers: int | None = None    # None = auto (cpu_count // 2)
```

#### 4.5 Progress reporting

Chunked diarization provides much better progress visibility:

```
"Running speaker diarization: chunk 2/6 (parallel, 3 workers)... 33% ETA ~45s"
```

Progress structure:
```python
{
    "phase": "chunked_diarization",
    "completed_chunks": int,
    "total_chunks": int,
    "parallel_workers": int,
    "eta_s": float | None,
    "elapsed_s": float,
}
```

Since chunks run in parallel, progress jumps in bursts (a batch of `max_workers`
chunks completes at once). ETA is calculated as:
```python
batches_remaining = ceil((total_chunks - completed_chunks) / max_workers)
avg_batch_time = elapsed / completed_batches
eta = batches_remaining * avg_batch_time
```

### GPU Behavior (Question 5 answer)

**On a machine with GPU (CUDA):**

| Component | GPU used? | Notes |
|-----------|-----------|-------|
| Pyannote segmentation | Yes | PyanNet runs on CUDA, ~10x speedup |
| Pyannote embeddings | Yes | WeSpeaker/ResNet on CUDA |
| Pyannote clustering | No | CPU-only (sklearn agglomerative) |
| GigaAM inference | Depends | Uses CUDA if torch detects it |
| faster-whisper | Yes | CTranslate2 auto-selects CUDA |

**With GPU, chunked diarization behavior changes:**
- Chunking is LESS beneficial with GPU because pyannote is already fast
- The GPU can only run one model at a time (unless multi-GPU)
- So `chunk_max_workers` should default to 1 on GPU (sequential chunks)
- The main benefit of chunking on GPU is **progress visibility** rather than speed

Auto-detection:
```python
def _default_chunk_workers(device: str) -> int:
    if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
        return 1  # GPU: sequential, don't compete for VRAM
    return max(1, (os.cpu_count() or 4) // 2)  # CPU: parallel
```

### Tests

| Test | File | What it verifies |
|------|------|-----------------|
| `test_chunked_splits_long_audio` | `tests/unit/test_chunked_diarization.py` | Audio > 7.5min is split into chunks |
| `test_chunked_short_audio_passthrough` | `tests/unit/test_chunked_diarization.py` | Short audio delegates to inner diarizer directly |
| `test_chunked_speaker_reconciliation` | `tests/unit/test_chunked_diarization.py` | Same speaker in overlap gets same ID |
| `test_chunked_new_speaker_in_later_chunk` | `tests/unit/test_chunked_diarization.py` | New speaker gets unique ID |
| `test_chunked_progress_events` | `tests/unit/test_chunked_diarization.py` | Correct chunk progress with ETA |
| `test_chunked_workers_auto_gpu` | `tests/unit/test_chunked_diarization.py` | Workers=1 when CUDA detected |
| `test_chunked_workers_auto_cpu` | `tests/unit/test_chunked_diarization.py` | Workers=cpu_count//2 on CPU |
| `test_chunked_config_fields` | `tests/unit/test_config.py` | New config fields validate correctly |

---

## Task Decomposition

### Phase 1: Quick Wins (no new dependencies)

| # | Task | Files | Est. complexity |
|---|------|-------|----------------|
| 1.1 | Create `NoneDiarizer` class | `diarization/none.py` | Small |
| 1.2 | Register "none" in `factory.py` | `diarization/factory.py` | Small |
| 1.3 | Handle "none" in `batch.py` path decision | `pipeline/batch.py` | Small |
| 1.4 | Add "none" to GUI dropdown | `gui/main.py` | Small |
| 1.5 | Write tests for "none" strategy | `tests/unit/test_diarization.py`, `test_diarization_factory.py`, `test_batch_pipeline.py` | Small |
| 1.6 | Create `ToolTip` + `create_help_icon` | `gui/tooltip.py` | Small |
| 1.7 | Add tooltips to all File tab controls | `gui/main.py` | Medium |
| 1.8 | Add tooltips to all Live Capture tab controls | `gui/main.py` | Medium |
| 1.9 | Write tooltip unit tests | `tests/unit/test_tooltip.py` | Small |

### Phase 2: Speaker Count UX

| # | Task | Files | Est. complexity |
|---|------|-------|----------------|
| 2.1 | Add Speaker Count preset dropdown to GUI | `gui/main.py` | Medium |
| 2.2 | Wire preset selection to min/max fields | `gui/main.py` | Small |
| 2.3 | Persist speaker count preset in settings | `gui/main.py`, `gui/helpers.py` | Small |
| 2.4 | Create `speaker_counter.py` (estimation) | `diarization/speaker_counter.py` | Medium |
| 2.5 | Add "Detect" button to GUI | `gui/main.py` | Medium |
| 2.6 | Write speaker counter tests | `tests/unit/test_speaker_counter.py` | Medium |
| 2.7 | Write GUI speaker count tests | `tests/unit/test_gui_flow.py` | Small |

### Phase 3: Parallel Window Transcription

| # | Task | Files | Est. complexity |
|---|------|-------|----------------|
| 3.1 | Extract `_transcribe_single_window` method | `pipeline/batch.py` | Small |
| 3.2 | Implement parallel batch loop in `_transcribe_diarized_windows` | `pipeline/batch.py` | Medium |
| 3.3 | Add `parallel_windows` config field | `config/models.py` | Small |
| 3.4 | Update progress reporting for batch mode | `pipeline/batch.py` | Small |
| 3.5 | Write parallel transcription tests | `tests/unit/test_batch_pipeline.py` | Medium |

### Phase 4: Chunked Diarization

| # | Task | Files | Est. complexity |
|---|------|-------|----------------|
| 4.1 | Add chunked config fields to `DiarizationMLConfig` | `config/models.py` | Small |
| 4.2 | Implement `ChunkedDiarizer` with split logic | `diarization/chunked.py` | Large |
| 4.3 | Implement speaker reconciliation across chunks | `diarization/chunked.py` | Medium |
| 4.4 | Integrate chunked diarizer into factory | `diarization/factory.py` | Medium |
| 4.5 | Add chunked progress reporting to batch pipeline | `pipeline/batch.py` | Medium |
| 4.6 | Auto-detect GPU for worker count | `diarization/chunked.py` | Small |
| 4.7 | Write chunked diarization tests | `tests/unit/test_chunked_diarization.py` | Large |
| 4.8 | Integration test: full pipeline with chunked diarization | `tests/integration/` | Medium |

---

## Progress & Timing Summary

### Current flow (10-min file, CPU, GigaAM + pyannote ML):

```
[Capture]      ████                          ~2s
[Preprocess]   ██                            ~1s
[Diarization]  ████████████████████████████   ~4-8 min  ← BOTTLENECK
[ASR windows]  ██████████                    ~30s
[Alignment]    █                             ~0.1s
Total:                                       ~5-9 min
```

### After optimization (same setup):

**With strategy="none" (single speaker):**
```
[Capture]      ████                          ~2s
[Preprocess]   ██                            ~1s
[ASR]          ██████████                    ~30s
[Diarization]  (skipped)                     ~0s
Total:                                       ~33s  (10-16x faster)
```

**With chunked diarization (4 cores, 3 workers):**
```
[Capture]      ████                          ~2s
[Preprocess]   ██                            ~1s
[Diarization]  ██████████                    ~1.5-3 min (3x faster)
[ASR windows]  ██████████                    ~30s
Total:                                       ~2.5-4 min (2-3x faster)
```

**With GPU:**
```
[Capture]      ████                          ~2s
[Preprocess]   ██                            ~1s
[Diarization]  ████                          ~15-30s
[ASR windows]  ████                          ~10s
Total:                                       ~30-45s
```

### Progress bar behavior

Each phase maps to a progress range:

| Phase | Progress range | Behavior |
|-------|---------------|----------|
| Capture + Preprocess | 0-15% | Quick jump |
| Diarization (or skip) | 15-55% | Heartbeat updates every 10s with ETA |
| ASR windows | 55-85% | Per-batch updates with ETA |
| Post-processing | 85-100% | Quick jump |

Status label format during diarization:
```
"Speaker diarization: chunk 2/6 (3 workers) — 01:23 elapsed, ETA ~00:45"
```

Status label format during ASR windows:
```
"Transcribing: window 12/30 (SPEAKER_A 2:15-2:39) — ETA ~00:18"
```

Elapsed time label (already exists in GUI, right side):
```
"03:45 elapsed"
```
