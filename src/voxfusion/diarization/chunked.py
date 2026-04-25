"""Chunked diarization wrapper for long audio files.

Splits audio into overlapping chunks, runs the inner diarizer on each chunk,
then stitches chunk-local speaker labels into one global speaker space. This
preserves progress visibility and parallel chunk execution without leaking
per-chunk local speaker IDs into the final result.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING

import numpy as np

from voxfusion.diarization.alignment import SpeakerTurn
from voxfusion.diarization.stitching import stitch_chunk_speakers
from voxfusion.logging import get_logger
from voxfusion.models.audio import AudioChunk
from voxfusion.models.diarization import DiarizedSegment
from voxfusion.models.transcription import TranscriptionSegment

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

# Minimum audio duration before chunking is applied (seconds).
# Below this threshold we delegate directly to the inner diarizer.
_MIN_DURATION_FOR_CHUNKING_FACTOR = 1.5  # chunk_duration × this factor


def _default_max_workers(device: str) -> int:
    """Return a sensible default worker count based on the compute device."""
    try:
        import torch

        if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
            # One GPU — serial is best; no VRAM contention.
            return 1
    except ImportError:
        pass
    return max(1, (os.cpu_count() or 4) // 2)


def _slice_chunk(audio: AudioChunk, start_s: float, end_s: float) -> AudioChunk:
    sr = audio.sample_rate
    start_idx = max(0, int(round(start_s * sr)))
    end_idx = min(audio.num_samples, int(round(end_s * sr)))
    return AudioChunk(
        samples=np.ascontiguousarray(audio.samples[start_idx:end_idx], dtype=np.float32),
        sample_rate=sr,
        channels=audio.channels,
        timestamp_start=start_s,
        timestamp_end=end_s,
        source=audio.source,
        dtype="float32",
    )


def _build_chunk_boundaries(
    duration: float,
    chunk_duration_s: float,
    chunk_overlap_s: float,
) -> list[tuple[float, float, float, float]]:
    """Return list of (chunk_start, chunk_end, keep_start, keep_end) tuples.

    *chunk_start/chunk_end* are the actual audio slice boundaries including
    overlap. *keep_start/keep_end* are the non-overlapping regions that are
    kept when composing final turns across chunks.
    """
    boundaries: list[tuple[float, float, float, float]] = []
    cursor = 0.0
    while cursor < duration:
        chunk_end = min(cursor + chunk_duration_s, duration)
        read_end = min(chunk_end + chunk_overlap_s, duration) if chunk_end < duration else chunk_end
        keep_start = cursor
        keep_end = chunk_end
        boundaries.append((cursor, read_end, keep_start, keep_end))
        cursor = chunk_end
    return boundaries


def _coerce_absolute_turns(
    turns: list[SpeakerTurn],
    *,
    chunk_start: float,
    chunk_end: float,
) -> list[SpeakerTurn]:
    """Return turns in absolute audio coordinates.

    `PyAnnoteDiarizer` already returns absolute coordinates when the input
    `AudioChunk` carries a non-zero `timestamp_start`. Older/simple diarizers may
    still return chunk-local offsets. Accept both forms here so chunked
    diarization does not double-shift already-absolute turns.
    """
    if not turns:
        return []

    tolerance_s = 1.0
    if all(
        turn.start_time >= (chunk_start - tolerance_s)
        and turn.end_time <= (chunk_end + tolerance_s)
        for turn in turns
    ):
        return turns

    return [
        SpeakerTurn(
            speaker_id=turn.speaker_id,
            start_time=turn.start_time + chunk_start,
            end_time=turn.end_time + chunk_start,
        )
        for turn in turns
    ]


class ChunkedDiarizer:
    """Run an inner diarizer chunk-by-chunk on long audio files.

    For audio shorter than ``chunk_duration_s * 1.5``, the inner diarizer is
    called directly. For longer audio, the file is split into overlapping
    chunks, each processed independently, and local chunk speaker IDs are
    stitched into one deterministic global speaker space.

    Args:
        inner_diarizer_factory: Zero-argument callable returning a fresh inner
            diarizer instance. Called once (or once per worker when parallel).
        chunk_duration_s: Target duration of each chunk in seconds.
        chunk_overlap_s: Overlap between consecutive chunks used for speaker
            stitching.
        max_workers: Parallel workers. 1 = sequential. None = auto-detect.
        device: Compute device of the inner diarizer ("auto", "cpu", "cuda").
            Used for auto-detecting ``max_workers``.
        on_chunk_progress: Optional callback ``(completed, total)`` called
            after each chunk completes.
    """

    def __init__(
        self,
        inner_diarizer_factory: Callable[[], object],
        *,
        chunk_duration_s: float = 300.0,
        chunk_overlap_s: float = 10.0,
        max_workers: int | None = None,
        device: str = "auto",
        on_chunk_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        self._factory = inner_diarizer_factory
        self._chunk_duration_s = chunk_duration_s
        self._chunk_overlap_s = chunk_overlap_s
        self._device = device
        self._max_workers = max_workers if max_workers is not None else _default_max_workers(device)
        self._on_chunk_progress = on_chunk_progress
        self._inner: object | None = None

    def _get_inner(self) -> object:
        if self._inner is None:
            self._inner = self._factory()
        return self._inner

    async def diarize(
        self,
        segments: list[TranscriptionSegment],
        audio: AudioChunk | None = None,
    ) -> list[DiarizedSegment]:
        """Delegate to inner diarizer — chunking applies only to turn detection."""
        inner = self._get_inner()
        return await inner.diarize(segments, audio)  # type: ignore[union-attr]

    async def diarize_stream(
        self,
        segment_stream: AsyncIterator[tuple[TranscriptionSegment, AudioChunk]],
    ) -> AsyncIterator[DiarizedSegment]:
        """Delegate to inner diarizer."""
        inner = self._get_inner()
        async for item in inner.diarize_stream(segment_stream):  # type: ignore[union-attr]
            yield item

    async def diarize_turns(self, audio: AudioChunk) -> list[SpeakerTurn]:
        """Diarize audio in chunks and return stitched speaker turns.

        Short audio (< ``chunk_duration_s * 1.5``) is passed directly to the
        inner diarizer without chunking.
        """
        min_for_chunking = self._chunk_duration_s * _MIN_DURATION_FOR_CHUNKING_FACTOR
        if audio.duration <= min_for_chunking:
            log.info(
                "chunked_diarizer.passthrough",
                duration_s=round(audio.duration, 2),
                threshold_s=round(min_for_chunking, 2),
            )
            inner = self._get_inner()
            return await inner.diarize_turns(audio)  # type: ignore[union-attr]

        boundaries = _build_chunk_boundaries(
            audio.duration, self._chunk_duration_s, self._chunk_overlap_s
        )
        total_chunks = len(boundaries)
        log.info(
            "chunked_diarizer.start",
            duration_s=round(audio.duration, 2),
            total_chunks=total_chunks,
            chunk_duration_s=self._chunk_duration_s,
            chunk_overlap_s=self._chunk_overlap_s,
            max_workers=self._max_workers,
        )

        if self._max_workers == 1:
            return await self._process_sequential(audio, boundaries)
        return await self._process_parallel(audio, boundaries)

    async def _process_sequential(
        self,
        audio: AudioChunk,
        boundaries: list[tuple[float, float, float, float]],
    ) -> list[SpeakerTurn]:
        total_chunks = len(boundaries)
        per_chunk_turns: list[list[SpeakerTurn]] = []

        for chunk_idx, (chunk_start, chunk_end, keep_start, keep_end) in enumerate(
            boundaries, start=1
        ):
            chunk_audio = _slice_chunk(audio, chunk_start, chunk_end)
            inner = self._get_inner()
            local_turns = await inner.diarize_turns(chunk_audio)  # type: ignore[union-attr]

            abs_turns = _coerce_absolute_turns(
                local_turns,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
            )
            per_chunk_turns.append(abs_turns)

            log.info(
                "chunked_diarizer.chunk_done",
                chunk=chunk_idx,
                total_chunks=total_chunks,
                keep_start_s=round(keep_start, 2),
                keep_end_s=round(keep_end, 2),
            )
            if self._on_chunk_progress:
                self._on_chunk_progress(chunk_idx, total_chunks)

        return self._stitch_chunks(boundaries, per_chunk_turns)

    async def _process_parallel(
        self,
        audio: AudioChunk,
        boundaries: list[tuple[float, float, float, float]],
    ) -> list[SpeakerTurn]:
        """Process chunks in parallel, then stitch the local speakers globally."""
        total_chunks = len(boundaries)
        semaphore = asyncio.Semaphore(self._max_workers)
        completed_count = 0

        async def _run_chunk(
            chunk_idx: int,
            chunk_start: float,
            chunk_end: float,
        ) -> list[SpeakerTurn]:
            nonlocal completed_count
            chunk_audio = _slice_chunk(audio, chunk_start, chunk_end)
            async with semaphore:
                inner = self._factory()
                local_turns = await inner.diarize_turns(chunk_audio)  # type: ignore[union-attr]
            abs_turns = _coerce_absolute_turns(
                local_turns,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
            )
            completed_count += 1
            log.info(
                "chunked_diarizer.chunk_done_parallel",
                chunk=chunk_idx,
                total_chunks=total_chunks,
                completed=completed_count,
            )
            if self._on_chunk_progress:
                self._on_chunk_progress(completed_count, total_chunks)
            return abs_turns

        per_chunk_turns = await asyncio.gather(
            *[
                _run_chunk(i + 1, chunk_start, chunk_end)
                for i, (chunk_start, chunk_end, _ks, _ke) in enumerate(boundaries)
            ]
        )
        return self._stitch_chunks(boundaries, per_chunk_turns)

    def _stitch_chunks(
        self,
        boundaries: list[tuple[float, float, float, float]],
        per_chunk_turns: list[list[SpeakerTurn]],
    ) -> list[SpeakerTurn]:
        mappings = stitch_chunk_speakers(
            per_chunk_turns,
            boundaries=boundaries,
            chunk_overlap_s=self._chunk_overlap_s,
        )
        global_turns: list[SpeakerTurn] = []
        for (_chunk_start, _chunk_end, keep_start, keep_end), abs_turns, mapping in zip(
            boundaries,
            per_chunk_turns,
            mappings,
            strict=False,
        ):
            for turn in abs_turns:
                keep_s = max(turn.start_time, keep_start)
                keep_e = min(turn.end_time, keep_end)
                if keep_e <= keep_s:
                    continue
                global_turns.append(
                    SpeakerTurn(
                        speaker_id=mapping.get(turn.speaker_id, turn.speaker_id),
                        start_time=keep_s,
                        end_time=keep_e,
                    )
                )
        return global_turns
