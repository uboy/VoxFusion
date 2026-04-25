"""Global speaker stitching for chunk-local diarization results."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import TypeAlias

from voxfusion.diarization.alignment import SpeakerTurn

ChunkBoundary: TypeAlias = tuple[float, float, float, float]
ChunkSpeakerKey: TypeAlias = tuple[int, str]


@dataclass(frozen=True)
class _ScoredEdge:
    left: ChunkSpeakerKey
    right: ChunkSpeakerKey
    score: float


class _UnionFind:
    def __init__(self, items: list[ChunkSpeakerKey]) -> None:
        self._parent = {item: item for item in items}

    def find(self, item: ChunkSpeakerKey) -> ChunkSpeakerKey:
        parent = self._parent[item]
        if parent != item:
            parent = self.find(parent)
            self._parent[item] = parent
        return parent

    def union(self, left: ChunkSpeakerKey, right: ChunkSpeakerKey) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self._parent[right_root] = left_root


def _overlap_seconds(
    t_start: float,
    t_end: float,
    window_start: float,
    window_end: float,
) -> float:
    return max(0.0, min(t_end, window_end) - max(t_start, window_start))


def _shared_overlap(
    left_turn: SpeakerTurn,
    right_turn: SpeakerTurn,
    overlap_start: float,
    overlap_end: float,
) -> float:
    shared_start = max(left_turn.start_time, right_turn.start_time, overlap_start)
    shared_end = min(left_turn.end_time, right_turn.end_time, overlap_end)
    return max(0.0, shared_end - shared_start)


def _speaker_turns_by_id(turns: list[SpeakerTurn]) -> dict[str, list[SpeakerTurn]]:
    grouped: dict[str, list[SpeakerTurn]] = {}
    for turn in turns:
        grouped.setdefault(turn.speaker_id, []).append(turn)
    return grouped


def _boundary_pair_scores(
    left_turns: list[SpeakerTurn],
    right_turns: list[SpeakerTurn],
    *,
    overlap_start: float,
    overlap_end: float,
) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    for left_turn in left_turns:
        if (
            _overlap_seconds(left_turn.start_time, left_turn.end_time, overlap_start, overlap_end)
            <= 0
        ):
            continue
        for right_turn in right_turns:
            shared = _shared_overlap(left_turn, right_turn, overlap_start, overlap_end)
            if shared <= 0:
                continue
            key = (left_turn.speaker_id, right_turn.speaker_id)
            scores[key] = scores.get(key, 0.0) + shared
    return scores


def _maximize_boundary_matches(
    left_ids: tuple[str, ...],
    right_ids: tuple[str, ...],
    pair_scores: dict[tuple[str, str], float],
) -> list[tuple[str, str, float]]:
    """Return the maximum-weight one-to-one match across one chunk boundary."""

    @cache
    def _solve(left_index: int, used_mask: int) -> tuple[float, tuple[tuple[str, str, float], ...]]:
        if left_index >= len(left_ids):
            return 0.0, ()

        best_score, best_pairs = _solve(left_index + 1, used_mask)
        left_id = left_ids[left_index]
        for right_index, right_id in enumerate(right_ids):
            if used_mask & (1 << right_index):
                continue
            score = pair_scores.get((left_id, right_id), 0.0)
            if score <= 0:
                continue
            tail_score, tail_pairs = _solve(left_index + 1, used_mask | (1 << right_index))
            total_score = score + tail_score
            if total_score > best_score:
                best_score = total_score
                best_pairs = ((left_id, right_id, score), *tail_pairs)
        return best_score, best_pairs

    return list(_solve(0, 0)[1])


def stitch_chunk_speakers(
    per_chunk_turns: list[list[SpeakerTurn]],
    *,
    boundaries: list[ChunkBoundary],
    chunk_overlap_s: float,
) -> list[dict[str, str]]:
    """Map chunk-local speaker ids onto one deterministic global speaker space."""
    local_nodes: list[ChunkSpeakerKey] = []
    node_turns: dict[ChunkSpeakerKey, list[SpeakerTurn]] = {}
    for chunk_index, turns in enumerate(per_chunk_turns):
        grouped = _speaker_turns_by_id(turns)
        for speaker_id, speaker_turns in grouped.items():
            key = (chunk_index, speaker_id)
            local_nodes.append(key)
            node_turns[key] = speaker_turns

    if not local_nodes:
        return [{} for _ in per_chunk_turns]

    union_find = _UnionFind(local_nodes)
    for boundary_index in range(1, len(per_chunk_turns)):
        overlap_start = boundaries[boundary_index][0]
        overlap_end = min(boundaries[boundary_index - 1][1], overlap_start + chunk_overlap_s)
        pair_scores = _boundary_pair_scores(
            per_chunk_turns[boundary_index - 1],
            per_chunk_turns[boundary_index],
            overlap_start=overlap_start,
            overlap_end=overlap_end,
        )
        if not pair_scores:
            continue
        left_ids = tuple(sorted({left_id for left_id, _right_id in pair_scores}))
        right_ids = tuple(sorted({right_id for _left_id, right_id in pair_scores}))
        for left_id, right_id, score in _maximize_boundary_matches(
            left_ids, right_ids, pair_scores
        ):
            if score <= 0:
                continue
            union_find.union((boundary_index - 1, left_id), (boundary_index, right_id))

    components: dict[ChunkSpeakerKey, list[ChunkSpeakerKey]] = {}
    for node in local_nodes:
        components.setdefault(union_find.find(node), []).append(node)

    ordered_components = sorted(
        components.values(),
        key=lambda component: min(
            turn.start_time for node in component for turn in node_turns[node]
        ),
    )
    component_to_global_id = {
        tuple(sorted(component)): f"SPEAKER_{index:02d}"
        for index, component in enumerate(ordered_components)
    }

    per_chunk_mappings: list[dict[str, str]] = [{} for _ in per_chunk_turns]
    for component in ordered_components:
        global_id = component_to_global_id[tuple(sorted(component))]
        for chunk_index, local_speaker_id in component:
            per_chunk_mappings[chunk_index][local_speaker_id] = global_id
    return per_chunk_mappings


__all__ = [
    "_maximize_boundary_matches",
    "stitch_chunk_speakers",
]
