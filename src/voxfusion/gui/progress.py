"""Global tqdm progress helpers for thread-safe GUI logging."""

from __future__ import annotations

import threading

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency fallback
    tqdm = None  # type: ignore[assignment]


class _NullProgress:
    """No-op progress object used when tqdm is unavailable."""

    def update(self, _value: int = 1) -> None:
        return

    def close(self) -> None:
        return


_progress_lock = threading.Lock()
_progress_bars: dict[str, tqdm | _NullProgress] = {}


def get_stage_progress(stage: str, total: int | None = None) -> tqdm | _NullProgress:
    """Return one shared progress bar per stage name."""
    with _progress_lock:
        progress = _progress_bars.get(stage)
        if progress is not None:
            return progress

        if tqdm is None:
            progress = _NullProgress()
        else:
            progress = tqdm(
                total=total,
                desc=stage,
                ascii=True,
                dynamic_ncols=True,
                leave=False,
                disable=True,
            )

        _progress_bars[stage] = progress
        return progress


def close_all_progress() -> None:
    """Close and clear all shared progress bars."""
    with _progress_lock:
        for progress in _progress_bars.values():
            progress.close()
        _progress_bars.clear()
