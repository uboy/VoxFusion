"""Tests for logging-time runtime environment bootstrap."""

from __future__ import annotations

from pathlib import Path

from voxfusion.logging import _ensure_runtime_environment_defaults, _should_suppress_log_message


def test_runtime_environment_defaults_set_noise_controls(tmp_path: Path, monkeypatch) -> None:
    mpl_dir = tmp_path / "mplconfig"

    monkeypatch.setenv("MPLCONFIGDIR", str(mpl_dir))
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_TELEMETRY", raising=False)
    monkeypatch.delenv("PYANNOTE_METRICS_ENABLED", raising=False)

    _ensure_runtime_environment_defaults()

    assert mpl_dir.is_dir()
    assert Path(mpl_dir) == Path(mpl_dir)
    assert _should_suppress_log_message("Could not save font_manager cache [Errno 13]")
    assert _should_suppress_log_message(
        "Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work"
    )


def test_runtime_environment_defaults_preserve_requested_mpl_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mpl_dir = tmp_path / "custom-mpl"
    monkeypatch.setenv("MPLCONFIGDIR", str(mpl_dir))

    _ensure_runtime_environment_defaults()

    assert Path(mpl_dir).is_dir()
