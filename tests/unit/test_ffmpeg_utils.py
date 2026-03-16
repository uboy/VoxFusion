"""Tests for FFmpeg helper utilities."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from voxfusion.media.ffmpeg import (
    build_linear_overlay_filter_graph,
    detect_best_h264_encoder,
    recommended_encoder_workers,
)


def test_recommended_workers_for_hardware_encoder() -> None:
    assert recommended_encoder_workers(use_hardware_encoder=True, cpu_count=4) == 2
    assert recommended_encoder_workers(use_hardware_encoder=True, cpu_count=16) == 3


def test_recommended_workers_for_software_encoder() -> None:
    assert recommended_encoder_workers(use_hardware_encoder=False, cpu_count=1) == 1
    assert recommended_encoder_workers(use_hardware_encoder=False, cpu_count=12) == 8


def test_filter_graph_starts_with_background_label() -> None:
    graph = build_linear_overlay_filter_graph(["0:v", "1:v"])
    assert graph.startswith("color=c=black:s=1920x1080[bg];")
    assert "[bg][0:v]overlay=shortest=1[tmp0]" in graph
    assert "[tmp0][1:v]overlay=shortest=1[vout]" in graph


def test_filter_graph_without_layers_still_produces_output() -> None:
    graph = build_linear_overlay_filter_graph([])
    assert graph == "color=c=black:s=1920x1080[bg];[bg]copy[vout]"


def test_encoder_detection_falls_back_to_libx264(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_: object) -> SimpleNamespace:
        calls.append(cmd)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr("subprocess.run", fake_run)
    assert detect_best_h264_encoder() == "libx264"
    assert len(calls) == 3


def test_encoder_detection_uses_first_working_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], **_: object) -> SimpleNamespace:
        if "h264_qsv" in cmd:
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr("subprocess.run", fake_run)
    assert detect_best_h264_encoder() == "h264_qsv"
