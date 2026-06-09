"""Tests for device resolution in faster-whisper."""

from __future__ import annotations

from unittest.mock import patch


def test_resolve_device_cuda_returns_cuda() -> None:
    from voxfusion.asr.faster_whisper import _resolve_device

    with patch("voxfusion.asr.faster_whisper.has_ctranslate2_cuda", return_value=True):
        device, compute = _resolve_device("cuda")
    assert device == "cuda"
    assert compute == "float16"


def test_resolve_device_cuda_index_returns_cuda() -> None:
    from voxfusion.asr.faster_whisper import _resolve_device

    with patch("voxfusion.asr.faster_whisper.has_ctranslate2_cuda", return_value=True):
        device, compute = _resolve_device("cuda:0")
    assert device == "cuda"
    assert compute == "float16"


def test_resolve_device_auto_selects_cuda() -> None:
    from voxfusion.asr.faster_whisper import _resolve_device

    with patch("voxfusion.asr.faster_whisper.has_ctranslate2_cuda", return_value=True):
        device, compute = _resolve_device("auto")
    assert device == "cuda"
    assert compute == "float16"


def test_resolve_device_auto_falls_back_to_cpu() -> None:
    from voxfusion.asr.faster_whisper import _resolve_device

    with patch("voxfusion.asr.faster_whisper.has_ctranslate2_cuda", return_value=False):
        device, compute = _resolve_device("auto")
    assert device == "cpu"
    assert compute == "int8"


def test_resolve_device_cuda_falls_back_no_ctranslate2() -> None:
    from voxfusion.asr.faster_whisper import _resolve_device

    with patch("voxfusion.asr.faster_whisper.has_ctranslate2_cuda", return_value=False):
        device, compute = _resolve_device("cuda")
    assert device == "cpu"
    assert compute == "int8"


def test_resolve_device_cpu_stays_cpu() -> None:
    from voxfusion.asr.faster_whisper import _resolve_device

    device, compute = _resolve_device("cpu")
    assert device == "cpu"
    assert compute == "int8"
