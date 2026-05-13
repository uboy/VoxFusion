"""Tests for multi-GPU device resolution in faster-whisper."""

from __future__ import annotations

from unittest.mock import patch


def test_resolve_device_cuda_index_passthrough() -> None:
    from voxfusion.asr.faster_whisper import _resolve_device

    with patch("voxfusion.asr.faster_whisper.has_ctranslate2_cuda", return_value=True):
        device, compute = _resolve_device("cuda:2")
    assert device == "cuda:2"
    assert compute == "float16"


def test_resolve_device_auto_selects_best_gpu() -> None:
    from voxfusion.asr.faster_whisper import _resolve_device

    with (
        patch("voxfusion.asr.faster_whisper.has_ctranslate2_cuda", return_value=True),
        patch("voxfusion.asr.faster_whisper.select_best_gpu", return_value="cuda:1"),
    ):
        device, compute = _resolve_device("auto")
    assert device == "cuda:1"
    assert compute == "float16"


def test_resolve_device_cuda_generic_selects_best_gpu() -> None:
    from voxfusion.asr.faster_whisper import _resolve_device

    with (
        patch("voxfusion.asr.faster_whisper.has_ctranslate2_cuda", return_value=True),
        patch("voxfusion.asr.faster_whisper.select_best_gpu", return_value="cuda:2"),
    ):
        device, compute = _resolve_device("cuda")
    assert device == "cuda:2"
    assert compute == "float16"


def test_resolve_device_auto_falls_back_to_cpu_when_no_gpu() -> None:
    from voxfusion.asr.faster_whisper import _resolve_device

    with (
        patch("voxfusion.asr.faster_whisper.has_ctranslate2_cuda", return_value=True),
        patch("voxfusion.asr.faster_whisper.select_best_gpu", return_value=None),
    ):
        device, compute = _resolve_device("auto")
    assert device == "cpu"
    assert compute == "int8"


def test_resolve_device_cuda_index_falls_back_no_ctranslate2() -> None:
    from voxfusion.asr.faster_whisper import _resolve_device

    with patch("voxfusion.asr.faster_whisper.has_ctranslate2_cuda", return_value=False):
        device, compute = _resolve_device("cuda:1")
    assert device == "cpu"
    assert compute == "int8"
