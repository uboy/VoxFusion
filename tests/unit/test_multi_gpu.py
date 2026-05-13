"""Tests for multi-GPU VRAM detection and device selection."""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch


def _make_torch_mock(
    *,
    cuda_available: bool = True,
    device_count: int = 1,
    free_mb_per_device: list[int] | None = None,
    total_mb_per_device: list[int] | None = None,
    device_names: list[str] | None = None,
) -> types.ModuleType:
    """Build a fake torch module with configurable CUDA devices."""
    torch_mock = MagicMock()
    torch_mock.cuda.is_available.return_value = cuda_available
    torch_mock.cuda.device_count.return_value = device_count

    if free_mb_per_device is None:
        free_mb_per_device = [10000] * device_count
    if total_mb_per_device is None:
        total_mb_per_device = [32768] * device_count
    if device_names is None:
        device_names = [f"GPU-{i}" for i in range(device_count)]

    def mem_get_info(device_id=None):
        idx = device_id or 0
        free = free_mb_per_device[idx] * 1024 * 1024
        total = total_mb_per_device[idx] * 1024 * 1024
        return (free, total)

    torch_mock.cuda.mem_get_info = mem_get_info
    torch_mock.cuda.get_device_name = lambda idx: device_names[idx]
    return torch_mock


# ── select_best_gpu ──────────────────────────────────────────────────


def test_select_best_gpu_returns_first_gpu_with_enough_vram() -> None:
    from voxfusion.asr.cuda_utils import select_best_gpu

    torch_mock = _make_torch_mock(
        device_count=3,
        free_mb_per_device=[2000, 10000, 10000],
    )
    with patch.dict("sys.modules", {"torch": torch_mock}):
        result = select_best_gpu()
    assert result == "cuda:1"


def test_select_best_gpu_returns_none_when_all_gpus_full() -> None:
    from voxfusion.asr.cuda_utils import select_best_gpu

    torch_mock = _make_torch_mock(
        device_count=3,
        free_mb_per_device=[2000, 1000, 3000],
    )
    with patch.dict("sys.modules", {"torch": torch_mock}):
        result = select_best_gpu()
    assert result is None


def test_select_best_gpu_returns_none_when_no_cuda() -> None:
    from voxfusion.asr.cuda_utils import select_best_gpu

    torch_mock = _make_torch_mock(cuda_available=False)
    with patch.dict("sys.modules", {"torch": torch_mock}):
        result = select_best_gpu()
    assert result is None


def test_select_best_gpu_handles_exception() -> None:
    from voxfusion.asr.cuda_utils import select_best_gpu

    torch_mock = MagicMock()
    torch_mock.cuda.is_available.side_effect = RuntimeError("no driver")
    with patch.dict("sys.modules", {"torch": torch_mock}):
        result = select_best_gpu()
    assert result is None


def test_select_best_gpu_single_gpu_with_vram() -> None:
    from voxfusion.asr.cuda_utils import select_best_gpu

    torch_mock = _make_torch_mock(device_count=1, free_mb_per_device=[8000])
    with patch.dict("sys.modules", {"torch": torch_mock}):
        result = select_best_gpu()
    assert result == "cuda:0"


# ── _resolve_device with multi-GPU ──────────────────────────────────


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
