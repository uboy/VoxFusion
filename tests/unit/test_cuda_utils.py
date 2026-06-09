"""Tests for CUDA utility helpers."""

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
    zeros_succeeds: bool = True,
) -> types.ModuleType:
    torch_mock = MagicMock()
    torch_mock.cuda.is_available.return_value = cuda_available
    torch_mock.cuda.device_count.return_value = device_count
    torch_mock.zeros = MagicMock()
    if not zeros_succeeds:
        torch_mock.zeros.side_effect = RuntimeError("CUDA error")

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


# ── has_cuda_vram ────────────────────────────────────────────────────


def test_has_cuda_vram_true_when_enough_memory() -> None:
    from voxfusion.asr.cuda_utils import has_cuda_vram

    torch_mock = _make_torch_mock(free_mb_per_device=[8000])
    with patch.dict("sys.modules", {"torch": torch_mock}):
        assert has_cuda_vram() is True


def test_has_cuda_vram_false_when_low_memory() -> None:
    from voxfusion.asr.cuda_utils import has_cuda_vram

    torch_mock = _make_torch_mock(free_mb_per_device=[2000])
    with patch.dict("sys.modules", {"torch": torch_mock}):
        assert has_cuda_vram() is False


def test_has_cuda_vram_false_when_no_cuda() -> None:
    from voxfusion.asr.cuda_utils import has_cuda_vram

    torch_mock = _make_torch_mock(cuda_available=False)
    with patch.dict("sys.modules", {"torch": torch_mock}):
        assert has_cuda_vram() is False


def test_has_cuda_vram_handles_exception() -> None:
    from voxfusion.asr.cuda_utils import has_cuda_vram

    torch_mock = MagicMock()
    torch_mock.cuda.is_available.side_effect = RuntimeError("no driver")
    with patch.dict("sys.modules", {"torch": torch_mock}):
        assert has_cuda_vram() is False


# ── has_ctranslate2_cuda ─────────────────────────────────────────────


def test_has_ctranslate2_cuda_true() -> None:
    from voxfusion.asr.cuda_utils import has_ctranslate2_cuda

    ct2 = MagicMock()
    ct2.get_supported_compute_types.return_value = ["float16", "int8"]
    ct2.get_cuda_device_count.return_value = 1
    with (
        patch.dict("sys.modules", {"ctranslate2": ct2}),
        patch("ctypes.CDLL", return_value=MagicMock()),
    ):
        assert has_ctranslate2_cuda() is True


def test_has_ctranslate2_cuda_false_when_no_gpu_count() -> None:
    from voxfusion.asr.cuda_utils import has_ctranslate2_cuda

    ct2 = MagicMock()
    ct2.get_supported_compute_types.return_value = ["float16", "int8"]
    ct2.get_cuda_device_count.return_value = 0
    with patch.dict("sys.modules", {"ctranslate2": ct2}):
        assert has_ctranslate2_cuda() is False


def test_has_ctranslate2_cuda_false_when_no_libcublas() -> None:
    from voxfusion.asr.cuda_utils import has_ctranslate2_cuda

    ct2 = MagicMock()
    ct2.get_supported_compute_types.return_value = ["float16", "int8"]
    ct2.get_cuda_device_count.return_value = 1
    with (
        patch.dict("sys.modules", {"ctranslate2": ct2}),
        patch("ctypes.CDLL", side_effect=OSError("not found")),
    ):
        assert has_ctranslate2_cuda() is False


def test_has_ctranslate2_cuda_false_when_no_cuda_types() -> None:
    from voxfusion.asr.cuda_utils import has_ctranslate2_cuda

    ct2 = MagicMock()
    ct2.get_supported_compute_types.return_value = []
    with patch.dict("sys.modules", {"ctranslate2": ct2}):
        assert has_ctranslate2_cuda() is False


def test_has_ctranslate2_cuda_handles_import_error() -> None:
    from voxfusion.asr.cuda_utils import has_ctranslate2_cuda

    with patch.dict("sys.modules", {"ctranslate2": None}):
        assert has_ctranslate2_cuda() is False


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


def test_select_best_gpu_custom_threshold() -> None:
    from voxfusion.asr.cuda_utils import select_best_gpu

    torch_mock = _make_torch_mock(device_count=1, free_mb_per_device=[5000])
    with patch.dict("sys.modules", {"torch": torch_mock}):
        assert select_best_gpu(min_free_mb=6000) is None
        assert select_best_gpu(min_free_mb=4000) == "cuda:0"
