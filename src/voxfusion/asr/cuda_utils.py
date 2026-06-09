"""Shared CUDA availability helpers for ASR backends."""

from __future__ import annotations

from voxfusion.logging import get_logger

log = get_logger(__name__)

# Minimum free GPU VRAM (MB) to consider CUDA usable.
# faster-whisper large-v3 needs ~3 GB in float16; leave headroom.
MIN_CUDA_FREE_MB = 4000


def _cuda_runtime_probe() -> bool:
    """Return True if CUDA is actually usable at runtime.

    ``torch.cuda.is_available()`` may return True even when the driver is
    incompatible with the CUDA toolkit version (e.g. driver 575 with
    CUDA 13.0).  Allocating a small tensor on ``"cuda"`` catches this.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        torch.zeros(1, device="cuda")
        return True
    except Exception as exc:
        log.warning("cuda.runtime_probe_failed", error=str(exc))
        return False


def has_cuda_vram() -> bool:
    """Return True if CUDA is available and has enough free VRAM."""
    try:
        import torch

        if not _cuda_runtime_probe():
            return False
        free_bytes, _total = torch.cuda.mem_get_info()
        free_mb = free_bytes // (1024 * 1024)
        if free_mb < MIN_CUDA_FREE_MB:
            log.warning(
                "asr.cuda_vram_low",
                free_mb=free_mb,
                required_mb=MIN_CUDA_FREE_MB,
            )
            return False
        return True
    except Exception:
        return False


def has_ctranslate2_cuda() -> bool:
    """Return True if CTranslate2 supports CUDA on this system."""
    try:
        import ctranslate2

        if not bool(ctranslate2.get_supported_compute_types("cuda")):
            return False
        return _cuda_runtime_probe()
    except Exception:
        return False


def select_best_gpu(min_free_mb: int = MIN_CUDA_FREE_MB) -> str | None:
    """Find the first GPU with enough free VRAM and return its device string.

    Scans all visible CUDA devices in order and returns ``"cuda:N"`` for
    the first GPU with at least *min_free_mb* megabytes free.  Returns
    ``None`` when no GPU qualifies or CUDA is unavailable.
    """
    try:
        import torch

        if not _cuda_runtime_probe():
            return None

        device_count = torch.cuda.device_count()
        for device_id in range(device_count):
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_id)
            free_mb = free_bytes // (1024 * 1024)
            total_mb = total_bytes // (1024 * 1024)
            name = torch.cuda.get_device_name(device_id)
            if free_mb >= min_free_mb:
                log.info(
                    "cuda.gpu_selected",
                    device=f"cuda:{device_id}",
                    gpu_name=name,
                    free_mb=free_mb,
                    total_mb=total_mb,
                    index=device_id,
                )
                return f"cuda:{device_id}"

        log.warning(
            "cuda.no_gpu_with_enough_vram",
            devices_scanned=device_count,
            required_mb=min_free_mb,
        )
        return None
    except Exception:
        return None
