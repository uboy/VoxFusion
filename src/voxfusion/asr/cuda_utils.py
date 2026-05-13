"""Shared CUDA availability helpers for ASR backends."""

from __future__ import annotations

from voxfusion.logging import get_logger

log = get_logger(__name__)

# Minimum free GPU VRAM (MB) to consider CUDA usable.
# faster-whisper large-v3 needs ~3 GB in float16; leave headroom.
MIN_CUDA_FREE_MB = 4000


def has_cuda_vram() -> bool:
    """Return True if CUDA is available and has enough free VRAM."""
    try:
        import torch

        if not torch.cuda.is_available():
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

        return bool(ctranslate2.get_supported_compute_types("cuda"))
    except Exception:
        return False
