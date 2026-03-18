"""ASR engine factory with automatic backend selection.

Detection priority:
1. CUDA  — ``FasterWhisperEngine`` with device='cuda' (NVIDIA GPU)
2. OpenVINO — ``OpenVINOWhisperEngine`` (Intel CPU/GPU via optimum-intel)
3. CPU   — ``FasterWhisperEngine`` with device='cpu' (baseline)

Usage::

    engine, backend_name = create_asr_engine(config.asr)
    # backend_name is one of: 'cuda', 'openvino', 'cpu'
"""

from __future__ import annotations

from voxfusion.config.models import ASRConfig
from voxfusion.logging import get_logger

log = get_logger(__name__)


def _has_cuda() -> bool:
    """Return True if a CUDA-capable GPU is accessible via CTranslate2."""
    try:
        import ctranslate2
        types = ctranslate2.get_supported_compute_types("cuda")
        return bool(types)
    except Exception:
        return False


def _has_openvino() -> bool:
    """Return True if optimum-intel with OpenVINO backend is importable."""
    try:
        from optimum.intel.openvino import OVModelForSpeechSeq2Seq  # noqa: F401
        return True
    except ImportError:
        return False


def create_asr_engine(
    config: ASRConfig | None = None,
    *,
    prefer_openvino: bool = True,
) -> tuple[object, str]:
    """Create the best available ASR engine and return ``(engine, backend_name)``.

    Args:
        config: ASR configuration.  Defaults to ``ASRConfig()``.
        prefer_openvino: When ``True`` (default), try OpenVINO before CPU.

    Returns:
        A 2-tuple of ``(engine_instance, backend_name)`` where
        *backend_name* is ``'cuda'``, ``'openvino'``, or ``'cpu'``.
    """
    cfg = config or ASRConfig()

    if cfg.engine == "gigaam":
        from voxfusion.asr.gigaam_engine import GigaAMCTCEngine

        log.info("asr_factory.selected", backend="gigaam")
        return GigaAMCTCEngine(cfg), "gigaam"

    if cfg.engine == "parakeet":
        from voxfusion.asr.parakeet_engine import ParakeetASREngine

        log.info("asr_factory.selected", backend="parakeet")
        return ParakeetASREngine(cfg), "parakeet"

    if cfg.engine == "breeze":
        from voxfusion.asr.breeze_engine import BreezeASREngine

        log.info("asr_factory.selected", backend="breeze")
        return BreezeASREngine(cfg), "breeze"

    # ── 1. CUDA (NVIDIA) ────────────────────────────────────────────────────
    if _has_cuda():
        from voxfusion.asr.faster_whisper import FasterWhisperEngine
        cuda_cfg = ASRConfig(**{**cfg.model_dump(), "device": "cuda", "compute_type": "float16"})
        log.info("asr_factory.selected", backend="cuda")
        return FasterWhisperEngine(cuda_cfg), "cuda"

    # ── 2. OpenVINO (Intel) ─────────────────────────────────────────────────
    if prefer_openvino and _has_openvino():
        from voxfusion.asr.openvino_engine import OpenVINOWhisperEngine
        log.info("asr_factory.selected", backend="openvino")
        return OpenVINOWhisperEngine(cfg), "openvino"

    # ── 3. CPU (baseline) ───────────────────────────────────────────────────
    from voxfusion.asr.faster_whisper import FasterWhisperEngine
    log.info("asr_factory.selected", backend="cpu")
    return FasterWhisperEngine(cfg), "cpu"
