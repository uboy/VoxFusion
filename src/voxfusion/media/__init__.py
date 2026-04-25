"""Media and FFmpeg helper utilities."""

from voxfusion.media.extractor import (
    NEEDS_EXTRACTION_EXTENSIONS,
    extract_audio,
    extract_audio_async,
    needs_extraction,
)
from voxfusion.media.ffmpeg import (
    build_linear_overlay_filter_graph,
    detect_best_h264_encoder,
    recommended_encoder_workers,
)
from voxfusion.media.runtime_ffmpeg import (
    activate_ffmpeg_runtime,
    find_ffmpeg,
    install_ffmpeg_local,
)

__all__ = [
    "NEEDS_EXTRACTION_EXTENSIONS",
    "activate_ffmpeg_runtime",
    "build_linear_overlay_filter_graph",
    "detect_best_h264_encoder",
    "extract_audio",
    "extract_audio_async",
    "find_ffmpeg",
    "install_ffmpeg_local",
    "needs_extraction",
    "recommended_encoder_workers",
]
