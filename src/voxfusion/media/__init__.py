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

__all__ = [
    "NEEDS_EXTRACTION_EXTENSIONS",
    "build_linear_overlay_filter_graph",
    "detect_best_h264_encoder",
    "extract_audio",
    "extract_audio_async",
    "needs_extraction",
    "recommended_encoder_workers",
]
