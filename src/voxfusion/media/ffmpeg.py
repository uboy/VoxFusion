"""FFmpeg helpers for encoder selection and filter graph generation."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence


ENCODER_PRIORITY: tuple[str, ...] = ("h264_nvenc", "h264_qsv", "libx264")


def detect_best_h264_encoder(
    ffmpeg_binary: str = "ffmpeg",
    timeout_seconds: float = 8.0,
) -> str:
    """Detect the first available H.264 encoder via dry-run checks."""
    for encoder in ENCODER_PRIORITY:
        if _dry_run_encoder(ffmpeg_binary, encoder, timeout_seconds):
            return encoder
    return "libx264"


def _dry_run_encoder(
    ffmpeg_binary: str,
    encoder: str,
    timeout_seconds: float,
) -> bool:
    cmd = [
        ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=320x180:r=30",
        "-t",
        "0.1",
        "-an",
        "-c:v",
        encoder,
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def recommended_encoder_workers(
    use_hardware_encoder: bool,
    cpu_count: int | None = None,
) -> int:
    """Return recommended worker count with hardware-aware limits."""
    resolved_cpu = cpu_count or (os.cpu_count() or 1)
    if use_hardware_encoder:
        return 2 if resolved_cpu < 8 else 3
    return max(1, min(8, resolved_cpu - 1))


def build_linear_overlay_filter_graph(
    layers: Sequence[str],
    *,
    size: str = "1920x1080",
    background_color: str = "black",
    output_label: str = "vout",
) -> str:
    """Build a linear overlay graph that always initializes `[bg]` first."""
    graph_parts = [f"color=c={background_color}:s={size}[bg]"]

    if not layers:
        graph_parts.append(f"[bg]copy[{output_label}]")
        return ";".join(graph_parts)

    current = "bg"
    for index, layer in enumerate(layers):
        next_label = output_label if index == len(layers) - 1 else f"tmp{index}"
        graph_parts.append(f"[{current}][{layer}]overlay=shortest=1[{next_label}]")
        current = next_label

    return ";".join(graph_parts)
