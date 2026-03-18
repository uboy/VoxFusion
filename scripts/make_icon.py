"""Generate VoxFusion application icons.

Requires Pillow:
    pip install Pillow

Outputs:
    assets/voxfusion.ico   — Windows (multi-size: 16, 32, 48, 64, 128, 256)
    assets/voxfusion.png   — Linux / generic (256x256)
    assets/voxfusion.icns  — macOS (if Pillow ICNS support is available)
"""

from __future__ import annotations

import sys
from pathlib import Path


ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"

# Design constants
BG_COLOR = (22, 22, 40, 255)       # deep navy
ACCENT_COLOR = (0, 200, 230, 255)  # cyan

# Waveform bar heights (relative, centre bar = 1.0)
_BAR_HEIGHTS = [0.30, 0.58, 0.82, 1.00, 0.82, 0.58, 0.30]


def _require_pillow() -> tuple:
    try:
        from PIL import Image, ImageDraw  # type: ignore[import-not-found]
        return Image, ImageDraw
    except ImportError:
        sys.exit(
            "Pillow is not installed.\n"
            "Run:  pip install Pillow\n"
            "Then re-run this script."
        )


def make_frame(size: int) -> "Image":  # type: ignore[name-defined]
    """Draw a single VoxFusion icon frame at *size* x *size* pixels."""
    Image, ImageDraw = _require_pillow()

    img = Image.new("RGBA", (size, size), BG_COLOR)
    draw = ImageDraw.Draw(img)

    n = len(_BAR_HEIGHTS)
    margin = size * 0.12
    usable_w = size - 2 * margin
    # Each bar + gap of equal width → bar_w = usable_w / (n + n-1) = usable_w / (2n-1)
    bar_w = usable_w / (2 * n - 1)
    max_bar_h = (size - 2 * margin) * 0.78
    corner_r = max(1, int(bar_w * 0.35))

    for i, rel_h in enumerate(_BAR_HEIGHTS):
        bar_h = max(2, int(max_bar_h * rel_h))
        x0 = margin + i * 2 * bar_w
        x1 = x0 + bar_w
        y0 = (size - bar_h) / 2
        y1 = y0 + bar_h
        draw.rounded_rectangle([x0, y0, x1, y1], radius=corner_r, fill=ACCENT_COLOR)

    return img


def main() -> None:
    Image, _ImageDraw = _require_pillow()

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # Render at full 256px — Pillow's ICO plugin will downscale automatically
    base = make_frame(256)

    # --- Windows ICO (multi-size: Pillow resizes from the base frame) ---
    ico_path = ASSETS_DIR / "voxfusion.ico"
    base.save(
        ico_path,
        format="ICO",
        sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    print(f"Created: {ico_path}  ({ico_path.stat().st_size} bytes)")

    # --- Linux PNG (256px) ---
    png_path = ASSETS_DIR / "voxfusion.png"
    base.save(png_path, format="PNG")
    print(f"Created: {png_path}  ({png_path.stat().st_size} bytes)")

    # --- macOS ICNS ---
    icns_path = ASSETS_DIR / "voxfusion.icns"
    try:
        base.save(icns_path, format="ICNS")
        print(f"Created: {icns_path}  ({icns_path.stat().st_size} bytes)")
    except Exception as exc:
        print(f"ICNS skipped ({exc}). On macOS use iconutil or install Pillow with ICNS support.")

    print("\nDone. Commit assets/ and run 'python scripts/build_binaries.py' to build with the icon.")


if __name__ == "__main__":
    main()
