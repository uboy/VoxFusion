"""GUI entry points for VoxFusion."""

def main() -> int:
    """Lazy GUI entrypoint to avoid importing `voxfusion.gui.main` at package import time."""
    from voxfusion.gui.main import main as _main

    return _main()

__all__ = ["main"]
