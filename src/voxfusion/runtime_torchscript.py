"""TorchScript fallbacks for frozen/runtime-restricted environments."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager


def should_use_torchscript_source_fallback(torch_module: object | None = None) -> bool:
    """Return True when runtime conditions justify the TorchScript source shim."""
    forced = os.environ.get("VOXFUSION_FORCE_TORCHSCRIPT_SOURCE_FALLBACK", "").strip().lower()
    if forced in {"1", "true", "yes", "on"}:
        return True

    disabled = os.environ.get("VOXFUSION_DISABLE_TORCHSCRIPT_SOURCE_FALLBACK", "").strip().lower()
    if disabled in {"1", "true", "yes", "on"}:
        return False

    if getattr(sys, "frozen", False) or getattr(sys, "_MEIPASS", None):
        return True

    torch_file = getattr(torch_module, "__file__", "")
    return not bool(str(torch_file).strip())


def install_torchscript_source_fallback(torch_module: object) -> None:
    """Return original objects when TorchScript source inspection is unavailable."""
    jit = getattr(torch_module, "jit", None)
    if jit is None:
        return
    original_script = getattr(jit, "script", None)
    if original_script is None or getattr(original_script, "_voxfusion_safe_wrapper", False):
        return

    def _safe_script(obj: object, *args: object, **kwargs: object) -> object:
        try:
            return original_script(obj, *args, **kwargs)
        except (OSError, RuntimeError) as exc:
            if "requires source access" not in str(exc).lower():
                raise
            return obj

    setattr(_safe_script, "_voxfusion_safe_wrapper", True)
    jit.script = _safe_script  # type: ignore[assignment]


@contextmanager
def temporary_torchscript_source_fallback(torch_module: object):
    """Apply the fallback only for the current block and then restore TorchScript."""
    jit = getattr(torch_module, "jit", None)
    original_script = getattr(jit, "script", None)
    if jit is None or original_script is None or not should_use_torchscript_source_fallback(torch_module):
        yield
        return
    install_torchscript_source_fallback(torch_module)
    try:
        yield
    finally:
        current_script = getattr(jit, "script", None)
        if current_script is not None and getattr(current_script, "_voxfusion_safe_wrapper", False):
            jit.script = original_script  # type: ignore[assignment]
