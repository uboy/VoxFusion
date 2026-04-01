"""Tests for TorchScript fallback mode selection."""

from __future__ import annotations

import sys
import types

from voxfusion.runtime_torchscript import should_use_torchscript_source_fallback


def test_should_use_torchscript_source_fallback_for_fake_test_module() -> None:
    fake_torch = types.ModuleType("torch")
    assert should_use_torchscript_source_fallback(fake_torch)


def test_should_not_use_torchscript_source_fallback_for_source_runtime(
    monkeypatch,
) -> None:
    fake_torch = types.ModuleType("torch")
    fake_torch.__file__ = "C:/python/Lib/site-packages/torch/__init__.py"

    monkeypatch.setattr(sys, "frozen", False, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", None, raising=False)
    monkeypatch.delenv("VOXFUSION_FORCE_TORCHSCRIPT_SOURCE_FALLBACK", raising=False)
    monkeypatch.delenv("VOXFUSION_DISABLE_TORCHSCRIPT_SOURCE_FALLBACK", raising=False)

    assert not should_use_torchscript_source_fallback(fake_torch)


def test_should_use_torchscript_source_fallback_when_forced(
    monkeypatch,
) -> None:
    fake_torch = types.ModuleType("torch")
    fake_torch.__file__ = "C:/python/Lib/site-packages/torch/__init__.py"

    monkeypatch.setenv("VOXFUSION_FORCE_TORCHSCRIPT_SOURCE_FALLBACK", "1")

    assert should_use_torchscript_source_fallback(fake_torch)
