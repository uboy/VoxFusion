"""Tests for build_binaries import probing helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_build_binaries_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "build_binaries.py"
    spec = importlib.util.spec_from_file_location("build_binaries_under_test", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_is_installed_returns_false_when_find_spec_raises(monkeypatch) -> None:
    build_binaries = _load_build_binaries_module()

    def _raise(_name: str):
        raise ModuleNotFoundError("No module named 'pyannote'")

    monkeypatch.setattr(importlib.util, "find_spec", _raise)

    assert build_binaries._is_installed("pyannote.audio") is False


def test_pyannote_data_entries_returns_empty_when_find_spec_raises(monkeypatch) -> None:
    build_binaries = _load_build_binaries_module()

    def _raise(_name: str):
        raise ModuleNotFoundError("No module named 'pyannote'")

    monkeypatch.setattr(importlib.util, "find_spec", _raise)

    assert build_binaries._pyannote_data_entries() == []
