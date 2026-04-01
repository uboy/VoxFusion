"""Regression tests for pyannote data in frozen bundles."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_build_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "build_binaries.py"
    spec = importlib.util.spec_from_file_location("voxfusion_build_binaries_pyannote_test", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pyannote_data_entries_include_telemetry_config(tmp_path: Path, monkeypatch) -> None:
    module = _load_build_module()
    telemetry_dir = tmp_path / "pyannote" / "audio" / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    init_path = telemetry_dir / "__init__.py"
    init_path.write_text("", encoding="utf-8")
    config_path = telemetry_dir / "config.yaml"
    config_path.write_text("metrics_enabled: false\n", encoding="utf-8")

    fake_spec = type("Spec", (), {"origin": str(init_path)})()
    monkeypatch.setattr(
        module.importlib.util,
        "find_spec",
        lambda name: fake_spec if name == "pyannote.audio.telemetry" else None,
    )

    entries = module._pyannote_data_entries()

    assert entries == [f"{config_path}{module.os.pathsep}pyannote/audio/telemetry"]


def test_collect_all_packages_includes_pyannote_audio(monkeypatch) -> None:
    module = _load_build_module()

    monkeypatch.setattr(
        module,
        "_is_installed",
        lambda package: package in {"pyannote.audio", "pyaudiowpatch"},
    )

    packages = module._collect_all_packages(backends={"whisper"})

    assert "pyannote.audio" in packages


def test_hidden_imports_include_dynamic_speaker_counter() -> None:
    module = _load_build_module()

    hidden_imports = module._hidden_imports(backends={"whisper"})

    assert "voxfusion.diarization.speaker_counter" in hidden_imports


def test_package_file_data_entry_targets_package_directory(tmp_path: Path, monkeypatch) -> None:
    module = _load_build_module()
    package_dir = tmp_path / "lightning_fabric"
    package_dir.mkdir(parents=True, exist_ok=True)
    init_path = package_dir / "__init__.py"
    init_path.write_text("", encoding="utf-8")
    version_info = package_dir / "version.info"
    version_info.write_text("1.0.0\n", encoding="utf-8")

    fake_spec = type("Spec", (), {"origin": str(init_path)})()
    monkeypatch.setattr(
        module.importlib.util,
        "find_spec",
        lambda name: fake_spec if name == "lightning_fabric" else None,
    )

    entry = module._package_file_data_entry("lightning_fabric", "version.info")

    assert entry == f"{version_info}{module.os.pathsep}lightning_fabric"


def test_find_ffmpeg_binary_returns_absolute_path(tmp_path: Path, monkeypatch) -> None:
    module = _load_build_module()
    ffmpeg_path = tmp_path / "vendor" / "ffmpeg.exe"
    ffmpeg_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_path.write_bytes(b"ffmpeg")

    monkeypatch.setattr(module.shutil, "which", lambda _name: str(ffmpeg_path.relative_to(tmp_path)))
    monkeypatch.chdir(tmp_path)

    found = module._find_ffmpeg_binary()

    assert found == ffmpeg_path.resolve()
