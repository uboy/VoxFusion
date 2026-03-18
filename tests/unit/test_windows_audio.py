"""Unit tests for shared Windows audio device handling."""

import asyncio
import sys

from voxfusion.capture.wasapi import PyAudioLoopbackCapture, WASAPICapture
from voxfusion.capture.windows_audio import parse_windows_device_id
from voxfusion.capture.windows_factory import create_windows_capture_source


def test_parse_windows_device_id_supports_explicit_backend_prefix() -> None:
    assert parse_windows_device_id("pa:12") == ("pa", 12)
    assert parse_windows_device_id("sd:7") == ("sd", 7)


def test_parse_windows_device_id_treats_plain_int_as_default_backend() -> None:
    assert parse_windows_device_id(5, default_backend="sd") == ("sd", 5)
    assert parse_windows_device_id("9", default_backend="pa") == ("pa", 9)


def test_create_windows_capture_source_for_system_forwards_explicit_loopback_device(monkeypatch) -> None:
    from voxfusion.config.models import CaptureConfig

    seen: dict[str, object] = {}

    class StubLoopback:
        def __init__(self, *, device_id=None, config=None) -> None:
            seen["device_id"] = device_id
            seen["config"] = config

    monkeypatch.setattr("voxfusion.capture.windows_factory.RobustLoopbackCapture", StubLoopback)

    source = create_windows_capture_source(
        "system",
        CaptureConfig(),
        system_device_id="pa:42",
    )

    assert isinstance(source, StubLoopback)
    assert seen["device_id"] == "pa:42"


def test_create_windows_capture_source_rejects_pyaudio_device_for_microphone() -> None:
    from voxfusion.config.models import CaptureConfig

    try:
        create_windows_capture_source(
            "microphone",
            CaptureConfig(),
            microphone_device_id="pa:3",
        )
    except ValueError as exc:
        assert "Microphone capture requires a sounddevice/WASAPI device" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid microphone device backend.")


def test_pyaudio_loopback_capture_retries_channel_count(monkeypatch) -> None:
    from voxfusion.config.models import CaptureConfig

    attempts: list[int] = []

    class StubStream:
        def start_stream(self) -> None:
            return None

    class StubPyAudioInstance:
        def get_device_info_by_index(self, index: int) -> dict[str, object]:
            return {
                "index": index,
                "name": "Loopback Device",
                "defaultSampleRate": 48000,
                "maxInputChannels": 2,
            }

        def open(self, *, channels: int, **kwargs) -> StubStream:
            attempts.append(channels)
            if channels == 2:
                raise OSError("Errno -9998 Invalid number of channels")
            return StubStream()

        def terminate(self) -> None:
            return None

    class StubPyAudioModule:
        paFloat32 = object()

        @staticmethod
        def PyAudio() -> StubPyAudioInstance:
            return StubPyAudioInstance()

    monkeypatch.setitem(sys.modules, "pyaudiowpatch", StubPyAudioModule)

    capture = PyAudioLoopbackCapture(device_index=10, config=CaptureConfig(channels=2))
    asyncio.run(capture.start())

    assert attempts == [2, 1]
    assert capture.channels == 1


def test_wasapi_loopback_retries_after_invalid_channel_error(monkeypatch) -> None:
    from voxfusion.config.models import CaptureConfig

    attempts: list[int | None] = []

    class StubStream:
        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def close(self) -> None:
            return None

    class StubWasapiSettings:
        def __init__(self, *args, **kwargs) -> None:
            return None

    class StubSoundDeviceModule:
        default = type("Default", (), {"device": (0, 5)})()

        @staticmethod
        def query_hostapis() -> list[dict[str, object]]:
            return [{
                "name": "Windows WASAPI",
                "default_input_device": 0,
                "default_output_device": 5,
                "devices": [5],
            }]

        @staticmethod
        def query_devices(index: int | None = None):
            devices = [
                {"name": "Mic", "hostapi": 0, "max_input_channels": 2, "max_output_channels": 0, "default_samplerate": 48000},
                {"name": "Speaker", "hostapi": 0, "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 48000},
                {"name": "Speaker2", "hostapi": 0, "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 48000},
                {"name": "Speaker3", "hostapi": 0, "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 48000},
                {"name": "Speaker4", "hostapi": 0, "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 48000},
                {"name": "LoopbackTarget", "hostapi": 0, "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 48000},
            ]
            if index is None:
                return devices
            return devices[index]

        WasapiSettings = StubWasapiSettings

        @staticmethod
        def InputStream(**kwargs):
            attempts.append(kwargs.get("channels"))
            if kwargs.get("channels") == 2:
                raise OSError("Error opening InputStream: Invalid number of channels [PaErrorCode -9998]")
            return StubStream()

    monkeypatch.setitem(sys.modules, "sounddevice", StubSoundDeviceModule)
    monkeypatch.setattr(sys, "platform", "win32")

    capture = WASAPICapture(device_index=5, loopback=True, config=CaptureConfig(channels=2))
    asyncio.run(capture.start())

    assert attempts[:2] == [2, 1]
    assert capture.channels == 1
