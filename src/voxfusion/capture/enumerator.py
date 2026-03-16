"""Platform-specific audio device enumeration.

Uses ``sounddevice`` to list available input and output devices.
Loopback detection is platform-dependent and may require additional
drivers (e.g. WASAPI loopback on Windows, PulseAudio monitor on Linux).
"""

import sys

from voxfusion.logging import get_logger
from voxfusion.models.audio import AudioDeviceInfo

log = get_logger(__name__)


class SoundDeviceEnumerator:
    """Enumerate audio devices using the sounddevice library."""

    def _query_devices(self) -> list[dict]:  # type: ignore[type-arg]
        """Query all devices from sounddevice."""
        try:
            import sounddevice as sd

            return list(sd.query_devices())
        except ImportError:
            log.warning("sounddevice not installed — device enumeration unavailable")
            return []
        except Exception as exc:
            log.warning("device_enumeration_failed", error=str(exc))
            return []

    def _to_device_info(self, index: int, dev: dict) -> AudioDeviceInfo:  # type: ignore[type-arg]
        """Convert a sounddevice device dict to AudioDeviceInfo."""
        max_in = dev.get("max_input_channels", 0)
        max_out = dev.get("max_output_channels", 0)

        if max_in > 0 and max_out == 0:
            device_type = "input"
        elif max_out > 0 and max_in == 0:
            device_type = "loopback"
        else:
            device_type = "input"

        return AudioDeviceInfo(
            id=f"sd:{index}",
            name=dev["name"],
            sample_rate=int(dev.get("default_samplerate", 44100)),
            channels=max_in if max_in > 0 else max_out,
            device_type=device_type,
            is_default=False,
            platform_id=str(index),
        )

    def list_input_devices(self) -> list[AudioDeviceInfo]:
        """List input (microphone) devices."""
        return [
            self._to_device_info(i, d)
            for i, d in enumerate(self._query_devices())
            if d.get("max_input_channels", 0) > 0
        ]

    def list_loopback_devices(self) -> list[AudioDeviceInfo]:
        """List loopback/output devices.

        On Windows with WASAPI, output devices can be used as loopback.
        On Linux, PulseAudio monitor sources appear as input devices.
        """
        return [
            self._to_device_info(i, d)
            for i, d in enumerate(self._query_devices())
            if d.get("max_output_channels", 0) > 0
            and d.get("max_input_channels", 0) == 0
        ]

    def get_default_input_device(self) -> AudioDeviceInfo | None:
        """Return the system default input device."""
        try:
            import sounddevice as sd

            idx = sd.default.device[0]
            if idx is None or idx < 0:
                return None
            dev = sd.query_devices(idx)
            return self._to_device_info(idx, dev)
        except Exception:
            return None

    def get_default_loopback_device(self) -> AudioDeviceInfo | None:
        """Return the system default output device (for loopback)."""
        try:
            import sounddevice as sd

            idx = sd.default.device[1]
            if idx is None or idx < 0:
                return None
            dev = sd.query_devices(idx)
            return self._to_device_info(idx, dev)
        except Exception:
            return None
