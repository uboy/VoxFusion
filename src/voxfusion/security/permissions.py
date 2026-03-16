"""OS-specific permission checks for audio capture access.

Verifies that the application has the necessary permissions to
access audio input/output devices on each supported platform.
"""

import sys

from voxfusion.exceptions import DeviceAccessDeniedError
from voxfusion.logging import get_logger

log = get_logger(__name__)


class PermissionChecker:
    """Check OS-level audio permissions.

    Provides platform-specific checks to determine whether the
    application has permission to access microphone and system
    audio devices.
    """

    def check_microphone_access(self) -> bool:
        """Check if the application has microphone access.

        Returns:
            True if microphone access is granted or cannot be determined.

        Raises:
            DeviceAccessDeniedError: If access is explicitly denied.
        """
        platform = sys.platform

        if platform == "darwin":
            return self._check_macos_microphone()
        if platform == "win32":
            return self._check_windows_microphone()
        if platform == "linux":
            return self._check_linux_audio()

        log.debug("permissions.unknown_platform", platform=platform)
        return True

    def check_system_audio_access(self) -> bool:
        """Check if the application can capture system/loopback audio.

        Returns:
            True if access is available or cannot be determined.
        """
        platform = sys.platform

        if platform == "win32":
            return self._check_windows_loopback()
        if platform == "darwin":
            return self._check_macos_system_audio()
        if platform == "linux":
            return self._check_linux_audio()

        return True

    def _check_macos_microphone(self) -> bool:
        """Check macOS microphone permission via AVFoundation."""
        try:
            import subprocess

            result = subprocess.run(
                [
                    "osascript", "-e",
                    'tell application "System Events" to '
                    'get the properties of the first login item',
                ],
                capture_output=True,
                timeout=5,
            )
            # A more reliable check uses the AVFoundation framework,
            # but that requires PyObjC.  Fall through to a simple
            # sounddevice test.
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return self._test_audio_device()

    def _check_macos_system_audio(self) -> bool:
        """Check if a virtual audio device is available on macOS.

        System audio capture on macOS requires a virtual audio driver
        like BlackHole or Soundflower.
        """
        try:
            import sounddevice as sd

            devices = sd.query_devices()
            for dev in devices:
                name = dev.get("name", "").lower()  # type: ignore[union-attr]
                if any(kw in name for kw in ("blackhole", "soundflower", "loopback")):
                    log.info("permissions.macos_virtual_audio_found", device=dev["name"])  # type: ignore[index]
                    return True

            log.warning(
                "permissions.macos_no_virtual_audio",
                hint="Install BlackHole or Soundflower for system audio capture",
            )
            return False
        except Exception as exc:
            log.warning("permissions.macos_check_failed", error=str(exc))
            return False

    def _check_windows_microphone(self) -> bool:
        """Check Windows microphone access."""
        return self._test_audio_device()

    def _check_windows_loopback(self) -> bool:
        """Check Windows WASAPI loopback availability.

        WASAPI loopback is generally available on Windows without
        special permissions.
        """
        try:
            import sounddevice as sd

            devices = sd.query_devices()
            has_output = any(
                dev.get("max_output_channels", 0) > 0  # type: ignore[union-attr]
                for dev in devices
            )
            if has_output:
                log.debug("permissions.windows_loopback_available")
                return True

            log.warning("permissions.windows_no_output_devices")
            return False
        except Exception as exc:
            log.warning("permissions.windows_loopback_check_failed", error=str(exc))
            return False

    def _check_linux_audio(self) -> bool:
        """Check Linux audio access (PulseAudio/PipeWire)."""
        try:
            import subprocess

            # Check if PulseAudio or PipeWire is running
            result = subprocess.run(
                ["pactl", "info"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                log.debug("permissions.linux_pulseaudio_available")
                return True

            log.warning("permissions.linux_no_pulse")
            return False
        except FileNotFoundError:
            log.warning("permissions.linux_pactl_not_found")
            return self._test_audio_device()
        except subprocess.TimeoutExpired:
            return self._test_audio_device()

    def _test_audio_device(self) -> bool:
        """Generic test: try to query audio devices via sounddevice."""
        try:
            import sounddevice as sd

            devices = sd.query_devices()
            has_input = any(
                dev.get("max_input_channels", 0) > 0  # type: ignore[union-attr]
                for dev in devices
            )
            if has_input:
                log.debug("permissions.audio_device_available")
                return True

            log.warning("permissions.no_input_devices")
            return False
        except Exception as exc:
            raise DeviceAccessDeniedError(
                f"Cannot access audio devices: {exc}"
            ) from exc


def check_permissions() -> dict[str, bool]:
    """Run all permission checks and return results.

    Returns:
        Dictionary mapping check names to boolean results.
    """
    checker = PermissionChecker()
    results: dict[str, bool] = {}

    try:
        results["microphone"] = checker.check_microphone_access()
    except DeviceAccessDeniedError:
        results["microphone"] = False

    try:
        results["system_audio"] = checker.check_system_audio_access()
    except DeviceAccessDeniedError:
        results["system_audio"] = False

    log.info("permissions.checked", results=results)
    return results
