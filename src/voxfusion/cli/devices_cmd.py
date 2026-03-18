"""CLI command: voxfusion devices -- list available audio devices."""

import click

from voxfusion.cli.formatting import echo_table, echo_warning
from voxfusion.capture.windows_audio import (
    list_windows_loopback_devices,
    list_windows_microphone_devices,
)


@click.command("devices")
@click.option("--type", "-t", "device_type", type=click.Choice(["all", "input", "loopback"]),
              default="all", help="Filter by device type.")
@click.option("--diagnose", is_flag=True, default=False,
              help="Run loopback capture diagnostic and show what works.")
def devices(device_type: str, diagnose: bool) -> None:
    """List available audio capture devices."""
    rows: list[list[str]] = []
    if device_type in ("all", "input"):
        for dev in list_windows_microphone_devices():
            rows.append([dev.id, dev.name, "input", dev.backend, "WASAPI microphone"])
    if device_type in ("all", "loopback"):
        for dev in list_windows_loopback_devices():
            rows.append([dev.id, dev.name, "loopback", dev.backend, "Windows system audio"])

    if not rows:
        echo_warning("No audio devices found.")
    else:
        echo_table(
            headers=["ID", "Name", "Type", "Backend", "Purpose"],
            rows=rows,
        )

    if diagnose:
        _diagnose_loopback()


def _diagnose_loopback() -> None:
    """Test all system audio capture backends and print results."""
    import asyncio

    click.echo("\n=== System audio loopback diagnostic ===\n")

    # --- 1. sounddevice: WASAPI host API and output devices ---
    click.echo("[1] sounddevice / PortAudio")
    try:
        import sounddevice as sd
        hostapis = list(sd.query_hostapis())
        wasapi = next((h for h in hostapis if "wasapi" in h.get("name", "").lower()), None)
        if wasapi is None:
            click.echo("    WASAPI host API: NOT FOUND — audio driver may be outdated")
        else:
            click.echo(f"    WASAPI host API: OK (default_output={wasapi['default_output_device']})")
            wasapi_idx = hostapis.index(wasapi)
            out_devs = [
                (i, d) for i, d in enumerate(sd.query_devices())
                if d.get("hostapi") == wasapi_idx and d.get("max_output_channels", 0) > 0
            ]
            click.echo(f"    WASAPI output devices: {len(out_devs)}")
            for i, d in out_devs:
                click.echo(f"      [{i}] {d['name']}")
            if not out_devs:
                click.echo("    WARNING: no WASAPI output devices — loopback impossible via sounddevice")
    except Exception as exc:
        click.echo(f"    ERROR: {exc}")

    # --- 2. pyaudiowpatch loopback devices ---
    click.echo("\n[2] pyaudiowpatch (patched PortAudio)")
    try:
        import pyaudiowpatch as pyaudio
        pa = pyaudio.PyAudio()
        try:
            wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
            click.echo(f"    WASAPI available: YES (default_output={wasapi_info['defaultOutputDevice']})")
            loopback_devs = [
                pa.get_device_info_by_index(i)
                for i in range(pa.get_device_count())
                if pa.get_device_info_by_index(i).get("isLoopbackDevice")
                and pa.get_device_info_by_index(i).get("maxInputChannels", 0) > 0
            ]
            click.echo(f"    Loopback devices found: {len(loopback_devs)}")
            for d in loopback_devs:
                click.echo(f"      [{d['index']}] {d['name']}  ch={d['maxInputChannels']}  rate={d['defaultSampleRate']}")
            if not loopback_devs:
                click.echo("    WARNING: no loopback devices — pyaudiowpatch won't work on this machine")
        except Exception as exc:
            click.echo(f"    WASAPI query failed: {exc}")
        finally:
            pa.terminate()
    except ImportError:
        click.echo("    NOT INSTALLED — run: pip install pyaudiowpatch")
    except Exception as exc:
        click.echo(f"    ERROR loading pyaudiowpatch: {exc}")

    # --- 3. Live open test: try RobustLoopbackCapture for 1 second ---
    click.echo("\n[3] Live open test (1 second capture attempt)")
    async def _try_open() -> str:
        from voxfusion.capture.wasapi import RobustLoopbackCapture
        cap = RobustLoopbackCapture()
        try:
            await cap.start()
            backend = cap.device_name
            await asyncio.sleep(0.5)
            await cap.stop()
            return f"OK — backend: {backend}"
        except Exception as exc:
            return f"FAILED: {exc}"

    result = asyncio.run(_try_open())
    click.echo(f"    {result}")

    click.echo("\n=== End of diagnostic ===")
    if "FAILED" in result:
        click.echo(
            "\nSystem audio capture is not available on this machine.\n"
            "Solutions:\n"
            "  1. Enable Stereo Mix: Sound settings → Recording tab → right-click\n"
            "     → 'Show Disabled Devices' → right-click 'Stereo Mix' → Enable\n"
            "  2. Install VB-Audio Virtual Cable (free): https://vb-audio.com/Cable/\n"
            "     Then set 'CABLE Input' as default playback and use 'CABLE Output'\n"
            "     as the recording source in VoxFusion.\n"
        )
