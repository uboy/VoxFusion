"""CLI command: voxfusion devices -- list available audio devices."""

import click

from voxfusion.cli.formatting import echo_table, echo_warning


@click.command("devices")
@click.option("--type", "-t", "device_type", type=click.Choice(["all", "input", "loopback"]),
              default="all", help="Filter by device type.")
@click.option("--diagnose", is_flag=True, default=False,
              help="Run loopback capture diagnostic and show what works.")
def devices(device_type: str, diagnose: bool) -> None:
    """List available audio capture devices."""
    try:
        import sounddevice as sd
    except ImportError:
        raise click.ClickException(
            "sounddevice is not installed. Install with: pip install sounddevice"
        )

    try:
        all_devices = sd.query_devices()
    except Exception as exc:
        raise click.ClickException(f"Failed to query audio devices: {exc}") from exc

    rows: list[list[str]] = []
    for i, dev in enumerate(all_devices):
        max_in = dev.get("max_input_channels", 0)
        max_out = dev.get("max_output_channels", 0)

        if device_type == "input" and max_in == 0:
            continue
        if device_type == "loopback" and max_out == 0:
            continue

        dtype = "input" if max_in > 0 and max_out == 0 else (
            "output" if max_out > 0 and max_in == 0 else "input/output"
        )

        rows.append([
            str(i),
            dev["name"],
            dtype,
            str(max_in),
            str(int(dev.get("default_samplerate", 0))),
        ])

    if not rows:
        echo_warning("No audio devices found.")
    else:
        echo_table(
            headers=["ID", "Name", "Type", "Channels", "Sample Rate"],
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
