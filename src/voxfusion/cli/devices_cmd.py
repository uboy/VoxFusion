"""CLI command: voxfusion devices -- list available audio devices."""

import click

from voxfusion.cli.formatting import echo_table, echo_warning


@click.command("devices")
@click.option("--type", "-t", "device_type", type=click.Choice(["all", "input", "loopback"]),
              default="all", help="Filter by device type.")
def devices(device_type: str) -> None:
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
        return

    echo_table(
        headers=["ID", "Name", "Type", "Channels", "Sample Rate"],
        rows=rows,
    )
