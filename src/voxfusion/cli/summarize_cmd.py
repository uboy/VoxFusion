"""CLI command: voxfusion summarize -- LLM-powered transcript post-processing.

Sends a transcript file to an Open WebUI instance and streams the response.

Examples::

    voxfusion summarize meeting.txt
    voxfusion summarize meeting.txt --model qwen2.5:32b --url http://192.168.1.10:3000
    voxfusion summarize meeting.txt --api-key my-token --output summary.md
    voxfusion summarize meeting.txt --no-stream --output summary.txt
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click

from voxfusion.llm.client import DEFAULT_BASE_URL, DEFAULT_MODEL, LLMError, complete, stream_completion
from voxfusion.llm.prompts import BUILTIN_PROMPTS, build_messages
from voxfusion.logging import configure_logging, get_logger

log = get_logger(__name__)

_PROMPT_CHOICES = sorted(BUILTIN_PROMPTS.keys())


@click.command("summarize")
@click.argument("transcript_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--url", "-u",
    default=DEFAULT_BASE_URL,
    show_default=True,
    envvar="VOXFUSION_LLM_URL",
    help="Open WebUI base URL.",
)
@click.option(
    "--model", "-m",
    default=DEFAULT_MODEL,
    show_default=True,
    envvar="VOXFUSION_LLM_MODEL",
    help="Model name as it appears in Open WebUI (e.g. qwen2.5:32b).",
)
@click.option(
    "--api-key", "-k",
    default="",
    envvar="VOXFUSION_LLM_API_KEY",
    help="Bearer token / API key for Open WebUI (optional).",
)
@click.option(
    "--prompt", "-p",
    default="summarize",
    type=click.Choice(_PROMPT_CHOICES),
    show_default=True,
    help="Built-in prompt template to use.",
)
@click.option(
    "--output", "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write LLM output to this file instead of stdout.",
)
@click.option(
    "--no-stream",
    is_flag=True,
    default=False,
    help="Collect full response before printing (disables streaming output).",
)
@click.pass_context
def summarize(
    ctx: click.Context,
    transcript_file: Path,
    url: str,
    model: str,
    api_key: str,
    prompt: str,
    output: Path | None,
    no_stream: bool,
) -> None:
    """Process a transcript file with an LLM via Open WebUI.

    Reads TRANSCRIPT_FILE, applies the chosen prompt template, and sends
    the request to the Open WebUI API.  Output is streamed to stdout by
    default or written to a file with --output.

    Environment variables:
        VOXFUSION_LLM_URL      Base URL of the Open WebUI instance.
        VOXFUSION_LLM_MODEL    Model name.
        VOXFUSION_LLM_API_KEY  Bearer token (if authentication is required).
    """
    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)
    configure_logging("DEBUG" if verbose else ("ERROR" if quiet else "INFO"))

    transcript_text = transcript_file.read_text(encoding="utf-8").strip()
    if not transcript_text:
        raise click.ClickException(f"Transcript file is empty: {transcript_file}")

    messages = build_messages(prompt, transcript_text)

    if not quiet:
        click.echo(
            f"Summarizing: {transcript_file.name}  |  model: {model}  |  url: {url}",
            err=True,
        )
        if output:
            click.echo(f"Output -> {output}", err=True)
        click.echo("", err=True)

    try:
        if no_stream:
            result = asyncio.run(
                complete(messages, base_url=url, model=model, api_key=api_key)
            )
            _write_or_print(result, output)
        else:
            asyncio.run(
                _stream_to_output(messages, url=url, model=model, api_key=api_key, output=output)
            )
    except LLMError as exc:
        raise click.ClickException(str(exc)) from exc
    except KeyboardInterrupt:
        click.echo("\nInterrupted.", err=True)
        sys.exit(130)

    if not quiet and output:
        click.echo(f"\nSaved to: {output}", err=True)


async def _stream_to_output(
    messages: list[dict[str, str]],
    *,
    url: str,
    model: str,
    api_key: str,
    output: Path | None,
) -> None:
    """Stream LLM response to stdout or collect and write to file."""
    if output is None:
        # Stream directly to stdout
        async for token in stream_completion(messages, base_url=url, model=model, api_key=api_key):
            click.echo(token, nl=False)
        click.echo()  # final newline
    else:
        # Collect then write (streaming to file doesn't add much value)
        parts: list[str] = []
        async for token in stream_completion(messages, base_url=url, model=model, api_key=api_key):
            parts.append(token)
            click.echo(".", nl=False, err=True)  # progress indicator
        click.echo(" done", err=True)
        _write_or_print("".join(parts), output)


def _write_or_print(text: str, output: Path | None) -> None:
    if output is None:
        click.echo(text)
    else:
        output.write_text(text, encoding="utf-8")
