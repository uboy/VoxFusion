"""CLI output formatting helpers: tables, colors, progress indicators."""

import click


def format_table(headers: list[str], rows: list[list[str]], padding: int = 2) -> str:
    """Format data as an aligned text table.

    Args:
        headers: Column header labels.
        rows: List of rows, each a list of cell values.
        padding: Spaces between columns.
    """
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(cell))

    sep = " " * padding
    header_line = sep.join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    divider = sep.join("-" * w for w in col_widths)

    lines = [header_line, divider]
    for row in rows:
        line = sep.join(
            (row[i] if i < len(row) else "").ljust(col_widths[i])
            for i in range(len(headers))
        )
        lines.append(line)

    return "\n".join(lines)


def echo_table(headers: list[str], rows: list[list[str]]) -> None:
    """Print a formatted table to stdout."""
    click.echo(format_table(headers, rows))


def echo_key_value(label: str, value: str, label_width: int = 20) -> None:
    """Print a key-value pair with aligned label."""
    click.echo(f"  {label:<{label_width}} {value}")


def echo_success(message: str) -> None:
    """Print a success message in green."""
    click.echo(click.style(message, fg="green"))


def echo_warning(message: str) -> None:
    """Print a warning message in yellow."""
    click.echo(click.style(message, fg="yellow"), err=True)


def echo_error(message: str) -> None:
    """Print an error message in red."""
    click.echo(click.style(message, fg="red"), err=True)
