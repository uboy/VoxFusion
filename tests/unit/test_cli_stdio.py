"""Tests for CLI stdio bootstrap helpers."""

from __future__ import annotations

import voxfusion.cli.main as cli_main


class _FakeStream:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def reconfigure(self, **kwargs: str) -> None:
        self.calls.append(kwargs)


def test_configure_utf8_stdio_reconfigures_stdout_and_stderr(monkeypatch) -> None:
    fake_stdout = _FakeStream()
    fake_stderr = _FakeStream()

    monkeypatch.setattr(cli_main.sys, "stdout", fake_stdout)
    monkeypatch.setattr(cli_main.sys, "stderr", fake_stderr)

    cli_main._configure_utf8_stdio()

    assert fake_stdout.calls == [{"encoding": "utf-8", "errors": "replace"}]
    assert fake_stderr.calls == [{"encoding": "utf-8", "errors": "replace"}]
