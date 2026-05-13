"""Tests for interactive HF token prompt in diarization factory."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from voxfusion.config.models import DiarizationConfig, DiarizationMLConfig

_NO_TOKEN_ENV_KEYS = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "VOXFUSION_DIARIZATION__ML__HF_AUTH_TOKEN",
)


def _config_without_token() -> DiarizationConfig:
    return DiarizationConfig(ml=DiarizationMLConfig(hf_auth_token=None))


def _no_token_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in _NO_TOKEN_ENV_KEYS}


@pytest.fixture(autouse=True)
def _clean_token_state(tmp_path: Path) -> None:
    """Reset in-memory token cache and point token file at tmp_path."""
    from voxfusion.diarization.factory import _reset_interactive_token

    _reset_interactive_token()
    token_file = tmp_path / "hf_token"
    with patch("voxfusion.diarization.factory._HF_TOKEN_FILE", token_file):
        yield
    _reset_interactive_token()


def test_prompt_returns_none_for_non_tty() -> None:
    from voxfusion.diarization.factory import _prompt_hf_token_interactively

    with patch("sys.stdin") as mock_stdin:
        mock_stdin.isatty.return_value = False
        result = _prompt_hf_token_interactively()
    assert result is None


def test_prompt_returns_token_on_tty() -> None:
    from voxfusion.diarization.factory import _prompt_hf_token_interactively

    with patch("sys.stdin") as mock_stdin:
        mock_stdin.isatty.return_value = True
        mock_stdin.readline.return_value = "hf_test_token_123\n"
        result = _prompt_hf_token_interactively()
    assert result == "hf_test_token_123"


def test_prompt_returns_none_on_empty_input() -> None:
    from voxfusion.diarization.factory import _prompt_hf_token_interactively

    with patch("sys.stdin") as mock_stdin:
        mock_stdin.isatty.return_value = True
        mock_stdin.readline.return_value = "\n"
        result = _prompt_hf_token_interactively()
    assert result is None


def test_ml_prerequisites_does_not_prompt_when_not_interactive() -> None:
    from voxfusion.diarization.factory import _ml_prerequisites

    with patch.dict(os.environ, _no_token_env(), clear=True):
        ok, reason, _source = _ml_prerequisites(_config_without_token(), interactive=False)
    assert ok is False
    assert "token" in (reason or "").lower()


def test_ml_prerequisites_uses_env_var_before_prompt() -> None:
    from voxfusion.diarization.factory import _ml_prerequisites

    with patch.dict(os.environ, {**_no_token_env(), "HF_TOKEN": "hf_from_env"}):
        ok, _reason, source = _ml_prerequisites(_config_without_token(), interactive=True)
    assert ok is True
    assert source == "env:HF_TOKEN"


def test_interactive_token_cached_across_calls() -> None:
    from voxfusion.diarization.factory import _ml_prerequisites

    with patch.dict(os.environ, _no_token_env(), clear=True):
        with patch(
            "voxfusion.diarization.factory._prompt_hf_token_interactively",
            return_value="hf_cached",
        ):
            ok1, _, s1 = _ml_prerequisites(_config_without_token(), interactive=True)
        with patch("voxfusion.diarization.factory._prompt_hf_token_interactively") as mock_prompt:
            ok2, _, s2 = _ml_prerequisites(_config_without_token(), interactive=True)
            mock_prompt.assert_not_called()

    assert ok1 is True
    assert ok2 is True
    assert s1 == "interactive"
    assert s2 == "interactive (cached)"


def test_channel_strategy_does_not_prompt_even_when_interactive() -> None:
    """Channel strategy must never trigger the interactive HF token prompt."""
    from voxfusion.diarization.factory import create_diarizer
    from voxfusion.diarization.none import NoneDiarizer

    with (
        patch.dict(os.environ, _no_token_env(), clear=True),
        patch("voxfusion.diarization.factory._prompt_hf_token_interactively") as mock_prompt,
    ):
        selection = create_diarizer(
            _config_without_token().__class__(
                strategy="channel",
                ml=DiarizationMLConfig(hf_auth_token=None),
            ),
            mode="file",
            interactive=True,
        )
        mock_prompt.assert_not_called()

    assert selection.resolved_strategy == "channel"
    assert not isinstance(selection.engine, NoneDiarizer)


# ── Token persistence tests ──────────────────────────────────────────


def test_interactive_token_is_saved_to_disk(tmp_path: Path) -> None:
    """Interactive token must be persisted to the token file."""
    from voxfusion.diarization.factory import _ml_prerequisites

    token_file = tmp_path / "hf_token"
    with patch("voxfusion.diarization.factory._HF_TOKEN_FILE", token_file):
        with patch.dict(os.environ, _no_token_env(), clear=True):
            with patch(
                "voxfusion.diarization.factory._prompt_hf_token_interactively",
                return_value="hf_persist_test",
            ):
                ok, _, source = _ml_prerequisites(
                    _config_without_token(), interactive=True
                )

    assert ok is True
    assert source == "interactive"
    assert token_file.read_text().strip() == "hf_persist_test"


def test_saved_token_is_loaded_without_prompt(tmp_path: Path) -> None:
    """A previously saved token file must be loaded without prompting."""
    from voxfusion.diarization.factory import _ml_prerequisites

    token_file = tmp_path / "hf_token"
    token_file.write_text("hf_saved_token\n")

    with patch("voxfusion.diarization.factory._HF_TOKEN_FILE", token_file):
        with patch.dict(os.environ, _no_token_env(), clear=True):
            with patch(
                "voxfusion.diarization.factory._prompt_hf_token_interactively"
            ) as mock_prompt:
                ok, _, source = _ml_prerequisites(
                    _config_without_token(), interactive=True
                )
                mock_prompt.assert_not_called()

    assert ok is True
    assert source == "saved"


def test_env_var_takes_priority_over_saved_token(tmp_path: Path) -> None:
    """Environment variable must override the saved token file."""
    from voxfusion.diarization.factory import _ml_prerequisites

    token_file = tmp_path / "hf_token"
    token_file.write_text("hf_saved\n")

    with patch("voxfusion.diarization.factory._HF_TOKEN_FILE", token_file):
        with patch.dict(os.environ, {**_no_token_env(), "HF_TOKEN": "hf_from_env"}):
            ok, _, source = _ml_prerequisites(_config_without_token(), interactive=False)

    assert ok is True
    assert source == "env:HF_TOKEN"


def test_corrupt_token_file_treated_as_missing(tmp_path: Path) -> None:
    """Empty or missing token file should not break anything."""
    from voxfusion.diarization.factory import _load_saved_token

    token_file = tmp_path / "hf_token"
    token_file.write_text("")
    with patch("voxfusion.diarization.factory._HF_TOKEN_FILE", token_file):
        assert _load_saved_token() is None

    missing = tmp_path / "nonexistent"
    with patch("voxfusion.diarization.factory._HF_TOKEN_FILE", missing):
        assert _load_saved_token() is None
