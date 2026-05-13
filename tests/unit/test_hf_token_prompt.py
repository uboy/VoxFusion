"""Tests for interactive HF token prompt in diarization factory."""

from __future__ import annotations

import os
from unittest.mock import patch

from voxfusion.config.models import DiarizationConfig, DiarizationMLConfig


def _config_without_token() -> DiarizationConfig:
    return DiarizationConfig(ml=DiarizationMLConfig(hf_auth_token=None))


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
    from voxfusion.diarization.factory import _ml_prerequisites, _reset_interactive_token

    _reset_interactive_token()
    env = {k: v for k, v in os.environ.items()
           if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "VOXFUSION_DIARIZATION__ML__HF_AUTH_TOKEN")}
    with patch.dict(os.environ, env, clear=True):
        ok, reason, _source = _ml_prerequisites(_config_without_token(), interactive=False)
    assert ok is False
    assert "token" in (reason or "").lower()


def test_ml_prerequisites_uses_env_var_before_prompt() -> None:
    from voxfusion.diarization.factory import _ml_prerequisites, _reset_interactive_token

    _reset_interactive_token()
    with patch.dict(os.environ, {"HF_TOKEN": "hf_from_env"}):
        ok, _reason, source = _ml_prerequisites(_config_without_token(), interactive=True)
    assert ok is True
    assert source == "env:HF_TOKEN"


def test_interactive_token_cached_across_calls() -> None:
    from voxfusion.diarization.factory import _ml_prerequisites, _reset_interactive_token

    _reset_interactive_token()
    env = {k: v for k, v in os.environ.items()
           if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "VOXFUSION_DIARIZATION__ML__HF_AUTH_TOKEN")}

    with patch.dict(os.environ, env, clear=True):
        with patch("voxfusion.diarization.factory._prompt_hf_token_interactively", return_value="hf_cached"):
            ok1, _, s1 = _ml_prerequisites(_config_without_token(), interactive=True)
        # Second call should use cache, not prompt again
        with patch("voxfusion.diarization.factory._prompt_hf_token_interactively") as mock_prompt:
            ok2, _, s2 = _ml_prerequisites(_config_without_token(), interactive=True)
            mock_prompt.assert_not_called()

    assert ok1 is True
    assert ok2 is True
    assert s1 == "interactive"
    assert s2 == "interactive (cached)"
