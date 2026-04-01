"""Tests for GUI/runtime noise suppression helpers."""

import os

from voxfusion.gui.runtime import TextRedirector
from voxfusion.gui.runtime import _configure_gui_noise_controls
from voxfusion.logging import _should_suppress_log_message


def test_should_suppress_known_safe_dependency_noise() -> None:
    assert _should_suppress_log_message("NumExpr defaulting to 8 threads.")
    assert _should_suppress_log_message(
        "OneLogger: Setting error_handling_strategy to DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR"
    )
    assert _should_suppress_log_message(
        "Megatron num_microbatches_calculator not found, using Apex version."
    )
    assert _should_suppress_log_message(
        "'(MaxRetryError(...))' thrown while requesting HEAD https://huggingface.co/pyannote/speaker-diarization-3.1/resolve/main/config.yaml"
    )
    assert _should_suppress_log_message("Retrying in 1s [Retry 1/5].")
    assert _should_suppress_log_message(
        "Found only 2 clusters. Using a smaller value than 12 for `min_cluster_size` might help."
    )
    assert not _should_suppress_log_message("real pipeline failure")


def test_configure_gui_noise_controls_sets_runtime_env(monkeypatch) -> None:
    monkeypatch.delenv("HF_HUB_DISABLE_SYMLINKS_WARNING", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
    monkeypatch.delenv("NUMEXPR_MAX_THREADS", raising=False)
    monkeypatch.delenv("NUMEXPR_NUM_THREADS", raising=False)

    _configure_gui_noise_controls()

    assert os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] == "1"
    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert int(os.environ["NUMEXPR_MAX_THREADS"]) >= 1
    assert os.environ["NUMEXPR_NUM_THREADS"] == os.environ["NUMEXPR_MAX_THREADS"]


def test_text_redirector_suppresses_known_gui_noise_lines() -> None:
    class _FakeWidget:
        def after(self, *_args, **_kwargs) -> None:
            pass

    redirector = TextRedirector(_FakeWidget())

    clean = redirector._sanitize(  # noqa: SLF001
        "useful line\n"
        "[NeMo W] Megatron num_microbatches_calculator not found, using Apex version.\n"
        "still useful\n"
    )

    assert clean == "useful line\nstill useful\n"


def test_text_redirector_suppresses_hf_retry_and_cluster_noise_lines() -> None:
    class _FakeWidget:
        def after(self, *_args, **_kwargs) -> None:
            pass

    redirector = TextRedirector(_FakeWidget())

    clean = redirector._sanitize(  # noqa: SLF001
        "useful line\n"
        "'(MaxRetryError(...))' thrown while requesting HEAD https://huggingface.co/pyannote/speaker-diarization-3.1/resolve/main/config.yaml\n"
        "INFO | Retrying in 1s [Retry 1/5].\n"
        "Found only 2 clusters. Using a smaller value than 12 for `min_cluster_size` might help.\n"
        "still useful\n"
    )

    assert clean == "useful line\nstill useful\n"
