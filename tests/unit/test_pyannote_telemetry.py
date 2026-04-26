"""Tests for pyannote-audio telemetry suppression.

pyannote.audio 4.x includes an OpenTelemetry exporter that posts metrics to
https://otel.pyannote.ai/v1/metrics every 60 seconds via a background thread.
We disable this at module import time in both gigaam_engine and pyannote_engine.

This test verifies that:
1. The telemetry flag is set to False (no data recording)
2. The MeterProvider is shut down (background thread stopped)
"""

import threading

import pytest

pyannote = pytest.importorskip("pyannote")


@pytest.mark.unit
def test_pyannote_telemetry_is_disabled_after_gigaam_import() -> None:
    """Importing gigaam_engine should disable pyannote telemetry."""
    # Import first time (triggers _disable_pyannote_telemetry)
    import importlib

    import voxfusion.asr.gigaam_engine as gigaam_module

    # Reload to ensure the function runs in this test
    importlib.reload(gigaam_module)

    # Check the telemetry flag is disabled via pyannote's API
    from pyannote.audio.telemetry.metrics import is_metrics_enabled

    assert is_metrics_enabled() is False, (
        "PYANNOTE_METRICS_ENABLED should be False after importing gigaam_engine"
    )

    # Check the MeterProvider is shut down (no active background thread)
    from opentelemetry import metrics

    provider = metrics.get_meter_provider()
    # A shut-down provider has no active readers/exporters
    assert hasattr(provider, "_shutdown"), "MeterProvider should be shut down"


@pytest.mark.unit
def test_pyannote_telemetry_is_disabled_after_diarization_import() -> None:
    """Importing pyannote_engine should disable pyannote telemetry."""
    import importlib

    import voxfusion.diarization.pyannote_engine as diarization_module

    # Reload to ensure the function runs
    importlib.reload(diarization_module)

    # Check the telemetry flag is disabled via pyannote's API
    from pyannote.audio.telemetry.metrics import is_metrics_enabled

    assert is_metrics_enabled() is False, (
        "PYANNOTE_METRICS_ENABLED should be False after importing pyannote_engine"
    )

    # Check the MeterProvider is shut down
    from opentelemetry import metrics

    provider = metrics.get_meter_provider()
    assert hasattr(provider, "_shutdown"), "MeterProvider should be shut down"


@pytest.mark.unit
def test_no_pyannote_telemetry_threads_running() -> None:
    """No pyannote telemetry threads should be running after imports."""
    import importlib

    import voxfusion.asr.gigaam_engine
    import voxfusion.diarization.pyannote_engine

    # Reload both modules to ensure telemetry suppression runs
    importlib.reload(voxfusion.asr.gigaam_engine)
    importlib.reload(voxfusion.diarization.pyannote_engine)

    # Check for any telemetry-related threads
    thread_names = [t.name for t in threading.enumerate()]

    # pyannote uses PeriodicExportingMetricReader which creates threads
    # with names like "PeriodicMetricsExporterThread" or similar
    telemetry_threads = [name for name in thread_names if "metric" in name.lower()]

    assert len(telemetry_threads) == 0, (
        f"Found telemetry threads running: {telemetry_threads}. "
        "pyannote telemetry should be completely disabled."
    )
