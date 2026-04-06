"""GUI logging tests for Open WebUI interactions."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock

from voxfusion.gui.main import TranscriptionGUI
from voxfusion.llm.client import LLMModelDescriptor

gui_main = importlib.import_module("voxfusion.gui.main")


class _FakeVar:
    def __init__(self, value: str = "") -> None:
        self._value = value

    def get(self) -> str:
        return self._value

    def set(self, value: str) -> None:
        self._value = value


class _ImmediateThread:
    def __init__(self, target, daemon: bool = False) -> None:
        del daemon
        self._target = target

    def start(self) -> None:
        self._target()


def test_start_llm_summarize_logs_request(monkeypatch) -> None:
    fake_log = MagicMock()
    worker_instances: list[object] = []

    class _FakeWorker:
        def __init__(self, **kwargs: object) -> None:
            worker_instances.append(self)
            self.kwargs = kwargs

        def start(self) -> None:
            return None

    gui = object.__new__(TranscriptionGUI)
    gui._llm_worker = None
    gui._llm_url_var = _FakeVar("http://openwebui:3000")
    gui._llm_model_var = _FakeVar("llama3.2:3b")
    gui._llm_key_var = _FakeVar("secret")
    gui._llm_prompt_var = _FakeVar("summarize")
    gui._llm_context_var = _FakeVar("")
    gui._llm_custom_user_prompt = ""
    gui._llm_model_contexts = {"llama3.2:3b": 8192}
    gui._llm_summarize_btn = MagicMock()
    gui._llm_status_label = MagicMock()
    gui._persist_gui_settings = MagicMock()
    gui._clear_llm_output = MagicMock()
    gui._refresh_file_workflow = MagicMock()
    gui._schedule_llm_token = MagicMock()
    gui._schedule_llm_error = MagicMock()
    gui._schedule_llm_finished = MagicMock()
    gui._get_file_transcript_text = lambda: "[00:00:01] [SPEAKER_00] Hello\n[00:00:03] [SPEAKER_01] World"
    gui._tr = lambda key, **kwargs: key if not kwargs else f"{key}:{kwargs}"
    gui._llm_last_error_message = None

    monkeypatch.setattr(gui_main, "log", fake_log)
    monkeypatch.setattr(gui_main, "LLMWorker", _FakeWorker)

    TranscriptionGUI._start_llm_summarize(gui)

    assert worker_instances
    fake_log.info.assert_any_call(
        "gui.llm_summarize_requested",
        base_url="http://openwebui:3000",
        model="llama3.2:3b",
        prompt_name="summarize",
        transcript_chars=59,
        transcript_lines=2,
        api_key_present=True,
        custom_user_prompt=False,
        context_tokens_resolved=8192,
        context_source="model_metadata",
    )
    assert worker_instances[0].kwargs["context_limit_tokens"] == 8192


def test_on_llm_models_loaded_logs_failure(monkeypatch) -> None:
    fake_log = MagicMock()

    gui = object.__new__(TranscriptionGUI)
    gui._llm_model_refreshing = True
    gui._llm_url_var = _FakeVar("http://openwebui:3000")
    gui._llm_status_label = MagicMock()
    gui._cached_llm_models = []
    gui._tr = lambda key, **kwargs: key if not kwargs else f"{key}:{kwargs}"

    monkeypatch.setattr(gui_main, "log", fake_log)

    TranscriptionGUI._on_llm_models_loaded(gui, [], "HTTP 503")

    fake_log.error.assert_called_once_with(
        "gui.llm_models_load_failed",
        base_url="http://openwebui:3000",
        error="HTTP 503",
    )


def test_on_llm_models_loaded_uses_cached_models_on_failure(monkeypatch) -> None:
    fake_log = MagicMock()

    gui = object.__new__(TranscriptionGUI)
    gui._llm_model_refreshing = True
    gui._llm_url_var = _FakeVar("http://openwebui:3000")
    gui._llm_model_var = _FakeVar("qwen2.5:32b")
    gui._cached_llm_models = ["qwen2.5-7b", "ohoswiki"]
    gui._available_llm_models = []
    gui._llm_model_combo = MagicMock()
    gui._llm_status_label = MagicMock()
    gui._persist_gui_settings = MagicMock()
    gui._tr = lambda key, **kwargs: key if not kwargs else f"{key}:{kwargs}"

    monkeypatch.setattr(gui_main, "log", fake_log)

    TranscriptionGUI._on_llm_models_loaded(gui, [], "HTTP 503")

    assert gui._available_llm_models == ["qwen2.5-7b", "ohoswiki"]
    assert gui._llm_model_var.get() == "qwen2.5-7b"
    fake_log.warning.assert_any_call(
        "gui.llm_models_loaded_from_cache",
        base_url="http://openwebui:3000",
        model_count=2,
        selected_model="qwen2.5-7b",
        error="HTTP 503",
    )



def test_show_llm_error_logs_error(monkeypatch) -> None:
    fake_log = MagicMock()

    gui = object.__new__(TranscriptionGUI)
    gui._llm_url_var = _FakeVar("http://openwebui:3000")
    gui._llm_model_var = _FakeVar("qwen3:32b")
    gui._llm_prompt_var = _FakeVar("summarize")
    gui._llm_status_label = MagicMock()
    gui._append_llm_token = MagicMock()
    gui._refresh_file_workflow = MagicMock()
    gui._tr = lambda key, **kwargs: key if not kwargs else f"{key}:{kwargs}"
    gui._llm_last_error_message = None

    monkeypatch.setattr(gui_main, "log", fake_log)

    TranscriptionGUI._show_llm_error(gui, "HTTP 503: backend busy")

    fake_log.error.assert_called_once_with(
        "gui.llm_error",
        base_url="http://openwebui:3000",
        model="qwen3:32b",
        prompt_name="summarize",
        error="HTTP 503: backend busy",
    )


def test_probe_llm_model_logs_success(monkeypatch) -> None:
    fake_log = MagicMock()

    async def _fake_complete(messages, *, base_url, model, api_key, timeout_read):
        assert messages == gui_main._LLM_PROBE_MESSAGES
        assert base_url == "http://openwebui:3000"
        assert model == "llama3.2:3b"
        assert api_key == "secret"
        assert timeout_read == gui_main._LLM_PROBE_TIMEOUT_READ
        return "OK"

    gui = object.__new__(TranscriptionGUI)
    gui._llm_worker = None
    gui._file_worker = None
    gui._llm_model_refreshing = False
    gui._llm_probe_running = False
    gui._llm_url_var = _FakeVar("http://openwebui:3000")
    gui._llm_model_var = _FakeVar("llama3.2:3b")
    gui._llm_key_var = _FakeVar("secret")
    gui._llm_status_label = MagicMock()
    gui._persist_gui_settings = MagicMock()
    gui._refresh_file_workflow = MagicMock()
    gui.root = type("Root", (), {"after": staticmethod(lambda _delay, fn, *args: fn(*args))})()
    gui._tr = lambda key, **kwargs: key if not kwargs else f"{key}:{kwargs}"

    monkeypatch.setattr(gui_main, "log", fake_log)
    monkeypatch.setattr(gui_main, "complete", _fake_complete)
    monkeypatch.setattr(gui_main.threading, "Thread", _ImmediateThread)

    TranscriptionGUI._probe_llm_model(gui)

    fake_log.info.assert_any_call(
        "gui.llm_probe_requested",
        base_url="http://openwebui:3000",
        model="llama3.2:3b",
        timeout_read=gui_main._LLM_PROBE_TIMEOUT_READ,
        api_key_present=True,
    )
    fake_log.info.assert_any_call(
        "gui.llm_probe_succeeded",
        base_url="http://openwebui:3000",
        model="llama3.2:3b",
        response_preview="OK",
    )


def test_probe_llm_model_logs_failure(monkeypatch) -> None:
    fake_log = MagicMock()

    async def _fake_complete(messages, *, base_url, model, api_key, timeout_read):
        del messages, base_url, model, api_key, timeout_read
        raise RuntimeError("HTTP 503")

    gui = object.__new__(TranscriptionGUI)
    gui._llm_worker = None
    gui._file_worker = None
    gui._llm_model_refreshing = False
    gui._llm_probe_running = False
    gui._llm_url_var = _FakeVar("http://openwebui:3000")
    gui._llm_model_var = _FakeVar("qwen3:32b")
    gui._llm_key_var = _FakeVar("secret")
    gui._llm_status_label = MagicMock()
    gui._persist_gui_settings = MagicMock()
    gui._refresh_file_workflow = MagicMock()
    gui.root = type("Root", (), {"after": staticmethod(lambda _delay, fn, *args: fn(*args))})()
    gui._tr = lambda key, **kwargs: key if not kwargs else f"{key}:{kwargs}"

    monkeypatch.setattr(gui_main, "log", fake_log)
    monkeypatch.setattr(gui_main, "complete", _fake_complete)
    monkeypatch.setattr(gui_main.threading, "Thread", _ImmediateThread)

    TranscriptionGUI._probe_llm_model(gui)

    fake_log.error.assert_any_call(
        "gui.llm_probe_failed",
        base_url="http://openwebui:3000",
        model="qwen3:32b",
        error="HTTP 503",
    )


def test_on_llm_models_loaded_caches_context_metadata(monkeypatch) -> None:
    fake_log = MagicMock()

    gui = object.__new__(TranscriptionGUI)
    gui._llm_model_refreshing = True
    gui._llm_url_var = _FakeVar("http://openwebui:3000")
    gui._llm_model_var = _FakeVar("qwen2.5-7b")
    gui._llm_context_var = _FakeVar("")
    gui._cached_llm_models = []
    gui._cached_llm_model_contexts = {}
    gui._available_llm_models = []
    gui._llm_model_contexts = {}
    gui._llm_model_combo = MagicMock()
    gui._llm_status_label = MagicMock()
    gui._llm_context_hint_label = MagicMock()
    gui._persist_gui_settings = MagicMock()
    gui._tr = lambda key, **kwargs: key if not kwargs else f"{key}:{kwargs}"

    monkeypatch.setattr(gui_main, "log", fake_log)

    TranscriptionGUI._on_llm_models_loaded(
        gui,
        [LLMModelDescriptor(id="qwen2.5-7b", context_tokens=32768)],
        None,
    )

    assert gui._cached_llm_model_contexts == {"qwen2.5-7b": 32768}
    assert gui._llm_model_contexts == {"qwen2.5-7b": 32768}
    gui._llm_context_hint_label.configure.assert_called()
