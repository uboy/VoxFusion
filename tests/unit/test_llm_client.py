"""Unit tests for Open WebUI client helpers."""

from voxfusion.llm.client import _extract_model_ids


def test_extract_model_ids_from_data_payload() -> None:
    payload = {"data": [{"id": "qwen2.5:32b"}, {"id": "llama3.1:8b"}]}
    assert _extract_model_ids(payload) == ["llama3.1:8b", "qwen2.5:32b"]


def test_extract_model_ids_from_models_payload() -> None:
    payload = {"models": [{"name": "gemma3:12b"}, {"model": "qwen3:14b"}]}
    assert _extract_model_ids(payload) == ["gemma3:12b", "qwen3:14b"]
