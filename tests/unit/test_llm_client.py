"""Unit tests for Open WebUI client helpers."""

from voxfusion.llm.client import _extract_model_ids, extract_model_catalog


def test_extract_model_ids_from_data_payload() -> None:
    payload = {"data": [{"id": "qwen2.5:32b"}, {"id": "llama3.1:8b"}]}
    assert _extract_model_ids(payload) == ["llama3.1:8b", "qwen2.5:32b"]


def test_extract_model_ids_from_models_payload() -> None:
    payload = {"models": [{"name": "gemma3:12b"}, {"model": "qwen3:14b"}]}
    assert _extract_model_ids(payload) == ["gemma3:12b", "qwen3:14b"]


def test_extract_model_catalog_reads_context_tokens_from_common_metadata() -> None:
    payload = {
        "data": [
            {"id": "qwen2.5-7b", "context_length": 32768},
            {"id": "llama3.2:3b", "details": {"num_ctx": "8k"}},
        ]
    }

    catalog = extract_model_catalog(payload)

    assert [(item.id, item.context_tokens) for item in catalog] == [
        ("llama3.2:3b", 8192),
        ("qwen2.5-7b", 32768),
    ]
