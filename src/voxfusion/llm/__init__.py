"""LLM post-processing utilities — Open WebUI / OpenAI-compatible client."""

from voxfusion.llm.client import LLMError, complete, stream_completion
from voxfusion.llm.prompts import BUILTIN_PROMPTS, build_messages

__all__ = [
    "BUILTIN_PROMPTS",
    "LLMError",
    "build_messages",
    "complete",
    "stream_completion",
]
