"""Built-in prompt templates for LLM transcript post-processing."""

from __future__ import annotations

_SUMMARIZE_SYSTEM = """You are an expert meeting summarizer. Your task is to read a speech transcript and produce a concise, well-structured summary. Follow all instructions exactly. Do not add information that is not present in the transcript. Do not skip any significant topic or decision."""

_SUMMARIZE_USER = """Below is a transcript of a conversation/meeting. Each line starts with [HH:MM:SS] [SPEAKER] followed by the spoken text.

Your task:
1. Write a SHORT SUMMARY (3-5 sentences) of the entire conversation.
2. List the KEY TOPICS discussed (bullet points, max 8 items).
3. List all DECISIONS or CONCLUSIONS that were explicitly stated (bullet points). If none — write "No explicit decisions recorded."
4. List ACTION ITEMS with responsible person if mentioned (bullet points). If none — write "No action items."
5. If there are multiple distinct speakers, briefly note WHO said WHAT on each key point (1 sentence per speaker contribution).

Rules:
- Write in the SAME LANGUAGE as the transcript (if Russian — answer in Russian, if English — answer in English).
- Be concise. No padding, no filler phrases.
- Do not invent or assume information not present in the transcript.
- If the transcript is unclear or incomplete, note it briefly.

OUTPUT FORMAT (use exactly these section headers):

## Summary
<3-5 sentence overview>

## Key Topics
- <topic 1>
- <topic 2>

## Decisions & Conclusions
- <decision 1>

## Action Items
- <action> — <person responsible> (if known)

## Speaker Contributions
- <SPEAKER_ID>: <key point>

---

TRANSCRIPT:
{transcript}
"""

_CHUNK_TASK_PREFIX = """The transcript below is chunk {chunk_index} of {chunk_count} from a longer transcript.
Apply the requested task to THIS CHUNK ONLY.
Your output will be merged with outputs from other chunks later, so:
- capture only facts present in this chunk
- do not speculate about content outside this chunk
- keep the same language and structure requested by the task
"""

_MERGE_CHUNK_OUTPUTS_USER = """Below are partial outputs generated from multiple transcript chunks of one longer conversation.

Merge them into one final response for the full transcript.

Rules:
- preserve the requested language and output structure
- deduplicate repeated topics, decisions, and action items
- keep only facts supported by the partial outputs
- if the same point appears multiple times, keep one clear consolidated version

Original task template:
{task_template}

PARTIAL OUTPUTS:
{chunk_outputs}
"""

BUILTIN_PROMPTS: dict[str, dict[str, str]] = {
    "summarize": {
        "system": _SUMMARIZE_SYSTEM,
        "user": _SUMMARIZE_USER,
        "description": "Summarize transcript: key topics, decisions, and action items",
    },
}


def _resolve_prompt_texts(
    template_name: str,
    *,
    custom_system: str | None = None,
    custom_user: str | None = None,
) -> tuple[str, str]:
    tmpl = BUILTIN_PROMPTS[template_name]
    return custom_system or tmpl["system"], custom_user or tmpl["user"]


def build_messages(
    template_name: str,
    transcript: str,
    *,
    custom_system: str | None = None,
    custom_user: str | None = None,
) -> list[dict[str, str]]:
    """Build an OpenAI-compatible messages list from a template and transcript."""
    system_text, user_template = _resolve_prompt_texts(
        template_name,
        custom_system=custom_system,
        custom_user=custom_user,
    )
    user_text = user_template.format(transcript=transcript)
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]


def build_chunk_messages(
    template_name: str,
    transcript: str,
    *,
    chunk_index: int,
    chunk_count: int,
    custom_system: str | None = None,
    custom_user: str | None = None,
) -> list[dict[str, str]]:
    """Build a chunk-scoped prompt for one transcript fragment."""
    system_text, user_template = _resolve_prompt_texts(
        template_name,
        custom_system=custom_system,
        custom_user=custom_user,
    )
    task_text = user_template.format(transcript=transcript)
    user_text = (
        _CHUNK_TASK_PREFIX.format(chunk_index=chunk_index, chunk_count=chunk_count)
        + "\n\n"
        + task_text
    )
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]


def build_merge_messages(
    template_name: str,
    chunk_outputs: str,
    *,
    custom_system: str | None = None,
    custom_user: str | None = None,
) -> list[dict[str, str]]:
    """Build a final merge prompt from partial chunk outputs."""
    system_text, user_template = _resolve_prompt_texts(
        template_name,
        custom_system=custom_system,
        custom_user=custom_user,
    )
    task_template = user_template.replace("{transcript}", "<full transcript omitted during merge>")
    user_text = _MERGE_CHUNK_OUTPUTS_USER.format(
        task_template=task_template,
        chunk_outputs=chunk_outputs,
    )
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
