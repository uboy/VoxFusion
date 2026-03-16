"""Built-in prompt templates for LLM transcript post-processing."""

from __future__ import annotations

_SUMMARIZE_SYSTEM = """\
You are an expert meeting summarizer. Your task is to read a speech transcript \
and produce a concise, well-structured summary. Follow all instructions exactly. \
Do not add information that is not present in the transcript. \
Do not skip any significant topic or decision.\
"""

_SUMMARIZE_USER = """\
Below is a transcript of a conversation/meeting. \
Each line starts with [HH:MM:SS] [SPEAKER] followed by the spoken text.

Your task:
1. Write a SHORT SUMMARY (3-5 sentences) of the entire conversation.
2. List the KEY TOPICS discussed (bullet points, max 8 items).
3. List all DECISIONS or CONCLUSIONS that were explicitly stated (bullet points). \
If none — write "No explicit decisions recorded."
4. List ACTION ITEMS with responsible person if mentioned (bullet points). \
If none — write "No action items."
5. If there are multiple distinct speakers, briefly note WHO said WHAT on each \
key point (1 sentence per speaker contribution).

Rules:
- Write in the SAME LANGUAGE as the transcript \
(if Russian — answer in Russian, if English — answer in English).
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

BUILTIN_PROMPTS: dict[str, dict[str, str]] = {
    "summarize": {
        "system": _SUMMARIZE_SYSTEM,
        "user": _SUMMARIZE_USER,
        "description": "Summarize transcript: key topics, decisions, and action items",
    },
}


def build_messages(
    template_name: str,
    transcript: str,
    *,
    custom_system: str | None = None,
    custom_user: str | None = None,
) -> list[dict[str, str]]:
    """Build an OpenAI-compatible messages list from a template and transcript.

    Args:
        template_name: Key from :data:`BUILTIN_PROMPTS` (e.g. ``"summarize"``).
        transcript: The raw transcript text to substitute into the user prompt.
        custom_system: Override the system prompt from the template.
        custom_user: Override the user prompt template (must contain ``{transcript}``).

    Returns:
        A list of ``{"role": ..., "content": ...}`` dicts.
    """
    tmpl = BUILTIN_PROMPTS[template_name]
    system_text = custom_system or tmpl["system"]
    user_template = custom_user or tmpl["user"]
    user_text = user_template.format(transcript=transcript)
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
