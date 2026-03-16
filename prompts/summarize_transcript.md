# Промпт: Краткое изложение транскрипции

Используй этот промпт для генерации пересказа транскрибированного разговора / совещания / записи.
Работает на моделях от 7B до 32B+ параметров (Qwen 2.5, Llama 3.x, Mistral и аналогах).

---

## Системный промпт (system prompt)

```
You are an expert meeting summarizer. Your task is to read a speech transcript and produce a concise, well-structured summary. Follow all instructions exactly. Do not add information that is not present in the transcript. Do not skip any significant topic or decision.
```

---

## Пользовательский промпт (user prompt)

```
Below is a transcript of a conversation/meeting. Each line starts with [HH:MM:SS] [SPEAKER] followed by the spoken text. Some lines may be in Russian or mixed Russian/English.

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
...

## Decisions & Conclusions
- <decision 1>
...

## Action Items
- <action> — <person responsible> (if known)
...

## Speaker Contributions
- <SPEAKER_ID>: <key point>
...

---

TRANSCRIPT:
{transcript}
```

---

## Как использовать

### Подстановка транскрипции

Замени `{transcript}` на текст из файла `.txt`, выгруженного из VoxFusion (Save... → .txt).

Формат VoxFusion-вывода:
```
[00:00:05] [mic]    Добрый день, начинаем совещание.
[00:00:12] [system] Я готов, слушаю.
[00:00:18] [mic]    Обсудим план на следующий квартал...
```

### Пример вызова через API (Python)

```python
import anthropic  # или openai, или ollama

transcript = open("result.txt", encoding="utf-8").read()

prompt = """...<вставь user prompt выше>...""".replace("{transcript}", transcript)

# Claude API
client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-haiku-4-5-20251001",  # быстро и дёшево
    max_tokens=2048,
    system="You are an expert meeting summarizer. ...",
    messages=[{"role": "user", "content": prompt}],
)
print(response.content[0].text)
```

---

## Рекомендации по выбору модели

### Для пересказа транскрипций рекомендуется:

| Модель | Где запускать | Качество | Скорость | Примечание |
|--------|--------------|----------|----------|------------|
| **Qwen2.5-32B-Instruct** | Ollama / vLLM | Отличное | Среднее | Лучший выбор для локального запуска; отлично работает с русским языком |
| **Qwen2.5-72B-Instruct** | vLLM / облако | Превосходное | Медленнее | Если есть GPU >40 GB или несколько GPU |
| **Llama-3.3-70B-Instruct** | vLLM / Groq | Отличное | Быстро на Groq | Хорошо следует структурированным инструкциям |
| **Mistral-Nemo-12B** | Ollama | Хорошее | Быстро | Компромисс для слабого железа (8+ GB VRAM) |
| **claude-haiku-4-5** | Anthropic API | Превосходное | Очень быстро | Дёшево ($0.25/Mtok), лучшее соотношение цена/качество |
| **gpt-4o-mini** | OpenAI API | Отличное | Быстро | Хорошая альтернатива |

### Важные нюансы для слабых/средних моделей (7B–32B):

1. **Длинные транскрипции** — модели до 32B хуже справляются с текстами >4000 слов.
   Решение: разбей транскрипцию на части по ~2000 слов с перекрытием 100 слов,
   суммаризируй каждую часть отдельно, затем суммаризируй полученные части.

2. **Следование формату** — Qwen2.5 и Llama 3.3 лучше других соблюдают заданную структуру.
   Mistral 7B/12B иногда игнорирует заголовки секций — добавь `You MUST use the exact headers`.

3. **Язык** — для русского языка Qwen2.5 значительно лучше Llama и Mistral.

4. **Температура** — используй `temperature=0.1–0.3` для детерминированного структурированного вывода.

---

## Промпт для длинных транскрипций (chunked summary)

Если запись длиннее ~30 минут, используй двухпроходный подход:

**Шаг 1 — суммаризация каждого чанка:**
```
This is PART {n} of {total} of a transcript. Summarize only this part in 3-5 sentences. Preserve speaker labels and any decisions or action items.

TRANSCRIPT PART {n}/{total}:
{chunk_text}
```

**Шаг 2 — финальный пересказ по частичным саммари:**
```
Below are {total} partial summaries of a long meeting transcript. Combine them into one final structured summary using the format below.

PARTIAL SUMMARIES:
{all_partial_summaries}

OUTPUT FORMAT:
## Summary
## Key Topics
## Decisions & Conclusions
## Action Items
## Speaker Contributions
```
