# 🎙️ VoxFusion — Руководство по потоковой записи и переводу

## 🚀 Быстрый старт

VoxFusion теперь поддерживает **потоковую запись звука с транскрипцией и переводом в реальном времени**!

### Что уже работает:

✅ **Запись звука** — микрофон, системный звук, или оба сразу
✅ **Транскрипция в реальном времени** — faster-whisper с GPU/CPU
✅ **Определение языка** — автоматическое определение языка речи
✅ **Диаризация спикеров** — разделение по каналам (микрофон/система)
✅ **Перевод** — 4 бэкенда (Argos, NLLB, DeepL, LibreTranslate)
✅ **Форматы вывода** — JSON, SRT, VTT, TXT

---

## 📋 Команды

### 1. Просмотр доступных устройств

```bash
python -m voxfusion devices
```

Выведет список всех доступных микрофонов и устройств воспроизведения с их ID.

### 2. Простая запись с микрофона (только транскрипция)

```bash
python -m voxfusion capture
```

**Что происходит:**
- Записывает с микрофона по умолчанию
- Транскрибирует речь в реальном времени
- Выводит результат в текстовом формате
- **Ctrl+C** для остановки

### 3. Запись с переводом на русский

```bash
python -m voxfusion capture --translate ru
```

**Пример вывода:**
```
[1] SPEAKER_LOCAL (en): Hello, how are you?
    → Translation (ru): Привет, как дела?

[2] SPEAKER_LOCAL (en): The weather is nice today.
    → Translation (ru): Погода сегодня хорошая.
```

### 4. Запись системного звука (всё что играет на компьютере)

```bash
python -m voxfusion capture --source system
```

Это захватывает:
- Видео в браузере (YouTube, Netflix, и т.д.)
- Музыку из медиаплеера
- Звук из Skype/Zoom/Teams
- Любые другие приложения с аудио

### 5. Запись микрофона И системного звука одновременно

```bash
python -m voxfusion capture --source both --translate ru
```

**Используется для:**
- Записи онлайн-звонков (Skype, Zoom, Discord)
- Вы говорите в микрофон → `SPEAKER_LOCAL`
- Собеседник через систему → `SPEAKER_REMOTE`
- Оба переводятся на русский

### 6. Выбор конкретного микрофона

```bash
# Сначала смотрим список
python -m voxfusion devices

# Потом выбираем нужный (например, ID=1)
python -m voxfusion capture --device 1 --translate ru
```

### 7. Запись на определённое время (30 секунд)

```bash
python -m voxfusion capture --duration 30 --translate ru
```

### 8. Вывод в JSON формате

```bash
python -m voxfusion capture --output-format json --translate ru
```

JSON содержит:
- Точные временные метки
- Уверенность распознавания
- Информацию о спикерах
- Оригинальный и переведённый текст

### 9. Принудительное указание языка источника

```bash
python -m voxfusion capture --language en --translate ru
```

Полезно если автоопределение языка работает неправильно.

---

## 🔧 Установка бэкендов для перевода

### Argos Translate (рекомендуется, offline)

```bash
pip install argostranslate
```

**Загрузка языковых пакетов:**
```bash
python -m argostranslate.package install en ru  # English → Russian
python -m argostranslate.package install ru en  # Russian → English
python -m argostranslate.package install en de  # English → German
```

**Преимущества:**
- ✅ Работает офлайн
- ✅ Бесплатно
- ✅ MIT лицензия
- ✅ Не требует API ключей

### NLLB (Meta, offline, нужен GPU для скорости)

```bash
pip install transformers sentencepiece sacremoses
```

Использование:
```bash
python -m voxfusion capture --translate ru
# В config.yaml установить: translation.backend: nllb
```

### DeepL (API, лучшее качество)

```bash
pip install deepl
export DEEPL_API_KEY=your_key_here
```

Использование:
```bash
python -m voxfusion capture --translate ru
# В config.yaml установить: translation.backend: deepl
```

### LibreTranslate (API, self-hosted опция)

```bash
# Используется по умолчанию: https://libretranslate.com
python -m voxfusion capture --translate ru
# В config.yaml установить: translation.backend: libretranslate
```

---

## ⚙️ Конфигурация

Создайте файл `config.yaml`:

```yaml
# Настройки записи
capture:
  sources: ["microphone"]    # или ["system"] или ["microphone", "system"]
  sample_rate: 16000
  chunk_duration_ms: 500
  buffer_size: 10

# Настройки распознавания речи
asr:
  model_size: "small"        # tiny, base, small, medium, large-v3
  device: "auto"             # auto, cpu, cuda
  compute_type: "int8"       # int8, float16, float32
  language: null             # null = автоопределение, или "en", "ru", и т.д.
  word_timestamps: false
  beam_size: 5

# Настройки перевода
translation:
  enabled: true              # включить/выключить перевод
  target_language: "ru"      # целевой язык
  backend: "argos"           # argos, nllb, deepl, libretranslate
  cache:
    enabled: true            # кэширование переводов
    max_size: 10000
    ttl: 3600                # секунды

# Формат вывода
output:
  format: "txt"              # json, srt, vtt, txt
  include_word_timestamps: false
  include_translation: true
  include_confidence: true

# Диаризация (разделение спикеров)
diarization:
  strategy: "channel"        # channel, ml, hybrid
  channel_map:
    microphone: "SPEAKER_LOCAL"
    system: "SPEAKER_REMOTE"
```

Использование своего конфига:
```bash
python -m voxfusion capture --config my_config.yaml --translate ru
```

---

## 🧪 Тестирование

### 1. Запуск всех тестов

```bash
python -m pytest tests/unit/ -v
```

**Результат:** ✅ 134 теста пройдены за 4.84 секунды

### 2. Тесты перевода

```bash
python -m pytest tests/unit/test_translation_backends.py -v
python -m pytest tests/unit/test_translation_cache.py -v
```

### 3. Простой тест потоковой записи

```bash
python test_streaming.py
```

Это запустит запись на 30 секунд и покажет транскрипцию в реальном времени.

---

## 📊 Примеры использования

### Пример 1: Транскрипция YouTube видео

```bash
# 1. Откройте YouTube видео в браузере
# 2. Запустите захват системного звука
python -m voxfusion capture --source system --output-format srt > youtube.srt

# Теперь у вас есть SRT субтитры!
```

### Пример 2: Перевод онлайн-звонка Zoom

```bash
# Во время звонка запустите:
python -m voxfusion capture --source both --translate ru --output-format json > call.json

# Получите:
# - Вашу речь: SPEAKER_LOCAL
# - Речь собеседника: SPEAKER_REMOTE
# - Оба переведены на русский
```

### Пример 3: Транскрипция лекции

```bash
# Запись длинной лекции с сохранением в файл
python -m voxfusion capture \
  --source microphone \
  --duration 3600 \
  --output-format json \
  --translate ru > lecture.json

# Или в текстовый формат
python -m voxfusion capture \
  --source microphone \
  --output-format txt > lecture.txt
```

### Пример 4: Живые субтитры для видеоконференций

```bash
# Показывает субтитры в реальном времени на экране
python -m voxfusion capture --source both --output-format txt --translate ru
```

---

## 🐛 Решение проблем

### Проблема: "sounddevice is not installed"

**Решение:**
```bash
pip install sounddevice
```

### Проблема: "argostranslate is not installed"

**Решение:**
```bash
pip install argostranslate
# Затем загрузите языковые пакеты (см. выше)
```

### Проблема: Не слышно системного звука (source=system)

**Решение (Windows):**
1. Откройте "Параметры звука" → "Дополнительные параметры звука"
2. Во вкладке "Запись" должно быть устройство "Стерео микшер" или "Loopback"
3. Если его нет, обновите драйверы аудио

### Проблема: Модель faster-whisper не загружается

**Решение:**
```bash
# Загрузите модель вручную
pip install faster-whisper --upgrade

# Проверьте версию
python -c "import faster_whisper; print(faster_whisper.__version__)"
```

### Проблема: Медленная транскрипция

**Решение:**
```bash
# Используйте меньшую модель
python -m voxfusion capture --model tiny --translate ru

# Или используйте GPU (если доступно)
# В config.yaml: asr.device: "cuda"
```

---

## 📝 Дополнительные возможности

### Сохранение в файл с автоматическим именем

```bash
# С временной меткой
python -m voxfusion capture --translate ru > "capture_$(date +%Y%m%d_%H%M%S).txt"
```

### Использование с другими программами (pipe)

```bash
# Отправка результатов в другую программу
python -m voxfusion capture --output-format json | jq '.segments[].text'

# Фильтрация только русских сегментов
python -m voxfusion capture --output-format json | jq '.segments[] | select(.language=="ru")'
```

### Мониторинг в реальном времени

```bash
# Показывать только переведённый текст
python -m voxfusion capture --translate ru --output-format txt | grep "→ Translation"
```

---

## 🎯 Производительность

### Рекомендуемые настройки для разных сценариев:

**Максимальное качество (медленно):**
```yaml
asr:
  model_size: "large-v3"
  device: "cuda"
  compute_type: "float16"
  beam_size: 10
```

**Сбалансированное (рекомендуется):**
```yaml
asr:
  model_size: "small"
  device: "auto"
  compute_type: "int8"
  beam_size: 5
```

**Максимальная скорость (жертвуя качеством):**
```yaml
asr:
  model_size: "tiny"
  device: "cpu"
  compute_type: "int8"
  beam_size: 1
```

---

## ✅ Итоговый чеклист

- [x] ✅ Потоковая запись звука (микрофон/система)
- [x] ✅ Транскрипция в реальном времени (faster-whisper)
- [x] ✅ Автоопределение языка
- [x] ✅ Диаризация спикеров (channel-based)
- [x] ✅ Перевод на лету (4 бэкенда)
- [x] ✅ Кэширование переводов
- [x] ✅ Множественные форматы вывода (JSON/SRT/VTT/TXT)
- [x] ✅ 134 юнит-теста (все проходят)
- [x] ✅ CLI интерфейс
- [x] ✅ Конфигурация через YAML
- [x] ✅ Поддержка Windows (WASAPI)

---

## 🚀 Следующие шаги

1. **Попробуйте базовую запись:**
   ```bash
   python -m voxfusion capture
   ```

2. **Добавьте перевод:**
   ```bash
   # Установите Argos
   pip install argostranslate

   # Загрузите языковой пакет
   python -m argostranslate.package install en ru

   # Запустите с переводом
   python -m voxfusion capture --translate ru
   ```

3. **Экспериментируйте с настройками:**
   - Разные модели ASR (tiny → large-v3)
   - Разные источники звука (microphone/system/both)
   - Разные форматы вывода (json/srt/vtt/txt)
   - Разные бэкенды перевода (argos/nllb/deepl)

---

## 📞 Поддержка

Если что-то не работает:
1. Проверьте версию Python: `python --version` (нужна 3.11+)
2. Обновите зависимости: `pip install -e . --upgrade`
3. Запустите тесты: `python -m pytest tests/unit/ -v`
4. Включите verbose режим: `python -m voxfusion -v capture`

---

**Готово! 🎉 Теперь можно записывать и переводить звук в реальном времени!**
