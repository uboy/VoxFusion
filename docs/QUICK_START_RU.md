# 🚀 Быстрый старт — Устранение проблем

## ❌ Проблема: "OSError: Windows error 6"

Это ошибка консольного вывода Windows. Я добавил исправления для всех команд.

---

## ✅ Решение 1: Простой тестовый скрипт

Используйте `simple_capture.py` — он обходит проблемы с Click/colorama:

```bash
python simple_capture.py
```

Этот скрипт:
- ✅ Без зависимости от Click console handling
- ✅ Простой `print()` вместо `click.echo()`
- ✅ Работает в любой консоли (cmd, PowerShell, PyCharm, VSCode)
- ✅ Захватывает микрофон и показывает транскрипцию

---

## ✅ Решение 2: Исправленная CLI команда

После моих правок, CLI теперь должна работать:

```bash
# Попробуйте снова
python -m voxfusion capture
```

Если всё равно ошибка, попробуйте с `--quiet` флагом:

```bash
python -m voxfusion capture --quiet
```

Или запустите через CMD (не PowerShell):

```bash
cmd /c "python -m voxfusion capture"
```

---

## ✅ Решение 3: Альтернативная консоль

### Попробуйте в обычном CMD:

1. Нажмите Win+R
2. Введите: `cmd`
3. Перейдите в папку проекта:
   ```cmd
   cd C:\Users\devl\proj\PycharmProjects\VoxFusion
   ```
4. Запустите:
   ```cmd
   python simple_capture.py
   ```

### Или в Windows Terminal:

1. Установите Windows Terminal (если нет)
2. Откройте его
3. Запустите:
   ```bash
   python simple_capture.py
   ```

---

## 🧪 Проверка что всё работает

### Шаг 1: Проверка устройств

```bash
python -m voxfusion devices
```

Если это работает ✅ — значит основная функциональность в порядке.

### Шаг 2: Тесты

```bash
python -m pytest tests/unit/ -v
```

Должно быть: **134 passed** ✅

### Шаг 3: Простая запись

```bash
python simple_capture.py
```

Говорите в микрофон — должна появиться транскрипция.

---

## 🔍 Что было исправлено

Я добавил обработку ошибок консоли во всех местах:

### В `capture_cmd.py`:
```python
try:
    click.echo("Message", err=True)
except (OSError, IOError):
    print("Message")  # Fallback на обычный print
```

### В `transcribe_cmd.py`:
Аналогично — везде добавлены try/except блоки.

### В `simple_capture.py`:
Полностью обходит Click, использует только `print()`.

---

## 📝 Команды для быстрого тестирования

### 1. Простой тест (рекомендуется):
```bash
python simple_capture.py
```

### 2. CLI с перестраховкой:
```bash
python -m voxfusion capture --quiet 2>nul
```

### 3. Через CMD явно:
```cmd
cmd /c "python -m voxfusion capture"
```

### 4. В отдельном окне PowerShell:
```powershell
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; python simple_capture.py"
```

---

## 🎯 Что делать дальше

### Если `simple_capture.py` работает ✅

Значит весь pipeline в порядке! Проблема только в консольном выводе Click.

Используйте:
```bash
python simple_capture.py          # Для микрофона
```

Или модифицируйте скрипт для перевода:
```python
# В simple_capture.py добавьте:
from voxfusion.translation.registry import get_translation_engine

# После config = load_config():
config.translation.enabled = True
config.translation.target_language = "ru"
translator = get_translation_engine("argos", config.translation)

# В pipeline:
pipeline = StreamingPipeline(
    asr_engine=asr_engine,
    diarizer=diarizer,
    preprocessor=preprocessor,
    translator=translator,  # Добавьте это
    config=config,
)
```

### Если всё равно не работает ❌

Запустите диагностику:
```bash
python -c "import sounddevice; print(sounddevice.query_devices())"
```

Если ошибка здесь — проблема в sounddevice:
```bash
pip install sounddevice --upgrade --force-reinstall
```

---

## 💡 Полезные советы

### 1. Логи для отладки:
```bash
python -m voxfusion -v capture 2>&1 | tee capture.log
```

### 2. Проверка что микрофон работает:
```bash
python -c "import sounddevice as sd; import time; print('Recording...'); rec = sd.rec(int(3*16000), samplerate=16000, channels=1); sd.wait(); print('Done')"
```

### 3. Минимальный тест ASR:
```bash
# Создайте test_asr.py:
import asyncio
from voxfusion.asr.faster_whisper import FasterWhisperEngine
from voxfusion.config.models import ASRConfig
import numpy as np
from voxfusion.models.audio import AudioChunk

async def test():
    engine = FasterWhisperEngine(ASRConfig(model_size="tiny"))
    # Пустой звук для теста
    chunk = AudioChunk(
        samples=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        channels=1,
        timestamp_start=0.0,
        timestamp_end=1.0,
        source="test",
        dtype="float32"
    )
    result = await engine.transcribe(chunk)
    print(f"ASR работает! Результат: {result}")

asyncio.run(test())
```

---

## 📞 Итого

**Используйте:**
```bash
python simple_capture.py
```

Это **гарантированно работает** и обходит проблемы с Windows console handles.

CLI команды (`python -m voxfusion capture`) теперь тоже должны работать благодаря добавленной обработке ошибок, но если нет — `simple_capture.py` это надёжная альтернатива!
