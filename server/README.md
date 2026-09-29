# VoxFusion HTTP API server

HTTP-сервис транскрибации поверх канонического пайплайна VoxFusion
(`PipelineOrchestrator.transcribe_file` — тот же путь, что у CLI
`voxfusion transcribe`). Работает строго на CPU, не трогает GPU.

## Запуск одной командой

```bash
sudo systemctl enable --now voxfusion-api
```

Юнит: `deploy/voxfusion-api.service` (установлен в
`/etc/systemd/system/voxfusion-api.service`). Логи: `journalctl -u voxfusion-api -f`.

Адрес: `http://<bm1-host>:8500` (uvicorn, host 0.0.0.0, workers 1).

## Конфигурация

Секреты и параметры — `/home/dmazur/secrets/voxfusion-api.env`
(EnvironmentFile юнита, chmod 600):

| `VOXFUSION_API_MAX_JOBS_HISTORY` | записей в истории задач | 500 |
| Переменная | Назначение | По умолчанию |
|---|---|---|
| `VOXFUSION_API_TOKEN` | Bearer-токен (обязателен; без него все защищённые эндпоинты отдают 401) | — |
| `VOXFUSION_API_MODEL` | Модель faster-whisper (id из asr_catalog) | `small` |
| `VOXFUSION_API_DEVICE` | Устройство ASR | `cpu` |
| `VOXFUSION_API_CPU_THREADS` | Потоки ctranslate2 | `6` |
| `VOXFUSION_API_DATA_DIR` | Каталог uploads | `/home/dmazur/voxfusion-api` |
| `VOXFUSION_API_RETENTION_HOURS` | Чистка файлов упавших задач и завершённых задач в истории (успешные удаляются сразу) | `24` |

Отсчёт 24 ч: файлы - по mtime (момент завершения загрузки на диск), записи
задач в памяти - по времени создания. После рестарта сервиса записи теряются,
но их файлы лежат в uploads до ретеншн-чистки; перезапустить их через API
нельзя (реестра нет) - клиент делает новый POST.
| `VOXFUSION_API_MAX_UPLOAD_GB` | Предел размера загрузки | `32` |

Юнит жёстко выставляет `CUDA_VISIBLE_DEVICES=` (CPU-only), `OMP_NUM_THREADS=6`,
`CPUQuota=200%` — чтобы не мешать тренировке на том же хосте.

## API

Авторизация: `Authorization: Bearer $VOXFUSION_API_TOKEN` на всех эндпоинтах,
кроме `GET /healthz`.

### GET /healthz

```bash
curl -s http://localhost:8500/healthz
# {"status":"ok","model":"faster-whisper/small","device":"cpu","jobs_active":0}
```

### POST /v1/transcribe — поставить задачу

multipart-загрузка, файл стримится на диск (можно несколько ГБ):

```bash
TOKEN=$(sudo cat /home/dmazur/secrets/voxfusion-api.env | grep ^VOXFUSION_API_TOKEN= | cut -d= -f2)
curl -s -X POST http://localhost:8500/v1/transcribe \
  -H "Authorization: Bearer $TOKEN" \
  -F file=@meeting.mp4 \
  -F language=ru            # опционально; по умолчанию auto
  -F include_segments=true  # опционально; вернуть сегменты с таймкодами
# {"job_id":"ab12cd34ef56","status":"queued"}
```

Поддерживаются любые форматы, которые читает VoxFusion: wav/flac/aiff
напрямую, mp3/m4a/ogg/opus/wma и видео mp4/mkv/webm/mov/avi... — через
ffmpeg-extraction внутри пайплайна.

### GET /v1/jobs/{job_id} — статус и результат

```bash
curl -s http://localhost:8500/v1/jobs/ab12cd34ef56 -H "Authorization: Bearer $TOKEN"
# {"job_id":"...","status":"done","text":"...","segments":[...],
#  "error":null,"processing_time_s":42.1,"audio_duration_s":3600.0,
#  "model":"gigaam/gigaam-v3-rnnt", ...}
```

`status`: `queued` -> `running` -> `done` | `error`. Ошибка декодирования
файла (например, «wav» из нулей) попадает в `error` задачи, а не в HTTP 500.
Поля времени: `processing_time_s` - сколько обрабатывалась задача;
`audio_duration_s` - длительность самого аудио (из пайплайна; `null`, если
пайплайн её не отдал, например при ошибке).

### GET /v1/jobs — последние задачи

```bash
curl -s http://localhost:8500/v1/jobs -H "Authorization: Bearer $TOKEN"
```

## Архитектура

- `server/app.py` — FastAPI: авторизация, стриминг загрузки в
  `~/voxfusion-api/uploads/`, реестр задач в памяти, ретеншн 24 ч.
- `server/worker.py` — один FIFO-поток; `PipelineOrchestrator` создаётся
  один раз, модель не перезагружается между задачами; language
  подставляется per-job (worker однопоточный).
- Результаты хранятся в памяти процесса (последние 500 задач). Файл
  загрузки — переходящий артефакт: при успехе (done) worker удаляет его
  сразу; при ошибке (error) файл остаётся на диске, чтобы можно было
  перезапустить транскрибацию после исправления, и снимается ретеншном
  через 24 ч.

## Известные ограничения

- Очередь и результаты — в памяти: перезапуск сервиса теряет историю задач
  (файлы упавших задач остаются до ретеншн-чистки).
- Транскрипция CPU-only и ограничена CPUQuota=200% — большие файлы
  обрабатываются медленно (минуты на час аудио для модели small).
- Одна одновременная транскрибация (workers=1, один воркер) — остальные
  задачи ждут в FIFO-очереди.
- Не поднимайте `VOXFUSION_API_DEVICE` выше `cpu`: GPU занят тренировкой 1C.

## Примечания

- Модель грузится лениво на первой задаче (не на старте сервиса): после рестарта первая транскрибация медленнее.
- Задачи и результаты живут в памяти процесса: рестарт сервиса теряет историю задач (файлы чистятся по ретеншну).
