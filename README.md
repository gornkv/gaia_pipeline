# GAIA Inspect Pipeline

Плановый runner запускает один model backend и выполняет подходящие строки CSV
параллельно. Централизованного диспетчера и базы данных нет.

## Подготовка

Нужны Python 3.10+ и токен Hugging Face с доступом к `gaia-benchmark/GAIA`.
Docker нужен только для локального backend `llama_docker`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/inspect-tool-support post-install
.venv/bin/python -m playwright install chromium
```

Перед запуском экспортируйте секрет:

- создайте токен на [Hugging Face → Access Tokens](https://huggingface.co/settings/tokens);
- запросите доступ и примите условия на странице [GAIA dataset](https://huggingface.co/datasets/gaia-benchmark/GAIA).

Если токен уже создан, откройте страницу Access Tokens и нажмите значок
копирования у нужного токена. Если эта машина уже авторизована через
`hf auth login`, токен можно перенести из локального хранилища прямо в окружение,
не выводя его в терминал:

```bash
export HF_TOKEN="$(.venv/bin/python -c 'from huggingface_hub import get_token; print(get_token() or "")')"
test -n "$HF_TOKEN" && echo 'HF_TOKEN loaded'
```

Скопировать локально сохранённый токен в буфер обмена macOS:

```bash
.venv/bin/python -c 'from huggingface_hub import get_token; print(get_token() or "", end="")' | pbcopy
```

```bash
export HF_TOKEN=hf_...
```

Чтобы использовать шаблон, один раз скопируйте его и заполните значения:

```bash
cp .env.example .env
```

Runner автоматически загружает корневой `.env`. Переменные, переданные при
запуске или уже экспортированные в shell, имеют приоритет над файлом. Например,
`PARALLEL=4 .venv/bin/python runner/start.py` переопределит только `PARALLEL`.

## План

По умолчанию runner читает [plans/default.csv](plans/default.csv). Одна строка —
один запуск sample. Пустой `status` означает pending. `job_id` должен быть
уникальным целым числом; `sample_no` — порядковый номер sample внутри выбранных
`task` и `split`.

Чистый шаблон с двумя задачами находится в
[plans/default.csv.example](plans/default.csv.example). Чтобы заменить текущий
рабочий план шаблоном:

```bash
cp plans/default.csv.example plans/default.csv
```

Сейчас поддерживаются `task`: `gaia`, `gaia_level1`, `gaia_level2`,
`gaia_level3`; `split`: `validation` или `test`.

В `feature` можно указать `baseline`, одну фичу или несколько через запятую без
префикса `FEATURE_`. Порядок сохраняется и задаёт порядок middleware, например
`voi,context_compaction`.

`token_limit` зарезервирован и пока не применяется. После выполнения worker
заполняет `tokens_used`, `result`, число и время tool calls, общее время и путь к
Inspect-логу.

`status` принимает `running`, `done`, `error` или `interrupted`. Поле `result`
уточняет итог: `solved`, `unsolved`, `timeout`, `sample_error`, `error` либо
`interrupted`. Успешно выполненная, но нерешённая задача имеет `status=done` и
`result=unsolved`.

Колонка `error` содержит только `1` или пустое значение. Подробности находятся в
`eval_log`, `_state/jobs/<job_id>/inspect.stderr` либо
`_state/jobs/<job_id>/error.log`, поэтому traceback не раздувает CSV.

Во время работы строка хранит `worker_id`, `started_at` и `heartbeat_at`. При
следующем запуске `running` без свежего heartbeat переводится в `interrupted`,
но не запускается повторно автоматически. Порог задаёт `WORKER_STALE_AFTER`, по
умолчанию 120 секунд; для повтора очистите `status` вручную.

CSV можно дополнять во время прогона. Worker перечитывает его под короткой
блокировкой перед каждым изменением, пишет атомарно и сохраняет `.csv.bak`.
Сохранение файла из IDE ровно одновременно с worker всё ещё может создать
конфликт, потому что редактор не использует эту блокировку.

На Ubuntu изменения удобно наблюдать стандартной утилитой `watch`:

```bash
watch -d -n 1 'cat plans/default.csv'
```

Другой план задаётся переменной окружения:

```bash
PLAN=plans/experiment.csv ... .venv/bin/python runner/start.py
```

Worker забирает только строки, где одновременно совпадают `model` и `backend`;
остальные строки остаются для других worker'ов.

## Локальный тест через Docker

Локальный backend использует официальный CPU image llama.cpp и небольшую
Qwen3.5-0.8B. Первый запуск скачивает image и около 750 MB файлов модели; они
остаются в `_state/llama-cache`.

```bash
MODEL='unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M' \
BACKEND=llama_docker \
PARALLEL=1 \
MODEL_CONTEXT_LIMIT=24000 \
.venv/bin/python runner/start.py
```

Чтобы выполнить другой CSV:

```bash
PLAN=plans/local.csv \
MODEL='unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M' \
BACKEND=llama_docker \
MODEL_CONTEXT_LIMIT=24000 \
.venv/bin/python runner/start.py
```

## vLLM на GPU

На GPU-машине один раз установите vLLM в окружение runner:

```bash
.venv/bin/pip install 'vllm>=0.22.1'
```

`GPU`, `MODEL`, `BACKEND` и `PARALLEL` передаются окружением. Для GPU 0:

```bash
GPU=0 \
BACKEND=vllm \
MODEL='QuantTrio/Qwen3.5-9B-AWQ' \
PARALLEL=4 \
MODEL_CONTEXT_LIMIT=65536 \
.venv/bin/python runner/start.py
```

Для второй карты замените `GPU=0` на `GPU=1`. По умолчанию каждому GPU
назначаются отдельные model/scaffold-порты. Второй worker на уже занятом GPU
завершится с ошибкой.

Параллельность не уменьшается автоматически: vLLM получает `PARALLEL` как
`--max-num-seqs`, а ошибка запуска или нехватка памяти останавливает worker.

Дополнительные настройки при необходимости:

```text
MODEL_PORT=18080
SCAFFOLD_PORT_BASE=19080
# Обязательный единый лимит backend и контекстных техник.
MODEL_CONTEXT_LIMIT=24000
GPU_MEMORY_UTILIZATION=0.90
VLLM_BIN=vllm
TOOL_CALL_PARSER=qwen3_coder
LLAMA_DOCKER_IMAGE=ghcr.io/ggml-org/llama.cpp:server
LLAMA_CACHE=_state/llama-cache
BASE_MODEL_REQUEST_TIMEOUT=1800
GAIA_SAMPLE_TIME_LIMIT=1800
GAIA_MESSAGE_LIMIT=100
GAIA_TOOL_TIMEOUT=120
GAIA_MAX_ATTEMPTS=1
WORKER_STALE_AFTER=120
```

## Изоляция задач

Model backend живёт весь прогон. Каждая задача получает собственные scaffold,
inspect-tool-support, runtime-каталог и Inspect-лог. После завершения задачи её
процессы останавливаются; модель остаётся загруженной для следующей строки.

Sandbox и браузер автоматически следуют способу запуска backend: если имя
backend заканчивается на `docker`, они запускаются в Docker; иначе — локально.

## Логи

Все runtime-артефакты находятся в игнорируемом Git каталоге `_state/`:

```text
_state/
├── worker-<backend или gpu>/
│   ├── backend.log
│   └── worker.lock
└── jobs/<job_id>/
    ├── scaffold.log
    ├── tool-support.log
    ├── inspect.stdout
    ├── inspect.stderr
    ├── error.log
    ├── runtime/
    └── inspect-logs/*.eval
```

Для конкретной строки плана начинайте с `_state/jobs/<job_id>/`. Структурированный
результат находится в `.eval`; `error.log` создаётся только при ошибке самого
runner, а ошибки Inspect дополнительно видны в `inspect.stderr`.
