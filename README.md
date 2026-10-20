# Benchmark Pipeline

Плановый runner запускает один model backend и выполняет подходящие строки CSV
параллельно. Централизованного диспетчера и базы данных нет.

## Подготовка

Нужны Python 3.10+. Дополнительные зависимости и доступы описаны в каталоге
выбранного benchmark, например [benchmark/gaia](benchmark/gaia/README.md).
Docker нужен только для benchmark/backend, которые его используют.

```bash
python3 -m venv .venv
export XDG_CACHE_HOME="$PWD/_state/cache"
export XDG_DATA_HOME="$PWD/_state/cache/data"
export HF_HOME="$PWD/_state/cache/huggingface"
export INSPECT_EVALS_CACHE_DIR="$PWD/_state/cache/inspect_evals"
export PLAYWRIGHT_BROWSERS_PATH="$PWD/_state/cache/playwright"
export PIP_CACHE_DIR="$PWD/_state/cache/pip"
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -r benchmark/gaia/requirements.txt
.venv/bin/inspect-tool-support post-install
.venv/bin/python -m playwright install chromium
```

Перед запуском экспортируйте секрет:

- создайте токен на [Hugging Face → Access Tokens](https://huggingface.co/settings/tokens);
- примите условия доступа для выбранной модели и dataset, если они ограничены.

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

Допустимые `task` и `split` определяет выбранный через `BENCHMARK` адаптер.
Настройки GAIA перечислены в [benchmark/gaia](benchmark/gaia/README.md).

В `feature` можно указать `baseline`, одну фичу или несколько через запятую без
префикса `FEATURE_`. Порядок сохраняется и задаёт порядок middleware, например
`voi,context_compaction`.

В `feature_envs` задаются дополнительные переменные окружения конкретной строки:
`VOI_TOOL_BUDGET=10` или `"VOI_TOOL_BUDGET=10,CTX_OPTION=value"`. Значение с
запятыми необходимо заключать в CSV-кавычки.

`token_limit` зарезервирован и пока не применяется. После выполнения worker
заполняет `tokens_used` максимальным размером входного контекста одного model
request, а также `result`, число и время tool calls, общее время и путь к Inspect-логу.

`status` принимает `running`, `done`, `error` или `interrupted`. Поле `result`
уточняет итог: `solved`, `unsolved`, `context_overflow`, `timeout`,
`sample_error`, `error` либо `interrupted`. Успешно выполненная, но нерешённая
задача имеет `status=done` и `result=unsolved`.

Колонка `error` содержит только `1` или пустое значение. Подробности находятся в
`eval_log`, `_state/logs/<worker>-<timestamp>/jobs/<job_id>-<timestamp>/inspect.stderr`
либо соседний `error.log`, поэтому traceback не раздувает CSV.

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
остаются в `_state/cache/llama`.

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

Внешний OpenAI-compatible API не запускается и не останавливается worker'ом:

```bash
BACKEND=external \
MODEL='provider/model-name' \
MODEL_API_BASE_URL='https://api.example.com/v1' \
MODEL_API_KEY='secret' \
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
LLAMA_CACHE=_state/cache/llama
# Timeout model request вычисляется как MODEL_CONTEXT_LIMIT / скорость.
MODEL_AVERAGE_TOKENS_PER_SECOND=20
SAMPLE_TIME_LIMIT=1800
MESSAGE_LIMIT=100
TOOL_TIMEOUT=120
WORKER_STALE_AFTER=120
```

## Изоляция задач

Model backend живёт весь прогон. Каждая задача получает собственные scaffold,
inspect-tool-support, временный runtime-каталог и Inspect-лог. Runtime создаётся
коротким путём в системном temp из-за лимита длины Unix-сокета и удаляется после
задачи; все диагностические файлы сразу пишутся в `_state/logs`. После завершения
задачи её процессы останавливаются; модель остаётся загруженной для следующей строки.

При `SIGKILL`, падении ОС или потере питания cleanup не выполняется, поэтому в
системном temp может остаться каталог `benchmark-<job_id>-*`. В нём находится только
неработающий сокет, полезные для разбора логи не теряются. После аварии строка
остаётся `running`, пока следующий worker не отметит её `interrupted`; для повтора
нужно вручную очистить runtime-поля строки в CSV.

Sandbox и браузер автоматически следуют способу запуска backend: если имя
backend заканчивается на `docker`, они запускаются в Docker; иначе — локально.

## Логи

Все постоянные логи и управляемые проектом кэши находятся в игнорируемом Git
каталоге `_state/`:

```text
_state/
├── cache/
│   ├── llama/
│   ├── docker/
│   ├── data/
│   ├── huggingface/
│   ├── inspect_ai/
│   ├── inspect_evals/
│   ├── playwright/
│   ├── torch/
│   ├── torchinductor/
│   ├── triton/
│   └── vllm/
└── logs/
    └── worker-<backend или gpu>-<UTC timestamp>/
        ├── backend.log
        └── jobs/
            └── <job_id>-<UTC timestamp>/
                ├── scaffold.log
                ├── tool-support.log
                ├── inspect.stdout
                ├── inspect.stderr
                ├── error.log
                └── inspect-logs/*.eval
```

Для конкретной строки плана начинайте с каталога worker, указанного в `worker_id`,
затем переходите в `jobs/<job_id>-<timestamp>/`. Структурированный результат
находится в `.eval`; `error.log` создаётся только при ошибке самого runner, а
ошибки Inspect дополнительно видны в `inspect.stderr`. Стабильные lock-файлы
лежат прямо в корне `_state`, поскольку lock внутри уникального каталога
запуска не защищал бы от второго worker.

Старые каталоги, созданные до появления worker-run группировки, находятся в
`_state/logs/legacy-unassigned/jobs`: прежний формат не записывал связь каталога
с конкретным запуском worker, поэтому надёжно восстановить её задним числом нельзя.
