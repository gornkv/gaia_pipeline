# GAIA benchmark

Установка после общих зависимостей проекта:

```bash
.venv/bin/pip install -r benchmark/gaia/requirements.txt
```

Для dataset нужен `HF_TOKEN` аккаунта, который принял условия на странице
[GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA).

Выбор benchmark:

```bash
BENCHMARK=gaia .venv/bin/python runner/start.py
```

Необязательная настройка: `GAIA_MAX_ATTEMPTS=1`.

`run.py` содержит Inspect task definitions, параметры запуска GAIA и ленивую
подготовку Docker sandbox. При параллельном старте только одна job скачивает
отсутствующий image; остальные ждут lock.
