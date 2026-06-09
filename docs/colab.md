1. Откройте Google Colab.
2. Создайте notebook: File → New notebook.
3. Выберите:
   Runtime → Change runtime type.
4. В поле Hardware accelerator выберите GPU.
5. Нажмите Save.
6. Нажмите Connect справа сверху.

```sh
%cd /content
!git clone https://github.com/gornkv/gaia_pipeline.git
```

затем зайти в файлы, показать скрытые, создать .env скопировав все из .env.example
добавить HF_TOKEN, аккаунт должен согласится с правилами на https://huggingface.co/datasets/gaia-benchmark/GAIA
установить BASE_MODEL_RUNNER_TYPE=llama16GB

Этот режим скачивает готовый CUDA llama.cpp server из релизов ai-dock/llama.cpp-cuda и запускает GGUF:

```env
BASE_MODEL_RUNNER_TYPE=llama16GB
```

Команда для запуска:
```sh
%cd /content/gaia_pipeline
!sh runner/start.sh
```

хелпер для быстрого запуска

```sh
%cd /content

![ ! -d gaia_pipeline/.git ] && \
  git clone https://github.com/gornkv/gaia_pipeline.git && \
  cp gaia_pipeline/.env.example gaia_pipeline/.env && \
  echo 'HF_TOKEN=hf_token' >> gaia_pipeline/.env

%cd /content/gaia_pipeline
!git pull
!sh runner/start.sh

```

вьювер логов
```sh
!nohup _state/.venv/bin/inspect view --host 0.0.0.0 --port 7575 --log-dir inspect-logs start > /tmp/inspect-view.log 2>&1 &

from google.colab.output import eval_js
print(eval_js("google.colab.kernel.proxyPort(7575)"))
```
