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
если нужно хранить `_state` на Google Drive между Colab-сессиями, перед запуском выполнить Python-ячейку:

```python
from google.colab import drive
drive.mount("/content/drive")
```

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

```python
from google.colab import drive

%cd /content
![ ! -d gaia_pipeline/.git ] && \
  git clone https://github.com/gornkv/gaia_pipeline.git && \
  cd gaia_pipeline && \
  cp .env.example .env && \
  echo 'HF_TOKEN=hf_aaaaaaaaaaaaaaaaa' >> .env && \
  echo 'GAIA_SAMPLE_START=1' >> .env && \
  echo 'GAIA_SAMPLE_END=1' >> .env && \
  sleep 1

drive.mount("/content/drive")
%cd /content
!mkdir -p drive/MyDrive/gaia_pipeline/_state
![ ! -L gaia_pipeline/_state ] && \
  rm -rf gaia_pipeline/_state && \
  ln -s /content/drive/MyDrive/gaia_pipeline/_state gaia_pipeline/_state

%cd /content/gaia_pipeline
!git pull
!sh runner/start.sh
```

Потом https://drive.google.com/drive/my-drive и там в корне будет папка gaia_pipeline

вьювер логов
```sh
!nohup .venv/bin/inspect view --host 0.0.0.0 --port 7575 --log-dir _state/inspect-logs start > /tmp/inspect-view.log 2>&1 &

from google.colab.output import eval_js
print(eval_js("google.colab.kernel.proxyPort(7575)"))
```
