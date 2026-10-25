"""Нативный llama.cpp с выбором ускорителя: общий KV-пул и отдельный лимит каждого слота."""

import fcntl
import os
import platform
import shutil
import subprocess
import tarfile
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
LLAMA_VERSION = "11850"
PLATFORMS = {
    ("Linux", "x86_64"): ("linux-64", "cpu_mkl_*"),
    ("Linux", "aarch64"): ("linux-aarch64", "cpu_openblas_*"),
    ("Darwin", "x86_64"): ("osx-64", "cpu_accelerate_*"),
    ("Darwin", "arm64"): ("osx-arm64", "cpu_accelerate_*"),
}


def install() -> None:
    """Обеспечивает доступность llama-server в PATH текущего процесса."""
    if shutil.which("llama-server"):
        return
    subdir, build = PLATFORMS[platform.system(), platform.machine()]
    accelerator = os.environ["BACKEND_LLAMA_ACCELERATOR"]
    if accelerator == "cuda":
        build = "cuda129_*"
    directory = ROOT / "_state/cache/llama-conda"
    prefix = directory / f"{LLAMA_VERSION}-{subdir}-{accelerator}"
    # Сервер из кэша запускается по тому же имени, что и системная установка.
    os.environ["PATH"] = os.pathsep.join([str(prefix / "bin"), *os.get_exec_path()])
    runtime = ROOT / "_state/runtime/llama"
    runtime.mkdir(parents=True, exist_ok=True)
    ready = runtime / f"{prefix.name}.ready"
    if ready.exists():
        return
    with (runtime / "install.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # Пока мы ждали lock, другой worker мог завершить установку.
        if ready.exists():
            return
        directory.mkdir(parents=True, exist_ok=True)
        manager = directory / f"micromamba-{subdir}"
        if not manager.is_file():
            archive = directory / f"micromamba-{subdir}.tar.bz2"
            with urlopen(f"https://micro.mamba.pm/api/micromamba/{subdir}/latest", timeout=120) as response, archive.open("wb") as output:
                shutil.copyfileobj(response, output)
            # Извлекаем только исполняемый файл, без произвольных путей из архива.
            temporary = manager.with_suffix(".tmp")
            with tarfile.open(archive) as package:
                member = package.getmember("bin/micromamba")
                if not member.isfile():
                    raise RuntimeError("Micromamba archive has no regular executable")
                with package.extractfile(member) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output)
            temporary.chmod(0o755)
            temporary.replace(manager)
        log_path = ROOT / "_state/logs/install/llama.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as log:
            subprocess.run([
                str(manager), "--no-rc", "--root-prefix", str(directory / "root"),
                "create", "--yes", "--prefix", str(prefix), "--override-channels",
                "--channel", "conda-forge", f"llama.cpp={LLAMA_VERSION}={build}",
            ], check=True, stdout=log, stderr=subprocess.STDOUT)
        ready.touch()


def enrich_payload(payload: dict) -> dict:
    budget = os.environ["BACKEND_LLAMA_THINKING_BUDGET_LIMIT"]
    if budget and int(budget) > 0:
        payload["reasoning_budget_tokens"] = int(budget)
    return payload


def run(port: int) -> None:
    install()
    model = os.environ["WORKER_MODEL"]
    cmd = ["llama-server", "-hf", model,
           "--alias", model, "--api-key", "local", "--host", "127.0.0.1", "--port", str(port),
           "--ctx-size", os.environ["BACKEND_LLAMA_KV_POOL_SIZE"], "--parallel", os.environ["WORKER_PARALLEL"],
           # Слоты делят один KV-пул, но каждый диалог ограничен WORKER_MODEL_CONTEXT_LIMIT.
           "--kv-unified", "--kv-unified-per-slot", os.environ["WORKER_MODEL_CONTEXT_LIMIT"],
           # Continuous batching смешивает работу слотов; prompt cache переиспользует префикс.
           # Не сохраняем idle-слоты в дополнительный RAM-кэш автоматически.
           "--cont-batching", "--cache-prompt", "--no-cache-idle-slots",
           # Jinja включает шаблон модели и tools; mmproj отключён, запуск только текстовый.
           "--jinja", "--no-mmproj", "--metrics",
           # Переносим слои на WORKER_GPU_NUM; q4_0 уменьшает расход памяти KV-кэша ценой квантизации.
           "--n-gpu-layers", "99", "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
           # Flash Attention снижает рабочую память; дополнительный RAM prompt-кэш отключён.
           "--flash-attn", "on", "--cache-ram", "0",
           # Не подгоняем профиль автоматически; логический batch и физический microbatch.
           "--fit", "off", "--batch-size", os.environ['BACKEND_LLAMA_BATCH_SIZE'],
           "--ubatch-size", os.environ['BACKEND_LLAMA_UBATCH_SIZE']]
    if os.environ["BACKEND_LLAMA_ACCELERATOR"] == "cpu":
        cmd.extend(["--device", "none"])
    # На CUDA выбираем ту же WORKER_GPU_NUM, для которой worker удерживает блокировку.
    if os.environ["WORKER_GPU_NUM"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["WORKER_GPU_NUM"]
    # Заменяем процесс Python сервером: PID и группа процессов сохраняются для остановки worker.
    os.execvp(cmd[0], cmd)
