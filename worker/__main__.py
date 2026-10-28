import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import venv


def main():
    root = Path(__file__).resolve().parent.parent
    python = root / ".venv/bin/python"
    requirements = [root / "requirements.txt", *sorted(root.glob("benchmarks/*/requirements.txt"))]
    fingerprint = hashlib.sha256(b"".join(path.read_bytes() for path in [Path(__file__), *requirements])).hexdigest()
    runtime = root / "_state/runtime/python"
    runtime.mkdir(parents=True, exist_ok=True)
    ready = runtime / "install.ready"
    # Один процесс готовит окружение; остальные запуски используют готовую .venv.
    with (runtime / "install.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not python.is_file() or not ready.exists() or ready.read_text() != fingerprint:
            venv.EnvBuilder(with_pip=True).create(root / ".venv")
            log = root / "_state/logs/install/python.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            print(f"Installing Python dependencies; log: {log}", flush=True)
            with log.open("a") as output:
                subprocess.run([str(python), "-m", "pip", "install", *[arg for path in requirements for arg in ("-r", str(path))]],
                               check=True, stdout=output, stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))
            ready.write_text(fingerprint)
    os.execv(str(python), [str(python), "-m", "worker.worker"])


if __name__ == "__main__":
    main()
