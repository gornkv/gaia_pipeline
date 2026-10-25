"""Запуск нативного сервера через python -m backends.llama."""

import sys
from . import run

run(int(sys.argv[1]))
