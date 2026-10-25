"""Запуск прокси через python -m backends.external."""

import sys
from . import run

run(int(sys.argv[1]))
