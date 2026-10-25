"""Запускает scaffold с портом и каталогом логов, переданными job напрямую."""

import sys
from pathlib import Path

import uvicorn

from .app import create_app

uvicorn.run(create_app(Path(sys.argv[2])), host="127.0.0.1", port=int(sys.argv[1]), ws="none")
