"""Внешняя модель с локальным окружением инструментов benchmark."""

# Прокси и параметры запросов общие: изменения external автоматически применяются здесь.
from backends.external import command, enrich_payload


def sandbox() -> str:
    return "local"
