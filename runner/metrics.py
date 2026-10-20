"""Преобразует один sample из Inspect EvalLog в компактные поля CSV."""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def _dump(value: Any) -> Any:
    # Inspect использует Pydantic events; словари упрощают доступ к полям разных версий.
    return value.model_dump() if hasattr(value, "model_dump") else value


def _seconds(start: Any, end: Any) -> float:
    """Считает неотрицательную длительность из datetime или ISO timestamp."""
    if not start or not end:
        return 0.0
    if isinstance(start, str):
        start = datetime.fromisoformat(start.replace("Z", "+00:00"))
    if isinstance(end, str):
        end = datetime.fromisoformat(end.replace("Z", "+00:00"))
    return max(0.0, (end - start).total_seconds())


def _tool_stats(tools: list[dict[str, Any]], time_limit: float) -> tuple[float, int]:
    durations = [_seconds(event.get("timestamp"), event.get("completed")) for event in tools]
    return sum(durations), sum(duration > time_limit for duration in durations)


def _max_context_tokens(events: list[dict[str, Any]]) -> int:
    """Возвращает наибольший входной контекст, включая отклонённый backend запрос."""
    maximum = 0
    for event in events:
        if event.get("event") != "model":
            continue
        usage = (_dump(event.get("output")) or {}).get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        if not isinstance(prompt_tokens, int):
            input_tokens = usage.get("input_tokens") or 0
            cached_tokens = usage.get("input_tokens_cache_read") or 0
            prompt_tokens = input_tokens + cached_tokens
        maximum = max(maximum, prompt_tokens)
        error = str(event.get("error") or "")
        match = re.search(r"request \((\d+) tokens\).*context size", error)
        if match:
            maximum = max(maximum, int(match.group(1)))
    return maximum


def _is_context_overflow(events: list[dict[str, Any]]) -> bool:
    return any(
        event.get("event") == "model"
        and re.search(r"request \(\d+ tokens\).*context size", str(event.get("error") or ""))
        for event in events
    )


def read_metrics(path: Path) -> dict[str, str]:
    """Извлекает результат, токены и стоимость tools из закрытого eval-лога."""
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(path))
    sample = log.samples[0]
    events = [_dump(event) for event in (sample.events or [])]
    tools = [event for event in events if event.get("event") == "tool"]
    # Это сумма времени tools, поэтому при параллельности она может превысить wall time задачи.
    tool_time_limit = float(os.getenv("TOOL_TIMEOUT", "180"))
    tool_time, tool_timeouts = _tool_stats(tools, tool_time_limit)
    max_context_tokens = _max_context_tokens(events)
    scores = [_dump(score).get("value") for score in (sample.scores or {}).values()]
    timed_out = any(
        event.get("event") == "sample_limit" and event.get("type") == "time" for event in events
    )
    return {
        # Ошибка выполнения отличается от корректно завершённого, но неверного ответа.
        "result": "context_overflow" if _is_context_overflow(events) else "timeout" if timed_out else "sample_error" if sample.error else "solved" if any(
            score == "C" or score is True or isinstance(score, (int, float)) and score > 0 for score in scores
        ) else "unsolved",
        "tool_calls": str(len(tools)),
        # Один ToolEvent — один вызов; тип ошибки не создаёт дополнительного инкремента.
        "tool_timeouts": str(tool_timeouts),
        "tool_time_s": f"{tool_time:.3f}",
        "tokens_used": str(max_context_tokens),
        # Подробности уже находятся в .eval и inspect.stderr; CSV хранит только признак ошибки.
        "error": "1" if sample.error else "",
    }
