"""Преобразует один sample из Inspect EvalLog в компактные поля CSV."""

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


def read_metrics(path: Path) -> dict[str, str]:
    """Извлекает результат, токены и стоимость tools из закрытого eval-лога."""
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(path))
    sample = log.samples[0]
    events = [_dump(event) for event in (sample.events or [])]
    tools = [event for event in events if event.get("event") == "tool"]
    # Это сумма времени tools, поэтому при параллельности она может превысить wall time задачи.
    tool_time = sum(_seconds(event.get("timestamp"), event.get("completed")) for event in tools)
    usage = sum(getattr(item, "total_tokens", 0) or 0 for item in (sample.model_usage or {}).values())
    scores = [_dump(score).get("value") for score in (sample.scores or {}).values()]
    timed_out = any(
        event.get("event") == "sample_limit" and event.get("type") == "time" for event in events
    )
    return {
        # Ошибка выполнения отличается от корректно завершённого, но неверного ответа.
        "result": "timeout" if timed_out else "sample_error" if sample.error else "solved" if any(
            score == "C" or score is True or isinstance(score, (int, float)) and score > 0 for score in scores
        ) else "unsolved",
        "tool_calls": str(len(tools)),
        "tool_time_s": f"{tool_time:.3f}",
        "tokens_used": str(usage),
        # Подробности уже находятся в .eval и inspect.stderr; CSV хранит только признак ошибки.
        "error": "1" if sample.error else "",
    }
