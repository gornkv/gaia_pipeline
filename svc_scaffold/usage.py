from __future__ import annotations

# Per-request accounting of every generation a scaffold request makes.
#
# inspect only ever sees the one response the proxy returns, so discarded
# branches and helper summariser calls never reached its token counts. Each
# call records itself here instead, and the API layer attaches the totals.

from contextvars import ContextVar
from typing import Any

_acc: ContextVar[dict | None] = ContextVar("scaffold_usage", default=None)

_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def new_accumulator() -> dict:
    """Start accounting for one inspect-visible request."""
    acc = {f: 0 for f in _FIELDS}
    acc["generations"] = 0
    acc["helper_generations"] = 0
    acc["helper_total_tokens"] = 0
    _acc.set(acc)
    return acc


def record(usage: Any, *, helper: bool = False) -> None:
    """Add one generation. Called for every base-model and helper call."""
    acc = _acc.get()
    if acc is None:
        return
    if helper:
        acc["helper_generations"] += 1
        if isinstance(usage, dict) and isinstance(usage.get("total_tokens"), int):
            acc["helper_total_tokens"] += usage["total_tokens"]
        return
    acc["generations"] += 1
    if not isinstance(usage, dict):
        return
    for f in _FIELDS:
        v = usage.get(f)
        if isinstance(v, int):
            acc[f] += v
