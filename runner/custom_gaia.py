from typing import Literal

from inspect_ai import Task, task
from inspect_evals.gaia.gaia import default_solver, gaia as upstream_gaia


def _gaia(
    subset: Literal["2023_all", "2023_level1", "2023_level2", "2023_level3"],
    split: Literal["test", "validation"] = "validation",
    max_attempts: int = 1,
    code_timeout: int = 180,
) -> Task:
    return upstream_gaia(
        solver=default_solver(max_attempts, code_timeout),
        max_attempts=max_attempts,
        subset=subset,
        split=split,
    )


@task
def gaia(**kwargs: object) -> Task:
    return _gaia("2023_all", **kwargs)


@task
def gaia_level1(**kwargs: object) -> Task:
    return _gaia("2023_level1", **kwargs)


@task
def gaia_level2(**kwargs: object) -> Task:
    return _gaia("2023_level2", **kwargs)


@task
def gaia_level3(**kwargs: object) -> Task:
    return _gaia("2023_level3", **kwargs)
