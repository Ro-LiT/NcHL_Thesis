"""Central metadata for every EvoGym task in the frozen protocol."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    """Pinned task difficulty, horizon, and thesis selection status."""

    env_id: str
    difficulty: str
    expected_horizon: int
    selected: bool


_TASKS = (
    TaskSpec("Walker-v0", "easy", 500, True),
    TaskSpec("Carrier-v0", "not preclassified", 500, True),
    TaskSpec("Jumper-v0", "easy", 500, True),
    TaskSpec("Thrower-v0", "not preclassified", 300, True),
)
_TASKS_BY_ID = {task.env_id: task for task in _TASKS}


def get_task_spec(env_id: str) -> TaskSpec:
    """Return task metadata or raise a descriptive lookup error."""

    try:
        return _TASKS_BY_ID[env_id]
    except KeyError as error:
        choices = ", ".join(_TASKS_BY_ID)
        raise KeyError(f"unknown task {env_id!r}; choose one of: {choices}") from error


def selected_tasks() -> tuple[TaskSpec, ...]:
    """Return the four official thesis tasks."""

    return tuple(task for task in _TASKS if task.selected)


def audited_tasks() -> tuple[TaskSpec, ...]:
    """Return every task in the official audit catalog."""

    return _TASKS
