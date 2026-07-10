from typing import Literal, TypeAlias, get_args

from hypertorch.utils import StrEnum


class TaskEnum(StrEnum):
    """
    Enum for supported hypergraph learning tasks.
    """

    HYPERLINK_PREDICTION = "hyperlink-prediction"
    NODE_CLASSIFICATION = "node-classification"


TaskLiteral: TypeAlias = Literal["hyperlink-prediction", "node-classification"]
"""Literal type for supported hypergraph learning tasks."""


Task: TypeAlias = TaskEnum | TaskLiteral
"""Type for supported hypergraph learning tasks, either as a TaskEnum or a string literal."""


def is_hyperedge_related_task(task: Task) -> bool:
    """
    Check if the task uses hyperedge-level targets and operations.

    Returns:
        is_hyperedge_related: True if the task is hyperedge-related, False otherwise.
    """
    # For now, we only support hyperlink prediction as a hyperedge-related task
    return task == TaskEnum.HYPERLINK_PREDICTION


def is_node_related_task(task: Task) -> bool:
    """
    Check if the task uses node-level targets and operations.

    Returns:
        is_node_related: True if the task is node-related, False otherwise.
    """
    # For now, we only support node classification as a node-related task
    return task == TaskEnum.NODE_CLASSIFICATION


def validate_task(task: Task) -> None:
    """
    Validate the learning task.

    Raises:
        ValueError: If the task is unsupported.
    """
    valid_tasks = get_args(TaskLiteral)
    if task not in valid_tasks:
        raise ValueError(f"'task' must be one of {valid_tasks}, got {task!r}.")
