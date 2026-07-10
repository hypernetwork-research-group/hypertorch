import pytest
import re

from typing import get_args, cast
from hypertorch.types import (
    Task,
    TaskEnum,
    TaskLiteral,
    is_hyperedge_related_task,
    is_node_related_task,
    validate_task,
)


def test_task_enum_values_match_supported_task_strings():
    assert TaskEnum.HYPERLINK_PREDICTION.value == "hyperlink-prediction"
    assert TaskEnum.NODE_CLASSIFICATION.value == "node-classification"


def test_task_literal_matches_supported_task_strings():
    assert get_args(TaskLiteral) == ("hyperlink-prediction", "node-classification")


@pytest.mark.parametrize(
    "task, expected_is_hyperedge_related, expected_is_node_related",
    [
        pytest.param("hyperlink-prediction", True, False, id="hyperlink_prediction_literal"),
        pytest.param(
            TaskEnum.HYPERLINK_PREDICTION,
            True,
            False,
            id="hyperlink_prediction_enum",
        ),
        pytest.param("node-classification", False, True, id="node_classification_literal"),
        pytest.param(
            TaskEnum.NODE_CLASSIFICATION,
            False,
            True,
            id="node_classification_enum",
        ),
    ],
)
def test_task_relation_helpers(task, expected_is_hyperedge_related, expected_is_node_related):
    assert is_hyperedge_related_task(task) is expected_is_hyperedge_related
    assert is_node_related_task(task) is expected_is_node_related


@pytest.mark.parametrize(
    "task",
    [
        pytest.param("hyperlink-prediction", id="hyperlink_prediction_literal"),
        pytest.param(TaskEnum.HYPERLINK_PREDICTION, id="hyperlink_prediction_enum"),
        pytest.param("node-classification", id="node_classification_literal"),
        pytest.param(TaskEnum.NODE_CLASSIFICATION, id="node_classification_enum"),
    ],
)
def test_validate_task_accepts_supported_tasks(task):
    validate_task(task)


def test_validate_task_rejects_unsupported_task():
    unsupported_task = cast(Task, "unsupported-task")

    with pytest.raises(
        ValueError,
        match=re.escape(
            "'task' must be one of ('hyperlink-prediction', 'node-classification'), "
            "got 'unsupported-task'."
        ),
    ):
        validate_task(unsupported_task)
