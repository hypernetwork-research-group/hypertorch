import pytest

from hyperbench.utils import (
    is_assigned_to_all,
    is_assigned_to_first,
    is_inductive_setting,
    is_transductive_setting,
    is_transductive_split,
)


@pytest.mark.parametrize(
    "node_space_assignment, expected",
    [
        pytest.param("all", True, id="all"),
        pytest.param("first", False, id="first"),
        pytest.param(None, False, id="none"),
    ],
)
def test_is_assigned_to_all(node_space_assignment, expected):
    assert is_assigned_to_all(node_space_assignment) == expected


@pytest.mark.parametrize(
    "node_space_assignment, expected",
    [
        pytest.param("first", True, id="first"),
        pytest.param("all", False, id="all"),
        pytest.param(None, False, id="none"),
    ],
)
def test_is_assigned_to_first(node_space_assignment, expected):
    assert is_assigned_to_first(node_space_assignment) == expected


@pytest.mark.parametrize(
    "node_space_setting, expected",
    [
        pytest.param("inductive", True, id="inductive"),
        pytest.param("transductive", False, id="transductive"),
        pytest.param(None, False, id="none"),
    ],
)
def test_is_inductive_setting(node_space_setting, expected):
    assert is_inductive_setting(node_space_setting) == expected


@pytest.mark.parametrize(
    "node_space_setting, expected",
    [
        pytest.param("transductive", True, id="transductive"),
        pytest.param("inductive", False, id="inductive"),
        pytest.param(None, False, id="none"),
    ],
)
def test_is_transductive_setting(node_space_setting, expected):
    assert is_transductive_setting(node_space_setting) == expected


@pytest.mark.parametrize(
    "node_space_setting, assign_node_space_to, split_num, expected",
    [
        pytest.param("inductive", "all", 0, False, id="inductive_all_first_split"),
        pytest.param("inductive", "first", 0, False, id="inductive_first_first_split"),
        pytest.param(None, "all", 0, False, id="none_setting"),
        pytest.param("transductive", "all", 0, True, id="transductive_all_first_split"),
        pytest.param("transductive", "all", 2, True, id="transductive_all_later_split"),
        pytest.param("transductive", "first", 0, True, id="transductive_first_first_split"),
        pytest.param("transductive", "first", 1, False, id="transductive_first_later_split"),
        pytest.param("transductive", None, 0, False, id="transductive_none_assignment"),
    ],
)
def test_is_transductive_split(node_space_setting, assign_node_space_to, split_num, expected):
    assert is_transductive_split(node_space_setting, assign_node_space_to, split_num) == expected
