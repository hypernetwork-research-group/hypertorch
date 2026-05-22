import pytest
import torch

from hyperbench.utils import (
    assign_hyperedge_label_to_nodes,
    is_inductive_setting,
    is_transductive_setting,
)


def test_assign_hyperedge_label_to_nodes_maps_labels_to_node_sets():
    hyperedge_index = torch.tensor(
        [
            [2, 0, 1, 3, 4, 2],
            [1, 0, 0, 2, 2, 1],
        ]
    )
    y = torch.tensor([1, 0, 1])

    labels_by_nodes = assign_hyperedge_label_to_nodes(
        hyperedge_index=hyperedge_index,
        y=y,
        num_hyperedges=3,
    )

    expected: dict[frozenset[int], float] = {
        frozenset({0, 1}): 1,
        frozenset({2}): 0,
        frozenset({3, 4}): 1,
    }

    assert labels_by_nodes.keys() == expected.keys()
    for nodes, expected_label in expected.items():
        assert torch.equal(torch.as_tensor(labels_by_nodes[nodes]), torch.as_tensor(expected_label))


def test_assign_hyperedge_label_to_nodes_includes_empty_hyperedge_slots():
    hyperedge_index = torch.tensor([[0, 1], [0, 0]])
    y = torch.tensor([1, 0])

    labels_by_nodes = assign_hyperedge_label_to_nodes(
        hyperedge_index=hyperedge_index,
        y=y,
        num_hyperedges=2,
    )

    expected: dict[frozenset[int], float] = {
        frozenset({0, 1}): 1,
        frozenset(): 0,
    }

    assert labels_by_nodes.keys() == expected.keys()
    for nodes, expected_label in expected.items():
        assert torch.equal(torch.as_tensor(labels_by_nodes[nodes]), torch.as_tensor(expected_label))


def test_assign_hyperedge_label_to_nodes_returns_empty_mapping_without_hyperedges():
    hyperedge_index = torch.empty((2, 0), dtype=torch.long)
    y = torch.empty((0,), dtype=torch.float)

    labels_by_nodes = assign_hyperedge_label_to_nodes(
        hyperedge_index=hyperedge_index,
        y=y,
        num_hyperedges=0,
    )

    assert labels_by_nodes == {}


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
