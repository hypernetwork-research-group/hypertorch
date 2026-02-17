import pytest
import torch

from hyperbench.utils import (
    empty_nodefeatures,
    empty_hyperedgeindex,
    empty_edgeattr,
    to_non_empty_edgeattr,
    to_0based_ids,
)


def test_empty_edgeindex():
    result = empty_hyperedgeindex()

    assert result.shape == (2, 0)
    assert result.dtype == torch.float32


def test_empty_edgeattr_zero_edges():
    result = empty_edgeattr(num_edges=0)

    assert result.shape == (0, 0)


def test_empty_edgeattr_single_edge():
    result = empty_edgeattr(num_edges=1)

    assert result.shape == (1, 0)


def test_empty_edgeattr_with_edges():
    result = empty_edgeattr(num_edges=5)

    assert result.shape == (5, 0)


def test_to_non_empty_edgeattr_with_none():
    result = to_non_empty_edgeattr(edge_attr=None)

    assert result.shape == (0, 0)


def test_to_non_empty_edgeattr_with_tensor():
    edge_attr = torch.tensor([[0.5], [0.7], [0.9]])
    result = to_non_empty_edgeattr(edge_attr)

    assert torch.equal(result, edge_attr)
    assert result.shape == (3, 1)


def test_to_non_empty_edgeattr_with_empty_tensor():
    edge_attr = torch.empty((0, 3))
    result = to_non_empty_edgeattr(edge_attr)

    assert torch.equal(result, edge_attr)
    assert result.shape == (0, 3)


def test_to_non_empty_edgeattr_with_multi_dimensional():
    edge_attr = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    result = to_non_empty_edgeattr(edge_attr)

    assert torch.equal(result, edge_attr)
    assert result.shape == (2, 3)


def test_empty_nodefeatures():
    result = empty_nodefeatures()

    assert result.shape == (0, 0)


@pytest.mark.parametrize(
    "original_ids, ids_to_rebase, expected_result",
    [
        pytest.param(
            torch.tensor([1, 3, 3, 7]),
            torch.tensor([3, 7]),
            torch.tensor([0, 0, 1]),
            id="with_ids_to_rebase",
        ),
        pytest.param(
            torch.tensor([1, 3, 3, 7]),
            torch.tensor([1, 3, 7]),
            torch.tensor([0, 1, 1, 2]),
            id="with_ids_to_rebase_all",
        ),
        pytest.param(
            torch.tensor([5, 3, 5, 8]),
            None,
            torch.tensor([1, 0, 1, 2]),
            id="without_ids_to_rebase",
        ),
    ],
)
def test_to_0based_ids(original_ids, ids_to_rebase, expected_result):
    result = to_0based_ids(original_ids, ids_to_rebase)

    assert torch.equal(result, expected_result)
