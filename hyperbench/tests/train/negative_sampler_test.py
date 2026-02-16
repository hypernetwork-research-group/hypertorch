import pytest
import torch

from hyperbench.train import RandomNegativeSampler
from hyperbench.types import HData


@pytest.fixture
def mock_hdata_with_attr():
    return HData(
        x=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        edge_index=torch.tensor([[0, 1, 2], [0, 1, 2]]),
        edge_attr=torch.tensor([[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]),
        num_nodes=3,
        num_edges=3,
    )


@pytest.fixture
def mock_hdata_no_attr():
    return HData(
        x=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        edge_index=torch.tensor([[0, 1, 2], [0, 0, 1]]),
        edge_attr=None,
        num_nodes=3,
        num_edges=2,
    )


def test_random_negative_sampler_invalid_args():
    with pytest.raises(ValueError, match="num_negative_samples must be positive, got 0"):
        RandomNegativeSampler(num_negative_samples=0, num_nodes_per_sample=2)

    with pytest.raises(ValueError, match="num_nodes_per_sample must be positive, got 0"):
        RandomNegativeSampler(num_negative_samples=2, num_nodes_per_sample=0)


def test_random_negative_sampler_sample_too_many_nodes(mock_hdata_with_attr):
    sampler = RandomNegativeSampler(num_negative_samples=2, num_nodes_per_sample=10)
    with pytest.raises(
        ValueError,
        match="Asked to create samples with 10 nodes, but only 3 nodes are available",
    ):
        sampler.sample(mock_hdata_with_attr)


def test_random_negative_sampler_with_edge_attr(mock_hdata_with_attr):
    sampler = RandomNegativeSampler(num_negative_samples=2, num_nodes_per_sample=2)
    result = sampler.sample(mock_hdata_with_attr)

    assert result.num_edges == 2
    assert result.x.shape[0] <= mock_hdata_with_attr.x.shape[0]
    assert result.edge_index.shape[0] == 2
    assert result.edge_index.shape[1] == 4  # 2 negative hyperedges * 2 nodes per negative hyperedge
    assert result.edge_attr is not None
    assert result.edge_attr.shape[0] == 2


def test_random_negative_sampler_sample_no_edge_attr(mock_hdata_no_attr):
    sampler = RandomNegativeSampler(num_negative_samples=1, num_nodes_per_sample=2)
    result = sampler.sample(mock_hdata_no_attr)

    assert result.num_edges == 1
    assert result.x.shape[0] <= mock_hdata_no_attr.x.shape[0]
    assert result.edge_index.shape[0] == 2
    assert result.edge_index.shape[1] == 2  # 1 negative hyperedge * 2 nodes per negative hyperedge
    assert result.edge_attr is None


def test_random_negative_sampler_sample_unique_nodes(mock_hdata_with_attr):
    sampler = RandomNegativeSampler(num_negative_samples=3, num_nodes_per_sample=2)
    result = sampler.sample(mock_hdata_with_attr)

    node_ids = result.edge_index[0]
    edge_ids = result.edge_index[1]

    # All node indices in hyperedge_index should be valid
    assert torch.all(node_ids < mock_hdata_with_attr.num_nodes)

    # No duplicate node indices within a single hyperedge
    for edge_id in edge_ids.unique():
        edge_mask = torch.isin(edge_ids, edge_id)
        unique_edge_nodes = node_ids[edge_mask].unique()

        assert len(unique_edge_nodes) == sampler.num_nodes_per_sample


@pytest.mark.parametrize(
    "return_0based_negatives",
    [
        pytest.param(True, id="return_0based_negatives=True"),
        pytest.param(False, id="return_0based_negatives=False"),
    ],
)
def test_random_negative_sampler_sample_nodes_depending_on_return_0based_negatives(
    mock_hdata_no_attr, return_0based_negatives
):
    sampler = RandomNegativeSampler(
        num_negative_samples=1,
        num_nodes_per_sample=2,
        return_0based_negatives=return_0based_negatives,
    )
    result = sampler.sample(mock_hdata_no_attr)

    node_ids = result.edge_index[0]

    assert torch.all(node_ids >= 0)
    assert torch.all(node_ids < mock_hdata_no_attr.num_nodes)

    max_node_id = max(node_ids)
    if return_0based_negatives:
        for node_id in range(max_node_id + 1):
            assert node_id in node_ids
