import pytest
import torch
import re

from unittest.mock import MagicMock
from hyperbench.nn import HyperedgeAttrsEnricher, HyperedgeWeightsEnricher, NodeEnricher
from hyperbench.train import (
    GeneratedNodesNegativeSampler,
    RandomNegativeSampler,
    SameNodeSpaceNegativeSampler,
)
from hyperbench.types import HData


@pytest.fixture
def mock_hdata_with_attr():
    return HData(
        x=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        hyperedge_index=torch.tensor([[0, 1, 2], [0, 1, 2]]),
        hyperedge_attr=torch.tensor([[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]),
        num_nodes=3,
        num_hyperedges=3,
    )


@pytest.fixture
def mock_hdata_no_attr():
    return HData(
        x=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        hyperedge_index=torch.tensor([[0, 1, 2], [0, 0, 1]]),
        hyperedge_attr=None,
        num_nodes=3,
        num_hyperedges=3,
    )


def test_random_negative_sampler_invalid_args():
    with pytest.raises(ValueError, match="num_negative_samples must be positive, got 0"):
        RandomNegativeSampler(num_negative_samples=0, num_nodes_per_sample=2)

    with pytest.raises(ValueError, match="num_nodes_per_sample must be positive, got 0"):
        RandomNegativeSampler(num_negative_samples=2, num_nodes_per_sample=0)

    with pytest.raises(ValueError, match="max_retry must be positive, got 0"):
        RandomNegativeSampler(num_negative_samples=2, num_nodes_per_sample=1, max_retry=0)


def test_random_negative_sampler_is_same_node_space_negative_sampler():
    sampler = RandomNegativeSampler(num_negative_samples=1, num_nodes_per_sample=1)

    assert isinstance(sampler, SameNodeSpaceNegativeSampler)


def test_generated_nodes_negative_sampler_initializes_enrichers():
    class DummyGeneratedNodesNegativeSampler(GeneratedNodesNegativeSampler):
        def sample(self, hdata: HData, seed: int | None = None) -> HData:
            return hdata

    node_feature_enricher = MagicMock(spec=NodeEnricher)
    hyperedge_attr_enricher = MagicMock(spec=HyperedgeAttrsEnricher)
    hyperedge_weights_enricher = MagicMock(spec=HyperedgeWeightsEnricher)

    sampler = DummyGeneratedNodesNegativeSampler(
        node_feature_enricher=node_feature_enricher,
        hyperedge_attr_enricher=hyperedge_attr_enricher,
        hyperedge_weights_enricher=hyperedge_weights_enricher,
        return_0based_negatives=True,
    )

    assert sampler.return_0based_negatives is True
    assert sampler.node_feature_enricher is node_feature_enricher
    assert sampler.hyperedge_attr_enricher is hyperedge_attr_enricher
    assert sampler.hyperedge_weights_enricher is hyperedge_weights_enricher


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

    assert result.num_hyperedges == 2
    assert result.x.shape[0] <= mock_hdata_with_attr.x.shape[0]
    assert result.hyperedge_index.shape[0] == 2
    assert (
        result.hyperedge_index.shape[1] == 4
    )  # 2 negative hyperedges * 2 nodes per negative hyperedge
    assert (
        3 in result.hyperedge_index[1] and 4 in result.hyperedge_index[1]
    )  # New hyperedge IDs (3, 4) should be present
    assert result.hyperedge_attr is not None
    assert result.hyperedge_attr.shape[0] == 2


def test_random_negative_sampler_sample_no_edge_attr(mock_hdata_no_attr):
    sampler = RandomNegativeSampler(num_negative_samples=1, num_nodes_per_sample=2)
    result = sampler.sample(mock_hdata_no_attr)

    assert result.num_hyperedges == 1
    assert result.x.shape[0] <= mock_hdata_no_attr.x.shape[0]
    assert result.hyperedge_index.shape[0] == 2
    assert (
        result.hyperedge_index.shape[1] == 2
    )  # 1 negative hyperedge * 2 nodes per negative hyperedge
    assert 3 in result.hyperedge_index[1]  # New hyperedge ID (3) should be present
    assert result.hyperedge_attr is None


def test_random_negative_sampler_sample_with_seed_is_reproducible(mock_hdata_with_attr):
    sampler = RandomNegativeSampler(num_negative_samples=3, num_nodes_per_sample=2)

    result_a = sampler.sample(mock_hdata_with_attr, seed=123)
    result_b = sampler.sample(mock_hdata_with_attr, seed=123)

    assert torch.equal(result_a.x, result_b.x)
    assert torch.equal(result_a.hyperedge_index, result_b.hyperedge_index)
    assert result_a.hyperedge_attr is not None
    assert result_b.hyperedge_attr is not None
    assert torch.equal(result_a.hyperedge_attr, result_b.hyperedge_attr)
    assert torch.equal(result_a.y, result_b.y)


def test_random_negative_sampler_handles_missing_global_node_ids(mock_hdata_no_attr):
    mock_hdata_no_attr.global_node_ids = None

    sampler = RandomNegativeSampler(num_negative_samples=1, num_nodes_per_sample=2)
    result = sampler.sample(mock_hdata_no_attr)

    assert result.num_hyperedges == 1
    assert result.global_node_ids is not None
    assert torch.equal(result.global_node_ids, torch.arange(result.num_nodes))


def test_random_negative_sampler_sample_unique_nodes(mock_hdata_no_attr):
    sampler = RandomNegativeSampler(num_negative_samples=2, num_nodes_per_sample=2)
    result = sampler.sample(mock_hdata_no_attr)

    node_ids = result.hyperedge_index[0]
    hyperedge_ids = result.hyperedge_index[1]

    # All node indices in hyperedge_index should be valid
    assert torch.all(node_ids < mock_hdata_no_attr.num_nodes)

    # No duplicate node indices within a single hyperedge
    for hyperedge_id in hyperedge_ids.unique():
        hyperedge_mask = torch.isin(hyperedge_ids, hyperedge_id)
        unique_edge_nodes = node_ids[hyperedge_mask].unique()

        assert len(unique_edge_nodes) == sampler.num_nodes_per_sample


def test_random_negative_sampler_rejects_positive_hyperedges():
    hdata = HData(
        x=torch.tensor([[1.0], [2.0], [3.0]]),
        hyperedge_index=torch.tensor([[0, 1, 0, 2], [0, 0, 1, 1]]),
        num_nodes=3,
        num_hyperedges=2,
    )
    sampler = RandomNegativeSampler(
        num_negative_samples=1,
        num_nodes_per_sample=2,
        max_retry=100,
    )

    result = sampler.sample(hdata, seed=123)

    sampled_nodes = result.hyperedge_index[0].tolist()
    assert set(sampled_nodes) == {1, 2}


def test_random_negative_sampler_rejects_duplicate_negative_hyperedges():
    hdata = HData(
        x=torch.tensor([[1.0], [2.0], [3.0]]),
        hyperedge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=3,
        num_hyperedges=0,
    )
    sampler = RandomNegativeSampler(
        num_negative_samples=3,
        num_nodes_per_sample=1,
        max_retry=100,
    )

    result = sampler.sample(hdata, seed=123)

    sampled_nodes = result.hyperedge_index[0].tolist()
    assert sorted(sampled_nodes) == [0, 1, 2]


def test_random_negative_sampler_fails_when_unique_negatives_are_unavailable():
    hdata = HData(
        x=torch.tensor([[1.0], [2.0], [3.0]]),
        hyperedge_index=torch.tensor([[0, 1, 0, 2, 1, 2], [0, 0, 1, 1, 2, 2]]),
        num_nodes=3,
        num_hyperedges=3,
    )
    sampler = RandomNegativeSampler(num_negative_samples=1, num_nodes_per_sample=2)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Asked to create 1 unique negative samples with 2 nodes each, but only 0 are available"
        ),
    ):
        sampler.sample(hdata)


def test_random_negative_sampler_fails_when_retry_budget_is_exhausted(monkeypatch):
    hdata = HData(
        x=torch.tensor([[1.0], [2.0], [3.0]]),
        hyperedge_index=torch.tensor([[0, 1], [0, 0]]),
        num_nodes=3,
        num_hyperedges=1,
    )
    sampler = RandomNegativeSampler(
        num_negative_samples=1,
        num_nodes_per_sample=2,
        max_retry=2,
    )

    def sample_positive_hyperedge(**kwargs):
        return torch.tensor([0, 1], device=kwargs["input"].device)

    monkeypatch.setattr(torch, "multinomial", sample_positive_hyperedge)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Unable to sample 1 unique negative hyperedges after 2 attempts. "
            "Increase max_retry or request fewer samples."
        ),
    ):
        sampler.sample(hdata)


def test_random_negative_sampler_sample_new_hyperedges(mock_hdata_no_attr):
    sampler = RandomNegativeSampler(num_negative_samples=2, num_nodes_per_sample=2)
    result = sampler.sample(mock_hdata_no_attr)

    hyperedge_ids = result.hyperedge_index[1]

    # All node indices in hyperedge_index should be valid
    new_hyperedge_id_offset = mock_hdata_no_attr.num_hyperedges + sampler.num_negative_samples
    assert torch.all(hyperedge_ids < new_hyperedge_id_offset)

    hyperedge_id_offset = mock_hdata_no_attr.num_hyperedges
    for hyperedge_id in range(hyperedge_id_offset, new_hyperedge_id_offset):
        assert hyperedge_id in hyperedge_ids


@pytest.mark.parametrize(
    "return_0based_negatives",
    [
        pytest.param(True, id="return_0based_negatives=True"),
        pytest.param(False, id="return_0based_negatives=False"),
    ],
)
def test_random_negative_sampler_sample_depends_on_return_0based_negatives(
    mock_hdata_no_attr,
    return_0based_negatives,
):
    sampler = RandomNegativeSampler(
        num_negative_samples=1,
        num_nodes_per_sample=2,
        return_0based_negatives=return_0based_negatives,
    )
    result = sampler.sample(mock_hdata_no_attr)

    node_ids = result.hyperedge_index[0]

    assert torch.all(node_ids >= 0)
    assert torch.all(node_ids < mock_hdata_no_attr.num_nodes)

    if return_0based_negatives:
        for node_id in range(max(node_ids) + 1):
            assert node_id in node_ids

    hyperedge_ids = result.hyperedge_index[1]
    assert torch.all(hyperedge_ids >= 0)
    assert torch.all(
        hyperedge_ids < mock_hdata_no_attr.num_hyperedges + sampler.num_negative_samples
    )

    if return_0based_negatives:
        for hyperedge_id in range(max(hyperedge_ids) + 1):
            assert hyperedge_id in hyperedge_ids


def test_random_negative_sampler_uses_hyperedge_enrichers(mock_hdata_no_attr):
    hyperedge_attr = torch.tensor([[10.0, 11.0], [12.0, 13.0]])
    hyperedge_weights = torch.tensor([0.2, 0.3])
    hyperedge_attr_enricher = MagicMock(spec=HyperedgeAttrsEnricher)
    hyperedge_weights_enricher = MagicMock(spec=HyperedgeWeightsEnricher)
    hyperedge_attr_enricher.enrich.return_value = hyperedge_attr
    hyperedge_weights_enricher.enrich.return_value = hyperedge_weights

    sampler = RandomNegativeSampler(
        num_negative_samples=2,
        num_nodes_per_sample=2,
        hyperedge_attr_enricher=hyperedge_attr_enricher,
        hyperedge_weights_enricher=hyperedge_weights_enricher,
    )
    result = sampler.sample(mock_hdata_no_attr)

    assert result.hyperedge_attr is not None
    assert result.hyperedge_weights is not None
    assert torch.equal(result.hyperedge_attr, hyperedge_attr)
    assert torch.equal(result.hyperedge_weights, hyperedge_weights)

    attr_enricher_index = hyperedge_attr_enricher.enrich.call_args.args[0]
    weights_enricher_index = hyperedge_weights_enricher.enrich.call_args.args[0]
    assert torch.equal(attr_enricher_index, weights_enricher_index)
    assert torch.equal(attr_enricher_index[1].unique(sorted=True), torch.arange(2))
    for node_id in range(int(attr_enricher_index[0].max().item()) + 1):
        assert node_id in attr_enricher_index[0]
