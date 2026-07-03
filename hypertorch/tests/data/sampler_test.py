import pytest
import torch
import re

from typing import Any, cast
from hypertorch.data import (
    BaseSampler,
    HyperedgeSampler,
    NodeSampler,
    SamplingStrategyEnum,
    create_sampler_from_strategy,
)
from hypertorch.types import HData, TaskEnum


@pytest.fixture
def mock_four_node_two_hyperedge_hdata():
    x = torch.ones((4, 1), dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.long)
    return HData(x=x, hyperedge_index=hyperedge_index, num_nodes=4, num_hyperedges=2)


@pytest.fixture
def mock_single_node_single_hyperedge_hdata():
    x = torch.ones((1, 1), dtype=torch.float)
    hyperedge_index = torch.tensor([[0], [0]], dtype=torch.long)
    return HData(x=x, hyperedge_index=hyperedge_index, num_nodes=1, num_hyperedges=1)


@pytest.fixture
def mock_empty_hdata():
    x = torch.ones((0, 0), dtype=torch.float)
    hyperedge_index = torch.zeros((2, 0), dtype=torch.long)
    return HData(x=x, hyperedge_index=hyperedge_index, num_nodes=0, num_hyperedges=0)


def test_base_sampler_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseSampler()


@pytest.mark.parametrize(
    "sampling_strategy",
    [
        pytest.param("hyperedge", id="hyperedge_literal"),
        pytest.param(SamplingStrategyEnum.HYPEREDGE, id="hyperedge_enum"),
    ],
)
def test_create_sampler_from_strategy_hyperedge(sampling_strategy):
    sampler = create_sampler_from_strategy(sampling_strategy)
    assert isinstance(sampler, HyperedgeSampler)


@pytest.mark.parametrize(
    "sampling_strategy",
    [
        pytest.param("node", id="node_literal"),
        pytest.param(SamplingStrategyEnum.NODE, id="node_enum"),
    ],
)
def test_create_sampler_from_strategy_node(sampling_strategy):
    sampler = create_sampler_from_strategy(sampling_strategy)
    assert isinstance(sampler, NodeSampler)


def test_create_sampler_from_strategy_rejects_unsupported_strategy():
    with pytest.raises(
        ValueError,
        match=re.escape("Unsupported sampling strategy: 'edge'."),
    ):
        create_sampler_from_strategy(cast(Any, "edge"))


def test_hyperedge_sampling_single_index(mock_four_node_two_hyperedge_hdata):
    sampler = HyperedgeSampler()
    result = sampler.sample(0, mock_four_node_two_hyperedge_hdata)

    # hyperedge 0 has nodes 0 and 1
    assert result.hyperedge_index.shape == (2, 2)
    assert result.hyperedge_index.dtype == torch.long
    assert result.num_hyperedges == 1
    assert torch.equal(result.hyperedge_index[0], torch.tensor([0, 1], dtype=torch.long))
    assert torch.equal(result.hyperedge_index[1], torch.tensor([0, 0], dtype=torch.long))
    assert result.hyperedge_attr is None
    assert result.x.shape == (0, 0)


def test_hyperedge_sampling_index_list(mock_four_node_two_hyperedge_hdata):
    sampler = HyperedgeSampler()
    result = sampler.sample([0, 1], mock_four_node_two_hyperedge_hdata)

    # hyperedge 0 (nodes 0, 1) + hyperedge 1 (nodes 2, 3)
    assert result.hyperedge_index.shape == (2, 4)
    assert result.num_hyperedges == 2
    assert torch.equal(result.hyperedge_index[0], torch.tensor([0, 1, 2, 3], dtype=torch.long))
    assert torch.equal(result.hyperedge_index[1], torch.tensor([0, 0, 1, 1], dtype=torch.long))
    assert result.hyperedge_attr is None
    assert result.x.shape == (0, 0)


def test_hyperedge_sampling_len(mock_four_node_two_hyperedge_hdata):
    sampler = HyperedgeSampler()
    assert sampler.len(mock_four_node_two_hyperedge_hdata) == 2


def test_hyperedge_sampling_len_counts_only_target_hyperedges_for_hyperlink_prediction():
    hdata = HData(
        x=torch.arange(4, dtype=torch.float).unsqueeze(1),
        hyperedge_index=torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.long),
        target_hyperedge_mask=torch.tensor([False, True], dtype=torch.bool),
    )

    assert HyperedgeSampler().len(hdata) == 1


def test_hyperedge_sampling_maps_positions_to_target_hyperedge_ids():
    hdata = HData(
        x=torch.arange(6, dtype=torch.float).unsqueeze(1),
        hyperedge_index=torch.tensor(
            [[0, 1, 2, 3, 4, 5], [0, 0, 1, 1, 2, 2]],
            dtype=torch.long,
        ),
        target_hyperedge_mask=torch.tensor([False, True, True], dtype=torch.bool),
    )

    # Hyperedge 0 is not a target, hyperedges 1 and 2 are targets
    # So, sampling hyperedge 0 returns the first target hyperedge (1)
    # and sampling hyperedge 1 returns the second target hyperedge (2)
    result = HyperedgeSampler().sample(0, hdata)

    assert torch.equal(
        result.hyperedge_index,
        torch.tensor([[2, 3], [1, 1]], dtype=torch.long),
    )


def test_sample_single_node_graph_hyperedge_sampler(mock_single_node_single_hyperedge_hdata):
    sampler = HyperedgeSampler()
    result = sampler.sample(0, mock_single_node_single_hyperedge_hdata)

    assert result.hyperedge_index.shape == (2, 1)
    assert result.num_hyperedges == 1
    assert torch.equal(result.hyperedge_index[0], torch.tensor([0], dtype=torch.long))
    assert torch.equal(result.hyperedge_index[1], torch.tensor([0], dtype=torch.long))


def test_sample_hyperedge_of_size_one_hyperedge_sampler(mock_single_node_single_hyperedge_hdata):
    sampler = HyperedgeSampler()
    result = sampler.sample(0, mock_single_node_single_hyperedge_hdata)

    # Hyperedge 0 has only node 0
    assert result.hyperedge_index.shape == (2, 1)
    assert result.num_hyperedges == 1
    assert torch.equal(result.hyperedge_index[0], torch.tensor([0], dtype=torch.long))
    assert torch.equal(result.hyperedge_index[1], torch.tensor([0], dtype=torch.long))


def test_node_sampling_single_index(mock_four_node_two_hyperedge_hdata):
    sampler = NodeSampler()
    result = sampler.sample(0, mock_four_node_two_hyperedge_hdata)

    # Node 0 is in hyperedge 0 (nodes 0, 1), so we get all incidences of hyperedge 0
    assert result.hyperedge_index.shape == (2, 2)
    assert result.hyperedge_index.dtype == torch.long
    assert result.num_hyperedges == 1
    assert torch.equal(result.hyperedge_index[0], torch.tensor([0, 1], dtype=torch.long))
    assert torch.equal(result.hyperedge_index[1], torch.tensor([0, 0], dtype=torch.long))
    assert result.hyperedge_attr is None
    assert result.x.shape == (0, 0)


def test_node_sampling_index_list(mock_four_node_two_hyperedge_hdata):
    sampler = NodeSampler()
    result = sampler.sample([0, 2], mock_four_node_two_hyperedge_hdata)

    # Node 0 -> hyperedge 0 (nodes 0, 1)
    # Node 2 -> hyperedge 1 (nodes 2, 3)
    assert result.hyperedge_index.shape == (2, 4)
    assert result.num_hyperedges == 2
    assert torch.equal(result.hyperedge_index[0], torch.tensor([0, 1, 2, 3], dtype=torch.long))
    assert torch.equal(result.hyperedge_index[1], torch.tensor([0, 0, 1, 1], dtype=torch.long))
    assert result.hyperedge_attr is None
    assert result.x.shape == (0, 0)


def test_node_sampling_len(mock_four_node_two_hyperedge_hdata):
    sampler = NodeSampler()
    assert sampler.len(mock_four_node_two_hyperedge_hdata) == 4


def test_node_sampling_len_counts_all_nodes_for_hyperlink_prediction():
    hdata = HData(
        x=torch.arange(5, dtype=torch.float).unsqueeze(1),
        hyperedge_index=torch.tensor([[0, 1, 2, 2, 3, 4], [0, 0, 0, 1, 1, 1]], dtype=torch.long),
        target_node_mask=torch.tensor([False, True, False, True, False], dtype=torch.bool),
        y=torch.arange(2, dtype=torch.float),
        task=TaskEnum.HYPERLINK_PREDICTION,
    )

    assert NodeSampler().len(hdata) == 5


def test_node_sampling_len_counts_only_target_nodes_for_node_classification():
    hdata = HData(
        x=torch.arange(5, dtype=torch.float).unsqueeze(1),
        hyperedge_index=torch.tensor([[0, 1, 2, 2, 3, 4], [0, 0, 0, 1, 1, 1]], dtype=torch.long),
        target_node_mask=torch.tensor([False, True, False, True, False], dtype=torch.bool),
        y=torch.arange(5, dtype=torch.long),
        task=TaskEnum.NODE_CLASSIFICATION,
    )

    assert NodeSampler().len(hdata) == 2


def test_node_sampling_marks_only_seed_targets_in_context_for_node_classification():
    hdata = HData(
        x=torch.arange(5, dtype=torch.float).unsqueeze(1),
        hyperedge_index=torch.tensor([[0, 1, 2, 2, 3, 4], [0, 0, 0, 1, 1, 1]], dtype=torch.long),
        target_node_mask=torch.tensor([False, True, False, True, False], dtype=torch.bool),
        y=torch.arange(5, dtype=torch.long),
        task=TaskEnum.NODE_CLASSIFICATION,
    )

    result = NodeSampler().sample([0, 1], hdata)

    assert torch.equal(
        result.hyperedge_index,
        torch.tensor([[0, 1, 2, 2, 3, 4], [0, 0, 0, 1, 1, 1]], dtype=torch.long),
    )
    assert torch.equal(
        result.target_node_mask,
        torch.tensor([False, True, False, True, False], dtype=torch.bool),
    )


def test_node_sampling_validates_target_node_index_bounds_for_node_classification():
    hdata = HData(
        x=torch.arange(5, dtype=torch.float).unsqueeze(1),
        hyperedge_index=torch.tensor([[0, 1, 2, 2, 3, 4], [0, 0, 0, 1, 1, 1]], dtype=torch.long),
        target_node_mask=torch.tensor([False, True, False, True, False], dtype=torch.bool),
        y=torch.arange(5, dtype=torch.long),
        task=TaskEnum.NODE_CLASSIFICATION,
    )

    with pytest.raises(IndexError, match=re.escape("Node ID 2 is out of bounds (0, 1).")):
        NodeSampler().sample(2, hdata)


def test_sample_single_node_graph_node_sampler(mock_single_node_single_hyperedge_hdata):
    sampler = NodeSampler()
    result = sampler.sample(0, mock_single_node_single_hyperedge_hdata)

    assert result.hyperedge_index.shape == (2, 1)
    assert result.num_hyperedges == 1
    assert torch.equal(result.hyperedge_index[0], torch.tensor([0], dtype=torch.long))
    assert torch.equal(result.hyperedge_index[1], torch.tensor([0], dtype=torch.long))


def test_sample_hyperedge_of_size_one_node_sampler(mock_single_node_single_hyperedge_hdata):
    sampler = NodeSampler()
    result = sampler.sample(0, mock_single_node_single_hyperedge_hdata)

    # Node 0 is in hyperedge 0 which has only node 0
    assert result.hyperedge_index.shape == (2, 1)
    assert result.num_hyperedges == 1
    assert torch.equal(result.hyperedge_index[0], torch.tensor([0], dtype=torch.long))
    assert torch.equal(result.hyperedge_index[1], torch.tensor([0], dtype=torch.long))


@pytest.mark.parametrize(
    "sampler",
    [
        pytest.param(NodeSampler(), id="node_sampler"),
        pytest.param(HyperedgeSampler(), id="hyperedge_sampler"),
    ],
)
def test_sample_empty_index_raises(mock_four_node_two_hyperedge_hdata, sampler):
    with pytest.raises(ValueError, match=re.escape("Index list cannot be empty.")):
        sampler.sample([], mock_four_node_two_hyperedge_hdata)


@pytest.mark.parametrize(
    "sampler, label",
    [
        pytest.param(NodeSampler(), "Node ID", id="node_sampler_node_id_invalid"),
        pytest.param(
            HyperedgeSampler(), "Hyperedge ID", id="hyperedge_sampler_hyperedge_id_invalid"
        ),
    ],
)
def test_sample_index_out_of_bounds_raises(mock_four_node_two_hyperedge_hdata, sampler, label):
    with pytest.raises(IndexError, match=rf"{label} 99 is out of bounds"):
        sampler.sample(99, mock_four_node_two_hyperedge_hdata)


@pytest.mark.parametrize(
    "sampler, index_list",
    [
        pytest.param(
            NodeSampler(), [0, 1, 2, 3, 4], id="node_sampler_index_list_too_large"
        ),  # 5 > 4 nodes
        pytest.param(
            HyperedgeSampler(), [0, 1, 2], id="hyperedge_sampler_index_list_too_large"
        ),  # 3 > 2 hyperedges
    ],
)
def test_sample_index_list_too_large_raises(
    mock_four_node_two_hyperedge_hdata, sampler, index_list
):
    with pytest.raises(ValueError, match=re.compile("Index list length .* cannot exceed")):
        sampler.sample(index_list, mock_four_node_two_hyperedge_hdata)


@pytest.mark.parametrize(
    "sampler",
    [
        pytest.param(NodeSampler(), id="node_sampler"),
        pytest.param(HyperedgeSampler(), id="hyperedge_sampler"),
    ],
)
def test_sample_returns_correct_hdata(mock_four_node_two_hyperedge_hdata, sampler):
    result = sampler.sample(0, mock_four_node_two_hyperedge_hdata)

    assert result.x.shape == (0, 0)
    assert result.hyperedge_attr is None


@pytest.mark.parametrize(
    "sampler",
    [
        pytest.param(NodeSampler(), id="node_sampler"),
        pytest.param(HyperedgeSampler(), id="hyperedge_sampler"),
    ],
)
def test_sample_empty_hyperedge_index_len(mock_empty_hdata, sampler):
    assert sampler.len(mock_empty_hdata) == 0


@pytest.mark.parametrize(
    "sampler",
    [
        pytest.param(NodeSampler(), id="node_sampler"),
        pytest.param(HyperedgeSampler(), id="hyperedge_sampler"),
    ],
)
def test_sample_empty_hyperedge_index_empty_list_raises(mock_empty_hdata, sampler):
    with pytest.raises(ValueError, match=re.escape("Index list cannot be empty.")):
        sampler.sample([], mock_empty_hdata)


@pytest.mark.parametrize(
    "sampler, index",
    [
        pytest.param(HyperedgeSampler(), "0", id="hyperedge_string_index"),
        pytest.param(HyperedgeSampler(), True, id="hyperedge_bool_index"),
        pytest.param(NodeSampler(), "0", id="node_string_index"),
        pytest.param(NodeSampler(), False, id="node_bool_index"),
    ],
)
def test_sample_rejects_non_integer_index(mock_four_node_two_hyperedge_hdata, sampler, index):
    with pytest.raises(
        TypeError,
        match=re.escape("Index must be an integer or a list of integers."),
    ):
        sampler.sample(cast(Any, index), mock_four_node_two_hyperedge_hdata)


@pytest.mark.parametrize(
    "sampler, index",
    [
        pytest.param(HyperedgeSampler(), [0, "1"], id="hyperedge_list_with_string"),
        pytest.param(HyperedgeSampler(), [0, False], id="hyperedge_list_with_bool"),
        pytest.param(NodeSampler(), [0, "1"], id="node_list_with_string"),
        pytest.param(NodeSampler(), [0, True], id="node_list_with_bool"),
    ],
)
def test_sample_rejects_index_list_with_non_integer_items(
    mock_four_node_two_hyperedge_hdata, sampler, index
):
    with pytest.raises(
        TypeError,
        match=re.escape("Index list must contain only integers."),
    ):
        sampler.sample(cast(Any, index), mock_four_node_two_hyperedge_hdata)
