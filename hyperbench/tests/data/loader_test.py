import pytest
import torch

from hyperbench.data import DataLoader, Dataset
from hyperbench.types import HData
from unittest.mock import MagicMock
from hyperbench import utils


@pytest.fixture
def mock_dataset_single_sample():
    # Full dataset: 3 nodes (0,1,2), 2 hyperedges (0,1)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    hyperedge_index = torch.tensor([[0, 1, 1, 2], [0, 0, 1, 1]])
    hyperedge_attr = torch.tensor([[0.5], [0.7]])
    hdata = HData(x=x, hyperedge_index=hyperedge_index, hyperedge_attr=hyperedge_attr)

    dataset = MagicMock(spec=Dataset)
    dataset.hdata = hdata
    dataset.__len__.return_value = 3
    # __getitem__ returns HData with global IDs and no features
    dataset.__getitem__.return_value = HData.from_hyperedge_index(hyperedge_index)

    return dataset


@pytest.fixture
def mock_dataset_single_sample_with_weights():
    # Full dataset: 3 nodes (0,1,2), 2 hyperedges (0,1)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    hyperedge_index = torch.tensor([[0, 1, 1, 2], [0, 0, 1, 1]])
    hyperedge_attr = torch.tensor([[0.5], [0.7]])
    hyperedge_weights = torch.tensor([[0.8], [0.9]])
    hdata = HData(
        x=x,
        hyperedge_index=hyperedge_index,
        hyperedge_weights=hyperedge_weights,
        hyperedge_attr=hyperedge_attr,
    )

    dataset = MagicMock(spec=Dataset)
    dataset.hdata = hdata
    dataset.__len__.return_value = 3
    # __getitem__ returns HData with global IDs and no features
    dataset.__getitem__.return_value = HData.from_hyperedge_index(hyperedge_index)

    return dataset


@pytest.fixture
def mock_dataset_multiple_samples():
    # Full dataset: 5 nodes (0-4), 3 hyperedges (0-2)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    full_hyperedge_index = torch.tensor([[0, 1, 1, 2, 3, 4], [0, 0, 1, 1, 2, 2]])
    hdata = HData(x=x, hyperedge_index=full_hyperedge_index)

    # Sample 0: global nodes (0, 1, 2) in global hyperedges (0, 1)
    sample0 = HData.from_hyperedge_index(torch.tensor([[0, 1, 1, 2], [0, 0, 1, 1]]))
    # Sample 1: global nodes (3, 4) in global hyperedge 2
    sample1 = HData.from_hyperedge_index(torch.tensor([[3, 4], [2, 2]]))

    dataset = MagicMock(spec=Dataset)
    dataset.hdata = hdata
    dataset.__len__.return_value = 5
    dataset.__getitem__.side_effect = lambda idx: [sample0, sample1][idx]

    return dataset


def test_initialization_with_default_params(mock_dataset_single_sample):
    loader = DataLoader(mock_dataset_single_sample)

    assert loader.batch_size == 1
    assert loader.dataset == mock_dataset_single_sample


def test_initialization_with_custom_params(mock_dataset_single_sample):
    loader = DataLoader(mock_dataset_single_sample, batch_size=4, shuffle=True, num_workers=0)

    assert loader.batch_size == 4
    assert loader.dataset == mock_dataset_single_sample

    # num_workers is used to test that kwargs are passed correctly
    assert loader.num_workers == 0


def test_collate_single_sample(mock_dataset_single_sample):
    loader = DataLoader(mock_dataset_single_sample, batch_size=1)

    sample = mock_dataset_single_sample[0]
    batched = loader.collate([sample])

    # Features come from cached dataset.hdata indexed by global node/hyperedge IDs
    expected_x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    expected_hyperedge_attr = torch.tensor([[0.5], [0.7]])
    expected_hyeredge_index = torch.tensor([[0, 1, 1, 2], [0, 0, 1, 1]])

    assert torch.equal(batched.x, expected_x)
    assert batched.hyperedge_index.shape == (2, 4)
    assert torch.equal(batched.hyperedge_index, expected_hyeredge_index)
    assert torch.equal(
        utils.to_non_empty_edgeattr(batched.hyperedge_attr),
        expected_hyperedge_attr,
    )
    assert batched.num_nodes == 3
    assert batched.num_hyperedges == 2


def test_collate_single_sample_with_weights(mock_dataset_single_sample_with_weights):
    loader = DataLoader(mock_dataset_single_sample_with_weights, batch_size=1)

    sample = mock_dataset_single_sample_with_weights[0]
    batched = loader.collate([sample])

    # Features come from cached dataset.hdata indexed by global node/hyperedge IDs
    expected_x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    expected_hyperedge_attr = torch.tensor([[0.5], [0.7]])
    expected_hyeredge_index = torch.tensor([[0, 1, 1, 2], [0, 0, 1, 1]])
    expected_hyperedge_weights = torch.tensor([[0.8], [0.9]])

    assert torch.equal(batched.x, expected_x)
    assert batched.hyperedge_index.shape == (2, 4)
    assert torch.equal(batched.hyperedge_index, expected_hyeredge_index)
    assert torch.equal(
        utils.to_non_empty_edgeattr(batched.hyperedge_attr),
        expected_hyperedge_attr,
    )
    assert batched.num_nodes == 3
    assert batched.num_hyperedges == 2
    assert torch.equal(
        utils.to_non_empty_edgeattr(batched.hyperedge_weights), expected_hyperedge_weights
    )


def test_collate_single_sample_rebases_to_0based():
    # Full dataset: 5 nodes (0-4), 3 hyperedges (0-2)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    full_hyperedge_index = torch.tensor([[0, 1, 1, 2, 3, 4], [0, 0, 1, 1, 2, 2]])
    hyperedge_attr = torch.tensor([[0.1], [0.2], [0.3]])
    hdata = HData(x=x, hyperedge_index=full_hyperedge_index, hyperedge_attr=hyperedge_attr)

    # Single sample with non-0-based global IDs: nodes [3, 4] in hyperedge [2]
    sample = HData.from_hyperedge_index(torch.tensor([[3, 4], [2, 2]]))

    dataset = MagicMock(spec=Dataset)
    dataset.hdata = hdata
    dataset.__len__.return_value = 5
    dataset.__getitem__.return_value = sample

    loader = DataLoader(dataset, batch_size=1)
    batched = loader.collate([sample])

    # Global nodes [3, 4] should be rebased to [0, 1]
    expected_hyperedge_index = torch.tensor([[0, 1], [0, 0]])
    assert torch.equal(batched.hyperedge_index, expected_hyperedge_index)

    # Features should come from global indices 3 and 4
    expected_x = torch.tensor([[7.0, 8.0], [9.0, 10.0]])
    assert torch.equal(batched.x, expected_x)

    # Hyperedge attr should come from global hyperedge 2
    expected_hyperedge_attr = torch.tensor([[0.3]])
    assert torch.equal(utils.to_non_empty_edgeattr(batched.hyperedge_attr), expected_hyperedge_attr)

    assert batched.num_nodes == 2
    assert batched.num_hyperedges == 1


def test_collate_two_samples_no_edge_attr(mock_dataset_multiple_samples):
    loader = DataLoader(mock_dataset_multiple_samples, batch_size=2)

    sample0 = mock_dataset_multiple_samples[0]
    sample1 = mock_dataset_multiple_samples[1]
    batched = loader.collate([sample0, sample1])

    # Features indexed from cached dataset.hdata by global node IDs [0, 1, 2, 3, 4]
    expected_x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    # Global IDs already 0-based: no remapping change needed for hyperedge_index
    expected_hyperedge_index = torch.tensor([[0, 1, 1, 2, 3, 4], [0, 0, 1, 1, 2, 2]])

    assert torch.equal(batched.x, expected_x)
    assert batched.num_nodes == 5
    assert torch.equal(batched.hyperedge_index, expected_hyperedge_index)
    assert batched.num_hyperedges == 3
    assert batched.hyperedge_attr is None


def test_collate_two_samples_with_edge_attr(mock_dataset_multiple_samples):
    # hyperedge_attr comes from cached dataset.hdata, not from batch items
    mock_dataset_multiple_samples.hdata.hyperedge_attr = torch.tensor([[0.5], [0.7], [0.9]])

    sample0 = mock_dataset_multiple_samples[0]
    sample1 = mock_dataset_multiple_samples[1]

    loader = DataLoader(mock_dataset_multiple_samples, batch_size=2)
    batched = loader.collate([sample0, sample1])

    # Indexed from cached hdata by global hyperedge IDs [0,1,2]
    expected_hyperedge_attr = torch.tensor([[0.5], [0.7], [0.9]])
    assert torch.equal(utils.to_non_empty_edgeattr(batched.hyperedge_attr), expected_hyperedge_attr)


def test_collate_three_samples():
    # Full dataset: 6 nodes (0-5), 4 hyperedges (0-3)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    full_hyperedge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 3, 3]])
    hdata = HData(x=x, hyperedge_index=full_hyperedge_index)

    # Sample 0: global nodes [0,1] in global hyperedges [0,1]
    sample0 = HData.from_hyperedge_index(torch.tensor([[0, 1], [0, 1]]))
    # Sample 1: global node [2] in global hyperedge [2]
    sample1 = HData.from_hyperedge_index(torch.tensor([[2], [2]]))
    # Sample 2: global nodes [3,4,5] in global hyperedge [3]
    sample2 = HData.from_hyperedge_index(torch.tensor([[3, 4, 5], [3, 3, 3]]))

    dataset = MagicMock(spec=Dataset)
    dataset.hdata = hdata
    dataset.__len__.return_value = 6
    dataset.__getitem__.side_effect = lambda idx: [sample0, sample1, sample2][idx]

    loader = DataLoader(dataset, batch_size=3)
    batched = loader.collate([sample0, sample1, sample2])

    total_nodes_from_all_samples = (
        2 + 1 + 3
    )  # 6 nodes total, 2 from sample0, 1 from sample1, 3 from sample2
    assert batched.num_nodes == total_nodes_from_all_samples
    assert batched.x.size(0) == total_nodes_from_all_samples

    total_hyperedges_from_all_samples = (
        2 + 1 + 1
    )  # 4 hyperedges total, 2 from sample0, 1 from sample1, 1 from sample2
    assert batched.num_hyperedges == total_hyperedges_from_all_samples

    # Sample 0: nodes [0,1], edges [0,1]
    # Sample 1: nodes [2], edges [2] (offset by 2 nodes, 2 edges)
    # Sample 2: nodes [3,4,5], edges [3] (offset by 3 nodes, 3 edges)
    expected_hyperedge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 3, 3]])
    assert torch.equal(batched.hyperedge_index, expected_hyperedge_index)


def test_collate_empty_hyperedge_index():
    # Dataset with some nodes but empty hyperedge_index
    hdata = HData(x=torch.empty((2, 2)), hyperedge_index=torch.empty((2, 0), dtype=torch.long))

    sample0 = HData.from_hyperedge_index(torch.empty((2, 0), dtype=torch.long))
    sample1 = HData.from_hyperedge_index(torch.empty((2, 0), dtype=torch.long))

    dataset = MagicMock(spec=Dataset)
    dataset.hdata = hdata
    dataset.__len__.return_value = 2
    dataset.__getitem__.side_effect = lambda idx: [sample0, sample1][idx]

    loader = DataLoader(dataset, batch_size=2)
    batched = loader.collate([sample0, sample1])

    assert batched.num_nodes == 0
    assert batched.hyperedge_index.size(1) == 0
    assert batched.num_hyperedges == 0


def test_collate_multi_dimensional_hyperedge_attributes():
    # Dataset: 4 nodes, 2 hyperedges with 3-dim attrs
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    hyperedge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    hyperedge_attr = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    hdata = HData(x=x, hyperedge_index=hyperedge_index, hyperedge_attr=hyperedge_attr)

    # Sample 0: global nodes [0,1] in global hyperedge [0]
    sample0 = HData.from_hyperedge_index(torch.tensor([[0, 1], [0, 0]]))
    # Sample 1: global nodes [2,3] in global hyperedge [1]
    sample1 = HData.from_hyperedge_index(torch.tensor([[2, 3], [1, 1]]))

    dataset = MagicMock(spec=Dataset)
    dataset.hdata = hdata

    loader = DataLoader(dataset, batch_size=2)
    batched = loader.collate([sample0, sample1])

    expected_hyperedge_attr = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    assert torch.equal(utils.to_non_empty_edgeattr(batched.hyperedge_attr), expected_hyperedge_attr)


def test_collate_when_dataset_no_hyperedge_attr_presence():
    # When cached dataset has no hyperedge_attr, batched result has no hyperedge_attr
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 1]])
    hdata = HData(x=x, hyperedge_index=hyperedge_index)  # No hyperedge_attr

    sample0 = HData.from_hyperedge_index(torch.tensor([[0, 1], [0, 0]]))
    sample1 = HData.from_hyperedge_index(torch.tensor([[2], [1]]))

    dataset = MagicMock(spec=Dataset)
    dataset.hdata = hdata

    loader = DataLoader(dataset, batch_size=2)
    batched = loader.collate([sample0, sample1])

    assert batched.hyperedge_attr is None


def test_collate_sample_full_hypergraph_returns_cached_hdata(mock_dataset_single_sample):
    loader = DataLoader(mock_dataset_single_sample, sample_full_hypergraph=True)

    batch = [mock_dataset_single_sample[0]]
    batched = loader.collate(batch)

    expected_hdata: HData = mock_dataset_single_sample.hdata
    assert torch.equal(batched.x, expected_hdata.x)
    assert torch.equal(batched.hyperedge_index, expected_hdata.hyperedge_index)
    assert torch.equal(
        utils.to_non_empty_edgeattr(batched.hyperedge_attr), expected_hdata.hyperedge_attr
    )


def test_collate_with_explicit_num_nodes_and_edges():
    # num_nodes and num_hyperedges are derived from unique IDs in hyperedge_index
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 1]])
    hdata = HData(x=x, hyperedge_index=hyperedge_index)

    sample0 = HData.from_hyperedge_index(torch.tensor([[0, 1], [0, 0]]))
    sample1 = HData.from_hyperedge_index(torch.tensor([[2], [1]]))

    dataset = MagicMock(spec=Dataset)
    dataset.hdata = hdata
    dataset.__len__.return_value = 3
    dataset.__getitem__.side_effect = lambda idx: [sample0, sample1][idx]

    loader = DataLoader(dataset, batch_size=2)
    batched = loader.collate([sample0, sample1])

    assert batched.num_nodes == 2 + 1  # 3 nodes total, 2 from sample0 and 1 from sample1
    assert batched.num_hyperedges == 1 + 1  # 2 hyperedges total, 1 from sample0 and 1 from sample1


def test_iteration_over_dataloader():
    n_samples = 5

    # Full dataset: 10 nodes (2 per sample), 5 hyperedges (1 per sample)
    x = torch.randn(10, 3)
    node_row = list(range(10))
    edge_row = [i // 2 for i in range(10)]
    hdata = HData(x=x, hyperedge_index=torch.tensor([node_row, edge_row]))

    # Each sample has 2 unique global nodes and 1 unique global hyperedge
    data_list = [
        HData.from_hyperedge_index(torch.tensor([[i * 2, i * 2 + 1], [i, i]]))
        for i in range(n_samples)
    ]

    dataset = MagicMock(spec=Dataset)
    dataset.hdata = hdata
    dataset.__len__.return_value = n_samples
    dataset.__getitem__.side_effect = lambda idx: data_list[idx]

    loader = DataLoader(dataset, batch_size=2)

    batch_count = 0
    for batch in loader:
        batch_count += 1
        assert batch.x.size(1) == 3  # 3 features per node
        dataset.__getitem__.assert_called()

    assert batch_count == 3  # 5 samples with batch_size=2 -> 3 batches (2 + 2 + 1)
    assert dataset.__getitem__.call_count == n_samples  # Ensure all samples were accessed


def test_collate_with_hyperedge_sampled_batch():
    # Full dataset with 4 nodes and 2 hyperedges
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    hyperedge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 2]])
    hyperedge_attr = torch.tensor([[0.5], [0.7], [0.9]])
    hdata = HData(x=x, hyperedge_index=hyperedge_index, hyperedge_attr=hyperedge_attr)

    # hyperedge 0 -> nodes 0 and 1
    sample0 = HData.from_hyperedge_index(torch.tensor([[0, 1], [0, 0]]))
    # hyperedge 2 -> node 3
    sample1 = HData.from_hyperedge_index(torch.tensor([[3], [2]]))

    dataset = MagicMock(spec=Dataset)
    dataset.hdata = hdata

    loader = DataLoader(dataset, batch_size=2)
    batched = loader.collate([sample0, sample1])

    assert batched.num_nodes == 3  # Nodes 0, 1 from sample0 and node 3 from sample1
    assert batched.num_hyperedges == 2  # Hyperedges 0 from sample0 and 2 from sample1

    expected_x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [7.0, 8.0]])
    assert torch.equal(batched.x, expected_x)

    # 0-based rebasing:
    # - global nodes [0, 1, 3] -> local nodes [0, 1, 2]
    # - global hyperedges [0, 2] -> local hyperedges [0, 1]
    expected_hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 1]])
    assert torch.equal(batched.hyperedge_index, expected_hyperedge_index)

    expected_hyperedge_attr = torch.tensor([[0.5], [0.9]])  # From global hyperedges 0 and 2
    assert torch.equal(utils.to_non_empty_edgeattr(batched.hyperedge_attr), expected_hyperedge_attr)


def test_collate_with_node_sampled_batch():
    # Full dataset with 4 nodes and 2 hyperedges
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    hyperedge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 2]])
    hdata = HData(x=x, hyperedge_index=hyperedge_index)

    # Samples contain node's incident hyperedges to simulate NODE strategy
    # Node 0 -> hyperedge 0 (nodes 0, 1)
    sample0 = HData.from_hyperedge_index(torch.tensor([[0, 1], [0, 0]]))
    # Node 3 -> hyperedge 2 (nodes 3)
    sample1 = HData.from_hyperedge_index(torch.tensor([[3], [2]]))

    dataset = MagicMock(spec=Dataset)
    dataset.hdata = hdata

    loader = DataLoader(dataset, batch_size=2)
    batched = loader.collate([sample0, sample1])

    assert batched.num_nodes == 3  # Nodes 0, 1 from sample0 and node 3 from sample1
    assert batched.num_hyperedges == 2  # Hyperedges 0 from sample0 and 2 from sample1

    expected_x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [7.0, 8.0]])
    assert torch.equal(batched.x, expected_x)

    # 0-based rebasing:
    # - global nodes [0, 1, 3] -> local nodes [0, 1, 2]
    # - global hyperedges [0, 2] -> local hyperedges [0, 1]
    expected_hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 1]])
    assert torch.equal(batched.hyperedge_index, expected_hyperedge_index)

    assert batched.hyperedge_attr is None
