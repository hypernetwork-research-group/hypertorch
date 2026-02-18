import pytest
import torch

from hyperbench.data import DataLoader, Dataset
from hyperbench.types import HData
from unittest.mock import MagicMock
from hyperbench import utils


@pytest.fixture
def mock_dataset_single_sample():
    # Sample: 3 nodes (0, 1, 2), 2 hyperedges (0, 1)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    hyperedge_index = torch.tensor([[0, 1, 1, 2], [0, 0, 1, 1]])
    hyperedge_attr = torch.tensor([[0.5], [0.7]])

    data = HData(x=x, hyperedge_index=hyperedge_index, hyperedge_attr=hyperedge_attr)

    dataset = MagicMock(spec=Dataset)
    dataset.__len__.return_value = 1
    dataset.__getitem__.return_value = data

    return dataset


@pytest.fixture
def mock_dataset_multiple_samples():
    # Sample 0: 3 nodes, 2 hyperedges
    x0 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    hyperedge_index0 = torch.tensor([[0, 1, 1, 2], [0, 0, 1, 1]])
    data0 = HData(x=x0, hyperedge_index=hyperedge_index0)

    # Sample 1: 2 nodes, 1 hyperedge
    x1 = torch.tensor([[7.0, 8.0], [9.0, 10.0]])
    hyperedge_index1 = torch.tensor([[0, 1], [0, 0]])
    data1 = HData(x=x1, hyperedge_index=hyperedge_index1)

    dataset = MagicMock(spec=Dataset)
    dataset.__len__.return_value = 2
    dataset.__getitem__.side_effect = lambda idx: [data0, data1][idx]

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

    sample: HData = mock_dataset_single_sample[0]
    batched = loader.collate([sample])

    assert torch.equal(batched.x, sample.x)
    assert torch.equal(batched.hyperedge_index, sample.hyperedge_index)
    assert torch.equal(
        utils.to_non_empty_edgeattr(batched.hyperedge_attr),
        utils.to_non_empty_edgeattr(sample.hyperedge_attr),
    )
    assert batched.num_nodes == 3
    assert batched.num_hyperedges == 2


def test_collate_two_samples_no_edge_attr(mock_dataset_multiple_samples):
    loader = DataLoader(mock_dataset_multiple_samples, batch_size=2)

    sample0: HData = mock_dataset_multiple_samples[0]
    sample1: HData = mock_dataset_multiple_samples[1]
    batched = loader.collate([sample0, sample1])

    # Check node features are concatenated
    expected_x = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],  # Sample 0
            [7.0, 8.0],
            [9.0, 10.0],  # Sample 1
        ]
    )
    assert torch.equal(batched.x, expected_x)
    assert batched.num_nodes == 5

    # Check hyperedge_index nodes from Sample 1 are offset by 3 and hyperedges are offset by 2
    # Sample 0: nodes [0,1,2], edges [0,1]
    # Sample 1: nodes [3,4], edges [2] (offset by 3 nodes, 2 edges)
    expected_hyperedge_index = torch.tensor([[0, 1, 1, 2, 3, 4], [0, 0, 1, 1, 2, 2]])
    assert torch.equal(batched.hyperedge_index, expected_hyperedge_index)
    assert batched.num_hyperedges == 3

    assert batched.hyperedge_attr is None


def test_collate_two_samples_with_edge_attr(mock_dataset_multiple_samples):
    sample0: HData = mock_dataset_multiple_samples[0]
    sample0.hyperedge_attr = torch.tensor([[0.5], [0.7]])

    sample1: HData = mock_dataset_multiple_samples[1]
    sample1.hyperedge_attr = torch.tensor([[0.9]])

    mock_dataset_multiple_samples.__getitem__.side_effect = lambda idx: [
        sample0,
        sample1,
    ][idx]

    loader = DataLoader(mock_dataset_multiple_samples, batch_size=2)
    batched = loader.collate([sample0, sample1])

    expected_hyperedge_attr = torch.tensor([[0.5], [0.7], [0.9]])
    assert torch.equal(utils.to_non_empty_edgeattr(batched.hyperedge_attr), expected_hyperedge_attr)


def test_collate_three_samples():
    # Sample 0: 2 nodes, 2 hyperedges
    sample0 = HData(
        x=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        hyperedge_index=torch.tensor([[0, 1], [0, 1]]),
    )

    # Sample 1: 1 node, 1 hyperedge
    sample1 = HData(x=torch.tensor([[5.0, 6.0]]), hyperedge_index=torch.tensor([[0], [0]]))

    # Sample 2: 3 nodes, 1 hyperedge
    sample2 = HData(
        x=torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
        hyperedge_index=torch.tensor([[0, 1, 2], [0, 0, 0]]),
    )

    dataset = MagicMock(spec=Dataset)
    dataset.__len__.return_value = 3
    dataset.__getitem__.side_effect = lambda idx: [sample0, sample1, sample2][idx]

    loader = DataLoader(dataset, batch_size=3)

    batched = loader.collate([sample0, sample1, sample2])

    total_nodes_from_all_samples = (
        2 + 1 + 3
    )  # 6 nodes total, 2 from data0, 1 from data1, 3 from data2
    assert batched.num_nodes == total_nodes_from_all_samples
    assert batched.x.size(0) == total_nodes_from_all_samples

    total_hyperedges_from_all_samples = (
        2 + 1 + 1
    )  # 4 hyperedges total, 2 from data0, 1 from data1, 1 from data2
    assert batched.num_hyperedges == total_hyperedges_from_all_samples

    # Sample 0: nodes [0,1], edges [0,1]
    # Sample 1: nodes [2], edges [2] (offset by 2 nodes, 2 edges)
    # Sample 2: nodes [3,4,5], edges [3] (offset by 3 nodes, 3 edges)
    expected_hyperedge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 3, 3]])
    assert torch.equal(batched.hyperedge_index, expected_hyperedge_index)


def test_collate_empty_edge_index():
    sample0 = HData(x=torch.empty((1, 0)), hyperedge_index=torch.empty((2, 0)))
    sample1 = HData(x=torch.empty((1, 0)), hyperedge_index=torch.empty((2, 0)))

    dataset = MagicMock(spec=Dataset)
    dataset.__len__.return_value = 2
    dataset.__getitem__.side_effect = lambda idx: [sample0, sample1][idx]

    loader = DataLoader(dataset, batch_size=2)

    batched = loader.collate([sample0, sample1])

    assert batched.num_nodes == 2
    assert batched.hyperedge_index.size(1) == 0

    assert batched.num_hyperedges == 0


def test_collate_multi_dimensional_edge_attributes(mock_dataset_multiple_samples):
    sample0: HData = mock_dataset_multiple_samples[0]
    sample0.hyperedge_attr = torch.tensor([[0.1, 0.2, 0.3]])
    sample1: HData = mock_dataset_multiple_samples[1]
    sample1.hyperedge_attr = torch.tensor([[0.4, 0.5, 0.6]])

    dataset = MagicMock(spec=Dataset)
    dataset.__getitem__.side_effect = lambda idx: [sample0, sample1][idx]

    loader = DataLoader(dataset, batch_size=2)

    batched = loader.collate([sample0, sample1])

    expected_hyperedge_attr = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    assert torch.equal(utils.to_non_empty_edgeattr(batched.hyperedge_attr), expected_hyperedge_attr)


def test_collate_mixed_edge_attr_presence(mock_dataset_multiple_samples):
    sample0_with_edge_attr: HData = mock_dataset_multiple_samples[0]
    sample0_with_edge_attr.hyperedge_attr = torch.tensor([[0.1, 0.2, 0.3]])
    sample1_no_edge_attr: HData = mock_dataset_multiple_samples[1]

    dataset = MagicMock(spec=Dataset)
    dataset.__getitem__.side_effect = lambda idx: [
        sample0_with_edge_attr,
        sample1_no_edge_attr,
    ][idx]

    loader = DataLoader(dataset, batch_size=2)

    batched = loader.collate([sample0_with_edge_attr, sample1_no_edge_attr])

    # Only sample0 has hyperedge_attr, so only that should be in the batch
    expected_hyperedge_attr = torch.tensor([[0.1, 0.2, 0.3]])
    assert torch.equal(utils.to_non_empty_edgeattr(batched.hyperedge_attr), expected_hyperedge_attr)


def test_collate_with_explicit_num_nodes_and_edges():
    sample0 = HData(
        x=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        hyperedge_index=torch.tensor([[0, 1], [0, 0]]),
        num_nodes=2,
        num_hyperedges=1,
    )
    sample1 = HData(
        x=torch.tensor([[5.0, 6.0]]),
        hyperedge_index=torch.tensor([[0], [0]]),
        num_nodes=1,
        num_hyperedges=1,
    )

    dataset = MagicMock(spec=Dataset)
    dataset.__len__.return_value = 2
    dataset.__getitem__.side_effect = lambda idx: [sample0, sample1][idx]

    loader = DataLoader(dataset, batch_size=2)

    batched = loader.collate([sample0, sample1])

    assert batched.num_nodes == 2 + 1  # 3 nodes total, 2 from data0 and 1 from data1
    assert batched.num_hyperedges == 1 + 1  # 2 hyperedges total, 1 from data0 and 1 from data1


def test_iteration_over_dataloader():
    n_samples = 5

    # 5 samples with 2 nodes and 1 hyperedge each
    data_list = [
        HData(x=torch.randn(2, 3), hyperedge_index=torch.tensor([[0, 1], [0, 0]]))
        for _ in range(n_samples)
    ]

    dataset = MagicMock(spec=Dataset)
    dataset.__len__.return_value = n_samples
    dataset.__getitem__.side_effect = lambda idx: data_list[idx]

    loader = DataLoader(dataset, batch_size=2)

    batch_count = 0
    for batch in loader:
        batch_count += 1
        assert isinstance(batch, HData)
        assert batch.x.size(1) == 3  # 3 features per node
        dataset.__getitem__.assert_called()

    assert batch_count == 3  # 5 samples with batch_size=2 should give us 3 batches (2 + 2 + 1)

    assert dataset.__getitem__.call_count == n_samples  # Ensure all samples were accessed
