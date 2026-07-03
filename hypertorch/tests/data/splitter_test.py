import pytest
import torch
import re

from typing import Any, cast
from hypertorch.data import (
    Dataset,
    HyperedgeDatasetSplitter,
    HyperedgeHDataSplitter,
    HyperedgeIDSplitter,
    NodeDatasetSplitter,
    NodeHDataSplitter,
    NodeIDSplitter,
    SamplingStrategyEnum,
    SparseHyperedgeDatasetSplitter,
    SparseHyperedgeHDataSplitter,
    Splitter,
)
from hypertorch.types import HData


@pytest.fixture
def mock_hdata_five_hyperedges() -> HData:
    x = torch.ones((4, 1), dtype=torch.float32)
    hyperedge_index = torch.tensor(
        [
            [0, 1, 2, 3, 0],
            [0, 1, 2, 3, 4],
        ],
        dtype=torch.long,
    )
    return HData(x=x, hyperedge_index=hyperedge_index)


def test_splitter_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        Splitter()


def test_hyperedge_hdata_splitter_materializes_inductive_split():
    hdata = HData(
        x=torch.tensor([[10.0], [20.0], [30.0], [40.0]], dtype=torch.float),
        hyperedge_index=torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.long),
        y=torch.tensor([1.0, 0.0], dtype=torch.float),
    )

    split_hdata = HyperedgeHDataSplitter(node_space_setting="inductive").split(
        to_split=hdata, split_hyperedge_ids=torch.tensor([1], dtype=torch.long)
    )

    assert split_hdata.num_nodes == 2
    assert split_hdata.num_hyperedges == 1
    assert torch.equal(split_hdata.x, torch.tensor([[30.0], [40.0]], dtype=torch.float))
    assert torch.equal(
        split_hdata.hyperedge_index, torch.tensor([[0, 1], [0, 0]], dtype=torch.long)
    )
    assert torch.equal(split_hdata.y, torch.tensor([0.0], dtype=torch.float))


def test_hyperedge_hdata_splitter_materializes_transductive_split():
    hdata = HData(
        x=torch.tensor([[10.0], [20.0], [30.0], [40.0]], dtype=torch.float),
        hyperedge_index=torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.long),
        hyperedge_weights=torch.tensor([0.25, 0.75], dtype=torch.float),
        hyperedge_attr=torch.tensor([[1.0], [2.0]], dtype=torch.float),
        global_node_ids=torch.tensor([100, 200, 300, 400], dtype=torch.long),
        y=torch.tensor([1.0, 0.0], dtype=torch.float),
    )

    split_hdata = HyperedgeHDataSplitter(node_space_setting="transductive").split(
        to_split=hdata, split_hyperedge_ids=torch.tensor([1], dtype=torch.long)
    )

    assert split_hdata.num_nodes == hdata.num_nodes
    assert split_hdata.num_hyperedges == hdata.num_hyperedges
    assert torch.equal(split_hdata.x, hdata.x)
    assert split_hdata.global_node_ids is not None
    assert hdata.global_node_ids is not None
    assert torch.equal(split_hdata.global_node_ids, hdata.global_node_ids)
    assert torch.equal(split_hdata.hyperedge_index, hdata.hyperedge_index)
    assert torch.equal(split_hdata.y, hdata.y)
    assert split_hdata.hyperedge_weights is not None
    assert hdata.hyperedge_weights is not None
    assert torch.equal(split_hdata.hyperedge_weights, hdata.hyperedge_weights)
    assert split_hdata.hyperedge_attr is not None
    assert hdata.hyperedge_attr is not None
    assert torch.equal(split_hdata.hyperedge_attr, hdata.hyperedge_attr)
    assert torch.equal(
        split_hdata.target_hyperedge_mask,
        torch.tensor([False, True], dtype=torch.bool),
    )


def test_hyperedge_hdata_splitter_rejects_node_related_task():
    hdata = HData(
        x=torch.tensor([[10.0], [20.0]], dtype=torch.float),
        hyperedge_index=torch.tensor([[0, 1], [0, 0]], dtype=torch.long),
        task="node-classification",
    )

    with pytest.raises(
        ValueError,
        match=re.escape("Cannot split dataset with task 'node-classification'"),
    ):
        HyperedgeHDataSplitter().split(
            to_split=hdata,
            split_hyperedge_ids=torch.tensor([0], dtype=torch.long),
        )


def test_sparse_hyperedge_hdata_splitter_rejects_node_related_task():
    hdata = HData(
        x=torch.tensor([[10.0], [20.0]], dtype=torch.float),
        hyperedge_index=torch.tensor([[0, 1], [0, 0]], dtype=torch.long),
        task="node-classification",
    )

    with pytest.raises(
        ValueError,
        match=re.escape("Cannot split dataset with task 'node-classification'"),
    ):
        SparseHyperedgeHDataSplitter().split(
            to_split=hdata,
            split_hyperedge_ids=torch.tensor([0], dtype=torch.long),
        )


def test_node_hdata_splitter_materializes_inductive_split():
    hdata = HData(
        x=torch.tensor([[10.0], [20.0], [30.0], [40.0]], dtype=torch.float),
        hyperedge_index=torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.long),
        hyperedge_weights=torch.tensor([0.25, 0.75], dtype=torch.float),
        hyperedge_attr=torch.tensor([[1.0], [2.0]], dtype=torch.float),
        global_node_ids=torch.tensor([100, 200, 300, 400], dtype=torch.long),
        y=torch.tensor([10, 11, 12, 13], dtype=torch.long),
        task="node-classification",
    )

    split_hdata = NodeHDataSplitter(node_space_setting="inductive").split(
        to_split=hdata,
        split_node_ids=torch.tensor([1, 3], dtype=torch.long),
    )

    assert split_hdata.num_nodes == 2
    assert split_hdata.num_hyperedges == 2
    assert torch.equal(split_hdata.x, torch.tensor([[20.0], [40.0]], dtype=torch.float))
    assert torch.equal(
        split_hdata.hyperedge_index,
        torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
    )
    assert torch.equal(split_hdata.global_node_ids, torch.tensor([200, 400], dtype=torch.long))
    assert torch.equal(split_hdata.y, torch.tensor([11, 13], dtype=torch.long))
    assert torch.equal(split_hdata.target_node_mask, torch.tensor([True, True]))
    assert split_hdata.hyperedge_weights is not None
    assert torch.equal(split_hdata.hyperedge_weights, torch.tensor([0.25, 0.75]))
    assert split_hdata.hyperedge_attr is not None
    assert torch.equal(split_hdata.hyperedge_attr, torch.tensor([[1.0], [2.0]]))


def test_node_hdata_splitter_materializes_transductive_split():
    hdata = HData(
        x=torch.tensor([[10.0], [20.0], [30.0], [40.0]], dtype=torch.float),
        hyperedge_index=torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.long),
        hyperedge_weights=torch.tensor([0.25, 0.75], dtype=torch.float),
        hyperedge_attr=torch.tensor([[1.0], [2.0]], dtype=torch.float),
        global_node_ids=torch.tensor([100, 200, 300, 400], dtype=torch.long),
        y=torch.tensor([10, 11, 12, 13], dtype=torch.long),
        task="node-classification",
    )

    split_hdata = NodeHDataSplitter(node_space_setting="transductive").split(
        to_split=hdata,
        split_node_ids=torch.tensor([1, 3], dtype=torch.long),
    )

    assert split_hdata.num_nodes == hdata.num_nodes
    assert split_hdata.num_hyperedges == hdata.num_hyperedges
    assert torch.equal(split_hdata.x, hdata.x)
    assert torch.equal(split_hdata.hyperedge_index, hdata.hyperedge_index)
    assert torch.equal(split_hdata.global_node_ids, hdata.global_node_ids)
    assert torch.equal(split_hdata.y, hdata.y)
    assert torch.equal(
        split_hdata.target_node_mask,
        torch.tensor([False, True, False, True], dtype=torch.bool),
    )
    assert split_hdata.hyperedge_weights is not None
    assert hdata.hyperedge_weights is not None
    assert torch.equal(split_hdata.hyperedge_weights, hdata.hyperedge_weights)
    assert split_hdata.hyperedge_attr is not None
    assert hdata.hyperedge_attr is not None
    assert torch.equal(split_hdata.hyperedge_attr, hdata.hyperedge_attr)


def test_node_hdata_splitter_rejects_empty_node_ids():
    hdata = HData(
        x=torch.tensor([[10.0], [20.0]], dtype=torch.float),
        hyperedge_index=torch.tensor([[0, 1], [0, 0]], dtype=torch.long),
        task="node-classification",
    )

    with pytest.raises(ValueError, match=re.escape("'split_node_ids' cannot be empty.")):
        NodeHDataSplitter().split(
            to_split=hdata,
            split_node_ids=torch.empty(0, dtype=torch.long),
        )


def test_node_hdata_splitter_rejects_hyperedge_related_task():
    hdata = HData(
        x=torch.tensor([[10.0], [20.0]], dtype=torch.float),
        hyperedge_index=torch.tensor([[0, 1], [0, 0]], dtype=torch.long),
    )

    with pytest.raises(
        ValueError,
        match=re.escape("Cannot split dataset with task 'hyperlink-prediction'"),
    ):
        NodeHDataSplitter().split(
            to_split=hdata,
            split_node_ids=torch.tensor([0], dtype=torch.long),
        )


def test_hyperedge_dataset_splitter_materializes_datasets_and_final_ratios():
    hdata = HData(
        x=torch.arange(4, dtype=torch.float32).unsqueeze(1),
        hyperedge_index=torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.long),
    )
    dataset = Dataset.from_hdata(hdata, sampling_strategy=SamplingStrategyEnum.NODE)

    split_datasets, final_ratios = HyperedgeDatasetSplitter(node_space_setting="inductive").split(
        to_split=dataset, ratios=[0.5, 0.5]
    )

    assert final_ratios == [0.5, 0.5]
    assert [split.hdata.num_hyperedges for split in split_datasets] == [1, 1]
    assert [split.sampling_strategy for split in split_datasets] == [
        SamplingStrategyEnum.NODE,
        SamplingStrategyEnum.NODE,
    ]


def test_hyperedge_dataset_splitter_materializes_transductive_datasets():
    hdata = HData(
        x=torch.arange(4, dtype=torch.float32).unsqueeze(1),
        hyperedge_index=torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long),
    )
    dataset = Dataset.from_hdata(hdata)

    split_datasets, final_ratios = HyperedgeDatasetSplitter(
        node_space_setting="transductive"
    ).split(to_split=dataset, ratios=[0.5, 0.5])

    assert final_ratios == [0.5, 0.5]
    assert [split.hdata.num_nodes for split in split_datasets] == [hdata.num_nodes, hdata.num_nodes]
    assert [split.hdata.num_hyperedges for split in split_datasets] == [
        hdata.num_hyperedges,
        hdata.num_hyperedges,
    ]
    assert torch.equal(
        split_datasets[0].hdata.target_hyperedge_mask,
        torch.tensor([True, True, False, False], dtype=torch.bool),
    )
    assert torch.equal(
        split_datasets[1].hdata.target_hyperedge_mask,
        torch.tensor([False, False, True, True], dtype=torch.bool),
    )


def test_hyperedge_dataset_splitter_nested_split_partitions_only_target_hyperedges():
    hdata = HData(
        x=torch.arange(4, dtype=torch.float32).unsqueeze(1),
        hyperedge_index=torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long),
        target_hyperedge_mask=torch.tensor([False, True, True, False], dtype=torch.bool),
    )
    dataset = Dataset.from_hdata(hdata)

    split_datasets, final_ratios = HyperedgeDatasetSplitter().split(
        to_split=dataset,
        ratios=[0.5, 0.5],
    )

    assert final_ratios == [0.5, 0.5]
    assert torch.equal(
        split_datasets[0].hdata.target_hyperedge_mask,
        torch.tensor([False, True, False, False], dtype=torch.bool),
    )
    assert torch.equal(
        split_datasets[1].hdata.target_hyperedge_mask,
        torch.tensor([False, False, True, False], dtype=torch.bool),
    )


def test_hyperedge_dataset_splitter_shuffle_is_deterministic_with_seed():
    hdata = HData(
        x=torch.arange(6, dtype=torch.float32).unsqueeze(1),
        hyperedge_index=torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], dtype=torch.long),
    )
    dataset = Dataset.from_hdata(hdata)
    splitter = HyperedgeDatasetSplitter(node_space_setting="transductive", shuffle=True, seed=123)

    split_datasets_a, final_ratios_a = splitter.split(to_split=dataset, ratios=[0.5, 0.5])
    split_datasets_b, final_ratios_b = splitter.split(to_split=dataset, ratios=[0.5, 0.5])

    assert final_ratios_a == final_ratios_b
    assert [
        split_dataset.hdata.target_hyperedge_mask.tolist() for split_dataset in split_datasets_a
    ] == [split_dataset.hdata.target_hyperedge_mask.tolist() for split_dataset in split_datasets_b]


def test_hyperedge_dataset_splitter_rejects_node_related_task():
    hdata = HData(
        x=torch.arange(4, dtype=torch.float32).unsqueeze(1),
        hyperedge_index=torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.long),
        task="node-classification",
    )
    dataset = Dataset.from_hdata(hdata, sampling_strategy=SamplingStrategyEnum.NODE)

    with pytest.raises(
        ValueError,
        match=re.escape("Cannot split dataset with task 'node-classification'"),
    ):
        HyperedgeDatasetSplitter().split(to_split=dataset, ratios=[0.5, 0.5])


def test_hyperedge_dataset_splitter_raises_when_ratios_do_not_sum_to_one(
    mock_hdata_five_hyperedges,
):
    dataset = Dataset.from_hdata(mock_hdata_five_hyperedges)

    with pytest.raises(
        ValueError,
        match=re.escape("'ratios' must sum to 1.0"),
    ):
        HyperedgeDatasetSplitter().split(to_split=dataset, ratios=[0.5, 0.25])


def test_hyperedge_dataset_splitter_ignores_cover_all_nodes_in_train_split(
    mock_hdata_five_hyperedges,
):
    dataset = Dataset.from_hdata(mock_hdata_five_hyperedges)

    split_datasets, final_ratios = HyperedgeDatasetSplitter().split(
        to_split=dataset,
        ratios=[0.5, 0.5],
        cover_all_nodes_in_train_split=True,
    )

    assert final_ratios == [0.4, 0.6]
    assert torch.equal(
        split_datasets[0].hdata.target_hyperedge_mask,
        torch.tensor([True, True, False, False, False], dtype=torch.bool),
    )
    assert torch.equal(
        split_datasets[1].hdata.target_hyperedge_mask,
        torch.tensor([False, False, True, True, True], dtype=torch.bool),
    )


def test_sparse_hyperedge_dataset_splitter_rejects_node_related_task():
    hdata = HData(
        x=torch.arange(4, dtype=torch.float32).unsqueeze(1),
        hyperedge_index=torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.long),
        task="node-classification",
    )
    dataset = Dataset.from_hdata(hdata, sampling_strategy=SamplingStrategyEnum.NODE)

    with pytest.raises(
        ValueError,
        match=re.escape("Cannot split dataset with task 'node-classification'"),
    ):
        SparseHyperedgeDatasetSplitter().split(to_split=dataset, ratios=[0.5, 0.5])


def test_sparse_hyperedge_dataset_splitter_uses_train_split_idx_for_transductive_split():
    hdata = HData(
        x=torch.arange(4, dtype=torch.float32).unsqueeze(1),
        hyperedge_index=torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long),
    )
    dataset = Dataset.from_hdata(hdata)

    split_datasets, final_ratios = SparseHyperedgeDatasetSplitter(
        node_space_setting="transductive"
    ).split(to_split=dataset, ratios=[0.5, 0.5], train_split_idx=1)

    assert final_ratios == [0.5, 0.5]
    assert split_datasets[0].hdata.num_nodes == 2
    assert split_datasets[1].hdata.num_nodes == hdata.num_nodes


def test_sparse_hyperedge_dataset_splitter_split_raises_when_train_split_idx_is_out_of_bounds():
    hdata = HData(
        x=torch.arange(4, dtype=torch.float).unsqueeze(1),
        hyperedge_index=torch.tensor([[0, 1, 2, 3, 0], [0, 1, 2, 3, 4]], dtype=torch.long),
    )
    dataset = Dataset.from_hdata(hdata)

    with pytest.raises(
        ValueError, match=re.escape("'train_split_idx' must be between 0 and 1 inclusive, got 2.")
    ):
        SparseHyperedgeDatasetSplitter(node_space_setting="transductive").split(
            to_split=dataset, ratios=[0.5, 0.5], train_split_idx=2
        )


def test_sparse_hyperedge_dataset_splitter_raises_for_non_transductive_train_split_idx():
    hdata = HData(
        x=torch.arange(4, dtype=torch.float).unsqueeze(1),
        hyperedge_index=torch.tensor([[0, 1, 2, 3, 0], [0, 1, 2, 3, 4]], dtype=torch.long),
    )
    dataset = Dataset.from_hdata(hdata)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "'train_split_idx' is only relevant when 'node_space_setting' is 'transductive'"
        ),
    ):
        SparseHyperedgeDatasetSplitter(node_space_setting="inductive").split(
            to_split=dataset, ratios=[0.5, 0.5], train_split_idx=1
        )


def test_sparse_hyperedge_dataset_splitter_rebalances_first_split_to_cover_all_nodes():
    hdata = HData(
        x=torch.arange(4, dtype=torch.float).unsqueeze(1),
        hyperedge_index=torch.tensor([[0, 1, 2, 3, 0], [0, 1, 2, 3, 4]], dtype=torch.long),
        global_node_ids=torch.tensor([100, 200, 300, 400], dtype=torch.long),
        y=torch.arange(5, dtype=torch.float),
    )
    dataset = Dataset.from_hdata(hdata)

    split_datasets, _ = SparseHyperedgeDatasetSplitter().split(
        to_split=dataset, ratios=[0.75, 0.25], cover_all_nodes_in_train_split=True
    )

    train_dataset = split_datasets[0]
    test_dataset = split_datasets[1]

    assert train_dataset.hdata.num_nodes == dataset.hdata.num_nodes
    assert torch.equal(train_dataset.hdata.x, dataset.hdata.x)
    assert torch.equal(
        train_dataset.hdata.hyperedge_index[0].unique(sorted=True),
        torch.arange(hdata.num_nodes, dtype=torch.long),
    )
    # The only way to cover all 4 nodes with 75% of the hyperedges is to:
    # - Put 3 hyperedges in the train split.
    # - Put 1 hyperedge in the test split.
    assert train_dataset.hdata.num_hyperedges == dataset.hdata.num_hyperedges - 1
    assert test_dataset.hdata.num_hyperedges == 1

    split_labels = torch.cat([train_dataset.hdata.y, test_dataset.hdata.y])
    assert split_labels.unique().numel() == dataset.hdata.num_hyperedges
    assert torch.equal(split_labels.sort().values, hdata.y)


def test_sparse_hyperedge_dataset_splitter_returns_final_ratios_when_train_coverage_is_enabled():
    hdata = HData(
        x=torch.arange(4, dtype=torch.float).unsqueeze(1),
        hyperedge_index=torch.tensor([[0, 1, 2, 3, 0], [0, 1, 2, 3, 4]], dtype=torch.long),
    )
    dataset = Dataset.from_hdata(hdata)

    split_datasets, final_ratios = SparseHyperedgeDatasetSplitter().split(
        to_split=dataset, ratios=[0.75, 0.25], cover_all_nodes_in_train_split=True
    )

    assert [split_dataset.hdata.num_hyperedges for split_dataset in split_datasets] == [4, 1]
    assert final_ratios == pytest.approx([0.8, 0.2])


def test_sparse_hyperedge_dataset_splitter_keeps_ratios_when_train_covers_all_nodes():
    hdata = HData(
        x=torch.arange(4, dtype=torch.float).unsqueeze(1),
        hyperedge_index=torch.tensor(
            [
                [0, 1, 2, 3, 0, 2, 1, 3],
                [0, 0, 1, 1, 2, 2, 3, 3],
            ],
            dtype=torch.long,
        ),
    )
    dataset = Dataset.from_hdata(hdata)

    split_datasets, final_ratios = SparseHyperedgeDatasetSplitter().split(
        to_split=dataset, ratios=[0.5, 0.5], cover_all_nodes_in_train_split=True
    )

    assert [split_dataset.hdata.num_hyperedges for split_dataset in split_datasets] == [2, 2]
    assert final_ratios == pytest.approx([0.5, 0.5])
    assert torch.equal(
        split_datasets[0].hdata.hyperedge_index[0].unique(sorted=True),
        torch.arange(hdata.num_nodes, dtype=torch.long),
    )


def test_sparse_hyperedge_dataset_splitter_raises_when_rebalancing_empties_split():
    hdata = HData(
        x=torch.arange(4, dtype=torch.float).unsqueeze(1),
        hyperedge_index=torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long),
    )
    dataset = Dataset.from_hdata(hdata)

    with pytest.raises(
        ValueError,
        match=re.escape("Splitting produced splits"),
    ):
        SparseHyperedgeDatasetSplitter().split(
            to_split=dataset, ratios=[0.75, 0.25], cover_all_nodes_in_train_split=True
        )


def test_sparse_hyperedge_dataset_splitter_raises_when_node_is_missing_from_all_hyperedges():
    hdata = HData(
        x=torch.arange(4, dtype=torch.float).unsqueeze(1),
        hyperedge_index=torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long),
    )
    dataset = Dataset.from_hdata(hdata)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Cannot create a transductive first split covering all nodes because these "
            "node ids do not appear in any hyperedge: [3]."
        ),
    ):
        SparseHyperedgeDatasetSplitter().split(
            to_split=dataset, ratios=[0.5, 0.5], cover_all_nodes_in_train_split=True
        )


def test_node_dataset_splitter_materializes_inductive_datasets_and_final_ratios():
    hdata = HData(
        x=torch.arange(4, dtype=torch.float32).unsqueeze(1),
        hyperedge_index=torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.long),
        y=torch.tensor([10, 11, 12, 13], dtype=torch.long),
        task="node-classification",
    )
    dataset = Dataset.from_hdata(hdata, sampling_strategy=SamplingStrategyEnum.NODE)

    split_datasets, final_ratios = NodeDatasetSplitter(node_space_setting="inductive").split(
        to_split=dataset,
        ratios=[0.25, 0.75],
    )

    assert final_ratios == [0.25, 0.75]
    assert [split.sampling_strategy for split in split_datasets] == [
        SamplingStrategyEnum.NODE,
        SamplingStrategyEnum.NODE,
    ]
    assert [split.hdata.num_nodes for split in split_datasets] == [1, 3]
    assert torch.equal(split_datasets[0].hdata.global_node_ids, torch.tensor([0], dtype=torch.long))
    assert torch.equal(
        split_datasets[1].hdata.global_node_ids,
        torch.tensor([1, 2, 3], dtype=torch.long),
    )
    assert torch.equal(split_datasets[0].hdata.target_node_mask, torch.tensor([True]))
    assert torch.equal(
        split_datasets[1].hdata.target_node_mask,
        torch.tensor([True, True, True]),
    )


def test_node_dataset_splitter_materializes_transductive_datasets_and_final_ratios():
    hdata = HData(
        x=torch.arange(4, dtype=torch.float32).unsqueeze(1),
        hyperedge_index=torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.long),
        y=torch.tensor([10, 11, 12, 13], dtype=torch.long),
        task="node-classification",
    )
    dataset = Dataset.from_hdata(hdata, sampling_strategy=SamplingStrategyEnum.NODE)

    split_datasets, final_ratios = NodeDatasetSplitter(node_space_setting="transductive").split(
        to_split=dataset,
        ratios=[0.25, 0.75],
    )

    assert final_ratios == [0.25, 0.75]
    assert [split.sampling_strategy for split in split_datasets] == [
        SamplingStrategyEnum.NODE,
        SamplingStrategyEnum.NODE,
    ]
    assert torch.equal(
        split_datasets[0].hdata.target_node_mask,
        torch.tensor([True, False, False, False], dtype=torch.bool),
    )
    assert torch.equal(
        split_datasets[1].hdata.target_node_mask,
        torch.tensor([False, True, True, True], dtype=torch.bool),
    )


def test_node_dataset_splitter_nested_split_partitions_only_target_nodes():
    hdata = HData(
        x=torch.arange(4, dtype=torch.float32).unsqueeze(1),
        hyperedge_index=torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.long),
        y=torch.tensor([10, 11, 12, 13], dtype=torch.long),
        target_node_mask=torch.tensor([False, True, True, False], dtype=torch.bool),
        task="node-classification",
    )
    dataset = Dataset.from_hdata(hdata, sampling_strategy=SamplingStrategyEnum.NODE)

    split_datasets, final_ratios = NodeDatasetSplitter(node_space_setting="transductive").split(
        to_split=dataset,
        ratios=[0.5, 0.5],
    )

    assert final_ratios == [0.5, 0.5]
    assert torch.equal(
        split_datasets[0].hdata.target_node_mask,
        torch.tensor([False, True, False, False], dtype=torch.bool),
    )
    assert torch.equal(
        split_datasets[1].hdata.target_node_mask,
        torch.tensor([False, False, True, False], dtype=torch.bool),
    )


def test_node_dataset_splitter_rejects_hyperedge_related_task():
    hdata = HData(
        x=torch.arange(4, dtype=torch.float32).unsqueeze(1),
        hyperedge_index=torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.long),
    )
    dataset = Dataset.from_hdata(hdata)

    with pytest.raises(
        ValueError,
        match=re.escape("Cannot split dataset with task 'hyperlink-prediction'"),
    ):
        NodeDatasetSplitter().split(to_split=dataset, ratios=[0.5, 0.5])


def test_node_dataset_splitter_inductive_shuffle_is_deterministic_with_seed():
    hdata = HData(
        x=torch.arange(6, dtype=torch.float32).unsqueeze(1),
        hyperedge_index=torch.tensor([[0, 1, 2, 3, 4, 5], [0, 0, 1, 1, 2, 2]], dtype=torch.long),
        y=torch.arange(6, dtype=torch.long),
        task="node-classification",
    )
    dataset = Dataset.from_hdata(hdata, sampling_strategy=SamplingStrategyEnum.NODE)
    splitter = NodeDatasetSplitter(node_space_setting="inductive", shuffle=True, seed=123)

    split_datasets_a, final_ratios_a = splitter.split(to_split=dataset, ratios=[0.5, 0.5])
    split_datasets_b, final_ratios_b = splitter.split(to_split=dataset, ratios=[0.5, 0.5])

    assert final_ratios_a == final_ratios_b
    assert [split_dataset.hdata.global_node_ids.tolist() for split_dataset in split_datasets_a] == [
        split_dataset.hdata.global_node_ids.tolist() for split_dataset in split_datasets_b
    ]
    assert [split_dataset.hdata.hyperedge_index.tolist() for split_dataset in split_datasets_a] == [
        split_dataset.hdata.hyperedge_index.tolist() for split_dataset in split_datasets_b
    ]


def test_node_dataset_splitter_shuffle_is_deterministic_with_seed():
    hdata = HData(
        x=torch.arange(6, dtype=torch.float32).unsqueeze(1),
        hyperedge_index=torch.tensor([[0, 1, 2, 3, 4, 5], [0, 0, 1, 1, 2, 2]], dtype=torch.long),
        y=torch.arange(6, dtype=torch.long),
        task="node-classification",
    )
    dataset = Dataset.from_hdata(hdata, sampling_strategy=SamplingStrategyEnum.NODE)
    splitter = NodeDatasetSplitter(node_space_setting="transductive", shuffle=True, seed=123)

    split_datasets_a, final_ratios_a = splitter.split(to_split=dataset, ratios=[0.5, 0.5])
    split_datasets_b, final_ratios_b = splitter.split(to_split=dataset, ratios=[0.5, 0.5])

    assert final_ratios_a == final_ratios_b
    assert [
        split_dataset.hdata.target_node_mask.tolist() for split_dataset in split_datasets_a
    ] == [split_dataset.hdata.target_node_mask.tolist() for split_dataset in split_datasets_b]


def test_hyperedge_id_splitter_get_split_ratios_returns_zero_ratios_when_all_splits_are_empty():
    hdata = HData.empty()
    splitter = HyperedgeIDSplitter(
        hyperedge_index=hdata.hyperedge_index,
        num_nodes=hdata.num_nodes,
        num_hyperedges=hdata.num_hyperedges,
    )
    empty_split = torch.empty(0, dtype=torch.long)

    split_ratios = splitter.get_split_ratios([empty_split, empty_split])

    assert split_ratios == [0.0, 0.0]


def test_hyperedge_id_splitter_get_hyperedge_ids_permutation_returns_original_order_without_shuffle(
    mock_hdata_five_hyperedges,
):
    splitter = HyperedgeIDSplitter(
        hyperedge_index=mock_hdata_five_hyperedges.hyperedge_index,
        num_nodes=mock_hdata_five_hyperedges.num_nodes,
        num_hyperedges=mock_hdata_five_hyperedges.num_hyperedges,
    )
    permutation = splitter.get_hyperedge_ids_permutation(shuffle=False, seed=123)

    assert torch.equal(permutation, torch.arange(5, dtype=torch.long))


def test_hyperedge_id_splitter_get_hyperedge_ids_permutation_is_deterministic_with_seed(
    mock_hdata_five_hyperedges,
):
    shuffle = True
    seed = 123

    splitter = HyperedgeIDSplitter(
        hyperedge_index=mock_hdata_five_hyperedges.hyperedge_index,
        num_nodes=mock_hdata_five_hyperedges.num_nodes,
        num_hyperedges=mock_hdata_five_hyperedges.num_hyperedges,
    )
    permutation_a = splitter.get_hyperedge_ids_permutation(shuffle, seed)
    permutation_b = splitter.get_hyperedge_ids_permutation(shuffle, seed)

    assert torch.equal(permutation_a, permutation_b)
    assert torch.equal(permutation_a.sort().values, torch.arange(5, dtype=torch.long))


def test_hyperedge_id_splitter_split_cumulative_floor_boundaries_and_last_split_absorbs_remainder(
    mock_hdata_five_hyperedges,
):
    hyperedge_ids = torch.arange(5, dtype=torch.long)

    splitter = HyperedgeIDSplitter(
        hyperedge_index=mock_hdata_five_hyperedges.hyperedge_index,
        num_nodes=mock_hdata_five_hyperedges.num_nodes,
        num_hyperedges=mock_hdata_five_hyperedges.num_hyperedges,
    )
    split_hyperedge_ids, final_ratios = splitter.split(
        to_split=hyperedge_ids, ratios=[0.5, 0.25, 0.25]
    )

    assert [split.tolist() for split in split_hyperedge_ids] == [[0, 1], [2], [3, 4]]
    assert final_ratios == [0.4, 0.2, 0.4]


@pytest.mark.parametrize(
    ("hyperedge_index", "expected_split_sizes", "expected_final_ratios"),
    [
        pytest.param(
            torch.tensor([[0, 1, 2, 3, 0], [0, 1, 2, 3, 4]], dtype=torch.long),
            [3, 2],
            # 3/5 and 2/5 as we ensure splits don't get more then requested,
            # in this way, all later splits get at least what they requested,
            # except the last one that might get slightly more due to rounding.
            # This effect is mitigated the more hyperedges we have, as the ratios get closer to the
            # requested ones.
            [0.6, 0.4],
            id="five_hyperedges_rounds_train_up",
        ),
        pytest.param(
            torch.stack(
                [
                    torch.arange(500, dtype=torch.long) % 4,  # 4 nodes
                    torch.arange(
                        500,
                        dtype=torch.long,
                    ),  # 500 hyperedges, 125 per node, so we can split according to the ratios
                ]
            ),
            [375, 125],
            [0.75, 0.25],
            id="many_hyperedges_matches_requested_ratios",
        ),
    ],
)
def test_hyperedge_id_splitter_split_returns_expected_cumulative_ratios(
    hyperedge_index,
    expected_split_sizes,
    expected_final_ratios,
):
    hdata = HData(
        x=torch.arange(4, dtype=torch.float).unsqueeze(1),
        hyperedge_index=hyperedge_index,
    )

    splitter = HyperedgeIDSplitter(
        hyperedge_index=hdata.hyperedge_index,
        num_nodes=hdata.num_nodes,
        num_hyperedges=hdata.num_hyperedges,
    )
    split_hyperedge_ids, final_ratios = splitter.split(hyperedge_index[1], ratios=[0.75, 0.25])

    assert [len(split) for split in split_hyperedge_ids] == expected_split_sizes
    assert final_ratios == pytest.approx(expected_final_ratios)


def test_hyperedge_id_splitter_split_covers_all_nodes_moves_best_covering_he_in_first_split():
    x = torch.ones((4, 1), dtype=torch.float32)
    hyperedge_index = torch.tensor(
        [
            [0, 1, 2, 3, 0, 1, 2, 3],
            [0, 1, 2, 3, 4, 4, 4, 4],
        ],
        dtype=torch.long,
    )
    hdata = HData(x=x, hyperedge_index=hyperedge_index)
    splitter = HyperedgeIDSplitter(
        hyperedge_index=hdata.hyperedge_index,
        num_nodes=hdata.num_nodes,
        num_hyperedges=hdata.num_hyperedges,
    )

    hyperedge_ids_by_split = [
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([2, 3, 4], dtype=torch.long),
    ]

    updated_split_hyperedge_ids, final_ratios = splitter.ensure_split_covers_all_nodes(
        hyperedge_ids_by_split
    )

    assert set(updated_split_hyperedge_ids[0].tolist()) == {0, 1, 4}
    assert set(updated_split_hyperedge_ids[1].tolist()) == {2, 3}
    assert final_ratios == [0.6, 0.4]


def test_hyperedge_id_splitter_ensure_split_covers_all_nodes_rejects_empty_splits(
    mock_hdata_five_hyperedges,
):
    splitter = HyperedgeIDSplitter(
        hyperedge_index=mock_hdata_five_hyperedges.hyperedge_index,
        num_nodes=mock_hdata_five_hyperedges.num_nodes,
        num_hyperedges=mock_hdata_five_hyperedges.num_hyperedges,
    )

    with pytest.raises(ValueError, match="'hyperedge_ids_by_split' cannot be empty"):
        splitter.ensure_split_covers_all_nodes(hyperedge_ids_by_split=[])


def test_hyperedge_id_splitter_ensure_split_covers_all_nodes_rejects_invalid_split_idx(
    mock_hdata_five_hyperedges,
):
    splitter = HyperedgeIDSplitter(
        hyperedge_index=mock_hdata_five_hyperedges.hyperedge_index,
        num_nodes=mock_hdata_five_hyperedges.num_nodes,
        num_hyperedges=mock_hdata_five_hyperedges.num_hyperedges,
    )

    with pytest.raises(
        ValueError, match=re.escape("'split_idx' must be between 0 and 0 inclusive, got 1.")
    ):
        splitter.ensure_split_covers_all_nodes(
            hyperedge_ids_by_split=[torch.tensor([0], dtype=torch.long)],
            split_idx=1,
        )


def test_hyperedge_id_splitter_ensure_split_covers_all_nodes_raises_when_node_is_missing():
    x = torch.ones((4, 1), dtype=torch.float32)
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long)
    hdata = HData(x=x, hyperedge_index=hyperedge_index)
    splitter = HyperedgeIDSplitter(
        hyperedge_index=hdata.hyperedge_index,
        num_nodes=hdata.num_nodes,
        num_hyperedges=hdata.num_hyperedges,
    )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Cannot create a transductive first split covering all nodes because these "
            "node ids do not appear in any hyperedge: [3]."
        ),
    ):
        splitter.ensure_split_covers_all_nodes(
            hyperedge_ids_by_split=[
                torch.tensor([0], dtype=torch.long),
                torch.tensor([1], dtype=torch.long),
            ]
        )


def test_hyperedge_id_splitter_validate_splits_have_hyperedges_raises_when_a_split_is_empty(
    mock_hdata_five_hyperedges,
):
    with pytest.raises(
        ValueError,
        match=re.escape("Splitting produced splits"),
    ):
        splitter = HyperedgeIDSplitter(
            hyperedge_index=mock_hdata_five_hyperedges.hyperedge_index,
            num_nodes=mock_hdata_five_hyperedges.num_nodes,
            num_hyperedges=mock_hdata_five_hyperedges.num_hyperedges,
        )
        splitter.validate_splits_have_hyperedges(
            hyperedge_ids_by_split=[
                torch.tensor([0, 1, 2], dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.tensor([3, 4], dtype=torch.long),
            ]
        )


@pytest.mark.parametrize(
    "ratios, expected_exception, expected_message",
    [
        pytest.param([], ValueError, "'ratios' cannot be empty.", id="empty"),
        pytest.param([0.5, 0.0, 0.5], ValueError, "'ratios' must be positive, got 0.0.", id="zero"),
        pytest.param(
            [0.5, float("inf")], ValueError, "'ratios' must be finite, got inf.", id="infinite"
        ),
        pytest.param([0.5, 0.25], ValueError, "'ratios' must sum to 1.0", id="sum_to_one"),
    ],
)
def test_hyperedge_id_splitter_validates_ratio_values(
    mock_hdata_five_hyperedges, ratios, expected_exception, expected_message
):
    hyperedge_ids = torch.arange(5, dtype=torch.long)
    splitter = HyperedgeIDSplitter(
        hyperedge_index=mock_hdata_five_hyperedges.hyperedge_index,
        num_nodes=mock_hdata_five_hyperedges.num_nodes,
        num_hyperedges=mock_hdata_five_hyperedges.num_hyperedges,
    )

    with pytest.raises(expected_exception, match=re.escape(expected_message)):
        splitter.split(hyperedge_ids, ratios=cast(Any, ratios))


def test_node_id_splitter_get_split_ratios_returns_zero_ratios_when_all_splits_are_empty():
    splitter = NodeIDSplitter(num_nodes=0, device=torch.device("cpu"))
    empty_split = torch.empty(0, dtype=torch.long)

    split_ratios = splitter.get_split_ratios([empty_split, empty_split])

    assert split_ratios == [0.0, 0.0]


@pytest.mark.parametrize(
    "shuffle, seed",
    [
        pytest.param(False, 123, id="no_shuffle_with_seed"),
        pytest.param(None, 123, id="none_shuffle_with_seed"),
    ],
)
def test_node_id_splitter_get_node_ids_permutation_returns_original_order_without_shuffle(
    shuffle, seed
):
    splitter = NodeIDSplitter(num_nodes=5, device=torch.device("cpu"))

    permutation = splitter.get_node_ids_permutation(shuffle=shuffle, seed=seed)

    assert torch.equal(permutation, torch.arange(5, dtype=torch.long))


def test_node_id_splitter_get_node_ids_permutation_is_deterministic_with_seed():
    splitter = NodeIDSplitter(num_nodes=5, device=torch.device("cpu"))

    permutation_a = splitter.get_node_ids_permutation(shuffle=True, seed=123)
    permutation_b = splitter.get_node_ids_permutation(shuffle=True, seed=123)

    assert torch.equal(permutation_a, permutation_b)
    assert torch.equal(permutation_a.sort().values, torch.arange(5, dtype=torch.long))


def test_node_id_splitter_split_cumulative_floor_boundaries_and_last_split_absorbs_remainder():
    node_ids = torch.arange(5, dtype=torch.long)
    splitter = NodeIDSplitter(num_nodes=5, device=torch.device("cpu"))

    split_node_ids, final_ratios = splitter.split(
        to_split=node_ids,
        ratios=[0.5, 0.25, 0.25],
    )

    assert [split.tolist() for split in split_node_ids] == [[0, 1], [2], [3, 4]]
    assert final_ratios == [0.4, 0.2, 0.4]


@pytest.mark.parametrize(
    "ratios, expected_message",
    [
        pytest.param([], "'ratios' cannot be empty.", id="empty"),
        pytest.param([0.5, 0.0, 0.5], "'ratios' must be positive, got 0.0.", id="zero"),
        pytest.param([0.5, float("inf")], "'ratios' must be finite, got inf.", id="infinite"),
        pytest.param([0.5, 0.25], "'ratios' must sum to 1.0", id="sum_not_one"),
    ],
)
def test_node_id_splitter_split_validates_ratio_values(ratios, expected_message):
    splitter = NodeIDSplitter(num_nodes=5, device=torch.device("cpu"))

    with pytest.raises(ValueError, match=re.escape(expected_message)):
        splitter.split(
            to_split=torch.arange(5, dtype=torch.long),
            ratios=cast(Any, ratios),
        )


def test_node_id_splitter_validate_splits_have_nodes_raises_when_a_split_is_empty():
    splitter = NodeIDSplitter(num_nodes=5, device=torch.device("cpu"))

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Splitting produced splits [1] with no nodes. Final ratios: [0.6, 0.0, 0.4]."
        ),
    ):
        splitter.validate_splits_have_nodes(
            node_ids_by_split=[
                torch.tensor([0, 1, 2], dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.tensor([3, 4], dtype=torch.long),
            ]
        )
