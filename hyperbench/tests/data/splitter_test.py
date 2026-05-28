import pytest
import torch
import re

from typing import Any, cast
from hyperbench.data import (
    Dataset,
    DatasetSplitter,
    DefaultDatasetSplitter,
    DefaultHDataSplitter,
    HDataSplitter,
    HyperedgeIDSplitter,
    SamplingStrategy,
    TensorSplitter,
)
from hyperbench.types import HData


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


def test_hdata_splitter_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        HDataSplitter()


def test_dataset_splitter_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        DatasetSplitter()


def test_tensor_splitter_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        TensorSplitter()


def test_default_hdata_splitter_materializes_inductive_split():
    hdata = HData(
        x=torch.tensor([[10.0], [20.0], [30.0], [40.0]]),
        hyperedge_index=torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.long),
        y=torch.tensor([1.0, 0.0]),
    )

    split_hdata = DefaultHDataSplitter(
        split_hyperedge_ids=torch.tensor([1], dtype=torch.long),
        node_space_setting="inductive",
    ).split(to_split=hdata)

    assert split_hdata.num_nodes == 2
    assert split_hdata.num_hyperedges == 1
    assert torch.equal(split_hdata.x, torch.tensor([[30.0], [40.0]]))
    assert torch.equal(split_hdata.hyperedge_index, torch.tensor([[0, 1], [0, 0]]))
    assert torch.equal(split_hdata.y, torch.tensor([0.0]))


def test_default_hdata_splitter_materializes_transductive_split():
    hdata = HData(
        x=torch.tensor([[10.0], [20.0], [30.0], [40.0]]),
        hyperedge_index=torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.long),
        global_node_ids=torch.tensor([100, 200, 300, 400]),
        y=torch.tensor([1.0, 0.0]),
    )

    split_hdata = DefaultHDataSplitter(
        split_hyperedge_ids=torch.tensor([1], dtype=torch.long),
        node_space_setting="transductive",
    ).split(to_split=hdata)

    assert split_hdata.num_nodes == hdata.num_nodes
    assert split_hdata.num_hyperedges == 1
    assert torch.equal(split_hdata.x, hdata.x)
    assert split_hdata.global_node_ids is not None
    assert hdata.global_node_ids is not None
    assert torch.equal(split_hdata.global_node_ids, hdata.global_node_ids)
    assert torch.equal(split_hdata.hyperedge_index, torch.tensor([[2, 3], [0, 0]]))
    assert torch.equal(split_hdata.y, torch.tensor([0.0]))


def test_default_dataset_splitter_materializes_datasets_and_final_ratios():
    hdata = HData(
        x=torch.arange(4, dtype=torch.float32).unsqueeze(1),
        hyperedge_index=torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.long),
    )
    dataset = Dataset.from_hdata(hdata, sampling_strategy=SamplingStrategy.NODE)

    split_datasets, final_ratios = DefaultDatasetSplitter(
        ratios=[0.5, 0.5],
        node_space_setting="inductive",
    ).split(to_split=dataset)

    assert final_ratios == [0.5, 0.5]
    assert [split.hdata.num_hyperedges for split in split_datasets] == [1, 1]
    assert [split.sampling_strategy for split in split_datasets] == [
        SamplingStrategy.NODE,
        SamplingStrategy.NODE,
    ]


def test_default_dataset_splitter_uses_train_split_idx_for_transductive_split():
    hdata = HData(
        x=torch.arange(4, dtype=torch.float32).unsqueeze(1),
        hyperedge_index=torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long),
    )
    dataset = Dataset.from_hdata(hdata)

    split_datasets, final_ratios = DefaultDatasetSplitter(
        ratios=[0.5, 0.5],
        node_space_setting="transductive",
        train_split_idx=1,
    ).split(to_split=dataset)

    assert final_ratios == [0.5, 0.5]
    assert split_datasets[0].hdata.num_nodes == 2
    assert split_datasets[1].hdata.num_nodes == hdata.num_nodes


def test_default_dataset_splitter_raises_when_ratios_do_not_sum_to_one(mock_hdata_five_hyperedges):
    dataset = Dataset.from_hdata(mock_hdata_five_hyperedges)

    with pytest.raises(
        ValueError,
        match=re.escape("'ratios' must sum to 1.0"),
    ):
        DefaultDatasetSplitter(ratios=[0.5, 0.25]).split(to_split=dataset)


def test_hyperedge_id_splitter_get_split_ratios_returns_zero_ratios_when_all_splits_are_empty():
    hdata = HData.empty()
    splitter = HyperedgeIDSplitter(
        hyperedge_index=hdata.hyperedge_index,
        num_nodes=hdata.num_nodes,
        num_hyperedges=hdata.num_hyperedges,
        ratios=[0.5, 0.5],
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
        ratios=[0.5, 0.5],
    )
    permutation = splitter.get_hyperedge_ids_permutation(shuffle=False, seed=123)

    assert torch.equal(permutation, torch.arange(5))


def test_hyperedge_id_splitter_get_hyperedge_ids_permutation_is_deterministic_with_seed(
    mock_hdata_five_hyperedges,
):
    shuffle = True
    seed = 123

    splitter = HyperedgeIDSplitter(
        hyperedge_index=mock_hdata_five_hyperedges.hyperedge_index,
        num_nodes=mock_hdata_five_hyperedges.num_nodes,
        num_hyperedges=mock_hdata_five_hyperedges.num_hyperedges,
        ratios=[0.5, 0.5],
    )
    permutation_a = splitter.get_hyperedge_ids_permutation(shuffle, seed)
    permutation_b = splitter.get_hyperedge_ids_permutation(shuffle, seed)

    assert torch.equal(permutation_a, permutation_b)
    assert torch.equal(permutation_a.sort().values, torch.arange(5))


def test_hyperedge_id_splitter_split_uses_cumulative_floor_boundaries_and_last_split_absorbs_remainder(
    mock_hdata_five_hyperedges,
):
    hyperedge_ids = torch.arange(5)

    splitter = HyperedgeIDSplitter(
        hyperedge_index=mock_hdata_five_hyperedges.hyperedge_index,
        num_nodes=mock_hdata_five_hyperedges.num_nodes,
        num_hyperedges=mock_hdata_five_hyperedges.num_hyperedges,
        ratios=[0.5, 0.25, 0.25],
    )
    split_hyperedge_ids, final_ratios = splitter.split(hyperedge_ids)

    assert [split.tolist() for split in split_hyperedge_ids] == [[0, 1], [2], [3, 4]]
    assert final_ratios == [0.4, 0.2, 0.4]


@pytest.mark.parametrize(
    "ratios, expected_exception, expected_message",
    [
        pytest.param([], ValueError, "'ratios' cannot be empty.", id="empty"),
        pytest.param([0.5, 0.0, 0.5], ValueError, "'ratios' must be positive, got 0.0.", id="zero"),
        pytest.param(
            [0.5, float("inf")], ValueError, "'ratios' must be finite, got inf.", id="infinite"
        ),
    ],
)
def test_split_validates_ratio_values(
    mock_hdata_five_hyperedges, ratios, expected_exception, expected_message
):
    hyperedge_ids = torch.arange(5)
    splitter = HyperedgeIDSplitter(mock_hdata_five_hyperedges)

    with pytest.raises(expected_exception, match=re.escape(expected_message)):
        splitter.split(hyperedge_ids, cast(Any, ratios))


def test_hyperedge_id_splitter_ensure_split_covers_all_nodes_moves_best_covering_hyperedge_into_first_split():
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
        ratios=[0.5, 0.5],
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
    splitter = HyperedgeIDSplitter(mock_hdata_five_hyperedges)

    with pytest.raises(ValueError, match="'hyperedge_ids_by_split' cannot be empty"):
        splitter.ensure_split_covers_all_nodes([])


def test_hyperedge_id_splitter_ensure_split_covers_all_nodes_rejects_invalid_split_idx(
    mock_hdata_five_hyperedges,
):
    splitter = HyperedgeIDSplitter(mock_hdata_five_hyperedges)

    with pytest.raises(ValueError, match="split_idx must reference an existing split"):
        splitter.ensure_split_covers_all_nodes(
            [torch.tensor([0], dtype=torch.long)],
            split_idx=1,
        )


def test_hyperedge_id_splitter_ensure_split_covers_all_nodes_raises_when_a_node_is_missing_from_hypergraph():
    x = torch.ones((4, 1), dtype=torch.float32)
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long)
    hdata = HData(x=x, hyperedge_index=hyperedge_index)
    splitter = HyperedgeIDSplitter(
        hyperedge_index=hdata.hyperedge_index,
        num_nodes=hdata.num_nodes,
        num_hyperedges=hdata.num_hyperedges,
        ratios=[0.5, 0.5],
    )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Cannot create a transductive first split covering all nodes because these node ids do not appear in any hyperedge: [3]."
        ),
    ):
        splitter.ensure_split_covers_all_nodes(
            [
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
            ratios=[0.5, 0.5],
        )
        splitter.validate_splits_have_hyperedges(
            [
                torch.tensor([0, 1, 2], dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.tensor([3, 4], dtype=torch.long),
            ]
        )
