import torch
import pytest
import re

from typing import Any, cast
from hyperbench.data import HyperedgeIDSplitter
from hyperbench.types import HData


@pytest.fixture
def mock_hdata_five_hyperedges():
    x = torch.ones((4, 1), dtype=torch.float)
    hyperedge_index = torch.tensor(
        [
            [0, 1, 2, 3, 0],
            [0, 1, 2, 3, 4],
        ],
        dtype=torch.long,
    )
    return HData(x=x, hyperedge_index=hyperedge_index)


def test_get_split_ratios_returns_zero_ratios_when_all_splits_are_empty():
    splitter = HyperedgeIDSplitter(HData.empty())
    empty_split = torch.empty(0, dtype=torch.long)

    split_ratios = splitter.get_split_ratios([empty_split, empty_split])

    assert split_ratios == [0.0, 0.0]


def test_get_hyperedge_ids_permutation_returns_original_order_without_shuffle(
    mock_hdata_five_hyperedges,
):
    splitter = HyperedgeIDSplitter(mock_hdata_five_hyperedges)
    permutation = splitter.get_hyperedge_ids_permutation(shuffle=False, seed=123)

    assert torch.equal(permutation, torch.arange(5))


def test_get_hyperedge_ids_permutation_is_deterministic_with_seed(
    mock_hdata_five_hyperedges,
):
    shuffle = True
    seed = 123

    splitter = HyperedgeIDSplitter(mock_hdata_five_hyperedges)
    permutation_a = splitter.get_hyperedge_ids_permutation(shuffle, seed)
    permutation_b = splitter.get_hyperedge_ids_permutation(shuffle, seed)

    assert torch.equal(permutation_a, permutation_b)
    assert torch.equal(permutation_a.sort().values, torch.arange(5))


def test_split_uses_cumulative_floor_boundaries_and_last_split_absorbs_remainder(
    mock_hdata_five_hyperedges,
):
    hyperedge_ids = torch.arange(5)

    splitter = HyperedgeIDSplitter(mock_hdata_five_hyperedges)
    split_hyperedge_ids, final_ratios = splitter.split(hyperedge_ids, [0.5, 0.25, 0.25])

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


def test_ensure_split_covers_all_nodes_moves_best_covering_hyperedge_into_first_split():
    x = torch.ones((4, 1), dtype=torch.float)
    hyperedge_index = torch.tensor(
        [
            [0, 1, 2, 3, 0, 1, 2, 3],
            [0, 1, 2, 3, 4, 4, 4, 4],
        ],
        dtype=torch.long,
    )
    hdata = HData(x=x, hyperedge_index=hyperedge_index)
    splitter = HyperedgeIDSplitter(hdata)

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


def test_ensure_split_covers_all_nodes_rejects_empty_splits(mock_hdata_five_hyperedges):
    splitter = HyperedgeIDSplitter(mock_hdata_five_hyperedges)

    with pytest.raises(ValueError, match="'hyperedge_ids_by_split' cannot be empty"):
        splitter.ensure_split_covers_all_nodes([])


def test_ensure_split_covers_all_nodes_rejects_invalid_split_idx(
    mock_hdata_five_hyperedges,
):
    splitter = HyperedgeIDSplitter(mock_hdata_five_hyperedges)

    with pytest.raises(ValueError, match="split_idx must reference an existing split"):
        splitter.ensure_split_covers_all_nodes(
            [torch.tensor([0], dtype=torch.long)],
            split_idx=1,
        )


def test_ensure_split_covers_all_nodes_raises_when_a_node_is_missing_from_hypergraph():
    x = torch.ones((4, 1), dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long)
    hdata = HData(x=x, hyperedge_index=hyperedge_index)
    splitter = HyperedgeIDSplitter(hdata)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Cannot create a transductive first split covering all nodes because these "
            "node ids do not appear in any hyperedge: [3]."
        ),
    ):
        splitter.ensure_split_covers_all_nodes(
            [
                torch.tensor([0], dtype=torch.long),
                torch.tensor([1], dtype=torch.long),
            ]
        )


def test_validate_splits_have_hyperedges_raises_when_a_split_is_empty(
    mock_hdata_five_hyperedges,
):
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Cannot create dataset splits because splits [1] contain no hyperedges. "
            "Final ratios: [0.6, 0.0, 0.4]."
        ),
    ):
        splitter = HyperedgeIDSplitter(mock_hdata_five_hyperedges)
        splitter.validate_splits_have_hyperedges(
            [
                torch.tensor([0, 1, 2], dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.tensor([3, 4], dtype=torch.long),
            ]
        )
