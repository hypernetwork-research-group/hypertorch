import pytest
import torch

from unittest.mock import MagicMock
from torch import Tensor
from hyperbench import utils
from hyperbench.nn import NodeEnricher, HyperedgeEnricher
from hyperbench.types import HData


@pytest.fixture
def mock_hdata():
    x = torch.randn(5, 4)  # 5 nodes with 4 features each
    hyperedge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 0],  # node IDs
            [0, 0, 1, 1, 2, 2],
        ]
    )  # hyperedge IDs
    hyperedge_attr = torch.randn(3, 2)  # 3 hyperedges with 2 features each

    return HData(x=x, hyperedge_index=hyperedge_index, hyperedge_attr=hyperedge_attr)


@pytest.fixture
def mock_hdata_stats():
    x = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
        ],
        dtype=torch.float,
    )
    hyperedge_index = torch.tensor(
        [
            [0, 1, 2, 2, 3],
            [0, 0, 0, 1, 1],
        ]
    )
    return HData(x=x, hyperedge_index=hyperedge_index)


@pytest.mark.parametrize(
    "explicit_num_nodes, expected_num_nodes",
    [
        pytest.param(None, 7, id="inferred_from_x"),
        pytest.param(10, 10, id="explicit_overrides_x"),
        pytest.param(0, 0, id="explicit_zero"),
    ],
)
def test_init_num_nodes(explicit_num_nodes, expected_num_nodes):
    hyperedge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6], [0, 0, 0, 0, 0, 0, 0]])
    num_nodes = hyperedge_index[0].size(0)
    x = torch.randn(num_nodes, 3)

    data = HData(x=x, hyperedge_index=hyperedge_index, num_nodes=explicit_num_nodes)

    assert data.num_nodes == expected_num_nodes


@pytest.mark.parametrize(
    "hyperedge_index, explicit_num_hyperedges, expected_num_hyperedges",
    [
        pytest.param(
            torch.tensor([[0, 1, 2, 3], [0, 0, 1, 2]]),
            None,
            3,
            id="inferred_from_hyperedge_index",
        ),
        pytest.param(
            torch.tensor([[0, 1], [0, 0]]),
            5,
            5,
            id="explicit_overrides_hyperedge_index",
        ),
        pytest.param(
            torch.zeros((2, 0), dtype=torch.long),
            None,
            0,
            id="inferred_zero_from_empty_hyperedge_index",
        ),
    ],
)
def test_init_num_hyperedges(hyperedge_index, explicit_num_hyperedges, expected_num_hyperedges):
    x = torch.randn(4, 3)
    data = HData(x=x, hyperedge_index=hyperedge_index, num_hyperedges=explicit_num_hyperedges)

    assert data.num_hyperedges == expected_num_hyperedges


def test_init_default_y_is_ones():
    x = torch.randn(3, 2)
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 1]])
    data = HData(x=x, hyperedge_index=hyperedge_index)

    assert torch.equal(data.y, torch.ones(2, dtype=torch.float))


def test_init_uses_explicit_y():
    x = torch.randn(3, 2)
    hyperedge_index = torch.tensor([[0, 1], [0, 0]])
    y = torch.tensor([0.5])
    data = HData(x=x, hyperedge_index=hyperedge_index, y=y)

    assert torch.equal(data.y, y)


def test_init_stores_hyperedge_attr():
    x = torch.randn(3, 2)
    hyperedge_index = torch.tensor([[0, 1], [0, 0]])
    hyperedge_attr = torch.randn(1, 4)

    data = HData(x=x, hyperedge_index=hyperedge_index, hyperedge_attr=hyperedge_attr)

    assert torch.equal(utils.to_non_empty_edgeattr(data.hyperedge_attr), hyperedge_attr)


def test_init_hyperedge_attr_defaults_to_none():
    x = torch.randn(3, 2)
    hyperedge_index = torch.tensor([[0, 1], [0, 0]])
    data = HData(x=x, hyperedge_index=hyperedge_index)

    assert data.hyperedge_attr is None


def test_repr_contains_class_name_and_fields(mock_hdata):
    r = repr(mock_hdata)

    assert "HData" in r
    assert f"num_nodes={mock_hdata.num_nodes}" in r
    assert f"num_hyperedges={mock_hdata.num_hyperedges}" in r
    assert f"x_shape={mock_hdata.x.shape}" in r
    assert f"hyperedge_index_shape={mock_hdata.hyperedge_index.shape}" in r


def test_repr_shows_none_hyperedge_attr_when_absent():
    x = torch.randn(3, 2)
    hyperedge_index = torch.tensor([[0, 1], [0, 0]])
    data = HData(x=x, hyperedge_index=hyperedge_index)

    assert "hyperedge_attr_shape=None" in repr(data)


def test_empty_returns_empty_hdata():
    data = HData.empty()

    assert data.x is not None
    assert isinstance(data.x, Tensor)
    assert data.x.shape == (0, 0)

    assert data.hyperedge_index is not None
    assert isinstance(data.hyperedge_index, Tensor)
    assert data.hyperedge_index.shape == (2, 0)

    assert data.hyperedge_attr is None
    assert data.num_nodes == 0
    assert data.num_hyperedges == 0


@pytest.mark.parametrize(
    "hyperedge_index, expected_num_nodes, expected_num_hyperedges",
    [
        pytest.param(
            torch.tensor([[0, 1, 2], [0, 0, 1]]),
            3,
            2,
            id="standard",
        ),
        pytest.param(
            torch.tensor([[0, 0, 1, 2, 3, 4], [0, 1, 0, 1, 2, 2]]),
            5,
            3,
            id="nodes_in_multiple_hyperedges",
        ),
        pytest.param(
            torch.zeros((2, 0), dtype=torch.long),
            0,
            0,
            id="empty_hyperedge_index",
        ),
    ],
)
def test_from_hyperedge_index_counts(hyperedge_index, expected_num_nodes, expected_num_hyperedges):
    data = HData.from_hyperedge_index(hyperedge_index)

    assert data.num_nodes == expected_num_nodes
    assert data.num_hyperedges == expected_num_hyperedges
    assert torch.equal(data.hyperedge_index, hyperedge_index)
    assert data.x.shape == (0, 0)
    assert data.hyperedge_attr is None


def test_from_hyperedge_index_has_empty_features():
    hyperedge_index = torch.tensor([[0, 1], [0, 0]])
    data = HData.from_hyperedge_index(hyperedge_index)

    assert data.x.shape == (0, 0)
    assert data.hyperedge_attr is None


def test_hdata_to_cpu(mock_hdata):
    returned = mock_hdata.to("cpu")

    assert returned is mock_hdata
    assert mock_hdata.x.device.type == "cpu"
    assert mock_hdata.hyperedge_index.device.type == "cpu"
    assert mock_hdata.hyperedge_attr is not None
    assert mock_hdata.hyperedge_attr.device.type == "cpu"


def test_hdata_to_cpu_handles_none_hyperedge_attr(mock_hdata):
    mock_hdata.hyperedge_attr = None
    returned = mock_hdata.to("cpu")

    assert returned is mock_hdata
    assert mock_hdata.x.device.type == "cpu"
    assert mock_hdata.hyperedge_index.device.type == "cpu"
    assert mock_hdata.hyperedge_attr is None


def test_hdata_to_cpu_handles_none_global_node_ids(mock_hdata):
    mock_hdata.global_node_ids = None
    returned = mock_hdata.to("cpu")

    assert returned is mock_hdata
    assert mock_hdata.x.device.type == "cpu"
    assert mock_hdata.hyperedge_index.device.type == "cpu"
    assert mock_hdata.global_node_ids is None


def test_hdata_to_cpu_moves_hyperedge_weights():
    x = torch.randn(3, 2)
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 0]])
    hyperedge_weights = torch.tensor([0.25])
    hdata = HData(
        x=x,
        hyperedge_index=hyperedge_index,
        hyperedge_weights=hyperedge_weights,
    )

    returned = hdata.to("cpu")

    assert returned is hdata
    assert hdata.hyperedge_weights is not None
    assert torch.equal(hdata.hyperedge_weights, hyperedge_weights)
    assert hdata.hyperedge_weights.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_hdata_to_cuda_moves_hyperedge_weights():
    x = torch.randn(3, 2)
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 0]])
    hyperedge_weights = torch.tensor([0.25])
    hdata = HData(
        x=x,
        hyperedge_index=hyperedge_index,
        hyperedge_weights=hyperedge_weights,
    )

    returned = hdata.to("cuda")

    assert returned is hdata
    assert hdata.hyperedge_weights is not None
    assert torch.equal(hdata.hyperedge_weights.cpu(), hyperedge_weights)
    assert hdata.hyperedge_weights.device.type == "cuda"


@pytest.mark.skipif(not torch.mps.is_available(), reason="MPS not available")
def test_hdata_to_mps_moves_hyperedge_weights():
    x = torch.randn(3, 2)
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 0]])
    hyperedge_weights = torch.tensor([0.25])
    hdata = HData(
        x=x,
        hyperedge_index=hyperedge_index,
        hyperedge_weights=hyperedge_weights,
    )

    returned = hdata.to("mps")

    assert returned is hdata
    assert hdata.hyperedge_weights is not None
    assert torch.equal(hdata.hyperedge_weights.cpu(), hyperedge_weights)
    assert hdata.hyperedge_weights.device.type == "mps"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_hdata_to_cuda(mock_hdata):
    returned = mock_hdata.to("cuda")

    assert returned is mock_hdata
    assert mock_hdata.x.device.type == "cuda"
    assert mock_hdata.hyperedge_index.device.type == "cuda"
    assert mock_hdata.hyperedge_attr is not None
    assert mock_hdata.hyperedge_attr.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_hdata_to_cuda_handles_none_hyperedge_attr(mock_hdata):
    mock_hdata.hyperedge_attr = None
    returned = mock_hdata.to("cuda")

    assert returned is mock_hdata
    assert mock_hdata.x.device.type == "cuda"
    assert mock_hdata.hyperedge_index.device.type == "cuda"
    assert mock_hdata.hyperedge_attr is None


@pytest.mark.skipif(not torch.mps.is_available(), reason="MPS not available")
def test_hdata_to_mps(mock_hdata):
    returned = mock_hdata.to("mps")

    assert returned is mock_hdata
    assert mock_hdata.x.device.type == "mps"
    assert mock_hdata.hyperedge_index.device.type == "mps"
    assert mock_hdata.hyperedge_attr is not None
    assert mock_hdata.hyperedge_attr.device.type == "mps"


@pytest.mark.skipif(not torch.mps.is_available(), reason="MPS not available")
def test_hdata_to_mps_handles_none_hyperedge_attr(mock_hdata):
    mock_hdata.hyperedge_attr = None
    returned = mock_hdata.to("mps")

    assert returned is mock_hdata
    assert mock_hdata.x.device.type == "mps"
    assert mock_hdata.hyperedge_index.device.type == "mps"
    assert mock_hdata.hyperedge_attr is None


def test_cat_same_node_space_raises_on_empty_list():
    with pytest.raises(ValueError, match="At least one instance is required."):
        HData.cat_same_node_space([])


def test_cat_same_node_space_raises_on_overlapping_hyperedge_ids():
    x = torch.randn(3, 4)
    hdata1 = HData(x=x, hyperedge_index=torch.tensor([[0, 1], [0, 0]]))
    hdata2 = HData(x=x, hyperedge_index=torch.tensor([[1, 2], [0, 0]]))

    with pytest.raises(
        ValueError,
        match="Overlapping hyperedge IDs found across instances. Ensure each instance uses distinct hyperedge IDs.",
    ):
        HData.cat_same_node_space([hdata1, hdata2])


def test_cat_same_node_space_single_instance():
    x = torch.randn(3, 4)
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 0]])
    hdata = HData(x=x, hyperedge_index=hyperedge_index)

    result = HData.cat_same_node_space([hdata])

    assert result.num_nodes == 3
    assert result.num_hyperedges == 1
    assert torch.equal(result.x, x)
    assert torch.equal(result.hyperedge_index, hyperedge_index)


def test_cat_same_node_space_concatenates_hyperedges():
    x = torch.randn(5, 4)
    hdata1 = HData(x=x, hyperedge_index=torch.tensor([[0, 1], [0, 0]]))
    hdata2 = HData(x=x, hyperedge_index=torch.tensor([[2, 3, 4], [1, 1, 1]]))
    expected_hyperedge_index = torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 1, 1]])

    result = HData.cat_same_node_space([hdata1, hdata2])

    assert result.num_nodes == 5
    assert result.num_hyperedges == 2
    assert torch.equal(result.hyperedge_index, expected_hyperedge_index)


def test_cat_same_node_space_concatenates_labels():
    x = torch.randn(4, 2)
    hdata1 = HData(
        x=x,
        hyperedge_index=torch.tensor([[0, 1], [0, 0]]),
        y=torch.tensor([1.0]),
    )
    hdata2 = HData(
        x=x,
        hyperedge_index=torch.tensor([[2, 3], [1, 1]]),
        y=torch.tensor([0.0]),
    )

    result = HData.cat_same_node_space([hdata1, hdata2])

    assert torch.equal(result.y, torch.tensor([1.0, 0.0]))


def test_cat_same_node_space_uses_largest_x_when_not_provided():
    x_large = torch.randn(3, 1)
    x_small = torch.randn(2, 1)
    hdata1 = HData(x=x_large, hyperedge_index=torch.tensor([[0, 1, 2], [0, 0, 0]]))
    hdata2 = HData(x=x_small, hyperedge_index=torch.tensor([[0, 2], [1, 1]]))
    expected_hyperedge_index = torch.tensor([[0, 1, 2, 0, 2], [0, 0, 0, 1, 1]])

    result = HData.cat_same_node_space([hdata1, hdata2])

    assert torch.equal(result.x, x_large)
    assert torch.equal(result.hyperedge_index, expected_hyperedge_index)


def test_cat_same_node_space_uses_provided_x():
    x = torch.randn(2, 4)
    hdata1 = HData(x=x, hyperedge_index=torch.tensor([[0, 1], [0, 0]]))
    hdata2 = HData(x=x, hyperedge_index=torch.tensor([[2, 3], [1, 1]]))

    custom_x = torch.randn(4, 4)
    result = HData.cat_same_node_space([hdata1, hdata2], x=custom_x)

    assert torch.equal(result.x, custom_x)


def test_cat_same_node_space_concatenates_hyperedge_attr():
    x = torch.randn(4, 2)
    attr1 = torch.randn(1, 3)
    attr2 = torch.randn(1, 3)
    hdata1 = HData(x=x, hyperedge_index=torch.tensor([[0, 1], [0, 0]]), hyperedge_attr=attr1)
    hdata2 = HData(x=x, hyperedge_index=torch.tensor([[2, 3], [1, 1]]), hyperedge_attr=attr2)

    result = HData.cat_same_node_space([hdata1, hdata2])

    assert result.hyperedge_attr is not None
    assert torch.equal(result.hyperedge_attr, torch.cat([attr1, attr2], dim=0))


def test_cat_same_node_space_concatenates_hyperedge_weights():
    x = torch.randn(4, 2)
    weights1 = torch.tensor([0.25])
    weights2 = torch.tensor([0.75])
    hdata1 = HData(x=x, hyperedge_index=torch.tensor([[0, 1], [0, 0]]), hyperedge_weights=weights1)
    hdata2 = HData(x=x, hyperedge_index=torch.tensor([[2, 3], [1, 1]]), hyperedge_weights=weights2)

    result = HData.cat_same_node_space([hdata1, hdata2])

    assert result.hyperedge_weights is not None
    assert torch.equal(result.hyperedge_weights, torch.cat([weights1, weights2], dim=0))


def test_cat_same_node_space_drops_hyperedge_weights_when_partially_missing():
    x = torch.randn(4, 2)
    hdata1 = HData(
        x=x, hyperedge_index=torch.tensor([[0, 1], [0, 0]]), hyperedge_weights=torch.tensor([0.25])
    )
    hdata2 = HData(x=x, hyperedge_index=torch.tensor([[2, 3], [1, 1]]), hyperedge_weights=None)

    result = HData.cat_same_node_space([hdata1, hdata2])

    assert result.hyperedge_weights is None


def test_cat_same_node_space_drops_hyperedge_attr_when_partially_missing():
    x = torch.randn(4, 2)
    hdata1 = HData(
        x=x, hyperedge_index=torch.tensor([[0, 1], [0, 0]]), hyperedge_attr=torch.randn(1, 3)
    )
    hdata2 = HData(x=x, hyperedge_index=torch.tensor([[2, 3], [1, 1]]), hyperedge_attr=None)

    result = HData.cat_same_node_space([hdata1, hdata2])

    assert result.hyperedge_attr is None


@pytest.mark.parametrize(
    "split_ids, expected_num_nodes, expected_num_hyperedges, expected_hyperedge_index",
    [
        pytest.param(
            torch.tensor([0]), 3, 1, torch.tensor([[0, 1, 2], [0, 0, 0]]), id="first_hyperedge"
        ),
        pytest.param(
            torch.tensor([1]), 2, 1, torch.tensor([[0, 1], [0, 0]]), id="second_hyperedge"
        ),  # nodes and hyperedges are mapped to be 0-based
        pytest.param(
            torch.tensor([0, 1]),
            4,
            2,
            torch.tensor([[0, 1, 2, 2, 3], [0, 0, 0, 1, 1]]),
            id="both_hyperedges",
        ),
    ],
)
def test_split_counts(
    split_ids, expected_num_nodes, expected_num_hyperedges, expected_hyperedge_index
):
    x = torch.randn(4, 2)
    hyperedge_index = torch.tensor([[0, 1, 2, 2, 3], [0, 0, 0, 1, 1]])
    hdata = HData(x=x, hyperedge_index=hyperedge_index)

    result = HData.split(hdata, split_hyperedge_ids=split_ids)

    assert result.num_nodes == expected_num_nodes
    assert result.num_hyperedges == expected_num_hyperedges
    assert torch.equal(result.hyperedge_index, expected_hyperedge_index)


def test_split_subsets_node_features():
    x = torch.tensor([[10.0], [20.0], [30.0], [40.0], [50.0]])
    hyperedge_index = torch.tensor([[0, 1, 3, 4], [0, 0, 1, 1]])
    hdata = HData(x=x, hyperedge_index=hyperedge_index)

    hyperedge_ids = torch.tensor([1])  # Split by hyperedge 1, which includes nodes 3 and 4
    result = HData.split(hdata, split_hyperedge_ids=hyperedge_ids)

    # Only nodes 3 and 4 should be included
    assert result.num_nodes == 2
    assert torch.equal(result.x, torch.tensor([[40.0], [50.0]]))


def test_split_subsets_labels():
    x = torch.randn(4, 2)
    hyperedge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    y = torch.tensor([1.0, 0.0])
    hdata = HData(x=x, hyperedge_index=hyperedge_index, y=y)

    hyperedge_ids = torch.tensor([1])  # Split by hyperedge 1, which has label 0.0
    result = HData.split(hdata, split_hyperedge_ids=hyperedge_ids)

    assert torch.equal(result.y, torch.tensor([0.0]))


def test_split_handles_none_global_node_ids():
    x = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
    hyperedge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    hdata = HData(x=x, hyperedge_index=hyperedge_index)
    hdata.global_node_ids = None

    result = HData.split(hdata, split_hyperedge_ids=torch.tensor([1]))

    assert result.global_node_ids is not None
    assert torch.equal(result.global_node_ids, torch.arange(result.num_nodes))


def test_split_subsets_edge_attr():
    x = torch.randn(4, 2)
    hyperedge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_attr = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    hdata = HData(x=x, hyperedge_index=hyperedge_index, hyperedge_attr=edge_attr)

    hyperedge_ids = torch.tensor([1])  # Split by hyperedge 1, which has hyperedge_attr [3.0, 4.0]
    result = HData.split(hdata, split_hyperedge_ids=hyperedge_ids)

    assert result.hyperedge_attr is not None
    assert torch.equal(result.hyperedge_attr, torch.tensor([[3.0, 4.0]]))


def test_split_handles_none_edge_attr():
    x = torch.randn(4, 2)
    hyperedge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    hdata = HData(x=x, hyperedge_index=hyperedge_index, hyperedge_attr=None)

    hyperedge_ids = torch.tensor([1])  # Split by hyperedge 1, which has hyperedge_attr None
    result = HData.split(hdata, split_hyperedge_ids=hyperedge_ids)

    assert result.hyperedge_attr is None


def test_split_subsets_hyperedge_weights():
    x = torch.randn(4, 2)
    hyperedge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    hyperedge_weights = torch.tensor([0.25, 0.75])
    hdata = HData(x=x, hyperedge_index=hyperedge_index, hyperedge_weights=hyperedge_weights)

    hyperedge_ids = torch.tensor([1])
    result = HData.split(hdata, split_hyperedge_ids=hyperedge_ids)

    assert result.hyperedge_weights is not None
    assert torch.equal(result.hyperedge_weights, torch.tensor([0.75]))


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(0.0, id="zeros"),
        pytest.param(1.0, id="ones"),
        pytest.param(0.5, id="half"),
        pytest.param(-1.0, id="negative"),
    ],
)
def test_with_y_to_sets_all_labels_to_value(mock_hdata, value):
    hdata = mock_hdata.with_y_to(value)
    expected_y = torch.full((mock_hdata.num_hyperedges,), value, dtype=torch.float)

    assert torch.equal(hdata.y, expected_y)


def test_with_y_to_preserves_other_fields(mock_hdata):
    hdata = mock_hdata.with_y_to(0.5)
    expected_y = torch.full((mock_hdata.num_hyperedges,), 0.5, dtype=torch.float)

    assert torch.equal(hdata.x, mock_hdata.x)
    assert torch.equal(hdata.hyperedge_index, mock_hdata.hyperedge_index)
    assert torch.equal(hdata.y, expected_y)
    assert hdata.num_nodes == mock_hdata.num_nodes
    assert hdata.num_hyperedges == mock_hdata.num_hyperedges


def test_with_y_ones_returns_all_ones(mock_hdata):
    hdata = mock_hdata.with_y_ones()

    assert torch.equal(hdata.y, torch.ones(mock_hdata.num_hyperedges, dtype=torch.float))


def test_with_y_zeros_returns_all_zeros(mock_hdata):
    hdata = mock_hdata.with_y_zeros()

    assert torch.equal(hdata.y, torch.zeros(mock_hdata.num_hyperedges, dtype=torch.float))


def test_enrich_node_features_replace(mock_hdata):
    enricher = MagicMock(spec=NodeEnricher)
    enriched_x = torch.randn(5, 3)
    enricher.enrich.return_value = enriched_x

    result = mock_hdata.enrich_node_features(enricher)

    enricher.enrich.assert_called_once_with(mock_hdata.hyperedge_index)
    assert torch.equal(result.x, enriched_x)


def test_enrich_node_features_concatenate(mock_hdata):
    original_x = mock_hdata.x.clone()

    enricher = MagicMock(spec=NodeEnricher)
    enriched_x = torch.randn(5, 3)
    enricher.enrich.return_value = enriched_x

    result = mock_hdata.enrich_node_features(enricher, enrichment_mode="concatenate")

    enricher.enrich.assert_called_once_with(mock_hdata.hyperedge_index)
    expected_x = torch.cat([original_x, enriched_x], dim=1)
    assert torch.equal(result.x, expected_x)
    assert result.x.shape == (5, 7)  # 4 original + 3 enriched


def test_enrich_hyperedge_weights_replace():
    x = torch.tensor([[1.0], [2.0], [3.0]])
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 1]])
    hyperedge_attr = torch.tensor([[10.0, 11.0], [20.0, 21.0]])
    hyperedge_weights = torch.tensor([0.1, 0.2])
    global_node_ids = torch.tensor([10, 20, 30])
    y = torch.tensor([1.0, 0.0])
    hdata = HData(
        x=x,
        hyperedge_index=hyperedge_index,
        hyperedge_weights=hyperedge_weights,
        hyperedge_attr=hyperedge_attr,
        global_node_ids=global_node_ids,
        y=y,
    )

    enriched_weights = torch.tensor([0.5, 0.9])
    enricher = MagicMock(spec=HyperedgeEnricher)
    enricher.enrich.return_value = enriched_weights

    result = hdata.enrich_hyperedge_weights(enricher)

    enricher.enrich.assert_called_once_with(hyperedge_index)
    assert torch.equal(utils.to_non_empty_edgeattr(result.hyperedge_weights), enriched_weights)
    assert torch.equal(result.x, x)
    assert torch.equal(result.hyperedge_index, hyperedge_index)
    assert torch.equal(utils.to_non_empty_edgeattr(result.hyperedge_attr), hyperedge_attr)
    assert torch.equal(utils.to_non_empty_edgeattr(result.global_node_ids), global_node_ids)
    assert torch.equal(result.y, y)


def test_enrich_hyperedge_weights_concatenate():
    x = torch.tensor([[1.0], [2.0], [3.0]])
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 1]])
    hdata = HData(x=x, hyperedge_index=hyperedge_index, hyperedge_weights=None)

    enriched_weights = torch.tensor([0.3, 0.7])
    enricher = MagicMock(spec=HyperedgeEnricher)
    enricher.enrich.return_value = enriched_weights

    result = hdata.enrich_hyperedge_weights(enricher, enrichment_mode="concatenate")

    enricher.enrich.assert_called_once_with(hyperedge_index)
    assert torch.equal(utils.to_non_empty_edgeattr(result.hyperedge_weights), enriched_weights)


def test_enrich_hyperedge_attr_replace():
    x = torch.tensor([[1.0], [2.0], [3.0]])
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 1]])
    hyperedge_weights = torch.tensor([0.1, 0.2])
    hyperedge_attr = torch.tensor([[10.0], [20.0]])
    global_node_ids = torch.tensor([10, 20, 30])
    y = torch.tensor([1.0, 0.0])
    hdata = HData(
        x=x,
        hyperedge_index=hyperedge_index,
        hyperedge_weights=hyperedge_weights,
        hyperedge_attr=hyperedge_attr,
        global_node_ids=global_node_ids,
        y=y,
    )

    enriched_attr = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    enricher = MagicMock(spec=HyperedgeEnricher)
    enricher.enrich.return_value = enriched_attr

    result = hdata.enrich_hyperedge_attr(enricher)

    enricher.enrich.assert_called_once_with(hyperedge_index)
    assert torch.equal(utils.to_non_empty_edgeattr(result.hyperedge_attr), enriched_attr)
    assert torch.equal(result.x, x)
    assert torch.equal(result.hyperedge_index, hyperedge_index)
    assert torch.equal(utils.to_non_empty_edgeattr(result.hyperedge_weights), hyperedge_weights)
    assert torch.equal(utils.to_non_empty_edgeattr(result.global_node_ids), global_node_ids)
    assert torch.equal(result.y, y)


def test_enrich_hyperedge_attr_concatenate():
    x = torch.tensor([[1.0], [2.0], [3.0]])
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 1]])
    hdata = HData(x=x, hyperedge_index=hyperedge_index, hyperedge_attr=None)

    enriched_attr = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    enricher = MagicMock(spec=HyperedgeEnricher)
    enricher.enrich.return_value = enriched_attr

    result = hdata.enrich_hyperedge_attr(enricher, enrichment_mode="concatenate")

    enricher.enrich.assert_called_once_with(hyperedge_index)
    assert torch.equal(utils.to_non_empty_edgeattr(result.hyperedge_attr), enriched_attr)


def test_get_device_if_all_consistent_returns_device_when_all_consistent():
    x = torch.randn(3, 2)
    hyperedge_index = torch.tensor([[0, 1], [0, 0]])
    hdata = HData(x=x, hyperedge_index=hyperedge_index)

    assert hdata.get_device_if_all_consistent() == torch.device("cpu")


def test_get_device_if_all_consistent_raises_on_mixed_devices():
    x = torch.randn(3, 2)
    hyperedge_index = torch.tensor([[0, 1], [0, 0]])
    hdata = HData(x=x, hyperedge_index=hyperedge_index)

    # Mock a different device on x
    hdata.x = MagicMock(device=torch.device("cuda:0"))

    with pytest.raises(ValueError, match="Inconsistent device placement"):
        hdata.get_device_if_all_consistent()


def test_get_device_if_all_consistent_includes_edge_attr():
    x = torch.randn(3, 2)
    hyperedge_index = torch.tensor([[0, 1], [0, 0]])
    hyperedge_attr = torch.randn(1, 4)
    hdata = HData(x=x, hyperedge_index=hyperedge_index, hyperedge_attr=hyperedge_attr)

    # All on CPU, but hyperedge_attr on different device
    hdata.hyperedge_attr = MagicMock(device=torch.device("cuda:0"))

    with pytest.raises(ValueError, match="Inconsistent device placement"):
        hdata.get_device_if_all_consistent()


def test_get_device_if_all_consistent_handles_none_global_node_ids():
    x = torch.randn(3, 2)
    hyperedge_index = torch.tensor([[0, 1], [0, 0]])
    hyperedge_attr = torch.randn(1, 4)
    hdata = HData(x=x, hyperedge_index=hyperedge_index, hyperedge_attr=hyperedge_attr)
    hdata.global_node_ids = None

    assert hdata.get_device_if_all_consistent() == torch.device("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_raises_on_inconsistent_device_placement_on_cuda():
    x = torch.randn(3, 4).to("cuda")  # CUDA
    hyperedge_index = torch.tensor([[0, 1], [0, 0]])  # CPU

    with pytest.raises(ValueError, match="Inconsistent device placement"):
        HData(x=x, hyperedge_index=hyperedge_index)


@pytest.mark.skipif(not torch.mps.is_available(), reason="MPS not available")
def test_raises_on_inconsistent_device_placement_on_mps():
    x = torch.randn(3, 4).to("mps")  # MPS
    hyperedge_index = torch.tensor([[0, 1], [0, 0]])  # CPU

    with pytest.raises(ValueError, match="Inconsistent device placement"):
        HData(x=x, hyperedge_index=hyperedge_index)


def test_shuffle_preserves_num_nodes_and_num_hyperedges(mock_hdata):
    shuffled_hdata = mock_hdata.shuffle(seed=42)

    assert shuffled_hdata.num_nodes == mock_hdata.num_nodes
    assert shuffled_hdata.num_hyperedges == mock_hdata.num_hyperedges


def test_shuffle_preserves_incidence_structure(mock_hdata):
    shuffled_hdata = mock_hdata.shuffle(seed=7)

    def nodes_per_hyperegde(hyperedge_index, num_hyperedge):
        hyperedges = set()
        for hyperedge_id in range(num_hyperedge):
            hyperedge_mask = hyperedge_index[1] == hyperedge_id
            nodes_in_hyperedge = tuple(sorted(hyperedge_index[0][hyperedge_mask].tolist()))
            hyperedges.add(nodes_in_hyperedge)
        return hyperedges

    original_hyperedges = nodes_per_hyperegde(mock_hdata.hyperedge_index, mock_hdata.num_hyperedges)
    shuffled_hyperedges = nodes_per_hyperegde(
        shuffled_hdata.hyperedge_index, shuffled_hdata.num_hyperedges
    )

    assert original_hyperedges == shuffled_hyperedges


def test_shuffle_matches_labels_and_attr_with_correct_hyperedge():
    x = torch.randn(4, 2)
    # Hyperedge 0 has nodes {0, 1}, hyperedge 1 has nodes {2, 3}
    hyperedge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    y = torch.tensor([1.0, 0.0])
    hyperedge_attr = torch.tensor([[10.0], [20.0]])
    hdata = HData(x=x, hyperedge_index=hyperedge_index, y=y, hyperedge_attr=hyperedge_attr)

    shuffled_hdata = hdata.shuffle(seed=42)

    # For each new hyperedge ID, find which nodes it has and verify the label/attr match
    for new_hyperedge_id in range(shuffled_hdata.num_hyperedges):
        new_hyperedge_mask = shuffled_hdata.hyperedge_index[1] == new_hyperedge_id
        new_nodes = set(shuffled_hdata.hyperedge_index[0][new_hyperedge_mask].tolist())

        # Find the original hyperedge with the same nodes
        for old_hyperedge_id in range(hdata.num_hyperedges):
            old_hyperedge_mask = hdata.hyperedge_index[1] == old_hyperedge_id
            old_nodes = set(hdata.hyperedge_index[0][old_hyperedge_mask].tolist())
            if old_nodes == new_nodes:
                assert shuffled_hdata.y[new_hyperedge_id] == hdata.y[old_hyperedge_id]
                assert torch.equal(
                    utils.to_non_empty_edgeattr(shuffled_hdata.hyperedge_attr)[new_hyperedge_id],
                    utils.to_non_empty_edgeattr(hdata.hyperedge_attr)[old_hyperedge_id],
                )
                break


def test_shuffle_permutes_labels(mock_hdata):
    mock_hdata.y = torch.tensor([1.0, 0.0, 0.5])
    shuffled_hdata = mock_hdata.shuffle(seed=42)

    # Same multiset of labels
    assert sorted(shuffled_hdata.y.tolist()) == sorted(mock_hdata.y.tolist())


def test_shuffle_permutes_hyperedge_attr(mock_hdata):
    mock_hdata.hyperedge_attr = torch.tensor([[10.0], [20.0], [30.0]])
    shuffled_hdata = mock_hdata.shuffle(seed=42)

    # Same multiset of attribute rows
    original_attr = {tuple(attrs.tolist()) for attrs in mock_hdata.hyperedge_attr}
    shuffled_attr = {tuple(attrs.tolist()) for attrs in shuffled_hdata.hyperedge_attr}

    assert original_attr == shuffled_attr


def test_shuffle_handles_none_hyperedge_attr(mock_hdata):
    mock_hdata.hyperedge_attr = None
    shuffled_hdata = mock_hdata.shuffle(seed=42)

    assert shuffled_hdata.hyperedge_attr is None


def test_shuffle_does_not_modify_x(mock_hdata):
    shuffled_hdata = mock_hdata.shuffle(seed=42)

    assert torch.equal(shuffled_hdata.x, mock_hdata.x)


def test_shuffle_with_seed_is_reproducible():
    x = torch.randn(5, 4)
    hyperedge_index = torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 1, 2]])
    y = torch.tensor([1.0, 0.0, 0.5])
    hdata = HData(x=x, hyperedge_index=hyperedge_index, y=y)

    shuffled_hdata1 = hdata.shuffle(seed=123)
    shuffled_hdata2 = hdata.shuffle(seed=123)

    assert torch.equal(shuffled_hdata1.hyperedge_index, shuffled_hdata2.hyperedge_index)
    assert torch.equal(shuffled_hdata1.y, shuffled_hdata2.y)


def test_shuffle_with_no_seed_set(mock_hdata):
    shuffled_hdata1 = mock_hdata.shuffle()

    assert shuffled_hdata1.num_nodes == mock_hdata.num_nodes
    assert shuffled_hdata1.num_hyperedges == mock_hdata.num_hyperedges
    assert shuffled_hdata1.hyperedge_index.shape == mock_hdata.hyperedge_index.shape


def test_stats_returns_correct_statistics(mock_hdata_stats):
    expected_stats = {
        "shape_x": torch.Size([4, 4]),
        "shape_hyperedge_attr": None,
        "shape_hyperedge_weights": None,
        "num_nodes": 4,
        "num_hyperedges": 2,
        "avg_degree_node_raw": 1.25,
        "avg_degree_node": 1,
        "avg_degree_hyperedge_raw": 2.5,
        "avg_degree_hyperedge": 2,
        "node_degree_max": 2,
        "hyperedge_degree_max": 3,
        "node_degree_median": 1,
        "hyperedge_degree_median": 2,
        "distribution_node_degree": [1, 1, 2, 1],
        "distribution_hyperedge_size": [3, 2],
        "distribution_node_degree_hist": {1: 3, 2: 1},
        "distribution_hyperedge_size_hist": {2: 1, 3: 1},
    }

    stats = mock_hdata_stats.stats()

    assert stats == expected_stats


@pytest.mark.parametrize(
    "hyperedge_index, k, expected_hyperedge_index",
    [
        pytest.param(
            torch.tensor([[0, 1, 2], [0, 0, 0]]),
            4,
            torch.zeros((2, 0), dtype=torch.long),
            id="single_hyperedge_below_k_removed",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2], [0, 0, 0]]),
            3,
            torch.tensor([[0, 1, 2], [0, 0, 0]]),
            id="single_hyperedge_at_exact_k_kept",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2, 3, 4], [0, 0, 0, 1, 1]]),
            3,
            torch.tensor([[0, 1, 2], [0, 0, 0]]),
            id="two_hyperedges_first_kept_second_removed",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 1, 1]]),
            3,
            torch.tensor([[0, 1, 2], [0, 0, 0]]),
            id="two_hyperedges_second_kept_first_removed",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2, 3, 4, 5], [0, 0, 0, 1, 1, 1]]),
            3,
            torch.tensor([[0, 1, 2, 3, 4, 5], [0, 0, 0, 1, 1, 1]]),
            id="two_hyperedges_both_kept",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2, 3, 4, 5], [0, 0, 1, 1, 2, 2]]),
            3,
            torch.zeros((2, 0), dtype=torch.long),
            id="three_hyperedges_all_removed",
        ),
    ],
)
def test_remove_hyperedges_with_fewer_than_k_nodes(hyperedge_index, k, expected_hyperedge_index):
    num_nodes = hyperedge_index[0].max().item() + 1
    num_hyperedges = hyperedge_index[1].unique().shape[0]
    x = torch.randn(num_nodes, 4)
    y = torch.randn(num_hyperedges)
    hyperedge_attr = torch.randn(num_hyperedges, 2)
    hdata = HData(x=x, hyperedge_index=hyperedge_index, y=y, hyperedge_attr=hyperedge_attr)

    result = hdata.remove_hyperedges_with_fewer_than_k_nodes(k)

    expected_num_nodes = expected_hyperedge_index[0].unique().shape[0]
    expected_num_hyperedges = expected_hyperedge_index[1].unique().shape[0]

    assert torch.equal(result.hyperedge_index, expected_hyperedge_index)
    assert result.x.shape[0] == expected_num_nodes
    assert result.y.shape[0] == expected_num_hyperedges
    assert utils.to_non_empty_edgeattr(result.hyperedge_attr).shape[0] == expected_num_hyperedges


@pytest.mark.parametrize(
    "hyperedge_index, k, x, expected_x",
    [
        pytest.param(
            torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 1, 1]]),
            3,
            torch.tensor([[10.0], [20.0], [30.0], [40.0], [50.0]]),
            torch.tensor([[30.0], [40.0], [50.0]]),
            id="disjoint_nodes_first_hyperedge_removed",
        ),
        pytest.param(
            # Hyperedge 0: nodes {0, 2} -> 2 nodes (removed), hyperedge 1: nodes {1, 2, 3} -> 3 nodes (kept)
            # Node 2 is shared, so it survives because hyperedge 1 is kept
            # Node 0 is the only node removed as it is only in the removed hyperedge 0
            torch.tensor([[0, 2, 1, 2, 3], [0, 0, 1, 1, 1]]),
            3,
            torch.tensor([[10.0], [20.0], [30.0], [40.0]]),
            torch.tensor([[20.0], [30.0], [40.0]]),
            id="shared_node_survives_with_kept_hyperedge",
        ),
    ],
)
def test_remove_hyperedges_with_fewer_than_k_nodes_subsets_x(hyperedge_index, k, x, expected_x):
    hdata = HData(x=x, hyperedge_index=hyperedge_index)
    result = hdata.remove_hyperedges_with_fewer_than_k_nodes(k=k)

    assert torch.equal(result.x, expected_x)


@pytest.mark.parametrize(
    "hyperedge_index, k, y, expected_y",
    [
        pytest.param(
            torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 1, 1]]),
            3,
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.0]),
            id="disjoint_nodes_first_hyperedge_removed",
        ),
        pytest.param(
            # Hyperedge 0: nodes {0, 2} -> 2 nodes (removed). hyperedge 1: nodes {1, 2, 3} -> 3 nodes (kept)
            # Node 2 is shared, so y for hyperedge 1 must survive
            torch.tensor([[0, 2, 1, 2, 3], [0, 0, 1, 1, 1]]),
            3,
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.0]),
            id="shared_node_y_of_kept_hyperedge_survives",
        ),
    ],
)
def test_remove_hyperedges_with_fewer_than_k_nodes_subsets_y(hyperedge_index, k, y, expected_y):
    num_nodes = hyperedge_index[0].max().item() + 1
    x = torch.randn(num_nodes, 2)
    hdata = HData(x=x, hyperedge_index=hyperedge_index, y=y)
    result = hdata.remove_hyperedges_with_fewer_than_k_nodes(k=k)

    assert torch.equal(result.y, expected_y)


@pytest.mark.parametrize(
    "hyperedge_index, k, hyperedge_attr, expected_attr",
    [
        pytest.param(
            torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 1, 1]]),
            3,
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[3.0, 4.0]]),
            id="disjoint_nodes_first_hyperedge_removed",
        ),
        pytest.param(
            # Hyperedge 0: nodes {0, 2} -> 2 nodes (removed), hyperedge 1: nodes {1, 2, 3} -> 3 nodes (kept)
            # Node 2 is shared, so attr for hyperedge 1 must survive
            torch.tensor([[0, 2, 1, 2, 3], [0, 0, 1, 1, 1]]),
            3,
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[3.0, 4.0]]),
            id="shared_node_attr_of_kept_hyperedge_survives",
        ),
    ],
)
def test_remove_hyperedges_with_fewer_than_k_nodes_subsets_hyperedge_attr(
    hyperedge_index, k, hyperedge_attr, expected_attr
):
    num_nodes = hyperedge_index[0].max().item() + 1
    x = torch.randn(num_nodes, 2)
    hdata = HData(x=x, hyperedge_index=hyperedge_index, hyperedge_attr=hyperedge_attr)
    result = hdata.remove_hyperedges_with_fewer_than_k_nodes(k=k)

    assert result.hyperedge_attr is not None
    assert torch.equal(result.hyperedge_attr, expected_attr)


def test_remove_hyperedges_with_fewer_than_k_nodes_keeps_none_hyperedge_attr():
    x = torch.randn(3, 2)
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 0]])
    hdata = HData(x=x, hyperedge_index=hyperedge_index, hyperedge_attr=None)

    result = hdata.remove_hyperedges_with_fewer_than_k_nodes(k=1)

    assert result.hyperedge_attr is None


def test_remove_hyperedges_with_fewer_than_k_nodes_handles_none_global_node_ids():
    x = torch.randn(5, 2)
    hyperedge_index = torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 1, 1]])
    hdata = HData(x=x, hyperedge_index=hyperedge_index)
    hdata.global_node_ids = None

    result = hdata.remove_hyperedges_with_fewer_than_k_nodes(k=3)

    assert result.global_node_ids is not None
    assert torch.equal(result.global_node_ids, torch.arange(result.num_nodes))


def test_remove_hyperedges_with_fewer_than_k_nodes_subsets_hyperedge_weights():
    x = torch.randn(5, 2)
    hyperedge_index = torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 1, 1]])
    hyperedge_weights = torch.tensor([0.25, 0.75])
    hdata = HData(x=x, hyperedge_index=hyperedge_index, hyperedge_weights=hyperedge_weights)

    result = hdata.remove_hyperedges_with_fewer_than_k_nodes(k=3)

    assert result.hyperedge_weights is not None
    assert torch.equal(result.hyperedge_weights, torch.tensor([0.75]))


def test_remove_hyperedges_with_fewer_than_k_nodes_rebases_hyperedge_index():
    # Hyperedge 0 (nodes 0,1) removed, while hyperedge 1 (nodes 2,3,4) kept.
    # After filtering, surviving nodes 2, 3, and 4, and hyperedge 1 must be rebased to 0-based.
    x = torch.randn(5, 2)
    hyperedge_index = torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 1, 1]])
    hdata = HData(x=x, hyperedge_index=hyperedge_index)

    result = hdata.remove_hyperedges_with_fewer_than_k_nodes(k=3)

    assert torch.equal(result.hyperedge_index, torch.tensor([[0, 1, 2], [0, 0, 0]]))


def test_stats_with_empty_hdata():
    empty_hdata = HData.empty()

    expected_stats = {
        "shape_x": torch.Size([0, 0]),
        "shape_hyperedge_attr": None,
        "shape_hyperedge_weights": None,
        "num_nodes": 0,
        "num_hyperedges": 0,
        "avg_degree_node_raw": 0,
        "avg_degree_node": 0,
        "avg_degree_hyperedge_raw": 0,
        "avg_degree_hyperedge": 0,
        "node_degree_max": 0,
        "hyperedge_degree_max": 0,
        "node_degree_median": 0,
        "hyperedge_degree_median": 0,
        "distribution_node_degree": [],
        "distribution_hyperedge_size": [],
        "distribution_node_degree_hist": {},
        "distribution_hyperedge_size_hist": {},
    }

    stats = empty_hdata.stats()

    assert stats == expected_stats
