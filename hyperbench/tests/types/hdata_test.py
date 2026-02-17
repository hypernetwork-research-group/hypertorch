import pytest
import torch

from torch import Tensor
from hyperbench import utils
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

    return HData(x=x, edge_index=hyperedge_index, edge_attr=hyperedge_attr)


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

    data = HData(x=x, edge_index=hyperedge_index, num_nodes=explicit_num_nodes)

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
    data = HData(x=x, edge_index=hyperedge_index, num_edges=explicit_num_hyperedges)

    assert data.num_edges == expected_num_hyperedges


def test_init_default_y_is_ones():
    x = torch.randn(3, 2)
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 1]])
    data = HData(x=x, edge_index=hyperedge_index)

    assert torch.equal(data.y, torch.ones(2, dtype=torch.float))


def test_init_uses_explicit_y():
    x = torch.randn(3, 2)
    hyperedge_index = torch.tensor([[0, 1], [0, 0]])
    y = torch.tensor([0.5])
    data = HData(x=x, edge_index=hyperedge_index, y=y)

    assert torch.equal(data.y, y)


def test_init_stores_hyperedge_attr():
    x = torch.randn(3, 2)
    hyperedge_index = torch.tensor([[0, 1], [0, 0]])
    hyperedge_attr = torch.randn(1, 4)

    data = HData(x=x, edge_index=hyperedge_index, edge_attr=hyperedge_attr)

    assert torch.equal(utils.to_non_empty_edgeattr(data.edge_attr), hyperedge_attr)


def test_init_hyperedge_attr_defaults_to_none():
    x = torch.randn(3, 2)
    hyperedge_index = torch.tensor([[0, 1], [0, 0]])
    data = HData(x=x, edge_index=hyperedge_index)

    assert data.edge_attr is None


def test_repr_contains_class_name_and_fields(mock_hdata):
    r = repr(mock_hdata)

    assert "HData" in r
    assert f"num_nodes={mock_hdata.num_nodes}" in r
    assert f"num_edges={mock_hdata.num_edges}" in r
    assert f"x_shape={mock_hdata.x.shape}" in r
    assert f"edge_index_shape={mock_hdata.edge_index.shape}" in r


def test_repr_shows_none_edge_attr_when_absent():
    x = torch.randn(3, 2)
    edge_index = torch.tensor([[0, 1], [0, 0]])
    data = HData(x=x, edge_index=edge_index)

    assert "edge_attr_shape=None" in repr(data)


def test_empty_returns_empty_hdata():
    data = HData.empty()

    assert data.x is not None
    assert isinstance(data.x, Tensor)
    assert data.x.shape == (0, 0)

    assert data.edge_index is not None
    assert isinstance(data.edge_index, Tensor)
    assert data.edge_index.shape == (2, 0)

    assert data.edge_attr is None
    assert data.num_nodes == 0
    assert data.num_edges == 0


def test_hdata_to_cpu(mock_hdata):
    returned = mock_hdata.to("cpu")

    assert returned is mock_hdata
    assert mock_hdata.x.device.type == "cpu"
    assert mock_hdata.edge_index.device.type == "cpu"
    assert mock_hdata.edge_attr is not None
    assert mock_hdata.edge_attr.device.type == "cpu"


def test_hdata_to_cpu_handles_none_edge_attr(mock_hdata):
    mock_hdata.edge_attr = None
    returned = mock_hdata.to("cpu")

    assert returned is mock_hdata
    assert mock_hdata.x.device.type == "cpu"
    assert mock_hdata.edge_index.device.type == "cpu"
    assert mock_hdata.edge_attr is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_hdata_to_cuda(mock_hdata):
    returned = mock_hdata.to("cuda")

    assert returned is mock_hdata
    assert mock_hdata.x.device.type == "cuda"
    assert mock_hdata.edge_index.device.type == "cuda"
    assert mock_hdata.edge_attr is not None
    assert mock_hdata.edge_attr.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_hdata_to_cuda_handles_none_edge_attr(mock_hdata):
    mock_hdata.edge_attr = None
    returned = mock_hdata.to("cuda")

    assert returned is mock_hdata
    assert mock_hdata.x.device.type == "cuda"
    assert mock_hdata.edge_index.device.type == "cuda"
    assert mock_hdata.edge_attr is None


@pytest.mark.skipif(not torch.mps.is_available(), reason="MPS not available")
def test_hdata_to_mps(mock_hdata):
    returned = mock_hdata.to("mps")

    assert returned is mock_hdata
    assert mock_hdata.x.device.type == "mps"
    assert mock_hdata.edge_index.device.type == "mps"
    assert mock_hdata.edge_attr is not None
    assert mock_hdata.edge_attr.device.type == "mps"


@pytest.mark.skipif(not torch.mps.is_available(), reason="MPS not available")
def test_hdata_to_mps_handles_none_edge_attr(mock_hdata):
    mock_hdata.edge_attr = None
    returned = mock_hdata.to("mps")

    assert returned is mock_hdata
    assert mock_hdata.x.device.type == "mps"
    assert mock_hdata.edge_index.device.type == "mps"
    assert mock_hdata.edge_attr is None


def test_cat_same_node_space_raises_on_empty_list():
    with pytest.raises(ValueError, match="At least one instance is required."):
        HData.cat_same_node_space([])


def test_cat_same_node_space_raises_on_overlapping_hyperedge_ids():
    x = torch.randn(3, 4)
    hdata1 = HData(x=x, edge_index=torch.tensor([[0, 1], [0, 0]]))
    hdata2 = HData(x=x, edge_index=torch.tensor([[1, 2], [0, 0]]))

    with pytest.raises(
        ValueError,
        match="Overlapping hyperedge IDs found across instances. Ensure each instance uses distinct hyperedge IDs.",
    ):
        HData.cat_same_node_space([hdata1, hdata2])


def test_cat_same_node_space_single_instance():
    x = torch.randn(3, 4)
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 0]])
    hdata = HData(x=x, edge_index=hyperedge_index)

    result = HData.cat_same_node_space([hdata])

    assert result.num_nodes == 3
    assert result.num_edges == 1
    assert torch.equal(result.x, x)
    assert torch.equal(result.edge_index, hyperedge_index)


def test_cat_same_node_space_concatenates_hyperedges():
    x = torch.randn(5, 4)
    hdata1 = HData(x=x, edge_index=torch.tensor([[0, 1], [0, 0]]))
    hdata2 = HData(x=x, edge_index=torch.tensor([[2, 3, 4], [1, 1, 1]]))
    expected_hyperedge_index = torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 1, 1]])

    result = HData.cat_same_node_space([hdata1, hdata2])

    assert result.num_nodes == 5
    assert result.num_edges == 2
    assert torch.equal(result.edge_index, expected_hyperedge_index)


def test_cat_same_node_space_concatenates_labels():
    x = torch.randn(4, 2)
    hdata1 = HData(
        x=x,
        edge_index=torch.tensor([[0, 1], [0, 0]]),
        y=torch.tensor([1.0]),
    )
    hdata2 = HData(
        x=x,
        edge_index=torch.tensor([[2, 3], [1, 1]]),
        y=torch.tensor([0.0]),
    )

    result = HData.cat_same_node_space([hdata1, hdata2])

    assert torch.equal(result.y, torch.tensor([1.0, 0.0]))


def test_cat_same_node_space_uses_largest_x_when_not_provided():
    x_large = torch.randn(3, 1)
    x_small = torch.randn(2, 1)
    hdata1 = HData(x=x_large, edge_index=torch.tensor([[0, 1, 2], [0, 0, 0]]))
    hdata2 = HData(x=x_small, edge_index=torch.tensor([[0, 2], [1, 1]]))
    expected_hyperedge_index = torch.tensor([[0, 1, 2, 0, 2], [0, 0, 0, 1, 1]])

    result = HData.cat_same_node_space([hdata1, hdata2])

    assert torch.equal(result.x, x_large)
    assert torch.equal(result.edge_index, expected_hyperedge_index)


def test_cat_same_node_space_uses_provided_x():
    x = torch.randn(2, 4)
    hdata1 = HData(x=x, edge_index=torch.tensor([[0, 1], [0, 0]]))
    hdata2 = HData(x=x, edge_index=torch.tensor([[2, 3], [1, 1]]))

    custom_x = torch.randn(4, 4)
    result = HData.cat_same_node_space([hdata1, hdata2], x=custom_x)

    assert torch.equal(result.x, custom_x)


def test_cat_same_node_space_concatenates_hyperedge_attr():
    x = torch.randn(4, 2)
    attr1 = torch.randn(1, 3)
    attr2 = torch.randn(1, 3)
    hdata1 = HData(x=x, edge_index=torch.tensor([[0, 1], [0, 0]]), edge_attr=attr1)
    hdata2 = HData(x=x, edge_index=torch.tensor([[2, 3], [1, 1]]), edge_attr=attr2)

    result = HData.cat_same_node_space([hdata1, hdata2])

    assert result.edge_attr is not None
    assert torch.equal(result.edge_attr, torch.cat([attr1, attr2], dim=0))


def test_cat_same_node_space_drops_hyperedge_attr_when_partially_missing():
    x = torch.randn(4, 2)
    hdata1 = HData(x=x, edge_index=torch.tensor([[0, 1], [0, 0]]), edge_attr=torch.randn(1, 3))
    hdata2 = HData(x=x, edge_index=torch.tensor([[2, 3], [1, 1]]), edge_attr=None)

    result = HData.cat_same_node_space([hdata1, hdata2])

    assert result.edge_attr is None


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
    hdata = HData(x=x, edge_index=hyperedge_index)

    result = HData.split(hdata, split_hyperedge_ids=split_ids)

    assert result.num_nodes == expected_num_nodes
    assert result.num_edges == expected_num_hyperedges
    assert torch.equal(result.edge_index, expected_hyperedge_index)


def test_split_subsets_node_features():
    x = torch.tensor([[10.0], [20.0], [30.0], [40.0], [50.0]])
    hyperedge_index = torch.tensor([[0, 1, 3, 4], [0, 0, 1, 1]])
    hdata = HData(x=x, edge_index=hyperedge_index)

    hyperedge_ids = torch.tensor([1])  # Split by hyperedge 1, which includes nodes 3 and 4
    result = HData.split(hdata, split_hyperedge_ids=hyperedge_ids)

    # Only nodes 3 and 4 should be included
    assert result.num_nodes == 2
    assert torch.equal(result.x, torch.tensor([[40.0], [50.0]]))


def test_split_subsets_labels():
    x = torch.randn(4, 2)
    hyperedge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    y = torch.tensor([1.0, 0.0])
    hdata = HData(x=x, edge_index=hyperedge_index, y=y)

    hyperedge_ids = torch.tensor([1])  # Split by hyperedge 1, which has label 0.0
    result = HData.split(hdata, split_hyperedge_ids=hyperedge_ids)

    assert torch.equal(result.y, torch.tensor([0.0]))


def test_split_subsets_edge_attr():
    x = torch.randn(4, 2)
    hyperedge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_attr = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    hdata = HData(x=x, edge_index=hyperedge_index, edge_attr=edge_attr)

    hyperedge_ids = torch.tensor([1])  # Split by hyperedge 1, which has hyperedge_attr [3.0, 4.0]
    result = HData.split(hdata, split_hyperedge_ids=hyperedge_ids)

    assert result.edge_attr is not None
    assert torch.equal(result.edge_attr, torch.tensor([[3.0, 4.0]]))


def test_split_handles_none_edge_attr():
    x = torch.randn(4, 2)
    hyperedge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    hdata = HData(x=x, edge_index=hyperedge_index, edge_attr=None)

    hyperedge_ids = torch.tensor([1])  # Split by hyperedge 1, which has hyperedge_attr None
    result = HData.split(hdata, split_hyperedge_ids=hyperedge_ids)

    assert result.edge_attr is None


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
    result = mock_hdata.with_y_to(value)
    expected_y = torch.full((mock_hdata.num_edges,), value, dtype=torch.float)

    assert torch.equal(result.y, expected_y)


def test_with_y_to_preserves_other_fields(mock_hdata):
    result = mock_hdata.with_y_to(0.5)
    expected_y = torch.full((mock_hdata.num_edges,), 0.5, dtype=torch.float)

    assert torch.equal(result.x, mock_hdata.x)
    assert torch.equal(result.edge_index, mock_hdata.edge_index)
    assert torch.equal(result.y, expected_y)
    assert result.num_nodes == mock_hdata.num_nodes
    assert result.num_edges == mock_hdata.num_edges


def test_with_y_ones_returns_all_ones(mock_hdata):
    result = mock_hdata.with_y_ones()

    assert torch.equal(result.y, torch.ones(mock_hdata.num_edges, dtype=torch.float))


def test_with_y_zeros_returns_all_zeros(mock_hdata):
    result = mock_hdata.with_y_zeros()

    assert torch.equal(result.y, torch.zeros(mock_hdata.num_edges, dtype=torch.float))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_raises_on_inconsistent_device_placement_on_cuda():
    x = torch.randn(3, 4).to("cuda")  # CUDA
    hyperedge_index = torch.tensor([[0, 1], [0, 0]])  # CPU

    with pytest.raises(ValueError, match="Inconsistent device placement"):
        HData(x=x, edge_index=hyperedge_index)


@pytest.mark.skipif(not torch.mps.is_available(), reason="MPS not available")
def test_raises_on_inconsistent_device_placement_on_mps():
    x = torch.randn(3, 4).to("mps")  # MPS
    hyperedge_index = torch.tensor([[0, 1], [0, 0]])  # CPU

    with pytest.raises(ValueError, match="Inconsistent device placement"):
        HData(x=x, edge_index=hyperedge_index)
