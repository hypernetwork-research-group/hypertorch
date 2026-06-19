import pytest
import torch
import re

from unittest.mock import MagicMock
from typing import Any, cast
from hyperbench import utils
from hyperbench.data import (
    DefaultHDataSplitter,
    HyperedgeEnricher,
    NegativeSampler,
    NodeEnricher,
    RandomNegativeSampler,
    Splitter,
)
from hyperbench.types import HData
from hyperbench.utils import assign_hyperedge_label_to_nodes


@pytest.fixture
def mock_hdata() -> HData:
    x = torch.randn(5, 4, dtype=torch.float)  # 5 nodes with 4 features each
    hyperedge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 0],  # node IDs
            [0, 0, 1, 1, 2, 2],
        ],
        dtype=torch.long,
    )  # hyperedge IDs
    hyperedge_attr = torch.randn(3, 2, dtype=torch.float)  # 3 hyperedges with 2 features each

    return HData(x=x, hyperedge_index=hyperedge_index, hyperedge_attr=hyperedge_attr)


@pytest.fixture
def mock_hdata_stats() -> HData:
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
        ],
        dtype=torch.long,
    )
    return HData(x=x, hyperedge_index=hyperedge_index)


@pytest.fixture
def hdata_with_all_mutable_tensors() -> HData:
    x = torch.arange(10, dtype=torch.float).reshape(5, 2)
    hyperedge_index = torch.tensor([[0, 1, 2, 2, 3, 4], [0, 0, 1, 1, 2, 2]], dtype=torch.long)
    hyperedge_weights = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float)
    hyperedge_attr = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=torch.float)
    global_node_ids = torch.tensor([10, 20, 30, 40, 50], dtype=torch.long)
    y = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float)
    return HData(
        x=x,
        hyperedge_index=hyperedge_index,
        hyperedge_weights=hyperedge_weights,
        hyperedge_attr=hyperedge_attr,
        global_node_ids=global_node_ids,
        y=y,
    )


@pytest.fixture
def mock_negative_sampler() -> tuple[NegativeSampler, MagicMock]:
    def sample(data: HData, seed: int | None = None) -> HData:
        negative_nodes = torch.tensor([0, 2], dtype=torch.long, device=data.device)
        negative_hyperedge_id = torch.full(
            size=negative_nodes.shape,
            fill_value=data.num_hyperedges,
            dtype=torch.long,
            device=data.device,
        )
        return HData(
            x=data.x,
            hyperedge_index=torch.stack([negative_nodes, negative_hyperedge_id]),
            num_nodes=data.num_nodes,
            num_hyperedges=1,
            global_node_ids=data.global_node_ids,
            y=torch.zeros(1, dtype=torch.float, device=data.device),
        )

    sampler = MagicMock(spec=NegativeSampler)
    sampler.sample.side_effect = sample
    return cast(NegativeSampler, sampler), sampler


@pytest.mark.parametrize(
    "explicit_num_nodes, expected_num_nodes",
    [
        pytest.param(None, 7, id="inferred_from_x"),
        pytest.param(10, 10, id="explicit_allows_isolated_nodes"),
    ],
)
def test_init_num_nodes(explicit_num_nodes, expected_num_nodes):
    hyperedge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6], [0, 0, 0, 0, 0, 0, 0]], dtype=torch.long)
    x = torch.randn(expected_num_nodes, 3, dtype=torch.float)

    data = HData(x=x, hyperedge_index=hyperedge_index, num_nodes=explicit_num_nodes)

    assert data.num_nodes == expected_num_nodes


def test_init_raises_when_num_nodes_is_too_small_for_hyperedge_index():
    x = torch.randn(2, 2, dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "'num_nodes' is too small for 'hyperedge_index'. "
            "Got num_nodes=2, but 'hyperedge_index' contains 3 unique node IDs."
        ),
    ):
        HData(x=x, hyperedge_index=hyperedge_index, num_nodes=2)


@pytest.mark.parametrize(
    "hyperedge_index, explicit_num_hyperedges, expected_num_hyperedges",
    [
        pytest.param(
            torch.tensor([[0, 1, 2, 3], [0, 0, 1, 2]], dtype=torch.long),
            None,
            3,
            id="inferred_from_hyperedge_index",
        ),
        pytest.param(
            torch.tensor([[0, 1], [0, 0]], dtype=torch.long),
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
    x = torch.randn(4, 3, dtype=torch.float)
    data = HData(x=x, hyperedge_index=hyperedge_index, num_hyperedges=explicit_num_hyperedges)

    assert data.num_hyperedges == expected_num_hyperedges


def test_init_raises_when_num_hyperedges_is_too_small_for_hyperedge_index():
    x = torch.randn(2, 2, dtype=torch.float)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "'num_hyperedges' is too small for 'hyperedge_index'. "
            "Got num_hyperedges=2, but 'hyperedge_index' contains 3 unique hyperedge IDs."
        ),
    ):
        HData(
            x=x,
            hyperedge_index=torch.tensor([[0, 1, 0], [0, 1, 2]], dtype=torch.long),
            num_hyperedges=2,
        )


def test_init_default_y_is_ones():
    x = torch.randn(3, 2, dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long)
    data = HData(x=x, hyperedge_index=hyperedge_index)

    assert data.y.dtype == torch.float
    assert torch.equal(data.y, torch.ones(2, dtype=torch.float))


def test_init_default_global_node_ids_are_long_and_on_x_device():
    x = torch.randn(3, 2, dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long)

    data = HData(x=x, hyperedge_index=hyperedge_index)

    assert data.global_node_ids.dtype == torch.long
    assert data.global_node_ids.device == x.device


def test_init_uses_explicit_y():
    x = torch.randn(3, 2, dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)
    y = torch.tensor([0.5], dtype=torch.float64)
    data = HData(x=x, hyperedge_index=hyperedge_index, y=y)

    assert data.y.dtype == y.dtype
    assert torch.equal(data.y, y)


def test_init_stores_hyperedge_attr():
    x = torch.randn(3, 2, dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)
    hyperedge_attr = torch.randn(1, 4, dtype=torch.float64)

    data = HData(x=x, hyperedge_index=hyperedge_index, hyperedge_attr=hyperedge_attr)

    assert utils.to_non_empty_edgeattr(data.hyperedge_attr).dtype == hyperedge_attr.dtype
    assert torch.equal(utils.to_non_empty_edgeattr(data.hyperedge_attr), hyperedge_attr)


def test_init_hyperedge_attr_defaults_to_none():
    x = torch.randn(3, 2, dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)
    data = HData(x=x, hyperedge_index=hyperedge_index)

    assert data.hyperedge_attr is None


@pytest.mark.parametrize(
    "kwargs, expected_message",
    [
        pytest.param(
            {
                "x": torch.randn(3, dtype=torch.float),
                "hyperedge_index": torch.tensor([[0, 1], [0, 0]], dtype=torch.long),
            },
            "'x' must be a 2D tensor, got shape (3,).",
            id="x_not_2d",
        ),
        pytest.param(
            {
                "x": torch.ones((3, 2), dtype=torch.long),
                "hyperedge_index": torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long),
            },
            "'x' must have a floating-point dtype, got torch.int64.",
            id="x_not_floating",
        ),
        pytest.param(
            {
                "x": torch.randn(3, 2, dtype=torch.float),
                "hyperedge_index": torch.tensor([0, 1], dtype=torch.long),
            },
            "'hyperedge_index' must have shape (2, num_incidences), got (2,).",
            id="hyperedge_index_not_2d",
        ),
        pytest.param(
            {
                "x": torch.randn(3, 2, dtype=torch.float),
                "hyperedge_index": torch.tensor([[0, 1, 2]], dtype=torch.long),
            },
            "'hyperedge_index' must have shape (2, num_incidences), got (1, 3).",
            id="hyperedge_index_wrong_rows",
        ),
        pytest.param(
            {
                "x": torch.randn(3, 2, dtype=torch.float),
                "hyperedge_index": torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float),
            },
            "'hyperedge_index' must have dtype torch.long, got torch.float32.",
            id="hyperedge_index_not_long",
        ),
        pytest.param(
            {
                "x": torch.randn(3, 2, dtype=torch.float),
                "hyperedge_index": torch.tensor([[-1, 1], [0, 0]], dtype=torch.long),
            },
            "'hyperedge_index' cannot contain negative node or hyperedge IDs.",
            id="hyperedge_index_negative_id",
        ),
        pytest.param(
            {
                "x": torch.randn(2, 2, dtype=torch.float),
                "hyperedge_index": torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long),
            },
            (
                "'x' must have one feature row per node, or be 'torch.empty((0, 0))' "
                "if there are no nodes. Got x.shape=(2, 2) but num_nodes=3."
            ),
            id="x_rows_do_not_match_num_nodes",
        ),
        pytest.param(
            {
                "x": torch.randn(3, 2, dtype=torch.float),
                "hyperedge_index": torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long),
                "global_node_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
            },
            "'global_node_ids' must be a 1D tensor, got shape (1, 3).",
            id="global_node_ids_not_1d",
        ),
        pytest.param(
            {
                "x": torch.randn(3, 2, dtype=torch.float),
                "hyperedge_index": torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long),
                "global_node_ids": torch.tensor([0.0, 1.0, 2.0], dtype=torch.float),
            },
            "'global_node_ids' must have dtype torch.long, got torch.float32.",
            id="global_node_ids_not_long",
        ),
        pytest.param(
            {
                "x": torch.randn(3, 2, dtype=torch.float),
                "hyperedge_index": torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long),
                "global_node_ids": torch.tensor([0, 1], dtype=torch.long),
            },
            "'global_node_ids' must have one entry per node. Got size=2 but num_nodes=3.",
            id="global_node_ids_wrong_length",
        ),
        pytest.param(
            {
                "x": torch.randn(3, 2, dtype=torch.float),
                "hyperedge_index": torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long),
                "y": torch.tensor([[1.0, 0.0]], dtype=torch.float),
            },
            "'y' must be a 1D tensor, got shape (1, 2).",
            id="y_not_1d",
        ),
        pytest.param(
            {
                "x": torch.randn(3, 2, dtype=torch.float),
                "hyperedge_index": torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long),
                "y": torch.tensor([1, 0], dtype=torch.long),
            },
            "'y' must have a floating-point dtype, got torch.int64.",
            id="y_not_floating",
        ),
        pytest.param(
            {
                "x": torch.randn(3, 2, dtype=torch.float),
                "hyperedge_index": torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long),
                "y": torch.tensor([1.0], dtype=torch.float),
            },
            "'y' must have one entry per hyperedge. Got 1 entries but num_hyperedges=2.",
            id="y_wrong_length",
        ),
        pytest.param(
            {
                "x": torch.randn(3, 2, dtype=torch.float),
                "hyperedge_index": torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long),
                "hyperedge_weights": torch.tensor([[0.25, 0.75]], dtype=torch.float),
            },
            "'hyperedge_weights' must be a 1D tensor, got shape (1, 2).",
            id="hyperedge_weights_not_1d",
        ),
        pytest.param(
            {
                "x": torch.randn(3, 2, dtype=torch.float),
                "hyperedge_index": torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long),
                "hyperedge_weights": torch.tensor([1, 2], dtype=torch.long),
            },
            "'hyperedge_weights' must have a floating-point dtype, got torch.int64.",
            id="hyperedge_weights_not_floating",
        ),
        pytest.param(
            {
                "x": torch.randn(3, 2, dtype=torch.float),
                "hyperedge_index": torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long),
                "hyperedge_weights": torch.tensor([0.25], dtype=torch.float),
            },
            (
                "'hyperedge_weights' must have one entry per hyperedge. "
                "Got size=1 but num_hyperedges=2."
            ),
            id="hyperedge_weights_wrong_length",
        ),
        pytest.param(
            {
                "x": torch.randn(3, 2, dtype=torch.float),
                "hyperedge_index": torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long),
                "hyperedge_attr": torch.tensor([1.0, 2.0], dtype=torch.float),
            },
            "'hyperedge_attr' must be a 2D tensor, got shape (2,).",
            id="hyperedge_attr_not_2d",
        ),
        pytest.param(
            {
                "x": torch.randn(3, 2, dtype=torch.float),
                "hyperedge_index": torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long),
                "hyperedge_attr": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
            },
            "'hyperedge_attr' must have a floating-point dtype, got torch.int64.",
            id="hyperedge_attr_not_floating",
        ),
        pytest.param(
            {
                "x": torch.randn(3, 2, dtype=torch.float),
                "hyperedge_index": torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long),
                "hyperedge_attr": torch.randn(1, 4, dtype=torch.float),
            },
            "'hyperedge_attr' must have one row per hyperedge. Got size=1 but num_hyperedges=2.",
            id="hyperedge_attr_wrong_rows",
        ),
    ],
)
def test_init_validates_input_values(kwargs, expected_message):
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        HData(**kwargs)


@pytest.mark.parametrize(
    "kwargs, expected_message",
    [
        pytest.param(
            {
                "x": torch.empty((0, 1), dtype=torch.float),
                "hyperedge_index": torch.empty((2, 0), dtype=torch.long),
                "num_nodes": -1,
            },
            "'num_nodes' must be non-negative, got -1.",
            id="negative_num_nodes",
        ),
        pytest.param(
            {
                "x": torch.empty((0, 1), dtype=torch.float),
                "hyperedge_index": torch.empty((2, 0), dtype=torch.long),
                "num_hyperedges": -1,
            },
            "'num_hyperedges' must be non-negative, got -1.",
            id="negative_num_hyperedges",
        ),
    ],
)
def test_init_validates_non_negative_number_of_nodes_and_hyperedges(kwargs, expected_message):
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        HData(**kwargs)


def test_repr_contains_class_name_and_fields(mock_hdata):
    r = repr(mock_hdata)

    assert "HData" in r
    assert f"num_nodes={mock_hdata.num_nodes}" in r
    assert f"num_hyperedges={mock_hdata.num_hyperedges}" in r
    assert f"x_shape={mock_hdata.x.shape}" in r
    assert f"hyperedge_index_shape={mock_hdata.hyperedge_index.shape}" in r


def test_repr_shows_none_hyperedge_attr_when_absent():
    x = torch.randn(3, 2, dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)
    data = HData(x=x, hyperedge_index=hyperedge_index)

    assert "hyperedge_attr_shape=None" in repr(data)


def test_empty_returns_empty_hdata():
    data = HData.empty()

    assert data.x is not None
    assert data.x.shape == (0, 0)
    assert data.x.dtype == torch.float

    assert data.hyperedge_index is not None
    assert data.hyperedge_index.shape == (2, 0)
    assert data.hyperedge_index.dtype == torch.long

    assert data.hyperedge_attr is None
    assert data.hyperedge_weights is None

    assert data.num_nodes == 0
    assert data.num_hyperedges == 0

    assert data.global_node_ids is not None
    assert data.global_node_ids.shape == (0,)
    assert data.global_node_ids.dtype == torch.long

    assert data.y is not None
    assert data.y.shape == (0,)
    assert data.y.dtype == torch.float


@pytest.mark.parametrize(
    "hyperedge_index, expected_num_nodes, expected_num_hyperedges",
    [
        pytest.param(
            torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long),
            3,
            2,
            id="standard",
        ),
        pytest.param(
            torch.tensor([[0, 0, 1, 2, 3, 4], [0, 1, 0, 1, 2, 2]], dtype=torch.long),
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
    assert data.hyperedge_index.dtype == torch.long
    assert data.x.shape == (0, 0)
    assert data.x.dtype == torch.float
    assert data.hyperedge_attr is None


def test_from_hyperedge_index_has_empty_features():
    hyperedge_index = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)
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


def test_hdata_to_cpu_moves_hyperedge_weights():
    x = torch.randn(3, 2, dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long)
    hyperedge_weights = torch.tensor([0.25], dtype=torch.float)
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
    x = torch.randn(3, 2, dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long)
    hyperedge_weights = torch.tensor([0.25], dtype=torch.float)
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
    x = torch.randn(3, 2, dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long)
    hyperedge_weights = torch.tensor([0.25], dtype=torch.float)
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
    with pytest.raises(ValueError, match=re.escape("'hdatas' cannot be empty.")):
        HData.cat_same_node_space([])


def test_cat_same_node_space_raises_on_overlapping_hyperedge_ids():
    x = torch.randn(3, 4, dtype=torch.float)
    hdata1 = HData(x=x, hyperedge_index=torch.tensor([[0, 1], [0, 0]], dtype=torch.long))
    hdata2 = HData(x=x, hyperedge_index=torch.tensor([[1, 2], [0, 0]], dtype=torch.long))

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Overlapping hyperedge IDs found across instances. Ensure each "
            "instance uses distinct hyperedge IDs."
        ),
    ):
        HData.cat_same_node_space([hdata1, hdata2])


def test_cat_same_node_space_single_instance():
    x = torch.randn(3, 4, dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long)
    hdata = HData(x=x, hyperedge_index=hyperedge_index)

    result = HData.cat_same_node_space([hdata])

    assert result.num_nodes == 3
    assert result.num_hyperedges == 1
    assert torch.equal(result.x, x)
    assert torch.equal(result.hyperedge_index, hyperedge_index)


def test_cat_same_node_space_concatenates_hyperedges():
    x = torch.randn(5, 4, dtype=torch.float)
    hdata1 = HData(x=x, hyperedge_index=torch.tensor([[0, 1], [0, 0]], dtype=torch.long))
    hdata2 = HData(x=x, hyperedge_index=torch.tensor([[2, 3, 4], [1, 1, 1]], dtype=torch.long))
    expected_hyperedge_index = torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 1, 1]], dtype=torch.long)

    result = HData.cat_same_node_space([hdata1, hdata2])

    assert result.num_nodes == 5
    assert result.num_hyperedges == 2
    assert torch.equal(result.hyperedge_index, expected_hyperedge_index)


def test_cat_same_node_space_concatenates_labels():
    x = torch.randn(4, 2, dtype=torch.float)
    hdata1 = HData(
        x=x,
        hyperedge_index=torch.tensor([[0, 1], [0, 0]], dtype=torch.long),
        y=torch.tensor([1.0], dtype=torch.float),
    )
    hdata2 = HData(
        x=x,
        hyperedge_index=torch.tensor([[2, 3], [1, 1]], dtype=torch.long),
        y=torch.tensor([0.0], dtype=torch.float),
    )

    result = HData.cat_same_node_space([hdata1, hdata2])

    assert torch.equal(result.y, torch.tensor([1.0, 0.0], dtype=torch.float))


def test_cat_same_node_space_uses_largest_x_when_not_provided():
    x_large = torch.randn(3, 1, dtype=torch.float)
    x_small = torch.randn(2, 1, dtype=torch.float)
    global_node_ids = torch.tensor([10, 20, 30], dtype=torch.long)
    hdata1 = HData(
        x=x_large,
        hyperedge_index=torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long),
        global_node_ids=global_node_ids,
    )
    hdata2 = HData(x=x_small, hyperedge_index=torch.tensor([[0, 2], [1, 1]], dtype=torch.long))

    expected_hyperedge_index = torch.tensor([[0, 1, 2, 0, 2], [0, 0, 0, 1, 1]], dtype=torch.long)

    result = HData.cat_same_node_space([hdata1, hdata2])

    assert torch.equal(result.x, x_large)
    assert result.global_node_ids is not None
    assert torch.equal(result.global_node_ids, global_node_ids)
    assert torch.equal(result.hyperedge_index, expected_hyperedge_index)


def test_cat_same_node_space_uses_provided_x_and_global_node_ids():
    x = torch.randn(2, 4, dtype=torch.float)
    hdata1 = HData(x=x, hyperedge_index=torch.tensor([[0, 1], [0, 0]], dtype=torch.long))
    hdata2 = HData(x=x, hyperedge_index=torch.tensor([[2, 3], [1, 1]], dtype=torch.long))

    custom_x = torch.randn(4, 4, dtype=torch.float)
    custom_global_node_ids = torch.tensor([10, 20, 30, 40], dtype=torch.long)

    result = HData.cat_same_node_space(
        hdatas=[hdata1, hdata2],
        x=custom_x,
        global_node_ids=custom_global_node_ids,
    )

    assert torch.equal(result.x, custom_x)
    assert result.global_node_ids is not None
    assert torch.equal(result.global_node_ids, custom_global_node_ids)


def test_cat_same_node_space_raises_when_only_x_is_provided():
    x = torch.randn(2, 4, dtype=torch.float)
    hdata1 = HData(x=x, hyperedge_index=torch.tensor([[0, 1], [0, 0]], dtype=torch.long))
    hdata2 = HData(x=x, hyperedge_index=torch.tensor([[2, 3], [1, 1]], dtype=torch.long))

    with pytest.raises(
        ValueError,
        match=re.escape(
            "If 'x' is provided, 'global_node_ids' must also be provided to ensure consistency."
        ),
    ):
        HData.cat_same_node_space([hdata1, hdata2], x=torch.randn(4, 4, dtype=torch.float))


def test_cat_same_node_space_raises_when_only_global_node_ids_are_provided():
    x = torch.randn(2, 4, dtype=torch.float)
    hdata1 = HData(x=x, hyperedge_index=torch.tensor([[0, 1], [0, 0]], dtype=torch.long))
    hdata2 = HData(x=x, hyperedge_index=torch.tensor([[2, 3], [1, 1]], dtype=torch.long))

    with pytest.raises(
        ValueError,
        match=re.escape(
            "If 'global_node_ids' is provided, 'x' must also be provided to ensure consistency."
        ),
    ):
        HData.cat_same_node_space(
            [hdata1, hdata2], global_node_ids=torch.arange(4, dtype=torch.long)
        )


def test_cat_same_node_space_validates_global_node_ids_alignment():
    x = torch.randn(2, 4, dtype=torch.float)
    hdata1 = HData(x=x, hyperedge_index=torch.tensor([[0, 1], [0, 0]], dtype=torch.long))
    hdata2 = HData(x=x, hyperedge_index=torch.tensor([[2, 3], [1, 1]], dtype=torch.long))

    with pytest.raises(
        ValueError,
        match=re.escape(
            "'global_node_ids' must have one entry per node. Got size=3 but num_nodes=4."
        ),
    ):
        HData.cat_same_node_space(
            [hdata1, hdata2],
            x=torch.randn(4, 4, dtype=torch.float),
            global_node_ids=torch.arange(3, dtype=torch.long),
        )


def test_cat_same_node_space_concatenates_hyperedge_attr():
    x = torch.randn(4, 2, dtype=torch.float)
    attr1 = torch.randn(1, 3, dtype=torch.float)
    attr2 = torch.randn(1, 3, dtype=torch.float)
    hdata1 = HData(
        x=x, hyperedge_index=torch.tensor([[0, 1], [0, 0]], dtype=torch.long), hyperedge_attr=attr1
    )
    hdata2 = HData(
        x=x, hyperedge_index=torch.tensor([[2, 3], [1, 1]], dtype=torch.long), hyperedge_attr=attr2
    )

    result = HData.cat_same_node_space([hdata1, hdata2])

    assert result.hyperedge_attr is not None
    assert torch.equal(result.hyperedge_attr, torch.cat([attr1, attr2], dim=0))


def test_cat_same_node_space_concatenates_hyperedge_weights():
    x = torch.randn(4, 2, dtype=torch.float)
    weights1 = torch.tensor([0.25], dtype=torch.float64)
    weights2 = torch.tensor([0.75], dtype=torch.float64)
    hdata1 = HData(
        x=x,
        hyperedge_index=torch.tensor([[0, 1], [0, 0]], dtype=torch.long),
        hyperedge_weights=weights1,
    )
    hdata2 = HData(
        x=x,
        hyperedge_index=torch.tensor([[2, 3], [1, 1]], dtype=torch.long),
        hyperedge_weights=weights2,
    )

    result = HData.cat_same_node_space([hdata1, hdata2])

    assert result.hyperedge_weights is not None
    assert result.hyperedge_weights.dtype == weights1.dtype
    assert torch.equal(result.hyperedge_weights, torch.cat([weights1, weights2], dim=0))


def test_cat_same_node_space_drops_hyperedge_weights_when_partially_missing():
    x = torch.randn(4, 2, dtype=torch.float)
    hdata1 = HData(
        x=x,
        hyperedge_index=torch.tensor([[0, 1], [0, 0]], dtype=torch.long),
        hyperedge_weights=torch.tensor([0.25], dtype=torch.float),
    )
    hdata2 = HData(
        x=x,
        hyperedge_index=torch.tensor([[2, 3], [1, 1]], dtype=torch.long),
        hyperedge_weights=None,
    )

    result = HData.cat_same_node_space([hdata1, hdata2])

    assert result.hyperedge_weights is None


def test_cat_same_node_space_drops_hyperedge_attr_when_partially_missing():
    x = torch.randn(4, 2, dtype=torch.float)
    hdata1 = HData(
        x=x,
        hyperedge_index=torch.tensor([[0, 1], [0, 0]], dtype=torch.long),
        hyperedge_attr=torch.randn(1, 3, dtype=torch.float),
    )
    hdata2 = HData(
        x=x, hyperedge_index=torch.tensor([[2, 3], [1, 1]], dtype=torch.long), hyperedge_attr=None
    )

    result = HData.cat_same_node_space([hdata1, hdata2])

    assert result.hyperedge_attr is None


def test_cat_same_node_space_does_not_share_mutable_storage_with_inputs(
    hdata_with_all_mutable_tensors,
):
    hdata = hdata_with_all_mutable_tensors
    other_hdata = HData(
        x=hdata.x,
        hyperedge_index=torch.tensor([[0, 4], [3, 3]], dtype=torch.long),
        hyperedge_weights=torch.tensor([0.4], dtype=torch.float),
        hyperedge_attr=torch.tensor([[4.0, 40.0]], dtype=torch.float),
        global_node_ids=hdata.global_node_ids,
        y=torch.tensor([0.0], dtype=torch.float),
    )

    result = HData.cat_same_node_space([hdata, other_hdata])

    __assert_mutating_result_keeps_source_tensors_unchanged(hdata, result)
    __assert_mutating_result_keeps_source_tensors_unchanged(other_hdata, result)


def test_cat_same_node_space_clones_provided_x_and_global_node_ids(hdata_with_all_mutable_tensors):
    custom_x = torch.full_like(
        hdata_with_all_mutable_tensors.x, 9.0, dtype=hdata_with_all_mutable_tensors.x.dtype
    )
    custom_global_node_ids = (
        torch.arange(hdata_with_all_mutable_tensors.num_nodes, dtype=torch.long) + 100
    )

    original_custom_x = custom_x.clone()
    original_custom_global_node_ids = custom_global_node_ids.clone()

    result = HData.cat_same_node_space(
        hdatas=[hdata_with_all_mutable_tensors], x=custom_x, global_node_ids=custom_global_node_ids
    )
    result.x.flatten()[0].add_(1)

    assert torch.equal(custom_x, original_custom_x)
    assert torch.equal(custom_global_node_ids, original_custom_global_node_ids)


def test_add_negative_samples_combines_positive_and_negative_hyperedges(mock_negative_sampler):
    hdata = HData(
        x=torch.arange(4, dtype=torch.float).unsqueeze(1),
        hyperedge_index=torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.long),
    )
    sampler, sampler_mock = mock_negative_sampler

    result = hdata.add_negative_samples(sampler, seed=42)

    assert result.num_nodes == hdata.num_nodes
    assert result.num_hyperedges == hdata.num_hyperedges + 1
    assert torch.equal(result.x, hdata.x)
    assert assign_hyperedge_label_to_nodes(
        result.hyperedge_index, result.y, result.num_hyperedges
    ) == {
        frozenset({0, 1}): 1,
        frozenset({2, 3}): 1,
        frozenset({0, 2}): 0,
    }
    sampler_mock.sample.assert_called_once_with(hdata, seed=42)


def test_add_negative_samples_returns_new_hdata_and_keeps_source_unchanged(
    mock_negative_sampler,
):
    original_hyperedge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.long)
    hdata = HData(
        x=torch.arange(4, dtype=torch.float).unsqueeze(1),
        hyperedge_index=original_hyperedge_index,
    )
    sampler, _ = mock_negative_sampler

    result = hdata.add_negative_samples(sampler, seed=42)

    assert result is not hdata
    assert result.hyperedge_index is not hdata.hyperedge_index
    assert torch.equal(hdata.hyperedge_index, original_hyperedge_index)
    assert torch.equal(hdata.y, torch.ones(hdata.num_hyperedges, dtype=torch.float))


def test_add_negative_samples_with_seed_is_reproducible():
    hdata = HData(
        x=torch.arange(5, dtype=torch.float).unsqueeze(1),
        hyperedge_index=torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 1, 2]], dtype=torch.long),
    )
    sampler = RandomNegativeSampler(num_negative_samples=3, num_nodes_per_sample=2)

    result_a = hdata.add_negative_samples(sampler, seed=123)
    result_b = hdata.add_negative_samples(sampler, seed=123)

    assert torch.equal(result_a.hyperedge_index, result_b.hyperedge_index)
    assert torch.equal(result_a.y, result_b.y)


@pytest.mark.parametrize(
    "split_ids, expected_num_nodes, expected_num_hyperedges, expected_hyperedge_index",
    [
        pytest.param(
            torch.tensor([0], dtype=torch.long),
            3,
            1,
            torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long),
            id="first_hyperedge",
        ),
        pytest.param(
            torch.tensor([1], dtype=torch.long),
            2,
            1,
            torch.tensor([[0, 1], [0, 0]], dtype=torch.long),
            id="second_hyperedge",
        ),  # nodes and hyperedges are mapped to be 0-based
        pytest.param(
            torch.tensor([0, 1], dtype=torch.long),
            4,
            2,
            torch.tensor([[0, 1, 2, 2, 3], [0, 0, 0, 1, 1]], dtype=torch.long),
            id="both_hyperedges",
        ),
        pytest.param(
            torch.tensor([0], dtype=torch.long),
            3,
            1,
            torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long),
            id="subset_hyperedges",
        ),
    ],
)
def test_split_inductive_counts(
    split_ids, expected_num_nodes, expected_num_hyperedges, expected_hyperedge_index
):
    x = torch.randn(4, 2, dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2, 2, 3], [0, 0, 0, 1, 1]], dtype=torch.long)
    hdata = HData(x=x, hyperedge_index=hyperedge_index)

    result = HData.split(
        hdata,
        split_hyperedge_ids=split_ids,
        node_space_setting="inductive",
    )

    assert result.num_nodes == expected_num_nodes
    assert result.num_hyperedges == expected_num_hyperedges
    assert torch.equal(result.hyperedge_index, expected_hyperedge_index)


@pytest.mark.parametrize(
    "split_ids, expected_num_nodes, expected_num_hyperedges, expected_hyperedge_index",
    [
        pytest.param(
            torch.tensor([0], dtype=torch.long),
            4,
            1,
            torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long),
            id="first_hyperedge",
        ),
        pytest.param(
            torch.tensor([1], dtype=torch.long),
            4,
            1,
            torch.tensor([[2, 3], [0, 0]], dtype=torch.long),
            id="second_hyperedge",
        ),
        pytest.param(
            torch.tensor([0, 1], dtype=torch.long),
            4,
            2,
            torch.tensor([[0, 1, 2, 2, 3], [0, 0, 0, 1, 1]], dtype=torch.long),
            id="both_hyperedges",
        ),
        pytest.param(
            torch.tensor([0], dtype=torch.long),
            4,
            1,
            torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long),
            id="subset_hyperedges",
        ),
    ],
)
def test_split_transductive_counts(
    split_ids, expected_num_nodes, expected_num_hyperedges, expected_hyperedge_index
):
    x = torch.randn(4, 2, dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2, 2, 3], [0, 0, 0, 1, 1]], dtype=torch.long)
    hdata = HData(x=x, hyperedge_index=hyperedge_index)

    result = HData.split(
        hdata,
        split_hyperedge_ids=split_ids,
        node_space_setting="transductive",
    )

    assert result.num_nodes == expected_num_nodes
    assert result.num_hyperedges == expected_num_hyperedges
    assert torch.equal(result.hyperedge_index, expected_hyperedge_index)


def test_default_hdata_splitter_materializes_explicit_hyperedge_ids():
    x = torch.randn(4, 2, dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2, 2, 3], [0, 0, 0, 1, 1]], dtype=torch.long)
    hdata = HData(x=x, hyperedge_index=hyperedge_index)

    result = DefaultHDataSplitter(node_space_setting="inductive").split(
        to_split=hdata, split_hyperedge_ids=torch.tensor([1], dtype=torch.long)
    )

    assert result.num_nodes == 2
    assert result.num_hyperedges == 1
    assert torch.equal(result.hyperedge_index, torch.tensor([[0, 1], [0, 0]], dtype=torch.long))
    assert torch.equal(result.x, x[torch.tensor([2, 3], dtype=torch.long)])


def test_split_delegates_to_custom_hdata_splitter():
    hdata = HData(
        x=torch.randn(2, 1, dtype=torch.float),
        hyperedge_index=torch.tensor([[0, 1], [0, 0]], dtype=torch.long),
    )

    expected_splitted_hdata = HData(
        x=torch.randn(2, 1, dtype=torch.float),
        hyperedge_index=torch.tensor([[0, 1], [0, 0]], dtype=torch.long),
    )

    class CustomHDataSplitter(Splitter[HData, Any]):
        def split(self, to_split: HData, **kwargs) -> HData:
            assert to_split is hdata
            return expected_splitted_hdata

    splitter = CustomHDataSplitter()
    result = HData.split(hdata, splitter=splitter)

    assert result is expected_splitted_hdata


def test_split_raises_on_invalid_node_space_setting():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "'node_space_setting' must be one of 'transductive' or 'inductive', got 'semi'."
        ),
    ):
        hdata = HData(
            x=torch.randn(2, 1, dtype=torch.float),
            hyperedge_index=torch.tensor([[0, 1], [0, 0]], dtype=torch.long),
        )
        HData.split(
            hdata,
            split_hyperedge_ids=torch.tensor([0], dtype=torch.long),
            node_space_setting=cast(Any, "semi"),
        )


def test_split_raises_when_split_hyperedge_ids_and_splitter_are_missing():
    hdata = HData(
        x=torch.randn(2, 1, dtype=torch.float),
        hyperedge_index=torch.tensor([[0, 1], [0, 0]], dtype=torch.long),
    )

    with pytest.raises(
        ValueError,
        match=re.escape("'split_hyperedge_ids' must be provided when 'splitter' is not provided."),
    ):
        HData.split(hdata)


def test_split_inductive_subsets_node_features():
    x = torch.tensor([[10.0], [20.0], [30.0], [40.0], [50.0]], dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 3, 4], [0, 0, 1, 1]], dtype=torch.long)
    hdata = HData(x=x, hyperedge_index=hyperedge_index)

    hyperedge_ids = torch.tensor(
        [1], dtype=torch.long
    )  # Split by hyperedge 1, which includes nodes 3 and 4
    result = HData.split(
        hdata,
        split_hyperedge_ids=hyperedge_ids,
        node_space_setting="inductive",
    )

    # Only nodes 3 and 4 should be included
    assert result.num_nodes == 2
    assert torch.equal(result.x, torch.tensor([[40.0], [50.0]], dtype=torch.float))


def test_split_subsets_labels():
    x = torch.randn(4, 2, dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.long)
    y = torch.tensor([1.0, 0.0], dtype=torch.float)
    hdata = HData(x=x, hyperedge_index=hyperedge_index, y=y)

    hyperedge_ids = torch.tensor([1], dtype=torch.long)  # Split by hyperedge 1, which has label 0.0
    result = HData.split(hdata, split_hyperedge_ids=hyperedge_ids)

    assert torch.equal(result.y, torch.tensor([0.0], dtype=torch.float))


@pytest.mark.parametrize(
    "node_space_setting, split_hyperedge_ids, expected_global_node_ids",
    [
        pytest.param(
            "transductive",
            torch.tensor([1], dtype=torch.long),
            torch.arange(4, dtype=torch.long),
            id="transductive",
        ),
        pytest.param(
            "inductive",
            torch.tensor([1], dtype=torch.long),
            torch.tensor([2, 3], dtype=torch.long),
            id="inductive",
        ),
    ],
)
def test_split_handles_none_global_node_ids(
    node_space_setting, split_hyperedge_ids, expected_global_node_ids
):
    x = torch.tensor([[10.0], [20.0], [30.0], [40.0]], dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.long)
    hdata = HData(x=x, hyperedge_index=hyperedge_index, global_node_ids=None)

    result = HData.split(
        hdata,
        split_hyperedge_ids=split_hyperedge_ids,
        node_space_setting=node_space_setting,
    )

    assert result.global_node_ids is not None
    assert torch.equal(result.global_node_ids, expected_global_node_ids)


def test_split_transductive_keeps_full_x_and_global_node_ids():
    x = torch.tensor([[10.0], [20.0], [30.0], [40.0], [50.0]], dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 2, 3, 4], [0, 0, 1, 1]], dtype=torch.long)
    global_node_ids = torch.tensor([10, 20, 30, 40, 50], dtype=torch.long)
    hdata = HData(
        x=x,
        hyperedge_index=hyperedge_index,
        global_node_ids=global_node_ids,
        y=torch.tensor([1.0, 0.0], dtype=torch.float),
    )

    result = HData.split(
        hdata,
        split_hyperedge_ids=torch.tensor([1], dtype=torch.long),
        node_space_setting="transductive",
    )

    assert result.num_nodes == hdata.num_nodes
    assert torch.equal(result.x, x)
    assert result.global_node_ids is not None
    assert torch.equal(result.global_node_ids, global_node_ids)
    assert torch.equal(result.hyperedge_index, torch.tensor([[3, 4], [0, 0]], dtype=torch.long))
    assert torch.equal(result.y, torch.tensor([0.0], dtype=torch.float))


def test_split_transductive_handles_none_global_node_ids():
    x = torch.tensor([[10.0], [20.0], [30.0], [40.0], [50.0]], dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 2, 3, 4], [0, 0, 1, 1]], dtype=torch.long)
    hdata = HData(
        x=x,
        hyperedge_index=hyperedge_index,
        y=torch.tensor([1.0, 0.0], dtype=torch.float),
        global_node_ids=None,
    )

    result = HData.split(
        hdata,
        split_hyperedge_ids=torch.tensor([1], dtype=torch.long),
        node_space_setting="transductive",
    )

    assert result.global_node_ids is not None
    assert torch.equal(result.global_node_ids, torch.arange(hdata.num_nodes, dtype=torch.long))


def test_split_transductive_does_not_share_mutable_storage_with_source(
    hdata_with_all_mutable_tensors,
):
    hdata = hdata_with_all_mutable_tensors

    result = HData.split(
        hdata,
        split_hyperedge_ids=torch.tensor([1], dtype=torch.long),
        node_space_setting="transductive",
    )

    __assert_mutating_result_keeps_source_tensors_unchanged(hdata, result)


def test_split_subsets_edge_attr():
    x = torch.randn(4, 2, dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.long)
    edge_attr = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float)
    hdata = HData(x=x, hyperedge_index=hyperedge_index, hyperedge_attr=edge_attr)

    hyperedge_ids = torch.tensor(
        [1], dtype=torch.long
    )  # Split by hyperedge 1, which has hyperedge_attr [3.0, 4.0]
    result = HData.split(hdata, split_hyperedge_ids=hyperedge_ids)

    assert result.hyperedge_attr is not None
    assert torch.equal(result.hyperedge_attr, torch.tensor([[3.0, 4.0]], dtype=torch.float))


def test_split_handles_none_edge_attr():
    x = torch.randn(4, 2, dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.long)
    hdata = HData(x=x, hyperedge_index=hyperedge_index, hyperedge_attr=None)

    hyperedge_ids = torch.tensor(
        [1], dtype=torch.long
    )  # Split by hyperedge 1, which has hyperedge_attr None
    result = HData.split(hdata, split_hyperedge_ids=hyperedge_ids)

    assert result.hyperedge_attr is None


def test_split_subsets_hyperedge_weights():
    x = torch.randn(4, 2, dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.long)
    hyperedge_weights = torch.tensor([0.25, 0.75], dtype=torch.float)
    hdata = HData(x=x, hyperedge_index=hyperedge_index, hyperedge_weights=hyperedge_weights)

    hyperedge_ids = torch.tensor([1], dtype=torch.long)
    result = HData.split(hdata, split_hyperedge_ids=hyperedge_ids)

    assert result.hyperedge_weights is not None
    assert torch.equal(result.hyperedge_weights, torch.tensor([0.75], dtype=torch.float))


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


def test_with_y_to_does_not_share_mutable_storage_with_source(hdata_with_all_mutable_tensors):
    hdata = hdata_with_all_mutable_tensors

    result = hdata.with_y_to(0.5)

    __assert_mutating_result_keeps_source_tensors_unchanged(hdata, result)


def test_enrich_node_features_replace(mock_hdata):
    enricher = MagicMock(spec=NodeEnricher)
    enriched_x = torch.randn(5, 3, dtype=torch.float)
    enricher.enrich.return_value = enriched_x

    result = mock_hdata.enrich_node_features(enricher)

    enricher.enrich.assert_called_once_with(mock_hdata.hyperedge_index)
    assert torch.equal(result.x, enriched_x)


def test_enrich_node_features_concatenate(mock_hdata):
    original_x = mock_hdata.x.clone()

    enricher = MagicMock(spec=NodeEnricher)
    enriched_x = torch.randn(5, 3, dtype=torch.float)
    enricher.enrich.return_value = enriched_x

    result = mock_hdata.enrich_node_features(enricher, enrichment_mode="concatenate")

    enricher.enrich.assert_called_once_with(mock_hdata.hyperedge_index)
    expected_x = torch.cat([original_x, enriched_x], dim=1)
    assert torch.equal(result.x, expected_x)
    assert result.x.shape == (5, 7)  # 4 original + 3 enriched


@pytest.mark.parametrize(
    "enrich_method",
    [
        pytest.param("enrich_node_features", id="node_features"),
        pytest.param("enrich_hyperedge_weights", id="hyperedge_weights"),
        pytest.param("enrich_hyperedge_attr", id="hyperedge_attr"),
    ],
)
def test_enrich_rejects_invalid_enrichment_mode(mock_hdata, enrich_method):
    enricher_spec = NodeEnricher if enrich_method == "enrich_node_features" else HyperedgeEnricher
    enricher = MagicMock(spec=enricher_spec)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "'enrichment_mode' must be one of 'replace', 'concatenate', or None, got 'append'."
        ),
    ):
        getattr(mock_hdata, enrich_method)(enricher, enrichment_mode=cast(Any, "append"))

    enricher.enrich.assert_not_called()


@pytest.mark.parametrize(
    "enrichment_mode",
    [
        pytest.param("replace", id="replace"),
        pytest.param("concatenate", id="concatenate"),
    ],
)
def test_enrich_node_features_replace_preserves_global_node_ids(mock_hdata, enrichment_mode):
    global_node_ids = torch.tensor([10, 20, 30, 40, 50], dtype=torch.long)
    mock_hdata.global_node_ids = global_node_ids

    enricher = MagicMock(spec=NodeEnricher)
    enricher.enrich.return_value = torch.randn(5, 3, dtype=torch.float)

    result = mock_hdata.enrich_node_features(enricher, enrichment_mode=enrichment_mode)

    assert result.global_node_ids is not None
    assert torch.equal(result.global_node_ids, global_node_ids)


def test_enrich_node_features_from_aligns_by_global_node_ids():
    source_hdata = HData(
        x=torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=torch.float),
        hyperedge_index=torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long),
        global_node_ids=torch.tensor([100, 200, 300], dtype=torch.long),
    )
    target_hdata = HData(
        x=torch.tensor([[0.0], [0.0]], dtype=torch.float),
        hyperedge_index=torch.tensor([[0, 1], [0, 0]], dtype=torch.long),
        global_node_ids=torch.tensor([300, 100], dtype=torch.long),
        y=torch.tensor([0.0], dtype=torch.float),
    )

    result = target_hdata.enrich_node_features_from(source_hdata)

    assert torch.equal(result.x, torch.tensor([[3.0, 30.0], [1.0, 10.0]], dtype=torch.float))
    assert torch.equal(result.hyperedge_index, target_hdata.hyperedge_index)
    assert result.hyperedge_weights is None
    assert result.hyperedge_attr is None
    assert result.global_node_ids is not None
    assert torch.equal(
        result.global_node_ids, utils.to_non_empty_edgeattr(target_hdata.global_node_ids)
    )
    assert torch.equal(result.y, target_hdata.y)


def test_enrich_node_features_from_raises_when_source_rows_do_not_match_global_node_ids():
    source_hdata = HData(
        x=torch.empty((0, 0), dtype=torch.float),
        hyperedge_index=torch.tensor([[0, 1], [0, 0]], dtype=torch.long),
    )
    target_hdata = HData(
        x=torch.tensor([[0.0]], dtype=torch.float),
        hyperedge_index=torch.tensor([[0], [0]], dtype=torch.long),
        global_node_ids=torch.tensor([0], dtype=torch.long),
    )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Expected 'hdata_with_features.x' rows to align with "
            "hdata_with_features.global_node_ids."
        ),
    ):
        target_hdata.enrich_node_features_from(source_hdata)


def test_enrich_node_features_from_raises_when_target_node_missing_from_source():
    source_hdata = HData(
        x=torch.tensor([[1.0], [2.0]], dtype=torch.float),
        hyperedge_index=torch.tensor([[0, 1], [0, 0]], dtype=torch.long),
        global_node_ids=torch.tensor([10, 20], dtype=torch.long),
    )
    target_hdata = HData(
        x=torch.tensor([[0.0], [0.0]], dtype=torch.float),
        hyperedge_index=torch.tensor([[0, 1], [0, 0]], dtype=torch.long),
        global_node_ids=torch.tensor([10, 30], dtype=torch.long),
    )

    with pytest.raises(
        ValueError,
        match=re.escape("Missing node features for target global_node_ids: [30]."),
    ):
        target_hdata.enrich_node_features_from(source_hdata)


@pytest.mark.parametrize(
    "fill_value, expected_x",
    [
        pytest.param(
            0.5, torch.tensor([[1.0, 10.0], [0.5, 0.5]], dtype=torch.float), id="scalar_fill_value"
        ),
        pytest.param(
            [7.0, 8.0],
            torch.tensor([[1.0, 10.0], [7.0, 8.0]], dtype=torch.float),
            id="vector_fill_value",
        ),
        pytest.param(
            torch.tensor([7.0, 8.0], dtype=torch.float),
            torch.tensor([[1.0, 10.0], [7.0, 8.0]], dtype=torch.float),
            id="tensor_fill_value",
        ),
        pytest.param(
            [0.5],
            torch.tensor([[1.0, 10.0], [0.5, 0.5]], dtype=torch.float),
            id="missing_dimensions_scalar_vector_fill_value",
        ),
        pytest.param(
            torch.tensor(0.5, dtype=torch.float),
            torch.tensor([[1.0, 10.0], [0.5, 0.5]], dtype=torch.float),
            id="missing_dimensions_scalar_tensor_fill_value",
        ),
    ],
)
def test_enrich_node_features_from_inductive_fill_value(fill_value, expected_x):
    source_hdata = HData(
        x=torch.tensor([[1.0, 10.0], [2.0, 20.0]], dtype=torch.float),
        hyperedge_index=torch.tensor([[0, 1], [0, 0]], dtype=torch.long),
        global_node_ids=torch.tensor([10, 20], dtype=torch.long),
    )
    target_hdata = HData(
        x=torch.tensor([[0.0], [0.0]], dtype=torch.float),
        hyperedge_index=torch.tensor([[0, 1], [0, 0]], dtype=torch.long),
        global_node_ids=torch.tensor([10, 30], dtype=torch.long),
    )

    result = target_hdata.enrich_node_features_from(
        source_hdata,
        node_space_setting="inductive",
        fill_value=fill_value,
    )

    assert torch.equal(result.x, expected_x)


def test_enrich_node_features_from_inductive_raises_without_fill_value():
    source_hdata = HData(
        x=torch.tensor([[1.0, 10.0], [2.0, 20.0]], dtype=torch.float),
        hyperedge_index=torch.tensor([[0, 1], [0, 0]], dtype=torch.long),
        global_node_ids=torch.tensor([10, 20], dtype=torch.long),
    )
    target_hdata = HData(
        x=torch.tensor([[0.0], [0.0]], dtype=torch.float),
        hyperedge_index=torch.tensor([[0, 1], [0, 0]], dtype=torch.long),
        global_node_ids=torch.tensor([10, 30], dtype=torch.long),
    )

    with pytest.raises(
        ValueError,
        match=re.escape("'fill_value' must be provided when node_space_setting='inductive'."),
    ):
        target_hdata.enrich_node_features_from(
            source_hdata,
            node_space_setting="inductive",
        )


def test_enrich_node_features_from_transductive_raises_when_fill_value_provided():
    source_hdata = HData(
        x=torch.tensor([[1.0, 10.0], [2.0, 20.0]], dtype=torch.float),
        hyperedge_index=torch.tensor([[0, 1], [0, 0]], dtype=torch.long),
        global_node_ids=torch.tensor([10, 20], dtype=torch.long),
    )
    target_hdata = HData(
        x=torch.tensor([[0.0]], dtype=torch.float),
        hyperedge_index=torch.tensor([[0], [0]], dtype=torch.long),
        global_node_ids=torch.tensor([10], dtype=torch.long),
    )

    with pytest.raises(
        ValueError,
        match=re.escape("'fill_value' cannot be provided when node_space_setting='transductive'."),
    ):
        target_hdata.enrich_node_features_from(
            source_hdata,
            node_space_setting="transductive",
            fill_value=0.0,
        )


def test_enrich_node_features_from_non_transductive_raises_on_fill_value_shape_mismatch():
    source_hdata = HData(
        x=torch.tensor([[1.0, 10.0], [2.0, 20.0]], dtype=torch.float),
        hyperedge_index=torch.tensor([[0, 1], [0, 0]], dtype=torch.long),
        global_node_ids=torch.tensor([10, 20], dtype=torch.long),
    )
    target_hdata = HData(
        x=torch.tensor([[0.0], [0.0]], dtype=torch.float),
        hyperedge_index=torch.tensor([[0, 1], [0, 0]], dtype=torch.long),
        global_node_ids=torch.tensor([10, 30], dtype=torch.long),
    )

    with pytest.raises(
        ValueError,
        match=re.escape("Expected 'fill_value' to define exactly 2 features, got shape (3,)."),
    ):
        target_hdata.enrich_node_features_from(
            source_hdata,
            node_space_setting="inductive",
            fill_value=[1.0, 2.0, 3.0],
        )


def test_enrich_methods_do_not_share_mutable_storage_with_source(hdata_with_all_mutable_tensors):
    node_enricher = MagicMock(spec=NodeEnricher)
    node_enricher.enrich.return_value = torch.ones(
        (hdata_with_all_mutable_tensors.num_nodes, 1), dtype=hdata_with_all_mutable_tensors.x.dtype
    )

    feature_source_hdata = hdata_with_all_mutable_tensors.clone()
    feature_source_hdata.x = torch.full(
        (hdata_with_all_mutable_tensors.num_nodes, 2), 7.0, dtype=torch.float
    )

    weight_enricher = MagicMock(spec=HyperedgeEnricher)
    weight_enricher.enrich.return_value = torch.full(
        (hdata_with_all_mutable_tensors.num_hyperedges,),
        0.8,
        dtype=torch.float,
    )

    attr_enricher = MagicMock(spec=HyperedgeEnricher)
    attr_enricher.enrich.return_value = torch.full(
        (hdata_with_all_mutable_tensors.num_hyperedges, 2),
        8.0,
        dtype=torch.float,
    )

    results = [
        hdata_with_all_mutable_tensors.enrich_node_features(
            node_enricher, enrichment_mode="concatenate"
        ),
        hdata_with_all_mutable_tensors.enrich_node_features_from(feature_source_hdata),
        hdata_with_all_mutable_tensors.enrich_hyperedge_weights(weight_enricher),
        hdata_with_all_mutable_tensors.enrich_hyperedge_attr(attr_enricher),
    ]

    for result in results:
        __assert_mutating_result_keeps_source_tensors_unchanged(
            hdata_with_all_mutable_tensors, result
        )


def test_enrich_node_features_from_raises_on_invalid_node_space_setting():
    source_hdata = HData(
        x=torch.tensor([[1.0]], dtype=torch.float),
        hyperedge_index=torch.tensor([[0], [0]], dtype=torch.long),
        global_node_ids=torch.tensor([10], dtype=torch.long),
    )
    target_hdata = HData(
        x=torch.tensor([[0.0]], dtype=torch.float),
        hyperedge_index=torch.tensor([[0], [0]], dtype=torch.long),
        global_node_ids=torch.tensor([10], dtype=torch.long),
    )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "'node_space_setting' must be one of 'transductive' or 'inductive', got 'semi'."
        ),
    ):
        target_hdata.enrich_node_features_from(
            source_hdata,
            node_space_setting=cast(Any, "semi"),
        )


def test_enrich_hyperedge_weights_replace():
    x = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long)
    hyperedge_attr = torch.tensor([[10.0, 11.0], [20.0, 21.0]], dtype=torch.float)
    hyperedge_weights = torch.tensor([0.1, 0.2], dtype=torch.float)
    global_node_ids = torch.tensor([10, 20, 30], dtype=torch.long)
    y = torch.tensor([1.0, 0.0], dtype=torch.float)
    hdata = HData(
        x=x,
        hyperedge_index=hyperedge_index,
        hyperedge_weights=hyperedge_weights,
        hyperedge_attr=hyperedge_attr,
        global_node_ids=global_node_ids,
        y=y,
    )

    enriched_weights = torch.tensor([0.5, 0.9], dtype=torch.float)
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
    x = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long)
    hdata = HData(x=x, hyperedge_index=hyperedge_index, hyperedge_weights=None)

    enriched_weights = torch.tensor([0.3, 0.7], dtype=torch.float)
    enricher = MagicMock(spec=HyperedgeEnricher)
    enricher.enrich.return_value = enriched_weights

    result = hdata.enrich_hyperedge_weights(enricher, enrichment_mode="concatenate")

    enricher.enrich.assert_called_once_with(hyperedge_index)
    assert torch.equal(utils.to_non_empty_edgeattr(result.hyperedge_weights), enriched_weights)


def test_enrich_hyperedge_weights_concatenate_after_hyperedge_index_expansion():
    x = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long)
    hdata = HData(
        x=x,
        hyperedge_index=hyperedge_index,
        hyperedge_weights=torch.tensor([0.1, 0.2], dtype=torch.float),
    )
    hdata.hyperedge_index = torch.tensor([[0, 1, 2, 0], [0, 0, 1, 2]], dtype=torch.long)
    hdata.num_hyperedges = 3
    hdata.y = torch.ones(3, dtype=torch.float)

    enricher = MagicMock(spec=HyperedgeEnricher)
    enricher.enrich.return_value = torch.tensor([0.7], dtype=torch.float)

    result = hdata.enrich_hyperedge_weights(enricher, enrichment_mode="concatenate")

    enricher.enrich.assert_called_once_with(hdata.hyperedge_index)
    assert torch.equal(
        utils.to_non_empty_edgeattr(result.hyperedge_weights),
        torch.tensor([0.1, 0.2, 0.7], dtype=torch.float),
    )


def test_enrich_hyperedge_attr_replace():
    x = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long)
    hyperedge_weights = torch.tensor([0.1, 0.2], dtype=torch.float)
    hyperedge_attr = torch.tensor([[10.0], [20.0]], dtype=torch.float)
    global_node_ids = torch.tensor([10, 20, 30], dtype=torch.long)
    y = torch.tensor([1.0, 0.0], dtype=torch.float)
    hdata = HData(
        x=x,
        hyperedge_index=hyperedge_index,
        hyperedge_weights=hyperedge_weights,
        hyperedge_attr=hyperedge_attr,
        global_node_ids=global_node_ids,
        y=y,
    )

    enriched_attr = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float)
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
    x = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long)
    hdata = HData(x=x, hyperedge_index=hyperedge_index, hyperedge_attr=None)

    enriched_attr = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float)
    enricher = MagicMock(spec=HyperedgeEnricher)
    enricher.enrich.return_value = enriched_attr

    result = hdata.enrich_hyperedge_attr(enricher, enrichment_mode="concatenate")

    enricher.enrich.assert_called_once_with(hyperedge_index)
    assert torch.equal(utils.to_non_empty_edgeattr(result.hyperedge_attr), enriched_attr)


def test_get_device_if_all_consistent_returns_device_when_all_consistent():
    x = torch.randn(3, 2, dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)
    hdata = HData(x=x, hyperedge_index=hyperedge_index)

    assert hdata.get_device_if_all_consistent() == torch.device("cpu")


def test_get_device_if_all_consistent_raises_on_mixed_devices():
    x = torch.randn(3, 2, dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)
    hdata = HData(x=x, hyperedge_index=hyperedge_index)

    # Mock a different device on x
    hdata.x = MagicMock(device=torch.device("cuda:0"))

    with pytest.raises(ValueError, match="Inconsistent device placement"):
        hdata.get_device_if_all_consistent()


def test_get_device_if_all_consistent_includes_edge_attr():
    x = torch.randn(3, 2, dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)
    hyperedge_attr = torch.randn(1, 4, dtype=torch.float)
    hdata = HData(x=x, hyperedge_index=hyperedge_index, hyperedge_attr=hyperedge_attr)

    # All on CPU, but hyperedge_attr on different device
    hdata.hyperedge_attr = MagicMock(device=torch.device("cuda:0"))

    with pytest.raises(ValueError, match="Inconsistent device placement"):
        hdata.get_device_if_all_consistent()


def test_get_device_if_all_consistent_handles_none_global_node_ids():
    x = torch.randn(3, 2, dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)
    hyperedge_attr = torch.randn(1, 4, dtype=torch.float)
    hdata = HData(
        x=x,
        hyperedge_index=hyperedge_index,
        hyperedge_attr=hyperedge_attr,
        global_node_ids=None,
    )

    assert hdata.get_device_if_all_consistent() == torch.device("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_raises_on_inconsistent_device_placement_on_cuda():
    x = torch.randn(3, 4, dtype=torch.float).to("cuda")  # CUDA
    hyperedge_index = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)  # CPU

    with pytest.raises(ValueError, match="Inconsistent device placement"):
        HData(x=x, hyperedge_index=hyperedge_index)


@pytest.mark.skipif(not torch.mps.is_available(), reason="MPS not available")
def test_raises_on_inconsistent_device_placement_on_mps():
    x = torch.randn(3, 4, dtype=torch.float).to("mps")  # MPS
    hyperedge_index = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)  # CPU

    with pytest.raises(ValueError, match="Inconsistent device placement"):
        HData(x=x, hyperedge_index=hyperedge_index)


def test_shuffle_preserves_num_nodes_and_num_hyperedges(mock_hdata):
    shuffled_hdata = mock_hdata.shuffle(seed=42)

    assert shuffled_hdata.num_nodes == mock_hdata.num_nodes
    assert shuffled_hdata.num_hyperedges == mock_hdata.num_hyperedges
    assert shuffled_hdata.hyperedge_index.dtype == torch.long


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
    x = torch.randn(4, 2, dtype=torch.float)
    # Hyperedge 0 has nodes {0, 1}, hyperedge 1 has nodes {2, 3}
    hyperedge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.long)
    y = torch.tensor([1.0, 0.0], dtype=torch.float)
    hyperedge_attr = torch.tensor([[10.0], [20.0]], dtype=torch.float)
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
    mock_hdata.y = torch.tensor([1.0, 0.0, 0.5], dtype=torch.float)
    shuffled_hdata = mock_hdata.shuffle(seed=42)

    # Same multiset of labels
    assert sorted(shuffled_hdata.y.tolist()) == sorted(mock_hdata.y.tolist())


def test_shuffle_permutes_hyperedge_attr(mock_hdata):
    mock_hdata.hyperedge_attr = torch.tensor([[10.0], [20.0], [30.0]], dtype=torch.float)
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
    x = torch.randn(5, 4, dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 1, 2]], dtype=torch.long)
    y = torch.tensor([1.0, 0.0, 0.5], dtype=torch.float)
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


def test_shuffle_does_not_share_mutable_storage_with_source(hdata_with_all_mutable_tensors):
    hdata = hdata_with_all_mutable_tensors

    result = hdata.shuffle(seed=42)

    __assert_mutating_result_keeps_source_tensors_unchanged(hdata, result)


def test_from_hyperedge_index_clones_caller_owned_tensor():
    hyperedge_index = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)
    original_hyperedge_index = hyperedge_index.clone()

    result = HData.from_hyperedge_index(hyperedge_index)
    result.hyperedge_index.flatten()[0].add_(1)

    assert torch.equal(hyperedge_index, original_hyperedge_index)


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
            torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long),
            4,
            torch.zeros((2, 0), dtype=torch.long),
            id="single_hyperedge_below_k_removed",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long),
            3,
            torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long),
            id="single_hyperedge_at_exact_k_kept",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2, 3, 4], [0, 0, 0, 1, 1]], dtype=torch.long),
            3,
            torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long),
            id="two_hyperedges_first_kept_second_removed",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 1, 1]], dtype=torch.long),
            3,
            torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long),
            id="two_hyperedges_second_kept_first_removed",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2, 3, 4, 5], [0, 0, 0, 1, 1, 1]], dtype=torch.long),
            3,
            torch.tensor([[0, 1, 2, 3, 4, 5], [0, 0, 0, 1, 1, 1]], dtype=torch.long),
            id="two_hyperedges_both_kept",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2, 3, 4, 5], [0, 0, 1, 1, 2, 2]], dtype=torch.long),
            3,
            torch.zeros((2, 0), dtype=torch.long),
            id="three_hyperedges_all_removed",
        ),
    ],
)
def test_remove_hyperedges_with_fewer_than_k_nodes(hyperedge_index, k, expected_hyperedge_index):
    num_nodes = hyperedge_index[0].max().item() + 1
    num_hyperedges = hyperedge_index[1].unique().shape[0]
    x = torch.randn(num_nodes, 4, dtype=torch.float)
    y = torch.randn(num_hyperedges, dtype=torch.float)
    hyperedge_attr = torch.randn(num_hyperedges, 2, dtype=torch.float)
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
            torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 1, 1]], dtype=torch.long),
            3,
            torch.tensor([[10.0], [20.0], [30.0], [40.0], [50.0]], dtype=torch.float),
            torch.tensor([[30.0], [40.0], [50.0]], dtype=torch.float),
            id="disjoint_nodes_first_hyperedge_removed",
        ),
        pytest.param(
            # Hyperedge 0: nodes {0, 2} -> 2 nodes (removed), hyperedge 1: nodes {1, 2, 3}
            #                                                            -> 3 nodes (kept)
            # Node 2 is shared, so it survives because hyperedge 1 is kept
            # Node 0 is the only node removed as it is only in the removed hyperedge 0
            torch.tensor([[0, 2, 1, 2, 3], [0, 0, 1, 1, 1]], dtype=torch.long),
            3,
            torch.tensor([[10.0], [20.0], [30.0], [40.0]], dtype=torch.float),
            torch.tensor([[20.0], [30.0], [40.0]], dtype=torch.float),
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
            torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 1, 1]], dtype=torch.long),
            3,
            torch.tensor([1.0, 0.0], dtype=torch.float),
            torch.tensor([0.0], dtype=torch.float),
            id="disjoint_nodes_first_hyperedge_removed",
        ),
        pytest.param(
            # Hyperedge 0: nodes {0, 2} -> 2 nodes (removed). hyperedge 1: nodes {1, 2, 3}
            #                                                            -> 3 nodes (kept)
            # Node 2 is shared, so y for hyperedge 1 must survive
            torch.tensor([[0, 2, 1, 2, 3], [0, 0, 1, 1, 1]], dtype=torch.long),
            3,
            torch.tensor([1.0, 0.0], dtype=torch.float),
            torch.tensor([0.0], dtype=torch.float),
            id="shared_node_y_of_kept_hyperedge_survives",
        ),
    ],
)
def test_remove_hyperedges_with_fewer_than_k_nodes_subsets_y(hyperedge_index, k, y, expected_y):
    num_nodes = hyperedge_index[0].max().item() + 1
    x = torch.randn(num_nodes, 2, dtype=torch.float)
    hdata = HData(x=x, hyperedge_index=hyperedge_index, y=y)
    result = hdata.remove_hyperedges_with_fewer_than_k_nodes(k=k)

    assert torch.equal(result.y, expected_y)


@pytest.mark.parametrize(
    "hyperedge_index, k, hyperedge_attr, expected_attr",
    [
        pytest.param(
            torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 1, 1]], dtype=torch.long),
            3,
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float),
            torch.tensor([[3.0, 4.0]], dtype=torch.float),
            id="disjoint_nodes_first_hyperedge_removed",
        ),
        pytest.param(
            # Hyperedge 0: nodes {0, 2} -> 2 nodes (removed), hyperedge 1: nodes {1, 2, 3}
            #                                                            -> 3 nodes (kept)
            # Node 2 is shared, so attr for hyperedge 1 must survive
            torch.tensor([[0, 2, 1, 2, 3], [0, 0, 1, 1, 1]], dtype=torch.long),
            3,
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float),
            torch.tensor([[3.0, 4.0]], dtype=torch.float),
            id="shared_node_attr_of_kept_hyperedge_survives",
        ),
    ],
)
def test_remove_hyperedges_with_fewer_than_k_nodes_subsets_hyperedge_attr(
    hyperedge_index, k, hyperedge_attr, expected_attr
):
    num_nodes = hyperedge_index[0].max().item() + 1
    x = torch.randn(num_nodes, 2, dtype=torch.float)
    hdata = HData(x=x, hyperedge_index=hyperedge_index, hyperedge_attr=hyperedge_attr)
    result = hdata.remove_hyperedges_with_fewer_than_k_nodes(k=k)

    assert result.hyperedge_attr is not None
    assert torch.equal(result.hyperedge_attr, expected_attr)


def test_remove_hyperedges_with_fewer_than_k_nodes_keeps_none_hyperedge_attr():
    x = torch.randn(3, 2, dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long)
    hdata = HData(x=x, hyperedge_index=hyperedge_index, hyperedge_attr=None)

    result = hdata.remove_hyperedges_with_fewer_than_k_nodes(k=1)

    assert result.hyperedge_attr is None


def test_remove_hyperedges_with_fewer_than_k_nodes_rejects_invalid_k():
    x = torch.randn(2, 1, dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)
    hdata = HData(x=x, hyperedge_index=hyperedge_index)

    with pytest.raises(ValueError, match="'k' must be positive"):
        hdata.remove_hyperedges_with_fewer_than_k_nodes(k=0)


def test_remove_hyperedges_with_fewer_than_k_nodes_subsets_global_node_ids_when_preserve_true():
    x = torch.randn(5, 2, dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 1, 1]], dtype=torch.long)
    hdata = HData(x=x, hyperedge_index=hyperedge_index)
    hdata.global_node_ids = torch.tensor([10, 20, 30, 40, 50], dtype=torch.long)

    result = hdata.remove_hyperedges_with_fewer_than_k_nodes(k=3, preserve_global_node_ids=True)

    assert result.global_node_ids is not None
    assert torch.equal(result.global_node_ids, torch.tensor([30, 40, 50], dtype=torch.long))


def test_remove_hyperedges_with_fewer_than_k_nodes_not_subset_global_node_ids_when_preserve_false():
    x = torch.randn(5, 2, dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 1, 1]], dtype=torch.long)
    hdata = HData(x=x, hyperedge_index=hyperedge_index)
    hdata.global_node_ids = torch.tensor([10, 20, 30, 40, 50], dtype=torch.long)

    result = hdata.remove_hyperedges_with_fewer_than_k_nodes(k=3, preserve_global_node_ids=False)

    assert result.global_node_ids is not None
    assert torch.equal(result.global_node_ids, torch.arange(result.num_nodes, dtype=torch.long))


def test_remove_hyperedges_with_fewer_than_k_nodes_handles_none_global_node_ids():
    x = torch.randn(5, 2, dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 1, 1]], dtype=torch.long)
    hdata = HData(x=x, hyperedge_index=hyperedge_index, global_node_ids=None)

    result = hdata.remove_hyperedges_with_fewer_than_k_nodes(k=3)

    assert torch.equal(result.hyperedge_index[0], torch.tensor([0, 1, 2], dtype=torch.long))
    assert result.global_node_ids is not None
    assert torch.equal(result.global_node_ids, torch.tensor([0, 1, 2], dtype=torch.long))


def test_remove_hyperedges_with_fewer_than_k_nodes_subsets_hyperedge_weights():
    x = torch.randn(5, 2, dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 1, 1]], dtype=torch.long)
    hyperedge_weights = torch.tensor([0.25, 0.75], dtype=torch.float)
    hdata = HData(x=x, hyperedge_index=hyperedge_index, hyperedge_weights=hyperedge_weights)

    result = hdata.remove_hyperedges_with_fewer_than_k_nodes(k=3)

    assert result.hyperedge_weights is not None
    assert torch.equal(result.hyperedge_weights, torch.tensor([0.75], dtype=torch.float))


def test_remove_hyperedges_with_fewer_than_k_nodes_rebases_hyperedge_index():
    # Hyperedge 0 (nodes 0,1) removed, while hyperedge 1 (nodes 2,3,4) kept.
    # After filtering, surviving nodes 2, 3, and 4, and hyperedge 1 must be rebased to 0-based.
    x = torch.randn(5, 2, dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 1, 1]], dtype=torch.long)
    hdata = HData(x=x, hyperedge_index=hyperedge_index)

    result = hdata.remove_hyperedges_with_fewer_than_k_nodes(k=3)

    assert torch.equal(
        result.hyperedge_index, torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long)
    )


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


def __assert_mutating_result_keeps_source_tensors_unchanged(
    source: HData,
    result: HData,
    field_names: tuple[str, ...] = (
        "x",
        "hyperedge_index",
        "hyperedge_weights",
        "hyperedge_attr",
        "global_node_ids",
        "y",
    ),
) -> None:
    original_tensors = {
        field_name: getattr(source, field_name).clone()
        for field_name in field_names
        if getattr(source, field_name) is not None
    }

    for field_name in field_names:
        source_tensor = getattr(source, field_name)
        result_tensor = getattr(result, field_name)
        if source_tensor is None or result_tensor is None or result_tensor.numel() == 0:
            continue

        result_tensor.flatten()[0].add_(1)
        assert torch.equal(source_tensor, original_tensors[field_name])
