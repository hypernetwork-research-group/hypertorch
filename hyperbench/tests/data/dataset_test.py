import pytest
import requests
import tempfile
import torch

from unittest.mock import patch, mock_open, MagicMock
from hyperbench.data import AlgebraDataset, Dataset, HIFConverter, SamplingStrategy
from hyperbench.nn import EnrichmentMode, NodeEnricher, HyperedgeEnricher
from hyperbench.types import HData, HIFHypergraph


@pytest.fixture
def mock_hdata() -> HData:
    x = torch.ones((3, 1), dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long)
    return HData(x=x, hyperedge_index=hyperedge_index)


@pytest.fixture
def mock_hdata_with_hyperedge_attr() -> HData:
    x = torch.ones((3, 1), dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long)
    hyperedge_attr = torch.ones((3, 1), dtype=torch.float)
    return HData(x=x, hyperedge_index=hyperedge_index, hyperedge_attr=hyperedge_attr)


@pytest.fixture
def mock_hdata_with_hyperedge_weights() -> HData:
    x = torch.ones((3, 1), dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long)
    hyperedge_weights = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float)
    return HData(x=x, hyperedge_index=hyperedge_index, hyperedge_weights=hyperedge_weights)


@pytest.fixture
def mock_sample_hypergraph():
    return HIFHypergraph(
        network_type="undirected",
        nodes=[{"node": "0"}, {"node": "1"}],
        hyperedges=[{"edge": "0"}],
        incidences=[{"node": "0", "edge": "0"}],
    )


@pytest.fixture
def mock_simple_hypergraph():
    return HIFHypergraph(
        network_type="undirected",
        nodes=[{"node": "0", "attrs": {}}, {"node": "1", "attrs": {}}],
        hyperedges=[{"edge": "0", "attrs": {}}],
        incidences=[{"node": "0", "edge": "0"}],
    )


@pytest.fixture
def mock_three_node_weighted_hypergraph():
    return HIFHypergraph(
        network_type="undirected",
        nodes=[
            {"node": "0", "attrs": {}},
            {"node": "1", "attrs": {}},
            {"node": "2", "attrs": {}},
        ],
        hyperedges=[
            {"edge": "0", "attrs": {"weight": 1.0}},
            {"edge": "1", "attrs": {"weight": 2.0}},
        ],
        incidences=[
            {"node": "0", "edge": "0"},
            {"node": "1", "edge": "0"},
            {"node": "2", "edge": "1"},
        ],
    )


@pytest.fixture
def mock_four_node_hypergraph():
    return HIFHypergraph(
        network_type="undirected",
        nodes=[
            {"node": "0", "attrs": {}},
            {"node": "1", "attrs": {}},
            {"node": "2", "attrs": {}},
            {"node": "3", "attrs": {}},
        ],
        hyperedges=[{"edge": "0", "attrs": {}}, {"edge": "1", "attrs": {}}],
        incidences=[
            {"node": "0", "edge": "0"},
            {"node": "1", "edge": "0"},
            {"node": "2", "edge": "1"},
            {"node": "3", "edge": "1"},
        ],
    )


@pytest.fixture
def mock_five_node_hypergraph():
    return HIFHypergraph(
        network_type="undirected",
        nodes=[
            {"node": "0", "attrs": {}},
            {"node": "1", "attrs": {}},
            {"node": "2", "attrs": {}},
            {"node": "3", "attrs": {}},
            {"node": "4", "attrs": {}},
        ],
        hyperedges=[{"edge": "0", "attrs": {}}],
        incidences=[{"node": "0", "edge": "0"}],
    )


@pytest.fixture
def mock_no_edge_attr_hypergraph():
    return HIFHypergraph(
        network_type="undirected",
        nodes=[
            {"node": "0", "attrs": {}},
            {"node": "1", "attrs": {}},
        ],
        hyperedges=[{"edge": "0"}],
        incidences=[
            {"node": "0", "edge": "0"},
            {"node": "1", "edge": "0"},
        ],
    )


@pytest.fixture
def mock_multiple_edges_attr_hypergraph():
    return HIFHypergraph(
        network_type="undirected",
        nodes=[
            {"node": "0", "attrs": {}},
            {"node": "1", "attrs": {}},
            {"node": "2", "attrs": {}},
            {"node": "3", "attrs": {}},
        ],
        hyperedges=[
            {"edge": "0", "attrs": {"weight": 1.0}},
            {"edge": "1", "attrs": {"weight": 2.0}},
            {"edge": "2", "attrs": {"weight": 3.0}},
        ],
        incidences=[
            {"node": "0", "edge": "0"},
            {"node": "1", "edge": "0"},
            {"node": "2", "edge": "1"},
            {"node": "3", "edge": "2"},
        ],
    )


def test_HIFConverter_num_nodes_and_edges():
    dataset_name = "ALGEBRA"
    mock_hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[{"node": str(i)} for i in range(20)],
        hyperedges=[{"edge": str(i)} for i in range(30)],
        incidences=[{"node": "0", "edge": "0"}],
    )

    with patch.object(HIFConverter, "load_from_hif", return_value=mock_hypergraph):
        hypergraph = HIFConverter.load_from_hif(dataset_name)

    assert hypergraph is not None
    assert hasattr(hypergraph, "nodes")
    assert hasattr(hypergraph, "hyperedges")
    assert hasattr(hypergraph, "incidences")
    assert hasattr(hypergraph, "metadata")
    assert hasattr(hypergraph, "network_type")

    assert hypergraph.num_nodes == 20
    assert hypergraph.num_hyperedges == 30


def test_HIFConverter_loads_invalid_dataset():
    dataset_name = "INVALID_DATASET"

    with pytest.raises(ValueError, match="Dataset 'INVALID_DATASET' not found"):
        HIFConverter.load_from_hif(dataset_name)


def test_HIFConverter_loads_invalid_hif_format():
    dataset_name = "ALGEBRA"

    invalid_hif_json = '{"network-type": "undirected", "nodes": []}'

    with (
        patch("hyperbench.data.dataset.requests.get") as mock_get,
        patch("hyperbench.data.dataset.validate_hif_json", return_value=False),
        patch("builtins.open", mock_open(read_data=invalid_hif_json)),
        patch("hyperbench.data.dataset.zstd.ZstdDecompressor"),
    ):
        mock_response = mock_get.return_value
        mock_response.status_code = 200
        mock_response.content = b"mock_zst_content"

        with pytest.raises(ValueError, match="Dataset 'algebra' is not HIF-compliant"):
            HIFConverter.load_from_hif(dataset_name)


def test_HIFConverter_stores_on_disk_when_save_on_disk_true():
    dataset_name = "ALGEBRA"

    mock_hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[{"node": "0"}, {"node": "1"}],
        hyperedges=[{"edge": "0"}],
        incidences=[{"node": "0", "edge": "0"}],
    )

    mock_hif_json = {
        "network-type": "undirected",
        "nodes": [{"node": "0"}, {"node": "1"}],
        "edges": [{"edge": "0"}],
        "incidences": [{"node": "0", "edge": "0"}],
    }

    with (
        patch("hyperbench.data.dataset.requests.get") as mock_get,
        patch("hyperbench.data.dataset.os.path.exists", return_value=False),
        patch("hyperbench.data.dataset.os.makedirs"),
        patch("builtins.open", mock_open()) as mock_file,
        patch("hyperbench.data.dataset.zstd.ZstdDecompressor") as mock_decomp,
        patch("hyperbench.data.dataset.json.load", return_value=mock_hif_json),
        patch("hyperbench.data.dataset.validate_hif_json", return_value=True),
        patch.object(HIFHypergraph, "from_hif", return_value=mock_hypergraph),
    ):
        # Mock successful download
        mock_response = mock_get.return_value
        mock_response.status_code = 200
        mock_response.content = b"mock_zst_content"

        # Mock decompressor
        mock_stream = mock_decomp.return_value.stream_reader.return_value
        mock_stream.__enter__ = lambda self: mock_stream
        mock_stream.__exit__ = lambda self, *args: None

        hypergraph = HIFConverter.load_from_hif(dataset_name, save_on_disk=True)

        assert hypergraph is not None
        assert hypergraph.network_type == "undirected"
        mock_get.assert_called_once()
        # Verify file was written to disk (not temp file)
        assert mock_file.call_count >= 2  # Once for write, once for read


def test_HIFConverter_uses_temp_file_when_save_on_disk_false():
    dataset_name = "ALGEBRA"

    mock_hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[{"node": "0"}, {"node": "1"}],
        hyperedges=[{"edge": "0"}],
        incidences=[{"node": "0", "edge": "0"}],
    )

    mock_hif_json = {
        "network-type": "undirected",
        "nodes": [{"node": "0"}, {"node": "1"}],
        "edges": [{"edge": "0"}],
        "incidences": [{"node": "0", "edge": "0"}],
    }

    with (
        patch("hyperbench.data.dataset.requests.get") as mock_get,
        patch("hyperbench.data.dataset.os.path.exists", return_value=False),
        patch("hyperbench.data.dataset.tempfile.NamedTemporaryFile") as mock_temp,
        patch("builtins.open", mock_open()),
        patch("hyperbench.data.dataset.zstd.ZstdDecompressor") as mock_decomp,
        patch("hyperbench.data.dataset.json.load", return_value=mock_hif_json),
        patch("hyperbench.data.dataset.validate_hif_json", return_value=True),
        patch.object(HIFHypergraph, "from_hif", return_value=mock_hypergraph),
    ):
        # Mock successful download
        mock_response = mock_get.return_value
        mock_response.status_code = 200
        mock_response.content = b"mock_zst_content"

        # Mock temp file
        mock_temp_file = mock_temp.return_value.__enter__.return_value
        mock_temp_file.name = "/tmp/fake_temp.json.zst"

        # Mock decompressor
        mock_stream = mock_decomp.return_value.stream_reader.return_value
        mock_stream.__enter__ = lambda self: mock_stream
        mock_stream.__exit__ = lambda self, *args: None

        hypergraph = HIFConverter.load_from_hif(dataset_name, save_on_disk=False)

        assert hypergraph is not None
        assert hypergraph.network_type == "undirected"
        mock_get.assert_called_once()
        # Verify temp file was used
        assert mock_temp.call_count >= 1


def test_HIFConverter_download_failure():
    dataset_name = "ALGEBRA"

    with (
        patch("hyperbench.data.dataset.requests.get") as mock_get,
        patch("hyperbench.data.dataset.hf_hub_download", side_effect=Exception("HFHub failed")),
        patch("hyperbench.data.dataset.os.path.exists", return_value=False),
    ):
        # Mock failed download
        mock_response = mock_get.return_value
        mock_response.status_code = 404
        mock_response.content = b""

        with pytest.warns(
            UserWarning,
            match=r"(?s)GitHub raw download failed for dataset 'algebra' with status code 404.*Falling back to Hugging Face Hub download for dataset",
        ):
            with pytest.raises(
                ValueError,
                match=r"Failed to download dataset 'algebra'",
            ):
                HIFConverter.load_from_hif(dataset_name)


def test_HIFConverter_falls_back_to_hf_hub_download_when_github_raw_download_fails(
    tmp_path, mock_sample_hypergraph
):
    dataset_name = "ALGEBRA"

    mock_hypergraph = mock_sample_hypergraph

    mock_hif_json = {
        "network_type": "undirected",
        "nodes": [{"node": "0"}, {"node": "1"}],
        "edges": [{"edge": "0"}],
        "incidences": [{"node": "0", "edge": "0"}],
    }

    fallback_file = tmp_path / "algebra.json.zst"
    fallback_file.write_bytes(b"mock_zst_content")

    created_temp_files = []
    original_named_tempfile = tempfile.NamedTemporaryFile

    def named_tempfile_side_effect(*args, **kwargs):
        temp_file = original_named_tempfile(*args, **kwargs)
        created_temp_files.append(temp_file)
        return temp_file

    with (
        patch("hyperbench.data.dataset.requests.get") as mock_get,
        patch(
            "hyperbench.data.dataset.hf_hub_download",
            return_value=str(fallback_file),
        ) as mock_hf_hub_download,
        patch("hyperbench.data.dataset.os.path.exists", return_value=False),
        patch(
            "hyperbench.data.dataset.tempfile.NamedTemporaryFile",
            side_effect=named_tempfile_side_effect,
        ),
        patch("hyperbench.data.dataset.zstd.ZstdDecompressor") as mock_decomp,
        patch("hyperbench.data.dataset.json.load", return_value=mock_hif_json),
        patch("hyperbench.data.dataset.validate_hif_json", return_value=True),
        patch.object(HIFHypergraph, "from_hif", return_value=mock_hypergraph),
    ):
        mock_response = mock_get.return_value
        mock_response.status_code = 404
        mock_response.content = b""

        def fake_copy_stream(src, dst):
            dst.write(b'{"network_type":"undirected","nodes":[],"edges":[],"incidences":[]}')
            return

        mock_decomp.return_value.copy_stream.side_effect = fake_copy_stream

        with pytest.warns(
            UserWarning,
            match=r"(?s)GitHub raw download failed for dataset 'algebra' with status code 404.*Falling back to Hugging Face Hub download for dataset",
        ):
            hypergraph = HIFConverter.load_from_hif(dataset_name, save_on_disk=False)

        assert hypergraph.network_type == "undirected"
        mock_get.assert_called_once()
        mock_hf_hub_download.assert_called_once()
        assert created_temp_files[0].name is not None
        assert fallback_file.read_bytes() == b"mock_zst_content"


def test_HIFConverter_download_raises_when_network_error():
    dataset_name = "ALGEBRA"

    with (
        patch("hyperbench.data.dataset.requests.get") as mock_get,
        patch("hyperbench.data.dataset.os.path.exists", return_value=False),
    ):
        # Mock network error
        mock_get.side_effect = requests.RequestException("Network error")

        with pytest.raises(requests.RequestException, match="Network error"):
            HIFConverter.load_from_hif(dataset_name)


def test_init_with_prepare_false_and_no_hdata_raises():
    with pytest.raises(ValueError, match="hdata must be provided when prepare is set to False."):
        Dataset(hdata=None, prepare=False)


def test_dataset_is_not_available():
    class FakeMockDataset(Dataset):
        DATASET_NAME = "FAKE"

    with pytest.raises(ValueError, match=r"Dataset 'FAKE' not found"):
        FakeMockDataset()


@pytest.mark.parametrize(
    "strategy, expected_len",
    [
        pytest.param(SamplingStrategy.NODE, 4, id="node_strategy"),
        pytest.param(SamplingStrategy.HYPEREDGE, 2, id="hyperedge_strategy"),
    ],
)
def test_dataset_is_available_with_all_strategies(
    strategy, expected_len, mock_four_node_hypergraph
):
    with patch.object(HIFConverter, "load_from_hif", return_value=mock_four_node_hypergraph):
        dataset = AlgebraDataset(sampling_strategy=strategy)

        assert dataset.DATASET_NAME == "ALGEBRA"
        assert dataset.hypergraph is not None
        assert len(dataset) == expected_len


def test_download_already_downloaded_dataset_uses_local_value(mock_four_node_hypergraph):
    with patch.object(HIFConverter, "load_from_hif", return_value=mock_four_node_hypergraph):
        dataset = AlgebraDataset()

    hg1 = dataset.download()
    hg2 = dataset.download()

    assert hg1 is hg2


def test_throw_when_dataset_name_is_none():
    class FakeMockDataset(Dataset):
        DATASET_NAME = None

    with pytest.raises(
        ValueError,
        match=r"Dataset name \(provided: None\) must be provided\.",
    ):
        FakeMockDataset()


def test_dataset_process_no_incidences():
    mock_hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[{"node": "0", "attrs": {}}, {"node": "1", "attrs": {}}],
        hyperedges=[{"edge": "0", "attrs": {}}],
        incidences=[],
    )

    with patch.object(HIFConverter, "load_from_hif", return_value=mock_hypergraph):
        dataset = AlgebraDataset()

        assert dataset.hdata is not None
        assert dataset.hdata.x.shape[0] == 2
        assert dataset.hdata.hyperedge_index.shape[0] == 2
        assert dataset.hdata.hyperedge_index.shape[1] == 2
        assert dataset.hdata.hyperedge_attr is not None
        assert dataset.hdata.hyperedge_attr.shape == (2, 0)
        assert dataset.hdata.hyperedge_attr[0].shape == (0,)
        assert dataset.hdata.hyperedge_attr[1].shape == (0,)


def test_dataset_process_with_edge_attributes():
    mock_hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[
            {"node": "0", "attrs": {}},
            {"node": "1", "attrs": {}},
            {"node": "2", "attrs": {}},
        ],
        hyperedges=[
            {"edge": "0", "attrs": {"weight": 1.0, "type": 2.0}},
            {"edge": "1", "attrs": {"weight": 3.0, "type": 0.1}},
        ],
        incidences=[
            {"node": "0", "edge": "0"},
            {"node": "1", "edge": "0"},
            {"node": "2", "edge": "1"},
        ],
    )

    with patch.object(HIFConverter, "load_from_hif", return_value=mock_hypergraph):
        dataset = AlgebraDataset()

    assert dataset.hdata is not None
    assert dataset.hdata.x.shape[0] == 3
    assert dataset.hdata.hyperedge_index.shape[0] == 2
    assert dataset.hdata.hyperedge_index.shape[1] == 3
    assert dataset.hdata.hyperedge_attr is not None
    # Two edges with two attributes each: shape [2, 2]
    assert dataset.hdata.hyperedge_attr.shape == (2, 2)
    # Attributes maintain dictionary insertion order (no sorting)

    assert torch.allclose(dataset.hdata.hyperedge_attr[0], torch.tensor([1.0, 2.0]))  # weight, type
    assert torch.allclose(dataset.hdata.hyperedge_attr[1], torch.tensor([3.0, 0.1]))  # weight, type


def test_dataset_process_without_edge_attributes(mock_no_edge_attr_hypergraph):
    with patch.object(HIFConverter, "load_from_hif", return_value=mock_no_edge_attr_hypergraph):
        dataset = AlgebraDataset()

    assert dataset.hdata is not None
    assert dataset.hdata.hyperedge_index.shape[0] == 2
    assert dataset.hdata.hyperedge_index.shape[1] == 2
    assert dataset.hdata.hyperedge_attr is None


def test_dataset_process_hyperedge_index_in_correct_format(mock_four_node_hypergraph):
    with patch.object(HIFConverter, "load_from_hif", return_value=mock_four_node_hypergraph):
        dataset = AlgebraDataset()

    assert dataset.hdata.hyperedge_index.shape == (2, 4)
    assert torch.allclose(dataset.hdata.hyperedge_index[0], torch.tensor([0, 1, 2, 3]))
    assert torch.allclose(dataset.hdata.hyperedge_index[1], torch.tensor([0, 0, 1, 1]))


def test_dataset_process_random_ids():
    mock_hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[
            {"node": "abc", "attrs": {}},
            {"node": "ss", "attrs": {}},
            {"node": "fewao", "attrs": {}},
        ],
        hyperedges=[{"edge": "0", "attrs": {}}, {"edge": "1", "attrs": {}}],
        incidences=[
            {"node": "abc", "edge": "0"},
            {"node": "ss", "edge": "0"},
            {"node": "fewao", "edge": "1"},
        ],
    )

    with patch.object(HIFConverter, "load_from_hif", return_value=mock_hypergraph):
        dataset = AlgebraDataset()

    assert dataset.hdata.hyperedge_index.shape == (2, 3)
    assert torch.allclose(dataset.hdata.hyperedge_index[0], torch.tensor([0, 1, 2]))
    assert torch.allclose(dataset.hdata.hyperedge_index[1], torch.tensor([0, 0, 1]))
    assert dataset.hdata.hyperedge_attr is not None
    assert dataset.hdata.hyperedge_attr.shape == (2, 0)  # 2 edges, 0 attributes each


@pytest.mark.parametrize(
    "strategy",
    [
        pytest.param(SamplingStrategy.NODE, id="node_strategy"),
        pytest.param(SamplingStrategy.HYPEREDGE, id="hyperedge_strategy"),
    ],
)
def test_getitem_index_list_empty(mock_simple_hypergraph, strategy):
    with patch.object(HIFConverter, "load_from_hif", return_value=mock_simple_hypergraph):
        dataset = AlgebraDataset(sampling_strategy=strategy)

    with pytest.raises(ValueError, match="Index list cannot be empty."):
        dataset[[]]


@pytest.mark.parametrize(
    "strategy, index_list, expected_message",
    [
        pytest.param(
            SamplingStrategy.NODE,
            [0, 1, 2, 3, 4],
            r"Index list length \(5\) cannot exceed the number of sampleable items \(4\)\.",
            id="node_strategy",
        ),
        pytest.param(
            SamplingStrategy.HYPEREDGE,
            [0, 1, 2],
            r"Index list length \(3\) cannot exceed the number of sampleable items \(2\)\.",
            id="hyperedge_strategy",
        ),
    ],
)
def test_getitem_raises_when_index_list_larger_than_max(
    mock_four_node_hypergraph, strategy, index_list, expected_message
):
    with patch.object(HIFConverter, "load_from_hif", return_value=mock_four_node_hypergraph):
        dataset = AlgebraDataset(sampling_strategy=strategy)

    with pytest.raises(ValueError, match=expected_message):
        dataset[index_list]


@pytest.mark.parametrize(
    "strategy, index, expected_message",
    [
        pytest.param(
            SamplingStrategy.NODE, 4, r"Node ID 4 is out of bounds \(0, 3\)\.", id="node_strategy"
        ),
        pytest.param(
            SamplingStrategy.HYPEREDGE,
            2,
            r"Hyperedge ID 2 is out of bounds \(0, 1\)\.",
            id="hyperedge_strategy",
        ),
    ],
)
def test_getitem_raises_when_index_out_of_bounds(
    mock_four_node_hypergraph, strategy, index, expected_message
):
    with patch.object(HIFConverter, "load_from_hif", return_value=mock_four_node_hypergraph):
        dataset = AlgebraDataset(sampling_strategy=strategy)

    with pytest.raises(IndexError, match=expected_message):
        dataset[index]


@pytest.mark.parametrize(
    "strategy, index, expected_shape, expected_num_hyperedges",
    [
        # When node 1 is selected, we get hyperedge 0 with nodes 0 and 1 -> 2 incidences, 1 hyperedge
        pytest.param(SamplingStrategy.NODE, 1, (2, 1), 1, id="node_strategy"),
        # When hyperedge 0 is selected, we get nodes 0 and 1 -> 2 incidences, 1 hyperedge
        pytest.param(SamplingStrategy.HYPEREDGE, 0, (2, 1), 1, id="hyperedge_strategy"),
    ],
)
def test_getitem_single_index(
    mock_sample_hypergraph, strategy, index, expected_shape, expected_num_hyperedges
):
    with patch.object(HIFConverter, "load_from_hif", return_value=mock_sample_hypergraph):
        dataset = AlgebraDataset(sampling_strategy=strategy)

    data = dataset[index]

    assert data.hyperedge_index.shape == expected_shape
    assert data.num_hyperedges == expected_num_hyperedges


@pytest.mark.parametrize(
    "strategy, index, expected_shape, expected_num_hyperedges",
    [
        # When nodes (0, 2, 3) -> hyperedge 0 (nodes 0, 1) + hyperedge 1 (nodes 2, 3) -> 4 incidences, 2 hyperedges
        pytest.param(SamplingStrategy.NODE, [0, 2, 3], (2, 4), 2, id="node_strategy"),
        # When hyperedge 0 (nodes 0, 1) + hyperedge 1 (nodes 2, 3) -> 4 incidences, 2 hyperedges
        pytest.param(SamplingStrategy.HYPEREDGE, [0, 1], (2, 4), 2, id="hyperedge_strategy"),
    ],
)
def test_getitem_when_list_index_provided(
    mock_four_node_hypergraph, strategy, index, expected_shape, expected_num_hyperedges
):
    with patch.object(HIFConverter, "load_from_hif", return_value=mock_four_node_hypergraph):
        dataset = AlgebraDataset(sampling_strategy=strategy)

    data = dataset[index]

    assert data.hyperedge_index.shape == expected_shape
    assert data.num_hyperedges == expected_num_hyperedges


@pytest.mark.parametrize(
    "strategy",
    [
        pytest.param(SamplingStrategy.NODE, id="node_strategy"),
        pytest.param(SamplingStrategy.HYPEREDGE, id="hyperedge_strategy"),
    ],
)
def test_getitem_with_edge_attr(mock_three_node_weighted_hypergraph, strategy):
    with patch.object(
        HIFConverter, "load_from_hif", return_value=mock_three_node_weighted_hypergraph
    ):
        dataset = AlgebraDataset(sampling_strategy=strategy)

    data = dataset[0]

    assert data.hyperedge_index.shape == (2, 2)
    assert data.num_hyperedges == 1
    assert data.hyperedge_attr is None


@pytest.mark.parametrize(
    "strategy",
    [
        pytest.param(SamplingStrategy.NODE, id="node_strategy"),
        pytest.param(SamplingStrategy.HYPEREDGE, id="hyperedge_strategy"),
    ],
)
def test_getitem_without_edge_attr(mock_no_edge_attr_hypergraph, strategy):
    with patch.object(HIFConverter, "load_from_hif", return_value=mock_no_edge_attr_hypergraph):
        dataset = AlgebraDataset(sampling_strategy=strategy)

    data = dataset[0]
    assert data.hyperedge_attr is None


@pytest.mark.parametrize(
    "strategy, index",
    [
        # When nodes 0,2 -> hyperedge 0 (nodes 0, 1) + hyperedge 1 (node 2) -> 2 hyperedges
        pytest.param(SamplingStrategy.NODE, [0, 2], id="node_strategy"),
        # When hyperedge 0 (nodes 0, 1) + hyperedge 1 (node 2) -> 2 hyperedges
        pytest.param(SamplingStrategy.HYPEREDGE, [0, 1], id="hyperedge_strategy"),
    ],
)
def test_getitem_with_multiple_edges_attr(mock_multiple_edges_attr_hypergraph, strategy, index):
    with patch.object(
        HIFConverter, "load_from_hif", return_value=mock_multiple_edges_attr_hypergraph
    ):
        dataset = AlgebraDataset(sampling_strategy=strategy)

    data = dataset[index]
    assert data.num_hyperedges == 2

    # Even though the original hypergraph has edge attributes, __getitem__ should return hyperedge_attr as None
    # as the hyperedge attributes are handled by the loader's collate function during batching
    assert data.hyperedge_attr is None


def test_getitem_hyperedge_attr_are_padded_with_zero_when_no_uniform_edges():
    mock_hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[
            {"node": "0", "attrs": {}},
            {"node": "1", "attrs": {}},
            {"node": "2", "attrs": {}},
            {"node": "3", "attrs": {}},
        ],
        hyperedges=[
            {"edge": "0", "attrs": {"weight": 1.0, "abc": 5.0}},
            {"edge": "1", "attrs": {"weight": 2.0}},  # Missing 'abc'
            {"edge": "2", "attrs": {"abc": 3.0}},  # Missing 'weight'
        ],
        incidences=[
            {"node": "0", "edge": "0"},
            {"node": "1", "edge": "0"},
            {"node": "2", "edge": "1"},
            {"node": "3", "edge": "2"},
        ],
    )

    with patch.object(HIFConverter, "load_from_hif", return_value=mock_hypergraph):
        dataset = AlgebraDataset()

    assert dataset.hdata.hyperedge_attr is not None
    assert dataset.hdata.hyperedge_attr.shape == (
        3,
        2,
    )  # 3 edges, 2 features each (weight, abc in insertion order)
    assert torch.allclose(
        dataset.hdata.hyperedge_attr[0], torch.tensor([1.0, 5.0])
    )  # weight=1.0, abc=5.0
    assert torch.allclose(
        dataset.hdata.hyperedge_attr[1], torch.tensor([2.0, 0.0])
    )  # weight=2.0, abc=0.0
    assert torch.allclose(
        dataset.hdata.hyperedge_attr[2], torch.tensor([0.0, 3.0])
    )  # weight=0.0, abc=3.0


def test_process_not_all_hyperedge_weights_():
    mock_hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[
            {"node": "0", "attrs": {}},
            {"node": "1", "attrs": {}},
            {"node": "2", "attrs": {}},
        ],
        hyperedges=[
            {"edge": "0", "weight": 1.5},
            {"edge": "1"},
            {"edge": "2", "weight": 2.5},
        ],
        incidences=[
            {"node": "0", "edge": "0"},
            {"node": "1", "edge": "1"},
            {"node": "2", "edge": "2"},
        ],
    )

    with patch.object(HIFConverter, "load_from_hif", return_value=mock_hypergraph):
        with pytest.raises(
            ValueError,
            match="Some hyperedges have weights while others do not. All hyperedges must either have weights or none.",
        ):
            dataset = AlgebraDataset()


def test_process_extracts_top_level_hyperedge_weights():
    mock_hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[
            {"node": "0", "attrs": {}},
            {"node": "1", "attrs": {}},
            {"node": "2", "attrs": {}},
        ],
        hyperedges=[
            {"edge": "0", "weight": 1.5},
            {"edge": "1", "weight": 3.0},
            {"edge": "2", "weight": 2.5},
        ],
        incidences=[
            {"node": "0", "edge": "0"},
            {"node": "1", "edge": "1"},
            {"node": "2", "edge": "2"},
        ],
    )

    with patch.object(HIFConverter, "load_from_hif", return_value=mock_hypergraph):
        dataset = AlgebraDataset()

    hyperedge_weights = dataset.hdata.hyperedge_weights
    assert hyperedge_weights is not None
    assert torch.allclose(hyperedge_weights[0], torch.tensor([1.5]))
    assert torch.allclose(hyperedge_weights[1], torch.tensor([3.0]))
    assert torch.allclose(hyperedge_weights[2], torch.tensor([2.5]))


def test_transform_attrs_empty_attrs():
    mock_hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[{"node": "0", "attrs": {}}],
        hyperedges=[{"edge": "0", "attrs": {}}],
        incidences=[{"node": "0", "edge": "0"}],
    )

    with patch.object(HIFConverter, "load_from_hif", return_value=mock_hypergraph):

        class TestDataset(Dataset):
            DATASET_NAME = "TEST"

        dataset = TestDataset()

        result = dataset.transform_attrs({})
        assert len(result) == 0

        attrs = {"name": "node1", "active": True}
        result = dataset.transform_attrs(attrs)
        assert len(result) == 0


def test_process_adds_padding_zero_when_inconsistent_node_attributes():
    mock_hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[
            {"node": "0", "attrs": {"weight": 1.0}},  # Missing 'score'
            {"node": "1", "attrs": {"weight": 2.0, "score": 0.8}},
            {"node": "2", "attrs": {"score": 0.5}},  # Missing 'weight'
        ],
        hyperedges=[{"edge": "0", "attrs": {}}],
        incidences=[
            {"node": "0", "edge": "0"},
            {"node": "1", "edge": "0"},
            {"node": "2", "edge": "0"},
        ],
    )

    with patch.object(HIFConverter, "load_from_hif", return_value=mock_hypergraph):

        class TestDataset(Dataset):
            DATASET_NAME = "TEST"

        dataset = TestDataset()

        assert dataset.hdata.x.shape == (
            3,
            2,
        )  # 3 nodes, 2 features each (weight, score in insertion order)
        assert torch.allclose(dataset.hdata.x[0], torch.tensor([1.0, 0.0]))  # weight=1.0, score=0.0
        assert torch.allclose(dataset.hdata.x[1], torch.tensor([2.0, 0.8]))  # weight=2.0, score=0.8
        assert torch.allclose(dataset.hdata.x[2], torch.tensor([0.0, 0.5]))  # weight=0.0, score=0.5


def test_process_with_no_node_attributes_fallback_to_one():
    mock_hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[
            {"node": "0", "attrs": {"name": "node0"}},
            {"node": "1", "attrs": {}},
        ],
        hyperedges=[{"edge": "0", "attrs": {}}],
        incidences=[{"node": "0", "edge": "0"}, {"node": "1", "edge": "0"}],
    )

    with patch.object(HIFConverter, "load_from_hif", return_value=mock_hypergraph):

        class TestDataset(Dataset):
            DATASET_NAME = "TEST"

        dataset = TestDataset()

        assert dataset.hdata.x.shape == (2, 1)
        assert torch.allclose(dataset.hdata.x, torch.tensor([[1.0], [1.0]]))


def test_process_with_single_node_attribute():
    mock_hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[
            {"node": "0", "attrs": {"weight": 1.5}},
            {"node": "1", "attrs": {"weight": 2.5}},
            {"node": "2", "attrs": {"weight": 3.5}},
        ],
        hyperedges=[{"edge": "0", "attrs": {}}],
        incidences=[
            {"node": "0", "edge": "0"},
            {"node": "1", "edge": "0"},
            {"node": "2", "edge": "0"},
        ],
    )

    with patch.object(HIFConverter, "load_from_hif", return_value=mock_hypergraph):

        class TestDataset(Dataset):
            DATASET_NAME = "TEST"

        dataset = TestDataset()

        # Single attribute should remain 2D: [num_nodes, 1]
        assert dataset.hdata.x.shape == (3, 1)
        assert torch.allclose(dataset.hdata.x, torch.tensor([[1.5], [2.5], [3.5]]))


def test_transform_attrs_adds_padding_zero_when_attr_keys_padding():
    mock_hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[{"node": "0", "attrs": {}}],
        hyperedges=[{"edge": "0", "attrs": {}}],
        incidences=[{"node": "0", "edge": "0"}],
    )

    with patch.object(HIFConverter, "load_from_hif", return_value=mock_hypergraph):

        class TestDataset(Dataset):
            DATASET_NAME = "TEST"

        dataset = TestDataset()

        # Test with attr_keys - should pad missing attributes with 0.0
        attrs = {"weight": 1.5}
        result = dataset.transform_attrs(attrs, attr_keys=["score", "weight", "age"])
        assert torch.allclose(
            result, torch.tensor([0.0, 1.5, 0.0])
        )  # score=0.0, weight=1.5, age=0.0

        # Test with all attributes present
        attrs = {"weight": 1.5, "score": 0.8, "age": 25.0}
        result = dataset.transform_attrs(attrs, attr_keys=["age", "score", "weight"])
        assert torch.allclose(
            result, torch.tensor([25.0, 0.8, 1.5])
        )  # age=25.0, score=0.8, weight=1.5

        # Test without attr_keys - maintains insertion order
        attrs = {"weight": 1.5, "score": 0.8}
        result = dataset.transform_attrs(attrs)
        assert torch.allclose(result, torch.tensor([1.5, 0.8]))  # weight, score (insertion order)


@pytest.mark.parametrize(
    "strategy, expected_len",
    [
        # mock_hdata: 3 nodes, 2 hyperedges
        pytest.param(SamplingStrategy.NODE, 3, id="node_strategy"),
        pytest.param(SamplingStrategy.HYPEREDGE, 2, id="hyperedge_strategy"),
    ],
)
def test_from_hdata(strategy, expected_len, mock_hdata):
    dataset = Dataset.from_hdata(mock_hdata, sampling_strategy=strategy)

    assert dataset.hdata is mock_hdata
    assert len(dataset) == expected_len


def test_from_hdata_download_raises(mock_hdata):
    dataset = Dataset.from_hdata(mock_hdata)

    with pytest.raises(ValueError, match="download can only be called for the original dataset."):
        dataset.download()


def test_from_hdata_process_raises(mock_hdata):
    dataset = Dataset.from_hdata(mock_hdata)

    with pytest.raises(ValueError, match="process can only be called for the original dataset."):
        dataset.process()


def test_enrich_node_features_replace(mock_hdata):
    dataset = Dataset.from_hdata(mock_hdata)

    enricher = MagicMock(spec=NodeEnricher)
    enriched_x = torch.randn(3, 4)
    enricher.enrich.return_value = enriched_x

    dataset.enrich_node_features(enricher)

    enricher.enrich.assert_called_once_with(mock_hdata.hyperedge_index)
    assert torch.equal(dataset.hdata.x, enriched_x)


def test_enrich_node_features_concatenate(mock_hdata):
    dataset = Dataset.from_hdata(mock_hdata)
    original_x = dataset.hdata.x.clone()

    enricher = MagicMock(spec=NodeEnricher)
    enriched_x = torch.randn(3, 4)
    enricher.enrich.return_value = enriched_x

    dataset.enrich_node_features(enricher, enrichment_mode="concatenate")

    enricher.enrich.assert_called_once_with(mock_hdata.hyperedge_index)
    expected_x = torch.cat([original_x, enriched_x], dim=1)
    assert torch.equal(dataset.hdata.x, expected_x)
    assert dataset.hdata.x.shape == (3, 5)  # 1 original + 4 enriched


def test_enrich_node_features_from_dataset():
    source_dataset = Dataset.from_hdata(
        HData(
            x=torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]),
            hyperedge_index=torch.tensor([[0, 1, 2], [0, 0, 1]]),
            global_node_ids=torch.tensor([100, 200, 300]),
        )
    )
    target_dataset = Dataset.from_hdata(
        HData(
            x=torch.tensor([[0.0], [0.0]]),
            hyperedge_index=torch.tensor([[0, 1], [0, 0]]),
            global_node_ids=torch.tensor([300, 100]),
        )
    )

    target_dataset.enrich_node_features_from(source_dataset)

    assert torch.equal(target_dataset.hdata.x, torch.tensor([[3.0, 30.0], [1.0, 10.0]]))


def test_enrich_node_features_from_propagates_hdata_validation_errors():
    source_dataset = Dataset.from_hdata(
        HData(
            x=torch.tensor([[1.0], [2.0]]),
            hyperedge_index=torch.tensor([[0, 1], [0, 0]]),
            global_node_ids=torch.tensor([10, 20]),
        )
    )
    target_dataset = Dataset.from_hdata(
        HData(
            x=torch.tensor([[0.0]]),
            hyperedge_index=torch.tensor([[0], [0]]),
            global_node_ids=torch.tensor([10]),
        )
    )
    target_dataset.hdata.global_node_ids = None

    with pytest.raises(
        ValueError,
        match="Both HData instances must define global_node_ids to align node features.",
    ):
        target_dataset.enrich_node_features_from(source_dataset)


def test_enrich_node_features_from_dataset_with_fill_value():
    source_dataset = Dataset.from_hdata(
        HData(
            x=torch.tensor([[1.0, 10.0], [2.0, 20.0]]),
            hyperedge_index=torch.tensor([[0, 1], [0, 0]]),
            global_node_ids=torch.tensor([10, 20]),
        )
    )
    target_dataset = Dataset.from_hdata(
        HData(
            x=torch.tensor([[0.0], [0.0]]),
            hyperedge_index=torch.tensor([[0, 1], [0, 0]]),
            global_node_ids=torch.tensor([10, 30]),
        )
    )

    target_dataset.enrich_node_features_from(
        source_dataset,
        node_space_setting="inductive",
        fill_value=[7.0, 8.0],
    )

    assert torch.equal(target_dataset.hdata.x, torch.tensor([[1.0, 10.0], [7.0, 8.0]]))


def test_enrich_hyperedge_attr_replace(mock_hdata):
    dataset = Dataset.from_hdata(mock_hdata)

    enricher = MagicMock(spec=HyperedgeEnricher)
    enriched_x = torch.randn(3, 4)
    enricher.enrich.return_value = enriched_x

    dataset.enrich_hyperedge_attr(enricher)

    enricher.enrich.assert_called_once_with(mock_hdata.hyperedge_index)
    hyperedge_attr = dataset.hdata.hyperedge_attr
    assert hyperedge_attr is not None
    assert torch.equal(hyperedge_attr, enriched_x)


def test_enrich_hyperedge_attr_concatenate(mock_hdata_with_hyperedge_attr):
    dataset = Dataset.from_hdata(mock_hdata_with_hyperedge_attr)
    original_hyperedge_attr = dataset.hdata.hyperedge_attr
    assert original_hyperedge_attr is not None
    original_hyperedge_attr = original_hyperedge_attr.clone()

    enricher = MagicMock(spec=HyperedgeEnricher)
    enriched_x = torch.randn(3, 4)
    enricher.enrich.return_value = enriched_x

    dataset.enrich_hyperedge_attr(enricher, enrichment_mode="concatenate")

    enricher.enrich.assert_called_once_with(mock_hdata_with_hyperedge_attr.hyperedge_index)
    expected_x = torch.cat([original_hyperedge_attr, enriched_x], dim=1)
    hyperedge_attr = dataset.hdata.hyperedge_attr
    assert hyperedge_attr is not None
    assert torch.equal(hyperedge_attr, expected_x)
    assert hyperedge_attr.shape == (3, 5)  # 1 original + 4 enriched


def test_enrich_hyperedge_weights_replace(mock_hdata):
    dataset = Dataset.from_hdata(mock_hdata)

    enricher = MagicMock(spec=HyperedgeEnricher)
    enriched_weights = torch.randn(3)
    enricher.enrich.return_value = enriched_weights

    dataset.enrich_hyperedge_weights(enricher)

    enricher.enrich.assert_called_once_with(mock_hdata.hyperedge_index)
    hyperedge_weights = dataset.hdata.hyperedge_weights
    assert hyperedge_weights is not None
    assert torch.equal(hyperedge_weights, enriched_weights)


def test_enrich_hyperedge_weights_concatenate(mock_hdata_with_hyperedge_weights):
    dataset = Dataset.from_hdata(mock_hdata_with_hyperedge_weights)
    original_weights = dataset.hdata.hyperedge_weights
    assert original_weights is not None
    original_weights = original_weights.clone()

    enricher = MagicMock(spec=HyperedgeEnricher)
    enriched_weights = torch.randn(3)
    enricher.enrich.return_value = enriched_weights

    dataset.enrich_hyperedge_weights(enricher, enrichment_mode="concatenate")

    enricher.enrich.assert_called_once_with(mock_hdata_with_hyperedge_weights.hyperedge_index)
    expected_weights = torch.cat([original_weights, enriched_weights], dim=0)
    hyperedge_weights = dataset.hdata.hyperedge_weights
    assert hyperedge_weights is not None
    assert torch.equal(hyperedge_weights, expected_weights)
    assert hyperedge_weights.shape == (6,)  # 3 original + 3 enriched


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
    num_nodes = hyperedge_index[0].max().item() + 1 if hyperedge_index.shape[1] > 0 else 0
    x = torch.ones((num_nodes, 1), dtype=torch.float)
    hdata = HData(x=x, hyperedge_index=hyperedge_index)
    dataset = Dataset.from_hdata(hdata)

    dataset.remove_hyperedges_with_fewer_than_k_nodes(k)

    expected_num_nodes = expected_hyperedge_index[0].unique().shape[0]
    expected_num_hyperedges = expected_hyperedge_index[1].unique().shape[0]

    assert torch.equal(dataset.hdata.hyperedge_index, expected_hyperedge_index)
    assert dataset.hdata.x.shape[0] == expected_num_nodes
    assert dataset.hdata.y.shape[0] == expected_num_hyperedges


def test_split_with_equal_ratios(mock_four_node_hypergraph):
    with patch.object(HIFConverter, "load_from_hif", return_value=mock_four_node_hypergraph):
        dataset = AlgebraDataset()

    splits = dataset.split([0.5, 0.5])

    assert len(splits) == 2
    assert (
        splits[0].hdata.num_hyperedges + splits[1].hdata.num_hyperedges
        == dataset.hdata.num_hyperedges
    )
    for split in splits:
        assert split.hdata.x is not None
        assert split.hdata.num_nodes > 0
        assert split.hdata.num_hyperedges > 0


def test_split_three_way(mock_multiple_edges_attr_hypergraph):
    with patch.object(
        HIFConverter, "load_from_hif", return_value=mock_multiple_edges_attr_hypergraph
    ):
        dataset = AlgebraDataset()

    splits = dataset.split([0.5, 0.25, 0.25])
    total_edges = sum(split.hdata.num_hyperedges for split in splits)

    assert len(splits) == 3
    assert total_edges == dataset.hdata.num_hyperedges

    for split in splits:
        assert split.hdata.x is not None
        assert split.hdata.num_nodes > 0
        assert split.hdata.num_hyperedges > 0


def test_split_raises_when_ratios_do_not_sum_to_one(mock_four_node_hypergraph):
    with patch.object(HIFConverter, "load_from_hif", return_value=mock_four_node_hypergraph):
        dataset = AlgebraDataset()

    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        dataset.split([0.8, 0.1, 0.05])


def test_split_with_shuffle_produces_deterministic_results_when_seed_provided(
    mock_four_node_hypergraph,
):
    with patch.object(HIFConverter, "load_from_hif", return_value=mock_four_node_hypergraph):
        dataset = AlgebraDataset()

    splits_a = dataset.split([0.5, 0.5], shuffle=True, seed=42)
    splits_b = dataset.split([0.5, 0.5], shuffle=True, seed=42)

    assert torch.equal(splits_a[0].hdata.hyperedge_index, splits_b[0].hdata.hyperedge_index)
    assert torch.equal(splits_a[1].hdata.hyperedge_index, splits_b[1].hdata.hyperedge_index)


def test_split_with_shuffle_when_no_seed_provided(
    mock_four_node_hypergraph,
):
    with patch.object(HIFConverter, "load_from_hif", return_value=mock_four_node_hypergraph):
        dataset = AlgebraDataset()

    splits = dataset.split([0.5, 0.5], shuffle=True)
    total_edges = sum(split.hdata.num_hyperedges for split in splits)

    assert len(splits) == 2
    assert total_edges == dataset.hdata.num_hyperedges

    for split in splits:
        assert split.hdata.x is not None
        assert split.hdata.num_nodes > 0
        assert split.hdata.num_hyperedges > 0


def test_split_preserves_edge_attr(mock_multiple_edges_attr_hypergraph):
    with patch.object(
        HIFConverter, "load_from_hif", return_value=mock_multiple_edges_attr_hypergraph
    ):
        dataset = AlgebraDataset()

    splits = dataset.split([0.5, 0.5])

    for split in splits:
        assert split.hdata.hyperedge_attr is not None
        assert split.hdata.hyperedge_attr.shape[0] == split.hdata.num_hyperedges


def test_split_without_edge_attr(mock_no_edge_attr_hypergraph):
    with patch.object(HIFConverter, "load_from_hif", return_value=mock_no_edge_attr_hypergraph):
        dataset = AlgebraDataset()

    splits = dataset.split([0.5, 0.5])

    for split in splits:
        assert split.hdata.hyperedge_attr is None


def test_split_transductive_default_preserves_first_split_node_space():
    hdata = HData(
        x=torch.arange(4, dtype=torch.float).unsqueeze(1),
        hyperedge_index=torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
        global_node_ids=torch.tensor([100, 200, 300, 400]),
    )
    dataset = Dataset.from_hdata(hdata)

    train_dataset, test_dataset = dataset.split([0.75, 0.25])

    assert train_dataset.hdata.num_nodes == dataset.hdata.num_nodes
    assert torch.equal(train_dataset.hdata.x, dataset.hdata.x)
    assert test_dataset.hdata.num_nodes == 1


def test_split_transductive_all_preserves_all_split_node_spaces():
    hdata = HData(
        x=torch.arange(4, dtype=torch.float).unsqueeze(1),
        hyperedge_index=torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
        global_node_ids=torch.tensor([100, 200, 300, 400]),
    )
    dataset = Dataset.from_hdata(hdata)

    train_dataset, test_dataset = dataset.split(
        [0.75, 0.25],
        node_space_setting="transductive",
        assign_node_space_to="all",
    )

    assert train_dataset.hdata.num_nodes == dataset.hdata.num_nodes
    assert test_dataset.hdata.num_nodes == dataset.hdata.num_nodes
    assert torch.equal(train_dataset.hdata.x, dataset.hdata.x)
    assert torch.equal(test_dataset.hdata.x, dataset.hdata.x)


def test_split_raises_when_node_space_provided_with_transductive_disabled():
    hdata = HData(
        x=torch.arange(4, dtype=torch.float).unsqueeze(1),
        hyperedge_index=torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
        global_node_ids=torch.tensor([100, 200, 300, 400]),
    )
    dataset = Dataset.from_hdata(hdata)

    with pytest.raises(
        ValueError,
        match="assign_node_space_to can only be provided when node_space_setting='transductive'.",
    ):
        dataset.split(
            [0.75, 0.25],
            node_space_setting="inductive",
            assign_node_space_to="first",
        )


def test_nested_transductive_split_supports_train_feature_reuse():
    hdata = HData(
        x=torch.arange(4, dtype=torch.float).unsqueeze(1),
        hyperedge_index=torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
        global_node_ids=torch.tensor([100, 200, 300, 400]),
    )
    dataset = Dataset.from_hdata(hdata)

    train_dataset, test_dataset = dataset.split(
        [0.75, 0.25],
        node_space_setting="transductive",
    )
    train_dataset, val_dataset = train_dataset.split(
        [2 / 3, 1 / 3],
        node_space_setting="transductive",
    )

    enricher = MagicMock(spec=NodeEnricher)
    enricher.enrich.return_value = torch.tensor(
        [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0]]
    )
    train_dataset.enrich_node_features(enricher, enrichment_mode="replace")
    val_dataset.enrich_node_features_from(train_dataset)
    test_dataset.enrich_node_features_from(train_dataset)

    assert torch.equal(val_dataset.hdata.x, torch.tensor([[30.0, 31.0]]))
    assert torch.equal(test_dataset.hdata.x, torch.tensor([[40.0, 41.0]]))


def test_to_device(mock_hdata):
    device = torch.device("cpu")

    dataset = Dataset.from_hdata(mock_hdata)

    result = dataset.to(device)

    assert result is dataset
    assert dataset.hdata.device == device


def test_load_from_hif_skips_download_when_file_exists():
    dataset_name = "ALGEBRA"

    sample_hif = {
        "network-type": "undirected",
        "nodes": [{"node": "0"}, {"node": "1"}],
        "edges": [{"edge": "0"}],
        "incidences": [{"node": "0", "edge": "0"}],
    }

    mock_hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[{"node": "0"}, {"node": "1"}],
        hyperedges=[{"edge": "0"}],
        incidences=[{"node": "0", "edge": "0"}],
    )

    with (
        patch("hyperbench.data.dataset.requests.get") as mock_get,
        patch("hyperbench.data.dataset.os.path.exists", return_value=True),
        patch("builtins.open", mock_open()) as mock_file,
        patch("hyperbench.data.dataset.zstd.ZstdDecompressor") as mock_decomp,
        patch("hyperbench.data.dataset.tempfile.NamedTemporaryFile") as mock_temp,
        patch("hyperbench.data.dataset.json.load", return_value=sample_hif),
        patch("hyperbench.data.dataset.validate_hif_json", return_value=True),
        patch.object(HIFHypergraph, "from_hif", return_value=mock_hypergraph),
    ):
        mock_dctx = mock_decomp.return_value
        mock_dctx.copy_stream = lambda input_f, tmp_file: None

        mock_temp_instance = mock_temp.return_value.__enter__.return_value
        mock_temp_instance.name = "/tmp/decompressed.json"

        result = HIFConverter.load_from_hif(dataset_name, save_on_disk=True)
        mock_get.assert_not_called()
        assert result == mock_hypergraph


def test_default_sampling_strategy_is_hyperedge(mock_four_node_hypergraph):
    with patch.object(HIFConverter, "load_from_hif", return_value=mock_four_node_hypergraph):
        dataset = AlgebraDataset()

    # Default strategy is HYPEREDGE, so len should be num_hyperedges (2), not num_nodes (4)
    assert dataset.sampling_strategy == SamplingStrategy.HYPEREDGE
    assert len(dataset) == 2


def test_explicit_node_sampling_strategy(mock_four_node_hypergraph):
    with patch.object(HIFConverter, "load_from_hif", return_value=mock_four_node_hypergraph):
        dataset = AlgebraDataset(sampling_strategy=SamplingStrategy.NODE)

    # NODE strategy, so len should be num_nodes (4), not num_hyperedges (2)
    assert dataset.sampling_strategy == SamplingStrategy.NODE
    assert len(dataset) == 4


@pytest.mark.parametrize(
    "strategy",
    [
        pytest.param(SamplingStrategy.NODE, id="node_strategy"),
        pytest.param(SamplingStrategy.HYPEREDGE, id="hyperedge_strategy"),
    ],
)
def test_split_preserves_sampling_strategy(mock_four_node_hypergraph, strategy):
    with patch.object(HIFConverter, "load_from_hif", return_value=mock_four_node_hypergraph):
        dataset = AlgebraDataset(sampling_strategy=strategy)

    splits = dataset.split([0.5, 0.5])

    for split in splits:
        assert split.sampling_strategy == strategy


def test_from_hdata_with_explicit_strategy(mock_hdata):
    dataset = Dataset.from_hdata(mock_hdata, sampling_strategy=SamplingStrategy.NODE)

    assert dataset.sampling_strategy == SamplingStrategy.NODE
    assert len(dataset) == 3  # mock_hdata has 3 nodes


def test_update_from_hdata_returns_new_dataset(mock_hdata):
    dataset = Dataset(hdata=mock_hdata, prepare=False)
    new_x = torch.ones((2, 1), dtype=torch.float)
    new_hyperedge_index = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)
    new_hdata = HData(x=new_x, hyperedge_index=new_hyperedge_index)

    result = dataset.update_from_hdata(new_hdata)

    assert result is not dataset
    assert result.hdata is new_hdata
    assert dataset.hdata is mock_hdata


def test_update_from_hdata_stores_provided_hdata(mock_hdata):
    dataset = Dataset(hdata=mock_hdata, prepare=False)
    new_x = torch.ones((2, 1), dtype=torch.float)
    new_hyperedge_index = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)
    new_hdata = HData(x=new_x, hyperedge_index=new_hyperedge_index)

    result = dataset.update_from_hdata(new_hdata)

    assert result.hdata is new_hdata


@pytest.mark.parametrize(
    "strategy, expected_len",
    [
        pytest.param(SamplingStrategy.NODE, 4, id="node_strategy"),
        pytest.param(SamplingStrategy.HYPEREDGE, 3, id="hyperedge_strategy"),
    ],
)
def test_update_from_hdata_inherits_sampling_strategy(mock_hdata, strategy, expected_len):
    dataset = Dataset(hdata=mock_hdata, sampling_strategy=strategy, prepare=False)
    new_x = torch.ones((4, 1), dtype=torch.float)
    new_hyperedge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 2]], dtype=torch.long)
    new_hdata = HData(x=new_x, hyperedge_index=new_hyperedge_index)

    result = dataset.update_from_hdata(new_hdata)

    assert result.sampling_strategy == strategy
    assert len(result) == expected_len


def test_update_from_hdata_preserves_subclass_type(mock_hdata):
    dataset = AlgebraDataset(hdata=mock_hdata, prepare=False)
    new_x = torch.ones((2, 1), dtype=torch.float)
    new_hyperedge_index = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)
    new_hdata = HData(x=new_x, hyperedge_index=new_hyperedge_index)

    result = dataset.update_from_hdata(new_hdata)

    assert type(result) is AlgebraDataset


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


def test_dataset_stats_computation(mock_hdata_stats):
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

    dataset = Dataset.from_hdata(mock_hdata_stats)

    stats = dataset.stats()
    assert stats == expected_stats
