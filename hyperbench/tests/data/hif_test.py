import json
import re
import pytest
import requests
import torch
import os

from unittest.mock import patch
from hyperbench.data import HIFLoader, HIFProcessor
from hyperbench.types import HData, HIFHypergraph


@pytest.fixture(autouse=True)
def mock_cache_root(tmp_path):
    with patch("hyperbench.data.hif.get_cache_dir", return_value=str(tmp_path)):
        yield


@pytest.fixture
def mock_sample_hypergraph() -> HIFHypergraph:
    return HIFHypergraph(
        network_type="undirected",
        nodes=[{"node": "0"}, {"node": "1"}],
        hyperedges=[{"edge": "0"}],
        incidences=[{"node": "0", "edge": "0"}],
    )


@pytest.fixture
def mock_simple_hypergraph() -> HIFHypergraph:
    return HIFHypergraph(
        network_type="undirected",
        nodes=[{"node": "0", "attrs": {}}, {"node": "1", "attrs": {}}],
        hyperedges=[{"edge": "0", "attrs": {}}],
        incidences=[{"node": "0", "edge": "0"}],
    )


@pytest.fixture
def mock_three_node_weighted_hypergraph() -> HIFHypergraph:
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
def mock_four_node_hypergraph() -> HIFHypergraph:
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
def mock_five_node_hypergraph() -> HIFHypergraph:
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
def mock_no_edge_attr_hypergraph() -> HIFHypergraph:
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
def mock_multiple_edges_attr_hypergraph() -> HIFHypergraph:
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


@pytest.fixture
def mock_hypergraph() -> HIFHypergraph:
    return HIFHypergraph(
        network_type="undirected",
        nodes=[{"node": "0", "attrs": {}}, {"node": "1", "attrs": {}}],
        hyperedges=[{"edge": "0", "attrs": {"weight": 1.0}}],
        incidences=[{"node": "0", "edge": "0"}, {"node": "1", "edge": "0"}],
    )


@pytest.fixture
def mock_hdata() -> HData:
    x = torch.ones((2, 1), dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)
    return HData(x=x, hyperedge_index=hyperedge_index)


def __write_hif_json(tmp_path, hypergraph: HIFHypergraph, filename: str = "sample.json") -> str:
    path = tmp_path / filename
    payload = __hif_payload(hypergraph)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return str(path)


def __hif_payload(hypergraph: HIFHypergraph) -> dict:
    return {
        "network-type": hypergraph.network_type,
        "metadata": hypergraph.metadata,
        "nodes": hypergraph.nodes,
        "edges": hypergraph.hyperedges,
        "incidences": hypergraph.incidences,
    }


def test_transform_attrs_empty_attrs():
    result = HIFProcessor.transform_attrs({})
    assert len(result) == 0

    attrs = {"name": "node1", "active": True}
    result = HIFProcessor.transform_attrs(attrs)
    assert len(result) == 0


def test_transform_attrs_adds_padding_zero_when_attr_keys_padding():
    attrs = {"weight": 1.5}
    result = HIFProcessor.transform_attrs(attrs, attr_keys=["score", "weight", "age"])
    assert torch.allclose(result, torch.tensor([0.0, 1.5, 0.0], dtype=torch.float))

    attrs = {"weight": 1.5, "score": 0.8, "age": 25.0}
    result = HIFProcessor.transform_attrs(attrs, attr_keys=["age", "score", "weight"])
    assert torch.allclose(result, torch.tensor([25.0, 0.8, 1.5], dtype=torch.float))

    attrs = {"weight": 1.5, "score": 0.8}
    result = HIFProcessor.transform_attrs(attrs)
    assert torch.allclose(result, torch.tensor([1.5, 0.8], dtype=torch.float))


def test_transform_hyperedge_attrs_adds_padding_zero_when_attr_keys_padding():
    attrs = {"weight": 2.5}
    result = HIFProcessor.transform_attrs(attrs, attr_keys=["score", "weight"])
    assert torch.allclose(result, torch.tensor([0.0, 2.5], dtype=torch.float))


def test_transform_node_attrs_adds_padding_zero_when_attr_keys_padding():
    attrs = {"weight": 2.5}
    result = HIFProcessor.transform_attrs(attrs, attr_keys=["score", "weight"])
    assert torch.allclose(result, torch.tensor([0.0, 2.5], dtype=torch.float))


def test_process_hypergraph_rejects_duplicate_node_ids():
    hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[{"node": "0"}, {"node": "0"}],
        hyperedges=[{"edge": "0"}],
        incidences=[{"node": "0", "edge": "0"}],
    )

    with pytest.raises(ValueError, match="HIF node IDs must be unique"):
        HIFProcessor.process_hypergraph(hypergraph)


def test_process_hypergraph_rejects_incidences_with_unknown_node_ids():
    hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[{"node": "0"}],
        hyperedges=[{"edge": "0"}],
        incidences=[{"node": "missing", "edge": "0"}],
    )

    with pytest.raises(ValueError, match="Incidence references unknown node id"):
        HIFProcessor.process_hypergraph(hypergraph)


def test_load_from_url_rejects_invalid_url():
    with pytest.raises(ValueError, match="Invalid URL"):
        HIFLoader.load_from_url("not-a-url")


def test_load_from_url_raises_when_status_is_not_200():
    with patch("hyperbench.data.hif.requests.get") as mock_get:
        mock_response = mock_get.return_value
        mock_response.status_code = 404

        with pytest.raises(ValueError, match="Failed to download dataset from URL"):
            HIFLoader.load_from_url("https://example.com/file.json.zst")


@pytest.mark.parametrize(
    "url, expected_message",
    [
        pytest.param(
            "https://example.com/algebra.json.zst.zst",
            re.escape(
                "Unsupported file format for URL 'https://example.com/algebra.json.zst.zst'"
                ". Expected .json or .json.zst"
            ),
            id="json-zst-zst",
        ),
        pytest.param(
            "https://example.com/algebra.zst.json.zst",
            re.escape(
                "URL 'https://example.com/algebra.zst.json.zst' has an unexpected filename "
                "format. Expected at most one dot in the base filename before the extension (e.g., "
                "dataset.json or dataset.json.zst)."
            ),
            id="zst-json-zst",
        ),
    ],
)
def test_load_from_url_rejects_double_extension_urls(url, expected_message):
    with patch("hyperbench.data.hif.requests.get") as mock_get:
        mock_response = mock_get.return_value
        mock_response.status_code = 200
        mock_response.content = b"{}"

        with pytest.raises(ValueError, match=expected_message):
            HIFLoader.load_from_url(url)

    mock_get.assert_called_once_with(url, timeout=20)


def test_load_from_path_raises_for_missing_file():
    with pytest.raises(ValueError, match="does not exist"):
        HIFLoader.load_from_path("/abc/does-not-exist.json.zst")


def test_load_from_path_raises_for_unsupported_extension(tmp_path):
    invalid = tmp_path / "sample.txt"
    invalid.write_text("{}")

    with pytest.raises(
        ValueError,
        match=re.compile(
            "Unsupported format for file '.*sample\\.txt'. Expected \\.json or \\.json\\.zst"
        ),
    ):
        HIFLoader.load_from_path(str(invalid))


@pytest.mark.parametrize(
    "fixture_name, expected_nodes, expected_hyperedges, expected_incidences, has_hyperedge_weights",
    [
        pytest.param("mock_sample_hypergraph", 2, 2, 2, False, id="sample_with_isolated_node"),
        pytest.param("mock_simple_hypergraph", 2, 2, 2, True, id="simple_with_empty_attrs"),
        pytest.param("mock_three_node_weighted_hypergraph", 3, 2, 3, True, id="weighted"),
        pytest.param("mock_four_node_hypergraph", 4, 2, 4, True, id="four_nodes_two_edges"),
        pytest.param("mock_five_node_hypergraph", 5, 5, 5, True, id="five_nodes_with_self_loops"),
        pytest.param("mock_no_edge_attr_hypergraph", 2, 1, 2, False, id="no_edge_attr"),
        pytest.param("mock_multiple_edges_attr_hypergraph", 4, 3, 4, True, id="multiple_weighted"),
    ],
)
def test_load_from_path_processes_hypergraph_cases(
    tmp_path,
    request,
    fixture_name,
    expected_nodes,
    expected_hyperedges,
    expected_incidences,
    has_hyperedge_weights,
):
    hypergraph = request.getfixturevalue(fixture_name)
    json_path = __write_hif_json(tmp_path, hypergraph, filename=f"{fixture_name}.json")

    with patch("hyperbench.data.hif.validate_hif_data", return_value=True):
        hdata = HIFLoader.load_from_path(json_path)

    assert hdata.num_nodes == expected_nodes
    assert hdata.num_hyperedges == expected_hyperedges
    assert hdata.hyperedge_index.shape[1] == expected_incidences
    assert (hdata.hyperedge_weights is not None) is has_hyperedge_weights


def test_load_from_path_reads_utf8_json(tmp_path):
    hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[{"node": "0", "attrs": {"label": "café ☕"}}],
        hyperedges=[{"edge": "0", "attrs": {"name": "東京"}}],
        incidences=[{"node": "0", "edge": "0"}],
        metadata={"description": "naïve façade"},
    )
    json_path = __write_hif_json(tmp_path, hypergraph, filename="utf8.json")

    with patch("hyperbench.data.hif.validate_hif_data", return_value=True):
        hdata = HIFLoader.load_from_path(json_path)

    assert hdata.num_nodes == 1
    assert hdata.num_hyperedges == 1
    assert hdata.hyperedge_index.shape[1] == 1


def test_load_from_path_zst_uses_decompress(tmp_path, mock_hypergraph):
    zst_path = tmp_path / "sample.json.zst"
    zst_path.write_bytes(b"dummy")
    payload = __hif_payload(mock_hypergraph)

    with (
        patch(
            "hyperbench.data.hif.from_zst_file_to_json", return_value=payload
        ) as mock_from_zst_file_to_json,
        patch("hyperbench.data.hif.validate_hif_data", return_value=True),
    ):
        hdata = HIFLoader.load_from_path(str(zst_path))

    mock_from_zst_file_to_json.assert_called_once_with(str(zst_path))
    assert hdata.num_nodes == 2
    assert hdata.num_hyperedges == 1


def test_load_from_path_raises_for_non_hif_compliant_json(tmp_path, mock_hypergraph):
    json_path = __write_hif_json(tmp_path, mock_hypergraph)

    with (
        patch("hyperbench.data.hif.validate_hif_data", return_value=False),
        pytest.raises(ValueError, match="is not HIF-compliant"),
    ):
        HIFLoader.load_from_path(json_path)


def test_load_from_url_processes_zst_and_saves_to_disk(tmp_path, mock_hypergraph):
    unique_name = f"algebra_{tmp_path.name}.json.zst"
    url = f"https://example.com/{unique_name}"
    payload = __hif_payload(mock_hypergraph)

    with (
        patch("hyperbench.data.hif.requests.get") as mock_get,
        patch(
            "hyperbench.data.hif.from_zst_bytes_to_json", return_value=payload
        ) as mock_from_zst_bytes_to_json,
        patch("hyperbench.data.hif.validate_hif_data", return_value=True),
        patch("hyperbench.data.hif.write_dataset_to_disk_as_zst") as mock_write_to_disk,
    ):
        mock_response = mock_get.return_value
        mock_response.status_code = 200
        mock_response.content = b"mock-zst-content"

        hdata = HIFLoader.load_from_url(url, save_on_disk=True)

    mock_from_zst_bytes_to_json.assert_called_once_with(b"mock-zst-content")
    mock_write_to_disk.assert_called_once_with(
        dataset_name=unique_name,
        content=b"mock-zst-content",
    )
    assert hdata.num_nodes == 2
    assert hdata.num_hyperedges == 1


def test_load_from_url_processes_zst_and_not_saves_to_disk(tmp_path, mock_hypergraph):
    unique_name = f"algebra_{tmp_path.name}.json.zst"
    url = f"https://example.com/{unique_name}"
    payload = __hif_payload(mock_hypergraph)

    with (
        patch("hyperbench.data.hif.requests.get") as mock_get,
        patch(
            "hyperbench.data.hif.from_zst_bytes_to_json", return_value=payload
        ) as mock_from_zst_bytes_to_json,
        patch("hyperbench.data.hif.validate_hif_data", return_value=True),
        patch("hyperbench.data.hif.write_dataset_to_disk_as_zst") as mock_write_to_disk,
    ):
        mock_response = mock_get.return_value
        mock_response.status_code = 200
        mock_response.content = b"mock-zst-content"

        hdata = HIFLoader.load_from_url(url, save_on_disk=False)

    mock_from_zst_bytes_to_json.assert_called_once_with(b"mock-zst-content")
    mock_write_to_disk.assert_not_called()

    assert hdata.num_nodes == 2
    assert hdata.num_hyperedges == 1


def test_load_from_url_processes_json_and_saves_compressed_copy(tmp_path, mock_hypergraph):
    unique_name = f"algebra_{tmp_path.name}.json"
    url = f"https://example.com/{unique_name}"
    payload = __hif_payload(mock_hypergraph)

    with (
        patch("hyperbench.data.hif.requests.get") as mock_get,
        patch(
            "hyperbench.data.hif.from_bytes_to_json", return_value=payload
        ) as mock_read_json_bytes,
        patch(
            "hyperbench.data.hif.compress_json_bytes_as_zst", return_value=b"compressed"
        ) as mock_compress,
        patch("hyperbench.data.hif.validate_hif_data", return_value=True),
        patch("hyperbench.data.hif.write_dataset_to_disk_as_zst") as mock_write_to_disk,
    ):
        mock_response = mock_get.return_value
        mock_response.status_code = 200
        mock_response.content = json.dumps(payload).encode("utf-8")

        hdata = HIFLoader.load_from_url(url, save_on_disk=True)

    mock_read_json_bytes.assert_called_once_with(json.dumps(payload).encode("utf-8"))
    mock_compress.assert_called_once_with(json.dumps(payload).encode("utf-8"))
    mock_write_to_disk.assert_called_once_with(
        dataset_name=unique_name,
        content=b"compressed",
    )
    assert hdata.num_nodes == 2
    assert hdata.num_hyperedges == 1


def test_load_from_url_processes_zst_without_saving_to_disk(tmp_path, mock_hypergraph):
    unique_name = f"algebra_{tmp_path.name}.json.zst"
    url = f"https://example.com/{unique_name}"
    payload = __hif_payload(mock_hypergraph)

    with (
        patch("hyperbench.data.hif.requests.get") as mock_get,
        patch(
            "hyperbench.data.hif.from_zst_bytes_to_json", return_value=payload
        ) as mock_from_zst_bytes_to_json,
        patch("hyperbench.data.hif.validate_hif_data", return_value=True),
        patch("hyperbench.data.hif.write_dataset_to_disk_as_zst") as mock_write_to_disk,
    ):
        mock_response = mock_get.return_value
        mock_response.status_code = 200
        mock_response.content = b"mock-zst-content"

        hdata = HIFLoader.load_from_url(url, save_on_disk=False)

    mock_write_to_disk.assert_not_called()
    mock_from_zst_bytes_to_json.assert_called_once_with(b"mock-zst-content")
    assert hdata.num_nodes == 2
    assert hdata.num_hyperedges == 1


def test_load_from_url_processes_json_without_saving_to_disk(tmp_path, mock_hypergraph):
    unique_name = f"algebra_{tmp_path.name}.json"
    url = f"https://example.com/{unique_name}"
    payload = __hif_payload(mock_hypergraph)

    with (
        patch("hyperbench.data.hif.requests.get") as mock_get,
        patch(
            "hyperbench.data.hif.from_bytes_to_json", return_value=payload
        ) as mock_read_json_bytes,
        patch("hyperbench.data.hif.compress_json_bytes_as_zst") as mock_compress,
        patch("hyperbench.data.hif.validate_hif_data", return_value=True),
        patch("hyperbench.data.hif.write_dataset_to_disk_as_zst") as mock_write_to_disk,
    ):
        mock_response = mock_get.return_value
        mock_response.status_code = 200
        mock_response.content = json.dumps(payload).encode("utf-8")

        hdata = HIFLoader.load_from_url(url, save_on_disk=False)

    mock_read_json_bytes.assert_called_once_with(json.dumps(payload).encode("utf-8"))
    mock_compress.assert_not_called()
    mock_write_to_disk.assert_not_called()
    assert hdata.num_nodes == 2
    assert hdata.num_hyperedges == 1


def test_load_from_path_processes_node_numeric_attrs_into_features(tmp_path):
    hypergraph = HIFHypergraph(
        network_type="undirected",
        nodes=[
            {"node": "0", "attrs": {"weight": 1.0, "score": 0.5}},
            {"node": "1", "attrs": {"weight": 2.0, "score": 1.5}},
        ],
        hyperedges=[{"edge": "0", "attrs": {}}],
        incidences=[{"node": "0", "edge": "0"}, {"node": "1", "edge": "0"}],
    )
    json_path = __write_hif_json(tmp_path, hypergraph, filename="nodes_with_attrs.json")

    with patch("hyperbench.data.hif.validate_hif_data", return_value=True):
        hdata = HIFLoader.load_from_path(json_path)

    assert hdata.x.shape == (2, 2)
    assert torch.allclose(hdata.x[0], torch.tensor([1.0, 0.5], dtype=torch.float))
    assert torch.allclose(hdata.x[1], torch.tensor([2.0, 1.5], dtype=torch.float))


def test_load_from_url_raises_for_unsupported_temp_extension(tmp_path):
    with patch("hyperbench.data.hif.requests.get") as mock_get:
        mock_response = mock_get.return_value
        mock_response.status_code = 200
        mock_response.content = b"bytes"

        with pytest.raises(ValueError, match="Unsupported file format"):
            HIFLoader.load_from_url("https://example.com/algebra.unknown")


def test_load_skips_download_when_file_exists(tmp_path, mock_hypergraph):
    payload = __hif_payload(mock_hypergraph)

    with (
        patch("hyperbench.data.hif.os.path.exists", return_value=True),
        patch("hyperbench.data.hif.requests.get") as mock_get,
        patch(
            "hyperbench.data.hif.from_zst_file_to_json", return_value=payload
        ) as mock_from_zst_file_to_json,
        patch("hyperbench.data.hif.validate_hif_data", return_value=True),
    ):
        result = HIFLoader.load_by_name("algebra", save_on_disk=True)

    mock_get.assert_not_called()
    mock_from_zst_file_to_json.assert_called_once()
    assert result.num_nodes == 2
    assert result.num_hyperedges == 1


def test_hifloader_falls_back_to_hf_hub_download_when_github_raw_download_fails(
    tmp_path, mock_hypergraph
):
    fallback_file = tmp_path / "algebra.json.zst"
    fallback_file.write_bytes(b"mock_zst_content")
    payload = __hif_payload(mock_hypergraph)

    with (
        patch("hyperbench.data.hif.os.path.exists", return_value=False),
        patch("hyperbench.data.hif.requests.get") as mock_get,
        patch("hyperbench.data.hif.hf_hub_download", return_value=str(fallback_file)) as _,
        patch("hyperbench.data.hif.from_zst_file_to_json", return_value=payload),
        patch("hyperbench.data.hif.validate_hif_data", return_value=True),
        pytest.warns(UserWarning, match="GitHub raw download failed"),
        pytest.raises(
            ValueError,
            match=re.escape(
                "Failed to download dataset 'algebra' from GitHub with "
                "status code 404 and no SHA provided for Hugging Face Hub fallback."
            ),
        ),
    ):
        mock_response = mock_get.return_value
        mock_response.status_code = 404
        mock_response.content = b""

        _ = HIFLoader.load_by_name("algebra", save_on_disk=False)


def test_hifloader_from_url_raise_error_on_wrong_extension():
    with patch("hyperbench.data.hif.requests.get") as mock_get:
        mock_response = mock_get.return_value
        mock_response.status_code = 200
        mock_response.content = b"bytes"

        with pytest.raises(
            ValueError,
            match=re.escape("Unsupported file format for URL 'https://example.com/algebra.txt'"),
        ):
            HIFLoader.load_from_url("https://example.com/algebra.txt")


def test_load_saves_downloaded_dataset_on_disk(tmp_path, mock_hypergraph):
    payload = __hif_payload(mock_hypergraph)

    with (
        patch("hyperbench.data.hif.os.path.exists", return_value=False),
        patch("hyperbench.data.hif.requests.get") as mock_get,
        patch("hyperbench.data.hif.from_zst_bytes_to_json", return_value=payload),
        patch("hyperbench.data.hif.validate_hif_data", return_value=True),
        patch("hyperbench.data.hif.os.path.abspath", return_value=str(tmp_path / "hif.py")),
        patch("hyperbench.data.hif.write_zst_file_to_disk") as mock_save_zst_file,
    ):
        mock_response = mock_get.return_value
        mock_response.status_code = 200
        mock_response.content = b"downloaded-content"

        result = HIFLoader.load_by_name("algebra", save_on_disk=True)

    mock_save_zst_file.assert_called_once()
    assert result.num_nodes == 2
    assert result.num_hyperedges == 1


def test_load_skips_saving_downloaded_dataset_when_save_on_disk_is_false(tmp_path, mock_hypergraph):
    payload = __hif_payload(mock_hypergraph)

    with (
        patch("hyperbench.data.hif.os.path.exists", return_value=False),
        patch("hyperbench.data.hif.requests.get") as mock_get,
        patch("hyperbench.data.hif.from_zst_bytes_to_json", return_value=payload),
        patch("hyperbench.data.hif.validate_hif_data", return_value=True),
        patch("hyperbench.data.hif.write_zst_file_to_disk") as mock_save_zst_file,
    ):
        mock_response = mock_get.return_value
        mock_response.status_code = 200
        mock_response.content = b"downloaded-content"

        result = HIFLoader.load_by_name("algebra", save_on_disk=False)

    mock_save_zst_file.assert_not_called()
    assert result.num_nodes == 2
    assert result.num_hyperedges == 1


def test_hifloader_download_raises_when_network_error():
    with (
        patch("hyperbench.data.hif.os.path.exists", return_value=False),
        patch(
            "hyperbench.data.hif.requests.get",
            side_effect=requests.RequestException("Network error"),
        ),
        pytest.raises(requests.RequestException, match="Network error"),
    ):
        HIFLoader.load_by_name("algebra")


def test_load_by_name_uses_hf_revision_when_github_download_fails(tmp_path, mock_hypergraph):
    hf_sha = "2bb641461e00c103fb5ef4fe6a30aad42500fc21"
    fallback_file = tmp_path / "algebra.json.zst"
    fallback_file.write_bytes(b"mock_zst_content")
    payload = __hif_payload(mock_hypergraph)

    response = requests.Response()
    response.status_code = 404
    response._content = b""

    with (
        patch("hyperbench.data.hif.os.path.exists", return_value=False),
        patch("hyperbench.data.hif.os.path.isdir", return_value=False),
        patch("hyperbench.data.hif.requests.get", return_value=response),
        patch(
            "hyperbench.data.hif.hf_hub_download", return_value=str(fallback_file)
        ) as mock_hf_hub_download,
        patch("hyperbench.data.hif.from_zst_file_to_json", return_value=payload),
        patch("hyperbench.data.hif.validate_hif_data", return_value=True),
        patch("hyperbench.data.hif.os.getenv", return_value=None),
        pytest.warns(UserWarning, match="GitHub raw download failed"),
    ):
        result = HIFLoader.load_by_name("algebra", hf_sha=hf_sha, save_on_disk=False)

    mock_hf_hub_download.assert_called_once_with(
        repo_id="HypernetworkRG/algebra",
        filename="algebra.json.zst",
        repo_type="dataset",
        revision=hf_sha,
        cache_dir=str(tmp_path / "hf_cache"),
        token=None,
    )
    assert result.num_nodes == 2
    assert result.num_hyperedges == 1
    assert not (tmp_path / "hf_cache" / "datasets--HypernetworkRG--algebra").exists()
    assert not (tmp_path / "hf_cache" / ".locks" / "datasets--HypernetworkRG--algebra").exists()


def test_load_by_name_skips_cache_cleanup_when_hf_cache_dir_is_missing(tmp_path, mock_hypergraph):
    hf_sha = "2bb641461e00c103fb5ef4fe6a30aad42500fc21"
    fallback_file = tmp_path / "algebra.json.zst"
    fallback_file.write_bytes(b"mock_zst_content")
    payload = __hif_payload(mock_hypergraph)

    response = requests.Response()
    response.status_code = 404
    response._content = b""

    with (
        patch("hyperbench.data.hif.os.path.exists", return_value=False),
        patch("hyperbench.data.hif.os.path.isdir", return_value=False),
        patch("hyperbench.data.hif.requests.get", return_value=response),
        patch(
            "hyperbench.data.hif.hf_hub_download", return_value=str(fallback_file)
        ) as mock_hf_hub_download,
        patch("hyperbench.data.hif.from_zst_file_to_json", return_value=payload),
        patch("hyperbench.data.hif.validate_hif_data", return_value=True),
        patch("hyperbench.data.hif.shutil.rmtree") as mock_rmtree,
        patch("hyperbench.data.hif.os.getenv", return_value=None),
        pytest.warns(UserWarning, match="GitHub raw download failed"),
    ):
        result = HIFLoader.load_by_name("algebra", hf_sha=hf_sha, save_on_disk=False)

    mock_hf_hub_download.assert_called_once_with(
        repo_id="HypernetworkRG/algebra",
        filename="algebra.json.zst",
        repo_type="dataset",
        revision=hf_sha,
        cache_dir=str(tmp_path / "hf_cache"),
        token=None,
    )
    mock_rmtree.assert_not_called()
    assert result.num_nodes == 2
    assert result.num_hyperedges == 1


def test_load_by_name_cleans_hf_cache_and_locks(tmp_path, mock_hypergraph):
    hf_sha = "2bb641461e00c103fb5ef4fe6a30aad42500fc21"
    fallback_file = tmp_path / "algebra.json.zst"
    fallback_file.write_bytes(b"mock_zst_content")
    payload = __hif_payload(mock_hypergraph)

    response = requests.Response()
    response.status_code = 404
    response._content = b""

    with (
        patch("hyperbench.data.hif.os.path.exists", return_value=False),
        patch("hyperbench.data.hif.os.path.isdir", return_value=True),
        patch("hyperbench.data.hif.requests.get", return_value=response),
        patch("hyperbench.data.hif.hf_hub_download", return_value=str(fallback_file)),
        patch("hyperbench.data.hif.from_zst_file_to_json", return_value=payload),
        patch("hyperbench.data.hif.validate_hif_data", return_value=True),
        patch("hyperbench.data.hif.shutil.rmtree") as mock_rmtree,
        pytest.warns(UserWarning, match="GitHub raw download failed"),
    ):
        result = HIFLoader.load_by_name("algebra", hf_sha=hf_sha, save_on_disk=False)

    cache_root = tmp_path / "hf_cache"
    path_prefix = "datasets--HypernetworkRG--algebra"
    mock_rmtree.assert_any_call(os.path.join(cache_root, path_prefix))
    mock_rmtree.assert_any_call(os.path.join(cache_root, ".locks", path_prefix))
    assert result.num_nodes == 2
    assert result.num_hyperedges == 1


def test_load_by_name_raises_when_hf_sha_is_missing_on_fallback():
    response = requests.Response()
    response.status_code = 404
    response._content = b""

    with (
        patch("hyperbench.data.hif.os.path.exists", return_value=False),
        patch("hyperbench.data.hif.requests.get", return_value=response),
        patch("hyperbench.data.hif.hf_hub_download") as mock_hf_hub_download,
        pytest.warns(UserWarning, match="GitHub raw download failed"),
        pytest.raises(
            ValueError,
            match="no SHA provided for Hugging Face Hub fallback",
        ),
    ):
        HIFLoader.load_by_name("algebra", save_on_disk=False)

    mock_hf_hub_download.assert_not_called()


def test_load_by_name_reads_hf_download_and_saves_its_content(tmp_path, mock_hypergraph):
    hf_sha = "2bb641461e00c103fb5ef4fe6a30aad42500fc21"
    hf_content = b"mock_zst_content"
    fallback_file = tmp_path / "algebra.json.zst"
    fallback_file.write_bytes(hf_content)
    payload = __hif_payload(mock_hypergraph)

    response = requests.Response()
    response.status_code = 404
    response._content = b""

    with (
        patch("hyperbench.data.hif.os.path.isdir", return_value=False),
        patch("hyperbench.data.hif.requests.get", return_value=response),
        patch("hyperbench.data.hif.hf_hub_download", return_value=str(fallback_file)),
        patch("hyperbench.data.hif.from_zst_file_to_json", return_value=payload),
        patch("hyperbench.data.hif.validate_hif_data", return_value=True),
        patch("hyperbench.data.hif.__file__", str(tmp_path / "hif.py")),
        pytest.warns(UserWarning, match="GitHub raw download failed"),
    ):
        result = HIFLoader.load_by_name("algebra", hf_sha=hf_sha, save_on_disk=True)

    saved = tmp_path / "datasets" / "algebra.json.zst"
    assert saved.exists()
    assert saved.read_bytes() == hf_content
    assert result.num_nodes == 2
    assert result.num_hyperedges == 1


def test_load_by_name_raises_warn_when_fail_to_cleanup_hf_cache(tmp_path, mock_hypergraph):
    hf_sha = "2bb641461e00c103fb5ef4fe6a30aad42500fc21"
    hf_content = b"mock_zst_content"
    fallback_file = tmp_path / "algebra.json.zst"
    fallback_file.write_bytes(hf_content)
    payload = __hif_payload(mock_hypergraph)
    response = requests.Response()
    response.status_code = 404
    response._content = b""

    with (
        patch("hyperbench.data.hif.os.path.exists", return_value=False),
        patch("hyperbench.data.hif.os.path.isdir", return_value=True),
        patch("hyperbench.data.hif.requests.get", return_value=response),
        patch("hyperbench.data.hif.hf_hub_download", return_value=str(fallback_file)),
        patch("hyperbench.data.hif.from_zst_file_to_json", return_value=payload),
        patch("hyperbench.data.hif.validate_hif_data", return_value=True),
        patch(
            "hyperbench.data.hif.shutil.rmtree",
            side_effect=FileNotFoundError(
                f"[Errno 2] No such file or directory: '"
                f"{tmp_path / 'hf_cache' / 'datasets--HypernetworkRG--algebra'}'"
            ),
        ),
        pytest.warns(UserWarning, match="Failed to clean up Hugging Face Hub cache"),
        pytest.warns(UserWarning, match="GitHub raw download failed"),
    ):
        result = HIFLoader.load_by_name("algebra", hf_sha=hf_sha, save_on_disk=False)

    assert result.num_nodes == 2
    assert result.num_hyperedges == 1


def test_load_by_name_raises_when_downloaded_hf_file_cannot_be_read(tmp_path):
    hf_sha = "2bb641461e00c103fb5ef4fe6a30aad42500fc21"
    response = requests.Response()
    response.status_code = 404
    response._content = b""

    with (
        patch("hyperbench.data.hif.os.path.exists", return_value=False),
        patch("hyperbench.data.hif.requests.get", return_value=response),
        patch(
            "hyperbench.data.hif.hf_hub_download",
            return_value=str(tmp_path / "downloaded.json.zst"),
        ) as mock_download,
        patch(
            "hyperbench.data.hif.from_zst_file_to_json",
            side_effect=ValueError(
                "Failed to read compressed JSON file 'downloaded.json.zst': missing file."
            ),
        ),
        pytest.warns(UserWarning, match="GitHub raw download failed"),
        pytest.raises(
            ValueError,
            match=re.escape(
                "Failed to read compressed JSON file 'downloaded.json.zst': missing file."
            ),
        ),
        patch("hyperbench.data.hif.os.getenv", return_value=None),
    ):
        HIFLoader.load_by_name("algebra", hf_sha=hf_sha)

    mock_download.assert_called_once_with(
        repo_id="HypernetworkRG/algebra",
        filename="algebra.json.zst",
        repo_type="dataset",
        revision=hf_sha,
        cache_dir=str(tmp_path / "hf_cache"),
        token=None,
    )


def test_load_by_name_raises_when_downloaded_hf_content_cannot_be_written(tmp_path):
    hf_sha = "2bb641461e00c103fb5ef4fe6a30aad42500fc21"
    response = requests.Response()
    response.status_code = 404
    response._content = b""
    payload = __hif_payload(
        HIFHypergraph(
            network_type="undirected",
            nodes=[{"node": "0", "attrs": {}}],
            hyperedges=[{"edge": "0", "attrs": {}}],
            incidences=[{"node": "0", "edge": "0"}],
        )
    )

    with (
        patch("hyperbench.data.hif.os.path.exists", return_value=False),
        patch("hyperbench.data.hif.requests.get", return_value=response),
        patch(
            "hyperbench.data.hif.hf_hub_download", return_value=str(tmp_path / "fallback.json.zst")
        ),
        patch("hyperbench.data.hif.from_zst_file_to_json", return_value=payload),
        patch("hyperbench.data.hif.validate_hif_data", return_value=True),
        patch("hyperbench.data.hif.shutil.copyfile", side_effect=OSError("disk full")),
        pytest.warns(UserWarning, match="GitHub raw download failed"),
        pytest.raises(
            ValueError,
            match=re.compile(
                "Failed to save downloaded dataset 'algebra' to disk at '.*algebra.json.zst': "
                "disk full."
            ),
        ),
    ):
        HIFLoader.load_by_name("algebra", hf_sha=hf_sha, save_on_disk=True)


def test_load_by_name_raises_when_saving_downloaded_dataset_fails(tmp_path):
    response = requests.Response()
    response.status_code = 200
    response._content = b"downloaded-content"

    with (
        patch("hyperbench.data.hif.os.path.exists", return_value=False),
        patch("hyperbench.data.hif.requests.get", return_value=response),
        patch(
            "hyperbench.data.hif.from_zst_bytes_to_json",
            return_value=__hif_payload(
                HIFHypergraph(
                    network_type="undirected",
                    nodes=[{"node": "0", "attrs": {}}],
                    hyperedges=[{"edge": "0", "attrs": {}}],
                    incidences=[{"node": "0", "edge": "0"}],
                )
            ),
        ),
        patch("hyperbench.data.hif.validate_hif_data", return_value=True),
        patch(
            "hyperbench.data.hif.write_zst_file_to_disk",
            side_effect=ValueError("Failed to save downloaded 'algebra.json.zst'"),
        ),
        pytest.raises(
            ValueError,
            match=re.escape("Failed to save downloaded 'algebra.json.zst'"),
        ),
    ):
        HIFLoader.load_by_name("algebra", save_on_disk=True)


def test_load_from_path_raises_error_when_json_file_cannot_be_read(tmp_path):
    json_path = tmp_path / "sample.json"
    json_path.write_text("{}", encoding="utf-8")

    with (
        patch("hyperbench.data.hif.validate_hif_data", return_value=True),
        patch("builtins.open", side_effect=OSError("missing file")),
        pytest.raises(ValueError, match=re.compile("Failed to read JSON file '.*sample.json'")),
    ):
        HIFLoader.load_from_path(str(json_path))


def test_hifloader_download_failure_when_hf_fallback_fails(tmp_path):
    hf_sha = "2bb641461e00c103fb5ef4fe6a30aad42500fc21"
    response = requests.Response()
    response.status_code = 404
    response._content = b""

    with (
        patch("hyperbench.data.hif.os.path.exists", return_value=False),
        patch("hyperbench.data.hif.requests.get", return_value=response),
        patch(
            "hyperbench.data.hif.hf_hub_download",
            side_effect=Exception("HFHub failed"),
        ) as mock_hf_hub_download,
        pytest.warns(UserWarning, match="GitHub raw download failed"),
        pytest.raises(
            ValueError,
            match=re.escape(
                "Failed to download dataset 'algebra' from GitHub and Hugging Face Hub. "
                "GitHub error: 404 | Hugging Face error: HFHub failed"
            ),
        ),
        patch("hyperbench.data.hif.os.getenv", return_value=None),
    ):
        HIFLoader.load_by_name("algebra", hf_sha=hf_sha)

    mock_hf_hub_download.assert_called_once_with(
        repo_id="HypernetworkRG/algebra",
        filename="algebra.json.zst",
        repo_type="dataset",
        revision=hf_sha,
        cache_dir=str(tmp_path / "hf_cache"),
        token=None,
    )


def test_hifloader_download_failure_when_hf_token_is_invalid(tmp_path):
    hf_sha = "2bb641461e00c103fb5ef4fe6a30aad42500fc21"
    response = requests.Response()
    response.status_code = 404
    response._content = b""

    with (
        patch("hyperbench.data.hif.os.path.exists", return_value=False),
        patch("hyperbench.data.hif.requests.get", return_value=response),
        patch(
            "hyperbench.data.hif.hf_hub_download",
            side_effect=Exception("HFHub failed"),
        ) as mock_hf_hub_download,
        pytest.warns(UserWarning, match="GitHub raw download failed"),
        pytest.raises(
            ValueError,
            match=(
                re.escape(
                    "Failed to download dataset 'algebra' from GitHub and Hugging Face Hub. "
                    "GitHub error: 404 | Hugging Face error: HFHub failed"
                )
            ),
        ),
        patch("hyperbench.data.hif.os.getenv", return_value="invalid_token"),
    ):
        HIFLoader.load_by_name("algebra", hf_sha=hf_sha)

    mock_hf_hub_download.assert_called_once_with(
        repo_id="HypernetworkRG/algebra",
        filename="algebra.json.zst",
        repo_type="dataset",
        revision=hf_sha,
        cache_dir=str(tmp_path / "hf_cache"),
        token="invalid_token",
    )
