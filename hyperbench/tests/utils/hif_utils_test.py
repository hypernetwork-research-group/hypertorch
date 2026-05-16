import requests
import json
import os
import pytest

from unittest.mock import patch, mock_open, MagicMock
from hyperbench.utils import (
    validate_hif_json,
    compress_to_zst,
    decompress_zst,
    get_hf_datasets_shas,
    get_hf_dataset_sha,
    get_gh_datasets_shas,
    get_gh_dataset_sha,
)
from hyperbench.tests import MOCK_BASE_PATH


def test_validate_hif_json():
    path_invalid = f"{MOCK_BASE_PATH}/hif_not_compliant.json"
    assert not validate_hif_json(path_invalid)

    path_valid = f"{MOCK_BASE_PATH}/hif_compliant.json"
    assert validate_hif_json(path_valid)


def test_validate_hif_json_with_url_success():
    path_valid = f"{MOCK_BASE_PATH}/hif_compliant.json"

    with patch("hyperbench.utils.hif_utils.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {"type": "object"}  # Minimal valid schema
        mock_get.return_value = mock_response

        validate_hif_json(path_valid)
        mock_get.assert_called_once_with(
            "https://raw.githubusercontent.com/HIF-org/HIF-standard/b691a3d2ec32100c0229ebe1151e9afad015c356/schemas/hif_schema.json",
            timeout=10,
        )


def test_validate_hif_json_with_url_timeout_fallback():
    path_valid = f"{MOCK_BASE_PATH}/hif_compliant.json"

    mock_file = mock_open(read_data='{"type": "object"}')
    mock_path = MagicMock()
    mock_path.open.return_value = mock_file()
    mock_files = MagicMock()
    mock_files.joinpath.return_value = mock_path

    with (
        patch("hyperbench.utils.hif_utils.requests.get") as mock_get,
        patch(
            "hyperbench.utils.hif_utils.resources.files", return_value=mock_files
        ) as mock_files_call,
    ):
        mock_get.side_effect = requests.Timeout("Connection timeout")
        validate_hif_json(path_valid)

        mock_files_call.assert_called_once_with("hyperbench.utils.schema")
        mock_files.joinpath.assert_called_once_with("hif_schema.json")
        mock_path.open.assert_called_once_with("r")


def test_validate_hif_json_with_url_request_exception_fallback():
    path_valid = f"{MOCK_BASE_PATH}/hif_compliant.json"
    mock_file = mock_open(read_data='{"type": "object"}')
    mock_path = MagicMock()
    mock_path.open.return_value = mock_file()
    mock_files = MagicMock()
    mock_files.joinpath.return_value = mock_path

    with (
        patch("hyperbench.utils.hif_utils.requests.get") as mock_get,
        patch(
            "hyperbench.utils.hif_utils.resources.files", return_value=mock_files
        ) as mock_files_call,
    ):
        mock_get.side_effect = requests.RequestException("Network error")
        validate_hif_json(path_valid)

        mock_files_call.assert_called_once_with("hyperbench.utils.schema")
        mock_files.joinpath.assert_called_once_with("hif_schema.json")
        mock_path.open.assert_called_once_with("r")


def test_compress_to_zst_returns_non_empty_bytes(tmp_path):
    json_path = tmp_path / "sample.json"
    json_path.write_text('{"nodes": [], "edges": [], "incidences": []}')

    compressed_content = compress_to_zst(str(json_path))

    assert isinstance(compressed_content, bytes)
    assert len(compressed_content) > 0


def test_decompress_returns_correct_json(tmp_path):
    expected_data = {
        "network-type": "undirected",
        "nodes": [{"node": "0", "attrs": {"weight": 1.0}}],
        "edges": [{"edge": "0", "attrs": {}}],
        "incidences": [{"node": "0", "edge": "0"}],
    }

    json_path = tmp_path / "sample.json"
    with open(json_path, "w") as f:
        json.dump(expected_data, f)

    compressed_content = compress_to_zst(str(json_path))
    zst_path = tmp_path / "sample.json.zst"
    zst_path.write_bytes(compressed_content)

    decompressed_path = decompress_zst(str(zst_path))

    assert decompressed_path.endswith(".json")
    assert os.path.exists(decompressed_path)

    with open(decompressed_path) as f:
        decompressed_data = json.load(f)

    assert decompressed_data == expected_data


def test_get_datasets_shas_returns_shas_and_none_on_failure():
    names = ["algebra", "missing-dataset"]

    def dataset_info_side_effect(*, repo_id):
        if repo_id.endswith("/missing-dataset"):
            raise RuntimeError("not found")
        info = MagicMock()
        info.sha = "sha-algebra"
        return info

    with (
        patch("hyperbench.utils.hif_utils.HfApi") as mock_hf_api,
        pytest.warns(UserWarning, match="missing-dataset: failed to retrieve SHA"),
    ):
        mock_hf_api.return_value.dataset_info.side_effect = dataset_info_side_effect

        result = get_hf_datasets_shas(names)

    assert result == {
        "algebra": "sha-algebra",
        "missing-dataset": None,
    }
    mock_hf_api.return_value.dataset_info.assert_any_call(
        repo_id="HypernetworkRG/algebra"
    )
    mock_hf_api.return_value.dataset_info.assert_any_call(
        repo_id="HypernetworkRG/missing-dataset"
    )


def test_get_dataset_sha_returns_sha():
    with patch("hyperbench.utils.hif_utils.HfApi") as mock_hf_api:
        info = MagicMock()
        info.sha = "sha-algebra"
        mock_hf_api.return_value.dataset_info.return_value = info

        result = get_hf_dataset_sha("algebra")

    assert result == "sha-algebra"
    mock_hf_api.return_value.dataset_info.assert_called_once_with(
        repo_id="HypernetworkRG/algebra"
    )


def test_get_dataset_sha_returns_none_on_failure():
    with (
        patch("hyperbench.utils.hif_utils.HfApi") as mock_hf_api,
        pytest.warns(UserWarning, match="algebra: failed to retrieve SHA"),
    ):
        mock_hf_api.return_value.dataset_info.side_effect = RuntimeError("boom")

        result = get_hf_dataset_sha("algebra")

    assert result is None
    mock_hf_api.return_value.dataset_info.assert_called_once_with(
        repo_id="HypernetworkRG/algebra"
    )


def test_get_gh_datasets_shas_returns_shas_and_none_on_failure():
    names = ["algebra", "missing-dataset"]

    def requests_get_side_effect(url, *, params):
        if params["path"] == "missing-dataset.json.zst":
            raise requests.RequestException("not found")
        mock_response = MagicMock()
        mock_response.json.return_value = [{"sha": "sha-algebra"}]
        return mock_response

    with (
        patch(
            "hyperbench.utils.hif_utils.requests.get",
            side_effect=requests_get_side_effect,
        ) as mock_requests_get,
        pytest.warns(UserWarning, match="missing-dataset: failed to retrieve SHA"),
    ):
        result = get_gh_datasets_shas(names)

    assert result == {
        "algebra": "sha-algebra",
        "missing-dataset": None,
    }
    mock_requests_get.assert_any_call(
        "https://api.github.com/repos/hypernetwork-research-group/datasets/commits",
        params={"path": "algebra.json.zst", "per_page": 1},
        timeout=10,
    )
    mock_requests_get.assert_any_call(
        "https://api.github.com/repos/hypernetwork-research-group/datasets/commits",
        params={"path": "missing-dataset.json.zst", "per_page": 1},
        timeout=10,
    )


def test_get_gh_dataset_trigger_no_commit():
    def requests_get_side_effect(url, *, params):
        mock_response = MagicMock()
        mock_response.json.return_value = []  # No commits found
        return mock_response

    with (
        patch(
            "hyperbench.utils.hif_utils.requests.get",
            side_effect=requests_get_side_effect,
        ) as mock_requests_get,
        pytest.warns(
            UserWarning, match="algebra: no commits found for algebra.json.zst"
        ),
    ):
        result = get_gh_dataset_sha("algebra", "owner", "repo")

    assert result is None
    mock_requests_get.assert_called_once_with(
        "https://api.github.com/repos/owner/repo/commits",
        params={"path": "algebra.json.zst", "per_page": 1},
        timeout=10,
    )
