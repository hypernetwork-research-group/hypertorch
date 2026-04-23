import requests
import json
import os

from unittest.mock import patch, mock_open, MagicMock
from hyperbench.utils import validate_hif_json, compress_to_zst, decompress_zst
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
            "https://raw.githubusercontent.com/HIF-org/HIF-standard/main/schemas/hif_schema.json",
            timeout=10,
        )


def test_validate_hif_json_with_url_timeout_fallback():
    path_valid = f"{MOCK_BASE_PATH}/hif_compliant.json"

    with (
        patch("hyperbench.utils.hif_utils.requests.get") as mock_get,
        patch("builtins.open", mock_open(read_data='{"type": "object"}')) as mock_file,
    ):
        mock_get.side_effect = requests.Timeout("Connection timeout")
        validate_hif_json(path_valid)
        # local file was opened
        calls = [str(call) for call in mock_file.call_args_list]
        assert any("../schema/hif_schema.json" in call for call in calls)


def test_validate_hif_json_with_url_request_exception_fallback():
    path_valid = f"{MOCK_BASE_PATH}/hif_compliant.json"

    with (
        patch("hyperbench.utils.hif_utils.requests.get") as mock_get,
        patch("builtins.open", mock_open(read_data='{"type": "object"}')) as mock_file,
    ):
        mock_get.side_effect = requests.RequestException("Network error")
        validate_hif_json(path_valid)
        # local file was opened
        calls = [str(call) for call in mock_file.call_args_list]
        assert any("../schema/hif_schema.json" in call for call in calls)


def test_compress_to_zst_returns_non_empty_bytes(tmp_path):
    json_path = tmp_path / "sample.json"
    json_path.write_text('{"nodes": [], "edges": [], "incidences": []}')

    compressed_content = compress_to_zst(str(json_path))

    assert isinstance(compressed_content, bytes)
    assert len(compressed_content) > 0


def test_decompress_zst_round_trip_preserves_json_content(tmp_path):
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

    with open(decompressed_path, "r") as f:
        decompressed_data = json.load(f)

    assert decompressed_data == expected_data
