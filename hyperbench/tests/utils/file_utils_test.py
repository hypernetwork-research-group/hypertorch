import os
import pytest
from hyperbench.utils import (
    compress_to_zst,
    decompress_zst,
    write_to_disk,
    named_temporary_file,
)
from unittest.mock import patch


def test_write_to_disk_writes_file_default_output_dir(tmp_path):
    dataset_name = "test_dataset"
    content = b"test content"

    # Force write_to_disk default branch to resolve under tmp_path.
    fake_module_file = tmp_path / "hyperbench" / "utils" / "file_utils.py"

    with patch(
        "hyperbench.utils.file_utils.os.path.abspath",
        return_value=str(fake_module_file),
    ):
        write_to_disk(dataset_name, content)

    expected_path = tmp_path / "hyperbench" / "data" / "datasets" / f"{dataset_name}.json.zst"
    assert expected_path.is_file()
    assert expected_path.read_bytes() == content


def test_write_to_disk_writes_file_optional_output_dir(tmp_path):
    dataset_name = "test_dataset"
    content = b"test content"
    output_dir = tmp_path

    write_to_disk(dataset_name, content, output_dir)

    expected_path = tmp_path / f"{dataset_name}.json.zst"
    assert expected_path.is_file()

    with open(expected_path, "rb") as f:
        file_content = f.read()
        assert file_content == content


def test_create_named_temporary_file(tmp_path):
    content = b"temporary file content"
    temp_file_path = named_temporary_file(content, suffix=".txt")

    assert os.path.isfile(temp_file_path)

    with open(temp_file_path, "rb") as f:
        file_content = f.read()
        assert file_content == content
        assert temp_file_path.endswith(".txt")


def test_decompress_zst_raises_value_error_when_input_missing(tmp_path):
    with (
        patch("builtins.open", side_effect=OSError("missing file")),
        patch("hyperbench.utils.file_utils.pretty_print_disk_space_stats") as mock_stats,
    ):
        mock_stats.return_value = None

        with pytest.raises(ValueError, match=r"Failed to decompress \.zst file"):
            decompress_zst(f"{tmp_path}/missing.json.zst")


def test_compress_to_zst_raises_value_error_when_input_missing(tmp_path):
    with (
        patch("builtins.open", side_effect=OSError("missing file")),
        patch("hyperbench.utils.file_utils.pretty_print_disk_space_stats") as mock_stats,
    ):
        mock_stats.return_value = None

        with pytest.raises(ValueError, match=r"Failed to compress JSON file"):
            compress_to_zst(f"{tmp_path}/missing.json")


def test_write_to_disk_raises_value_error_when_output_path_cannot_be_determined():
    with (
        patch("hyperbench.utils.file_utils.os.path.abspath", side_effect=OSError("boom")),
        patch("hyperbench.utils.file_utils.pretty_print_disk_space_stats") as mock_stats,
    ):
        mock_stats.return_value = None

        with pytest.raises(ValueError, match=r"Failed to determine output path for dataset"):
            write_to_disk("test_dataset", b"content")


def test_write_to_disk_raises_value_error_when_write_fails(tmp_path):
    with (
        patch("builtins.open", side_effect=OSError("cannot write")),
        patch("hyperbench.utils.file_utils.pretty_print_disk_space_stats") as mock_stats,
    ):
        mock_stats.return_value = None

        with pytest.raises(ValueError, match=r"Failed to write file"):
            write_to_disk("test_dataset", b"content", str(tmp_path))


def test_named_temporary_file_raises_value_error_when_creation_fails():
    with (
        patch(
            "hyperbench.utils.file_utils.tempfile.NamedTemporaryFile", side_effect=OSError("boom")
        ),
        patch("hyperbench.utils.file_utils.pretty_print_disk_space_stats") as mock_stats,
    ):
        mock_stats.return_value = None

        with pytest.raises(ValueError, match=r"Failed to create temporary file"):
            named_temporary_file(b"content")
