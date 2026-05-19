import os
from hyperbench.utils import write_to_disk, named_temporary_file
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
