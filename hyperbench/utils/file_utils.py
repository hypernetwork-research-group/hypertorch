import io
import os
import json
import zstandard as zstd

from typing import Any


def compress_json_bytes_as_zst(content: bytes) -> bytes:
    try:
        return zstd.ZstdCompressor().compress(content)
    except Exception as e:
        raise ValueError(f"Failed to compress JSON content: {e!s}.") from e


def read_json_bytes(content: bytes) -> dict[str, Any]:
    try:
        return json.loads(content.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"Failed to read JSON content: {e!s}.") from e


def read_json_file(json_filename: str) -> dict[str, Any]:
    try:
        with open(json_filename, encoding="utf-8") as json_file:
            return json.load(json_file)
    except Exception as e:
        raise ValueError(f"Failed to read JSON file {json_filename!r}: {e!s}.") from e


def from_zst_bytes_to_json(content: bytes) -> dict[str, Any]:
    try:
        with io.BytesIO(content) as input_zst_file:
            return __read_zst_stream(input_zst_file)
    except Exception as e:
        raise ValueError(f"Failed to read compressed JSON byte data: {e!s}.") from e


def from_zst_file_to_json(zst_filename: str) -> dict[str, Any]:
    try:
        with open(zst_filename, "rb") as input_zst_file:
            return __read_zst_stream(input_zst_file)
    except Exception as e:
        raise ValueError(f"Failed to read compressed JSON file {zst_filename!r}: {e!s}.") from e


def __read_zst_stream(input_zst_file: Any) -> dict[str, Any]:
    with (
        zstd.ZstdDecompressor().stream_reader(input_zst_file) as zst_reader,
        io.TextIOWrapper(zst_reader, encoding="utf-8") as text_reader,
    ):
        try:
            return json.load(text_reader)
        except Exception as e:
            raise ValueError(f"Failed to read JSON data for {input_zst_file.name!r}: {e!s}.") from e


def save_zst_file(zst_filename: str, content: bytes) -> None:
    try:
        os.makedirs(os.path.dirname(zst_filename), exist_ok=True)
        with open(zst_filename, "wb") as zst_file:
            zst_file.write(content)
    except Exception as e:
        raise ValueError(f"Failed to save downloaded {zst_filename!r}: {e!s}.") from e


def write_dataset_to_disk_as_zst(
    dataset_name: str, content: bytes, output_dir: str | None = None
) -> None:
    """
    Writes the compressed content to disk in the specified output directory or a default location.

    Args:
        dataset_name: The name of the dataset.
        content: The compressed content as bytes.
        output_dir: The directory to write the file to. If None, a default location is used.
    """
    try:
        if output_dir is not None:
            zst_filename = os.path.join(output_dir, f"{dataset_name}.json.zst")
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(current_dir, "..", "data", "datasets")
            zst_filename = os.path.join(output_dir, f"{dataset_name}.json.zst")
    except Exception as e:
        raise ValueError(
            f"Failed to determine output path for dataset {dataset_name!r}: {e!s}."
        ) from e

    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(zst_filename, "wb") as f:
            f.write(content)
    except Exception as e:
        raise ValueError(
            f"Failed to write file {zst_filename!r} to disk {output_dir!r}: {e!s}."
        ) from e
