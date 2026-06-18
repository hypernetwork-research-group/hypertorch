import io
import os
import json
import zstandard as zstd

from pathlib import Path
from typing import Any


def compress_json_bytes_as_zst(content: bytes) -> bytes:
    """
    Compress JSON bytes with Zstandard.

    Args:
        content: JSON byte payload to compress.

    Returns:
        content: Compressed byte payload.

    Raises:
        ValueError: If compression fails.
    """
    try:
        return zstd.ZstdCompressor().compress(content)
    except Exception as e:
        raise ValueError(f"Failed to compress JSON content: {e!s}.") from e


def from_bytes_to_json(content: bytes) -> dict[str, Any]:
    """
    Decode JSON data from bytes.

    Args:
        content: UTF-8 encoded JSON bytes.

    Returns:
        data: Parsed JSON object.

    Raises:
        ValueError: If the bytes cannot be decoded or parsed as JSON.
    """
    try:
        return json.loads(content.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"Failed to read JSON content: {e!s}.") from e


def from_file_to_json(json_filename: str) -> dict[str, Any]:
    """
    Load JSON data from a file.

    Args:
        json_filename: Path to the JSON file.

    Returns:
        data: Parsed JSON object.

    Raises:
        ValueError: If the file cannot be read or parsed as JSON.
    """
    try:
        with open(json_filename, encoding="utf-8") as json_file:
            return json.load(json_file)
    except Exception as e:
        raise ValueError(f"Failed to read JSON file {json_filename!r}: {e!s}.") from e


def from_zst_bytes_to_json(content: bytes) -> dict[str, Any]:
    """
    Decompress Zstandard bytes and parse the contained JSON.

    Args:
        content: Zstandard-compressed JSON bytes.

    Returns:
        data: Parsed JSON object.

    Raises:
        ValueError: If decompression or JSON parsing fails.
    """
    try:
        with io.BytesIO(content) as input_zst_file:
            return __read_zst_stream(input_zst_file)
    except Exception as e:
        raise ValueError(f"Failed to read compressed JSON byte data: {e!s}.") from e


def from_zst_file_to_json(zst_filename: str) -> dict[str, Any]:
    """
    Load JSON data from a Zstandard-compressed file.

    Args:
        zst_filename: Path to the compressed JSON file.

    Returns:
        data: Parsed JSON object.

    Raises:
        ValueError: If the file cannot be decompressed or parsed as JSON.
    """
    try:
        with open(zst_filename, "rb") as input_zst_file:
            return __read_zst_stream(input_zst_file)
    except Exception as e:
        raise ValueError(f"Failed to read compressed JSON file {zst_filename!r}: {e!s}.") from e


def write_zst_file_to_disk(zst_filename: str, content: bytes) -> None:
    """
    Write compressed bytes to disk.

    Args:
        zst_filename: Destination file path.
        content: Compressed bytes to write.

    Raises:
        ValueError: If the file cannot be written.
    """
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
        output_dir: The directory to write the file to. If ``None``, a default location is used.
            Defaults to ``None``.
    """
    try:
        if output_dir is not None:
            zst_filename = os.path.join(output_dir, f"{dataset_name}.json.zst")
        else:
            cache_dir = get_cache_dir()
            output_dir = os.path.join(cache_dir, "datasets")
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


def __read_zst_stream(input_zst_file: Any) -> dict[str, Any]:
    """
    Read a compressed JSON stream.

    Args:
        input_zst_file: Binary file-like object containing Zstandard-compressed JSON.

    Returns:
        data: Parsed JSON object.

    Raises:
        ValueError: If JSON parsing fails after decompression.
    """
    with (
        zstd.ZstdDecompressor().stream_reader(input_zst_file) as zst_reader,
        io.TextIOWrapper(zst_reader, encoding="utf-8") as text_reader,
    ):
        try:
            return json.load(text_reader)
        except Exception as e:
            raise ValueError(f"Failed to read JSON data for {input_zst_file.name!r}: {e!s}.") from e


MARKERS = ("pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", ".git")


def find_project_root() -> Path:
    """
    Find the nearest project root from the current working directory.

    Returns:
        root: The nearest directory containing a known project marker, or the current directory.
    """
    current = Path.cwd().resolve()

    if current.is_file():
        current = current.parent

    for directory in (current, *current.parents):
        if any((directory / marker).exists() for marker in MARKERS):
            return directory

    return current


def get_cache_dir(
    create: bool = True,
    env_var: str = "HYPERBENCH_CACHE_DIR",
) -> Path:
    """
    Resolve the HyperBench cache directory.

    Args:
        create: Whether to create the directory if it does not exist.
        env_var: Environment variable that can override the default cache path.
            Defaults to ``"HYPERBENCH_CACHE_DIR"``.

    Returns:
        cache_dir: Absolute cache directory path.
    """
    override = os.getenv(env_var)

    if override:
        cache_dir = Path(override).expanduser()
        if not cache_dir.is_absolute():
            cache_dir = Path.cwd() / cache_dir
    else:
        cache_dir = find_project_root() / ".hyperbench_cache"

    cache_dir = cache_dir.resolve()

    if create:
        cache_dir.mkdir(parents=True, exist_ok=True)

    return cache_dir
