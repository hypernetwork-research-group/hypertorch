import shutil
import os
import tempfile
import zstandard as zstd


def decompress_zst(zst_path: str) -> str:
    """
    Decompresses a .zst file and returns the path to the decompressed JSON file.

    Args:
        zst_path: The path to the .zst file to decompress.

    Returns:
        path: The path to the decompressed JSON file.
    """
    try:
        dctx = zstd.ZstdDecompressor()
        with (
            open(zst_path, "rb") as input_f,
            tempfile.NamedTemporaryFile(mode="wb", suffix=".json", delete=False) as tmp_file,
        ):
            dctx.copy_stream(input_f, tmp_file)
            output = tmp_file.name
    except Exception as e:
        raise ValueError(f"Failed to decompress .zst file {zst_path!r}: {e!s}") from e
    return output


def compress_to_zst(json_path: str) -> bytes:
    """
    Compresses a JSON file to .zst format and returns the compressed bytes.

    Args:
        json_path: The path to the JSON file to compress.

    Returns:
        content: The compressed content as bytes.
    """
    try:
        cctx = zstd.ZstdCompressor()
        with open(json_path, "rb") as input_f:
            compressed_content = cctx.compress(input_f.read())
        return compressed_content
    except Exception as e:
        raise ValueError(f"Failed to compress JSON file {json_path!r}: {e!s}") from e


def write_to_disk(dataset_name: str, content: bytes, output_dir: str | None = None) -> None:
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
            f"Failed to determine output path for dataset {dataset_name!r}: {e!s}"
        ) from e

    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(zst_filename, "wb") as f:
            f.write(content)
    except Exception as e:
        raise ValueError(
            f"Failed to write file {zst_filename!r} to disk {output_dir!r}: {e!s}"
        ) from e


def named_temporary_file(content: bytes, suffix: str = ".json.zst") -> str:
    try:
        with tempfile.NamedTemporaryFile(mode="wb", suffix=suffix, delete=False) as tmp_zst_file:
            tmp_zst_file.write(content)
            zst_filename = tmp_zst_file.name
    except Exception as e:
        raise ValueError(f"Failed to create temporary file: {e!s}") from e
    return zst_filename


def get_disk_space_stats(path: str = ".") -> tuple[float, float, float]:
    """Return free, used, and total disk space for ``path`` in gigabytes."""

    usage = shutil.disk_usage(path)
    gigabyte = 1024 * 1024 * 1024
    return usage.free / gigabyte, usage.used / gigabyte, usage.total / gigabyte


def pretty_print_disk_space_stats(path: str = ".") -> None:
    """Print free, used, and total disk space for ``path`` in gigabytes."""
    free, used, total = get_disk_space_stats(path)
    print(f"Disk: [Free: {free:.2f}, Used:{used:.2f}, Total:{total:.2f}]")
