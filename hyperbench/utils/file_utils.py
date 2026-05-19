import os
import tempfile
import zstandard as zstd


def decompress_zst(zst_path: str) -> str:
    """
    Decompresses a .zst file and returns the path to the decompressed JSON file.
    Args:
        zst_path: The path to the .zst file to decompress.
    Returns:
        The path to the decompressed JSON file.
    """
    dctx = zstd.ZstdDecompressor()
    with (
        open(zst_path, "rb") as input_f,
        tempfile.NamedTemporaryFile(mode="wb", suffix=".json", delete=False) as tmp_file,
    ):
        dctx.copy_stream(input_f, tmp_file)
        output = tmp_file.name
    return output


def compress_to_zst(json_path: str) -> bytes:
    """
    Compresses a JSON file to .zst format and returns the compressed bytes.

    Args:
        json_path: The path to the JSON file to compress.
    Returns:
        The compressed content as bytes.
    """
    cctx = zstd.ZstdCompressor()
    with open(json_path, "rb") as input_f:
        compressed_content = cctx.compress(input_f.read())
    return compressed_content


def write_to_disk(dataset_name: str, content: bytes, output_dir: str | None = None) -> None:
    """
    Writes the compressed content to disk in the specified output directory or a default location.
    Args:
        dataset_name: The name of the dataset.
        content: The compressed content as bytes.
        output_dir: The directory to write the file to. If None, a default location is used.
    """
    if output_dir is not None:
        zst_filename = os.path.join(output_dir, f"{dataset_name}.json.zst")
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, "..", "data", "datasets")
        zst_filename = os.path.join(output_dir, f"{dataset_name}.json.zst")

    os.makedirs(output_dir, exist_ok=True)

    with open(zst_filename, "wb") as f:
        f.write(content)


def named_temporary_file(content: bytes, suffix: str = ".json.zst") -> str:
    with tempfile.NamedTemporaryFile(mode="wb", suffix=suffix, delete=False) as tmp_zst_file:
        tmp_zst_file.write(content)
        zst_filename = tmp_zst_file.name
    return zst_filename
