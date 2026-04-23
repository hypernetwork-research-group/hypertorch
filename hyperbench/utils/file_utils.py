import zstandard as zstd
import tempfile
import os


def decompress_zst(zst_path: str) -> str:
    dctx = zstd.ZstdDecompressor()
    with (
        open(zst_path, "rb") as input_f,
        tempfile.NamedTemporaryFile(mode="wb", suffix=".json", delete=False) as tmp_file,
    ):
        dctx.copy_stream(input_f, tmp_file)
        output = tmp_file.name
    return output


def compress_to_zst(json_path: str) -> bytes:
    cctx = zstd.ZstdCompressor()
    with open(json_path, "rb") as input_f:
        compressed_content = cctx.compress(input_f.read())
    return compressed_content


def save_on_disk(dataset_name: str, content: bytes) -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(current_dir, "..", "data", "datasets")
    zst_filename = os.path.join(datasets_dir, f"{dataset_name}.json.zst")
    os.makedirs(datasets_dir, exist_ok=True)

    with open(zst_filename, "wb") as f:
        f.write(content)
