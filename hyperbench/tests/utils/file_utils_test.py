import json
from unittest.mock import patch

import pytest
import zstandard as zstd

from hyperbench.utils import (
    compress_json_bytes_as_zst,
    from_bytes_to_json,
    from_file_to_json,
    from_zst_bytes_to_json,
    from_zst_file_to_json,
    write_zst_file_to_disk,
    write_dataset_to_disk_as_zst,
)


@pytest.mark.parametrize(
    "input_bytes",
    [
        b'{"nodes": [1, 2], "ok": true}',
        b'{"name": "hyperbench", "count": 1}',
        b"{}",
        b'{"nodes": [1, 2], "ok": true, "hyperedges": []}',
    ],
)
def test_compress_json_bytes_as_zst_round_trip(input_bytes):
    content = input_bytes
    compressed = compress_json_bytes_as_zst(content)

    assert zstd.ZstdDecompressor().decompress(compressed) == content


def test_compress_json_bytes_as_zst_raises_on_compression_error():
    with (
        patch(
            "hyperbench.utils.file_utils.zstd.ZstdCompressor.compress",
            side_effect=RuntimeError("boom"),
        ),
        pytest.raises(ValueError, match=r"Failed to compress JSON content: boom\."),
    ):
        compress_json_bytes_as_zst(b"{}")


@pytest.mark.parametrize(
    "input_bytes, expected",
    [
        (b'{"nodes": [1, 2], "ok": true}', {"nodes": [1, 2], "ok": True}),
        (b'{"name": "hyperbench", "count": 1}', {"name": "hyperbench", "count": 1}),
        (b"{}", {}),
        (
            b'{"nodes": [1, 2], "ok": true, "hyperedges": []}',
            {"nodes": [1, 2], "ok": True, "hyperedges": []},
        ),
    ],
)
def test_read_json_bytes_returns_parsed_data(input_bytes, expected):
    result = from_bytes_to_json(input_bytes)

    assert result == expected


def test_read_json_bytes_raises_on_invalid_json():
    with pytest.raises(ValueError, match="Failed to read JSON content:"):
        from_bytes_to_json(b"{not valid json")


def test_read_json_file_returns_parsed_data(tmp_path):
    json_path = tmp_path / "sample.json"
    json_path.write_text('{"name": "hyperbench"}', encoding="utf-8")

    result = from_file_to_json(str(json_path))

    assert result == {"name": "hyperbench"}


def test_read_json_file_raises_on_missing_file(tmp_path):
    with pytest.raises(ValueError, match=r"Failed to read JSON file '.*missing\.json'"):
        from_file_to_json(str(tmp_path / "missing.json"))


def test_from_zst_bytes_to_json_returns_parsed_data():
    payload = {"name": "hyperbench", "items": [1, 2]}
    compressed = compress_json_bytes_as_zst(json.dumps(payload).encode("utf-8"))

    result = from_zst_bytes_to_json(compressed)

    assert result == payload


def test_from_zst_bytes_to_json_raises_on_invalid_compressed_content():
    with pytest.raises(ValueError, match="Failed to read compressed JSON byte data:"):
        from_zst_bytes_to_json(b"not-a-zst-stream")


def test_from_zst_file_to_json_returns_parsed_data(tmp_path):
    payload = {"name": "hyperbench"}
    zst_path = tmp_path / "sample.json.zst"
    zst_path.write_bytes(compress_json_bytes_as_zst(json.dumps(payload).encode("utf-8")))

    result = from_zst_file_to_json(str(zst_path))

    assert result == payload


def test_from_zst_file_to_json_raises_on_invalid_compressed_file(tmp_path):
    zst_path = tmp_path / "bad.json.zst"
    zst_path.write_bytes(b"not-zst")

    with pytest.raises(ValueError, match=r"Failed to read compressed JSON file '.*bad\.json\.zst'"):
        from_zst_file_to_json(str(zst_path))


def test_save_zst_file_writes_bytes(tmp_path):
    zst_path = tmp_path / "nested" / "sample.json.zst"

    write_zst_file_to_disk(str(zst_path), b"content")

    assert zst_path.read_bytes() == b"content"


def test_save_zst_file_raises_on_write_failure(tmp_path):
    zst_path = tmp_path / "sample.json.zst"

    with (
        patch("builtins.open", side_effect=OSError("disk full")),
        pytest.raises(ValueError, match=r"Failed to save downloaded '.*sample\.json\.zst'"),
    ):
        write_zst_file_to_disk(str(zst_path), b"content")


def test_write_dataset_to_disk_as_zst_writes_to_explicit_output_dir(tmp_path):
    output_dir = tmp_path / "datasets"

    write_dataset_to_disk_as_zst("algebra", b"content", output_dir=str(output_dir))

    assert (output_dir / "algebra.json.zst").read_bytes() == b"content"


def test_write_dataset_to_disk_as_zst_uses_default_output_dir(tmp_path):
    with patch(
        "hyperbench.utils.file_utils.__file__",
        str(tmp_path / "pkg" / "file_utils.py"),
    ):
        write_dataset_to_disk_as_zst("algebra", b"content")

    assert (tmp_path / "data" / "datasets" / "algebra.json.zst").read_bytes() == b"content"


def test_write_dataset_to_disk_as_zst_raises_when_path_cannot_be_determined():
    with (
        patch("hyperbench.utils.file_utils.os.path.abspath", side_effect=OSError("boom")),
        pytest.raises(ValueError, match="Failed to determine output path for dataset"),
    ):
        write_dataset_to_disk_as_zst("algebra", b"content")


def test_write_dataset_to_disk_as_zst_raises_on_write_failure(tmp_path):
    output_dir = tmp_path / "datasets"

    with (
        patch("builtins.open", side_effect=OSError("disk full")),
        pytest.raises(
            ValueError, match=r"Failed to write file '.*algebra\.json\.zst' to disk '.*datasets'"
        ),
    ):
        write_dataset_to_disk_as_zst("algebra", b"content", output_dir=str(output_dir))
