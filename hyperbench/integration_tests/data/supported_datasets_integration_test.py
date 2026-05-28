from unittest.mock import patch

import pytest
import requests

from hyperbench.data import (
    get_dataset_by_name,
    list_datasets,
)


@pytest.mark.parametrize(
    "dataset_name",
    [pytest.param(dataset_name, id=dataset_name) for dataset_name in list_datasets()],
)
@pytest.mark.integration
def test_all_supported_datasets_load(dataset_name):
    dataset = get_dataset_by_name(dataset_name)

    assert dataset.hdata is not None
    assert dataset.hdata.x is not None
    assert dataset.hdata.hyperedge_index.shape[0] == 2
    assert len(dataset) > 0


@pytest.mark.parametrize(
    "dataset_name",
    [pytest.param(dataset_name, id=dataset_name) for dataset_name in list_datasets()],
)
@pytest.mark.integration
def test_all_supported_datasets_load_from_hf(dataset_name):
    response = requests.Response()
    response.status_code = 404
    response._content = b""

    with (
        patch("hyperbench.data.hif.os.path.exists", return_value=False),
        patch("hyperbench.data.hif.requests.get", return_value=response) as mock_get,
        patch("hyperbench.data.hif.validate_hif_data", return_value=True),
        patch("hyperbench.data.hif.shutil.copyfile") as mock_copyfile,
        pytest.warns(UserWarning, match="GitHub raw download failed"),
    ):
        dataset = get_dataset_by_name(dataset_name)

    mock_get.assert_called_once()
    mock_copyfile.assert_called_once()
    assert dataset.hdata is not None
    assert dataset.hdata.x is not None
    assert dataset.hdata.hyperedge_index.shape[0] == 2
    assert len(dataset) > 0


if __name__ == "__main__":
    response = requests.Response()
    response.status_code = 404
    response._content = b""

    with (
        patch("hyperbench.data.hif.os.path.exists", return_value=False),
        patch("hyperbench.data.hif.requests.get", return_value=response) as mock_get,
        patch("hyperbench.data.hif.validate_hif_data", return_value=True),
        patch("hyperbench.data.hif.shutil.copyfile") as mock_copyfile,
        pytest.warns(UserWarning, match="GitHub raw download failed"),
    ):
        dataset = get_dataset_by_name("algebra")
    # from huggingface_hub import scan_cache_dir, constants
    # cache_dir=constants.HF_HUB_CACHE
    # print(f"Scanning Hugging Face Hub cache directory at {cache_dir!r}...")
    # hf_cache_info = scan_cache_dir()
