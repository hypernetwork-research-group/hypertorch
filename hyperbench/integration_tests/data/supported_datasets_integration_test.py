import os
from unittest.mock import patch

import pytest
import requests

from hyperbench.data import (
    get_dataset_by_name,
    list_datasets,
)


RATE_LIMIT_TERMS = ["429", "rate limit", "too many requests"]


@pytest.mark.flaky(reruns=3, reruns_delay=10, rerun_show_tracebacks=True)
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


@pytest.mark.flaky(reruns=1, reruns_delay=10 * 60, rerun_show_tracebacks=True)
@pytest.mark.parametrize(
    "dataset_name",
    [pytest.param(dataset_name, id=dataset_name) for dataset_name in list_datasets()],
)
@pytest.mark.integration
def test_all_supported_datasets_load_from_hf(dataset_name, request):
    response = requests.Response()
    response.status_code = 404
    response._content = b""
    dataset_path_suffix = os.path.join("hyperbench", "data", "datasets", f"{dataset_name}.json.zst")
    real_path_exists = os.path.exists

    def dataset_file_exists(path):
        if os.fspath(path).endswith(dataset_path_suffix):
            return False
        return real_path_exists(path)

    try:
        with (
            patch("hyperbench.data.hif.os.path.exists", side_effect=dataset_file_exists),
            patch("hyperbench.data.hif.requests.get", return_value=response) as mock_get,
            patch("hyperbench.data.hif.validate_hif_data", return_value=True),
            patch("hyperbench.data.hif.shutil.copyfile") as mock_copyfile,
            pytest.warns(UserWarning, match="GitHub raw download failed"),
        ):
            dataset = get_dataset_by_name(dataset_name)
    except Exception as e:
        message = str(e)

        execution_count = getattr(request.node, "execution_count", 1)
        max_attempts = request.node.get_closest_marker("flaky").kwargs.get("reruns", 0) + 1

        if execution_count == max_attempts and any(
            term in message.lower() for term in RATE_LIMIT_TERMS
        ):
            pytest.skip(f"Skipping {dataset_name} due to Hugging Face rate limit: {message}")

        raise

    mock_get.assert_called_once()
    mock_copyfile.assert_called_once()
    assert dataset.hdata is not None
    assert dataset.hdata.x is not None
    assert dataset.hdata.hyperedge_index.shape[0] == 2
    assert len(dataset) > 0
