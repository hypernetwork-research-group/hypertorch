from unittest.mock import patch

import pytest
import requests
from hypertorch.data import (
    get_dataset_by_name,
    list_datasets,
)

from hypertorch.integration_tests.common import (
    exclude_datasets,
    warn_ci,
)

excluded_datasets = exclude_datasets()

NETWORK_ERROR_TERMS = [
    "429",
    "rate limit",
    "too many requests",
    "connection error",
    "connection refused",
    "connection reset",
    "network error",
    "name resolution",
    "remote disconnected",
    "service unavailable",
    "temporary failure",
    "timed out",
    "unable to locate the file on the hub",
    "unable to find the requested files in the local cache",
    "nodename nor servname provided",
    "cannot send a request",
    "client has been closed",
]


def _exception_chain_text(exception: BaseException) -> str:
    parts: list[str] = []
    current: BaseException | None = exception
    seen_ids: set[int] = set()

    while current is not None and id(current) not in seen_ids:
        seen_ids.add(id(current))
        parts.append(str(current).lower())
        current = current.__cause__ or current.__context__

    return "\n".join(parts)


def _is_network_download_failure(exception: BaseException) -> bool:
    message = _exception_chain_text(exception)
    return any(term in message for term in NETWORK_ERROR_TERMS)


@pytest.mark.flaky(reruns=3, reruns_delay=10, rerun_show_tracebacks=True)
@pytest.mark.parametrize(
    "dataset_name",
    [
        pytest.param(dataset_name, id=f"{dataset_name}")
        for dataset_name in list_datasets()
        if dataset_name not in excluded_datasets
    ],
)
@pytest.mark.integration
def test_all_supported_datasets_load(dataset_name):
    dataset = get_dataset_by_name(dataset_name)

    assert dataset.hdata is not None
    assert dataset.hdata.x is not None
    assert dataset.hdata.hyperedge_index.shape[0] == 2
    assert len(dataset) > 0


@pytest.mark.flaky(reruns=3, reruns_delay=5 * 60, rerun_show_tracebacks=True)
@pytest.mark.integration
def test_all_supported_datasets_load_from_hf(request):
    datasets = list_datasets()
    response = requests.Response()
    response.status_code = 404
    response._content = b""

    for dataset_name in datasets:
        try:
            with (
                patch("hypertorch.data.hif.os.path.exists", return_value=False),
                patch("hypertorch.data.hif.requests.get", return_value=response) as mock_get,
                patch("hypertorch.data.hif.validate_hif_data", return_value=True),
                patch("hypertorch.data.hif.shutil.copyfile") as mock_copyfile,
                pytest.warns(UserWarning, match="GitHub raw download failed"),
            ):
                dataset = get_dataset_by_name(dataset_name)
        except Exception as e:
            message = str(e)
            execution_count = getattr(request.node, "execution_count", 1)

            if _is_network_download_failure(e):
                warn_ci(
                    f"Skipping 'test_all_supported_datasets_load_from_hf' for "
                    f"{dataset_name!r} because an upstream download failed due to network issues."
                )

                if execution_count > 2:
                    pytest.skip(
                        f"Skipping {dataset_name!r} due to "
                        f"Hugging Face network failure: {message!r}"
                    )

            raise

        mock_get.assert_called_once()
        mock_copyfile.assert_called_once()
        assert dataset.hdata is not None
        assert dataset.hdata.x is not None
        assert dataset.hdata.hyperedge_index.shape[0] == 2
        assert len(dataset) > 0
