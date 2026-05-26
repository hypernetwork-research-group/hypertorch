import pytest

from hyperbench.data import (
    list_datasets,
    get_dataset_by_name,
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
