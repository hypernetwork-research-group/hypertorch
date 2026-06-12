import pytest
import re
from hyperbench.data import AlgebraDataset, get_dataset_by_name, list_datasets
from hyperbench.types import HData
from unittest.mock import patch


def test_list_datasets_returns_correct_names():
    dataset_names = list_datasets()
    assert isinstance(dataset_names, list)
    assert all(isinstance(name, str) for name in dataset_names)
    assert len(dataset_names) > 0


def test_list_datasets_ignores_empty_name():
    names_before = list_datasets()

    class EmptyNameDataset(AlgebraDataset):
        DATASET_NAME = ""

    assert list_datasets() == names_before


def test_list_datasets_rejects_duplicate_name():
    names_before = list_datasets()

    with pytest.raises(ValueError, match=re.compile("Duplicate preloaded dataset name 'algebra'")):

        class DuplicateAlgebraDataset(AlgebraDataset):
            DATASET_NAME = AlgebraDataset.DATASET_NAME

    assert list_datasets() == names_before


def test_load_dataset_rejects_invalid_name():
    with (
        pytest.raises(ValueError, match=re.compile("Invalid dataset name None")),
        patch.object(AlgebraDataset, "DATASET_NAME", None),
    ):
        AlgebraDataset()


def test_load_dataset_rejects_invalid_hf_sha():
    with (
        pytest.raises(ValueError, match=re.compile("Invalid HF_SHA ''")),
        patch.object(AlgebraDataset, "HF_SHA", ""),
    ):
        AlgebraDataset()


def test_get_dataset_by_name_returns_dataset_instance():
    dataset_name = list_datasets()[0]
    expected_hdata = HData.empty()

    with patch(
        "hyperbench.data.supported_datasets.HIFLoader.load_by_name",
        return_value=expected_hdata,
    ):
        dataset = get_dataset_by_name(dataset_name)

    assert dataset_name == getattr(dataset, "DATASET_NAME", None)
    assert dataset.hdata is expected_hdata


def test_get_dataset_by_name_rejects_unknown_name():
    with pytest.raises(ValueError, match=re.compile("Dataset not found: missing-dataset")):
        get_dataset_by_name("missing-dataset")
