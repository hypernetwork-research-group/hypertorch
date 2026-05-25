import pytest
from hyperbench.data import AlgebraDataset, list_datasets
from unittest.mock import patch


def test_list_datasets_returns_correct_names():
    dataset_names = list_datasets()
    assert isinstance(dataset_names, list)
    assert all(isinstance(name, str) for name in dataset_names)
    assert len(dataset_names) > 0


def test_preloaded_dataset_ignores_empty_name():
    names_before = list_datasets()

    class EmptyNameDataset(AlgebraDataset):
        DATASET_NAME = ""

    assert list_datasets() == names_before


def test_preloaded_dataset_rejects_duplicate_name():
    names_before = list_datasets()

    with pytest.raises(ValueError, match=r"Duplicate preloaded dataset name 'algebra'"):

        class DuplicateAlgebraDataset(AlgebraDataset):
            DATASET_NAME = AlgebraDataset.DATASET_NAME

    assert list_datasets() == names_before


def test_load_dataset_with_invalid_name():
    with (
        pytest.raises(ValueError, match=r"Invalid dataset name None"),
        patch.object(AlgebraDataset, "DATASET_NAME", None),
    ):
        AlgebraDataset()


def test_load_dataset_with_invalid_hf_sha():
    with (
        pytest.raises(ValueError, match=r"Invalid HF_SHA 12345"),
        patch.object(AlgebraDataset, "HF_SHA", 12345),
    ):
        AlgebraDataset()
