from hyperbench.data import AlgebraDataset, list_datasets


def test_list_datasets():
    dataset_names = list_datasets()

    assert isinstance(dataset_names, list)
    assert all(isinstance(name, str) for name in dataset_names)
    assert len(dataset_names) > 0


def test_list_datasets_filters_non_string_names(monkeypatch):

    original_name = AlgebraDataset.DATASET_NAME
    monkeypatch.setattr(AlgebraDataset, "DATASET_NAME", 123, raising=False)

    dataset_names = list_datasets()

    assert isinstance(dataset_names, list)
    assert all(isinstance(name, str) for name in dataset_names)
    assert 123 not in dataset_names
    assert original_name not in dataset_names
