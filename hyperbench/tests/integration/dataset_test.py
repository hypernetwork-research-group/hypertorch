import pytest

from hyperbench.data import (
    AlgebraDataset,
    AmazonDataset,
    ContactHighSchoolDataset,
    ContactPrimarySchoolDataset,
    CoraDataset,
    CourseraDataset,
    DBLPDataset,
    EmailEnronDataset,
    EmailW3CDataset,
    GOTDataset,
    GeometryDataset,
    HIFLoader,
    IMDBDataset,
    MusicBluesReviewsDataset,
    NBADataset,
    NDCClassesDataset,
    NDCSubstancesDataset,
    PatentDataset,
    PubmedDataset,
    RestaurantReviewsDataset,
    ThreadsAskUbuntuDataset,
    ThreadsMathsxDataset,
    TwitterDataset,
    VegasBarsReviewsDataset,
)


SUPPORTED_DATASETS = (
    AlgebraDataset,
    AmazonDataset,
    ContactHighSchoolDataset,
    ContactPrimarySchoolDataset,
    CoraDataset,
    CourseraDataset,
    DBLPDataset,
    EmailEnronDataset,
    EmailW3CDataset,
    GeometryDataset,
    GOTDataset,
    IMDBDataset,
    MusicBluesReviewsDataset,
    NBADataset,
    NDCClassesDataset,
    NDCSubstancesDataset,
    PatentDataset,
    PubmedDataset,
    RestaurantReviewsDataset,
    ThreadsAskUbuntuDataset,
    ThreadsMathsxDataset,
    TwitterDataset,
    VegasBarsReviewsDataset,
)


@pytest.mark.parametrize(
    "dataset_cls",
    [pytest.param(dataset_cls, id=dataset_cls.DATASET_NAME) for dataset_cls in SUPPORTED_DATASETS],
)
@pytest.mark.integration
def test_all_supported_datasets_load(dataset_cls):
    # hdata = HIFLoader.load_by_name(
    #     dataset_cls.DATASET_NAME,
    #     hf_sha=dataset_cls.HF_SHA,
    #     save_on_disk=False,
    # )

    # dataset = dataset_cls(hdata=hdata)

<<<<<<< HEAD:hyperbench/tests/integration/dataset_test.py
    # assert dataset.hdata is not None
    # assert dataset.hdata.x is not None
    # assert dataset.hdata.hyperedge_index.shape[0] == 2
    # assert len(dataset) > 0
    pass
=======
    assert dataset.hdata is not None
    assert dataset.hdata.x is not None
    assert dataset.hdata.hyperedge_index.shape[0] == 2
    assert len(dataset) > 0


if __name__ == "__main__":
    for dataset_cls in SUPPORTED_DATASETS:
        print(f"Testing dataset: {dataset_cls.DATASET_NAME}")
        hdata = HIFLoader.load_by_name(
            dataset_cls.DATASET_NAME,
            hf_sha=dataset_cls.HF_SHA,
            save_on_disk=False,
        )
        dataset = dataset_cls(hdata=hdata)
        stats = dataset.stats()
        # pretty print stats
        print(f"Stats for {dataset_cls.DATASET_NAME}:")
        for key, value in stats.items():
            if key.startswith("distribution_"):
                continue
            print(f"  {key}: {value}")
        print("\n")
>>>>>>> 8990b36 (chore: move tests in proper place - add support for parallel testing):hyperbench/tests/integration/integration_dataset_test.py
