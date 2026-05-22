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
    hdata = HIFLoader.load_by_name(
        dataset_cls.DATASET_NAME,
        hf_sha=dataset_cls.HF_SHA,
        save_on_disk=False,
    )

    dataset = dataset_cls(hdata=hdata)

    assert dataset.hdata is not None
    assert dataset.hdata.x is not None
    assert dataset.hdata.hyperedge_index.shape[0] == 2
    assert len(dataset) > 0
