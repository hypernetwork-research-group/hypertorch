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
    list_datasets,
)


SUPPORTED_DATASETS = {
    dataset_cls.DATASET_NAME: dataset_cls
    for dataset_cls in (
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
}


@pytest.mark.parametrize(
    "dataset_name",
    [pytest.param(dataset_name, id=dataset_name) for dataset_name in list_datasets()],
)
@pytest.mark.integration
def test_all_supported_datasets_load(dataset_name):
    dataset_cls = SUPPORTED_DATASETS[dataset_name]
    hdata = HIFLoader.load_by_name(
        dataset_name,
        hf_sha=dataset_cls.HF_SHA,
        save_on_disk=False,
    )

    dataset = dataset_cls(hdata=hdata)

    assert dataset.hdata is not None
    assert dataset.hdata.x is not None
    assert dataset.hdata.hyperedge_index.shape[0] == 2
    assert len(dataset) > 0
