import pytest
import torch

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
from hyperbench.types import HData
from hyperbench.tests.integration.common import (
    gcn_model,
    hgnn_model,
    hgnnp_model,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated.*:DeprecationWarning"
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


@pytest.fixture
def mock_hdata() -> HData:
    x = torch.ones((3, 1), dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long)
    hyperedge_weights = torch.tensor([0.5, 0.7], dtype=torch.float)
    hyperedge_attr = torch.tensor([[0.5], [0.7], [0.9]], dtype=torch.float)
    return HData(
        x=x,
        hyperedge_index=hyperedge_index,
        hyperedge_weights=hyperedge_weights,
        hyperedge_attr=hyperedge_attr,
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


@pytest.mark.integration
def test_model_gcn():
    gcn_model()


@pytest.mark.integration
def test_model_hgnn():
    hgnn_model()


@pytest.mark.integration
def test_model_hgnnp():
    hgnnp_model()
