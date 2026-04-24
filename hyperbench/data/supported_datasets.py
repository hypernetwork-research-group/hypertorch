from hyperbench.data import HIFLoader
from hyperbench.data.dataset import Dataset
from hyperbench.data.sampling import SamplingStrategy


class PreloadedDataset(Dataset):
    """
    Base class for datasets that use default loading. Subclasses should specify the DATASET_NAME class variable.
    The dataset will be saved on disk after the first load.
    Args:
        hdata: Optional HData object. If None, the dataset will be loaded using the DATASET_NAME.
        sampling_strategy: The sampling strategy to use for this dataset. Default is SamplingStrategy.HYPEREDGE.
    """

    DATASET_NAME = ""

    def __init__(
        self,
        hdata=None,
        sampling_strategy: SamplingStrategy = SamplingStrategy.HYPEREDGE,
    ) -> None:
        super().__init__(hdata=hdata, sampling_strategy=sampling_strategy)
        if hdata is None:
            self.hdata = HIFLoader.load_by_name(self.DATASET_NAME, save_on_disk=True)


class AlgebraDataset(PreloadedDataset):
    DATASET_NAME = "algebra"


class AmazonDataset(PreloadedDataset):
    DATASET_NAME = "amazon"


class ContactHighSchoolDataset(PreloadedDataset):
    DATASET_NAME = "contact-high-school"


class ContactPrimarySchoolDataset(PreloadedDataset):
    DATASET_NAME = "contact-primary-school"


class CoraDataset(PreloadedDataset):
    DATASET_NAME = "cora"


class CourseraDataset(PreloadedDataset):
    DATASET_NAME = "coursera"


class DBLPDataset(PreloadedDataset):
    DATASET_NAME = "dblp"


class EmailEnronDataset(PreloadedDataset):
    DATASET_NAME = "email-Enron"


class EmailW3CDataset(PreloadedDataset):
    DATASET_NAME = "email-W3C"


class GeometryDataset(PreloadedDataset):
    DATASET_NAME = "geometry"


class GOTDataset(PreloadedDataset):
    DATASET_NAME = "got"


class IMDBDataset(PreloadedDataset):
    DATASET_NAME = "imdb"


class MusicBluesReviewsDataset(PreloadedDataset):
    DATASET_NAME = "music-blues-reviews"


class NBADataset(PreloadedDataset):
    DATASET_NAME = "nba"


class NDCClassesDataset(PreloadedDataset):
    DATASET_NAME = "NDC-classes"


class NDCSubstancesDataset(PreloadedDataset):
    DATASET_NAME = "NDC-substances"


class PatentDataset(PreloadedDataset):
    DATASET_NAME = "patent"


class PubmedDataset(PreloadedDataset):
    DATASET_NAME = "pubmed"


class RestaurantReviewsDataset(PreloadedDataset):
    DATASET_NAME = "restaurant-reviews"


class ThreadsAskUbuntuDataset(PreloadedDataset):
    DATASET_NAME = "threads-ask-ubuntu"


class ThreadsMathsxDataset(PreloadedDataset):
    DATASET_NAME = "threads-math-sx"


class TwitterDataset(PreloadedDataset):
    DATASET_NAME = "twitter"


class VegasBarsReviewsDataset(PreloadedDataset):
    DATASET_NAME = "vegas-bars-reviews"


if __name__ == "__main__":
    # test loading each dataset
    for dataset_cls in [
        AlgebraDataset,
        AmazonDataset,
        ContactHighSchoolDataset,
        # ContactPrimarySchoolDataset,
        # CoraDataset,
        # CourseraDataset,
        # DBLPDataset,
        # EmailEnronDataset,
        # EmailW3CDataset,
        # GeometryDataset,
        # GOTDataset,
        # IMDBDataset,
        # MusicBluesReviewsDataset,
        # NBADataset,
        # NDCClassesDataset,
        # NDCSubstancesDataset,
        # PatentDataset,
        # PubmedDataset,
        # RestaurantReviewsDataset,
        # ThreadsAskUbuntuDataset,
        # ThreadsMathsxDataset,
        # TwitterDataset,
        # VegasBarsReviewsDataset,
    ]:
        dataset = dataset_cls()
        print(dataset.hdata.num_nodes, dataset.hdata.num_hyperedges)
