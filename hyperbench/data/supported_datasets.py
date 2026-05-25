from typing import ClassVar
from hyperbench.data.hif import HIFLoader
from hyperbench.data.dataset import Dataset
from hyperbench.data.sampler import SamplingStrategy


class _PreloadedDataset(Dataset):
    """
    Base class for datasets that use default loading.
    Subclasses should specify the ``DATASET_NAME`` class variable.
    The dataset will be saved on disk after the first load.

    Args:
        hdata: Optional HData object. If ``None``, the dataset will be loaded using the ``DATASET_NAME``.
        sampling_strategy: The sampling strategy to use for this dataset. Default is ``SamplingStrategy.HYPEREDGE``.
    """

    DATASET_NAME: ClassVar[str] = ""
    HF_SHA: ClassVar[str | None] = None

    # Keys are public dataset names, values are the concrete dataset classes.
    _registry: ClassVar[dict[str, type["_PreloadedDataset"]]] = {}

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        dataset_name = cls.DATASET_NAME
        if not dataset_name:
            return

        # No duplicate DATASET_NAME should exist
        existing_cls = _PreloadedDataset._registry.get(dataset_name)
        if existing_cls is not None:
            raise ValueError(
                f"Duplicate preloaded dataset name {dataset_name!r}: "
                f"{existing_cls.__name__} and {cls.__name__}"
            )

        # Register the subclass as soon as Python creates the class object.
        # This happens when the module is imported, before users call list_datasets().
        _PreloadedDataset._registry[dataset_name] = cls

    def __init__(
        self,
        hdata=None,
        sampling_strategy: SamplingStrategy = SamplingStrategy.HYPEREDGE,
    ) -> None:
        self.__validate()
        super().__init__(hdata=hdata, sampling_strategy=sampling_strategy)
        if hdata is None:
            self.hdata = HIFLoader.load_by_name(
                self.DATASET_NAME,
                hf_sha=self.HF_SHA,
                save_on_disk=True,
            )

    def __validate(self) -> None:

        dataset_name = getattr(self, "DATASET_NAME", None)
        if not isinstance(dataset_name, str):
            raise ValueError(
                f"Invalid dataset name {dataset_name!r} for class {self.__class__.__name__}. "
                "DATASET_NAME must be a string."
            )

        hf_sha = getattr(self, "HF_SHA", None)
        if hf_sha is not None and not isinstance(hf_sha, str):
            raise ValueError(
                f"Invalid HF_SHA {hf_sha!r} for class {self.__class__.__name__}. "
                "HF_SHA must be a string or None."
            )


def list_datasets() -> list[str]:
    """Return supported preloaded dataset names in deterministic order."""

    # Iterating over a dict yields its keys.
    # Sorting keeps the public result stable across runs.
    return sorted(_PreloadedDataset._registry)


class AlgebraDataset(_PreloadedDataset):
    DATASET_NAME: ClassVar[str] = "algebra"
    HF_SHA: ClassVar[str | None] = "2bb641461e00c103fb5ef4fe6a30aad42500fc21"


class AmazonDataset(_PreloadedDataset):
    DATASET_NAME = "amazon"
    HF_SHA = "614f75d1847d233ee06da0cc3ee10f51220b8243"


class ContactHighSchoolDataset(_PreloadedDataset):
    DATASET_NAME = "contact-high-school"
    HF_SHA = "b991fde34631a357961a244a5c4d734cf3093199"


class ContactPrimarySchoolDataset(_PreloadedDataset):
    DATASET_NAME = "contact-primary-school"
    HF_SHA = "f6f5453777d1fc62f6305b17d131ec1e32cdbe66"


class CoraDataset(_PreloadedDataset):
    DATASET_NAME = "cora"
    HF_SHA = "91fda9ed324e2cce2430638747e9b032bd9c22ad"


class CourseraDataset(_PreloadedDataset):
    DATASET_NAME = "coursera"
    HF_SHA = "e68679a01af61c43292575839e451eb0bbeee202"


class DBLPDataset(_PreloadedDataset):
    DATASET_NAME = "dblp"
    HF_SHA = "151c360ed77042abebb9709fd3d738763d5c5044"


class EmailEnronDataset(_PreloadedDataset):
    DATASET_NAME = "email-Enron"
    HF_SHA = "05247a5441a6a337cdccee24c0060255815905be"


class EmailW3CDataset(_PreloadedDataset):
    DATASET_NAME = "email-W3C"
    HF_SHA = "18b8c795504388c1d075ffcea7eada281ec5e416"


class GeometryDataset(_PreloadedDataset):
    DATASET_NAME = "geometry"
    HF_SHA = "49a8647d6ff7361485c953949010155b0b522a12"


class GOTDataset(_PreloadedDataset):
    DATASET_NAME = "got"
    HF_SHA = "2efb505e5d82457f6e5ba21820c8d8f2298f0ece"


class IMDBDataset(_PreloadedDataset):
    DATASET_NAME = "imdb"
    HF_SHA = "c3a583313d1611b292933d77e725b11be2c39a05"


class MusicBluesReviewsDataset(_PreloadedDataset):
    DATASET_NAME = "music-blues-reviews"
    HF_SHA = "7d218b727097ed007e7f368ab91c064b3eeff184"


class NBADataset(_PreloadedDataset):
    DATASET_NAME = "nba"
    HF_SHA = "5b3b1c7e425bc407bc0843f443cdf889b51e1ca7"


class NDCClassesDataset(_PreloadedDataset):
    DATASET_NAME = "NDC-classes"
    HF_SHA = "c9bb31897646fb3f964ee4affe126f9885954d92"


class NDCSubstancesDataset(_PreloadedDataset):
    DATASET_NAME = "NDC-substances"
    HF_SHA = "bbdde0839ca5913a2535e6fe3ce397b990803af9"


class PatentDataset(_PreloadedDataset):
    DATASET_NAME = "patent"
    HF_SHA = "608b4fab97d17adbc01b0b4636b060a550231307"


class PubmedDataset(_PreloadedDataset):
    DATASET_NAME = "pubmed"
    HF_SHA = "b8f846a3c812b3b23f10bd69f65f739983f6a390"


class RestaurantReviewsDataset(_PreloadedDataset):
    DATASET_NAME = "restaurant-reviews"
    HF_SHA = "668a90391fcb968c786da7bc9e7bbc55e2832066"


class ThreadsAskUbuntuDataset(_PreloadedDataset):
    DATASET_NAME = "threads-ask-ubuntu"
    HF_SHA = "704c54c7f21b4e313ab6bb50bcd30f58ade469b6"


class ThreadsMathsxDataset(_PreloadedDataset):
    DATASET_NAME = "threads-math-sx"
    HF_SHA = "b024111c16fdb266e159a4c647ff1a31ec40db5b"


class TwitterDataset(_PreloadedDataset):
    DATASET_NAME = "twitter"
    HF_SHA = "d93c55af8e04cf70d65ed0059325009a21699a25"


class VegasBarsReviewsDataset(_PreloadedDataset):
    DATASET_NAME = "vegas-bars-reviews"
    HF_SHA = "4f1e4e4c87957679efc38c05129a694d315a8c9b"
