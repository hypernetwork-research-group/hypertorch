from typing import ClassVar
from hypertorch.types import HData, Task, TaskEnum, HIFHypergraph
from hypertorch.data.hif import HIFLoader
from hypertorch.data.dataset import Dataset
from hypertorch.data.sampler import SamplingStrategy, SamplingStrategyEnum


class _PreloadedDataset(Dataset):
    """
    Base class for datasets that use default loading.

    Subclasses should specify the ``DATASET_NAME`` class variable. The dataset will be saved on
    disk after the first load.

    Attributes:
        hdata: Loaded hypergraph data. If ``None``, the dataset will be loaded
            using ``DATASET_NAME``.
    """

    DATASET_NAME: ClassVar[str] = ""
    HF_SHA: ClassVar[str | None] = None

    # Keys are public dataset names, values are the concrete dataset classes.
    _registry: ClassVar[dict[str, type["_PreloadedDataset"]]] = {}

    def __init_subclass__(cls) -> None:
        """
        Register concrete preloaded dataset subclasses by dataset name.

        Raises:
            ValueError: If two subclasses use the same ``DATASET_NAME``.
        """
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
        hdata: HData | None = None,
        hif_hypergraph: HIFHypergraph | None = None,
        sampling_strategy: SamplingStrategy = SamplingStrategyEnum.HYPEREDGE,
        task: Task = TaskEnum.HYPERLINK_PREDICTION,
        save_on_disk: bool = True,
    ) -> None:
        """
        Initialize a preloaded dataset.

        Args:
            hdata: Optional preloaded data. If ``None``, data is loaded from ``DATASET_NAME``.
            hif_hypergraph: Optional preloaded HIF hypergraph. If hdata is ``None``, data is loaded
                from ``DATASET_NAME``.
            sampling_strategy: Sampling strategy used by the dataset.
            task: Task type for the dataset. Defaults to ``"hyperlink-prediction"``.
            save_on_disk: Whether downloaded data should be cached on disk.
        """
        self.__validate()
        super().__init__(
            hdata=hdata,
            hif_hypergraph=hif_hypergraph,
            sampling_strategy=sampling_strategy,
            task=task,
        )
        if hdata is None:
            self.hdata, self.hif_hypergraph = HIFLoader.load_by_name(
                dataset_name=self.DATASET_NAME,
                hf_sha=self.HF_SHA,
                task=task,
                save_on_disk=save_on_disk,
            )
            self.hdata = hdata
            self.hif_hypergraph = hif_hypergraph

    def __validate(self) -> None:
        """
        Validate concrete dataset metadata.

        Raises:
            ValueError: If ``DATASET_NAME`` or ``HF_SHA`` has an invalid value.
        """
        dataset_name = getattr(self, "DATASET_NAME", None)
        if not dataset_name:
            raise ValueError(
                f"Invalid dataset name {dataset_name!r} for class {self.__class__.__name__}. "
                "DATASET_NAME must be a non-empty string."
            )

        hf_sha = getattr(self, "HF_SHA", None)
        if hf_sha is not None and not hf_sha:
            raise ValueError(
                f"Invalid HF_SHA {hf_sha!r} for class {self.__class__.__name__}. "
                "HF_SHA must be None or a non-empty string."
            )


def list_datasets() -> list[str]:
    """
    Return supported preloaded dataset names in deterministic order.
    """
    return sorted(_PreloadedDataset._registry)


def get_dataset_by_name(dataset_name: str) -> Dataset:
    """
    Instantiate a supported dataset by name.

    Args:
        dataset_name: Name returned by ``list_datasets``.

    Returns:
        dataset: Loaded dataset instance.

    Raises:
        ValueError: If the dataset name is not registered.
    """
    dataset_cls = _PreloadedDataset._registry.get(dataset_name)
    if dataset_cls is None:
        raise ValueError(f"Dataset not found: {dataset_name}")
    return dataset_cls()


class AlgebraDataset(_PreloadedDataset):
    """
    Preloaded algebra dataset.
    """

    DATASET_NAME: ClassVar[str] = "algebra"
    HF_SHA: ClassVar[str | None] = "2bb641461e00c103fb5ef4fe6a30aad42500fc21"


class AmazonDataset(_PreloadedDataset):
    """
    Preloaded Amazon reviews dataset.
    """

    DATASET_NAME: ClassVar[str] = "amazon"
    HF_SHA: ClassVar[str | None] = "614f75d1847d233ee06da0cc3ee10f51220b8243"


class ContactHighSchoolDataset(_PreloadedDataset):
    """
    Preloaded high-school contact network dataset.
    """

    DATASET_NAME: ClassVar[str] = "contact-high-school"
    HF_SHA: ClassVar[str | None] = "b991fde34631a357961a244a5c4d734cf3093199"


class ContactPrimarySchoolDataset(_PreloadedDataset):
    """
    Preloaded primary-school contact network dataset.
    """

    DATASET_NAME: ClassVar[str] = "contact-primary-school"
    HF_SHA: ClassVar[str | None] = "f6f5453777d1fc62f6305b17d131ec1e32cdbe66"


class CoraDataset(_PreloadedDataset):
    """
    Preloaded Cora citation dataset.
    """

    DATASET_NAME: ClassVar[str] = "cora"
    HF_SHA: ClassVar[str | None] = "dc0f94770bd4f4f7174fa8d02318435330812b42"


class CourseraDataset(_PreloadedDataset):
    """
    Preloaded Coursera dataset.
    """

    DATASET_NAME: ClassVar[str] = "coursera"
    HF_SHA: ClassVar[str | None] = "e68679a01af61c43292575839e451eb0bbeee202"


class DBLPDataset(_PreloadedDataset):
    """
    Preloaded DBLP co-authorship dataset.
    """

    DATASET_NAME: ClassVar[str] = "dblp"
    HF_SHA: ClassVar[str | None] = "151c360ed77042abebb9709fd3d738763d5c5044"


class EmailEnronDataset(_PreloadedDataset):
    """
    Preloaded Enron email dataset.
    """

    DATASET_NAME: ClassVar[str] = "email-Enron"
    HF_SHA: ClassVar[str | None] = "05247a5441a6a337cdccee24c0060255815905be"


class EmailW3CDataset(_PreloadedDataset):
    """
    Preloaded W3C email dataset.
    """

    DATASET_NAME: ClassVar[str] = "email-W3C"
    HF_SHA: ClassVar[str | None] = "18b8c795504388c1d075ffcea7eada281ec5e416"


class GeometryDataset(_PreloadedDataset):
    """
    Preloaded geometry dataset.
    """

    DATASET_NAME: ClassVar[str] = "geometry"
    HF_SHA: ClassVar[str | None] = "49a8647d6ff7361485c953949010155b0b522a12"


class GOTDataset(_PreloadedDataset):
    """
    Preloaded Game of Thrones dataset.
    """

    DATASET_NAME: ClassVar[str] = "got"
    HF_SHA: ClassVar[str | None] = "2efb505e5d82457f6e5ba21820c8d8f2298f0ece"


class IMDBDataset(_PreloadedDataset):
    """
    Preloaded IMDB dataset.
    """

    DATASET_NAME: ClassVar[str] = "imdb"
    HF_SHA: ClassVar[str | None] = "c3a583313d1611b292933d77e725b11be2c39a05"


class MusicBluesReviewsDataset(_PreloadedDataset):
    """
    Preloaded music blues reviews dataset.
    """

    DATASET_NAME: ClassVar[str] = "music-blues-reviews"
    HF_SHA: ClassVar[str | None] = "7d218b727097ed007e7f368ab91c064b3eeff184"


class NBADataset(_PreloadedDataset):
    """
    Preloaded NBA dataset.
    """

    DATASET_NAME: ClassVar[str] = "nba"
    HF_SHA: ClassVar[str | None] = "5b3b1c7e425bc407bc0843f443cdf889b51e1ca7"


class NDCClassesDataset(_PreloadedDataset):
    """
    Preloaded NDC classes dataset.
    """

    DATASET_NAME: ClassVar[str] = "NDC-classes"
    HF_SHA: ClassVar[str | None] = "c9bb31897646fb3f964ee4affe126f9885954d92"


class NDCSubstancesDataset(_PreloadedDataset):
    """
    Preloaded NDC substances dataset.
    """

    DATASET_NAME: ClassVar[str] = "NDC-substances"
    HF_SHA: ClassVar[str | None] = "bbdde0839ca5913a2535e6fe3ce397b990803af9"


class PatentDataset(_PreloadedDataset):
    """
    Preloaded patent dataset.
    """

    DATASET_NAME: ClassVar[str] = "patent"
    HF_SHA: ClassVar[str | None] = "608b4fab97d17adbc01b0b4636b060a550231307"


class PubmedDataset(_PreloadedDataset):
    """
    Preloaded PubMed dataset.
    """

    DATASET_NAME: ClassVar[str] = "pubmed"
    HF_SHA: ClassVar[str | None] = "b8f846a3c812b3b23f10bd69f65f739983f6a390"


class RestaurantReviewsDataset(_PreloadedDataset):
    """
    Preloaded restaurant reviews dataset.
    """

    DATASET_NAME: ClassVar[str] = "restaurant-reviews"
    HF_SHA: ClassVar[str | None] = "668a90391fcb968c786da7bc9e7bbc55e2832066"


class ThreadsAskUbuntuDataset(_PreloadedDataset):
    """
    Preloaded Ask Ubuntu thread dataset.
    """

    DATASET_NAME: ClassVar[str] = "threads-ask-ubuntu"
    HF_SHA: ClassVar[str | None] = "704c54c7f21b4e313ab6bb50bcd30f58ade469b6"


class ThreadsMathsxDataset(_PreloadedDataset):
    """
    Preloaded Math StackExchange thread dataset.
    """

    DATASET_NAME: ClassVar[str] = "threads-math-sx"
    HF_SHA: ClassVar[str | None] = "b024111c16fdb266e159a4c647ff1a31ec40db5b"


class TwitterDataset(_PreloadedDataset):
    """
    Preloaded Twitter dataset.
    """

    DATASET_NAME: ClassVar[str] = "twitter"
    HF_SHA: ClassVar[str | None] = "d93c55af8e04cf70d65ed0059325009a21699a25"


class VegasBarsReviewsDataset(_PreloadedDataset):
    """
    Preloaded Vegas bars reviews dataset.
    """

    DATASET_NAME: ClassVar[str] = "vegas-bars-reviews"
    HF_SHA: ClassVar[str | None] = "4f1e4e4c87957679efc38c05129a694d315a8c9b"
