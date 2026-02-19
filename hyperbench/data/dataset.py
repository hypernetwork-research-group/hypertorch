import json
import os
import tempfile
import torch
import zstandard as zstd
import requests

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from hyperbench.types import HData, HIFHypergraph
from hyperbench.utils import validate_hif_json, to_0based_ids


class DatasetNames(Enum):
    """
    Enumeration of available datasets.
    """

    ALGEBRA = "algebra"
    AMAZON = "amazon"
    CONTACT_HIGH_SCHOOL = "contact-high-school"
    CONTACT_PRIMARY_SCHOOL = "contact-primary-school"
    CORA = "cora"
    COURSERA = "coursera"
    DBLP = "dblp"
    EMAIL_ENRON = "email-Enron"
    EMAIL_W3C = "email-W3C"
    GEOMETRY = "geometry"
    GOT = "got"
    IMDB = "imdb"
    MUSIC_BLUES_REVIEWS = "music-blues-reviews"
    NBA = "nba"
    NDC_CLASSES = "NDC-classes"
    NDC_SUBSTANCES = "NDC-substances"
    PATENT = "patent"
    PUBMED = "pubmed"
    RESTAURANT_REVIEWS = "restaurant-reviews"
    THREADS_ASK_UBUNTU = "threads-ask-ubuntu"
    THREADS_MATH_SX = "threads-math-sx"
    TWITTER = "twitter"
    VEGAS_BARS_REVIEWS = "vegas-bars-reviews"


class HIFConverter:
    """A utility class to load hypergraphs from HIF format."""

    @staticmethod
    def load_from_hif(dataset_name: Optional[str], save_on_disk: bool = False) -> HIFHypergraph:
        if dataset_name is None:
            raise ValueError(f"Dataset name (provided: {dataset_name}) must be provided.")
        if dataset_name not in DatasetNames.__members__:
            raise ValueError(f"Dataset '{dataset_name}' not found.")

        dataset_name = DatasetNames[dataset_name].value
        current_dir = os.path.dirname(os.path.abspath(__file__))
        zst_filename = os.path.join(current_dir, "datasets", f"{dataset_name}.json.zst")

        if not os.path.exists(zst_filename):
            github_dataset_repo = f"https://github.com/hypernetwork-research-group/datasets/blob/main/{dataset_name}.json.zst?raw=true"

            response = requests.get(github_dataset_repo)
            if response.status_code != 200:
                raise ValueError(
                    f"Failed to download dataset '{dataset_name}' from GitHub. Status code: {response.status_code}"
                )

            if save_on_disk:
                os.makedirs(os.path.join(current_dir, "datasets"), exist_ok=True)
                with open(zst_filename, "wb") as f:
                    f.write(response.content)
            else:
                # Create temporary file for downloaded zst content
                with tempfile.NamedTemporaryFile(
                    mode="wb", suffix=".json.zst", delete=False
                ) as tmp_zst_file:
                    tmp_zst_file.write(response.content)
                    zst_filename = tmp_zst_file.name

        # Decompress the downloaded zst file
        dctx = zstd.ZstdDecompressor()
        with (
            open(zst_filename, "rb") as input_f,
            tempfile.NamedTemporaryFile(mode="wb", suffix=".json", delete=False) as tmp_file,
        ):
            dctx.copy_stream(input_f, tmp_file)
            output = tmp_file.name

        with open(output, "r") as f:
            hiftext = json.load(f)
        if not validate_hif_json(output):
            raise ValueError(f"Dataset '{dataset_name}' is not HIF-compliant.")

        hypergraph = HIFHypergraph.from_hif(hiftext)
        return hypergraph


class Dataset(TorchDataset):
    DATASET_NAME = None

    def __init__(
        self,
        hdata: Optional[HData] = None,
        is_original: Optional[bool] = True,
    ) -> None:
        self.__is_original = is_original
        self.hypergraph: HIFHypergraph = self.download() if hdata is None else None
        self.hdata: HData = self.process() if hdata is None else hdata

    def __len__(self) -> int:
        return self.hdata.num_nodes

    def __getitem__(self, index: int | List[int]) -> HData:
        sampled_node_ids_list = self.__get_node_ids_to_sample(index)
        self.__validate_node_ids(sampled_node_ids_list)

        sampled_hyperedge_index, sampled_node_ids, sampled_hyperedge_ids = (
            self.__sample_hyperedge_index(sampled_node_ids_list)
        )

        new_hyperedge_index = self.__new_hyperedge_index(
            sampled_hyperedge_index, sampled_node_ids, sampled_hyperedge_ids
        )

        new_x = self.hdata.x[sampled_node_ids]
        new_y = self.hdata.y[sampled_hyperedge_ids]

        new_edge_attr = None
        if self.hdata.hyperedge_attr is not None and len(sampled_hyperedge_ids) > 0:
            new_edge_attr = self.hdata.hyperedge_attr[sampled_hyperedge_ids]

        return HData(
            x=new_x,
            hyperedge_index=new_hyperedge_index,
            hyperedge_attr=new_edge_attr,
            num_nodes=len(sampled_node_ids),
            num_hyperedges=len(sampled_hyperedge_ids),
            y=new_y,
        )

    @classmethod
    def from_hdata(cls, hdata: HData) -> "Dataset":
        """
        Create a :class:`Dataset` instance from an :class:`HData` object.

        Args:
            hdata: :class:`HData` object containing the hypergraph data.

        Returns:
            The :class:`Dataset` instance with the provided :class:`HData`.
        """
        return cls(hdata=hdata, is_original=False)

    def download(self) -> HIFHypergraph:
        """
        Load the hypergraph from HIF format using HIFConverter class.
        """
        if not self.__is_original:
            raise ValueError("download can only be called for the original dataset.")

        if hasattr(self, "hypergraph") and self.hypergraph is not None:
            return self.hypergraph
        hypergraph = HIFConverter.load_from_hif(self.DATASET_NAME, save_on_disk=True)
        return hypergraph

    def process(self) -> HData:
        """
        Process the loaded hypergraph into :class:`HData` format, mapping HIF structure to tensors.

        Returns:
            The processed hypergraph data.
        """
        if not self.__is_original:
            raise ValueError("process can only be called for the original dataset.")

        num_nodes = len(self.hypergraph.nodes)
        x = self.__process_x(num_nodes)

        # Remap node IDs to 0-based contiguous IDs (using indices) matching the x tensor order
        node_id_to_idx = {node.get("node"): idx for idx, node in enumerate(self.hypergraph.nodes)}
        # Initialize edge_set only with edges that have incidences, so that
        # we avoid inflating edge count due to isolated nodes/missing incidences
        hyperedge_id_to_idx: Dict[Any, int] = {}

        node_ids = []
        hyperedge_ids = []
        nodes_with_incidences = set()
        for incidence in self.hypergraph.incidences:
            node_id = incidence.get("node", 0)
            hyperedge_id = incidence.get("edge", 0)

            if hyperedge_id not in hyperedge_id_to_idx:
                # Hyperedges start from 0 and are assigned IDs in the order they are first encountered in incidences
                hyperedge_id_to_idx[hyperedge_id] = len(hyperedge_id_to_idx)

            node_ids.append(node_id_to_idx[node_id])
            hyperedge_ids.append(hyperedge_id_to_idx[hyperedge_id])
            nodes_with_incidences.add(node_id_to_idx[node_id])

        # Handle isolated nodes by assigning them to a new unique hyperedge (self-loop)
        for node_idx in range(num_nodes):
            if node_idx not in nodes_with_incidences:
                new_hyperedge_id = len(hyperedge_id_to_idx)
                # Unique dummy key to reserve the index in hyperedge_set
                hyperedge_id_to_idx[f"__self_loop_{node_idx}__"] = new_hyperedge_id
                node_ids.append(node_idx)
                hyperedge_ids.append(new_hyperedge_id)

        num_hyperedges = len(hyperedge_id_to_idx)
        hyperedge_attr = self.__process_hyperedge_attr(hyperedge_id_to_idx, num_hyperedges)

        hyperedge_index = torch.tensor([node_ids, hyperedge_ids], dtype=torch.long)

        return HData(x, hyperedge_index, hyperedge_attr, num_nodes, num_hyperedges)

    def split(
        self,
        ratios: List[float],
        shuffle: Optional[bool] = False,
        seed: Optional[int] = None,
    ) -> List["Dataset"]:
        """
        Split the dataset by hyperedges into partitions with contiguous 0-based IDs.

        Boundaries are computed using cumulative floor to prevent early splits from
        over-consuming edges. The last split absorbs any rounding remainder.

        Examples:
            With ``num_hyperedges = 3`` and ``ratios = [0.5, 0.25, 0.25]``:

            >>> cumulative_ratios = [0.5, 0.75, 1.0]

            Boundaries:

            - ``i=0`` -> ``end = int(0.5 * 3) = 1`` -> slice ``[0:1]`` -> 1 edge
            - ``i=1`` -> ``end = int(0.75 * 3) = 2`` -> slice ``[1:2]`` -> 1 edge
            - ``i=2`` -> ``end = 3`` (clamped) -> slice ``[2:3]`` -> 1 edge

        Args:
            ratios: List of floats summing to ``1.0``, e.g., ``[0.8, 0.1, 0.1]``.
            shuffle: Whether to shuffle hyperedges before splitting. Defaults to ``False`` for deterministic splits.
            seed: Optional random seed for reproducibility. Ignored if shuffle is set to ``False``.

        Returns:
            List of Dataset objects, one per split, each with contiguous IDs.
        """
        # Allow small imprecision in sum of ratios, but raise error if it's significant
        # Example: ratios = [0.8, 0.1, 0.1] -> sum = 1.0 (valid)
        #          ratios = [0.8, 0.1, 0.05] -> sum = 0.95 (invalid, raises ValueError)
        #          ratios = [0.8, 0.1, 0.1, 0.0000001] -> sum = 1.0000001 (valid, allows small imprecision)
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {sum(ratios)}.")

        device = self.hdata.device
        num_hyperedges = self.hdata.num_hyperedges
        hyperedge_ids_permutation = self.__get_hyperedge_ids_permutation(
            num_hyperedges, shuffle, seed
        )

        # Compute cumulative ratio boundaries to avoid independent rounding errors.
        # Independent rounding (e.g., round(0.5*3)=2, round(0.25*3)=1, round(0.25*3)=1 -> total=4)
        # can over-allocate edges to early splits and starve later ones.
        # Cumulative floor boundaries guarantee monotonically increasing cut points.
        # Example: ratios = [0.5, 0.25, 0.25], num_hyperedges = 3
        #          cumulative_ratios = [0.5, 0.75, 1.0]
        cumulative_ratios = []
        cumsum = 0.0
        for ratio in ratios:
            cumsum += ratio
            cumulative_ratios.append(cumsum)

        split_datasets = []
        start = 0
        for i in range(len(ratios)):
            if i == len(ratios) - 1:
                # Last split gets everything remaining, absorbing any rounding remainder
                # Example: start = 2, end = 3 -> permutation[2:3] = [2] (1 edge)
                end = num_hyperedges
            else:
                # Floor of cumulative boundary ensures early splits don't over-consume
                # Example: i=0 -> int(0.5 * 3) = int(1.5) = 1, end = 1
                #          i=1 -> int(0.75 * 3) = int(2.25) = 2, end = 2
                end = int(cumulative_ratios[i] * num_hyperedges)

            # Example: i=0 -> permutation[0:1] = [0] (1 edge)
            #          i=1 -> permutation[1:2] = [1] (1 edge)
            #          i=2 -> permutation[2:3] = [2] (1 edge)
            split_hyperedge_ids = hyperedge_ids_permutation[start:end]
            split_hdata = HData.split(self.hdata, split_hyperedge_ids).to(device=device)
            split_dataset = self.__class__(hdata=split_hdata, is_original=False)
            split_datasets.append(split_dataset)

            start = end

        return split_datasets

    def to(self, device: torch.device) -> "Dataset":
        """
        Move the dataset's HData to the specified device.

        Args:
            device: The target device (e.g., ``torch.device('cuda')`` or ``torch.device('cpu')``).

        Returns:
            A new Dataset instance with HData moved to the specified device.
        """
        self.hdata = self.hdata.to(device)
        return self

    def transform_node_attrs(
        self,
        attrs: Dict[str, Any],
        attr_keys: Optional[List[str]] = None,
    ) -> Tensor:
        return self.transform_attrs(attrs, attr_keys)

    def transform_hyperedge_attrs(
        self,
        attrs: Dict[str, Any],
        attr_keys: Optional[List[str]] = None,
    ) -> Tensor:
        return self.transform_attrs(attrs, attr_keys)

    def transform_attrs(
        self,
        attrs: Dict[str, Any],
        attr_keys: Optional[List[str]] = None,
    ) -> Tensor:
        """
        Extract and encode numeric attributes to tensor.
        Non-numeric attributes are discarded. Missing attributes are filled with ``0.0``.

        Args:
            attrs: Dictionary of attributes
            attr_keys: Optional list of attribute keys to encode. If provided, ensures consistent ordering and fill missing with ``0.0``.

        Returns:
            Tensor of numeric attribute values
        """
        numeric_attrs = {
            key: value
            for key, value in attrs.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }

        if attr_keys is not None:
            values = [float(numeric_attrs.get(key, 0.0)) for key in attr_keys]
            return torch.tensor(values, dtype=torch.float)

        if not numeric_attrs:
            return torch.tensor([], dtype=torch.float)

        values = [float(value) for value in numeric_attrs.values()]
        return torch.tensor(values, dtype=torch.float)

    def __collect_attr_keys(self, attr_keys: List[Dict[str, Any]]) -> List[str]:
        """
        Collect unique numeric attribute keys from a list of attribute dictionaries.

        Args:
            attr_keys: List of attribute dictionaries.

        Returns:
            List of unique numeric attribute keys.
        """
        unique_keys = []
        for attrs in attr_keys:
            for key, value in attrs.items():
                if key not in unique_keys and isinstance(value, (int, float)):
                    unique_keys.append(key)

        return unique_keys

    def __get_hyperedge_ids_permutation(
        self,
        num_hyperedges: int,
        shuffle: Optional[bool],
        seed: Optional[int],
    ) -> Tensor:
        device = self.hdata.device

        # Shuffle hyperedge IDs if shuffle is requested, otherwise keep original order for deterministic splits
        if shuffle:
            generator = torch.Generator(device=device)
            if seed is not None:
                generator.manual_seed(seed)

            random_hyperedge_ids_permutation = torch.randperm(
                n=num_hyperedges,
                generator=generator,
                device=device,
            )
            return random_hyperedge_ids_permutation

        ranged_hyperedge_ids_permutation = torch.arange(num_hyperedges, device=device)
        return ranged_hyperedge_ids_permutation

    def __get_node_ids_to_sample(self, id: int | List[int]) -> List[int]:
        """
        Get a list of node IDs to sample based on the provided index.

        Args:
            id: An integer or a list of integers representing node IDs to sample.

        Returns:
            List of node IDs to sample.

        Raises:
            ValueError: If the provided index is invalid (e.g., empty list or list length exceeds number of nodes).
        """
        if isinstance(id, list):
            if len(id) < 1:
                raise ValueError("Index list cannot be empty.")
            elif len(id) > self.__len__():
                raise ValueError(
                    "Index list length cannot exceed number of nodes in the hypergraph."
                )
            return list(set(id))

        return [id]

    def __new_hyperedge_index(
        self,
        sampled_hyperedge_index: Tensor,
        sampled_node_ids: Tensor,
        sampled_hyperedge_ids: Tensor,
    ) -> Tensor:
        """
        Create new hyperedge_index with 0-based node and hyperedge IDs.

        Args:
            sampled_hyperedge_index: Original hyperedge_index tensor with sampled incidences.
            sampled_node_ids: List of sampled original node IDs.
            sampled_hyperedge_ids: List of sampled original hyperedge IDs.

        Returns:
            New hyperedge_index tensor with 0-based node and edge IDs.
        """
        # Example: sampled_edge_index = [[1, 1, 3],
        #                                [0, 2, 2]]
        #          sampled_node_ids = [1, 3],
        #          sampled_edge_ids = [0, 2]
        #          -> new_node_ids = [0, 0, 1], new_edge_ids = [0, 1, 1]
        new_node_ids = to_0based_ids(sampled_hyperedge_index[0], sampled_node_ids)
        new_hyperedge_ids = to_0based_ids(sampled_hyperedge_index[1], sampled_hyperedge_ids)

        # Example: new_node_ids = [0, 1], new_hyperedge_ids = [0, 1]
        #          -> new_hyperedge_index = [[0, 1],
        #                                    [0, 1]]
        new_hyperedge_index = torch.stack([new_node_ids, new_hyperedge_ids], dim=0)
        return new_hyperedge_index

    def __process_hyperedge_attr(
        self,
        hyperedge_id_to_idx: Dict[Any, int],
        num_hyperedges: int,
    ) -> Optional[Tensor]:
        # hyperedge-attr: shape [num_hyperedges, num_hyperedge_attributes]
        hyperedge_attr = None
        has_hyperedges = self.hypergraph.edges is not None and len(self.hypergraph.edges) > 0
        has_any_hyperedge_attrs = has_hyperedges and any(
            "attrs" in edge for edge in self.hypergraph.edges
        )

        if has_any_hyperedge_attrs:
            hyperedge_id_to_attrs: Dict[Any, Dict[str, Any]] = {
                e.get("edge"): e.get("attrs", {}) for e in self.hypergraph.edges
            }

            hyperedge_attr_keys = self.__collect_attr_keys(list(hyperedge_id_to_attrs.values()))

            # Build attributes in exact order of hyperedge_set indices (0 to num_hyperedges - 1)
            hyperedge_idx_to_id = {idx: id for id, idx in hyperedge_id_to_idx.items()}

            attrs = []
            for hyperedge_idx in range(num_hyperedges):
                hyperedge_id = hyperedge_idx_to_id[hyperedge_idx]

                transformed_attrs = self.transform_hyperedge_attrs(
                    # If it's a real hyperedge, get its attrs; if self-loop, get empty dict
                    attrs=hyperedge_id_to_attrs.get(hyperedge_id, {}),
                    attr_keys=hyperedge_attr_keys,
                )
                attrs.append(transformed_attrs)

            hyperedge_attr = torch.stack(attrs)

        return hyperedge_attr

    def __process_x(self, num_nodes: int) -> Tensor:
        # Collect all attribute keys to have tensors of same size
        node_attr_keys = self.__collect_attr_keys(
            [node.get("attrs", {}) for node in self.hypergraph.nodes]
        )

        if node_attr_keys:
            x = torch.stack(
                [
                    self.transform_node_attrs(node.get("attrs", {}), attr_keys=node_attr_keys)
                    for node in self.hypergraph.nodes
                ]
            )
        else:
            # Fallback to ones if no node features, 1 is better as it can help during
            # training (e.g., avoid zero multiplication), especially in first epochs
            x = torch.ones((num_nodes, 1), dtype=torch.float)

        return x  # shape [num_nodes, num_node_features]

    def __sample_hyperedge_index(
        self,
        sampled_node_ids_list: List[int],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        hyperedge_index = self.hdata.hyperedge_index
        node_ids = hyperedge_index[0]
        hyperedge_ids = hyperedge_index[1]

        sampled_node_ids = torch.tensor(sampled_node_ids_list, device=node_ids.device)

        # Find incidences where the node is in our sampled node set
        # Example: hyperedge_index[0] = [0, 0, 1, 2, 3, 4], sampled_node_ids = [0, 3]
        #          -> node_incidence_mask = [True, True, False, False, True, False]
        node_incidence_mask = torch.isin(node_ids, sampled_node_ids)

        # Get unique hyperedges that have at least one sampled node
        # Example: hyperedge_index[1] = [0, 0, 0, 1, 2, 2], node_incidence_mask = [True, True, False, False, True, False]
        #          -> sampled_hyperedge_ids = [0, 2] as they connect to sampled nodes
        sampled_hyperedge_ids = hyperedge_ids[node_incidence_mask].unique()

        # Find all incidences for the sampled hyperedges (not just sampled nodes)
        # Example: hyperedge_index[1] = [0, 0, 0, 1, 2, 2], sampled_hyperedge_ids = [0, 2]
        #          -> hyperedge_incidence_mask = [True, True, True, False, True, True]
        hyperedge_incidence_mask = torch.isin(hyperedge_ids, sampled_hyperedge_ids)

        # Collect all node IDs that appear in the sampled hyperedges
        # Example: hyperedge_index[0] = [0, 0, 1, 2, 3, 4], hyperedge_incidence_mask = [True, True, True, False, True, True]
        #          -> node_ids_in_sampled_hyperedge = [0, 1, 3, 4]
        node_ids_in_sampled_hyperedge = node_ids[hyperedge_incidence_mask].unique()

        # Keep all incidences belonging to the sampled hyperedges
        # Example: hyperedge_index = [[0, 0, 1, 2, 3, 4],
        #                             [0, 0, 0, 1, 2, 2]],
        #          hyperedge_incidence_mask = [True, True, True, False, True, True]
        #          -> sampled_hyperedge_index = [[0, 0, 1, 3, 4],
        #                                        [0, 0, 0, 2, 2]]
        sampled_hyperedge_index = hyperedge_index[:, hyperedge_incidence_mask]
        return sampled_hyperedge_index, node_ids_in_sampled_hyperedge, sampled_hyperedge_ids

    def __validate_node_ids(self, node_ids: List[int]) -> None:
        """
        Validate that node IDs are within bounds of the hypergraph.

        Args:
            node_ids: List of node IDs to validate.

        Raises:
            IndexError: If any node ID is out of bounds.
        """
        for id in node_ids:
            if id < 0 or id >= self.__len__():
                raise IndexError(f"Node ID {id} is out of bounds (0, {self.__len__() - 1}).")


class AlgebraDataset(Dataset):
    DATASET_NAME = "ALGEBRA"


class AmazonDataset(Dataset):
    DATASET_NAME = "AMAZON"


class ContactHighSchoolDataset(Dataset):
    DATASET_NAME = "CONTACT_HIGH_SCHOOL"


class ContactPrimarySchoolDataset(Dataset):
    DATASET_NAME = "CONTACT_PRIMARY_SCHOOL"


class CoraDataset(Dataset):
    DATASET_NAME = "CORA"


class CourseraDataset(Dataset):
    DATASET_NAME = "COURSERA"


class DBLPDataset(Dataset):
    DATASET_NAME = "DBLP"


class EmailEnronDataset(Dataset):
    DATASET_NAME = "EMAIL_ENRON"


class EmailW3CDataset(Dataset):
    DATASET_NAME = "EMAIL_W3C"


class GeometryDataset(Dataset):
    DATASET_NAME = "GEOMETRY"


class GOTDataset(Dataset):
    DATASET_NAME = "GOT"


class IMDBDataset(Dataset):
    DATASET_NAME = "IMDB"


class MusicBluesReviewsDataset(Dataset):
    DATASET_NAME = "MUSIC_BLUES_REVIEWS"


class NBADataset(Dataset):
    DATASET_NAME = "NBA"


class NDCClassesDataset(Dataset):
    DATASET_NAME = "NDC_CLASSES"


class NDCSubstancesDataset(Dataset):
    DATASET_NAME = "NDC_SUBSTANCES"


class PatentDataset(Dataset):
    DATASET_NAME = "PATENT"


class PubmedDataset(Dataset):
    DATASET_NAME = "PUBMED"


class RestaurantReviewsDataset(Dataset):
    DATASET_NAME = "RESTAURANT_REVIEWS"


class ThreadsAskUbuntuDataset(Dataset):
    DATASET_NAME = "THREADS_ASK_UBUNTU"


class ThreadsMathsxDataset(Dataset):
    DATASET_NAME = "THREADS_MATH_SX"


class TwitterDataset(Dataset):
    DATASET_NAME = "TWITTER"


class VegasBarsReviewsDataset(Dataset):
    DATASET_NAME = "VEGAS_BARS_REVIEWS"
