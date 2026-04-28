import json
import os
import tempfile
import torch
import requests
import warnings
import zstandard as zstd

from enum import Enum
from huggingface_hub import hf_hub_download
from typing import Any, Dict, List, Optional
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from hyperbench.nn.enricher import EnrichmentMode, NodeEnricher, HyperedgeEnricher
from hyperbench.types import HData, HIFHypergraph, HyperedgeIndex
from hyperbench.utils import (
    NodeSpaceAssignment,
    NodeSpaceFiller,
    NodeSpaceSetting,
    is_inductive_setting,
    is_transductive_split,
    validate_hif_json,
)

from hyperbench.data.sampling import SamplingStrategy, create_sampler_from_strategy


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
                warnings.warn(
                    f"GitHub raw download failed for dataset '{dataset_name}' with status code {response.status_code}\n"
                    "Falling back to Hugging Face Hub download for dataset",
                    category=UserWarning,
                    stacklevel=2,
                )

                REPO_ID = f"HypernetworkRG/{dataset_name}"
                FILENAME = f"{dataset_name}.json.zst"

                with tempfile.NamedTemporaryFile(
                    mode="wb", suffix=".json.zst", delete=False
                ) as tmp_hf_file:
                    try:
                        downloaded_path = hf_hub_download(
                            repo_id=REPO_ID,
                            filename=FILENAME,
                            repo_type="dataset",
                        )
                    except Exception as e:
                        raise ValueError(
                            f"Failed to download dataset '{dataset_name}' from GitHub and Hugging Face Hub. GitHub error: {response.status_code} | Hugging Face error: {str(e)}"
                        )
                    with open(downloaded_path, "rb") as hf_file:
                        hf_content = hf_file.read()
                    tmp_hf_file.write(hf_content)

                response._content = hf_content

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
    """
    A dataset class for loading and processing hypergraph data.

    Attributes:
        DATASET_NAME: Class variable indicating the name of the dataset to load.
        hypergraph: The loaded hypergraph in HIF format. Can be ``None`` if initialized from an HData object.
        hdata: The processed hypergraph data in HData format.
        sampling_strategy: The strategy used for sampling sub-hypergraphs (e.g., by node IDs or hyperedge IDs).
            If not provided, defaults to ``SamplingStrategy.HYPEREDGE``.
    """

    DATASET_NAME = None

    def __init__(
        self,
        hdata: Optional[HData] = None,
        sampling_strategy: SamplingStrategy = SamplingStrategy.HYPEREDGE,
        prepare: bool = True,
    ) -> None:
        """
        Initialize the Dataset.

        Args:
            hdata: Optional HData object to initialize the dataset with.
                If provided, the dataset will be initialized with this data instead of loading and processing from HIF. Must be provided if prepare is set to ``False``.
            sampling_strategy: The sampling strategy to use for the dataset. If not provided, defaults to ``SamplingStrategy.HYPEREDGE``.
            prepare: Whether to load and process the original dataset from HIF format.
                If set to ``False``, the dataset will be initialized with the provided hdata instead. Defaults to ``True``.
        """
        self.__is_prepared = prepare
        self.__sampler = create_sampler_from_strategy(sampling_strategy)
        self.sampling_strategy = sampling_strategy

        if self.__is_prepared:
            self.hypergraph = self.download()
            self.hdata = self.process()
        else:
            if hdata is None:
                raise ValueError("hdata must be provided when prepare is set to False.")

            self.hypergraph = HIFHypergraph.empty()
            self.hdata = hdata

    def __len__(self) -> int:
        return self.__sampler.len(self.hdata)

    def __getitem__(self, index: int | List[int]) -> HData:
        """
        Sample a sub-hypergraph based on the sampling strategy and return it as HData.
        If:
        - Sampling by node IDs, the sub-hypergraph will contain all hyperedges incident to the sampled nodes and all nodes incident to those hyperedges.
        - Sampling by hyperedge IDs, the sub-hypergraph will contain all nodes incident to the sampled hyperedges.

        Args:
            index: An integer or a list of integers representing node or hyperedge IDs to sample, depending on the sampling strategy.

        Returns:
            An HData instance containing the sampled sub-hypergraph.

        Raises:
            ValueError: If the provided index is invalid (e.g., empty list or list length exceeds number of nodes/hyperedges).
            IndexError: If any node/hyperedge ID is out of bounds.
        """
        return self.__sampler.sample(index, self.hdata)

    @classmethod
    def from_hdata(
        cls,
        hdata: HData,
        sampling_strategy: SamplingStrategy = SamplingStrategy.HYPEREDGE,
    ) -> "Dataset":
        """
        Create a :class:`Dataset` instance from an :class:`HData` object.

        Args:
            hdata: :class:`HData` object containing the hypergraph data.
            sampling_strategy: The sampling strategy to use for the dataset. If not provided, defaults to ``SamplingStrategy.HYPEREDGE``.

        Returns:
            The :class:`Dataset` instance with the provided :class:`HData`.
        """
        return cls(hdata=hdata, sampling_strategy=sampling_strategy, prepare=False)

    def download(self) -> HIFHypergraph:
        """
        Load the hypergraph from HIF format using HIFConverter class.
        """
        if not self.__is_prepared:
            raise ValueError("download can only be called for the original dataset (prepare=True).")

        if hasattr(self, "hypergraph") and self.hypergraph is not None:
            return self.hypergraph

        return HIFConverter.load_from_hif(self.DATASET_NAME, save_on_disk=True)

    def process(self) -> HData:
        """
        Process the loaded hypergraph into :class:`HData` format, mapping HIF structure to tensors.

        Returns:
            The processed hypergraph data.
        """
        if not self.__is_prepared:
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

        hyperedge_weights = self.__process_hyperedge_weights()

        hyperedge_index = torch.tensor([node_ids, hyperedge_ids], dtype=torch.long)

        return HData(
            x=x,
            hyperedge_index=hyperedge_index,
            hyperedge_weights=hyperedge_weights,
            hyperedge_attr=hyperedge_attr,
            num_nodes=num_nodes,
            num_hyperedges=num_hyperedges,
            global_node_ids=HyperedgeIndex(hyperedge_index).node_ids,
        )

    def enrich_node_features(
        self,
        enricher: NodeEnricher,
        enrichment_mode: Optional[EnrichmentMode] = None,
    ) -> None:
        """
        Enrich node features using the provided node feature enricher.

        Args:
            enricher: An instance of NodeEnricher to generate structural node features from hypergraph topology.
            enrichment_mode: How to combine generated features with existing ``hdata.x``.
                ``concatenate`` appends new features as additional columns.
                ``replace`` substitutes ``hdata.x`` entirely.
        """
        self.hdata = self.hdata.enrich_node_features(enricher, enrichment_mode)

    def enrich_node_features_from(
        self,
        dataset_with_features: "Dataset",
        node_space_setting: NodeSpaceSetting = "transductive",
        fill_value: Optional[NodeSpaceFiller] = None,
    ) -> None:
        """
        Enrich node features from another dataset by copying features by ``global_node_ids``.

        Examples:
            In a transductive setting, the full node space is preserved across datasets:
            >>> val_dataset.enrich_node_features_from(train_dataset)

            In inductive setting, missing node features can be filled with 0.0:
            >>> test_dataset.enrich_node_features_from(
            ...     train_dataset,
            ...     node_space_setting="inductive",
            ...     fill_value=0.0,  # torch.tensor(0.0) also works and will be broadcast to the appropriate shape
            ... )

        Args:
            dataset_with_features: Source dataset providing node features.
            node_space_setting: The setting for the node space, determining how nodes are handled.
                ``transductive`` (default) preserves the full node space of the target dataset.
                ``inductive`` allows the target dataset to have a different node space, filling missing features with ``fill_value``.
            fill_value: Scalar or vector used to fill missing node features when ``node_space_setting`` is not transductive.

        Raises:
            ValueError: If the source dataset's node features cannot be aligned with the target dataset's nodes.
        """
        self.hdata = self.hdata.enrich_node_features_from(
            hdata_with_features=dataset_with_features.hdata,
            node_space_setting=node_space_setting,
            fill_value=fill_value,
        )

    def enrich_hyperedge_attr(
        self,
        enricher: HyperedgeEnricher,
        enrichment_mode: Optional[EnrichmentMode] = None,
    ) -> None:
        """Enrich hyperedge features using the provided hyperedge feature enricher.

        Args:
            enricher: An instance of HyperedgeEnricher to generate structural hyperedge features from hypergraph topology.
            enrichment_mode: How to combine generated features with existing ``hdata.hyperedge_attr``.
                ``concatenate`` appends new features as additional columns.
                ``replace`` substitutes ``hdata.hyperedge_attr`` entirely.
        """
        self.hdata = self.hdata.enrich_hyperedge_attr(enricher, enrichment_mode)

    def enrich_hyperedge_weights(
        self,
        enricher: HyperedgeEnricher,
        enrichment_mode: Optional[EnrichmentMode] = None,
    ) -> None:
        """Enrich hyperedge weights using the provided hyperedge weight enricher.

        Args:
            enricher: An instance of HyperedgeEnricher to generate structural hyperedge features from hypergraph topology.
            enrichment_mode: How to combine generated features with existing ``hdata.hyperedge_weights``.
                ``concatenate`` appends new features as additional columns.
                ``replace`` substitutes ``hdata.hyperedge_weights`` entirely.
        """
        self.hdata = self.hdata.enrich_hyperedge_weights(enricher, enrichment_mode)

    def update_from_hdata(self, hdata: HData) -> "Dataset":
        """
        Create a :class:`Dataset` instance from an :class:`HData` object.

        Args:
            hdata: :class:`HData` object containing the hypergraph data.

        Returns:
            The :class:`Dataset` instance with the provided :class:`HData`.
        """
        return self.__class__(hdata=hdata, sampling_strategy=self.sampling_strategy, prepare=False)

    def remove_hyperedges_with_fewer_than_k_nodes(self, k: int) -> None:
        """
        Remove hyperedges that have fewer than k incident nodes.

        Args:
            k: The minimum number of nodes a hyperedge must have to be retained.
        """
        self.hdata = self.hdata.remove_hyperedges_with_fewer_than_k_nodes(k)

    def split(
        self,
        ratios: List[float],
        shuffle: Optional[bool] = False,
        seed: Optional[int] = None,
        node_space_setting: NodeSpaceSetting = "transductive",
        assign_node_space_to: Optional[NodeSpaceAssignment] = "first",
    ) -> List["Dataset"]:
        """
        Split the dataset by hyperedges into partitions with contiguous 0-based hyperedge IDs.

        Boundaries are computed using cumulative floor to prevent early splits from
        over-consuming edges. The last split absorbs any rounding remainder.

        Examples:
            Transductive split keeping the full node space only on the first split (default):
            >>> train, test = dataset.split([0.8, 0.2])
            >>> train.hdata.num_nodes == dataset.hdata.num_nodes
            >>> test.hdata.num_nodes <= dataset.hdata.num_nodes

            Transductive split keeping the full node space on all splits:
            >>> train, test = dataset.split(
            ...     [0.8, 0.2],
            ...     node_space_setting="transductive",
            ...     assign_node_space_to="all",
            ... )
            >>> train.hdata.num_nodes == dataset.hdata.num_nodes
            >>> test.hdata.num_nodes == dataset.hdata.num_nodes

            Inductive split:
            >>> train, test = dataset.split(
            ...     [0.8, 0.2],
            ...     node_space_setting="inductive",
            ...     assign_node_space_to=None,
            ... )
            >>> train.hdata.num_nodes <= dataset.hdata.num_nodes
            >>> test.hdata.num_nodes <= dataset.hdata.num_nodes

        Args:
            ratios: List of floats summing to ``1.0``, e.g., ``[0.8, 0.1, 0.1]``.
            shuffle: Whether to shuffle hyperedges before splitting. Defaults to ``False`` for deterministic splits.
            seed: Optional random seed for reproducibility. Ignored if shuffle is set to ``False``.
            node_space_setting: Whether to preserve the full node space in the splits.
                ``transductive`` (default) ensures all nodes are present in every split,
                while ``inductive`` allows splits to have disjoint node spaces.
            assign_node_space_to: Which split(s) preserve the full node space when
                ``node_space_setting="transductive"``.
                ``first`` preserves only the first returned split. ``all`` preserves all splits.

        Returns:
            List of Dataset objects, one per split, each with contiguous IDs.
        """
        # Allow small imprecision in sum of ratios, but raise error if it's significant
        # Example: ratios = [0.8, 0.1, 0.1] -> sum = 1.0 (valid)
        #          ratios = [0.8, 0.1, 0.05] -> sum = 0.95 (invalid, raises ValueError)
        #          ratios = [0.8, 0.1, 0.1, 0.0000001] -> sum = 1.0000001 (valid, allows small imprecision)
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {sum(ratios)}.")
        if is_inductive_setting(node_space_setting) and assign_node_space_to is not None:
            raise ValueError(
                "assign_node_space_to can only be provided when node_space_setting='transductive'."
            )

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

            use_transductive_node_space = is_transductive_split(
                node_space_setting, assign_node_space_to, split_num=i
            )
            split_hdata = HData.split(
                self.hdata,
                split_hyperedge_ids,
                node_space_setting="transductive" if use_transductive_node_space else "inductive",
            ).to(device=device)

            split_dataset = self.__class__(
                hdata=split_hdata,
                sampling_strategy=self.sampling_strategy,
                prepare=False,
            )
            split_datasets.append(split_dataset)

            start = end

        return split_datasets

    def to(self, device: torch.device) -> "Dataset":
        """
        Move the dataset's HData to the specified device.

        Args:
            device: The target device (e.g., ``torch.device('cuda')`` or ``torch.device('cpu')``).

        Returns:
            The Dataset instance moved to the specified device.
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

    def __process_hyperedge_attr(
        self,
        hyperedge_id_to_idx: Dict[Any, int],
        num_hyperedges: int,
    ) -> Optional[Tensor]:
        # hyperedge-attr: shape [num_hyperedges, num_hyperedge_attributes]
        hyperedge_attr = None
        has_hyperedges = (
            self.hypergraph.hyperedges is not None and len(self.hypergraph.hyperedges) > 0
        )
        has_any_hyperedge_attrs = has_hyperedges and any(
            "attrs" in edge for edge in self.hypergraph.hyperedges
        )

        if has_any_hyperedge_attrs:
            hyperedge_id_to_attrs: Dict[Any, Dict[str, Any]] = {
                e.get("edge"): e.get("attrs", {}) for e in self.hypergraph.hyperedges
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

    def __process_hyperedge_weights(self) -> Optional[Tensor]:
        # Initialize the hyperedge weights tensor
        hyperedge_weights = None

        has_hyperedge_weights = self.hypergraph.hyperedges is not None and all(
            "weight" in edge for edge in self.hypergraph.hyperedges
        )

        if has_hyperedge_weights:
            weights = [edge.get("weight", 1.0) for edge in self.hypergraph.hyperedges]
            hyperedge_weights = torch.tensor(weights, dtype=torch.float)
        elif (
            has_hyperedge_weights is False
            and self.hypergraph.hyperedges is not None
            and any("weight" in edge for edge in self.hypergraph.hyperedges)
        ):
            raise ValueError(
                "Some hyperedges have weights while others do not. All hyperedges must either have weights or none."
            )

        return hyperedge_weights

    def stats(self) -> Dict[str, Any]:
        """
        Compute statistics for the dataset.
        This method currently delegates to the underlying HData's stats method.
        The fields returned in the dictionary include:
        - ``shape_x``: The shape of the node feature matrix ``x``.
        - ``shape_hyperedge_attr``: The shape of the hyperedge attribute matrix, or ``None`` if hyperedge attributes are not present.
        - ``num_nodes``: The number of nodes in the hypergraph.
        - ``num_hyperedges``: The number of hyperedges in the hypergraph.
        - ``avg_degree_node_raw``: The average degree of nodes, calculated as the mean number of hyperedges each node belongs to.
        - ``avg_degree_node``: The floored node average degree.
        - ``avg_degree_hyperedge_raw``: The average size of hyperedges, calculated as the mean number of nodes each hyperedge contains.
        - ``avg_degree_hyperedge``: The floored hyperedge average size.
        - ``node_degree_max``: The maximum degree of any node in the hypergraph.
        - ``hyperedge_degree_max``: The maximum size of any hyperedge in the hypergraph.
        - ``node_degree_median``: The median degree of nodes in the hypergraph.
        - ``hyperedge_degree_median``: The median size of hyperedges in the hypergraph.
        - ``distribution_node_degree``: A list where the value at index ``i`` represents the count of nodes with degree ``i``.
        - ``distribution_hyperedge_size``: A list where the value at index ``i`` represents the count of hyperedges with size ``i``.
        - ``distribution_node_degree_hist``: A dictionary where the keys are node degrees and the values are the count of nodes with that degree.
        - ``distribution_hyperedge_size_hist``: A dictionary where the keys are hyperedge sizes and the values are the count of hyperedges with that size.

        Returns:
            A dictionary containing various statistics about the hypergraph.
        """

        return self.hdata.stats()
