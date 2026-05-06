import json
import os
import requests
import tempfile
import torch
import warnings
import zstandard as zstd

from typing import Any, Dict, List, Optional, Literal
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from hyperbench.nn import EnrichmentMode, NodeEnricher, HyperedgeEnricher
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
from hyperbench.data.hif import HIFLoader, HIFProcessor
from hyperbench.nn.enricher import EnrichmentMode, NodeEnricher, HyperedgeEnricher


class Dataset(TorchDataset):
    """
    A dataset class for loading and processing hypergraph data.

    Args:
        DATASET_NAME: Class variable indicating the name of the dataset to load.
        hypergraph: The loaded hypergraph in HIF format. Can be ``None`` if initialized from an HData object.
        hdata: The processed hypergraph data in HData format.
        sampling_strategy: The strategy used for sampling sub-hypergraphs (e.g., by node IDs or hyperedge IDs).
            If not provided, defaults to ``SamplingStrategy.HYPEREDGE``.
    """

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
        return cls(hdata=hdata, sampling_strategy=sampling_strategy)

    @classmethod
    def from_url(
        cls,
        url: str,
        sampling_strategy: SamplingStrategy = SamplingStrategy.HYPEREDGE,
        save_on_disk: bool = False,
    ) -> "Dataset":
        """
        Create a :class:`Dataset` instance by loading a hypergraph from a URL pointing to a .json or .json.zst file in HIF format.

        Args:
            url: The URL to the .json or .json.zst file containing the HIF hypergraph data.
            sampling_strategy: The sampling strategy to use for the dataset. If not provided, defaults to ``SamplingStrategy.HYPEREDGE``.
            save_on_disk: Whether to save the downloaded file on disk.

        Returns:
            The :class:`Dataset` instance with the loaded hypergraph data.
        """
        hdata = HIFLoader.load_from_url(url=url, save_on_disk=save_on_disk)
        dataset = cls.from_hdata(hdata=hdata, sampling_strategy=sampling_strategy)
        return dataset

    @classmethod
    def from_path(
        cls,
        filepath: str,
        sampling_strategy: SamplingStrategy = SamplingStrategy.HYPEREDGE,
    ) -> "Dataset":
        """
        Create a :class:`Dataset` instance by loading a hypergraph from a local file path pointing to a .json or .json.zst file in HIF format.

        Args:
            filepath: The local file path to the .json or .json.zst file containing the HIF hypergraph data.
            sampling_strategy: The sampling strategy to use for the dataset. If not provided, defaults to ``SamplingStrategy.HYPEREDGE``.

        Returns:
            The :class:`Dataset` instance with the loaded hypergraph data.
        """
        hypergraph = HIFLoader.load_from_path(filepath=filepath)
        dataset = cls.from_hdata(hdata=hypergraph, sampling_strategy=sampling_strategy)
        return dataset

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
        return self.__class__(hdata=hdata, sampling_strategy=self.sampling_strategy)

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
        attrs: Dict[str, Any],
        attr_keys: Optional[List[str]] = None,
    ) -> Tensor:
        return HIFProcessor.transform_attrs(attrs, attr_keys)

    def transform_hyperedge_attrs(
        attrs: Dict[str, Any],
        attr_keys: Optional[List[str]] = None,
    ) -> Tensor:
        return HIFProcessor.transform_attrs(attrs, attr_keys)

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

    def __init__(
        self,
        hdata: Optional[HData] = None,
        sampling_strategy: SamplingStrategy = SamplingStrategy.HYPEREDGE,
    ) -> None:
        """
        Initialize the Dataset.

        Args:
            hdata: Optional HData object to initialize the dataset with.
                If provided, the dataset will be initialized with this data instead of loading and processing from HIF. Must be provided if prepare is set to ``False``.
            sampling_strategy: The sampling strategy to use for the dataset. If not provided, defaults to ``SamplingStrategy.HYPEREDGE``.
        """

        self.__sampler = create_sampler_from_strategy(sampling_strategy)
        self.sampling_strategy = sampling_strategy
        self.hdata = hdata if hdata is not None else HData.empty()

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
