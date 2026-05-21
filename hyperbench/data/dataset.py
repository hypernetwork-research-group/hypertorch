from __future__ import annotations

import torch

from typing import TYPE_CHECKING, Any
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from hyperbench.types import HData
from hyperbench.utils import (
    NodeSpaceFiller,
    NodeSpaceSetting,
    is_transductive_setting,
)

from hyperbench.data.hif import HIFLoader, HIFProcessor
from hyperbench.data.sampler import SamplingStrategy, create_sampler_from_strategy
from hyperbench.data.splitter import HyperedgeIDSplitter

if TYPE_CHECKING:
    from hyperbench.data import (
        EnrichmentMode,
        HyperedgeEnricher,
        NegativeSampler,
        NodeEnricher,
    )


class Dataset(TorchDataset):
    """
    A dataset class for loading and processing hypergraph data.
    Args:
        hdata: The processed hypergraph data in HData format.
        sampling_strategy: The strategy used for sampling sub-hypergraphs (e.g., by node IDs or hyperedge IDs).
            If not provided, defaults to ``SamplingStrategy.HYPEREDGE``.
    """

    def __init__(
        self,
        hdata: HData | None = None,
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

    def __getitem__(self, index: int | list[int]) -> HData:
        """
        Sample a sub-hypergraph based on the sampling strategy and return it as HData.
        If:
        - Sampling by node IDs, the sub-hypergraph will contain all hyperedges incident to the sampled nodes and all nodes incident to those hyperedges.
        - Sampling by hyperedge IDs, the sub-hypergraph will contain all nodes incident to the sampled hyperedges.

        Args:
            index: An integer or a list of integers representing node or hyperedge IDs to sample, depending on the sampling strategy.

        Returns:
            hdata: An HData instance containing the sampled sub-hypergraph.

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
    ) -> Dataset:
        """
        Create a `Dataset` instance from an `HData` object.

        Args:
            hdata: `HData` object containing the hypergraph data.
            sampling_strategy: The sampling strategy to use for the dataset. If not provided, defaults to ``SamplingStrategy.HYPEREDGE``.

        Returns:
            dataset: The `Dataset` instance with the provided `HData`.
        """
        return cls(hdata=hdata, sampling_strategy=sampling_strategy)

    @classmethod
    def from_url(
        cls,
        url: str,
        sampling_strategy: SamplingStrategy = SamplingStrategy.HYPEREDGE,
        save_on_disk: bool = False,
    ) -> Dataset:
        """
        Create a `Dataset` instance by loading a hypergraph from a URL pointing to a .json or .json.zst file in HIF format.

        Args:
            url: The URL to the .json or .json.zst file containing the HIF hypergraph data.
            sampling_strategy: The sampling strategy to use for the dataset. If not provided, defaults to ``SamplingStrategy.HYPEREDGE``.
            save_on_disk: Whether to save the downloaded file on disk.

        Returns:
            dataset: The `Dataset` instance with the loaded hypergraph data.
        """
        hdata = HIFLoader.load_from_url(url=url, save_on_disk=save_on_disk)
        dataset = cls.from_hdata(hdata=hdata, sampling_strategy=sampling_strategy)
        return dataset

    @classmethod
    def from_path(
        cls,
        filepath: str,
        sampling_strategy: SamplingStrategy = SamplingStrategy.HYPEREDGE,
    ) -> Dataset:
        """
        Create a `Dataset` instance by loading a hypergraph from a local file path pointing to a .json or .json.zst file in HIF format.

        Args:
            filepath: The local file path to the .json or .json.zst file containing the HIF hypergraph data.
            sampling_strategy: The sampling strategy to use for the dataset. If not provided, defaults to ``SamplingStrategy.HYPEREDGE``.

        Returns:
            dataset: The `Dataset` instance with the loaded hypergraph data.
        """
        hypergraph = HIFLoader.load_from_path(filepath=filepath)
        dataset = cls.from_hdata(hdata=hypergraph, sampling_strategy=sampling_strategy)
        return dataset

    def enrich_node_features(
        self,
        enricher: NodeEnricher,
        enrichment_mode: EnrichmentMode | None = None,
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
        dataset_with_features: Dataset,
        node_space_setting: NodeSpaceSetting = "transductive",
        fill_value: NodeSpaceFiller | None = None,
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
        enrichment_mode: EnrichmentMode | None = None,
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
        enrichment_mode: EnrichmentMode | None = None,
    ) -> None:
        """Enrich hyperedge weights using the provided hyperedge weight enricher.

        Args:
            enricher: An instance of HyperedgeEnricher to generate structural hyperedge features from hypergraph topology.
            enrichment_mode: How to combine generated features with existing ``hdata.hyperedge_weights``.
                ``concatenate`` appends new features as additional columns.
                ``replace`` substitutes ``hdata.hyperedge_weights`` entirely.
        """
        self.hdata = self.hdata.enrich_hyperedge_weights(enricher, enrichment_mode)

    def update_from_hdata(self, hdata: HData) -> Dataset:
        """
        Create a `Dataset` instance from an `HData` object.

        Args:
            hdata: `HData` object containing the hypergraph data.

        Returns:
            dataset: The `Dataset` instance with the provided `HData`.
        """
        return self.__class__(hdata=hdata, sampling_strategy=self.sampling_strategy)

    def add_negative_samples(
        self,
        negative_sampler: NegativeSampler,
        seed: int | None = None,
    ) -> Dataset:
        """
        Create a new `Dataset` with sampled negative hyperedges added.

        Args:
            negative_sampler: Sampler used to generate negative hyperedges from this dataset's ``hdata``.
            seed: Optional random seed used for both negative sampling and the final shuffle.

        Returns:
            dataset: A new `Dataset` instance with positives and sampled negatives.
        """
        hdata_with_negatives = self.hdata.clone()
        hdata_with_negatives = hdata_with_negatives.add_negative_samples(
            negative_sampler=negative_sampler,
            seed=seed,
        )
        return self.update_from_hdata(hdata_with_negatives)

    def remove_hyperedges_with_fewer_than_k_nodes(self, k: int) -> None:
        """
        Remove hyperedges that have fewer than k incident nodes.

        Args:
            k: The minimum number of nodes a hyperedge must have to be retained.
        """
        self.hdata = self.hdata.remove_hyperedges_with_fewer_than_k_nodes(k)

    def split(
        self,
        ratios: list[float],
        shuffle: bool | None = False,
        seed: int | None = None,
        node_space_setting: NodeSpaceSetting = "transductive",
    ) -> list[Dataset]:
        """
        Split the dataset by hyperedges into partitions with contiguous 0-based hyperedge IDs.

        Boundaries are computed using cumulative floor to prevent early splits from
        over-consuming edges. The last split absorbs any rounding remainder. In the
        transductive setting, the first split is rebalanced with real hyperedges
        from later splits when needed to cover the full node space.
        Splits that would end with zero hyperedges are rejected.
        Use ``split_with_ratios`` to get the final ratios after splitting.

        Examples:
            Transductive split keeping and covering the full node space on the first split:
            >>> train, test = dataset.split([0.8, 0.2])
            >>> train.hdata.num_nodes == dataset.hdata.num_nodes
            >>> test.hdata.num_nodes <= dataset.hdata.num_nodes

            Inductive split:
            >>> train, test = dataset.split(
            ...     [0.8, 0.2],
            ...     node_space_setting="inductive",
            ... )
            >>> train.hdata.num_nodes <= dataset.hdata.num_nodes
            >>> test.hdata.num_nodes <= dataset.hdata.num_nodes

        Args:
            ratios: List of floats summing to ``1.0``, e.g., ``[0.8, 0.1, 0.1]``.
            shuffle: Whether to shuffle hyperedges before splitting. Defaults to ``False`` for deterministic splits.
            seed: Optional random seed for reproducibility. Ignored if shuffle is set to ``False``.
            node_space_setting: Whether to preserve the full node space in the splits.
                ``transductive`` (default) preserves the full node space on the
                first split and ensures every node is incident to one of its
                selected hyperedges. ``inductive`` keeps each split's local node
                space. Ratios are approximate when transductive coverage requires
                moving hyperedges into the first split.

        Returns:
            datasets: List of Dataset objects, one per split, each with contiguous IDs.

        Raises:
            ValueError: If ratios do not sum to ``1.0``, a final split has zero
                hyperedges, or a transductive first split cannot cover the full
                node space.
        """
        split_datasets, _ = self.split_with_ratios(
            ratios=ratios,
            shuffle=shuffle,
            seed=seed,
            node_space_setting=node_space_setting,
        )
        return split_datasets

    def split_with_ratios(
        self,
        ratios: list[float],
        shuffle: bool | None = False,
        seed: int | None = None,
        node_space_setting: NodeSpaceSetting = "transductive",
    ) -> tuple[list["Dataset"], list[float]]:
        """Split the dataset and return the final hyperedge ratios.

        Final ratios are computed from split hyperedge counts after ratio
        boundaries and any transductive rebalancing have been applied.

        Args:
            ratios: List of floats summing to ``1.0``, e.g., ``[0.8, 0.1, 0.1]``.
            shuffle: Whether to shuffle hyperedges before splitting. Defaults to
                ``False`` for deterministic splits.
            seed: Optional random seed for reproducibility. Ignored if ``shuffle``
                is set to ``False``.
            node_space_setting: Whether to preserve the full node space in the
                splits. ``transductive`` (default) preserves the full node space
                on the first split and may move hyperedges from later splits to
                cover all nodes. ``inductive`` keeps each split's local node space.

        Returns:
            datasets_and_ratios: A tuple containing the split datasets and their
                final hyperedge-count ratios.

        Raises:
            ValueError: If ratios do not sum to ``1.0``, a final split has zero
                hyperedges, or a transductive first split cannot cover the full
                node space.
        """
        # Allow small imprecision in sum of ratios, but raise error if it's significant
        # Example: ratios = [0.8, 0.1, 0.1] -> sum = 1.0 (valid)
        #          ratios = [0.8, 0.1, 0.05] -> sum = 0.95 (invalid, raises ValueError)
        #          ratios = [0.8, 0.1, 0.1, 0.0000001] -> sum = 1.0000001 (valid, allows small imprecision)
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {sum(ratios)}.")
        device = self.hdata.device

        hyperedge_splitter = HyperedgeIDSplitter(self.hdata)
        hyperedge_ids_permutation = hyperedge_splitter.get_hyperedge_ids_permutation(shuffle, seed)
        hyperedge_ids_by_split, final_ratios = hyperedge_splitter.split(
            hyperedge_ids_permutation, ratios
        )
        if is_transductive_setting(node_space_setting):
            hyperedge_ids_by_split, final_ratios = hyperedge_splitter.ensure_split_covers_all_nodes(
                hyperedge_ids_by_split=hyperedge_ids_by_split,
                split_idx=0,
            )
        hyperedge_splitter.validate_splits_have_hyperedges(hyperedge_ids_by_split)

        split_datasets = []
        for split_num, split_hyperedge_ids in enumerate(hyperedge_ids_by_split):
            split_node_space_setting: NodeSpaceSetting = (
                "transductive"
                if is_transductive_setting(node_space_setting) and split_num == 0
                else "inductive"
            )
            split_hdata = HData.split(
                hdata=self.hdata,
                split_hyperedge_ids=split_hyperedge_ids,
                node_space_setting=split_node_space_setting,
            ).to(device=device)

            split_dataset = self.__class__(
                hdata=split_hdata,
                sampling_strategy=self.sampling_strategy,
            )
            split_datasets.append(split_dataset)

        return split_datasets, final_ratios

    def to(self, device: torch.device) -> Dataset:
        """
        Move the dataset's HData to the specified device.

        Args:
            device: The target device (e.g., ``torch.device('cuda')`` or ``torch.device('cpu')``).

        Returns:
            dataset: The Dataset instance moved to the specified device.
        """
        self.hdata = self.hdata.to(device)
        return self

    def transform_node_attrs(
        self,
        attrs: dict[str, Any],
        attr_keys: list[str] | None = None,
    ) -> Tensor:
        return HIFProcessor.transform_attrs(attrs, attr_keys)

    def transform_hyperedge_attrs(
        self,
        attrs: dict[str, Any],
        attr_keys: list[str] | None = None,
    ) -> Tensor:
        return HIFProcessor.transform_attrs(attrs, attr_keys)

    def stats(self) -> dict[str, Any]:
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
            stats: A dictionary containing various statistics about the hypergraph.
        """

        return self.hdata.stats()
