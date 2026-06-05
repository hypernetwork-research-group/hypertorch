from __future__ import annotations

import torch

from typing import TYPE_CHECKING, Any
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from hyperbench.types import HData
from hyperbench.utils import (
    NodeSpaceFiller,
    NodeSpaceSetting,
)

from hyperbench.data.hif import HIFLoader, HIFProcessor
from hyperbench.data.sampler import SamplingStrategy, create_sampler_from_strategy
from hyperbench.data.splitter import DefaultDatasetSplitter, Splitter

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
        sampling_strategy: The strategy used for sampling sub-hypergraphs
            (e.g., by node IDs or hyperedge IDs).
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
                If provided, the dataset will be initialized with this data instead of loading and
                processing from HIF. Must be provided if prepare is set to ``False``.
            sampling_strategy: The sampling strategy to use for the dataset. If not provided,
                defaults to ``SamplingStrategy.HYPEREDGE``.
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
            - Sampling by node IDs, the sub-hypergraph will contain all hyperedges incident to the
            sampled nodes and all nodes incident to those hyperedges.
            - Sampling by hyperedge IDs, the sub-hypergraph will contain all nodes incident to the
            sampled hyperedges.

        Args:
            index: An integer or a list of integers representing node or hyperedge IDs to sample,
                depending on the sampling strategy.

        Returns:
            hdata: An HData instance containing the sampled sub-hypergraph.

        Raises:
            ValueError: If the provided index is invalid (e.g., empty list or list length exceeds
                number of nodes/hyperedges).
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
            sampling_strategy: The sampling strategy to use for the dataset. If not provided,
                defaults to ``SamplingStrategy.HYPEREDGE``.

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
        Create a `Dataset` instance by loading a hypergraph from a URL pointing to a .json or
        .json.zst file in HIF format.

        Args:
            url: The URL to the .json or .json.zst file containing the HIF hypergraph data.
                sampling_strategy: The sampling strategy to use for the dataset. If not provided,
                defaults to ``SamplingStrategy.HYPEREDGE``.
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
        Create a `Dataset` instance by loading a hypergraph from a local file path pointing to a
        .json or .json.zst file in HIF format.

        Args:
            filepath: The local file path to the .json or .json.zst file containing the
                HIF hypergraph data.
            sampling_strategy: The sampling strategy to use for the dataset. If not provided,
                defaults to ``SamplingStrategy.HYPEREDGE``.

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
            enricher: An instance of NodeEnricher to generate structural node features from
                hypergraph topology.
            enrichment_mode: How to combine generated features with existing ``hdata.x``.
                ``concatenate`` appends new features to the existing ones as additional columns.
                ``replace`` substitutes ``hdata.x`` entirely.
                Defaults to ``replace`` if not provided.
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
            ...     fill_value=0.0,  # torch.tensor(0.0) also works and will be broadcast to the
            ...     appropriate shape
            ... )

        Args:
            dataset_with_features: Source dataset providing node features.
            node_space_setting: The setting for the node space, determining how nodes are handled.
                ``transductive`` (default) preserves the full node space of the target dataset.
                ``inductive`` allows the target dataset to have a different node space, filling
                missing features with ``fill_value``.
            fill_value: Scalar or vector used to fill missing node features when
                ``node_space_setting`` is not transductive.

        Raises:
            ValueError: If the source dataset's node features cannot be aligned with the target
                dataset's nodes.
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
        """
        Enrich hyperedge attributes using the provided hyperedge feature enricher.

        Args:
            enricher: An instance of HyperedgeEnricher to generate structural hyperedge
                attributes from hypergraph topology.
            enrichment_mode: How to combine generated attributes with existing
                ``hdata.hyperedge_attr``.
                ``concatenate`` appends new attributes to the existing ones as additional columns.
                ``replace`` substitutes ``hdata.hyperedge_attr`` entirely.
                Defaults to ``replace`` if not provided.
        """
        self.hdata = self.hdata.enrich_hyperedge_attr(enricher, enrichment_mode)

    def enrich_hyperedge_weights(
        self,
        enricher: HyperedgeEnricher,
        enrichment_mode: EnrichmentMode | None = None,
    ) -> None:
        """
        Enrich hyperedge weights using the provided hyperedge weight enricher.

        Args:
            enricher: An instance of HyperedgeEnricher to generate structural hyperedge weights
                from hypergraph topology.
            enrichment_mode: How to combine generated weights with existing
                ``hdata.hyperedge_weights``.
                ``concatenate`` appends new weights to the existing ones as additional columns.
                ``replace`` substitutes ``hdata.hyperedge_weights`` entirely.
                Defaults to ``replace`` if not provided.
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
            negative_sampler: Sampler used to generate negative hyperedges from
                this dataset's ``hdata``.
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

    def remove_hyperedges_with_fewer_than_k_nodes(
        self,
        k: int,
        preserve_global_node_ids: bool = False,
    ) -> None:
        """
        Remove hyperedges that have fewer than k incident nodes.

        Args:
            k: The minimum number of nodes a hyperedge must have to be retained.
            preserve_global_node_ids: Whether to preserve the global node IDs after removing hyperedges. Defaults to ``False``.
                If ``False``, the global node IDs will be reindexed to be contiguous after removing hyperedges.
                If ``True``, the global node IDs will be preserved, which may cause some models to raise
                as they may expect contiguous global node IDs.
        """
        self.hdata = self.hdata.remove_hyperedges_with_fewer_than_k_nodes(
            k, preserve_global_node_ids
        )

    def split(
        self,
        ratios: list[float] | None = None,
        shuffle: bool | None = False,
        seed: int | None = None,
        node_space_setting: NodeSpaceSetting = "transductive",
        cover_all_nodes_in_train_split: bool = False,
        train_split_idx: int = 0,
        splitter: Splitter[Dataset, Any] | None = None,
    ) -> list[Dataset]:
        """
        Split the dataset by hyperedges into partitions with contiguous 0-based hyperedge IDs.

        Boundaries are computed using cumulative floor to prevent early splits from
        over-consuming edges. The last split absorbs any rounding remainder.
        In the transductive setting, the first split keeps the full node space
        and can optionally be rebalanced with real hyperedges from later splits
        to cover every node.

        Splits that would end with zero hyperedges are rejected.

        Use ``split_with_ratios`` to also get the final ratios after splitting.

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
                shuffle: Whether to shuffle hyperedges before splitting. Defaults to ``False``
                for deterministic splits.
            seed: Optional random seed for reproducibility. Ignored if shuffle is set to ``False``.
            node_space_setting: Whether to preserve the full node space in the splits.
                ``transductive`` (default) preserves the full node space on the
                first split. ``inductive`` keeps each split's local node space.
            cover_all_nodes_in_train_split: Whether a transductive first split
                should move hyperedges from later splits until every node is
                incident to one of its selected hyperedges. Ratios are approximate
                when this coverage requires moving hyperedges into the first split.
            train_split_idx: The index of the split to treat as the train split. Defaults to ``0``,
                so the first split is the train split that gets the full node space in the
                transductive setting and is optionally rebalanced to cover all nodes.
                This is used only when ``node_space_setting=="transductive"`` and ``cover_all_nodes_in_train_split==True``,
                to determine which split should be rebalanced to cover all nodes.
                For the 'inductive' setting, splits are always returned based on the provided ratios.
            seed: Optional random seed for reproducibility. Ignored if shuffle is set to ``False``.
            splitter: Optional dataset splitter. When provided, it owns split
                construction and final-ratio reporting.

        Returns:
            datasets: List of Dataset objects, one per split, each with contiguous IDs.

        Raises:
            ValueError: If ratios do not sum to ``1.0``, a final split has zero
                hyperedges, or a requested transductive train-cover split cannot
                cover the full node space.
        """
        if splitter is not None:
            return splitter.split(self)

        if ratios is None:
            raise ValueError("'ratios' must be provided when no custom 'splitter' is provided.")

        splits, _ = DefaultDatasetSplitter(
            node_space_setting=node_space_setting,
            shuffle=shuffle,
            seed=seed,
        ).split(
            to_split=self,
            ratios=ratios,
            train_split_idx=train_split_idx,
            cover_all_nodes_in_train_split=cover_all_nodes_in_train_split,
        )
        return splits

    def split_with_ratios(
        self,
        ratios: list[float],
        shuffle: bool | None = False,
        seed: int | None = None,
        node_space_setting: NodeSpaceSetting = "transductive",
        cover_all_nodes_in_train_split: bool = False,
        train_split_idx: int = 0,
    ) -> tuple[list[Dataset], list[float]]:
        """
        Split the dataset and return the final hyperedge ratios.

        Boundaries are computed using cumulative floor to prevent early splits from
        over-consuming edges. The last split absorbs any rounding remainder.
        In the transductive setting, the first split keeps the full node space
        and can optionally be rebalanced with real hyperedges from later splits
        to cover every node.

        Splits that would end with zero hyperedges are rejected.

        Final ratios are computed from split hyperedge counts after ratio
        boundaries and any requested transductive rebalancing have been applied.

        To provide a custom splitting implementation, use the ``splitter``
        argument of the ``split`` method instead.

        Args:
            ratios: List of floats summing to ``1.0``, e.g., ``[0.8, 0.1, 0.1]``.
            shuffle: Whether to shuffle hyperedges before splitting. Defaults to
                ``False`` for deterministic splits.
            node_space_setting: Whether to preserve the full node space in the
                splits. ``transductive`` (default) preserves the full node space
                on the first split. ``inductive`` keeps each split's local node space.
            cover_all_nodes_in_train_split: Whether a transductive first split
                should move hyperedges from later splits until every node is
                incident to one of its selected hyperedges.
            train_split_idx: The index of the split to treat as the train split. Defaults to ``0``,
                so the first split is the train split that gets the full node space in the
                transductive setting and is optionally rebalanced to cover all nodes.
                This is used only when ``node_space_setting=="transductive"`` and ``cover_all_nodes_in_train_split==True``,
                to determine which split should be rebalanced to cover all nodes.
                For the 'inductive' setting, splits are always returned based on the provided ratios.
            seed: Optional random seed for reproducibility. Ignored if ``shuffle`` is set to ``False``.

        Returns:
            datasets_and_ratios: A tuple containing the split datasets and their
                final hyperedge-count ratios.

        Raises:
            ValueError: If ratios do not sum to ``1.0``, a final split has zero
                hyperedges, or a requested transductive train-cover split cannot
                cover the full node space.
        """
        return DefaultDatasetSplitter(
            node_space_setting=node_space_setting,
            shuffle=shuffle,
            seed=seed,
        ).split(
            to_split=self,
            ratios=ratios,
            train_split_idx=train_split_idx,
            cover_all_nodes_in_train_split=cover_all_nodes_in_train_split,
        )

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

        Fields:
            - ``shape_x``: The shape of the node feature matrix ``x``.
            - ``shape_hyperedge_attr``: The shape of the hyperedge attribute matrix, or ``None``
            if hyperedge attributes are not present.
            - ``num_nodes``: The number of nodes in the hypergraph.
            - ``num_hyperedges``: The number of hyperedges in the hypergraph.
            - ``avg_degree_node_raw``: The average degree of nodes, calculated as the mean number
             of hyperedges each node belongs to.
            - ``avg_degree_node``: The floored node average degree.
            - ``avg_degree_hyperedge_raw``: The average size of hyperedges, calculated as the
            mean number of nodes each hyperedge contains.
            - ``avg_degree_hyperedge``: The floored hyperedge average size.
            - ``node_degree_max``: The maximum degree of any node in the hypergraph.
            - ``hyperedge_degree_max``: The maximum size of any hyperedge in the hypergraph.
            - ``node_degree_median``: The median degree of nodes in the hypergraph.
            - ``hyperedge_degree_median``: The median size of hyperedges in the hypergraph.
            - ``distribution_node_degree``: A list where the value at index ``i`` represents
            the count of nodes with degree ``i``.
            - ``distribution_hyperedge_size``: A list where the value at index ``i`` represents
            the count of hyperedges with size ``i``.
            - ``distribution_node_degree_hist``: A dictionary where the keys are node degrees
             and the values are the count of nodes with that degree.
            - ``distribution_hyperedge_size_hist``: A dictionary where the keys are hyperedge
             sizes and the values are the count of hyperedges with that size.

        Returns:
            stats: A dictionary containing various statistics about the hypergraph.
        """

        return self.hdata.stats()
