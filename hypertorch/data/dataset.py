from __future__ import annotations

import torch

from typing import TYPE_CHECKING, Any
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from hypertorch.types import (
    HData,
    HIFHypergraph,
    Task,
    TaskEnum,
    is_hyperedge_related_task,
    is_node_related_task,
    validate_task,
)
from hypertorch.utils import (
    NodeSpaceFiller,
    NodeSpaceSetting,
)

from hypertorch.data.hif import HIFLoader, HIFProcessor
from hypertorch.data.sampler import (
    BaseSampler,
    SamplingStrategy,
    SamplingStrategyEnum,
    create_sampler_from_strategy,
)
from hypertorch.data.splitter import (
    HyperedgeDatasetSplitter,
    NodeDatasetSplitter,
    SparseHyperedgeDatasetSplitter,
    Splitter,
)

if TYPE_CHECKING:
    from hypertorch.data import (
        EnrichmentMode,
        HyperedgeEnricher,
        NegativeSampler,
        NodeEnricher,
    )


class Dataset(TorchDataset):
    """
    A dataset class for loading and processing hypergraph data.

    Attributes:
        hdata: The hypergraph data stored by the dataset.
        sampling_strategy: The strategy used for sampling sub-hypergraphs.
    """

    def __init__(
        self,
        hdata: HData | None = None,
        hif_hypergraph: HIFHypergraph | None = None,
        sampling_strategy: SamplingStrategy = SamplingStrategyEnum.HYPEREDGE,
        task: Task = TaskEnum.HYPERLINK_PREDICTION,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            hdata: The processed hypergraph data in HData format.
            hif_hypergraph: The original HIF hypergraph. If not provided, defaults to ``None``.
            sampling_strategy: The strategy used for sampling sub-hypergraphs
                (e.g., by node IDs or hyperedge IDs).
                If not provided, defaults to ``SamplingStrategy.HYPEREDGE``.
            task: Learning task used when the HData in input is not provided.
                Defaults to ``"hyperlink-prediction"``.
        """
        self.__sampler: BaseSampler = create_sampler_from_strategy(sampling_strategy)
        self.__hif_hypergraph: HIFHypergraph | None = (
            hif_hypergraph if hif_hypergraph is not None else None
        )

        self.sampling_strategy: SamplingStrategy = sampling_strategy
        self.task: Task = task
        validate_task(self.task)

        self.hdata: HData = hdata if hdata is not None else HData.empty(task=task)
        self.hif_hypergraph: HIFHypergraph | None = hif_hypergraph if hif_hypergraph is not None else None

    @property
    def hif_hypergraph(self) -> HIFHypergraph:
        """
        Return the original HIF hypergraph, if available.

        Returns:
            The HIF hypergraph from which the dataset was created, if available.

        Raises:
            ValueError: If no HIF hypergraph is available. This can be due to the dataset being
                created from a preprocessed HData without the original HIF hypergraph,
                or due to operations like splitting or sampling that do not preserve
                the original HIF hypergraph.
        """
        if self.__hif_hypergraph is None:
            raise ValueError(
                "HIF hypergraph is not available. This may occur if the dataset was created "
                "from a preprocessed HData without providing the original HIF hypergraph. "
                "It can also be a consequence of operations like splitting or sampling, "
                "which do not preserve the original HIF hypergraph."
            )
        return self.__hif_hypergraph

    @hif_hypergraph.setter
    def hif_hypergraph(self, hif_hypergraph: HIFHypergraph | None) -> None:
        self.__hif_hypergraph = hif_hypergraph

    @property
    def is_hyperedge_related_task(self) -> bool:
        """
        Check if the task uses hyperedge-level targets and operations.

        Returns:
            is_hyperedge_related: True if the task is hyperedge-related, False otherwise.
        """
        return is_hyperedge_related_task(self.task)

    @property
    def is_node_related_task(self) -> bool:
        """
        Check if the task uses node-level targets and operations.

        Returns:
            is_node_related: True if the task is node-related, False otherwise.
        """
        return is_node_related_task(self.task)

    @property
    def hif_hypergraph(self) -> HIFHypergraph:
        """
        Return the original HIF hypergraph, if available.

        Returns:
            The HIF hypergraph from which the dataset was created, if available.

        Raises:
            ValueError: If no HIF hypergraph is available. This can be due to the dataset being
                created from a preprocessed HData without the original HIF hypergraph,
                or due to operations like splitting or sampling that do not preserve
                the original HIF hypergraph.
        """
        if self.__hif_hypergraph is None:
            raise ValueError(
                "HIF hypergraph is not available. This may occur if the dataset was created "
                "from a preprocessed HData without providing the original HIF hypergraph. "
                "It can also be a consequence of operations like splitting or sampling, "
                "which do not preserve the original HIF hypergraph."
            )
        return self.__hif_hypergraph

    @hif_hypergraph.setter
    def hif_hypergraph(self, hif_hypergraph: HIFHypergraph | None) -> None:
        self.__hif_hypergraph = hif_hypergraph

    @property
    def is_hyperedge_related_task(self) -> bool:
        """
        Check if the task uses hyperedge-level targets and operations.

        Returns:
            is_hyperedge_related: True if the task is hyperedge-related, False otherwise.
        """
        return is_hyperedge_related_task(self.task)

    @property
    def is_node_related_task(self) -> bool:
        """
        Check if the task uses node-level targets and operations.

        Returns:
            is_node_related: True if the task is node-related, False otherwise.
        """
        return is_node_related_task(self.task)

    def __len__(self) -> int:
        """
        Return the number of sampleable items in the dataset.

        Note:
            The length of the dataset is determined by the sampling strategy. If the strategy is
            based on nodes, the length corresponds to the number of sampleable nodes.
            If the strategy is based on hyperedges, the length corresponds to the
            number of sampleable hyperedges.

        Examples:
            Assuming `sampling_strategy="node"`:
            >>> len(original_dataset)  # Returns the number of total nodes in the dataset
            >>> len(sampled_dataset)   # Returns the number of sampled nodes in the dataset
            ...                        # If sampled nodes are fewer than total nodes,
            ...                        # this will be less than the original dataset length

            Assuming `sampling_strategy="hyperedge"`:
            >>> len(original_dataset)  # Returns the number of total hyperedges in the dataset
            >>> len(sampled_dataset)   # Returns the number of sampled hyperedges in the dataset
            ...                        # If sampled hyperedges are fewer than total hyperedges,
            ...                        # this will be less than the original dataset length

        Returns:
            length: Number of sampleable nodes or hyperedges, depending on the sampling strategy.
        """
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
        hif_hypergraph: HIFHypergraph | None = None,
        sampling_strategy: SamplingStrategy = SamplingStrategyEnum.HYPEREDGE,
        task: Task = TaskEnum.HYPERLINK_PREDICTION,
        hif: HIFHypergraph | None = None,
    ) -> Dataset:
        """
        Create a `Dataset` instance from an `HData` object.

        Args:
            hdata: `HData` object containing the hypergraph data.
            hif_hypergraph: The original HIF hypergraph.
                If not provided, defaults to ``None``.
            sampling_strategy: The sampling strategy to use for the dataset. If not provided,
                defaults to ``SamplingStrategy.HYPEREDGE``.
            task: Learning task used when the HData. If not provided,
                defaults to ``"hyperlink-prediction"``.
            hif: The original hypergraph. If not provided, defaults to ``None``.

        Returns:
            dataset: The `Dataset` instance with the provided `HData`.
        """
        return cls(
            hdata=hdata,
            hif_hypergraph=hif_hypergraph,
            sampling_strategy=sampling_strategy,
            task=task,
        )

    @classmethod
    def from_url(
        cls,
        url: str,
        sampling_strategy: SamplingStrategyEnum = SamplingStrategyEnum.HYPEREDGE,
        task: Task = TaskEnum.HYPERLINK_PREDICTION,
        save_on_disk: bool = False,
    ) -> Dataset:
        """
        Create a `Dataset` instance by loading a hypergraph from a URL pointing to a .json or
        .json.zst file in HIF format.

        Args:
            url: The URL to the .json or .json.zst file containing the HIF hypergraph data.
                sampling_strategy: The sampling strategy to use for the dataset. If not provided,
                defaults to ``SamplingStrategy.HYPEREDGE``.
            task: Learning task used when the HData. If not provided,
                defaults to ``"hyperlink-prediction"``.
            save_on_disk: Whether to save the downloaded file on disk. Defaults to ``False``.

        Returns:
            dataset: The `Dataset` instance with the loaded hypergraph data.
        """
        hdata, hif_hypergraph = HIFLoader.load_from_url(
            url=url,
            task=task,
            save_on_disk=save_on_disk,
        )
        dataset = cls.from_hdata(
            hdata=hdata,
            hif_hypergraph=hif_hypergraph,
            sampling_strategy=sampling_strategy,
            task=task,
        )
        return dataset

    @classmethod
    def from_path(
        cls,
        filepath: str,
        sampling_strategy: SamplingStrategyEnum = SamplingStrategyEnum.HYPEREDGE,
        task: TaskEnum = TaskEnum.HYPERLINK_PREDICTION,
    ) -> Dataset:
        """
        Create a `Dataset` instance by loading a hypergraph from a local file path pointing to a
        .json or .json.zst file in HIF format.

        Args:
            filepath: The local file path to the .json or .json.zst file containing the
                HIF hypergraph data.
            sampling_strategy: The sampling strategy to use for the dataset. If not provided,
                defaults to ``SamplingStrategy.HYPEREDGE``.
            task: Learning task used when the HData. If not provided,
                defaults to ``"hyperlink-prediction"``.

        Returns:
            dataset: The `Dataset` instance with the loaded hypergraph data.
        """
        hdata, hif_hypergraph = HIFLoader.load_from_path(filepath=filepath, task=task)
        dataset = cls.from_hdata(
            hdata=hdata,
            hif_hypergraph=hif_hypergraph,
            sampling_strategy=sampling_strategy,
            task=task,
        )
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
                ``node_space_setting`` is not transductive. Defaults to ``None``.

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
        return self.__class__(
            hdata=hdata,
            sampling_strategy=self.sampling_strategy,
            task=self.task,
        )

    def add_negative_samples(
        self,
        negative_sampler: NegativeSampler,
        seed: int | None = None,
    ) -> Dataset:
        """
        Create a new Dataset with sampled negative hyperedges added.

        Args:
            negative_sampler: Sampler used to generate negative hyperedges from
                this dataset's ``hdata``.
            seed: Optional random seed used for both negative sampling and the final shuffle.
                Defaults to ``None``.

        Returns:
            dataset: A new Dataset instance with positives and sampled negatives.
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
            preserve_global_node_ids: Whether to preserve the global node IDs
                after removing hyperedges. Defaults to ``False``. If ``False``, the global node IDs
                will be reindexed to be contiguous after removing hyperedges.
                If ``True``, the global node IDs will be preserved, which may cause some models
                to raise as they may expect contiguous global node IDs.
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
        sparse_split_hyperedges: bool = False,
        cover_all_nodes_in_train_split: bool = False,
        train_split_idx: int = 0,
        splitter: Splitter[Dataset, Any] | None = None,
    ) -> list[Dataset]:
        """
        Split the dataset based on task into partitions:
        - For node classification, splits are based on nodes and their incident hyperedges.
        - For hyperlink prediction, splits are based on hyperedges and their incident nodes.

        Boundaries are computed using cumulative floor to prevent early splits from
        over-consuming edges. The last split absorbs any rounding remainder.
        In the transductive setting, node-classification splits keep the full node space on
        the train split. Hyperlink-prediction splits keep the full hypergraph as context by
        default and mark supervised hyperedges with ``target_hyperedge_mask``.

        Splits that would end with zero hyperedges are rejected.

        Use ``split_with_ratios`` to also get the final ratios after splitting.

        Examples:
            Transductive split:
            >>> train, test = dataset.split([0.8, 0.2])
            >>> train.hdata.num_nodes == dataset.hdata.num_nodes
            >>> int(train.hdata.target_hyperedge_mask.sum().item()) == len(train)

            Inductive split:
            >>> train, test = dataset.split(
            ...     [0.8, 0.2],
            ...     node_space_setting="inductive",
            ... )
            >>> train.hdata.num_nodes <= dataset.hdata.num_nodes
            >>> test.hdata.num_nodes <= dataset.hdata.num_nodes

        Args:
            ratios: List of floats summing to ``1.0``, e.g., ``[0.8, 0.1, 0.1]``.
            shuffle: Whether to shuffle hyperedges before splitting.
                Defaults to ``False`` for deterministic splits.
            seed: Optional random seed for reproducibility. Ignored if shuffle is set to ``False``.
            node_space_setting: Whether to preserve the full node space in the splits.
                ``transductive`` (default) preserves the full node space on the
                first split. ``inductive`` keeps each split's local node space.
            sparse_split_hyperedges: Whether hyperlink-prediction splits should use the
                sparse split behavior. Defaults to ``False``, which keeps the full
                hypergraph as context in transductive splits and marks supervised hyperedges with
                ``target_hyperedge_mask``.
            cover_all_nodes_in_train_split: Whether a transductive sparse hyperedge split
                should move hyperedges from later splits until every node is
                incident to one of its selected hyperedges. Ratios are approximate
                when this coverage requires moving hyperedges into the first split.
            train_split_idx: The index of the split to treat as the train split. Defaults to ``0``,
                so the first split is the train split that is optionally rebalanced to cover
                all nodes in sparse transductive hyperlink-prediction splits.
                This is used only when ``node_space_setting=="transductive"``
                and ``cover_all_nodes_in_train_split==True``,
                to determine which split should be rebalanced to cover all nodes.
                For the 'inductive' setting, splits are always returned based on
                the provided ratios.
            splitter: Optional dataset splitter. When provided, it owns split
                construction and final-ratio reporting. Defaults to ``None``.

        Returns:
            datasets: List of Dataset objects, one per split, each with contiguous IDs.

        Raises:
            ValueError: If ratios do not sum to ``1.0``, a final split has zero
                hyperedges, if train coverage is requested for dense hyperlink-prediction
                splits, or a requested sparse train-cover split cannot cover the full node
                space. If the task is node-related, raises a ValueError if
                sparse hyperedge splitting or train coverage is requested.
        """
        if splitter is not None:
            return splitter.split(self)

        if ratios is None:
            raise ValueError("'ratios' must be provided when no custom 'splitter' is provided.")

        if self.hdata.is_node_related_task:
            self.__validate_hyperedge_split_only_parameter(
                sparse_split_hyperedges=sparse_split_hyperedges,
                cover_all_nodes_in_train_split=cover_all_nodes_in_train_split,
                train_split_idx=train_split_idx,
            )

            splits, _ = NodeDatasetSplitter(
                node_space_setting=node_space_setting,
                shuffle=shuffle,
                seed=seed,
            ).split(
                to_split=self,
                ratios=ratios,
            )
            return splits

        if not self.hdata.is_hyperedge_related_task:
            raise ValueError(f"Unsupported task category for task={self.hdata.task!r}.")

        hyperedge_splitter_cls = (
            SparseHyperedgeDatasetSplitter if sparse_split_hyperedges else HyperedgeDatasetSplitter
        )
        splits, _ = hyperedge_splitter_cls(
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
        sparse_split_hyperedges: bool = False,
    ) -> tuple[list[Dataset], list[float]]:
        """
        Split the dataset based on task and return the final hyperedge ratios:
        - For node classification, splits are based on nodes and their incident hyperedges.
          For more details, look at the `NodeDatasetSplitter` class.
        - For hyperlink prediction, splits are based on hyperedges and their incident nodes.
          For more details, look at the `HyperedgeDatasetSplitter` class.

        Boundaries are computed using cumulative floor to prevent early splits from
        over-consuming edges. The last split absorbs any rounding remainder.
        In the transductive setting, node-classification splits keep the full node space on
        the train split. Hyperlink-prediction splits keep the full hypergraph as context by
        default and mark supervised hyperedges with ``target_hyperedge_mask``.

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
            sparse_split_hyperedges: Whether hyperlink-prediction splits should use the legacy
                sparse materialization behavior. Defaults to ``False``, which keeps the full
                hypergraph as context in transductive splits and marks supervised hyperedges
                with ``target_hyperedge_mask``.
            cover_all_nodes_in_train_split: Whether a transductive sparse hyperedge split
                should move hyperedges from later splits until every node is
                incident to one of its selected hyperedges.
            train_split_idx: The index of the split to treat as the train split. Defaults to ``0``,
                so the first split is the train split that is optionally rebalanced to cover
                all nodes in sparse transductive hyperlink-prediction splits.
                This is used only when ``node_space_setting=="transductive"``
                and ``cover_all_nodes_in_train_split==True``,
                to determine which split should be rebalanced to cover all nodes.
                For the 'inductive' setting, splits are always returned based on
                the provided ratios.
            seed: Optional random seed for reproducibility. Ignored if ``shuffle``
                is set to ``False``. Defaults to ``None``.

        Returns:
            datasets: List of Dataset instances, one per split, each with contiguous IDs.
            final_ratios: List of floats representing the actual ratios of target hyperedges
                in each split after splitting and any requested rebalancing.

        Raises:
            ValueError: If ratios do not sum to ``1.0``, a final split has zero
                hyperedges, or a requested transductive train-cover split cannot
                cover the full node space. If the task is node-related, raises a ValueError if
                sparse hyperedge splitting or train coverage is requested.
        """
        if self.hdata.is_node_related_task:
            self.__validate_hyperedge_split_only_parameter(
                sparse_split_hyperedges=sparse_split_hyperedges,
                cover_all_nodes_in_train_split=cover_all_nodes_in_train_split,
                train_split_idx=train_split_idx,
            )

            return NodeDatasetSplitter(
                node_space_setting=node_space_setting,
                shuffle=shuffle,
                seed=seed,
            ).split(
                to_split=self,
                ratios=ratios,
            )

        if not self.hdata.is_hyperedge_related_task:
            raise ValueError(f"Unsupported task category for task={self.hdata.task!r}.")

        hyperedge_splitter_cls = (
            SparseHyperedgeDatasetSplitter if sparse_split_hyperedges else HyperedgeDatasetSplitter
        )
        return hyperedge_splitter_cls(
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
        """
        Transform HIF node attributes into a numeric tensor.
        Overload this in case processing node attributes requires custom logic.

        Args:
            attrs: Attributes to transform.
            attr_keys: Optional attribute key order used for consistent output shape.
                Defaults to ``None``.

        Returns:
            attrs: Tensor containing numeric attribute values.
        """
        return HIFProcessor.transform_attrs(attrs, attr_keys)

    def transform_hyperedge_attrs(
        self,
        attrs: dict[str, Any],
        attr_keys: list[str] | None = None,
    ) -> Tensor:
        """
        Transform hyperedge attributes into a numeric tensor.
        Overload this in case processing hyperedge attributes requires custom logic.

        Args:
            attrs: Attributes to transform.
            attr_keys: Optional attribute key order used for consistent output shape.
                Defaults to ``None``.

        Returns:
            attrs: Tensor containing numeric attribute values.
        """
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

    def __validate_hyperedge_split_only_parameter(
        self,
        sparse_split_hyperedges: bool,
        cover_all_nodes_in_train_split: bool,
        train_split_idx: int,
    ):
        if sparse_split_hyperedges:
            raise ValueError(
                "Sparse hyperedge splitting is not applicable to node-related tasks. "
                f"Got sparse_split_hyperedges={sparse_split_hyperedges} "
                f"for task={self.hdata.task!r}."
            )
        if cover_all_nodes_in_train_split:
            raise ValueError(
                "Train coverage is not applicable to node-related tasks. "
                "Do not set 'cover_all_nodes_in_train_split'. "
                f"Got cover_all_nodes_in_train_split={cover_all_nodes_in_train_split} "
                f"for task={self.hdata.task!r}."
            )
        if train_split_idx != 0:
            raise ValueError(
                "Train coverage is not applicable to node-related tasks. "
                f"Got train_split_idx={train_split_idx} (should be omitted or 0) "
                f"for task={self.hdata.task!r}."
            )
