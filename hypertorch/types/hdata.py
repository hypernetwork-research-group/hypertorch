from __future__ import annotations

import torch

from torch import Tensor
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, get_args
from collections.abc import Sequence
from hypertorch.utils import (
    NodeSpaceFiller,
    NodeSpaceSetting,
    StrEnum,
    clone_optional_tensor,
    create_seeded_torch_generator,
    empty_hyperedgeindex,
    empty_nodefeatures,
    is_inductive_setting,
    is_transductive_setting,
    validate_floating_tensor_dtype,
    validate_is_non_empty,
    validate_is_non_negative,
    validate_is_positive,
    validate_long_tensor_dtype,
    validate_node_space_setting,
)

from hypertorch.types.hypergraph import HyperedgeIndex

if TYPE_CHECKING:
    from hypertorch.data import (
        EnrichmentMode,
        HyperedgeEnricher,
        NegativeSampler,
        NodeEnricher,
        Splitter,
    )


class TaskEnum(StrEnum):
    HYPERLINK_PREDICTION = "hyperlink-prediction"
    NODE_CLASSIFICATION = "node-classification"


TaskLiteral: TypeAlias = Literal["hyperlink-prediction", "node-classification"]


Task: TypeAlias = TaskEnum | TaskLiteral


class HData:
    """
    Class for representing hypergraph data in a format suitable for hypergraph learning tasks.

    Examples:
        >>> x = torch.randn(10, 16)  # 10 nodes with 16 features each
        >>> hyperedge_index = torch.tensor([[0, 0, 1, 1, 1],  # node IDs
        ...                                 [0, 1, 2, 3, 4]]) # hyperedge IDs
        >>> data = HData(x=x, hyperedge_index=hyperedge_index)

    Attributes:
        x: Node feature matrix of shape ``[num_nodes, num_features]``.
        hyperedge_index: Hyperedge connectivity in COO format of shape ``[2, num_incidences]``,
            where ``hyperedge_index[0]`` contains node IDs and ``hyperedge_index[1]``
            contains hyperedge IDs.
        hyperedge_weights: Optional tensor of shape ``[num_hyperedges]`` containing weights
            for each hyperedge.
        hyperedge_attr: Hyperedge feature matrix of
            shape ``[num_hyperedges, num_hyperedge_features]``.
            Features associated with each hyperedge (e.g., weights, timestamps, types).
        num_nodes: Number of nodes in the hypergraph. If ``None``, inferred as ``x.size(0)``.
        num_hyperedges: Number of hyperedges in the hypergraph.
            If ``None``, inferred as the number of unique hyperedge IDs in ``hyperedge_index[1]``.
        global_node_ids: Optional stable node IDs of shape ``[num_nodes]`` matching the
            row order of ``x``. Use this to preserve access to the canonical node space
            when ``hyperedge_index`` is rebased locally.
            If ``None``, defaults to ``torch.arange(num_nodes)``, assuming that these are the
            global node IDs in the same order as the rows of ``x``.
        target_node_mask: Optional boolean tensor of shape ``[num_nodes]`` identifying the
            supervised nodes for node-classification splits and batches.
        y: Labels for hyperedges, of shape ``[num_hyperedges]``.
            Used for supervised learning tasks. For unsupervised tasks, this can be ignored.
            Default is a tensor of ones, indicating all hyperedges are positive examples.
        task: Learning task used for sampling and collation on this HData instance.
        device: Device shared by all tensors in the instance.
    """

    def __init__(
        self,
        x: Tensor,
        hyperedge_index: Tensor,
        hyperedge_weights: Tensor | None = None,
        hyperedge_attr: Tensor | None = None,
        num_nodes: int | None = None,
        num_hyperedges: int | None = None,
        global_node_ids: Tensor | None = None,
        target_node_mask: Tensor | None = None,
        y: Tensor | None = None,
        task: Task = TaskEnum.HYPERLINK_PREDICTION,
    ):
        """
        Initialize hypergraph learning data.

        Args:
            x: Node feature matrix of shape ``[num_nodes, num_features]``.
            hyperedge_index: Hyperedge connectivity in COO format of shape
                ``[2, num_incidences]``.
            hyperedge_weights: Optional tensor of shape ``[num_hyperedges]`` containing weights
                for each hyperedge.
            hyperedge_attr: Optional hyperedge feature matrix of shape
                ``[num_hyperedges, num_hyperedge_features]``.
            num_nodes: Optional explicit number of nodes.
            num_hyperedges: Optional explicit number of hyperedges.
            global_node_ids: Optional stable node IDs matching rows of ``x``.
            target_node_mask: Optional supervised node mask for node classification.
            y: Optional labels for hyperedges.
            task: Learning task. Defaults to ``"hyperlink-prediction"``.
        """
        self.x: Tensor = x
        self.hyperedge_index: Tensor = hyperedge_index
        self.__validate_x_and_hyperedge_index_type_and_dim()

        self.hyperedge_weights: Tensor | None = hyperedge_weights
        self.hyperedge_attr: Tensor | None = hyperedge_attr

        hyperedge_index_wrapper = HyperedgeIndex(hyperedge_index)
        self.num_nodes: int = (
            num_nodes
            if num_nodes is not None
            # There should never be isolated nodes when HData is created by Dataset
            # as each isolated node gets its own self-loop hyperedge
            else hyperedge_index_wrapper.num_nodes_if_isolated_exist(num_nodes=x.size(0))
        )
        validate_is_non_negative("num_nodes", self.num_nodes)

        self.num_hyperedges: int = (
            num_hyperedges if num_hyperedges is not None else hyperedge_index_wrapper.num_hyperedges
        )
        validate_is_non_negative("num_hyperedges", self.num_hyperedges)

        self.global_node_ids: Tensor = (
            # torch.arange is to handle isolated nodes, as they are already considered
            # when computing self.num_nodes via num_nodes_if_isolated_exist
            global_node_ids
            if global_node_ids is not None
            else torch.arange(self.num_nodes, dtype=torch.long, device=self.x.device)
        )

        self.target_node_mask: Tensor = (
            target_node_mask
            if target_node_mask is not None
            else torch.ones(self.num_nodes, dtype=torch.bool, device=self.x.device)
        )

        self.task: Task = task
        self.y = self.__y(y)

        self.__validate()

        self.device: torch.device = self.get_device_if_all_consistent()

    def __repr__(self) -> str:
        """
        Return a shape-oriented representation of the data object.

        Returns:
            representation: Human-readable summary of tensor shapes and counts.
        """
        hyperedge_weights_shape = (
            self.hyperedge_weights.shape if self.hyperedge_weights is not None else None
        )
        hyperedge_attr_shape = (
            self.hyperedge_attr.shape if self.hyperedge_attr is not None else None
        )
        target_node_mask_shape = (
            str(self.target_node_mask.shape)
            if self.is_node_related_task
            else f"(ignored for task={self.task!r})"
        )

        return (
            f"{self.__class__.__name__}(\n"
            f"    num_nodes={self.num_nodes},\n"
            f"    num_hyperedges={self.num_hyperedges},\n"
            f"    x_shape={self.x.shape},\n"
            f"    global_node_ids_shape={self.global_node_ids.shape},\n"
            f"    target_node_mask_shape={target_node_mask_shape},\n"
            f"    hyperedge_index_shape={self.hyperedge_index.shape},\n"
            f"    hyperedge_weights_shape={hyperedge_weights_shape},\n"
            f"    hyperedge_attr_shape={hyperedge_attr_shape},\n"
            f"    y_shape={self.y.shape},\n"
            f"    task={self.task!r},\n"
            f"    device={self.device}\n"
            f")"
        )

    @classmethod
    def cat_same_node_space(
        cls,
        hdatas: Sequence[HData],
        x: Tensor | None = None,
        global_node_ids: Tensor | None = None,
    ) -> HData:
        """
        Concatenate `HData` instances that share the same node space, meaning nodes with
        the same ID in different instances are the same node.
        This is useful when combining positive and negative hyperedges that reference
        the same set of nodes.

        Notes:
            - ``x`` is derived from the instance with the largest number of nodes,
                if not provided explicitly.
                If there are conflicting features for the same node ID across instances,
                the features from the instance with the largest number of nodes will be used.
                If ``global_node_ids`` is provided explicitly, ``x`` must also be provided
                to ensure consistency.
            - ``hyperedge_index`` is the concatenation of all input hyperedge indices.
            - ``hyperedge_weights`` is the concatenation of all input hyperedge weights, if present.
                If some instances have hyperedge weights and others do not, the resulting
                ``hyperedge_weights`` will be set to ``None``.
            - ``hyperedge_attr`` is the concatenation of all input hyperedge attributes, if present.
                If some instances have hyperedge attributes and others do not, the resulting
                ``hyperedge_attr`` will be set to ``None``.
            - ``global_node_ids`` is derived from the instance with the largest number of nodes,
                if not provided explicitly.
                If ``x`` is provided explicitly, ``global_node_ids`` must be provided explicitly
                as well to ensure consistency.
            - ``y`` is the concatenation of all input labels.

        Examples:
            >>> x = torch.randn(5, 8)
            >>> pos = HData(x=x, hyperedge_index=torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 2, 2]]))
            >>> neg = HData(x=x, hyperedge_index=torch.tensor([[0, 2], [3, 3]]))
            >>> new = HData.cat_same_node_space([pos, neg])
            >>> new.num_nodes  # 5 — nodes [0, 1, 2, 3, 4]
            >>> new.num_hyperedges  # 4 — hyperedges [0, 1, 2, 3]

        Args:
            hdatas: One or more `HData` instances sharing the same node space.
            x: Optional node feature matrix to use for the resulting `HData`.
                If ``None``, the node features from the instance with the largest number of
                nodes will be used.
                If ``global_node_ids`` is provided explicitly, ``x`` must also be provided
                to ensure consistency. Defaults to ``None``.
            global_node_ids: Optional global node IDs for the resulting `HData`.
                If ``None``, the global node IDs from the instance with the largest number of
                nodes will be used. If ``x`` is provided explicitly, ``global_node_ids`` must
                also be provided to ensure consistency.
                If ``x`` is provided and there is no need for ``global_node_ids`` to preserve
                access to the canonical node space, it is recommended to use arbitrary global node
                IDs that are consistent with the feature rows of ``x``.
                For example, ``global_node_ids=torch.arange(x.size(0))``).
                Defaults to ``None``.

        Returns:
            hdata: A new `HData` with shared nodes and concatenated hyperedges.

        Raises:
            ValueError: If no HData instances are provided, if there are overlapping
                hyperedge IDs across instances,
                or if ``x`` and ``global_node_ids`` are not both provided when one of
                them is provided.
        """
        cls.__validate_can_perform_cat_same_node_space(hdatas, x, global_node_ids)

        hdata_with_largest_node_space = max(hdatas, key=lambda hdata: hdata.num_nodes)

        new_x = (x.clone() if x is not None else hdata_with_largest_node_space.x).clone()
        new_global_node_ids = (
            global_node_ids.clone()
            if global_node_ids is not None
            else hdata_with_largest_node_space.global_node_ids.clone()
        )
        new_target_node_mask = (
            hdata_with_largest_node_space.target_node_mask.clone()
            if hdata_with_largest_node_space.target_node_mask is not None
            else None
        )
        new_y = (
            # For node-based tasks, we must preserve the labels for the entire node space
            hdata_with_largest_node_space.y.clone()
            if hdata_with_largest_node_space.is_node_related_task
            else torch.cat([hdata.y for hdata in hdatas], dim=0)
        )
        new_hyperedge_index = torch.cat([hdata.hyperedge_index for hdata in hdatas], dim=1)

        hyperedge_attrs = []
        hyperedge_weights = []
        have_all_hyperedge_attr = all(hdata.hyperedge_attr is not None for hdata in hdatas)
        have_all_hyperedge_weights = all(hdata.hyperedge_weights is not None for hdata in hdatas)
        for hdata in hdatas:
            if have_all_hyperedge_attr and hdata.hyperedge_attr is not None:
                hyperedge_attrs.append(hdata.hyperedge_attr)
            if have_all_hyperedge_weights and hdata.hyperedge_weights is not None:
                hyperedge_weights.append(hdata.hyperedge_weights)
        new_hyperedge_attr = torch.cat(hyperedge_attrs, dim=0) if len(hyperedge_attrs) > 0 else None
        new_hyperedge_weights = (
            torch.cat(hyperedge_weights, dim=0) if len(hyperedge_weights) > 0 else None
        )

        return cls(
            x=new_x,
            hyperedge_index=new_hyperedge_index,
            hyperedge_weights=new_hyperedge_weights,
            hyperedge_attr=new_hyperedge_attr,
            num_nodes=new_x.size(0),
            num_hyperedges=new_y.size(0),
            global_node_ids=new_global_node_ids,
            target_node_mask=new_target_node_mask,
            y=new_y,
            task=hdata_with_largest_node_space.task,
        )

    def add_negative_samples(
        self,
        negative_sampler: NegativeSampler,
        seed: int | None = None,
    ) -> HData:
        """
        Return a new `HData` with sampled negative hyperedges added.

        Args:
            negative_sampler: Sampler used to generate negative hyperedges from this instance.
            seed: Optional random seed used for both negative sampling and the final shuffle.
                Defaults to ``None``.

        Returns:
            hdata: A new `HData` containing the original hyperedges and sampled negatives.
        """
        neg_hdata = negative_sampler.sample(self, seed=seed)
        hdata_with_negatives = self.cat_same_node_space([self, neg_hdata])
        return hdata_with_negatives.shuffle(seed=seed)

    @classmethod
    def empty(cls, task: Task = TaskEnum.HYPERLINK_PREDICTION) -> HData:
        """
        Create an empty HData instance.

        Args:
            task: Learning task for the empty HData. Defaults to ``"hyperlink-prediction"``.

        Returns:
            task: Learning task for the empty HData.

        Returns:
            hdata: Empty HData.
        """
        return cls(
            x=empty_nodefeatures(),
            hyperedge_index=empty_hyperedgeindex(),
            hyperedge_weights=None,
            hyperedge_attr=None,
            num_nodes=0,
            num_hyperedges=0,
            global_node_ids=None,
            target_node_mask=None,
            y=None,
            task=task,
        )

    @classmethod
    def from_hyperedge_index(
        cls,
        hyperedge_index: Tensor,
        task: Task = TaskEnum.HYPERLINK_PREDICTION,
    ) -> HData:
        """
        Build an `HData` from a given hyperedge index, with empty node features and
        hyperedge attributes.

        - Node features are initialized as an empty tensor of shape ``[0, 0]``.
        - Hyperedge attributes are set to ``None``.
        - Hyperedge weights are set to ``None``.
        - The number of nodes and hyperedges are inferred from the hyperedge index.

        Examples:
            >>> hyperedge_index = [[0, 0, 1, 2, 3, 4],
            ...                    [0, 0, 0, 1, 2, 2]]
            >>> num_nodes = 5
            >>> num_hyperedges = 3
            >>> x = []  # Empty node features with shape [0, 0]
            >>> hyperedge_attr = None
            >>> hyperedge_weights = None

        Args:
            hyperedge_index: Tensor of shape ``[2, num_incidences]`` representing
                the hypergraph connectivity.
            task: Learning task for the resulting `HData`. Defaults to ``"hyperlink-prediction"``.

        Returns:
            hdata: An `HData` instance with the given hyperedge index and default values
                for other attributes.
        """
        return cls(
            x=empty_nodefeatures(),
            hyperedge_index=hyperedge_index.clone(),
            hyperedge_weights=None,
            hyperedge_attr=None,
            global_node_ids=None,
            target_node_mask=None,
            y=None,
            task=task,
        )

    @classmethod
    def split(
        cls,
        hdata: HData,
        split_hyperedge_ids: Tensor | None = None,
        node_space_setting: NodeSpaceSetting = "transductive",
        splitter: Splitter[HData, Any] | None = None,
    ) -> HData:
        """
        Build an `HData` for a single split from the given hyperedge IDs.

        Examples:
            Transductive split (default) preserving the full node space:
            >>> split_hdata = HData.split(
            ...    hdata,
            ...    torch.tensor([1]),
            ...    node_space_setting="transductive")
            >>> split_hdata.x.shape[0] == hdata.x.shape[0]
            >>> split_hdata.hyperedge_index
            ... # node IDs stay in the original row space, hyperedge IDs are rebased

            Inductive split:
            >>> split_hdata = HData.split(hdata, torch.tensor([1]), node_space_setting="inductive")
            >>> split_hdata.x.shape[0]  # only nodes incident to hyperedge 1
            ... 2

        Args:
            hdata: The original `HData` containing the full hypergraph.
            split_hyperedge_ids: Tensor of hyperedge IDs to include in this split.
                It is assumed that the provided hyperedge IDs are valid and exist
                in ``hdata.hyperedge_index[1]``.
                It is mandatory to provide this argument unless a custom ``splitter`` is provided
                that owns split materialization.
            node_space_setting: Whether to preserve the full node space in the splits.
                ``transductive`` (default) ensures all node features are present in the split,
                while ``inductive`` allows splits to have disjoint node spaces.
            splitter: Optional HData splitter. When provided, it owns split materialization.
                Defaults to ``None``.

        Returns:
            hdata: The splitted instance with remapped node and hyperedge IDs.

        Raises:
            ValueError: If ``node_space_setting`` is not ``"transductive"`` or ``"inductive"``.
        """
        if splitter is not None:
            return splitter.split(to_split=hdata)

        if split_hyperedge_ids is None:
            raise ValueError(
                "'split_hyperedge_ids' must be provided when 'splitter' is not provided."
            )

        from hypertorch.data.splitter import HyperedgeHDataSplitter

        return HyperedgeHDataSplitter(node_space_setting=node_space_setting).split(
            to_split=hdata,
            split_hyperedge_ids=split_hyperedge_ids,
        )

    @property
    def is_hyperedge_related_task(self) -> bool:
        """
        Check if the task uses hyperedge-level targets and operations.

        Returns:
            is_hyperedge_related: True if the task is hyperedge-related, False otherwise.
        """
        # For now, we only support hyperlink prediction as a hyperedge-related task
        return self.task == TaskEnum.HYPERLINK_PREDICTION

    @property
    def is_node_related_task(self) -> bool:
        """
        Check if the task uses node-level targets and operations.

        Returns:
            is_node_related: True if the task is node-related, False otherwise.
        """
        # For now, we only support node classification as a node-related task
        return self.task == TaskEnum.NODE_CLASSIFICATION

    @property
    def sampleable_node_ids(self) -> Tensor:
        """
        Return node IDs that are eligible for sampling based on the task of this HData instance.
        """
        if self.is_node_related_task:
            # as_tuple=False returns a 2-D tensor where each row is the index for a nonzero value
            # so, we flatten it to get a 1-D tensor of the nonzero indices (eligible node IDs)
            # Example: target_node_mask = [False, True, False, True], then
            #          -> target_node_mask.nonzero(as_tuple=False) = [[1], [3]]
            #          -> flatten() = [1, 3], which are the eligible node IDs
            return self.target_node_mask.nonzero(as_tuple=False).flatten()
        return HyperedgeIndex(self.hyperedge_index).node_ids

    def enrich_node_features(
        self,
        enricher: NodeEnricher,
        enrichment_mode: EnrichmentMode | None = "replace",
    ) -> HData:
        """
        Enrich node features using the provided node feature enricher.

        Args:
            enricher: An instance of NodeEnricher to generate structural node features
                from hypergraph topology.
            enrichment_mode: How to combine generated features with existing ``hdata.x``.
                ``concatenate`` appends new features as additional columns.
                ``replace`` substitutes ``hdata.x`` entirely.
                Defaults to ``replace`` if not provided.
        """
        self.__validate_enrichment_mode(enrichment_mode)
        enriched_features = enricher.enrich(self.hyperedge_index)

        match enrichment_mode:
            case "concatenate":
                x = torch.cat([self.x, enriched_features], dim=1)
            case _:
                x = enriched_features

        return self.__class__(
            x=x,
            hyperedge_index=self.hyperedge_index.clone(),
            hyperedge_weights=clone_optional_tensor(self.hyperedge_weights),
            hyperedge_attr=clone_optional_tensor(self.hyperedge_attr),
            num_nodes=self.num_nodes,
            num_hyperedges=self.num_hyperedges,
            global_node_ids=self.global_node_ids.clone(),
            target_node_mask=self.target_node_mask.clone(),
            y=self.y.clone(),
            task=self.task,
        )

    def enrich_node_features_from(
        self,
        hdata_with_features: HData,
        node_space_setting: NodeSpaceSetting = "transductive",
        fill_value: NodeSpaceFiller | None = None,
    ) -> HData:
        """
        Copy node features from another `HData` by aligning features by ``global_node_ids``.

        Examples:
            Transductive enrichment (default) expecting the same node space in both
            source and target:
            >>> target = target.enrich_node_features_from(source, node_space_setting="transductive")

            Inductive with a scalar fill value:
            >>> target = target.enrich_node_features_from(
            ...     source,
            ...     node_space_setting="inductive",
            ...     fill_value=0.0,
            ... )

            Inductive with a feature vector fill value:
            >>> target = target.enrich_node_features_from(
            ...     source,
            ...     node_space_setting="inductive",
            ...     fill_value=[0.0, 1.0, 0.0],
            ... )

        Args:
            hdata_with_features: Source `HData` providing node features.
            node_space_setting: The setting for the node space, determining how nodes are handled.
                If ``"transductive"``, every target node is expected to exist in the source.
                If ``"inductive"``, the target dataset may have a different node space, and missing
                nodes are filled using ``fill_value``.
            fill_value: Scalar or vector used to fill missing node features when
                ``node_space_setting`` is not transductive.  Defaults to ``None``.

        Returns:
            hdata: A new `HData` with node features copied from ``hdata_with_features``.

        Raises:
            ValueError: If either instance lacks ``global_node_ids``, if the source feature rows
                do not align with the source node IDs, if ``fill_value`` is used with
                ``node_space_setting="transductive"``, or if ``fill_value`` is missing or
                malformed when ``node_space_setting="inductive"``.
        """
        source_global_node_ids = hdata_with_features.global_node_ids
        source_x = hdata_with_features.x
        if source_x.size(0) != source_global_node_ids.size(0):
            raise ValueError(
                "Expected 'hdata_with_features.x' rows to align with "
                "hdata_with_features.global_node_ids."
            )
        self.__validate_node_space_setting(node_space_setting, fill_value)

        target_global_node_ids = self.global_node_ids.detach().cpu().tolist()

        # We need the index of the features for each node in the source, as we will use
        # the index to track back
        # to the node feautures after we match the global node id in the target to the one that
        # is in the source
        source_feature_idx_by_global_node_id = {
            int(global_node_id): feature_idx
            for feature_idx, global_node_id in enumerate(
                source_global_node_ids.detach().cpu().tolist()
            )
        }

        fill_features = self.__to_fill_features(
            fill_value=fill_value,
            num_features=int(source_x.size(1)),
            dtype=source_x.dtype,
            device=source_x.device,
        )

        enriched_rows = []
        missing_global_node_ids = []
        for global_node_id in target_global_node_ids:
            source_feature_idx = source_feature_idx_by_global_node_id.get(int(global_node_id))
            if source_feature_idx is None:
                # Example: global_node_id = 30 is not present in the source
                #          -> strict transductive mode records it as
                #             missing and then raises an error
                #          -> non-transductive mode fills the features with
                #             fill_value and continues enriching the other nodes
                if is_transductive_setting(node_space_setting):
                    missing_global_node_ids.append(
                        int(global_node_id)
                    )  # record missing node for error message
                else:
                    enriched_rows.append(
                        fill_features
                    )  # fill missing node features with fill_value and
                continue

            # Match the global node IDs in the target to the corresponding
            # feature indices in the source
            # Example: source_global_node_ids = [10, 20, 30], source_x has shape (3, num_features)
            #          target_global_node_ids = [10, 30]
            #          -> source_feature_idx_by_global_node_id = {10: 0, 20: 1, 30: 2}
            #          -> pick source_x rows 0 and 2 for the target
            enriched_rows.append(source_x[source_feature_idx])

        if len(missing_global_node_ids) > 0:
            raise ValueError(
                f"Missing node features for target global_node_ids: {missing_global_node_ids}."
            )

        enriched_x = torch.stack(enriched_rows, dim=0).to(device=self.device)

        return self.__class__(
            x=enriched_x,
            hyperedge_index=self.hyperedge_index.clone(),
            hyperedge_weights=clone_optional_tensor(self.hyperedge_weights),
            hyperedge_attr=clone_optional_tensor(self.hyperedge_attr),
            num_nodes=self.num_nodes,
            num_hyperedges=self.num_hyperedges,
            global_node_ids=self.global_node_ids.clone(),
            target_node_mask=self.target_node_mask.clone(),
            y=self.y.clone(),
            task=self.task,
        )

    def enrich_hyperedge_weights(
        self,
        enricher: HyperedgeEnricher,
        enrichment_mode: EnrichmentMode | None = "replace",
    ) -> HData:
        """
        Enrich hyperedge weights using the provided hyperedge weight enricher.

        Args:
            enricher: An instance of HyperedgeEnricher to generate hyperedge weights from
                hypergraph topology.
            enrichment_mode: How to combine generated weights with
                existing ``hdata.hyperedge_weights``.
                ``concatenate`` appends new weights to the existing 1D tensor.
                ``replace`` substitutes ``hdata.hyperedge_weights`` entirely.
                Defaults to ``replace`` if not provided.

        Returns:
            hdata: A new `HData` with enriched hyperedge weights.
        """
        self.__validate_enrichment_mode(enrichment_mode)
        enriched_weights = enricher.enrich(self.hyperedge_index)

        match enrichment_mode:
            case "concatenate":
                hyperedge_weights = (
                    torch.cat([self.hyperedge_weights, enriched_weights], dim=0)
                    if self.hyperedge_weights is not None
                    else enriched_weights
                )
            case _:
                hyperedge_weights = enriched_weights

        return self.__class__(
            x=self.x.clone(),
            hyperedge_index=self.hyperedge_index.clone(),
            hyperedge_weights=hyperedge_weights,
            hyperedge_attr=clone_optional_tensor(self.hyperedge_attr),
            num_nodes=self.num_nodes,
            num_hyperedges=self.num_hyperedges,
            global_node_ids=self.global_node_ids.clone(),
            target_node_mask=self.target_node_mask.clone(),
            y=self.y.clone(),
            task=self.task,
        )

    def enrich_hyperedge_attr(
        self,
        enricher: HyperedgeEnricher,
        enrichment_mode: EnrichmentMode | None = "replace",
    ) -> HData:
        """
        Enrich hyperedge features using the provided hyperedge feature enricher.

        Args:
            enricher: An instance of HyperedgeEnricher to generate structural hyperedge
                features from hypergraph topology.
            enrichment_mode: How to combine generated features with
                existing ``hdata.hyperedge_attr``.
                ``concatenate`` appends new features as additional columns.
                ``replace`` substitutes ``hdata.hyperedge_attr`` entirely.
                Defaults to ``replace`` if not provided.
        """
        self.__validate_enrichment_mode(enrichment_mode)
        enriched_features = enricher.enrich(self.hyperedge_index)

        match enrichment_mode:
            case "concatenate":
                hyperedge_attr = (
                    torch.cat([self.hyperedge_attr, enriched_features], dim=1)
                    if self.hyperedge_attr is not None
                    else enriched_features
                )
            case _:
                hyperedge_attr = enriched_features

        return self.__class__(
            x=self.x.clone(),
            hyperedge_index=self.hyperedge_index.clone(),
            hyperedge_weights=clone_optional_tensor(self.hyperedge_weights),
            hyperedge_attr=hyperedge_attr,
            num_nodes=self.num_nodes,
            num_hyperedges=self.num_hyperedges,
            global_node_ids=self.global_node_ids.clone(),
            target_node_mask=self.target_node_mask.clone(),
            y=self.y.clone(),
            task=self.task,
        )

    def get_device_if_all_consistent(self) -> torch.device:
        """
        Check that all tensors are on the same device and return that device.

        If there are no tensors or if they are on different devices, return CPU.

        Returns:
            device: The common device if all tensors are on the same device, otherwise CPU.

        Raises:
            ValueError: If tensors are on different devices.
        """
        devices = {
            self.x.device,
            self.hyperedge_index.device,
            self.global_node_ids.device,
            self.target_node_mask.device,
            self.y.device,
        }

        if self.hyperedge_attr is not None:
            devices.add(self.hyperedge_attr.device)
        if self.hyperedge_weights is not None:
            devices.add(self.hyperedge_weights.device)

        if len(devices) > 1:
            raise ValueError(f"Inconsistent device placement: {devices}")

        return devices.pop() if len(devices) == 1 else torch.device("cpu")

    def remove_hyperedges_with_fewer_than_k_nodes(
        self,
        k: int,
        preserve_global_node_ids: bool = False,
    ) -> HData:
        """
        Remove hyperedges that have fewer than k incident nodes.

        Args:
            k: The minimum number of nodes a hyperedge must have to be retained.
            preserve_global_node_ids: Whether to preserve the global node IDs after
                removing hyperedges. Defaults to ``False``.
                If ``False``, the global node IDs will be reindexed to be contiguous after
                removing hyperedges.
                If ``True``, the global node IDs will be preserved, which may cause some models
                to raise as they may expect contiguous global node IDs.
        """
        validate_is_positive("k", k)

        hyperedge_index_wrapper = HyperedgeIndex(
            self.hyperedge_index.clone()
        ).remove_hyperedges_with_fewer_than_k_nodes(k)

        x = self.x[hyperedge_index_wrapper.node_ids]
        y = (
            self.y[hyperedge_index_wrapper.node_ids]
            if self.is_node_related_task
            else self.y[hyperedge_index_wrapper.hyperedge_ids]
        )
        target_node_mask = self.target_node_mask[hyperedge_index_wrapper.node_ids]

        global_node_ids = (
            self.global_node_ids[hyperedge_index_wrapper.node_ids]
            if preserve_global_node_ids
            else None
        )
        hyperedge_attr = (
            self.hyperedge_attr[hyperedge_index_wrapper.hyperedge_ids]
            if self.hyperedge_attr is not None
            else None
        )
        hyperedge_weights = (
            self.hyperedge_weights[hyperedge_index_wrapper.hyperedge_ids]
            if self.hyperedge_weights is not None
            else None
        )

        return self.__class__(
            x=x,
            hyperedge_index=hyperedge_index_wrapper.to_0based().item,
            hyperedge_weights=hyperedge_weights,
            hyperedge_attr=hyperedge_attr,
            num_nodes=hyperedge_index_wrapper.num_nodes,
            num_hyperedges=hyperedge_index_wrapper.num_hyperedges,
            global_node_ids=global_node_ids,
            target_node_mask=target_node_mask,
            y=y,
            task=self.task,
        )

    def shuffle(self, seed: int | None = None) -> HData:
        """
        Return a new `HData` instance with hyperedge IDs randomly reassigned.

        Each hyperedge keeps its original set of nodes, but is assigned a new ID
        via a random permutation.
        ``y`` and ``hyperedge_attr`` are reordered to match, so that ``y[new_id]``
        still corresponds to the correct hyperedge.
        Same for ``hyperedge_attr[new_id]`` if hyperedge attributes are present.

        Examples:
            >>> hyperedge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
            >>> y  = torch.tensor([1, 0])
            >>> hdata = HData(x=x, hyperedge_index=hyperedge_index, y=y)
            >>> shuffled_hdata = hdata.shuffle(seed=42)
            >>> shuffled_hdata.hyperedge_index  # hyperedges may be reassigned
            ... # e.g.,
            ...     [[0, 1, 2, 3],
            ...      [1, 1, 0, 0]]
            >>> shuffled_hdata.y  # labels are permuted to match new hyperedge IDs, e.g., [0, 1]

        Args:
            seed: Optional random seed for reproducibility. If ``None``, the shuffle
                will be non-deterministic. Defaults to ``None``.

        Returns:
            hdata: A new `HData` instance with hyperedge IDs, ``y``, and
                ``hyperedge_attr`` permuted.
        """
        generator = create_seeded_torch_generator(device=self.device, seed=seed)
        permutation = torch.randperm(
            self.num_hyperedges,
            generator=generator,
            dtype=torch.long,
            device=self.device,
        )

        # permutation[new_id] = old_id, so y[permutation] puts old labels into new slots
        # inverse_permutation[old_id] = new_id, used to remap hyperedge IDs in incidences
        # Example: permutation = [1, 2, 0] means new_id 0 gets old_id 1,
        #                   new_id 1 gets old_id 2, new_id 2 gets old_id 0
        #                   -> inverse_permutation = [2, 0, 1] means old_id 0 gets new_id 2,
        #                        old_id 1 gets new_id 0, old_id 2 gets new_id 1
        inverse_permutation = torch.empty_like(
            permutation,
            dtype=permutation.dtype,
            device=permutation.device,
        )
        inverse_permutation[permutation] = torch.arange(
            self.num_hyperedges,
            dtype=permutation.dtype,
            device=permutation.device,
        )

        new_hyperedge_index = self.hyperedge_index.clone()

        # Example: hyperedge_index = [[0, 1, 2, 3, 4],
        #                             [0, 0, 1, 1, 2]],
        #          inverse_permutation = [2, 0, 1] (new_id 0 -> old_id 2, new_id 1 ->
        #                                           old_id 0, new_id 2 -> old_id 1)
        #          -> new_hyperedge_index = [[0, 1, 2, 3, 4],
        #                                    [2, 2, 0, 0, 1]]
        old_hyperedge_ids = self.hyperedge_index[1]
        new_hyperedge_index[1] = inverse_permutation[old_hyperedge_ids]

        # Example: hyperedge_attr = [attr_0, attr_1, attr_2], permutation = [1, 2, 0]
        #          -> new_hyperedge_attr = [attr_1  (attr of old_id 1),
        #                                   attr_2 (attr of old_id 2),
        #                                   attr_0 (attr of old_id 0)]
        new_hyperedge_attr = (
            self.hyperedge_attr[permutation] if self.hyperedge_attr is not None else None
        )

        new_hyperedge_weights = (
            self.hyperedge_weights[permutation] if self.hyperedge_weights is not None else None
        )

        # Permutate only for tasks where y is related to hyperedges (e.g., hyperlink-prediction)
        # Example: y = [1, 1, 0], permutation = [1, 2, 0]
        #          -> new_y = [y[1], y[2], y[0]] = [1, 0, 1]
        new_y = self.y[permutation] if self.is_hyperedge_related_task else self.y.clone()

        return self.__class__(
            x=self.x.clone(),
            hyperedge_index=new_hyperedge_index,
            hyperedge_weights=new_hyperedge_weights,
            hyperedge_attr=new_hyperedge_attr,
            num_nodes=self.num_nodes,
            num_hyperedges=self.num_hyperedges,
            global_node_ids=self.global_node_ids.clone(),
            target_node_mask=self.target_node_mask.clone(),
            y=new_y,
            task=self.task,
        )

    def clone(self) -> HData:
        """
        Return a deep copy of this `HData`.

        Returns:
            hdata: A new `HData` that is a deep copy of this instance.
        """
        return self.__class__(
            x=self.x.clone(),
            hyperedge_index=self.hyperedge_index.clone(),
            hyperedge_weights=clone_optional_tensor(self.hyperedge_weights),
            hyperedge_attr=clone_optional_tensor(self.hyperedge_attr),
            num_nodes=self.num_nodes,
            num_hyperedges=self.num_hyperedges,
            global_node_ids=self.global_node_ids.clone(),
            target_node_mask=self.target_node_mask.clone(),
            y=self.y.clone(),
            task=self.task,
        )

    def to(self, device: torch.device | str, non_blocking: bool = False) -> HData:
        """
        Move all tensors to the specified device.

        Args:
            device: The target device (e.g., 'cpu', 'cuda:0').
            non_blocking: If ``True`` and the source and destination devices are both CUDA,
                the copy will be non-blocking. Defaults to ``False``.

        Returns:
            hdata: The `HData` instance with all tensors moved to the specified device.
        """
        self.x = self.x.to(device=device, non_blocking=non_blocking)
        self.hyperedge_index = self.hyperedge_index.to(device=device, non_blocking=non_blocking)
        self.global_node_ids = self.global_node_ids.to(device=device, non_blocking=non_blocking)
        self.target_node_mask = self.target_node_mask.to(device=device, non_blocking=non_blocking)
        self.y = self.y.to(device=device, non_blocking=non_blocking)

        if self.hyperedge_attr is not None:
            self.hyperedge_attr = self.hyperedge_attr.to(device=device, non_blocking=non_blocking)

        if self.hyperedge_weights is not None:
            self.hyperedge_weights = self.hyperedge_weights.to(
                device=device,
                non_blocking=non_blocking,
            )

        self.device = device if isinstance(device, torch.device) else torch.device(device)
        return self

    def with_target_node_mask(self, target_node_mask: Tensor) -> HData:
        """
        Return a copy of this instance with a ``target_node_mask`` attribute set to the given mask.

        Args:
            target_node_mask: A boolean tensor indicating which nodes are considered target nodes.

        Returns:
            hdata: A new `HData` instance with the same attributes except for ``target_node_mask``,
                which is set to the provided mask.
        """
        return self.__class__(
            x=self.x.clone(),
            hyperedge_index=self.hyperedge_index.clone(),
            hyperedge_weights=clone_optional_tensor(self.hyperedge_weights),
            hyperedge_attr=clone_optional_tensor(self.hyperedge_attr),
            num_nodes=self.num_nodes,
            num_hyperedges=self.num_hyperedges,
            global_node_ids=self.global_node_ids.clone(),
            target_node_mask=target_node_mask.clone(),
            y=self.y.clone(),
            task=self.task,
        )

    def with_y_to(self, value: float, size: int | None = None) -> HData:
        """
        Return a copy of this instance with a y attribute set to the given value.

        Args:
            value: The value to set for all entries in the y attribute.
            size: The size of the y tensor. If ``None``, the size will be inferred
                from the number of hyperedges in the instance.

        Returns:
            hdata: A new `HData` instance with the same attributes except for y,
                which is set to a tensor of the given value.
        """
        y_size = size if size is not None else self.num_hyperedges
        return self.__class__(
            x=self.x.clone(),
            hyperedge_index=self.hyperedge_index.clone(),
            hyperedge_weights=clone_optional_tensor(self.hyperedge_weights),
            hyperedge_attr=clone_optional_tensor(self.hyperedge_attr),
            num_nodes=self.num_nodes,
            num_hyperedges=self.num_hyperedges,
            global_node_ids=self.global_node_ids.clone(),
            target_node_mask=self.target_node_mask.clone(),
            y=torch.full((y_size,), value, dtype=torch.float, device=self.device),
            task=self.task,
        )

    def with_y_ones(self, size: int | None = None) -> HData:
        """
        Return a copy of this instance with a y attribute of all ones.

        Args:
            size: The size of the y tensor. If ``None``, the size will be inferred
                from the number of hyperedges in the instance.

        Returns:
            hdata: A new `HData` instance with the same attributes except for y, which is
                set to a tensor of ones.
        """
        return self.with_y_to(1.0, size=size)

    def with_y_zeros(self, size: int | None = None) -> HData:
        """
        Return a copy of this instance with a y attribute of all zeros.

        Args:
            size: The size of the y tensor. If ``None``, the size will be inferred
                from the number of hyperedges in the instance.

        Returns:
            hdata: A new `HData` instance with the same attributes except for y, which is
                set to a tensor of zeros.
        """
        return self.with_y_to(0.0, size=size)

    def stats(self) -> dict[str, Any]:
        """
        Compute statistics for the hypergraph data.

        Fields:
            - ``shape_x``: The shape of the node feature matrix ``x``.
            - ``shape_hyperedge_weights``: The shape of the hyperedge weights tensor, or
                ``None`` if hyperedge weights are not present.
            - ``shape_hyperedge_attr``: The shape of the hyperedge attribute matrix, or ``None``
                if hyperedge attributes are not present.
            - ``num_nodes``: The number of nodes in the hypergraph.
            - ``num_hyperedges``: The number of hyperedges in the hypergraph.
            - ``avg_degree_node_raw``: The average degree of nodes, calculated as the mean
                number of hyperedges each node belongs to.
            - ``avg_degree_node``: The floored node average degree.
            - ``avg_degree_hyperedge_raw``: The average size of hyperedges, calculated as
                the mean number of nodes each hyperedge contains.
            - ``avg_degree_hyperedge``: The floored hyperedge average size.
            - ``node_degree_max``: The maximum degree of any node in the hypergraph.
            - ``hyperedge_degree_max``: The maximum size of any hyperedge in the hypergraph.
            - ``node_degree_median``: The median degree of nodes in the hypergraph.
            - ``hyperedge_degree_median``: The median size of hyperedges in the hypergraph.
            - ``distribution_node_degree``: A list where the value at index ``i`` represents
                the count of nodes with degree ``i``.
            - ``distribution_hyperedge_size``: A list where the value at index ``i`` represents
                the count of hyperedges with size ``i``.
            - ``distribution_node_degree_hist``: A dictionary where the keys are node degrees and
                the values are the count of nodes with that degree.
            - ``distribution_hyperedge_size_hist``: A dictionary where the keys are hyperedge
                sizes and the values are the count of hyperedges with that size.

        Returns:
            stats: A dictionary containing various statistics about the hypergraph.
        """
        node_ids = self.hyperedge_index[0]
        hyperedge_ids = self.hyperedge_index[1]

        # Degree of each node = number of hyperedges it belongs to
        # Size of each hyperedge = number of nodes it contains
        if node_ids.numel() > 0:
            distribution_node_degree = torch.bincount(node_ids, minlength=self.num_nodes).float()
            distribution_hyperedge_size = torch.bincount(
                hyperedge_ids, minlength=self.num_hyperedges
            ).float()
        else:
            distribution_node_degree = torch.zeros(
                self.num_nodes, dtype=torch.float, device=self.device
            )
            distribution_hyperedge_size = torch.zeros(
                self.num_hyperedges, dtype=torch.float, device=self.device
            )

        if distribution_node_degree.numel() > 0:
            avg_degree_node_raw = distribution_node_degree.mean(dtype=torch.float).item()
            avg_degree_node = int(avg_degree_node_raw)
            avg_degree_hyperedge_raw = distribution_hyperedge_size.mean(dtype=torch.float).item()
            avg_degree_hyperedge = int(avg_degree_hyperedge_raw)
            node_degree_max = int(distribution_node_degree.max().item())
            hyperedge_degree_max = int(distribution_hyperedge_size.max().item())
            node_degree_median = int(distribution_node_degree.median().item())
            hyperedge_degree_median = int(distribution_hyperedge_size.median().item())
        else:
            avg_degree_node_raw = 0
            avg_degree_node = 0
            avg_degree_hyperedge_raw = 0
            avg_degree_hyperedge = 0
            node_degree_max = 0
            hyperedge_degree_max = 0
            node_degree_median = 0
            hyperedge_degree_median = 0

        # Histograms: index i holds count of nodes/hyperedges with degree/size i
        distribution_node_degree_hist = torch.bincount(distribution_node_degree.long())
        distribution_hyperedge_size_hist = torch.bincount(distribution_hyperedge_size.long())

        distribution_node_degree_hist = {
            i: int(count.item())
            for i, count in enumerate(distribution_node_degree_hist)
            if count.item() > 0
        }
        distribution_hyperedge_size_hist = {
            i: int(count.item())
            for i, count in enumerate(distribution_hyperedge_size_hist)
            if count.item() > 0
        }

        return {
            "shape_x": self.x.shape,
            "shape_hyperedge_weights": self.hyperedge_weights.shape
            if self.hyperedge_weights is not None
            else None,
            "shape_hyperedge_attr": self.hyperedge_attr.shape
            if self.hyperedge_attr is not None
            else None,
            "num_nodes": self.num_nodes,
            "num_hyperedges": self.num_hyperedges,
            "avg_degree_node_raw": avg_degree_node_raw,
            "avg_degree_node": avg_degree_node,
            "avg_degree_hyperedge_raw": avg_degree_hyperedge_raw,
            "avg_degree_hyperedge": avg_degree_hyperedge,
            "node_degree_max": node_degree_max,
            "hyperedge_degree_max": hyperedge_degree_max,
            "node_degree_median": node_degree_median,
            "hyperedge_degree_median": hyperedge_degree_median,
            "distribution_node_degree": distribution_node_degree.int().tolist(),
            "distribution_hyperedge_size": distribution_hyperedge_size.int().tolist(),
            "distribution_node_degree_hist": distribution_node_degree_hist,
            "distribution_hyperedge_size_hist": distribution_hyperedge_size_hist,
        }

    @classmethod
    def __validate_can_perform_cat_same_node_space(
        cls,
        hdatas: Sequence[HData],
        x: Tensor | None,
        global_node_ids: Tensor | None,
    ) -> None:
        """
        Validate inputs for concatenating HData objects in the same node space.

        Args:
            hdatas: HData objects to concatenate.
            x: Optional shared node feature matrix.
            global_node_ids: Optional shared global node IDs.

        Raises:
            ValueError: If required paired arguments are missing or hyperedge IDs overlap.
        """
        validate_is_non_empty("hdatas", hdatas)

        if x is not None and global_node_ids is None:
            raise ValueError(
                "If 'x' is provided, 'global_node_ids' must also be provided to ensure consistency."
            )
        if x is None and global_node_ids is not None:
            raise ValueError(
                "If 'global_node_ids' is provided, 'x' must also be provided to ensure consistency."
            )

        joint_hyperedge_ids = torch.cat([hdata.hyperedge_index[1].unique() for hdata in hdatas])
        unique_joint_hyperedge_ids = joint_hyperedge_ids.unique()
        if unique_joint_hyperedge_ids.size(0) != joint_hyperedge_ids.size(0):
            raise ValueError(
                "Overlapping hyperedge IDs found across instances. Ensure each "
                "instance uses distinct hyperedge IDs."
            )

        tasks = {hdata.task for hdata in hdatas}
        if len(tasks) > 1:
            raise ValueError(f"All HData instances must have the same task, got {tasks}.")

    def __to_fill_features(
        self,
        fill_value: NodeSpaceFiller | None,
        num_features: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        """
        Convert a fill value into a feature vector.

        Args:
            fill_value: Scalar or vector fill value.
            num_features: Required number of feature values.
            dtype: Desired tensor dtype.
            device: Desired tensor device.

        Returns:
            fill_features: Tensor of shape ``(num_features,)`` or an empty tensor.

        Raises:
            ValueError: If the fill value cannot be broadcast to the requested feature count.
        """
        if fill_value is None:
            return torch.empty((0,), dtype=dtype, device=device)

        if isinstance(fill_value, Tensor):
            fill_features = fill_value.to(dtype=dtype, device=device)
        elif isinstance(fill_value, (int, float)):
            fill_features = torch.full(
                (num_features,), float(fill_value), dtype=dtype, device=device
            )
        else:
            fill_features = torch.tensor(fill_value, dtype=dtype, device=device)

        # This can happen when fill_value is:
        # - A scalar tensor, e.g., tensor(0.0), which should be broadcasted to all features
        # - A list with a single value, e.g., [0.0], which should
        #   also be broadcasted to all features
        if fill_features.numel() == 1:
            fill_features = fill_features.repeat(num_features)

        if fill_features.dim() != 1 or fill_features.numel() != num_features:
            raise ValueError(
                f"Expected 'fill_value' to define exactly {num_features} features, got shape "
                f"{tuple(fill_features.shape)}."
            )
        return fill_features

    def __validate(self) -> None:
        """
        Validate all HData tensor fields.

        Raises:
            ValueError: If any field has an invalid shape, dtype, or count.
        """
        self.__validate_x()
        self.__validate_hyperedge_index()
        self.__validate_hyperedge_attr()
        self.__validate_hyperedge_weights()
        self.__validate_global_node_ids()
        self.__validate_target_node_mask()
        self.__validate_labels()
        self.__validate_task()

    def __validate_enrichment_mode(self, enrichment_mode: EnrichmentMode | None) -> None:
        """
        Validate a feature enrichment mode.

        Args:
            enrichment_mode: Optional enrichment mode to validate.

        Raises:
            ValueError: If the mode is unsupported.
        """
        if enrichment_mode is None or enrichment_mode in ("replace", "concatenate"):
            return

        raise ValueError(
            f"'enrichment_mode' must be one of 'replace', 'concatenate', "
            f"or None, got {enrichment_mode!r}."
        )

    def __validate_hyperedge_attr(self) -> None:
        """
        Validate optional hyperedge attributes.

        Raises:
            ValueError: If hyperedge attributes have an invalid dtype or shape.
        """
        if self.hyperedge_attr is None:
            return

        validate_floating_tensor_dtype("hyperedge_attr", self.hyperedge_attr)
        if self.hyperedge_attr.dim() != 2:
            raise ValueError(
                f"'hyperedge_attr' must be a 2D tensor, got shape "
                f"{tuple(self.hyperedge_attr.shape)}."
            )
        if self.hyperedge_attr.size(0) != self.num_hyperedges:
            raise ValueError(
                "'hyperedge_attr' must have one row per hyperedge. "
                f"Got size={self.hyperedge_attr.size(0)} but "
                f"num_hyperedges={self.num_hyperedges}."
            )

    def __validate_hyperedge_index(self) -> None:
        """
        Validate hyperedge index IDs against configured node and hyperedge counts.

        Raises:
            ValueError: If IDs are negative or counts are too small.
        """
        if self.hyperedge_index.numel() > 0 and bool((self.hyperedge_index < 0).any()):
            raise ValueError("'hyperedge_index' cannot contain negative node or hyperedge IDs.")

        unique_node_count = self.hyperedge_index[0].unique().size(0)
        if unique_node_count > self.num_nodes:
            raise ValueError(
                f"'num_nodes' is too small for 'hyperedge_index'. "
                f"Got num_nodes={self.num_nodes}, but 'hyperedge_index' contains "
                f"{unique_node_count} unique node IDs."
            )

        unique_hyperedge_count = self.hyperedge_index[1].unique().size(0)
        if unique_hyperedge_count > self.num_hyperedges:
            raise ValueError(
                f"'num_hyperedges' is too small for 'hyperedge_index'. "
                f"Got num_hyperedges={self.num_hyperedges}, but 'hyperedge_index' contains "
                f"{unique_hyperedge_count} unique hyperedge IDs."
            )

    def __validate_hyperedge_weights(self) -> None:
        """
        Validate optional hyperedge weights.

        Raises:
            ValueError: If hyperedge weights have an invalid dtype or shape.
        """
        if self.hyperedge_weights is None:
            return

        validate_floating_tensor_dtype("hyperedge_weights", self.hyperedge_weights)

        if self.hyperedge_weights.dim() != 1:
            raise ValueError(
                f"'hyperedge_weights' must be a 1D tensor, "
                f"got shape {tuple(self.hyperedge_weights.shape)}."
            )
        if self.hyperedge_weights.size(0) != self.num_hyperedges:
            raise ValueError(
                f"'hyperedge_weights' must have one entry per hyperedge. "
                f"Got size={self.hyperedge_weights.size(0)} but "
                f"num_hyperedges={self.num_hyperedges}."
            )

    def __validate_global_node_ids(self) -> None:
        """
        Validate global node IDs.

        Raises:
            ValueError: If global node IDs have an invalid dtype, shape, or length.
        """
        validate_long_tensor_dtype("global_node_ids", self.global_node_ids)
        if self.global_node_ids.dim() != 1:
            raise ValueError(
                f"'global_node_ids' must be a 1D tensor, got "
                f"shape {tuple(self.global_node_ids.shape)}."
            )
        if self.global_node_ids.size(0) != self.num_nodes:
            raise ValueError(
                f"'global_node_ids' must have one entry per node. "
                f"Got size={self.global_node_ids.size(0)} but num_nodes={self.num_nodes}."
            )

    def __validate_labels(self) -> None:
        """
        Validate labels.

        Raises:
            ValueError: If labels have an invalid dtype, shape, or length.
        """
        if self.y.dim() != 1:
            raise ValueError(f"'y' must be a 1D tensor, got shape {tuple(self.y.shape)}.")

        if self.is_node_related_task:
            validate_long_tensor_dtype("y", self.y)
            if self.y.size(0) != self.num_nodes:
                raise ValueError(
                    f"For task={self.task!r}, 'y' must have one entry per node. "
                    f"Got {self.y.size(0)} entries but num_nodes={self.num_nodes}."
                )
            return

        if self.is_hyperedge_related_task:
            validate_floating_tensor_dtype("y", self.y)
            if self.y.size(0) != self.num_hyperedges:
                raise ValueError(
                    f"For task={self.task!r}, 'y' must have one entry per hyperedge. "
                    f"Got {self.y.size(0)} entries but num_hyperedges={self.num_hyperedges}."
                )

    def __validate_target_node_mask(self) -> None:
        """
        Validate the optional supervised-node mask.

        Raises:
            ValueError: If the mask is incompatible with the configured task or node count.
        """
        # We currently only support target_node_mask for node classification tasks
        # For all other tasks, it is simply ignored
        if not self.is_node_related_task:
            return

        if self.target_node_mask.dtype != torch.bool:
            raise ValueError(
                f"'target_node_mask' must have dtype torch.bool, got {self.target_node_mask.dtype}."
            )
        if self.target_node_mask.dim() != 1:
            raise ValueError(
                f"'target_node_mask' must be a 1D tensor, "
                f"got shape {tuple(self.target_node_mask.shape)}."
            )
        if self.target_node_mask.size(0) != self.num_nodes:
            raise ValueError(
                f"'target_node_mask' must have one entry per node. "
                f"Got size={self.target_node_mask.size(0)} but num_nodes={self.num_nodes}."
            )

    def __validate_task(self) -> None:
        """
        Validate the learning task.

        Raises:
            ValueError: If the task is unsupported.
        """
        valid_tasks = get_args(TaskLiteral)
        if self.task not in valid_tasks:
            raise ValueError(f"'task' must be one of {valid_tasks}, got {self.task!r}.")

    def __validate_x(self) -> None:
        """
        Validate node feature row count.

        Raises:
            ValueError: If node features do not match the configured node count.
        """
        if self.x.size(0) not in (0, self.num_nodes):
            raise ValueError(
                f"'x' must have one feature row per node, or be 'torch.empty((0, 0))' "
                f"if there are no nodes. "
                f"Got x.shape={tuple(self.x.shape)} but num_nodes={self.num_nodes}."
            )

    def __validate_node_space_setting(
        self,
        node_space_setting: NodeSpaceSetting,
        fill_value: NodeSpaceFiller | None,
    ) -> None:
        """
        Validate node-space enrichment settings.

        Args:
            node_space_setting: Node-space setting to validate.
            fill_value: Optional fill value for missing nodes.

        Raises:
            ValueError: If the setting and fill value are incompatible.
        """
        validate_node_space_setting(node_space_setting)

        if is_transductive_setting(node_space_setting) and fill_value is not None:
            raise ValueError(
                "'fill_value' cannot be provided when node_space_setting='transductive'."
            )
        if is_inductive_setting(node_space_setting) and fill_value is None:
            raise ValueError("'fill_value' must be provided when node_space_setting='inductive'.")

    def __validate_x_and_hyperedge_index_type_and_dim(self) -> None:
        """
        Validate core tensor dtypes and dimensions.

        Raises:
            ValueError: If ``x`` or ``hyperedge_index`` has an invalid dtype or shape.
        """
        validate_floating_tensor_dtype("x", self.x)
        if self.x.dim() != 2:
            raise ValueError(f"'x' must be a 2D tensor, got shape {tuple(self.x.shape)}.")

        validate_long_tensor_dtype("hyperedge_index", self.hyperedge_index)
        if self.hyperedge_index.dim() != 2 or self.hyperedge_index.size(0) != 2:
            raise ValueError(
                f"'hyperedge_index' must have shape (2, num_incidences), got "
                f"{tuple(self.hyperedge_index.shape)}."
            )

    def __y(self, y: Tensor | None = None) -> Tensor:
        """
        Return labels as non-None tensor on the correct device.

        Args:
            y: Optional labels tensor. If ``None``, defaults to
                a tensor of ones for hyperlink prediction tasks
                or a tensor of zeros for node classification tasks.

        Returns:
            y: Labels tensor on the correct device.
        """
        if y is not None:
            return y
        if self.is_node_related_task:
            return torch.ones((self.num_nodes,), dtype=torch.long, device=self.x.device)
        return torch.ones((self.num_hyperedges,), dtype=torch.float, device=self.x.device)
