import torch

from torch import Tensor
from typing import Optional, Sequence, Dict, Any
from hyperbench.utils import (
    NodeSpaceFiller,
    NodeSpaceSetting,
    empty_hyperedgeindex,
    empty_nodefeatures,
    is_inductive_setting,
    is_transductive_setting,
    to_0based_ids,
)

from hyperbench.nn.enricher import EnrichmentMode, NodeEnricher, HyperedgeEnricher
from hyperbench.types.hypergraph import HyperedgeIndex


class HData:
    """
    Container for hypergraph data.

    Examples:
        >>> x = torch.randn(10, 16)  # 10 nodes with 16 features each
        >>> hyperedge_index = torch.tensor([[0, 0, 1, 1, 1],  # node IDs
        ...                                 [0, 1, 2, 3, 4]]) # hyperedge IDs
        >>> data = HData(x=x, hyperedge_index=hyperedge_index)

    Args:
        x: Node feature matrix of shape ``[num_nodes, num_features]``.
        hyperedge_index: Hyperedge connectivity in COO format of shape ``[2, num_incidences]``,
            where ``hyperedge_index[0]`` contains node IDs and ``hyperedge_index[1]`` contains hyperedge IDs.
        hyperedge_weights: Optional tensor of shape ``[num_hyperedges]`` containing weights for each hyperedge.
        hyperedge_attr: Hyperedge feature matrix of shape ``[num_hyperedges, num_hyperedge_features]``.
            Features associated with each hyperedge (e.g., weights, timestamps, types).
        num_nodes: Number of nodes in the hypergraph.
            If ``None``, inferred as ``x.size(0)``.
        num_hyperedges: Number of hyperedges in the hypergraph.
            If ``None``, inferred as the number of unique hyperedge IDs in ``hyperedge_index[1]``.
        y: Labels for hyperedges, of shape ``[num_hyperedges]``.
            Used for supervised learning tasks. For unsupervised tasks, this can be ignored.
            Default is a tensor of ones, indicating all hyperedges are positive examples.
        global_node_ids: Optional stable node IDs of shape ``[num_nodes]`` matching the row order of ``x``.
            Use this to preserve access to the canonical node space when ``hyperedge_index`` is rebased locally.
            If ``None``, defaults to ``torch.arange(num_nodes)``, assuming that these are the global node IDs in the same order as the rows of ``x``.
    """

    def __init__(
        self,
        x: Tensor,
        hyperedge_index: Tensor,
        hyperedge_weights: Optional[Tensor] = None,
        hyperedge_attr: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
        num_hyperedges: Optional[int] = None,
        global_node_ids: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
    ):
        self.x: Tensor = x

        self.hyperedge_index: Tensor = hyperedge_index

        self.hyperedge_weights: Optional[Tensor] = hyperedge_weights

        self.hyperedge_attr: Optional[Tensor] = hyperedge_attr

        hyperedge_index_wrapper = HyperedgeIndex(hyperedge_index)

        self.num_nodes: int = (
            num_nodes
            if num_nodes is not None
            # There should never be isolated nodes when HData is created by Dataset
            # as each isolted node gets its own self-loop hyperedge
            else hyperedge_index_wrapper.num_nodes_if_isolated_exist(num_nodes=x.size(0))
        )

        self.num_hyperedges: int = (
            num_hyperedges if num_hyperedges is not None else hyperedge_index_wrapper.num_hyperedges
        )

        self.global_node_ids: Optional[Tensor] = (
            # torch.arange is to handle isolated nodes, as they are already considered
            # when computing self.num_nodes via num_nodes_if_isolated_exist
            global_node_ids if global_node_ids is not None else torch.arange(self.num_nodes)
        )

        self.y = (
            y
            if y is not None
            else torch.ones((self.num_hyperedges,), dtype=torch.float, device=self.x.device)
        )

        self.device = self.get_device_if_all_consistent()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"    num_nodes={self.num_nodes},\n"
            f"    num_hyperedges={self.num_hyperedges},\n"
            f"    x_shape={self.x.shape},\n"
            f"    global_node_ids_shape={self.global_node_ids.shape if self.global_node_ids is not None else None},\n"
            f"    hyperedge_index_shape={self.hyperedge_index.shape},\n"
            f"    hyperedge_weights_shape={self.hyperedge_weights.shape if self.hyperedge_weights is not None else None},\n"
            f"    hyperedge_attr_shape={self.hyperedge_attr.shape if self.hyperedge_attr is not None else None},\n"
            f"    y_shape={self.y.shape if self.y is not None else None}\n"
            f"    device={self.device}\n"
            f")"
        )

    @classmethod
    def cat_same_node_space(cls, hdatas: Sequence["HData"], x: Optional[Tensor] = None) -> "HData":
        """
        Concatenate :class:`HData` instances that share the same node space, meaning nodes with the same ID in different instances are the same node.
        This is useful when combining positive and negative hyperedges that reference the same set of nodes.

        Notes:
            - ``x`` is derived from the instance with the largest number of nodes, if not provided explicitly. If there are conflicting features for the same node ID across instances, the features from the instance with the largest number of nodes will be used.
            - ``hyperedge_index`` is the concatenation of all input hyperedge indices.
            - ``hyperedge_weights`` is the concatenation of all input hyperedge weights, if present. If some instances have hyperedge weights and others do not, the resulting ``hyperedge_weights`` will be set to ``None``.
            - ``hyperedge_attr`` is the concatenation of all input hyperedge attributes, if present. If some instances have hyperedge attributes and others do not, the resulting ``hyperedge_attr`` will be set to ``None``.
            - ``y`` is the concatenation of all input labels.

        Examples:
            >>> x = torch.randn(5, 8)
            >>> pos = HData(x=x, hyperedge_index=torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 2, 2]]))
            >>> neg = HData(x=x, hyperedge_index=torch.tensor([[0, 2], [3, 3]]))
            >>> new = HData.cat_same_node_space([pos, neg])
            >>> new.num_nodes  # 5 — nodes [0, 1, 2, 3, 4]
            >>> new.num_hyperedges  # 4 — hyperedges [0, 1, 2, 3]

        Args:
            hdatas: One or more :class:`HData` instances sharing the same node space.
            x: Optional node feature matrix to use for the resulting :class:`HData`.
                If ``None``, the node features from the instance with the largest number of nodes will be used.

        Returns:
            A new :class:`HData` with shared nodes and concatenated hyperedges.

        Raises:
            ValueError: If the node counts do not match across inputs.
        """
        if len(hdatas) < 1:
            raise ValueError("At least one instance is required.")

        joint_hyperedge_ids = torch.cat([hdata.hyperedge_index[1].unique() for hdata in hdatas])
        unique_joint_hyperedge_ids = joint_hyperedge_ids.unique()
        if unique_joint_hyperedge_ids.size(0) != joint_hyperedge_ids.size(0):
            raise ValueError(
                "Overlapping hyperedge IDs found across instances. Ensure each instance uses distinct hyperedge IDs."
            )

        new_x = x if x is not None else max(hdatas, key=lambda hdata: hdata.num_nodes).x
        new_y = torch.cat([hdata.y for hdata in hdatas], dim=0)
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
            global_node_ids=max(hdatas, key=lambda hdata: hdata.num_nodes).global_node_ids,
            y=new_y,
        )

    @classmethod
    def empty(cls) -> "HData":
        return cls(
            x=empty_nodefeatures(),
            hyperedge_index=empty_hyperedgeindex(),
            hyperedge_weights=None,
            hyperedge_attr=None,
            num_nodes=0,
            num_hyperedges=0,
            global_node_ids=torch.empty(size=(0, 0), dtype=torch.long),
            y=None,
        )

    @classmethod
    def from_hyperedge_index(cls, hyperedge_index: Tensor) -> "HData":
        """
        Build an :class:`HData` from a given hyperedge index, with empty node features and hyperedge attributes.

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
            hyperedge_index: Tensor of shape ``[2, num_incidences]`` representing the hypergraph connectivity.

        Returns:
            An :class:`HData` instance with the given hyperedge index and default values for other attributes.
        """
        return cls(
            x=empty_nodefeatures(),
            hyperedge_index=hyperedge_index,
            hyperedge_weights=None,
            hyperedge_attr=None,
            global_node_ids=None,
            y=None,
        )

    @classmethod
    def split(
        cls,
        hdata: "HData",
        split_hyperedge_ids: Tensor,
        node_space_setting: NodeSpaceSetting = "transductive",
    ) -> "HData":
        """
        Build an :class:`HData` for a single split from the given hyperedge IDs.

        Examples:
            Transductive split (default) preserving the full node space:
            >>> split_hdata = HData.split(hdata, torch.tensor([1]), node_space_setting="transductive")
            >>> split_hdata.x.shape[0] == hdata.x.shape[0]
            >>> split_hdata.hyperedge_index
            ... # node IDs stay in the original row space, hyperedge IDs are rebased

            Inductive split:
            >>> split_hdata = HData.split(hdata, torch.tensor([1]), node_space_setting="inductive")
            >>> split_hdata.x.shape[0]  # only nodes incident to hyperedge 1
            ... 2

        Args:
            hdata: The original :class:`HData` containing the full hypergraph.
            split_hyperedge_ids: Tensor of hyperedge IDs to include in this split.
            node_space_setting: Whether to preserve the full node space in the splits.
                ``transductive`` (default) ensures all nodes are present in every split,
                while ``inductive`` allows splits to have disjoint node spaces.

        Returns:
            The splitted instance with remapped node and hyperedge IDs.
        """
        # Mask to keep only incidences belonging to selected hyperedges
        # Example: hyperedge_index = [[0, 0, 1, 2, 3, 4],
        #                             [0, 0, 0, 1, 2, 2]]
        #          split_hyperedge_ids = [0, 2]
        #          -> mask = [True, True, True, False, True, True]
        keep_mask = torch.isin(hdata.hyperedge_index[1], split_hyperedge_ids)

        # Example: hyperedge_index = [[0, 0, 1, 3, 4],
        #                             [0, 0, 0, 2, 2]]
        #          incidence [2, 1] is missing as 1 is not in split_hyperedge_ids = [0, 2]
        split_hyperedge_index = hdata.hyperedge_index[:, keep_mask].clone()

        # Example: split_hyperedge_index = [[2, 3, 4],
        #                                   [2, 2, 5]]
        #          -> split_unique_hyperedge_ids = [2, 5]
        split_unique_hyperedge_ids = split_hyperedge_index[1].unique()

        split_y = hdata.y[split_unique_hyperedge_ids]

        split_hyperedge_attr = None
        if hdata.hyperedge_attr is not None:
            split_hyperedge_attr = hdata.hyperedge_attr[split_unique_hyperedge_ids]

        split_hyperedge_weights = None
        if hdata.hyperedge_weights is not None:
            split_hyperedge_weights = hdata.hyperedge_weights[split_unique_hyperedge_ids]

        # We don't need to split nodes, so we split only hyperedges and rebase their IDs to 0-based
        if is_transductive_setting(node_space_setting):
            # Example: split_unique_hyperedge_ids = [2, 5]
            #          -> hyperedge 2 -> 0, hyperedge 5 -> 1
            split_hyperedge_index[1] = to_0based_ids(
                original_ids=split_hyperedge_index[1],
                ids_to_rebase=split_unique_hyperedge_ids,
            )
            return cls(
                x=hdata.x,
                hyperedge_index=split_hyperedge_index,
                hyperedge_weights=split_hyperedge_weights,
                hyperedge_attr=split_hyperedge_attr,
                num_nodes=hdata.num_nodes,
                num_hyperedges=len(split_unique_hyperedge_ids),
                global_node_ids=hdata.global_node_ids,
                y=split_y,
            )

        # Example: split_hyperedge_index = [[0, 0, 1, 3, 4],
        #                                   [0, 0, 0, 2, 2]]
        #          -> split_unique_node_ids = [0, 1, 3, 4]
        split_unique_node_ids = split_hyperedge_index[0].unique()

        split_hyperedge_index = (
            HyperedgeIndex(split_hyperedge_index)
            .to_0based(
                node_ids_to_rebase=split_unique_node_ids,
                hyperedge_ids_to_rebase=split_unique_hyperedge_ids,
            )
            .item
        )

        split_x = hdata.x[split_unique_node_ids]

        split_global_node_ids = None
        if hdata.global_node_ids is not None:
            split_global_node_ids = hdata.global_node_ids[split_unique_node_ids]

        return cls(
            x=split_x,
            hyperedge_index=split_hyperedge_index,
            hyperedge_weights=split_hyperedge_weights,
            hyperedge_attr=split_hyperedge_attr,
            num_nodes=len(split_unique_node_ids),
            num_hyperedges=len(split_unique_hyperedge_ids),
            global_node_ids=split_global_node_ids,
            y=split_y,
        )

    def enrich_node_features(
        self,
        enricher: NodeEnricher,
        enrichment_mode: Optional[EnrichmentMode] = None,
    ) -> "HData":
        """
        Enrich node features using the provided node feature enricher.

        Args:
            enricher: An instance of NodeEnricher to generate structural node features from hypergraph topology.
            enrichment_mode: How to combine generated features with existing ``hdata.x``.
                ``concatenate`` appends new features as additional columns.
                ``replace`` substitutes ``hdata.x`` entirely.
        """
        enriched_features = enricher.enrich(self.hyperedge_index)

        match enrichment_mode:
            case "concatenate":
                x = torch.cat([self.x, enriched_features], dim=1)
            case _:
                x = enriched_features

        return self.__class__(
            x=x,
            hyperedge_index=self.hyperedge_index,
            hyperedge_weights=self.hyperedge_weights,
            hyperedge_attr=self.hyperedge_attr,
            num_nodes=self.num_nodes,
            num_hyperedges=self.num_hyperedges,
            global_node_ids=self.global_node_ids,
            y=self.y,
        )

    def enrich_node_features_from(
        self,
        hdata_with_features: "HData",
        node_space_setting: NodeSpaceSetting = "transductive",
        fill_value: Optional[NodeSpaceFiller] = None,
    ) -> "HData":
        """
        Copy node features from another :class:`HData` by aligning features by ``global_node_ids``.

        Examples:
            Transductive enrichment (default) expecting the same node space in both source and target:
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
            hdata_with_features: Source :class:`HData` providing node features.
            node_space_setting: The setting for the node space, determining how nodes are handled.
                If ``"transductive"``, every target node is expected to exist in the source.
                If ``"inductive"``, the target dataset may have a different node space, and missing nodes are filled using ``fill_value``.
            fill_value: Scalar or vector used to fill missing node features when ``node_space_setting`` is not transductive.

        Returns:
            A new :class:`HData` with node features copied from ``hdata_with_features``.

        Raises:
            ValueError: If either instance lacks ``global_node_ids``, if the source feature rows
                do not align with the source node IDs, if ``fill_value`` is used with
                ``node_space_setting="transductive"``, or if ``fill_value`` is missing or malformed when ``node_space_setting="inductive"``.
        """
        source_global_node_ids = hdata_with_features.global_node_ids
        source_x = hdata_with_features.x
        if self.global_node_ids is None or source_global_node_ids is None:
            raise ValueError(
                "Both HData instances must define global_node_ids to align node features."
            )
        if source_x.size(0) != source_global_node_ids.size(0):
            raise ValueError(
                "Expected hdata_with_features.x rows to align with hdata_with_features.global_node_ids."
            )
        self.__validate_node_space_setting(node_space_setting, fill_value)

        target_global_node_ids = self.global_node_ids.detach().cpu().tolist()

        # We need the index of the features for each node in the source, as we will use the index to track back
        # to the node feautures after we match the global node id in the target to the one that is in the source
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
                #          -> strict transductive mode records it as missing and then raises an error
                #          -> non-transductive mode fills the features with fill_value and continues enriching the other nodes
                if is_transductive_setting(node_space_setting):
                    missing_global_node_ids.append(
                        int(global_node_id)
                    )  # record missing node for error message
                else:
                    enriched_rows.append(
                        fill_features
                    )  # fill missing node features with fill_value and
                continue

            # Match the global node IDs in the target to the corresponding feature indices in the source
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
            hyperedge_index=self.hyperedge_index,
            hyperedge_weights=self.hyperedge_weights,
            hyperedge_attr=self.hyperedge_attr,
            num_nodes=self.num_nodes,
            num_hyperedges=self.num_hyperedges,
            global_node_ids=self.global_node_ids,
            y=self.y,
        )

    def enrich_hyperedge_weights(
        self,
        enricher: HyperedgeEnricher,
        enrichment_mode: Optional[EnrichmentMode] = None,
    ) -> "HData":
        """Enrich hyperedge weights using the provided hyperedge weight enricher.
        Args:
            enricher: An instance of HyperedgeEnricher to generate hyperedge weights from hypergraph topology.
            enrichment_mode: How to combine generated weights with existing ``hdata.hyperedge_weights``.
                ``concatenate`` appends new weights to the existing 1D tensor.
                ``replace`` substitutes ``hdata.hyperedge_weights`` entirely.
        """
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
            x=self.x,
            hyperedge_index=self.hyperedge_index,
            hyperedge_weights=hyperedge_weights,
            hyperedge_attr=self.hyperedge_attr,
            num_nodes=self.num_nodes,
            num_hyperedges=self.num_hyperedges,
            global_node_ids=self.global_node_ids,
            y=self.y,
        )

    def enrich_hyperedge_attr(
        self,
        enricher: HyperedgeEnricher,
        enrichment_mode: Optional[EnrichmentMode] = None,
    ) -> "HData":
        """
        Enrich hyperedge features using the provided hyperedge feature enricher.

        Args:
            enricher: An instance of HyperedgeEnricher to generate structural hyperedge features from hypergraph topology.
            enrichment_mode: How to combine generated features with existing ``hdata.hyperedge_attr``.
                ``concatenate`` appends new features as additional columns.
                ``replace`` substitutes ``hdata.hyperedge_attr`` entirely.
        """
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
            x=self.x,
            hyperedge_index=self.hyperedge_index,
            hyperedge_weights=self.hyperedge_weights,
            hyperedge_attr=hyperedge_attr,
            num_nodes=self.num_nodes,
            num_hyperedges=self.num_hyperedges,
            global_node_ids=self.global_node_ids,
            y=self.y,
        )

    def get_device_if_all_consistent(self) -> torch.device:
        """
        Check that all tensors are on the same device and return that device.
        If there are no tensors or if they are on different devices, return CPU.

        Returns:
            The common device if all tensors are on the same device, otherwise CPU.

        Raises:
            ValueError: If tensors are on different devices.
        """
        devices = {self.x.device, self.hyperedge_index.device, self.y.device}
        if self.global_node_ids is not None:
            devices.add(self.global_node_ids.device)
        if self.hyperedge_attr is not None:
            devices.add(self.hyperedge_attr.device)
        if self.hyperedge_weights is not None:
            devices.add(self.hyperedge_weights.device)
        if len(devices) > 1:
            raise ValueError(f"Inconsistent device placement: {devices}")

        return devices.pop() if len(devices) == 1 else torch.device("cpu")

    def remove_hyperedges_with_fewer_than_k_nodes(self, k: int) -> "HData":
        hyperedge_index_wrapper = HyperedgeIndex(
            self.hyperedge_index
        ).remove_hyperedges_with_fewer_than_k_nodes(k)

        x = self.x[hyperedge_index_wrapper.node_ids]
        global_node_ids = None
        if self.global_node_ids is not None:
            global_node_ids = self.global_node_ids[hyperedge_index_wrapper.node_ids]
        y = self.y[hyperedge_index_wrapper.hyperedge_ids]

        hyperedge_attr = None
        if self.hyperedge_attr is not None:
            hyperedge_attr = self.hyperedge_attr[hyperedge_index_wrapper.hyperedge_ids]

        hyperedge_weights = None
        if self.hyperedge_weights is not None:
            hyperedge_weights = self.hyperedge_weights[hyperedge_index_wrapper.hyperedge_ids]

        return self.__class__(
            x=x,
            hyperedge_index=hyperedge_index_wrapper.to_0based().item,
            hyperedge_weights=hyperedge_weights,
            hyperedge_attr=hyperedge_attr,
            num_nodes=hyperedge_index_wrapper.num_nodes,
            num_hyperedges=hyperedge_index_wrapper.num_hyperedges,
            global_node_ids=global_node_ids,
            y=y,
        )

    def shuffle(self, seed: Optional[int] = None) -> "HData":
        """
        Return a new :class:`HData` instance with hyperedge IDs randomly reassigned.

        Each hyperedge keeps its original set of nodes, but is assigned a new ID via a random permutation.
        ``y`` and ``hyperedge_attr`` are reordered to match, so that ``y[new_id]`` still corresponds to the correct hyperedge.
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
            seed: Optional random seed for reproducibility. If ``None``, the shuffle will be non-deterministic.

        Returns:
            A new :class:`HData` instance with hyperedge IDs, ``y``, and ``hyperedge_attr`` permuted.
        """
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)

        permutation = torch.randperm(self.num_hyperedges, generator=generator, device=self.device)

        # permutation[new_id] = old_id, so y[permutation] puts old labels into new slots
        # inverse_permutation[old_id] = new_id, used to remap hyperedge IDs in incidences
        # Example: permutation = [1, 2, 0] means new_id 0 gets old_id 1, new_id 1 gets old_id 2, new_id 2 gets old_id 0
        #          -> inverse_permutation = [2, 0, 1] means old_id 0 gets new_id 2, old_id 1 gets new_id 0, old_id 2 gets new_id 1
        inverse_permutation = torch.empty_like(permutation)
        inverse_permutation[permutation] = torch.arange(self.num_hyperedges, device=self.device)

        new_hyperedge_index = self.hyperedge_index.clone()

        # Example: hyperedge_index = [[0, 1, 2, 3, 4],
        #                             [0, 0, 1, 1, 2]],
        #          inverse_permutation = [2, 0, 1] (new_id 0 -> old_id 2, new_id 1 -> old_id 0, new_id 2 -> old_id 1)
        #          -> new_hyperedge_index = [[0, 1, 2, 3, 4],
        #                                    [2, 2, 0, 0, 1]]
        old_hyperedge_ids = self.hyperedge_index[1]
        new_hyperedge_index[1] = inverse_permutation[old_hyperedge_ids]

        # Example: hyperedge_attr = [attr_0, attr_1, attr_2], permutation = [1, 2, 0]
        #          -> new_hyperedge_attr = [attr_1  (attr of old_id 1), attr_2 (attr of old_id 2), attr_0 (attr of old_id 0)]
        new_hyperedge_attr = (
            self.hyperedge_attr[permutation] if self.hyperedge_attr is not None else None
        )

        new_hyperedge_weights = (
            self.hyperedge_weights[permutation] if self.hyperedge_weights is not None else None
        )

        # Example: y = [1, 1, 0], permutation = [1, 2, 0]
        #          -> new_y = [y[1], y[2], y[0]] = [1, 0, 1]
        new_y = self.y[permutation]

        return self.__class__(
            x=self.x,
            hyperedge_index=new_hyperedge_index,
            hyperedge_weights=new_hyperedge_weights,
            hyperedge_attr=new_hyperedge_attr,
            num_nodes=self.num_nodes,
            num_hyperedges=self.num_hyperedges,
            global_node_ids=self.global_node_ids,
            y=new_y,
        )

    def clone(self) -> "HData":
        """
        Return a deep copy of this :class:`HData`.

        Returns:
            A new :class:`HData` that is a deep copy of this instance.
        """
        cloned_hyperedge_weights = (
            self.hyperedge_weights.clone() if self.hyperedge_weights is not None else None
        )
        cloned_hyperedge_attr = (
            self.hyperedge_attr.clone() if self.hyperedge_attr is not None else None
        )
        cloned_global_node_ids = (
            self.global_node_ids.clone() if self.global_node_ids is not None else None
        )
        return self.__class__(
            x=self.x.clone(),
            hyperedge_index=self.hyperedge_index.clone(),
            hyperedge_weights=cloned_hyperedge_weights,
            hyperedge_attr=cloned_hyperedge_attr,
            num_nodes=self.num_nodes,
            num_hyperedges=self.num_hyperedges,
            global_node_ids=cloned_global_node_ids,
            y=self.y.clone(),
        )

    def to(self, device: torch.device | str, non_blocking: bool = False) -> "HData":
        """
        Move all tensors to the specified device.

        Args:
            device: The target device (e.g., 'cpu', 'cuda:0').
            non_blocking: If ``True`` and the source and destination devices are both CUDA, the copy will be non-blocking.

        Returns:
            The :class:`HData` instance with all tensors moved to the specified device.
        """
        self.x = self.x.to(device=device, non_blocking=non_blocking)
        self.hyperedge_index = self.hyperedge_index.to(device=device, non_blocking=non_blocking)
        self.y = self.y.to(device=device, non_blocking=non_blocking)

        if self.global_node_ids is not None:
            self.global_node_ids = self.global_node_ids.to(device=device, non_blocking=non_blocking)

        if self.hyperedge_attr is not None:
            self.hyperedge_attr = self.hyperedge_attr.to(device=device, non_blocking=non_blocking)

        if self.hyperedge_weights is not None:
            self.hyperedge_weights = self.hyperedge_weights.to(
                device=device, non_blocking=non_blocking
            )

        self.device = device if isinstance(device, torch.device) else torch.device(device)
        return self

    def with_y_to(self, value: float) -> "HData":
        """
        Return a copy of this instance with a y attribute set to the given value.

        Args:
            value: The value to set for all entries in the y attribute.

        Returns:
            A new :class:`HData` instance with the same attributes except for y, which is set to a tensor of the given value.
        """
        return self.__class__(
            x=self.x,
            hyperedge_index=self.hyperedge_index,
            hyperedge_weights=self.hyperedge_weights,
            hyperedge_attr=self.hyperedge_attr,
            num_nodes=self.num_nodes,
            num_hyperedges=self.num_hyperedges,
            global_node_ids=self.global_node_ids,
            y=torch.full((self.num_hyperedges,), value, dtype=torch.float, device=self.device),
        )

    def with_y_ones(self) -> "HData":
        """Return a copy of this instance with a y attribute of all ones."""
        return self.with_y_to(1.0)

    def with_y_zeros(self) -> "HData":
        """Return a copy of this instance with a y attribute of all zeros."""
        return self.with_y_to(0.0)

    def stats(self) -> Dict[str, Any]:
        """
        Compute statistics for the hypergraph data.
        The fields returned in the dictionary include:
        - ``shape_x``: The shape of the node feature matrix ``x``.
        - ``shape_hyperedge_weights``: The shape of the hyperedge weights tensor, or ``None`` if hyperedge weights are not present.
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
            distribution_node_degree = torch.zeros(self.num_nodes, dtype=torch.float)
            distribution_hyperedge_size = torch.zeros(self.num_hyperedges, dtype=torch.float)

        num_nodes = self.num_nodes
        num_hyperedges = self.num_hyperedges

        if distribution_node_degree.numel() > 0:
            avg_degree_node_raw = distribution_node_degree.mean().item()
            avg_degree_node = int(avg_degree_node_raw)
            avg_degree_hyperedge_raw = distribution_hyperedge_size.mean().item()
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
            "num_nodes": num_nodes,
            "num_hyperedges": num_hyperedges,
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

    def __to_fill_features(
        self,
        fill_value: Optional[NodeSpaceFiller],
        num_features: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
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
        # - A list with a single value, e.g., [0.0], which should also be broadcasted to all features
        if fill_features.numel() == 1:
            fill_features = fill_features.repeat(num_features)

        if fill_features.dim() != 1 or fill_features.numel() != num_features:
            raise ValueError(
                f"Expected fill_value to define exactly {num_features} features, got shape "
                f"{tuple(fill_features.shape)}."
            )
        return fill_features

    def __validate_node_space_setting(
        self,
        node_space_setting: NodeSpaceSetting,
        fill_value: Optional[NodeSpaceFiller],
    ) -> None:
        if is_transductive_setting(node_space_setting) and fill_value is not None:
            raise ValueError(
                "fill_value cannot be provided when node_space_setting='transductive'."
            )
        if is_inductive_setting(node_space_setting) and fill_value is None:
            raise ValueError("fill_value must be provided when node_space_setting='inductive'.")
