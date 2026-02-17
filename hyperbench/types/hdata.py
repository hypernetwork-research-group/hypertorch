import torch

from torch import Tensor
from typing import Optional, Sequence
from hyperbench import utils

from .hypergraph import HyperedgeIndex


class HData:
    """
    Container for hypergraph data.

    Example:
        >>> x = torch.randn(10, 16)  # 10 nodes with 16 features each
        >>> edge_index = torch.tensor([[0, 0, 1, 1, 1],  # node IDs
        ...                            [0, 1, 2, 3, 4]]) # hyperedge IDs
        >>> data = HData(x, edge_index=edge_index)

    Attributes:
        x (Tensor): Node feature matrix of shape ``[num_nodes, num_features]``.
        edge_index (Tensor): Hyperedge connectivity in COO format of shape ``[2, num_incidences]``,
            where ``edge_index[0]`` contains node IDs and ``edge_index[1]`` contains hyperedge IDs.
        edge_attr (Tensor, optional): Hyperedge feature matrix of shape ``[num_edges, num_edge_features]``.
            Features associated with each hyperedge (e.g., weights, timestamps, types).
        num_nodes (int, optional): Number of nodes in the hypergraph.
            If ``None``, inferred as ``x.size(0)``.
        num_edges (int, optional): Number of hyperedges in the hypergraph.
            If ``None``, inferred as the number of unique hyperedge IDs in ``edge_index[1]``.
        y (Tensor, optional): Labels for hyperedges, of shape ``[num_edges]``.
            Used for supervised learning tasks. For unsupervised tasks, this can be ignored.
            Default is a tensor of ones, indicating all hyperedges are positive examples.
    """

    def __init__(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
        num_edges: Optional[int] = None,
        y: Optional[Tensor] = None,
    ):
        self.x: Tensor = x

        self.edge_index: Tensor = edge_index

        self.edge_attr: Optional[Tensor] = edge_attr

        self.num_nodes: int = num_nodes if num_nodes is not None else x.size(0)

        self.num_edges: int = (
            num_edges if num_edges is not None else HyperedgeIndex(edge_index).num_hyperedges
        )

        self.y = (
            y
            if y is not None
            else torch.ones((self.num_edges,), dtype=torch.float, device=self.x.device)
        )

        self.device = self.get_device_if_all_consistent()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"    num_nodes={self.num_nodes},\n"
            f"    num_edges={self.num_edges},\n"
            f"    x_shape={self.x.shape},\n"
            f"    edge_index_shape={self.edge_index.shape},\n"
            f"    edge_attr_shape={self.edge_attr.shape if self.edge_attr is not None else None},\n"
            f"    y_shape={self.y.shape if self.y is not None else None}\n"
            f"    device={self.device}\n"
            f")"
        )

    @classmethod
    def empty(cls) -> "HData":
        return cls(
            x=utils.empty_nodefeatures(),
            edge_index=utils.empty_edgeindex(),
            edge_attr=None,
            num_nodes=0,
            num_edges=0,
            y=None,
        )

    @classmethod
    def cat_same_node_space(cls, hdatas: Sequence["HData"], x: Optional[Tensor] = None) -> "HData":
        """
        Concatenate HData instances that share the same node space, meaning nodes with the same ID in different instances are the same node.
        This is useful when combining positive and negative hyperedges that reference the same set of nodes.

        Results:
            - ``x`` is derived from the instance with the largest number of nodes, if not provided explicitly. If there are conflicting features for the same node ID across instances, the features from the instance with the largest number of nodes will be used.
            - ``edge_index`` is the concatenation of all input edge indices.
            - ``edge_attr`` is the concatenation of all input edge attributes, if present. If some instances have edge attributes and others do not, the resulting ``edge_attr`` will be set to ``None``.
            - ``y`` is the concatenation of all input labels.

        Example:
            >>> x = torch.randn(5, 8)
            >>> pos = HData(x, torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 2, 2]]))
            >>> neg = HData(x, torch.tensor([[0, 2], [3, 3]]))
            >>> new = HData.cat_same_node_space([pos, neg])
            >>> new.num_nodes -> 5 -> [0, 1, 2, 3, 4]
            >>> new.num_edges -> 4 -> [0, 1, 2, 3]

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

        joint_hyperedge_ids = torch.cat([hdata.edge_index[1].unique() for hdata in hdatas])
        unique_joint_hyperedge_ids = joint_hyperedge_ids.unique()
        if unique_joint_hyperedge_ids.size(0) != joint_hyperedge_ids.size(0):
            raise ValueError(
                "Overlapping hyperedge IDs found across instances. Ensure each instance uses distinct hyperedge IDs."
            )

        new_x = x if x is not None else max(hdatas, key=lambda hdata: hdata.num_nodes).x
        new_y = torch.cat([hdata.y for hdata in hdatas], dim=0)
        new_hyperedge_index = torch.cat([hdata.edge_index for hdata in hdatas], dim=1)

        hyperedge_attrs = []
        have_all_hyperedge_attr = all(hdata.edge_attr is not None for hdata in hdatas)
        for hdata in hdatas:
            if have_all_hyperedge_attr and hdata.edge_attr is not None:
                hyperedge_attrs.append(hdata.edge_attr)
        new_hyperedge_attr = torch.cat(hyperedge_attrs, dim=0) if len(hyperedge_attrs) > 0 else None

        return cls(
            x=new_x,
            edge_index=new_hyperedge_index,
            edge_attr=new_hyperedge_attr,
            num_nodes=new_x.size(0),
            num_edges=new_y.size(0),
            y=new_y,
        )

    @classmethod
    def split(cls, hdata: "HData", split_hyperedge_ids: Tensor) -> "HData":
        """
        Build an :class:`HData` for a single split from the given hyperedge IDs.

        Example:
            >>> hyperedge_index = [[0, 0, 1, 2, 3, 4],
            >>>                    [0, 0, 0, 1, 2, 2]]
            >>> split_hyperedge_ids = [0, 2]
            >>> new_hyperedge_index = [[0, 0, 1, 2, 3],  # nodes 0 -> 0, 1 -> 1, 3 -> 2, 4 -> 3 (remapped to 0-based)
            >>>                        [0, 0, 0, 1, 1]]  # hyperedges 0 -> 0, 2 -> 1 (remapped to 0-based)
            >>> new_x = [x[0], x[1], x[3], x[4]]
            >>> new_edge_attr = [edge_attr[0], edge_attr[2]]
        Args:
            hdata: The original :class:`HData` containing the full hypergraph.
            split_hyperedge_ids: Tensor of hyperedge IDs to include in this split.

        Returns:
            The splitted instance with remapped node and hyperedge IDs.
        """
        # Mask to keep only incidences belonging to selected hyperedges
        # Example: hyperedge_index = [[0, 0, 1, 2, 3, 4],
        #                             [0, 0, 0, 1, 2, 2]]
        #          split_hyperedge_ids = [0, 2]
        #          -> mask = [True, True, True, False, True, True]
        keep_mask = torch.isin(hdata.edge_index[1], split_hyperedge_ids)

        # Example: hyperedge_index = [[0, 0, 1, 3, 4],
        #                             [0, 0, 0, 2, 2]]
        #          incidence [2, 1] is missing as 1 is not in split_hyperedge_ids = [0, 2]
        split_hyperedge_index = hdata.edge_index[:, keep_mask]

        # Example: split_hyperedge_index = [[0, 0, 1, 3, 4],
        #                                   [0, 0, 0, 2, 2]]
        #          -> split_unique_node_ids = [0, 1, 3, 4]
        #          -> split_unique_hyperedge_ids = [0, 2]
        split_unique_node_ids = split_hyperedge_index[0].unique()
        split_unique_hyperedge_ids = split_hyperedge_index[1].unique()

        split_hyperedge_index_wrapper = HyperedgeIndex(split_hyperedge_index).to_0based(
            node_ids_to_rebase=split_unique_node_ids,
            hyperedge_ids_to_rebase=split_unique_hyperedge_ids,
        )

        new_x = hdata.x[split_unique_node_ids]
        new_y = hdata.y[split_unique_hyperedge_ids]

        # Subset hyperedge_attr if present
        new_hyperedge_attr = None
        if hdata.edge_attr is not None:
            new_hyperedge_attr = hdata.edge_attr[split_unique_hyperedge_ids]

        return cls(
            x=new_x,
            edge_index=split_hyperedge_index_wrapper.item,
            edge_attr=new_hyperedge_attr,
            num_nodes=len(split_unique_node_ids),
            num_edges=len(split_unique_hyperedge_ids),
            y=new_y,
        )

    def get_device_if_all_consistent(self) -> torch.device:
        devices = {self.x.device, self.edge_index.device, self.y.device}
        if self.edge_attr is not None:
            devices.add(self.edge_attr.device)
        if len(devices) > 1:
            raise ValueError(f"Inconsistent device placement: {devices}")

        return devices.pop() if len(devices) == 1 else torch.device("cpu")

    def to(self, device: torch.device | str, non_blocking: bool = False) -> "HData":
        self.x = self.x.to(device=device, non_blocking=non_blocking)
        self.edge_index = self.edge_index.to(device=device, non_blocking=non_blocking)
        self.y = self.y.to(device=device, non_blocking=non_blocking)

        if self.edge_attr is not None:
            self.edge_attr = self.edge_attr.to(device=device, non_blocking=non_blocking)

        self.device = device if isinstance(device, torch.device) else torch.device(device)
        return self

    def with_y_to(self, value: float) -> "HData":
        """Return a copy of this instance with a y attribute set to the given value."""
        return HData(
            x=self.x,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            num_nodes=self.num_nodes,
            num_edges=self.num_edges,
            y=torch.full((self.num_edges,), value, dtype=torch.float, device=self.device),
        )

    def with_y_ones(self) -> "HData":
        """Return a copy of this instance with a y attribute of all ones."""
        return self.with_y_to(1.0)

    def with_y_zeros(self) -> "HData":
        """Return a copy of this instance with a y attribute of all zeros."""
        return self.with_y_to(0.0)
