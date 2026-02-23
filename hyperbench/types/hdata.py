import torch

from torch import Tensor
from typing import Optional, Sequence
from hyperbench.utils import empty_hyperedgeindex, empty_nodefeatures

from .hypergraph import HyperedgeIndex


class HData:
    """
    Container for hypergraph data.

    Examples:
        >>> x = torch.randn(10, 16)  # 10 nodes with 16 features each
        >>> hyperedge_index = torch.tensor([[0, 0, 1, 1, 1],  # node IDs
        ...                                 [0, 1, 2, 3, 4]]) # hyperedge IDs
        >>> data = HData(x, hyperedge_index=hyperedge_index)

    Attributes:
        x (Tensor): Node feature matrix of shape ``[num_nodes, num_features]``.
        hyperedge_index (Tensor): Hyperedge connectivity in COO format of shape ``[2, num_incidences]``,
            where ``hyperedge_index[0]`` contains node IDs and ``hyperedge_index[1]`` contains hyperedge IDs.
        hyperedge_attr (Tensor, optional): Hyperedge feature matrix of shape ``[num_hyperedges, num_hyperedge_features]``.
            Features associated with each hyperedge (e.g., weights, timestamps, types).
        num_nodes (int, optional): Number of nodes in the hypergraph.
            If ``None``, inferred as ``x.size(0)``.
        num_hyperedges (int, optional): Number of hyperedges in the hypergraph.
            If ``None``, inferred as the number of unique hyperedge IDs in ``hyperedge_index[1]``.
        y (Tensor, optional): Labels for hyperedges, of shape ``[num_hyperedges]``.
            Used for supervised learning tasks. For unsupervised tasks, this can be ignored.
            Default is a tensor of ones, indicating all hyperedges are positive examples.
    """

    def __init__(
        self,
        x: Tensor,
        hyperedge_index: Tensor,
        hyperedge_attr: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
        num_hyperedges: Optional[int] = None,
        y: Optional[Tensor] = None,
    ):
        self.x: Tensor = x

        self.hyperedge_index: Tensor = hyperedge_index

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
            f"    hyperedge_index_shape={self.hyperedge_index.shape},\n"
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
            - ``hyperedge_attr`` is the concatenation of all input hyperedge attributes, if present. If some instances have hyperedge attributes and others do not, the resulting ``hyperedge_attr`` will be set to ``None``.
            - ``y`` is the concatenation of all input labels.

        Examples:
            >>> x = torch.randn(5, 8)
            >>> pos = HData(x, torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 2, 2]]))
            >>> neg = HData(x, torch.tensor([[0, 2], [3, 3]]))
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
        have_all_hyperedge_attr = all(hdata.hyperedge_attr is not None for hdata in hdatas)
        for hdata in hdatas:
            if have_all_hyperedge_attr and hdata.hyperedge_attr is not None:
                hyperedge_attrs.append(hdata.hyperedge_attr)
        new_hyperedge_attr = torch.cat(hyperedge_attrs, dim=0) if len(hyperedge_attrs) > 0 else None

        return cls(
            x=new_x,
            hyperedge_index=new_hyperedge_index,
            hyperedge_attr=new_hyperedge_attr,
            num_nodes=new_x.size(0),
            num_hyperedges=new_y.size(0),
            y=new_y,
        )

    @classmethod
    def empty(cls) -> "HData":
        return cls(
            x=empty_nodefeatures(),
            hyperedge_index=empty_hyperedgeindex(),
            hyperedge_attr=None,
            num_nodes=0,
            num_hyperedges=0,
            y=None,
        )

    @classmethod
    def from_hyperedge_index(cls, hyperedge_index: Tensor) -> "HData":
        """
        Build an :class:`HData` from a given hyperedge index, with empty node features and hyperedge attributes.

        - Node features are initialized as an empty tensor of shape ``[0, 0]``.
        - Hyperedge attributes are set to ``None``.
        - The number of nodes and hyperedges are inferred from the hyperedge index.

        Examples:
            >>> hyperedge_index = [[0, 0, 1, 2, 3, 4],
            ...                    [0, 0, 0, 1, 2, 2]]
            >>> num_nodes = 5
            >>> num_hyperedges = 3
            >>> x = []  # Empty node features with shape [0, 0]
            >>> hyperedge_attr = None

        Args:
            hyperedge_index: Tensor of shape ``[2, num_incidences]`` representing the hypergraph connectivity.

        Returns:
            An :class:`HData` instance with the given hyperedge index and default values for other attributes.
        """
        return cls(
            x=empty_nodefeatures(),
            hyperedge_index=hyperedge_index,
            hyperedge_attr=None,
            y=None,
        )

    @classmethod
    def split(cls, hdata: "HData", split_hyperedge_ids: Tensor) -> "HData":
        """
        Build an :class:`HData` for a single split from the given hyperedge IDs.

        Examples:
            >>> hyperedge_index = [[0, 0, 1, 2, 3, 4],
            ...                    [0, 0, 0, 1, 2, 2]]
            >>> split_hyperedge_ids = [0, 2]
            >>> new_hyperedge_index = [[0, 0, 1, 2, 3],  # nodes 0 -> 0, 1 -> 1, 3 -> 2, 4 -> 3 (remapped to 0-based)
            ...                        [0, 0, 0, 1, 1]]  # hyperedges 0 -> 0, 2 -> 1 (remapped to 0-based)
            >>> new_x = [x[0], x[1], x[3], x[4]]
            >>> new_hyperedge_attr = [hyperedge_attr[0], hyperedge_attr[2]]

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
        keep_mask = torch.isin(hdata.hyperedge_index[1], split_hyperedge_ids)

        # Example: hyperedge_index = [[0, 0, 1, 3, 4],
        #                             [0, 0, 0, 2, 2]]
        #          incidence [2, 1] is missing as 1 is not in split_hyperedge_ids = [0, 2]
        split_hyperedge_index = hdata.hyperedge_index[:, keep_mask]

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
        if hdata.hyperedge_attr is not None:
            new_hyperedge_attr = hdata.hyperedge_attr[split_unique_hyperedge_ids]

        return cls(
            x=new_x,
            hyperedge_index=split_hyperedge_index_wrapper.item,
            hyperedge_attr=new_hyperedge_attr,
            num_nodes=len(split_unique_node_ids),
            num_hyperedges=len(split_unique_hyperedge_ids),
            y=new_y,
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
        if self.hyperedge_attr is not None:
            devices.add(self.hyperedge_attr.device)
        if len(devices) > 1:
            raise ValueError(f"Inconsistent device placement: {devices}")

        return devices.pop() if len(devices) == 1 else torch.device("cpu")

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

        if self.hyperedge_attr is not None:
            self.hyperedge_attr = self.hyperedge_attr.to(device=device, non_blocking=non_blocking)

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
        return HData(
            x=self.x,
            hyperedge_index=self.hyperedge_index,
            hyperedge_attr=self.hyperedge_attr,
            num_nodes=self.num_nodes,
            num_hyperedges=self.num_hyperedges,
            y=torch.full((self.num_hyperedges,), value, dtype=torch.float, device=self.device),
        )

    def with_y_ones(self) -> "HData":
        """Return a copy of this instance with a y attribute of all ones."""
        return self.with_y_to(1.0)

    def with_y_zeros(self) -> "HData":
        """Return a copy of this instance with a y attribute of all zeros."""
        return self.with_y_to(0.0)
