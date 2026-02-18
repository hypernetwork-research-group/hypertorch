import torch

from typing import List, Optional, Tuple
from torch import Tensor
from torch.utils.data import DataLoader as TorchDataLoader
from hyperbench.data import Dataset
from hyperbench.types import HData, HyperedgeIndex


class DataLoader(TorchDataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: Optional[bool] = False,
        **kwargs,
    ) -> None:
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate,
            **kwargs,
        )

    def collate(self, batch: List[HData]) -> HData:
        """
        Collates a list of HData objects into a single batched HData object.

        This function combines multiple separate hypergraph samples into a single
        batched representation suitable for mini-batch training. It handles:
        - Concatenating node features from all samples
        - Concatenating and offsetting hyperedge from all samples
        - Concatenating edge attributes from all samples, if present

        Examples:
            Given ``batch = [HData_0, HData_1]``:

            For node features:

            >>> HData_0.x.shape  # (3, 64) — 3 nodes with 64 features
            >>> HData_1.x.shape  # (2, 64) — 2 nodes with 64 features
            >>> x.shape  # (5, 64) — all 5 nodes concatenated

            For hyperedge index:

            - ``HData_0`` (3 nodes, 2 hyperedges):

            >>> hyperedge_index = [[0, 1, 1, 2],  # Nodes 0, 1, 1, 2
            ...                    [0, 0, 1, 1]]  # Hyperedge 0 contains {0,1}, Hyperedge 1 contains {1,2}

            - ``HData_1`` (2 nodes, 1 hyperedge):

            >>> hyperedge_index = [[0, 1],  # Nodes 0, 1
            ...                    [0, 0]]  # Hyperedge 0 contains {0,1}

            Batched result:

            >>> hyperedge_index = [[0, 1, 1, 2, 3, 4],  # Node indices: original then offset by 3
            ...                    [0, 0, 1, 1, 2, 2]]  # Hyperedge IDs: original then offset by 2

        Args:
            batch: List of HData objects to collate.

        Returns:
            A single :class:`HData` object containing the batched data.
        """
        x, total_nodes = self.__batch_x(batch)
        hyperedge_index, hyperedge_attr, total_hyperedges = self.__batch_hyperedges(batch)
        y = torch.cat([data.y for data in batch], dim=0)

        batched_data = HData(
            x=x,
            hyperedge_index=hyperedge_index,
            hyperedge_attr=hyperedge_attr,
            num_nodes=total_nodes,
            num_hyperedges=total_hyperedges,
            y=y,
        )

        return batched_data.to(batch[0].device)

    def __batch_x(self, batch: List[HData]) -> Tuple[Tensor, int]:
        """
        Concatenates node features from all samples in the batch.

        Examples:
            With shape being ``(num_nodes_in_sample, num_features)``.

            If batch contains 3 sample with node features:

            >>> Sample 0: x = [[1, 2], [3, 4]]           , shape: (2, 2)
            >>> Sample 1: x = [[5, 6]]                   , shape: (1, 2)
            >>> Sample 2: x = [[7, 8], [9, 10], [11, 12]], shape: (3, 2)

            Result:

            >>> x: [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
            >>> shape: (6, 2), where 6 = 2 + 1 + 3 total nodes.

        Args:
            batch: List of :class:`HData` objects.

        Returns:
            The concatenated node features with shape ``(total_nodes, num_features)``.
        """
        per_sample_x = [data.x for data in batch]

        # Stack all nodes along the node dimension from all samples into a single tensor
        batched_x = torch.cat(per_sample_x, dim=0)
        total_nodes = batched_x.size(0)

        return batched_x, total_nodes

    def __batch_hyperedges(self, batch: List[HData]) -> Tuple[Tensor, Optional[Tensor], int]:
        """
        Batches hyperedge indices and attributes, adjusting indices for concatenated nodes.
        Hyperedge indices must be offset so they point to the correct nodes in the batched node tensor.

        Examples:
            Sample 0 (3 nodes, 2 hyperedges):
                >>> hyperedge_index = [[0, 1, 1, 2], # Nodes 0, 1, 1, 2
                ...                    [0, 0, 1, 1]] # Hyperedge 0 contains {0,1}, Hyperedge 1 contains {1,2}
                >>> node_offset = 0
                >>> edge_offset = 0

            Sample 1 (2 nodes, 1 hyperedge):
                >>> hyperedge_index = [[0, 1], # Nodes 0, 1
                ...                    [0, 0]] # Hyperedge 0 contains {0,1}
                >>> node_offset = 3 # Previous samples have 3 nodes total
                >>> edge_offset = 2 # Previous samples have 2 hyperedges total
            Result:
                >>> hyperedge_index = [[0, 1, 1, 2, 3, 4], # Node indices: original then offset by 3, so 0->3, 1->4
                ...                    [0, 0, 1, 1, 2, 2]] # Hyperedge IDs: original then offset by 2, so 0->2, 0->2
                ...                     ^^^^^^^^^^  ^^^^
                ...                     Sample 0    Sample 1 (nodes +3, edges +2)

        Args:
            batch: List of :class:`HData` objects.

        Returns:
            The tuple containing:
                - batched_hyperedge_index: Concatenated and offset hyperedge indices, or ``None``.
                - batched_hyperedge_attr: Concatenated hyperedge attributes, or ``None``.
                - total_hyperedges: Total number of hyperedges across all batched samples.
        """
        hyperedge_indexes = []
        hyperedge_attrs = []
        node_offset = 0
        hyperedge_offset = 0

        for data in batch:
            # Offset nodes and hyperedge IDs (indices) in hyperedge_index
            offset_hyperedge_index = data.hyperedge_index.clone()
            offset_hyperedge_index[0] += node_offset
            offset_hyperedge_index[1] += hyperedge_offset
            hyperedge_indexes.append(offset_hyperedge_index)

            if data.hyperedge_attr is not None:
                hyperedge_attrs.append(data.hyperedge_attr)

            hyperedge_offset += data.num_hyperedges
            node_offset += data.num_nodes

        # Concatenate all hyperedge_index tensors along the incidence dimension, so that we get a shape of (2, total_hyperedges)
        batched_hyperedge_index = torch.cat(hyperedge_indexes, dim=1)
        total_hyperedges = HyperedgeIndex(batched_hyperedge_index).num_hyperedges
        batched_hyperedge_attr = None
        if len(hyperedge_attrs) > 0:
            # Concatenate hyperedge attributes along dimension 0 (the hyperedge dimension)
            # hyperedge_attr typically has shape (num_hyperedges, num_hyperedge_features)
            # Result shape: (total_hyperedges, num_hyperedge_features)
            batched_hyperedge_attr = torch.cat(hyperedge_attrs, dim=0)

        return batched_hyperedge_index, batched_hyperedge_attr, total_hyperedges
